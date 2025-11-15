import pandas as pd
import numpy as np
import io
import time
import base64
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings
from typing import Dict, List, Any, Optional, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle non-JSON compliant float values
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return str(val)  # Convert inf/-inf to string
        return val
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
        # Handle regular Python floats that might be inf/nan
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return str(obj)
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def _analyze_null_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze null patterns in the dataset."""
    null_analysis = {
        "total_rows": len(df),
        "columns_with_nulls": [],
        "null_summary": {},
        "missing_data_heatmap": {},
        "recommendations": []
    }
    
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_percentage = float((null_count / len(df) * 100) if len(df) > 0 else 0)
        
        if null_count > 0:
            null_analysis["columns_with_nulls"].append(str(col))  # Ensure string type
            null_analysis["null_summary"][str(col)] = {  # Ensure string type
                "null_count": _convert_numpy_types(null_count),
                "null_percentage": _convert_numpy_types(round(null_percentage, 2)),
                "data_type": str(df[col].dtype),
                "non_null_count": _convert_numpy_types(df[col].count()),
                "suggested_strategy": _suggest_imputation_strategy(df[col], null_percentage)
            }
            
            # Generate recommendations based on null percentage
            if null_percentage > 70:
                null_analysis["recommendations"].append({
                    "column": str(col),  # Ensure string type
                    "action": "consider_dropping",
                    "reason": f"Column has {float(null_percentage):.1f}% missing values",
                    "priority": "high"
                })
            elif null_percentage > 30:
                null_analysis["recommendations"].append({
                    "column": str(col),  # Ensure string type
                    "action": "advanced_imputation",
                    "reason": f"Column has {float(null_percentage):.1f}% missing values - consider KNN or model-based imputation",
                    "priority": "medium"
                })
            else:
                null_analysis["recommendations"].append({
                    "column": str(col),  # Ensure string type
                    "action": "simple_imputation",
                    "reason": f"Column has {float(null_percentage):.1f}% missing values - suitable for mean/median/mode imputation",
                    "priority": "low"
                })
    
    return null_analysis

def _suggest_imputation_strategy(series: pd.Series, null_percentage: float) -> str:
    """Suggest the best imputation strategy for a column."""
    if null_percentage > 70:
        return "drop_column"
    elif null_percentage > 50:
        return "knn_imputation"
    elif pd.api.types.is_numeric_dtype(series):
        # Check for skewness to decide between mean and median
        non_null_data = series.dropna()
        if len(non_null_data) > 0:
            skewness = abs(non_null_data.skew()) if hasattr(non_null_data, 'skew') else 0
            return "median" if skewness > 1 else "mean"
        return "mean"
    elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
        return "mode"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "forward_fill"
    else:
        return "mode"

def _apply_imputation_strategy(df: pd.DataFrame, strategy_config: Dict[str, Any]) -> pd.DataFrame:
    """Apply imputation strategies to the dataframe."""
    df_cleaned = df.copy()
    imputation_log = []
    
    # Global strategy application
    global_strategy = strategy_config.get('global_strategy')
    if global_strategy == 'drop_rows':
        initial_rows = int(len(df_cleaned))
        df_cleaned = df_cleaned.dropna()
        rows_dropped = initial_rows - int(len(df_cleaned))
        imputation_log.append(f"Dropped {rows_dropped} rows with any null values")
    
    # Column-specific strategies
    column_strategies = strategy_config.get('column_strategies', {})
    
    for col, strategy in column_strategies.items():
        if col not in df_cleaned.columns:
            continue
            
        null_count_before = int(df_cleaned[col].isnull().sum())
        if null_count_before == 0:
            continue
            
        try:
            if strategy == 'drop_column':
                df_cleaned = df_cleaned.drop(columns=[col])
                imputation_log.append(f"Dropped column '{col}' (had {null_count_before} nulls)")
                
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                mean_val = float(df_cleaned[col].mean())
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                imputation_log.append(f"Filled {null_count_before} nulls in '{col}' with mean ({mean_val:.2f})")
                
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = float(df_cleaned[col].median())
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                imputation_log.append(f"Filled {null_count_before} nulls in '{col}' with median ({median_val:.2f})")
                
            elif strategy == 'mode':
                mode_val = df_cleaned[col].mode()
                if len(mode_val) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])
                    imputation_log.append(f"Filled {null_count_before} nulls in '{col}' with mode ({mode_val.iloc[0]})")
                    
            elif strategy == 'forward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                null_count_after = int(df_cleaned[col].isnull().sum())
                filled_count = null_count_before - null_count_after
                imputation_log.append(f"Forward filled {filled_count} nulls in '{col}'")
                
            elif strategy == 'backward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
                null_count_after = int(df_cleaned[col].isnull().sum())
                filled_count = null_count_before - null_count_after
                imputation_log.append(f"Backward filled {filled_count} nulls in '{col}'")
                
            elif strategy == 'constant':
                fill_value = strategy_config.get('fill_values', {}).get(col, 0)
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                imputation_log.append(f"Filled {null_count_before} nulls in '{col}' with constant ({fill_value})")
                
            elif strategy == 'knn_imputation':
                df_cleaned = _apply_knn_imputation(df_cleaned, [col], strategy_config.get('knn_neighbors', 5))
                null_count_after = int(df_cleaned[col].isnull().sum())
                filled_count = null_count_before - null_count_after
                imputation_log.append(f"KNN imputed {filled_count} nulls in '{col}'")
                
        except Exception as e:
            imputation_log.append(f"Error applying {strategy} to '{col}': {str(e)}")
    
    return df_cleaned, imputation_log

def _apply_knn_imputation(df: pd.DataFrame, columns: List[str], n_neighbors: int = 5) -> pd.DataFrame:
    """Apply KNN imputation to specified columns."""
    df_result = df.copy()
    
    # Select numeric columns for KNN imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [col for col in columns if col in numeric_cols]
    
    if not target_cols:
        return df_result
    
    try:
        # Use only numeric columns for KNN
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_numeric = df[numeric_cols]
        imputed_data = imputer.fit_transform(df_numeric)
        
        # Update only the target columns
        for i, col in enumerate(numeric_cols):
            if col in target_cols:
                df_result[col] = imputed_data[:, i]
                
    except Exception as e:
        # Fallback to simple imputation if KNN fails
        for col in target_cols:
            if pd.api.types.is_numeric_dtype(df_result[col]):
                df_result[col] = df_result[col].fillna(df_result[col].median())
    
    return df_result

def _generate_excel_output(df_cleaned: pd.DataFrame, original_filename: str) -> str:
    """Generate Excel file with cleaned data."""
    try:
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write cleaned data
            df_cleaned.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Original Rows',
                    'Cleaned Rows', 
                    'Rows Removed',
                    'Original Columns',
                    'Cleaned Columns',
                    'Columns Removed',
                    'Null Values Remaining'
                ],
                'Value': [
                    int(len(df_cleaned)),  # This will be updated with original count
                    int(len(df_cleaned)),
                    0,  # This will be calculated
                    int(len(df_cleaned.columns)),  # This will be updated
                    int(len(df_cleaned.columns)),
                    0,  # This will be calculated
                    int(df_cleaned.isnull().sum().sum())
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Cleaning_Summary', index=False)
        
        output.seek(0)
        excel_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
        
        # Generate filename
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
        new_filename = f"{base_name}_cleaned.xlsx"
        
        return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}"
        
    except Exception as e:
        return f"Error generating Excel file: {str(e)}"

def _calculate_cleaning_score(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Calculate cleaning effectiveness score."""
    original_nulls = int(original_df.isnull().sum().sum())
    remaining_nulls = int(cleaned_df.isnull().sum().sum())
    
    # Calculate metrics
    null_reduction_rate = ((original_nulls - remaining_nulls) / original_nulls * 100) if original_nulls > 0 else 100
    data_retention_rate = (int(len(cleaned_df)) / int(len(original_df)) * 100) if int(len(original_df)) > 0 else 0
    column_retention_rate = (int(len(cleaned_df.columns)) / int(len(original_df.columns)) * 100) if int(len(original_df.columns)) > 0 else 0
    
    # Calculate weighted score
    null_weight = float(config.get('null_reduction_weight', 0.5))
    data_weight = float(config.get('data_retention_weight', 0.3))
    column_weight = float(config.get('column_retention_weight', 0.2))
    
    overall_score = (
        float(null_reduction_rate) * null_weight +
        float(data_retention_rate) * data_weight +
        float(column_retention_rate) * column_weight
    )
    
    # Determine cleaning quality
    excellent_threshold = float(config.get('excellent_threshold', 90))
    good_threshold = float(config.get('good_threshold', 75))
    
    if overall_score >= excellent_threshold:
        quality = "excellent"
        quality_color = "green"
    elif overall_score >= good_threshold:
        quality = "good"
        quality_color = "yellow"
    else:
        quality = "needs_improvement"
        quality_color = "red"
    
    return _convert_numpy_types({
        "overall_score": round(overall_score, 1),
        "quality": quality,
        "quality_color": quality_color,
        "metrics": {
            "null_reduction_rate": round(null_reduction_rate, 1),
            "data_retention_rate": round(data_retention_rate, 1),
            "column_retention_rate": round(column_retention_rate, 1),
            "original_nulls": original_nulls,
            "remaining_nulls": remaining_nulls,
            "original_rows": len(original_df),
            "cleaned_rows": len(cleaned_df),
            "original_columns": len(original_df.columns),
            "cleaned_columns": len(cleaned_df.columns)
        },
        "weights_used": {
            "null_reduction_weight": null_weight,
            "data_retention_weight": data_weight,
            "column_retention_weight": column_weight
        }
    })

def _detect_null_row_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect row-level null issues in the DataFrame."""
    issues = []
    
    # Check for rows with null values
    for idx, row in df.iterrows():
        null_cols = row[row.isnull()].index.tolist()
        if null_cols:
            # Limit to first 100 issues for performance
            if len(issues) >= 100:
                break
                
            for col in null_cols:
                issues.append({
                    "row_index": int(idx),
                    "column": str(col),
                    "issue_type": "null_value",
                    "description": f"Null value found in column '{col}'",
                    "severity": "warning",
                    "value": None
                })
    
    return issues

def _generate_routing_info(cleaning_score: Dict[str, Any], remaining_nulls: int) -> Dict[str, Any]:
    """Generate routing information based on cleaning results."""
    quality = cleaning_score.get('quality', 'unknown')
    overall_score = cleaning_score.get('overall_score', 0)
    
    if quality == 'excellent' and remaining_nulls == 0:
        return {
            "status": "Fully Cleaned",
            "reason": f"Dataset successfully cleaned with score {float(overall_score)}/100. No null values remain.",
            "suggestion": "Data is ready for analysis. Consider running quality checks or profiling.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    elif quality == 'excellent' or quality == 'good':
        return {
            "status": "Well Cleaned",
            "reason": f"Dataset cleaned with {quality} quality (score: {float(overall_score)}/100). {int(remaining_nulls)} nulls remain.",
            "suggestion": "Consider additional cleaning if needed, or proceed with analysis.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    else:
        return {
            "status": "Needs More Cleaning",
            "reason": f"Cleaning quality needs improvement (score: {float(overall_score)}/100). {int(remaining_nulls)} nulls remain.",
            "suggestion": "Review cleaning strategies and consider more advanced imputation methods.",
            "suggested_agent_endpoint": "/run-tool/null-handler"
        }

def _process_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Process a single dataframe for null handling."""
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows_processed": 0, "nulls_handled": 0},
            "routing": {
                "status": "No Data",
                "reason": "Dataset is empty, no null handling needed.",
                "suggestion": "Provide a dataset with data to clean.",
                "suggested_agent_endpoint": "/run-tool/profile-my-data"
            },
            "data": {
                "cleaning_score": {"overall_score": 0, "quality": "no_data"},
                "summary": "Dataset is empty, no null handling performed."
            },
            "alerts": [{"level": "warning", "message": "Dataset is empty"}]
        }
    
    # Analyze null patterns
    null_analysis = _analyze_null_patterns(df)
    
    # Track row-level issues before cleaning
    row_level_issues = _detect_null_row_issues(df)
    
    # Apply cleaning strategies
    df_cleaned, imputation_log = _apply_imputation_strategy(df, config)
    
    # Calculate cleaning effectiveness
    cleaning_score = _calculate_cleaning_score(df, df_cleaned, config)
    
    # Generate alerts
    alerts = []
    remaining_nulls = int(df_cleaned.isnull().sum().sum())
    
    if remaining_nulls > 0:
        alerts.append({
            "level": "warning",
            "message": f"{int(remaining_nulls)} null values remain after cleaning",
            "type": "remaining_nulls",
            "details": {"remaining_nulls": _convert_numpy_types(remaining_nulls)}
        })
    
    if cleaning_score['quality'] == 'needs_improvement':
        alerts.append({
            "level": "warning",
            "message": f"Cleaning quality is below expectations (score: {float(cleaning_score['overall_score'])}/100)",
            "type": "low_quality",
            "details": cleaning_score['metrics']
        })
    
    # Generate summary
    original_nulls = cleaning_score['metrics']['original_nulls']
    nulls_handled = original_nulls - remaining_nulls
    
    summary = f"Null handling completed. Quality: {cleaning_score['quality']} (score: {float(cleaning_score['overall_score'])}/100). "
    summary += f"Processed {int(len(df))} rows, handled {int(nulls_handled)} null values, {int(remaining_nulls)} nulls remain. "
    summary += f"Applied {int(len(imputation_log))} cleaning operations."
    
    # Generate routing info
    routing_info = _generate_routing_info(cleaning_score, remaining_nulls)
    
    return {
        "status": "success",
        "metadata": _convert_numpy_types({
            "total_rows_processed": len(df_cleaned),
            "nulls_handled": nulls_handled,
            "original_nulls": original_nulls,
            "remaining_nulls": remaining_nulls,
            "total_issues": len(row_level_issues)
        }),
        "routing": routing_info,
        "data": {
            "cleaning_score": cleaning_score,
            "null_analysis": null_analysis,
            "imputation_log": imputation_log,
            "summary": summary,
            "cleaned_data_shape": list(df_cleaned.shape),
            "original_data_shape": list(df.shape),
            "row_level_issues": row_level_issues[:100]  # Limit to first 100 issues for performance
        },
        "alerts": alerts,
        "cleaned_dataframe": df_cleaned
    }

def _generate_excel_export(response: dict) -> dict:
    """
    Generate Excel export blob with complete JSON response.
    
    Args:
        response: Complete JSON response containing source_file, agent, audit, results, and summary
        
    Returns:
        dict: Excel export metadata and base64-encoded blob
    """
    import json
    
    # Extract components from response
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "NullHandler")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        # Create Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Response Overview
            overview_data = {
                "Field": ["Source File", "Agent", "Summary"],
                "Value": [
                    filename,
                    agent_name,
                    summary[:500] + "..." if len(summary) > 500 else summary  # Truncate long summaries
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name="Response Overview", index=False)
            
            # Sheet 2: Audit Summary
            if audit_trail:
                audit_summary_data = {
                    "Metric": [
                        "Agent Name",
                        "Agent Version", 
                        "Timestamp",
                        "Compute Time (seconds)",
                        "Total Sheets Processed",
                        "Total Rows Processed",
                        "Total Nulls Handled",
                        "Total Alerts Generated",
                        "Critical Alerts",
                        "Warning Alerts",
                        "Info Alerts"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", 0),
                        audit_trail.get("scores", {}).get("total_sheets_processed", 0),
                        audit_trail.get("scores", {}).get("total_rows_processed", 0),
                        audit_trail.get("scores", {}).get("total_nulls_handled", 0),
                        audit_trail.get("scores", {}).get("total_alerts_generated", 0),
                        audit_trail.get("scores", {}).get("critical_alerts", 0),
                        audit_trail.get("scores", {}).get("warning_alerts", 0),
                        audit_trail.get("scores", {}).get("info_alerts", 0)
                    ]
                }
                audit_summary_df = pd.DataFrame(audit_summary_data)
                audit_summary_df.to_excel(writer, sheet_name="Audit Summary", index=False)
            
            # Sheet 3: Fields Scanned
            if audit_trail.get("fields_scanned"):
                fields_df = pd.DataFrame({
                    "Field Name": audit_trail["fields_scanned"]
                })
                fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 4: Findings
            if audit_trail.get("findings"):
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 5: Actions
            if audit_trail.get("actions"):
                actions_df = pd.DataFrame({
                    "Action": audit_trail["actions"]
                })
                actions_df.to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 6: Configuration/Overrides
            if audit_trail.get("overrides"):
                config_data = {
                    "Parameter": list(audit_trail["overrides"].keys()),
                    "Value": list(audit_trail["overrides"].values())
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 7: Routing Information
            routing_data = []
            for sheet_name, sheet_results in results.items():
                if "routing" in sheet_results:
                    routing_info = sheet_results["routing"]
                    routing_data.append({
                        "Sheet/Dataset": sheet_name,
                        "Status": routing_info.get("status", ""),
                        "Reason": routing_info.get("reason", ""),
                        "Suggestion": routing_info.get("suggestion", ""),
                        "Suggested Agent Endpoint": routing_info.get("suggested_agent_endpoint", "")
                    })
            
            if routing_data:
                routing_df = pd.DataFrame(routing_data)
                routing_df.to_excel(writer, sheet_name="Routing", index=False)
            
            # Sheet 8+: Detailed cleaning results per sheet
            for sheet_name, sheet_result in results.items():
                if sheet_result.get("status") == "success" and "data" in sheet_result:
                    data = sheet_result["data"]
                    
                    # Cleaning summary for this sheet
                    if "cleaning_score" in data:
                        cleaning_score = data["cleaning_score"]
                        cleaning_data = {
                            "Metric": [
                                "Overall Score",
                                "Quality",
                                "Null Reduction Rate (%)",
                                "Data Retention Rate (%)",
                                "Column Retention Rate (%)",
                                "Original Nulls",
                                "Remaining Nulls",
                                "Original Rows",
                                "Cleaned Rows",
                                "Original Columns",
                                "Cleaned Columns"
                            ],
                            "Value": [
                                cleaning_score.get("overall_score", 0),
                                cleaning_score.get("quality", "unknown"),
                                cleaning_score.get("metrics", {}).get("null_reduction_rate", 0),
                                cleaning_score.get("metrics", {}).get("data_retention_rate", 0),
                                cleaning_score.get("metrics", {}).get("column_retention_rate", 0),
                                cleaning_score.get("metrics", {}).get("original_nulls", 0),
                                cleaning_score.get("metrics", {}).get("remaining_nulls", 0),
                                cleaning_score.get("metrics", {}).get("original_rows", 0),
                                cleaning_score.get("metrics", {}).get("cleaned_rows", 0),
                                cleaning_score.get("metrics", {}).get("original_columns", 0),
                                cleaning_score.get("metrics", {}).get("cleaned_columns", 0)
                            ]
                        }
                        safe_name = sheet_name[:25]
                        pd.DataFrame(cleaning_data).to_excel(writer, sheet_name=f"{safe_name}_Cleaning", index=False)
                        
                    # Add sheet-level metadata
                    if "metadata" in sheet_result:
                        metadata_data = {
                            "Metric": list(sheet_result["metadata"].keys()),
                            "Value": list(sheet_result["metadata"].values())
                        }
                        metadata_df = pd.DataFrame(metadata_data)
                        metadata_df.to_excel(writer, sheet_name=f"Meta_{safe_name}", index=False)
            
            # Sheet: Complete JSON Response (for reference)
            json_data = {
                "Component": ["Complete JSON Response"],
                "JSON Data": [json.dumps(response, indent=2, default=str)]
            }
            json_df = pd.DataFrame(json_data)
            json_df.to_excel(writer, sheet_name="Raw JSON", index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        # Build dynamic sheet list
        sheets_included = ["Response Overview"]
        if audit_trail:
            sheets_included.append("Audit Summary")
        if audit_trail.get("fields_scanned"):
            sheets_included.append("Fields Scanned")
        if audit_trail.get("findings"):
            sheets_included.append("Findings")
        if audit_trail.get("actions"):
            sheets_included.append("Actions")
        if audit_trail.get("overrides"):
            sheets_included.append("Configuration")
        
        # Check if any routing data exists
        has_routing = any("routing" in sheet_results for sheet_results in results.values())
        if has_routing:
            sheets_included.append("Routing")
        
        # Add cleaning result sheets
        for sheet_name in results.keys():
            safe_sheet_name = sheet_name[:25]
            sheets_included.extend([f"{safe_sheet_name}_Cleaning", f"Meta_{safe_sheet_name}"])
        
        sheets_included.append("Raw JSON")
        
        sheets_list = [sheet for sheet in sheets_included if sheet]  # Remove None values
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_null_handling_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": sheets_list,
            "download_ready": True
        }
        
    except Exception as e:
        # If Excel generation fails, return error info
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_null_handling_report.xlsx",
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }

def handle_nulls(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """Main function for the NullHandler agent."""
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    # Config should always be provided by routes.py from config.json
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not provided. This should be loaded from config.json by the route handler.")
    
    # Track processing data
    all_sheets_processed = []
    total_nulls_handled = 0
    total_rows_processed = 0
    
    try:
        # File handling logic
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_sheets_processed.append(sheet_name)
            
            sheet_result = _process_dataframe(df, config)
            results[sheet_name] = sheet_result
            
            total_nulls_handled += sheet_result['metadata']['nulls_handled']
            total_rows_processed += sheet_result['metadata']['total_rows_processed']
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                all_sheets_processed.append(sheet_name)
                
                sheet_result = _process_dataframe(df, config)
                results[sheet_name] = sheet_result
                
                total_nulls_handled += sheet_result['metadata']['nulls_handled']
                total_rows_processed += sheet_result['metadata']['total_rows_processed']
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Store cleaned data for future agents as Excel file blob
    updated_excel_export = {}
    
    # Generate Excel file with all cleaned data
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_result in results.items():
                if 'cleaned_dataframe' in sheet_result:
                    df_cleaned = sheet_result['cleaned_dataframe']
                    # Write each cleaned dataframe as a separate sheet
                    safe_sheet_name = sheet_name[:31]  # Excel sheet name limit
                    df_cleaned.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        # Create the updated_excel_export structure
        updated_excel_export = {
            "filename": f"{filename.rsplit('.', 1)[0]}_cleaned_data.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64
        }
        
    except Exception as e:
        # Fallback if Excel generation fails
        updated_excel_export = {
            "filename": f"{filename.rsplit('.', 1)[0]}_cleaned_data.xlsx",
            "size_bytes": 0,
            "format": "xlsx",
            "base64_data": "",
            "error": f"Failed to generate cleaned data Excel: {str(e)}"
        }
    
    # Extract audit trail data
    all_fields_scanned = []
    all_findings = []
    total_alerts_generated = 0
    
    for sheet_name, sheet_result in results.items():
        # Extract fields from each sheet
        if 'data' in sheet_result and 'null_analysis' in sheet_result['data']:
            null_analysis = sheet_result['data']['null_analysis']
            if 'columns_with_nulls' in null_analysis:
                all_fields_scanned.extend(null_analysis['columns_with_nulls'])
        
        # Extract findings from alerts
        alerts = sheet_result.get('alerts', [])
        total_alerts_generated += len(alerts)
        for alert in alerts:
            finding = {
                "severity": alert.get("level", "info"),
                "sheet": sheet_name,
                "issue": alert.get("message", ""),
                "category": "null_handling",
                "type": alert.get("type", "unknown")
            }
            all_findings.append(finding)
    
    # Populate overrides with actual configuration values being used
    effective_overrides = {
        "imputation_strategy": config.get("imputation_strategy"),
        "numeric_strategy": config.get("numeric_strategy"),
        "categorical_strategy": config.get("categorical_strategy"),
        "datetime_strategy": config.get("datetime_strategy"),
        "knn_neighbors": config.get("knn_neighbors"),
        "quality_threshold": config.get("quality_threshold")
    }
    
    # Build audit trail
    audit_trail = {
        "agent_name": "NullHandler",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),  # Keep for backward compatibility
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),  # Unique fields
        "findings": all_findings,
        "actions": [
            f"Processed {total_rows_processed} rows across {len(results)} sheet(s)",
            f"Handled {total_nulls_handled} null values",
            "Applied configured imputation strategies",
            "Generated cleaned dataset",
            "Created Excel export with cleaned data"
        ],
        "scores": _convert_numpy_types({
            "total_sheets_processed": len(results),
            "total_rows_processed": total_rows_processed,
            "total_nulls_handled": total_nulls_handled,
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info')
        }),
        "overrides": effective_overrides,
        "lineage": {}
    }
    
    # Remove cleaned_dataframe from results before JSON serialization
    for sheet_name, sheet_result in results.items():
        if 'cleaned_dataframe' in sheet_result:
            del sheet_result['cleaned_dataframe']
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("NullHandler", results, audit_trail)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "NullHandler",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "updated_excel_export": updated_excel_export  # Add cleaned data for future agents
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob
    
    return _convert_numpy_types(response)