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
        return float(obj)
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
            "remaining_nulls": remaining_nulls
        }),
        "routing": routing_info,
        "data": {
            "cleaning_score": cleaning_score,
            "null_analysis": null_analysis,
            "imputation_log": imputation_log,
            "summary": summary,
            "cleaned_data_shape": list(df_cleaned.shape),
            "original_data_shape": list(df.shape)
        },
        "alerts": alerts,
        "cleaned_dataframe": df_cleaned
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
    
    # Generate Excel output with cleaned data
    excel_export = ""
    if len(results) == 1:
        # Single sheet - use the cleaned dataframe
        sheet_result = list(results.values())[0]
        if 'cleaned_dataframe' in sheet_result:
            excel_export = _generate_excel_output(sheet_result['cleaned_dataframe'], filename)
    elif len(results) > 1:
        # Multiple sheets - combine or create multi-sheet Excel
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, sheet_result in results.items():
                    if 'cleaned_dataframe' in sheet_result:
                        sheet_result['cleaned_dataframe'].to_excel(writer, sheet_name=f"{sheet_name}_cleaned", index=False)
            
            output.seek(0)
            excel_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            excel_export = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}"
        except Exception as e:
            excel_export = f"Error generating multi-sheet Excel: {str(e)}"
    
    # Build audit trail
    audit_trail = {
        "agent_name": "NullHandler",
        "timestamp": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "sheets_processed": all_sheets_processed,
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
            "total_nulls_handled": total_nulls_handled
        }),
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Remove cleaned_dataframe from results before JSON serialization
    for sheet_name, sheet_result in results.items():
        if 'cleaned_dataframe' in sheet_result:
            del sheet_result['cleaned_dataframe']
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("NullHandler", results, audit_trail)
    
    return _convert_numpy_types({
        "source_file": filename,
        "agent": "NullHandler",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "excel_export": excel_export
    })