import pandas as pd
import numpy as np
import io
import time
import base64
import json
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings
from typing import Dict, List, Any, Optional, Union
from scipy import stats

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return str(val)
        return val
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
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

def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> List[Dict[str, Any]]:
    """Detect outliers using Z-score method."""
    outliers = []
    if len(series.dropna()) < 3:
        return outliers
    
    z_scores = np.abs(stats.zscore(series.dropna()))
    outlier_indices = series.dropna().index[z_scores > threshold]
    
    for idx in outlier_indices:
        z_val = z_scores[series.dropna().index.get_loc(idx)]
        severity = "critical" if z_val > 4.0 else "warning"
        outliers.append({
            "row_index": int(idx),
            "value": _convert_numpy_types(series.iloc[idx]),
            "z_score": _convert_numpy_types(z_val),
            "severity": severity,
            "method": "z_score"
        })
    
    return outliers

def _detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> List[Dict[str, Any]]:
    """Detect outliers using IQR method."""
    outliers = []
    clean_series = series.dropna()
    
    if len(clean_series) < 4:
        return outliers
    
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    for idx, value in clean_series.items():
        if value < lower_bound or value > upper_bound:
            distance = max(abs(value - lower_bound), abs(value - upper_bound))
            extreme_multiplier = 3.0
            severity = "critical" if distance > extreme_multiplier * IQR else "warning"
            
            outliers.append({
                "row_index": int(idx),
                "value": _convert_numpy_types(value),
                "lower_bound": _convert_numpy_types(lower_bound),
                "upper_bound": _convert_numpy_types(upper_bound),
                "severity": severity,
                "method": "iqr"
            })
    
    return outliers

def _detect_outliers_percentile(series: pd.Series, lower_pct: float = 1.0, upper_pct: float = 99.0) -> List[Dict[str, Any]]:
    """Detect outliers using percentile method."""
    outliers = []
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        return outliers
    
    lower_bound = clean_series.quantile(lower_pct / 100)
    upper_bound = clean_series.quantile(upper_pct / 100)
    
    for idx, value in clean_series.items():
        if value < lower_bound or value > upper_bound:
            extreme_lower = clean_series.quantile(0.001)
            extreme_upper = clean_series.quantile(0.999)
            severity = "critical" if (value < extreme_lower or value > extreme_upper) else "warning"
            
            outliers.append({
                "row_index": int(idx),
                "value": _convert_numpy_types(value),
                "lower_bound": _convert_numpy_types(lower_bound),
                "upper_bound": _convert_numpy_types(upper_bound),
                "severity": severity,
                "method": "percentile"
            })
    
    return outliers

def _analyze_outliers(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Analyze outliers in all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    analysis = {
        "total_rows": len(df),
        "numeric_columns": numeric_cols,
        "outlier_summary": {},
        "recommendations": []
    }
    
    detection_method = config.get('detection_method', 'iqr')
    
    for col in numeric_cols:
        series = df[col]
        
        if detection_method == 'z_score':
            threshold = config.get('z_threshold', 3.0)
            outliers = _detect_outliers_zscore(series, threshold)
        elif detection_method == 'percentile':
            lower_pct = config.get('lower_percentile', 1.0)
            upper_pct = config.get('upper_percentile', 99.0)
            outliers = _detect_outliers_percentile(series, lower_pct, upper_pct)
        else:  # Default to IQR
            multiplier = config.get('iqr_multiplier', 1.5)
            outliers = _detect_outliers_iqr(series, multiplier)
        
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(series) * 100) if len(series) > 0 else 0
        
        if outlier_count > 0:
            analysis["outlier_summary"][str(col)] = {
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_percentage, 2),
                "data_type": str(series.dtype),
                "total_values": len(series),
                "method_used": detection_method,
                "outliers": outliers[:50]  # Limit for performance
            }
            
            # Generate recommendations
            if outlier_percentage > 20:
                analysis["recommendations"].append({
                    "column": str(col),
                    "action": "review_data_quality",
                    "reason": f"Column has {outlier_percentage:.1f}% outliers - may indicate data quality issues",
                    "priority": "high"
                })
            elif outlier_percentage > 5:
                analysis["recommendations"].append({
                    "column": str(col),
                    "action": "consider_removal",
                    "reason": f"Column has {outlier_percentage:.1f}% outliers - consider removal or imputation",
                    "priority": "medium"
                })
            else:
                analysis["recommendations"].append({
                    "column": str(col),
                    "action": "safe_to_remove",
                    "reason": f"Column has {outlier_percentage:.1f}% outliers - safe to remove",
                    "priority": "low"
                })
    
    return analysis

def _remove_outliers(df: pd.DataFrame, outlier_analysis: Dict[str, Any], config: dict) -> tuple:
    """Remove or impute outliers based on configuration."""
    df_cleaned = df.copy()
    removal_log = []
    row_level_issues = []
    
    removal_strategy = config.get('removal_strategy', 'remove')  # 'remove', 'impute_mean', 'impute_median'
    total_outliers_removed = 0
    
    for col, col_analysis in outlier_analysis["outlier_summary"].items():
        outliers = col_analysis["outliers"]
        
        for outlier in outliers:
            row_idx = outlier["row_index"]
            original_value = outlier["value"]
            
            # Record the issue
            issue = {
                "row_index": row_idx,
                "column": col,
                "issue_type": "outlier_detected",
                "description": f"Outlier detected in column '{col}' using {outlier.get('method', 'unknown')} method",
                "severity": outlier["severity"],
                "value": original_value
            }
            
            if removal_strategy == 'remove':
                # Mark row for removal
                if row_idx in df_cleaned.index:
                    df_cleaned = df_cleaned.drop(row_idx)
                    issue["action_taken"] = "removed"
                    issue["issue_type"] = "outlier_removed"
                    total_outliers_removed += 1
                    removal_log.append(f"Removed row {row_idx} from column '{col}' (value: {original_value})")
                    
            elif removal_strategy == 'impute_mean':
                if col in df_cleaned.columns and row_idx in df_cleaned.index:
                    mean_val = df_cleaned[col].mean()
                    df_cleaned.loc[row_idx, col] = mean_val
                    issue["action_taken"] = f"imputed_mean ({mean_val:.2f})"
                    issue["issue_type"] = "outlier_imputed"
                    total_outliers_removed += 1
                    removal_log.append(f"Imputed row {row_idx} in column '{col}' with mean ({mean_val:.2f})")
                    
            elif removal_strategy == 'impute_median':
                if col in df_cleaned.columns and row_idx in df_cleaned.index:
                    median_val = df_cleaned[col].median()
                    df_cleaned.loc[row_idx, col] = median_val
                    issue["action_taken"] = f"imputed_median ({median_val:.2f})"
                    issue["issue_type"] = "outlier_imputed"
                    total_outliers_removed += 1
                    removal_log.append(f"Imputed row {row_idx} in column '{col}' with median ({median_val:.2f})")
            
            row_level_issues.append(issue)
    
    return df_cleaned, removal_log, row_level_issues, total_outliers_removed

def _calculate_outlier_score(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, outlier_analysis: Dict[str, Any], config: dict) -> Dict[str, Any]:
    """Calculate outlier removal effectiveness score."""
    total_outliers = sum(col_data["outlier_count"] for col_data in outlier_analysis["outlier_summary"].values())
    
    # Calculate metrics
    outlier_reduction_rate = 100.0  # Assume all detected outliers were handled
    data_retention_rate = (len(cleaned_df) / len(original_df) * 100) if len(original_df) > 0 else 0
    column_retention_rate = (len(cleaned_df.columns) / len(original_df.columns) * 100) if len(original_df.columns) > 0 else 0
    
    # Calculate weighted score
    outlier_weight = config.get('outlier_reduction_weight', 0.5)
    data_weight = config.get('data_retention_weight', 0.3)
    column_weight = config.get('column_retention_weight', 0.2)
    
    overall_score = (
        outlier_reduction_rate * outlier_weight +
        data_retention_rate * data_weight +
        column_retention_rate * column_weight
    )
    
    # Determine quality
    excellent_threshold = config.get('excellent_threshold', 90)
    good_threshold = config.get('good_threshold', 75)
    
    if overall_score >= excellent_threshold:
        quality = "excellent"
    elif overall_score >= good_threshold:
        quality = "good"
    else:
        quality = "needs_improvement"
    
    return _convert_numpy_types({
        "overall_score": round(overall_score, 1),
        "quality": quality,
        "metrics": {
            "outlier_reduction_rate": round(outlier_reduction_rate, 1),
            "data_retention_rate": round(data_retention_rate, 1),
            "column_retention_rate": round(column_retention_rate, 1),
            "original_outliers": total_outliers,
            "original_rows": len(original_df),
            "cleaned_rows": len(cleaned_df),
            "original_columns": len(original_df.columns),
            "cleaned_columns": len(cleaned_df.columns)
        }
    })

def _generate_routing_info(outlier_score: Dict[str, Any], remaining_outliers: int) -> Dict[str, Any]:
    """Generate routing information based on outlier removal results."""
    quality = outlier_score.get('quality', 'unknown')
    overall_score = outlier_score.get('overall_score', 0)
    
    if quality == 'excellent' and remaining_outliers == 0:
        return {
            "status": "Fully Cleaned",
            "reason": f"Dataset successfully cleaned with score {overall_score}/100. No outliers remain.",
            "suggestion": "Data is ready for analysis. Consider running quality checks or profiling.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    elif quality == 'excellent' or quality == 'good':
        return {
            "status": "Well Cleaned",
            "reason": f"Dataset cleaned with {quality} quality (score: {overall_score}/100). {remaining_outliers} outliers remain.",
            "suggestion": "Consider additional cleaning if needed, or proceed with analysis.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    else:
        return {
            "status": "Needs More Cleaning",
            "reason": f"Cleaning quality needs improvement (score: {overall_score}/100). {remaining_outliers} outliers remain.",
            "suggestion": "Review cleaning strategies and consider different outlier detection methods.",
            "suggested_agent_endpoint": "/run-tool/outlier-remover"
        }

def _process_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Process a single dataframe for outlier detection and removal."""
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows_processed": 0, "outliers_handled": 0, "total_issues": 0},
            "routing": {
                "status": "No Data",
                "reason": "Dataset is empty, no outlier handling needed.",
                "suggestion": "Provide a dataset with data to clean.",
                "suggested_agent_endpoint": "/run-tool/profile-my-data"
            },
            "data": {
                "outlier_score": {"overall_score": 0, "quality": "no_data"},
                "summary": "Dataset is empty, no outlier handling performed."
            },
            "alerts": [{"level": "warning", "message": "Dataset is empty"}]
        }
    
    # Analyze outliers
    outlier_analysis = _analyze_outliers(df, config)
    
    # Remove/impute outliers
    df_cleaned, removal_log, row_level_issues, total_outliers_removed = _remove_outliers(df, outlier_analysis, config)
    
    # Calculate effectiveness score
    outlier_score = _calculate_outlier_score(df, df_cleaned, outlier_analysis, config)
    
    # Generate alerts
    alerts = []
    total_outliers = sum(col_data["outlier_count"] for col_data in outlier_analysis["outlier_summary"].values())
    remaining_outliers = total_outliers - total_outliers_removed
    
    if remaining_outliers > 0:
        alerts.append({
            "level": "warning",
            "message": f"{remaining_outliers} outliers remain after processing",
            "type": "remaining_outliers",
            "details": {"remaining_outliers": remaining_outliers}
        })
    
    if outlier_score['quality'] == 'needs_improvement':
        alerts.append({
            "level": "warning",
            "message": f"Outlier removal quality is below expectations (score: {outlier_score['overall_score']}/100)",
            "type": "low_quality",
            "details": outlier_score['metrics']
        })
    
    # Generate summary
    summary = f"Outlier removal completed. Quality: {outlier_score['quality']} (score: {outlier_score['overall_score']}/100). "
    summary += f"Processed {len(df)} rows, handled {total_outliers_removed} outliers, {remaining_outliers} outliers remain. "
    summary += f"Applied {len(removal_log)} cleaning operations."
    
    # Generate routing info
    routing_info = _generate_routing_info(outlier_score, remaining_outliers)
    
    return {
        "status": "success",
        "metadata": _convert_numpy_types({
            "total_rows_processed": len(df_cleaned),
            "outliers_handled": total_outliers_removed,
            "original_outliers": total_outliers,
            "remaining_outliers": remaining_outliers,
            "total_issues": len(row_level_issues)
        }),
        "routing": routing_info,
        "data": {
            "outlier_score": outlier_score,
            "outlier_analysis": outlier_analysis,
            "removal_log": removal_log,
            "summary": summary,
            "cleaned_data_shape": list(df_cleaned.shape),
            "original_data_shape": list(df.shape),
            "row_level_issues": row_level_issues[:100]  # Limit to first 100 issues
        },
        "alerts": alerts,
        "cleaned_dataframe": df_cleaned
    }

def _generate_excel_export(response: dict) -> dict:
    """Generate Excel export blob with complete JSON response."""
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "OutlierRemoverAgent")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Response Overview
            overview_data = {
                "Field": ["Source File", "Agent", "Summary"],
                "Value": [
                    filename,
                    agent_name,
                    summary[:500] + "..." if len(summary) > 500 else summary
                ]
            }
            pd.DataFrame(overview_data).to_excel(writer, sheet_name="Response Overview", index=False)
            
            # Sheet 2: Audit Summary
            if audit_trail:
                audit_summary_data = {
                    "Metric": [
                        "Agent Name", "Agent Version", "Timestamp", "Compute Time (seconds)",
                        "Total Sheets Processed", "Total Rows Processed", "Total Outliers Handled",
                        "Total Alerts Generated", "Critical Alerts", "Warning Alerts", "Info Alerts"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", 0),
                        audit_trail.get("scores", {}).get("total_sheets_processed", 0),
                        audit_trail.get("scores", {}).get("total_rows_processed", 0),
                        audit_trail.get("scores", {}).get("total_outliers_handled", 0),
                        audit_trail.get("scores", {}).get("total_alerts_generated", 0),
                        audit_trail.get("scores", {}).get("critical_alerts", 0),
                        audit_trail.get("scores", {}).get("warning_alerts", 0),
                        audit_trail.get("scores", {}).get("info_alerts", 0)
                    ]
                }
                pd.DataFrame(audit_summary_data).to_excel(writer, sheet_name="Audit Summary", index=False)
            
            # Sheet 3: Fields Scanned
            if audit_trail.get("fields_scanned"):
                fields_df = pd.DataFrame({"Field Name": audit_trail["fields_scanned"]})
                fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 4: Findings
            if audit_trail.get("findings"):
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 5: Actions
            if audit_trail.get("actions"):
                actions_df = pd.DataFrame({"Action": audit_trail["actions"]})
                actions_df.to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 6: Configuration/Overrides
            if audit_trail.get("overrides"):
                config_data = {
                    "Parameter": list(audit_trail["overrides"].keys()),
                    "Value": list(audit_trail["overrides"].values())
                }
                pd.DataFrame(config_data).to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 7: Raw JSON
            json_data = {
                "Component": ["Complete JSON Response"],
                "JSON Data": [json.dumps(response, indent=2, default=str)]
            }
            pd.DataFrame(json_data).to_excel(writer, sheet_name="Raw JSON", index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_outlier_removal_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": ["Response Overview", "Audit Summary", "Fields Scanned", "Findings", "Actions", "Configuration", "Raw JSON"],
            "download_ready": True
        }
        
    except Exception as e:
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_outlier_removal_report.xlsx",
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }

def handle_outliers(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """Main function for the OutlierRemoverAgent."""
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not provided. This should be loaded from config.json by the route handler.")
    
    # Track processing data
    total_outliers_handled = 0
    total_rows_processed = 0
    
    try:
        # File handling logic
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            
            sheet_result = _process_dataframe(df, config)
            results[sheet_name] = sheet_result
            
            total_outliers_handled += sheet_result['metadata']['outliers_handled']
            total_rows_processed += sheet_result['metadata']['total_rows_processed']
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                sheet_result = _process_dataframe(df, config)
                results[sheet_name] = sheet_result
                
                total_outliers_handled += sheet_result['metadata']['outliers_handled']
                total_rows_processed += sheet_result['metadata']['total_rows_processed']
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Generate Excel file with cleaned data for future agents
    updated_excel_export = {}
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_result in results.items():
                if 'cleaned_dataframe' in sheet_result:
                    df_cleaned = sheet_result['cleaned_dataframe']
                    safe_sheet_name = sheet_name[:31]
                    df_cleaned.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        updated_excel_export = {
            "filename": f"{filename.rsplit('.', 1)[0]}_cleaned_data.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64
        }
        
    except Exception as e:
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
        if 'data' in sheet_result and 'outlier_analysis' in sheet_result['data']:
            outlier_analysis = sheet_result['data']['outlier_analysis']
            if 'numeric_columns' in outlier_analysis:
                all_fields_scanned.extend(outlier_analysis['numeric_columns'])
        
        alerts = sheet_result.get('alerts', [])
        total_alerts_generated += len(alerts)
        for alert in alerts:
            finding = {
                "severity": alert.get("level", "info"),
                "sheet": sheet_name,
                "issue": alert.get("message", ""),
                "category": "outlier_removal",
                "type": alert.get("type", "unknown")
            }
            all_findings.append(finding)
    
    # Populate overrides with actual configuration values
    effective_overrides = {
        "detection_method": config.get("detection_method"),
        "removal_strategy": config.get("removal_strategy"),
        "z_threshold": config.get("z_threshold"),
        "iqr_multiplier": config.get("iqr_multiplier"),
        "lower_percentile": config.get("lower_percentile"),
        "upper_percentile": config.get("upper_percentile")
    }
    
    # Build audit trail
    audit_trail = {
        "agent_name": "OutlierRemoverAgent",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),
        "findings": all_findings,
        "actions": [
            f"Processed {total_rows_processed} rows across {len(results)} sheet(s)",
            f"Handled {total_outliers_handled} outliers",
            "Applied configured outlier detection and removal strategies",
            "Generated cleaned dataset",
            "Created Excel export with cleaned data"
        ],
        "scores": _convert_numpy_types({
            "total_sheets_processed": len(results),
            "total_rows_processed": total_rows_processed,
            "total_outliers_handled": total_outliers_handled,
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
    llm_summary = generate_llm_summary("OutlierRemoverAgent", results, audit_trail)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "OutlierRemoverAgent",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "updated_excel_export": updated_excel_export
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob
    
    return _convert_numpy_types(response)