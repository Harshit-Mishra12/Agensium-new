"""
Unified Profiler Agent
Combines schema scanning and field profiling into a single comprehensive agent.
Follows the "Compute, Explain, Decide" agentic model.
"""

import pandas as pd
import numpy as np
import io
import time
import warnings
from datetime import datetime, timezone
from scipy.stats import entropy as scipy_entropy
from fastapi import HTTPException

from app.config import AGENT_ROUTES

AGENT_VERSION = "1.0.0"


def profile_dataset(file_contents: bytes, filename: str, config: dict, user_overrides: dict = None) -> dict:
    """
    Unified profiling agent that performs comprehensive schema scanning and field statistics.
    
    Args:
        file_contents: Raw bytes of the uploaded file
        filename: Name of the uploaded file
        config: Configuration dictionary containing thresholds and parameters
        user_overrides: Optional dict of user-provided threshold overrides for audit trail
        
    Returns:
        dict: Standardized JSON response with profile results, alerts, routing, and Excel export blob
    """
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    
    # Suppress pandas UserWarning during date inference
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
    
    try:
        # Determine file type and load data
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_contents))
            sheets = {filename.rsplit('.', 1)[0]: df}
        elif filename.lower().endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(io.BytesIO(file_contents))
            sheets = {sheet_name: excel_file.parse(sheet_name) for sheet_name in excel_file.sheet_names}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV and Excel files are supported.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Build results for each sheet/dataset
    results = {}
    all_fields_scanned = []
    all_findings = []
    total_columns_profiled = 0
    total_alerts_generated = 0
    
    for sheet_name, df in sheets.items():
        total_rows = len(df)
        
        if total_rows == 0:
            results[sheet_name] = {
                "status": "error",
                "metadata": {"total_rows": 0},
                "alerts": [{
                    "level": "critical",
                    "field": None,
                    "message": "Dataset is empty. No rows to profile."
                }],
                "routing": {
                    "status": "Needs Review",
                    "reason": "Empty dataset detected",
                    "suggestion": "Upload a file with data rows",
                    "suggested_agent_endpoint": None
                },
                "data": {"columns": {}}
            }
            continue
        
        # Initialize containers
        columns_profile = {}
        alerts = []
        fields_scanned = list(df.columns)
        all_fields_scanned.extend(fields_scanned)
        
        # Profile each column
        for col_name in df.columns:
            column_profile = _profile_column(df[col_name], col_name, total_rows, config, alerts)
            columns_profile[col_name] = column_profile
            total_columns_profiled += 1
        
        # Determine routing based on alerts
        routing = _determine_routing(alerts)
        total_alerts_generated += len(alerts)
        
        # Collect findings for audit trail
        findings = _extract_findings(columns_profile, alerts)
        all_findings.extend(findings)
        
        results[sheet_name] = {
            "status": "success",
            "metadata": {
                "total_rows": total_rows
            },
            "alerts": alerts,
            "routing": routing,
            "data": {
                "columns": columns_profile
            }
        }
    
    # Calculate compute time
    compute_time = round(time.time() - start_time, 2)
    
    # Build comprehensive audit trail
    audit_trail = {
        "agent_name": "UnifiedProfiler",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),  # Keep for backward compatibility
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": compute_time,
        "fields_scanned": list(set(all_fields_scanned)),  # Unique fields across all sheets
        "findings": all_findings,
        "actions": [
            f"Profiled {total_columns_profiled} columns across {len(results)} sheet(s)",
            f"Generated {total_alerts_generated} alert(s)",
            "Performed semantic type inference on all fields",
            "Calculated statistical measures for numeric fields",
            "Detected outliers using IQR method",
            "Computed entropy for categorical/text fields"
        ],
        "scores": {
            "total_columns_profiled": total_columns_profiled,
            "total_sheets_analyzed": len(results),
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info')
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Generate Excel export blob
    excel_blob = _generate_excel_export(results, filename, audit_trail, config)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "UnifiedProfiler",
        "audit": audit_trail,
        "results": results,
        "excel_export": excel_blob
    }
    
    return response


def _profile_column(series: pd.Series, col_name: str, total_rows: int, config: dict, alerts: list) -> dict:
    """
    Profile a single column with comprehensive statistics.
    
    Args:
        series: Pandas Series representing the column
        col_name: Name of the column
        total_rows: Total number of rows in the dataset
        config: Configuration dictionary
        alerts: List to append alerts to
        
    Returns:
        dict: Comprehensive profile for the column
    """
    profile = {
        "field_name": col_name,
        "data_type": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "null_percentage": round((series.isna().sum() / total_rows) * 100, 1)
    }
    
    # Check for high null percentage
    if profile["null_percentage"] > config.get("null_alert_threshold", 20.0):
        alerts.append({
            "level": "warning",
            "field": col_name,
            "message": f"High null percentage: {profile['null_percentage']}% of values are missing."
        })
    
    # Get non-null values for further analysis
    non_null_series = series.dropna()
    non_null_count = len(non_null_series)
    
    if non_null_count == 0:
        profile["semantic_type"] = "Unknown"
        alerts.append({
            "level": "warning",
            "field": col_name,
            "message": "Column contains only null values."
        })
        return profile
    
    # Infer semantic type and compute statistics
    semantic_type, stats = _infer_semantic_type_and_stats(non_null_series, col_name, total_rows, config, alerts)
    profile["semantic_type"] = semantic_type
    profile.update(stats)
    
    return profile


def _infer_semantic_type_and_stats(series: pd.Series, col_name: str, total_rows: int, config: dict, alerts: list) -> tuple:
    """
    Infer semantic type and compute appropriate statistics.
    
    Args:
        series: Non-null Pandas Series
        col_name: Name of the column
        total_rows: Total number of rows
        config: Configuration dictionary
        alerts: List to append alerts to
        
    Returns:
        tuple: (semantic_type, stats_dict)
    """
    dtype = series.dtype
    
    # Boolean type - treat as categorical
    if pd.api.types.is_bool_dtype(dtype):
        stats = _compute_categorical_stats(series, col_name, config, alerts)
        return "Categorical", stats
    
    # Numeric type (excluding boolean)
    if pd.api.types.is_numeric_dtype(dtype):
        stats = _compute_numeric_stats(series, col_name, config, alerts)
        return "Numeric", stats
    
    # Object type - could be Temporal, Categorical, or Free Text
    if dtype == 'object':
        # Try to parse as datetime
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                datetime_series = pd.to_datetime(series, errors='coerce')
            
            # If most values parse successfully, it's temporal
            if datetime_series.notna().sum() / len(series) > 0.8:
                stats = _compute_temporal_stats(datetime_series.dropna(), col_name)
                return "Temporal", stats
        except:
            pass
        
        # Determine if Categorical or Free Text
        unique_count = series.nunique()
        categorical_threshold = config.get("categorical_threshold", 50)
        categorical_ratio = config.get("categorical_ratio_threshold", 0.5)
        
        if unique_count < categorical_threshold or (unique_count / total_rows) < categorical_ratio:
            stats = _compute_categorical_stats(series, col_name, config, alerts)
            return "Categorical", stats
        else:
            stats = _compute_text_stats(series, col_name, config, alerts)
            return "Free Text", stats
    
    # Datetime type
    if pd.api.types.is_datetime64_any_dtype(dtype):
        stats = _compute_temporal_stats(series, col_name)
        return "Temporal", stats
    
    # Default fallback
    stats = _compute_categorical_stats(series, col_name, config, alerts)
    return "Categorical", stats


def _compute_numeric_stats(series: pd.Series, col_name: str, config: dict, alerts: list) -> dict:
    """Compute statistics for numeric columns."""
    # Convert boolean to int if needed (safety check)
    if pd.api.types.is_bool_dtype(series.dtype):
        series = series.astype(int)
    
    stats = {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": round(float(series.mean()), 2),
        "median": float(series.median()),
        "std_dev": round(float(series.std()), 2),
        "variance": round(float(series.var()), 2),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75))
    }
    
    # Check for low variance
    if stats["variance"] < 0.01 and stats["variance"] > 0:
        alerts.append({
            "level": "info",
            "field": col_name,
            "message": f"Low variance detected: {stats['variance']}. Values are very similar."
        })
    
    # Detect outliers using IQR method
    q1 = stats["p25"]
    q3 = stats["p75"]
    iqr = q3 - q1
    multiplier = config.get("outlier_iqr_multiplier", 1.5)
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_count = len(outliers)
    stats["outlier_count"] = outlier_count
    
    if outlier_count > 0:
        stats["outlier_sample"] = [float(x) for x in outliers.head(5).tolist()]
        
        # Check if outlier percentage exceeds threshold
        outlier_ratio = outlier_count / len(series)
        outlier_threshold = config.get("outlier_alert_threshold", 0.05)
        
        if outlier_ratio > outlier_threshold:
            alerts.append({
                "level": "warning",
                "field": col_name,
                "message": f"High outlier count: {outlier_count} outliers detected ({round(outlier_ratio * 100, 1)}% of non-null values)."
            })
    else:
        stats["outlier_sample"] = []
    
    return stats


def _compute_temporal_stats(series: pd.Series, col_name: str) -> dict:
    """Compute statistics for temporal columns."""
    earliest = series.min()
    latest = series.max()
    time_span = (latest - earliest).days
    
    stats = {
        "earliest_date": earliest.isoformat() if pd.notna(earliest) else None,
        "latest_date": latest.isoformat() if pd.notna(latest) else None,
        "time_span_days": int(time_span) if pd.notna(time_span) else 0
    }
    
    return stats


def _compute_categorical_stats(series: pd.Series, col_name: str, config: dict, alerts: list) -> dict:
    """Compute statistics for categorical columns."""
    unique_count = series.nunique()
    value_counts = series.value_counts()
    
    # Get top N values
    top_n = config.get("top_n_values", 10)
    top_values = [
        {"value": str(val), "count": int(count)}
        for val, count in value_counts.head(top_n).items()
    ]
    
    # Compute entropy
    probabilities = value_counts / len(series)
    entropy_value = round(float(scipy_entropy(probabilities)), 3)
    
    stats = {
        "unique_values_count": unique_count,
        "top_values": top_values,
        "entropy": entropy_value
    }
    
    # Check for single unique value
    if unique_count == 1:
        alerts.append({
            "level": "info",
            "field": col_name,
            "message": f"Column has only one unique value: '{top_values[0]['value']}'. Consider removing this constant column."
        })
    
    return stats


def _compute_text_stats(series: pd.Series, col_name: str, config: dict, alerts: list) -> dict:
    """Compute statistics for free text columns."""
    unique_count = series.nunique()
    value_counts = series.value_counts()
    
    # Get top N values
    top_n = config.get("top_n_values", 10)
    top_values = [
        {"value": str(val)[:100], "count": int(count)}  # Truncate long text
        for val, count in value_counts.head(top_n).items()
    ]
    
    # Compute entropy
    probabilities = value_counts / len(series)
    entropy_value = round(float(scipy_entropy(probabilities)), 3)
    
    stats = {
        "unique_values_count": unique_count,
        "top_values": top_values,
        "entropy": entropy_value
    }
    
    # Check for single unique value
    if unique_count == 1:
        alerts.append({
            "level": "info",
            "field": col_name,
            "message": f"Column has only one unique value. Consider removing this constant column."
        })
    
    return stats


def _determine_routing(alerts: list) -> dict:
    """
    Determine routing based on generated alerts.
    
    Args:
        alerts: List of alert dictionaries
        
    Returns:
        dict: Routing object with status, reason, suggestion, and endpoint
    """
    # Check for warning or critical alerts
    has_warnings = any(alert["level"] in ["warning", "critical"] for alert in alerts)
    
    if has_warnings:
        return {
            "status": "Needs Review",
            "reason": "Data quality issues detected (high nulls, outliers, or empty dataset).",
            "suggestion": "Review the alerts and consider using the data cleaning tool to address issues.",
            "suggested_agent_endpoint": AGENT_ROUTES.get("clean_data_tool", "/tools/clean-data")
        }
    else:
        return {
            "status": "Ready",
            "reason": "No significant data quality issues detected.",
            "suggestion": "Proceed to master data tool for entity resolution and deduplication.",
            "suggested_agent_endpoint": AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
        }


def _extract_findings(columns_profile: dict, alerts: list) -> list:
    """
    Extract structured findings from column profiles and alerts for audit trail.
    
    Args:
        columns_profile: Dictionary of column profiles
        alerts: List of alert dictionaries
        
    Returns:
        list: Structured findings with details about issues detected
    """
    findings = []
    
    # Convert alerts to findings with additional context
    for alert in alerts:
        finding = {
            "severity": alert["level"],
            "field": alert["field"],
            "issue": alert["message"],
            "category": _categorize_alert(alert["message"])
        }
        
        # Add specific details from column profile if available
        if alert["field"] and alert["field"] in columns_profile:
            col_profile = columns_profile[alert["field"]]
            finding["data_type"] = col_profile.get("data_type")
            finding["semantic_type"] = col_profile.get("semantic_type")
            finding["null_percentage"] = col_profile.get("null_percentage")
            
            # Add outlier details if present
            if "outlier_count" in col_profile:
                finding["outlier_count"] = col_profile["outlier_count"]
                finding["outlier_sample"] = col_profile.get("outlier_sample", [])
        
        findings.append(finding)
    
    return findings


def _categorize_alert(message: str) -> str:
    """Categorize alert message into a category."""
    message_lower = message.lower()
    if "null" in message_lower or "missing" in message_lower:
        return "data_completeness"
    elif "outlier" in message_lower:
        return "data_quality"
    elif "unique value" in message_lower or "constant" in message_lower:
        return "data_variance"
    elif "variance" in message_lower:
        return "statistical_anomaly"
    elif "empty" in message_lower:
        return "data_availability"
    else:
        return "general"


def _generate_excel_export(results: dict, filename: str, audit_trail: dict, config: dict) -> dict:
    """
    Generate Excel export blob with profiling results.
    
    Args:
        results: Profiling results dictionary
        filename: Source filename
        audit_trail: Audit trail information
        config: Configuration used
        
    Returns:
        dict: Excel export metadata and base64-encoded blob
    """
    import base64
    from io import BytesIO
    
    try:
        # Create Excel writer
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Summary
            summary_data = {
                "Metric": [
                    "Source File",
                    "Agent",
                    "Version",
                    "Timestamp",
                    "Compute Time (seconds)",
                    "Total Sheets Analyzed",
                    "Total Columns Profiled",
                    "Total Alerts Generated",
                    "Critical Alerts",
                    "Warning Alerts",
                    "Info Alerts"
                ],
                "Value": [
                    filename,
                    audit_trail["agent_name"],
                    audit_trail["agent_version"],
                    audit_trail["timestamp"],
                    audit_trail["compute_time_seconds"],
                    audit_trail["scores"]["total_sheets_analyzed"],
                    audit_trail["scores"]["total_columns_profiled"],
                    audit_trail["scores"]["total_alerts_generated"],
                    audit_trail["scores"]["critical_alerts"],
                    audit_trail["scores"]["warning_alerts"],
                    audit_trail["scores"]["info_alerts"]
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Sheet 2: Fields Scanned
            fields_df = pd.DataFrame({
                "Field Name": audit_trail["fields_scanned"]
            })
            fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 3: Findings
            if audit_trail["findings"]:
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 4: Actions
            actions_df = pd.DataFrame({
                "Action": audit_trail["actions"]
            })
            actions_df.to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 5: Configuration
            config_data = {
                "Parameter": list(config.keys()),
                "Value": list(config.values())
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 6+: Detailed profiles per sheet
            for sheet_name, sheet_results in results.items():
                if sheet_results["status"] == "success" and "data" in sheet_results:
                    columns_data = []
                    for col_name, col_profile in sheet_results["data"]["columns"].items():
                        row = {
                            "Field Name": col_name,
                            "Data Type": col_profile.get("data_type"),
                            "Semantic Type": col_profile.get("semantic_type"),
                            "Null Count": col_profile.get("null_count"),
                            "Null %": col_profile.get("null_percentage")
                        }
                        
                        # Add type-specific stats
                        if col_profile.get("semantic_type") == "Numeric":
                            row.update({
                                "Min": col_profile.get("min"),
                                "Max": col_profile.get("max"),
                                "Mean": col_profile.get("mean"),
                                "Median": col_profile.get("median"),
                                "Std Dev": col_profile.get("std_dev"),
                                "Outliers": col_profile.get("outlier_count")
                            })
                        elif col_profile.get("semantic_type") in ["Categorical", "Free Text"]:
                            row.update({
                                "Unique Values": col_profile.get("unique_values_count"),
                                "Entropy": col_profile.get("entropy")
                            })
                        elif col_profile.get("semantic_type") == "Temporal":
                            row.update({
                                "Earliest": col_profile.get("earliest_date"),
                                "Latest": col_profile.get("latest_date"),
                                "Time Span (days)": col_profile.get("time_span_days")
                            })
                        
                        columns_data.append(row)
                    
                    if columns_data:
                        profile_df = pd.DataFrame(columns_data)
                        safe_sheet_name = sheet_name[:31]  # Excel sheet name limit
                        profile_df.to_excel(writer, sheet_name=f"Profile_{safe_sheet_name}", index=False)
            
            # Sheet: Overrides (if any)
            if audit_trail["overrides"]:
                overrides_data = {
                    "Parameter": list(audit_trail["overrides"].keys()),
                    "User Value": list(audit_trail["overrides"].values())
                }
                overrides_df = pd.DataFrame(overrides_data)
                overrides_df.to_excel(writer, sheet_name="User Overrides", index=False)
        
        # Get the Excel file as bytes
        output.seek(0)
        excel_bytes = output.read()
        
        # Encode to base64
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_profile_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": [
                "Summary",
                "Fields Scanned",
                "Findings",
                "Actions",
                "Configuration",
                "User Overrides" if audit_trail["overrides"] else None
            ] + [f"Profile_{sheet}" for sheet in results.keys()],
            "download_ready": True
        }
        
    except Exception as e:
        return {
            "filename": None,
            "size_bytes": 0,
            "format": "xlsx",
            "base64_data": None,
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }
