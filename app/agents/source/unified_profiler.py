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
from app.agents.shared.chat_agent import generate_llm_summary

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
        row_level_issues = []
        fields_scanned = list(df.columns)
        all_fields_scanned.extend(fields_scanned)
        
        # Profile each column with row-level issue tracking
        for col_name in df.columns:
            column_profile, col_issues = _profile_column_with_issues(df, col_name, total_rows, config, alerts)
            columns_profile[col_name] = column_profile
            row_level_issues.extend(col_issues)
            total_columns_profiled += 1
        
        # Detect duplicate rows
        duplicate_issues = _detect_duplicate_rows(df)
        row_level_issues.extend(duplicate_issues)
        
        # Determine routing based on alerts
        routing = _determine_routing(alerts)
        total_alerts_generated += len(alerts)
        
        # Collect findings for audit trail
        findings = _extract_findings(columns_profile, alerts)
        all_findings.extend(findings)
        
        results[sheet_name] = {
            "status": "success",
            "metadata": {
                "total_rows": total_rows,
                "total_issues": len(row_level_issues)
            },
            "routing": routing,
            "data": {
                "columns": columns_profile,
                "row_level_issues": row_level_issues[:100]  # Limit to first 100 issues for performance
            },
            "issue_summary": _summarize_issues(row_level_issues)
        }
    
    # Calculate compute time
    compute_time = round(time.time() - start_time, 2)
    
    # Build comprehensive audit trail
    # Populate overrides with actual configuration values being used
    effective_overrides = {
        "null_alert_threshold": config.get("null_alert_threshold"),
        "categorical_threshold": config.get("categorical_threshold"),
        "categorical_ratio_threshold": config.get("categorical_ratio_threshold"),
        "top_n_values": config.get("top_n_values"),
        "outlier_iqr_multiplier": config.get("outlier_iqr_multiplier"),
        "outlier_alert_threshold": config.get("outlier_alert_threshold")
    }
    
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
        "overrides": effective_overrides,
        "lineage": {}
    }
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("UnifiedProfiler", results, audit_trail)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "UnifiedProfiler",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob
    
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


def _generate_excel_export(response: dict) -> dict:
    """
    Generate Excel export blob with complete JSON response.
    
    Args:
        response: Complete JSON response containing source_file, agent, audit, results, and summary
        
    Returns:
        dict: Excel export metadata and base64-encoded blob
    """
    import base64
    from io import BytesIO
    import json
    
    # Extract components from response
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "UnifiedProfiler")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        # Create Excel writer
        output = BytesIO()
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
                        "Total Sheets Analyzed",
                        "Total Columns Profiled",
                        "Total Alerts Generated",
                        "Critical Alerts",
                        "Warning Alerts",
                        "Info Alerts"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", ""),
                        audit_trail.get("scores", {}).get("total_sheets_analyzed", ""),
                        audit_trail.get("scores", {}).get("total_columns_profiled", ""),
                        audit_trail.get("scores", {}).get("total_alerts_generated", ""),
                        audit_trail.get("scores", {}).get("critical_alerts", ""),
                        audit_trail.get("scores", {}).get("warning_alerts", ""),
                        audit_trail.get("scores", {}).get("info_alerts", "")
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
            
            # Sheet 6: Overrides/Configuration
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
            
            # Sheet 8+: Detailed profiles per sheet
            for sheet_name, sheet_results in results.items():
                if sheet_results.get("status") == "success" and "data" in sheet_results:
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
                        
                    # Add sheet-level metadata
                    if "metadata" in sheet_results:
                        metadata_data = {
                            "Metric": list(sheet_results["metadata"].keys()),
                            "Value": list(sheet_results["metadata"].values())
                        }
                        metadata_df = pd.DataFrame(metadata_data)
                        metadata_df.to_excel(writer, sheet_name=f"Meta_{safe_sheet_name}", index=False)
            
            # Sheet: Complete JSON Response (for reference)
            json_data = {
                "Component": ["Complete JSON Response"],
                "JSON Data": [json.dumps(response, indent=2, default=str)]
            }
            json_df = pd.DataFrame(json_data)
            json_df.to_excel(writer, sheet_name="Raw JSON", index=False)
        
        # Get the Excel file as bytes
        output.seek(0)
        excel_bytes = output.read()
        
        # Encode to base64
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
        
        # Add profile sheets
        for sheet_name in results.keys():
            safe_sheet_name = sheet_name[:31]
            sheets_included.extend([f"Profile_{safe_sheet_name}", f"Meta_{safe_sheet_name}"])
        
        sheets_included.append("Raw JSON")
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_profile_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": [sheet for sheet in sheets_included if sheet],  # Remove None values
            "download_ready": True
        }
        
    except Exception as e:
        # If Excel generation fails, return error info
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_profile_report.xlsx",
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }


def _profile_column_with_issues(df: pd.DataFrame, col_name: str, total_rows: int, config: dict, alerts: list) -> tuple:
    """
    Profile a column and track row-level issues.
    
    Args:
        df: Full DataFrame
        col_name: Column name to profile
        total_rows: Total number of rows
        config: Configuration dictionary
        alerts: List to append alerts to
        
    Returns:
        tuple: (column_profile dict, list of row-level issues)
    """
    series = df[col_name]
    issues = []
    
    # Track null value rows
    null_mask = series.isna()
    null_indices = df.index[null_mask].tolist()
    if len(null_indices) > 0:
        # Limit to first 20 null rows for performance
        for idx in null_indices[:20]:
            issues.append({
                "row_index": int(idx),
                "column": col_name,
                "issue_type": "null_value",
                "severity": "warning" if len(null_indices) / total_rows > 0.2 else "info",
                "value": None,
                "message": f"Null value in column '{col_name}'"
            })
    
    # Get the regular column profile
    profile = _profile_column(series, col_name, total_rows, config, alerts)
    
    # Track outliers with row indices (for numeric columns)
    if profile.get("semantic_type") == "Numeric" and profile.get("outlier_count", 0) > 0:
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            q1 = profile.get("p25")
            q3 = profile.get("p75")
            iqr = q3 - q1
            multiplier = config.get("outlier_iqr_multiplier", 1.5)
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = df.index[outlier_mask].tolist()
            
            # Limit to first 20 outliers
            for idx in outlier_indices[:20]:
                value = series.iloc[idx] if idx < len(series) else None
                issues.append({
                    "row_index": int(idx),
                    "column": col_name,
                    "issue_type": "outlier",
                    "severity": "warning",
                    "value": float(value) if pd.notna(value) else None,
                    "message": f"Outlier detected in column '{col_name}': {value}",
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                })
    
    # Track single-value columns
    if profile.get("semantic_type") in ["Categorical", "Free Text"]:
        if profile.get("unique_values_count") == 1 and total_rows > 1:
            issues.append({
                "row_index": None,  # Affects all rows
                "column": col_name,
                "issue_type": "constant_column",
                "severity": "info",
                "value": str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else None,
                "message": f"Column '{col_name}' has only one unique value"
            })
    
    return profile, issues


def _detect_duplicate_rows(df: pd.DataFrame) -> list:
    """
    Detect duplicate rows and return their indices.
    
    Args:
        df: DataFrame to check for duplicates
        
    Returns:
        list: Row-level issues for duplicate rows
    """
    issues = []
    
    # Find duplicate rows
    duplicate_mask = df.duplicated(keep=False)
    duplicate_indices = df.index[duplicate_mask].tolist()
    
    if len(duplicate_indices) > 0:
        # Group duplicates together
        duplicate_groups = {}
        for idx in duplicate_indices:
            row_tuple = tuple(df.iloc[idx].values)
            if row_tuple not in duplicate_groups:
                duplicate_groups[row_tuple] = []
            duplicate_groups[row_tuple].append(int(idx))
        
        # Create issues for duplicate groups (limit to first 10 groups)
        for group_idx, (row_values, indices) in enumerate(list(duplicate_groups.items())[:10]):
            issues.append({
                "row_index": indices,  # List of all duplicate row indices
                "column": None,  # Affects entire row
                "issue_type": "duplicate_row",
                "severity": "warning",
                "value": None,
                "message": f"Duplicate row found at indices {indices[:5]}{'...' if len(indices) > 5 else ''}",
                "duplicate_count": len(indices)
            })
    
    return issues


def _summarize_issues(issues: list) -> dict:
    """
    Summarize row-level issues by type and severity.
    
    Args:
        issues: List of row-level issues
        
    Returns:
        dict: Summary statistics of issues
    """
    summary = {
        "total_issues": len(issues),
        "by_type": {},
        "by_severity": {},
        "by_column": {}
    }
    
    for issue in issues:
        # Count by type
        issue_type = issue.get("issue_type", "unknown")
        summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
        
        # Count by severity
        severity = issue.get("severity", "info")
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        # Count by column
        column = issue.get("column")
        if column:
            summary["by_column"][column] = summary["by_column"].get(column, 0) + 1
    
    return summary
