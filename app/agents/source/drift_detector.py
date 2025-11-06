import pandas as pd
import numpy as np
import io
import time
import base64
import sqlparse
from datetime import datetime, timezone
from fastapi import HTTPException
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# Import the central route registry
from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.4.0" # Enhanced with audit trail and Excel export

# --- Configuration for Drift Thresholds ---
P_VALUE_THRESHOLD = 0.05
PSI_ALERT_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.25

# --- Helper functions ---

def _calculate_psi(baseline_dist, current_dist):
    """Calculates the Population Stability Index."""
    baseline_dist = baseline_dist.replace(0, 0.0001)
    current_dist = current_dist.replace(0, 0.0001)
    psi_value = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
    return psi_value

def _read_data_to_sheets(file_contents: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Reads various data file formats into a dictionary of DataFrames."""
    file_extension = filename.split('.')[-1].lower()
    if file_extension == 'csv':
        df = pd.read_csv(io.BytesIO(file_contents))
        sheet_name = filename.rsplit('.', 1)[0]
        return {sheet_name: df}
    elif file_extension in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
    elif file_extension == 'json':
        df = pd.read_json(io.BytesIO(file_contents))
        sheet_name = filename.rsplit('.', 1)[0]
        return {sheet_name: df}
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported data file format for drift detection: {file_extension}")

def _parse_sql_schema(sql_contents: bytes) -> dict:
    """Parses a SQL script to extract {table_name: {column_names}}."""
    schema = {}
    sql_script = sql_contents.decode('utf-8', errors='ignore')
    statements = sqlparse.parse(sql_script)
    for stmt in statements:
        if stmt.get_type() == 'CREATE':
            tokens = [t for t in stmt.flatten()]
            try:
                table_keyword_index = [i for i, t in enumerate(tokens) if t.ttype is sqlparse.tokens.Keyword and t.value.upper() == 'TABLE'][0]
                table_name = tokens[table_keyword_index + 2].get_real_name()
                paren = next(t for t in tokens if isinstance(t, sqlparse.sql.Parenthesis))
                columns = {col.strip().split()[0].strip('`"[]') for col in paren.value.strip('()').split(',') if col.strip()}
                schema[table_name] = columns
            except (IndexError, StopIteration):
                continue
    return schema

# --- Comparison Logic ---

def _compare_sql_schemas(baseline_schema: dict, current_schema: dict):
    """Compares two parsed SQL schemas and generates a report."""
    alerts = []
    drift_details = {}
    
    baseline_tables = set(baseline_schema.keys())
    current_tables = set(current_schema.keys())
    
    new_tables = list(current_tables - baseline_tables)
    dropped_tables = list(baseline_tables - current_tables)
    common_tables = list(baseline_tables.intersection(current_tables))
    
    if new_tables:
        alerts.append({"level": "critical", "message": f"Schema changed: {len(new_tables)} new table(s) detected: {', '.join(new_tables)}."})
    if dropped_tables:
        alerts.append({"level": "critical", "message": f"Schema changed: {len(dropped_tables)} table(s) dropped: {', '.join(dropped_tables)}."})
        
    for table in common_tables:
        baseline_cols = baseline_schema.get(table, set())
        current_cols = current_schema.get(table, set())
        
        new_cols = list(current_cols - baseline_cols)
        dropped_cols = list(baseline_cols - current_cols)
        
        if new_cols or dropped_cols:
            drift_details[table] = {"new_columns": new_cols, "dropped_columns": dropped_cols}
            if new_cols:
                 alerts.append({"level": "critical", "message": f"Schema changed in table '{table}': {len(new_cols)} new column(s) detected: {', '.join(new_cols)}."})
            if dropped_cols:
                 alerts.append({"level": "critical", "message": f"Schema changed in table '{table}': {len(dropped_cols)} column(s) dropped: {', '.join(dropped_cols)}."})

    routing = {
        "status": "Needs Review" if alerts else "Ready",
        "reason": "Schema drift detected." if alerts else "No schema drift detected.",
        "suggestion": "Review schema changes and update baseline definitions." if alerts else "Proceed to master data tool for entity resolution and deduplication.",
        "suggested_agent_endpoint": AGENT_ROUTES.get('define_data_tool') if alerts else AGENT_ROUTES.get('master_my_data_tool', '/run-tool/master-my-data')
    }

    return {
        "status": "success", "metadata": {"baseline_tables": len(baseline_tables), "current_tables": len(current_tables)},
        "alerts": alerts, "routing": routing, "data": {"tables": drift_details}
    }


def _compare_dataframes(baseline_df: pd.DataFrame, current_df: pd.DataFrame):
    """Compares two dataframes and generates a detailed drift report."""
    if baseline_df.empty or current_df.empty:
        return {"status": "error", "message": "One or both datasets are empty."}

    baseline_cols = set(baseline_df.columns)
    current_cols = set(current_df.columns)
    
    common_cols = list(baseline_cols.intersection(current_cols))
    new_cols = list(current_cols - current_cols)
    dropped_cols = list(baseline_cols - current_cols)
    
    alerts = []
    drift_details = {}
    row_level_issues = []
    
    if new_cols:
        alerts.append({"level": "critical", "message": f"Schema changed: {len(new_cols)} new column(s) detected: {', '.join(new_cols)}."})
        for col in new_cols:
            drift_details[col] = {"status": "new_column"}
            
    if dropped_cols:
        alerts.append({"level": "critical", "message": f"Schema changed: {len(dropped_cols)} column(s) dropped: {', '.join(dropped_cols)}."})
        for col in dropped_cols:
            drift_details[col] = {"status": "dropped_column"}

    for col in common_cols:
        baseline_series = baseline_df[col].dropna()
        current_series = current_df[col].dropna()
        col_drift = {}

        if pd.api.types.is_numeric_dtype(baseline_series) and pd.api.types.is_numeric_dtype(current_series):
            ks_stat, p_value = ks_2samp(baseline_series, current_series)
            col_drift = {"type": "Numeric", "ks_statistic": round(ks_stat, 2), "p_value": round(p_value, 2)}
            if p_value < P_VALUE_THRESHOLD:
                drift_detected = True
                alerts.append({"level": "critical", "message": f"Significant distribution drift in numeric column '{col}' (p-value: {p_value:.2f})."})
                # Track row-level drift for numeric columns
                drift_issues = _detect_numeric_drift_rows(baseline_df, current_df, col, baseline_series, current_series)
                row_level_issues.extend(drift_issues)
            else:
                drift_detected = False
            col_drift["drift_detected"] = drift_detected

        else:
            baseline_counts = baseline_series.astype(str).value_counts(normalize=True)
            current_counts = current_series.astype(str).value_counts(normalize=True)
            all_categories = set(baseline_counts.index).union(set(current_counts.index))
            
            baseline_dist = baseline_counts.reindex(all_categories, fill_value=0)
            current_dist = current_counts.reindex(all_categories, fill_value=0)
            
            psi = _calculate_psi(baseline_dist, current_dist)
            js_div = jensenshannon(baseline_dist, current_dist)
            
            new_categories = list(set(current_counts.index) - set(baseline_counts.index))
            missing_categories = list(set(baseline_counts.index) - set(current_counts.index))
            
            col_drift = {
                "type": "Categorical", "psi": round(psi, 2), "js_divergence": round(js_div, 2),
                "new_categories": new_categories, "missing_categories": missing_categories
            }
            
            drift_detected = False
            if psi > PSI_CRITICAL_THRESHOLD:
                drift_detected = True
                alerts.append({"level": "critical", "message": f"Critical distribution drift in categorical column '{col}' (PSI: {psi:.2f})."})
                # Track row-level drift for categorical columns
                drift_issues = _detect_categorical_drift_rows(current_df, col, new_categories, missing_categories)
                row_level_issues.extend(drift_issues)
            elif psi > PSI_ALERT_THRESHOLD:
                drift_detected = True
                alerts.append({"level": "warning", "message": f"Moderate distribution drift in categorical column '{col}' (PSI: {psi:.2f})."})
                # Track row-level drift for categorical columns
                drift_issues = _detect_categorical_drift_rows(current_df, col, new_categories, missing_categories)
                row_level_issues.extend(drift_issues)
            
            col_drift["drift_detected"] = drift_detected
        
        drift_details[col] = col_drift

    if any(alert['level'] == 'critical' for alert in alerts):
        routing = {
            "status": "Needs Review", "reason": "Critical schema or data distribution drift detected.",
            "suggestion": "Review drift details and consider updating the baseline model or cleaning the new data.",
            "suggested_agent_endpoint": AGENT_ROUTES.get('define_data_tool') 
        }
    else:
        routing = {
            "status": "Ready", "reason": "No significant data drift detected.",
            "suggestion": "Data appears consistent. Proceed to master data tool for entity resolution and deduplication.",
            "suggested_agent_endpoint": AGENT_ROUTES.get('master_my_data_tool', '/run-tool/master-my-data')
        }

    return {
        "status": "success",
        "metadata": {
            "baseline_rows": len(baseline_df), 
            "current_rows": len(current_df),
            "total_issues": len(row_level_issues)
        },
        "alerts": alerts, 
        "routing": routing, 
        "data": {
            "columns": drift_details,
            "row_level_issues": row_level_issues[:100]  # Limit to first 100
        },
        "issue_summary": _summarize_drift_issues(row_level_issues)
    }

# --- Main Agent Function ---

def detect_drift(baseline_contents: bytes, current_contents: bytes, baseline_filename: str, current_filename: str, config: dict = None, user_overrides: dict = None):
    """
    Main function for the Drift Detector agent. Handles multiple file types.
    
    Args:
        baseline_contents: Raw bytes of baseline file
        current_contents: Raw bytes of current file
        baseline_filename: Name of baseline file
        current_filename: Name of current file
        config: Configuration dictionary (optional)
        user_overrides: User-provided overrides for audit trail (optional)
    
    Returns:
        dict: Standardized JSON response with drift results, audit trail, and Excel export
    """
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    
    if config is None:
        config = {
            'p_value_threshold': P_VALUE_THRESHOLD,
            'psi_alert_threshold': PSI_ALERT_THRESHOLD,
            'psi_critical_threshold': PSI_CRITICAL_THRESHOLD
        }
    
    baseline_ext = baseline_filename.split('.')[-1].lower()
    current_ext = current_filename.split('.')[-1].lower()

    if baseline_ext != current_ext:
        raise HTTPException(status_code=400, detail="Baseline and current files must be of the same type for drift detection.")

    results = {}
    
    try:
        if baseline_ext == 'sql':
            baseline_schema = _parse_sql_schema(baseline_contents)
            current_schema = _parse_sql_schema(current_contents)
            results['sql_schema_comparison'] = _compare_sql_schemas(baseline_schema, current_schema)
        
        elif baseline_ext in ['csv', 'json']:
            # For single-sheet files, ignore sheet names and compare directly
            baseline_sheets = _read_data_to_sheets(baseline_contents, baseline_filename)
            current_sheets = _read_data_to_sheets(current_contents, current_filename)
            
            baseline_df = list(baseline_sheets.values())[0]
            current_df = list(current_sheets.values())[0]
            
            # Use the baseline filename as the key for the result
            result_key = baseline_filename.rsplit('.', 1)[0]
            results[result_key] = _compare_dataframes(baseline_df, current_df)

        elif baseline_ext in ['xlsx', 'xls']:
            # For multi-sheet files, compare sheets with matching names
            baseline_sheets = _read_data_to_sheets(baseline_contents, baseline_filename)
            current_sheets = _read_data_to_sheets(current_contents, current_filename)
            
            common_sheets = set(baseline_sheets.keys()).intersection(set(current_sheets.keys()))
            if not common_sheets:
                raise ValueError("For Excel files, no matching sheet names were found between the baseline and current workbooks.")

            for sheet_name in common_sheets:
                baseline_df = baseline_sheets[sheet_name]
                current_df = current_sheets[sheet_name]
                results[sheet_name] = _compare_dataframes(baseline_df, current_df)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {baseline_ext}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Extract audit trail data
    all_fields_scanned = []
    all_findings = []
    total_alerts = 0
    drift_detected_count = 0
    
    for sheet_result in results.values():
        if sheet_result.get('status') == 'success':
            total_alerts += len(sheet_result.get('alerts', []))
            
            # Extract fields from drift details
            if 'data' in sheet_result:
                if 'columns' in sheet_result['data']:
                    all_fields_scanned.extend(sheet_result['data']['columns'].keys())
                    for col, drift_info in sheet_result['data']['columns'].items():
                        if drift_info.get('drift_detected'):
                            drift_detected_count += 1
                elif 'tables' in sheet_result['data']:
                    all_fields_scanned.extend(sheet_result['data']['tables'].keys())
            
            # Extract findings
            findings = _extract_findings_from_result(sheet_result)
            all_findings.extend(findings)
    
    # Build comprehensive audit trail
    audit_trail = {
        "agent_name": "DriftDetector",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),
        "findings": all_findings,
        "actions": [
            f"Compared baseline '{baseline_filename}' with current '{current_filename}'",
            f"Analyzed {len(results)} dataset(s) for drift",
            f"Generated {total_alerts} alert(s)",
            "Performed statistical drift detection (KS test for numeric, PSI for categorical)",
            "Detected schema changes (new/dropped columns)",
            f"Identified {drift_detected_count} field(s) with significant drift"
        ],
        "scores": {
            "total_datasets_analyzed": len(results),
            "total_alerts_generated": total_alerts,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "fields_with_drift": drift_detected_count,
            "total_fields_compared": len(set(all_fields_scanned))
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Generate Excel export
    excel_blob = _generate_excel_export(results, baseline_filename, current_filename, audit_trail, config)
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("DriftDetector", results, audit_trail)

    return {
        "source_file": {"baseline": baseline_filename, "current": current_filename},
        "agent": "DriftDetector",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "excel_export": excel_blob
    }


def _extract_findings_from_result(sheet_result: dict) -> list:
    """Extract structured findings from sheet result for audit trail."""
    findings = []
    
    for alert in sheet_result.get('alerts', []):
        finding = {
            "severity": alert.get("level", "info"),
            "issue": alert.get("message", ""),
            "category": "drift_detection"
        }
        findings.append(finding)
    
    return findings


def _generate_excel_export(results: dict, baseline_filename: str, current_filename: str, audit_trail: dict, config: dict) -> dict:
    """Generate Excel export blob with drift detection results."""
    from io import BytesIO
    
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Summary
            summary_data = {
                "Metric": [
                    "Baseline File",
                    "Current File",
                    "Agent",
                    "Version",
                    "Timestamp",
                    "Compute Time (seconds)",
                    "Total Datasets Analyzed",
                    "Total Alerts Generated",
                    "Critical Alerts",
                    "Warning Alerts",
                    "Fields with Drift",
                    "Total Fields Compared"
                ],
                "Value": [
                    baseline_filename,
                    current_filename,
                    audit_trail["agent_name"],
                    audit_trail["agent_version"],
                    audit_trail["timestamp"],
                    audit_trail["compute_time_seconds"],
                    audit_trail["scores"]["total_datasets_analyzed"],
                    audit_trail["scores"]["total_alerts_generated"],
                    audit_trail["scores"]["critical_alerts"],
                    audit_trail["scores"]["warning_alerts"],
                    audit_trail["scores"]["fields_with_drift"],
                    audit_trail["scores"]["total_fields_compared"]
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # Sheet 2: Fields Scanned
            if audit_trail["fields_scanned"]:
                pd.DataFrame({"Field Name": audit_trail["fields_scanned"]}).to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 3: Findings
            if audit_trail["findings"]:
                pd.DataFrame(audit_trail["findings"]).to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 4: Actions
            pd.DataFrame({"Action": audit_trail["actions"]}).to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 5: Configuration
            pd.DataFrame({"Parameter": list(config.keys()), "Value": list(config.values())}).to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 6+: Detailed drift results per dataset
            for dataset_name, dataset_result in results.items():
                if dataset_result.get("status") == "success" and "data" in dataset_result:
                    data = dataset_result["data"]
                    
                    if "columns" in data:
                        drift_data = []
                        for col, drift_info in data["columns"].items():
                            row = {
                                "Column": col,
                                "Type": drift_info.get("type", "N/A"),
                                "Drift Detected": drift_info.get("drift_detected", False)
                            }
                            
                            if drift_info.get("type") == "Numeric":
                                row.update({
                                    "KS Statistic": drift_info.get("ks_statistic"),
                                    "P-Value": drift_info.get("p_value")
                                })
                            elif drift_info.get("type") == "Categorical":
                                row.update({
                                    "PSI": drift_info.get("psi"),
                                    "JS Divergence": drift_info.get("js_divergence"),
                                    "New Categories": ", ".join(drift_info.get("new_categories", [])) if drift_info.get("new_categories") else "",
                                    "Missing Categories": ", ".join(drift_info.get("missing_categories", [])) if drift_info.get("missing_categories") else ""
                                })
                            
                            drift_data.append(row)
                        
                        if drift_data:
                            safe_name = dataset_name[:25]
                            pd.DataFrame(drift_data).to_excel(writer, sheet_name=f"{safe_name}_Drift", index=False)
            
            # Sheet: Overrides (if any)
            if audit_trail["overrides"]:
                pd.DataFrame({"Parameter": list(audit_trail["overrides"].keys()), "User Value": list(audit_trail["overrides"].values())}).to_excel(writer, sheet_name="User Overrides", index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        sheets_list = [
            "Summary",
            "Fields Scanned" if audit_trail["fields_scanned"] else None,
            "Findings" if audit_trail["findings"] else None,
            "Actions",
            "Configuration",
            "User Overrides" if audit_trail["overrides"] else None
        ]
        sheets_list = [s for s in sheets_list if s]
        sheets_list.extend([f"{dataset[:25]}_Drift" for dataset in results.keys()])
        
        return {
            "filename": f"{baseline_filename.rsplit('.', 1)[0]}_vs_{current_filename.rsplit('.', 1)[0]}_drift_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": sheets_list,
            "download_ready": True
        }
    except Exception as e:
        return {
            "filename": f"drift_report_{baseline_filename.rsplit('.', 1)[0]}_vs_{current_filename.rsplit('.', 1)[0]}.xlsx",
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }


def _detect_numeric_drift_rows(baseline_df: pd.DataFrame, current_df: pd.DataFrame, col: str, baseline_series: pd.Series, current_series: pd.Series) -> list:
    """
    Detect rows with significant numeric drift.
    
    Args:
        baseline_df: Baseline DataFrame
        current_df: Current DataFrame
        col: Column name
        baseline_series: Baseline series (non-null)
        current_series: Current series (non-null)
    
    Returns:
        list: Row-level drift issues
    """
    issues = []
    
    # Calculate baseline statistics
    baseline_mean = baseline_series.mean()
    baseline_std = baseline_series.std()
    
    # Find rows in current that are significantly different from baseline
    current_full = current_df[col]
    for idx in current_df.index[:20]:  # Limit to first 20
        value = current_full.iloc[idx] if idx < len(current_full) else None
        if pd.notna(value):
            # Check if value is more than 2 std deviations from baseline mean
            z_score = abs((value - baseline_mean) / baseline_std) if baseline_std > 0 else 0
            if z_score > 2:
                issues.append({
                    "row_index": int(idx),
                    "column": col,
                    "issue_type": "numeric_drift",
                    "severity": "warning",
                    "value": float(value),
                    "baseline_mean": round(float(baseline_mean), 2),
                    "baseline_std": round(float(baseline_std), 2),
                    "z_score": round(float(z_score), 2),
                    "message": f"Value {value} in column '{col}' deviates significantly from baseline (z-score: {z_score:.2f})"
                })
    
    return issues


def _detect_categorical_drift_rows(current_df: pd.DataFrame, col: str, new_categories: list, missing_categories: list) -> list:
    """
    Detect rows with categorical drift (new or unexpected categories).
    
    Args:
        current_df: Current DataFrame
        col: Column name
        new_categories: List of new categories not in baseline
        missing_categories: List of categories missing from current
    
    Returns:
        list: Row-level drift issues
    """
    issues = []
    
    # Track rows with new categories
    if new_categories:
        current_series = current_df[col].astype(str)
        for new_cat in new_categories[:10]:  # Limit to first 10 new categories
            matching_rows = current_df.index[current_series == new_cat].tolist()
            for idx in matching_rows[:5]:  # First 5 rows per category
                issues.append({
                    "row_index": int(idx),
                    "column": col,
                    "issue_type": "new_category",
                    "severity": "warning",
                    "value": new_cat,
                    "message": f"New category '{new_cat}' in column '{col}' not present in baseline"
                })
    
    return issues


def _summarize_drift_issues(issues: list) -> dict:
    """
    Summarize drift issues by type and severity.
    
    Args:
        issues: List of row-level drift issues
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_issues": len(issues),
        "by_type": {},
        "by_severity": {},
        "by_column": {}
    }
    
    for issue in issues:
        issue_type = issue.get("issue_type", "unknown")
        summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
        
        severity = issue.get("severity", "info")
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        column = issue.get("column")
        if column:
            summary["by_column"][column] = summary["by_column"].get(column, 0) + 1
    
    return summary
