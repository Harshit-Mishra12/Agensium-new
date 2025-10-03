import pandas as pd
import numpy as np
import io
import time
import sqlparse
from datetime import datetime, timezone
from fastapi import HTTPException
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

# Import the central route registry
from app.config import AGENT_ROUTES

AGENT_VERSION = "1.3.1" # Version updated for bug fix

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
        "suggestion": "Review schema changes and update baseline definitions." if alerts else "Schemas are consistent.",
        "suggested_agent_endpoint": AGENT_ROUTES.get('define_data_tool') if alerts else AGENT_ROUTES.get('clean_data_tool')
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
            elif psi > PSI_ALERT_THRESHOLD:
                drift_detected = True
                alerts.append({"level": "warning", "message": f"Moderate distribution drift in categorical column '{col}' (PSI: {psi:.2f})."})
            
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
            "suggestion": "Data appears consistent. As a next step, run the cleaning tool.",
            "suggested_agent_endpoint": AGENT_ROUTES.get('clean_data_tool')
        }

    return {
        "status": "success",
        "metadata": {"baseline_rows": len(baseline_df), "current_rows": len(current_df)},
        "alerts": alerts, "routing": routing, "data": {"columns": drift_details}
    }

# --- Main Agent Function ---

def detect_drift(baseline_contents: bytes, current_contents: bytes, baseline_filename: str, current_filename: str):
    """
    Main function for the Drift Detector agent. Handles multiple file types.
    """
    start_time = time.time()
    
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

    return {
        "source_file": {"baseline": baseline_filename, "current": current_filename},
        "agent": "DriftDetector",
        "audit": {
            "profile_date": datetime.now(timezone.utc).isoformat(),
            "agent_version": AGENT_VERSION,
            "compute_time_seconds": round(compute_time, 2)
        },
        "results": results
    }

