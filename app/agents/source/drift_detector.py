import pandas as pd
import io
from scipy.stats import ks_2samp
from fastapi import HTTPException

def _detect_drift_for_sheet(baseline_df: pd.DataFrame, current_df: pd.DataFrame):
    """
    Compares two DataFrames (baseline vs current) and detects drift.
    """
    if baseline_df.empty or current_df.empty:
        return {
            "metadata": {},
            "data": {"columns": {}, "message": "One or both datasets are empty."}
        }

    columns_drift = {}
    baseline_cols = set(baseline_df.columns)
    current_cols = set(current_df.columns)

    # Check for new and dropped columns
    for col in current_cols - baseline_cols:
        columns_drift[col] = {"schema_change": f"new column detected: {col}"}
    for col in baseline_cols - current_cols:
        columns_drift[col] = {"schema_change": f"column dropped: {col}"}

    # Check for drift in common columns
    for col in baseline_cols.intersection(current_cols):
        b_series = baseline_df[col].dropna()
        c_series = current_df[col].dropna()
        drift_info = {}

        if pd.api.types.is_numeric_dtype(b_series) and pd.api.types.is_numeric_dtype(c_series):
            stat, p_value = ks_2samp(b_series, c_series)
            drift_info = {"drift_score": round(stat, 2), "p_value": round(p_value, 2)}
            if b_series.mean() < c_series.mean():
                drift_info["direction"] = "increase in mean"
            else:
                drift_info["direction"] = "decrease in mean"
        else: # Categorical/Text
             b_counts = b_series.nunique()
             c_counts = c_series.nunique()
             drift_info = {"drift_score": round(abs(c_counts - b_counts) / b_counts, 2) if b_counts > 0 else 1}
             if c_counts > b_counts:
                 drift_info["direction"] = "new categories appeared"
        
        columns_drift[col] = drift_info

    return {
        "metadata": {"baseline_rows": len(baseline_df), "current_rows": len(current_df)},
        "data": {"columns": columns_drift}
    }


def detect_drift(baseline_contents: bytes, current_contents: bytes, filename: str):
    """
    Detects drift between a baseline and current file in a standardized format.
    """
    # For simplicity, this example only handles single-sheet CSVs for drift.
    # It can be extended to handle multi-sheet Excel like the other agents.
    try:
        baseline_df = pd.read_csv(io.BytesIO(baseline_contents))
        current_df = pd.read_csv(io.BytesIO(current_contents))
        
        sheet_name = filename.rsplit('.', 1)[0] # Use a representative name
        result = {"status": "success", **_detect_drift_for_sheet(baseline_df, current_df)}

        return {
            "source_files": {"baseline": f"baseline_{filename}", "current": f"current_{filename}"},
            "agent": "DriftDetector",
            "results": {sheet_name: result}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files. Error: {str(e)}")
