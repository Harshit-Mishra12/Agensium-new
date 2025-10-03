import pandas as pd
import io
import time
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings

# Import the central route registry
from app.config import AGENT_ROUTES

# Define the current version of the agent
AGENT_VERSION = "1.1.0"

def _calculate_readiness_score(df: pd.DataFrame):
    """
    Calculates the readiness score and generates explanations for a single DataFrame.
    """
    if df.empty:
        scores = {"overall": 0, "completeness": 0, "consistency": 0, "schema_health": 0}
        deductions = ["Dataset is empty, resulting in a score of 0."]
        return scores, deductions

    deductions = []
    
    # 1. Completeness Score (based on nulls)
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0
    completeness_score = 100 - null_percentage
    if null_cells > 0:
        deductions.append(f"Completeness: Score reduced by {null_percentage:.1f} points due to {null_percentage:.1f}% null values.")

    # 2. Consistency Score (based on duplicate rows)
    duplicate_rows = df.duplicated().sum()
    total_rows = len(df)
    duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
    consistency_score = 100 - duplicate_percentage
    if duplicate_rows > 0:
        deductions.append(f"Consistency: Score reduced by {duplicate_percentage:.1f} points due to {duplicate_rows} duplicate rows ({duplicate_percentage:.1f}%).")

    # 3. Schema Health Score (heuristic-based)
    schema_health_score = 100
    # Penalize for columns with very low variance
    for col in df.columns:
        if df[col].nunique() == 1 and len(df) > 1:
            schema_health_score -= 10
            deductions.append(f"Schema Health: Score reduced by 10 points because column '{col}' has only one unique value.")
    
    # Penalize for potential mixed-type object columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for col in df.select_dtypes(include=['object']).columns:
            if pd.to_numeric(df[col].dropna().iloc[:100], errors='coerce').notna().any():
                schema_health_score -= 5
                deductions.append(f"Schema Health: Score reduced by 5 points for potential mixed data types in column '{col}'.")
    
    schema_health_score = max(0, schema_health_score)

    # 4. Overall Score (weighted average)
    overall_score = (completeness_score * 0.4) + (consistency_score * 0.4) + (schema_health_score * 0.2)

    scores = {
        "overall": round(overall_score),
        "completeness": round(completeness_score),
        "consistency": round(consistency_score),
        "schema_health": round(schema_health_score)
    }
    
    return scores, deductions

def _profile_dataframe(df: pd.DataFrame):
    """
    Generates the full agentic response for a single DataFrame.
    """
    scores, deductions = _calculate_readiness_score(df)
    overall_score = scores['overall']
    alerts = []
    
    # Threshold-based routing and alerts
    if overall_score >= 85:
        routing_status = "Ready"
        reason = "Dataset meets readiness criteria."
        suggestion = "Proceed with downstream analytics or ML pipelines."
        endpoint = None # No immediate action needed
    elif overall_score >= 70:
        routing_status = "Needs Review"
        reason = "Dataset has moderate quality issues that should be reviewed."
        suggestion = "Run the 'Clean My Data' tool to address issues."
        endpoint = AGENT_ROUTES['clean_data_tool']
        alerts.append({"level": "warning", "message": f"Overall readiness score is {overall_score}. Manual review is recommended."})
    else:
        routing_status = "Not Ready"
        reason = "Dataset has significant quality issues and is not ready for use."
        suggestion = "Run the 'Clean My Data' tool to fix critical issues."
        endpoint = AGENT_ROUTES['clean_data_tool']
        alerts.append({"level": "critical", "message": f"Overall readiness score is {overall_score}. Data is not recommended for production use without cleaning."})

    routing = {
        "status": routing_status,
        "reason": reason,
        "suggestion": suggestion,
        "suggested_agent_endpoint": endpoint
    }
    
    return {
        "status": "success",
        "metadata": {"total_rows_analyzed": len(df)},
        "routing": routing,
        "alerts": alerts,
        "data": {
            "readiness_score": scores,
            "deductions": deductions
        }
    }

def rate_readiness(file_contents: bytes, filename: str):
    """
    Main function for the Readiness Rater agent.
    """
    start_time = time.time()
    file_extension = filename.split('.')[-1].lower()
    results = {}

    try:
        # File handling logic remains the same
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = _profile_dataframe(df)
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                results[sheet_name] = _profile_dataframe(df)
        # Add other file types (json, parquet) here if needed
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time

    return {
        "source_file": filename,
        "agent": "ReadinessRater",
        "audit": {
            "profile_date": datetime.now(timezone.utc).isoformat(),
            "agent_version": AGENT_VERSION,
            "compute_time_seconds": round(compute_time, 2)
        },
        "results": results
    }

