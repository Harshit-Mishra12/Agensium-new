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

def _determine_semantic_type(series: pd.Series):
    """
    Determines a richer semantic type for a column based on its data.
    """
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Could be enhanced to detect identifiers vs. measures
        return "Numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "Datetime"
    
    # For object/text types, distinguish between categorical and free text
    if series.dtype == 'object':
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        # Heuristic: if a text field has relatively few unique values, it's categorical.
        if unique_ratio < 0.5 and series.nunique() < 100:
            return "Categorical"
        else:
            return "Free Text"
            
    return "Other"


def _profile_dataframe(df: pd.DataFrame):
    """
    Generates a schema summary, alerts, and routing for a single DataFrame.
    """
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows": 0},
            "routing": {"status": "Ready", "reason": "Sheet is empty."},
            "alerts": [],
            "data": {"summary_table": []}
        }
        
    schema_summary = []
    alerts = []
    high_null_fields = []

    for col in df.columns:
        # 1. Null Counts (Absolute and Percentage)
        null_count = int(df[col].isnull().sum())
        total_count = len(df)
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0

        # 2. Data Type and Richer Semantic Type
        col_type = str(df[col].dtype)
        semantic_type = "Unknown"
        # Suppress pandas UserWarning for date parsing attempts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if 'object' in col_type:
                try:
                    # Attempt to convert a sample to datetime to infer type
                    pd.to_datetime(df[col].dropna().iloc[:100], errors='raise')
                    data_type = 'Date'
                    semantic_type = 'Datetime'
                except (ValueError, TypeError, IndexError):
                    data_type = 'Text'
                    semantic_type = _determine_semantic_type(df[col])
            elif 'int' in col_type:
                data_type = 'Integer'
                semantic_type = "Numeric"
            elif 'float' in col_type:
                data_type = 'Float'
                semantic_type = "Numeric"
            elif 'datetime' in col_type:
                data_type = 'Date'
                semantic_type = 'Datetime'
            else:
                data_type = col_type
                semantic_type = "Other"

        # 3. Structured Top-N Values with Frequencies
        top_values_counts = df[col].value_counts().nlargest(5)
        top_values_structured = [
            {"value": str(val), "count": int(count)}
            for val, count in top_values_counts.items()
        ]

        # 4. Agent Logic: Generate alerts for high nulls
        if null_percentage > 20:
            high_null_fields.append(col)
            alerts.append({
                "level": "warning",
                "field": col,
                "message": f"High percentage of nulls ({null_percentage:.1f}%) detected in '{col}'. This impacts data completeness."
            })

        schema_summary.append({
            "field": col,
            "data_type": data_type,
            "semantic_type": semantic_type,
            "null_count": null_count,
            "null_percentage": f"{null_percentage:.1f}%",
            "distinct_count": int(df[col].nunique()),
            "top_values": top_values_structured
        })
    
    # 5. Agent Logic: Determine routing based on findings
    if high_null_fields:
        routing = {
            "status": "Needs Review",
            "reason": f"High null percentages detected in fields: {', '.join(high_null_fields)}.",
            "suggestion": "Run the 'Clean My Data' tool to handle missing values.",
            # Use the route from the central config file
            "suggested_agent_endpoint": AGENT_ROUTES['clean_data_tool']
        }
    else:
        routing = {
            "status": "Ready",
            "reason": "Schema appears healthy with no major null value issues.",
            "suggestion": "Proceed with Field Profiler for deeper statistical analysis.",
            # Use the route from the central config file
            "suggested_agent_endpoint": AGENT_ROUTES['profile_fields']
        }

    return {
        "status": "success",
        "metadata": {"total_rows": len(df)},
        "routing": routing,
        "alerts": alerts,
        "data": {"summary_table": schema_summary}
    }

def scan_schema(file_contents: bytes, filename: str):
    """
    Main function for the Schema Scanner agent.
    Orchestrates profiling, adds audit metadata, and formats the final response.
    """
    start_time = time.time()
    file_extension = filename.split('.')[-1].lower()
    results = {}

    try:
        # This structure can be expanded to handle different file types as before
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = _profile_dataframe(df)
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                results[sheet_name] = _profile_dataframe(df)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format for Schema Scanner: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")
    
    end_time = time.time()
    compute_time = end_time - start_time

    # Construct the final, standardized JSON response
    return {
        "source_file": filename,
        "agent": "SchemaScanner",
        "audit": {
            "profile_date": datetime.now(timezone.utc).isoformat(),
            "agent_version": AGENT_VERSION,
            "compute_time_seconds": round(compute_time, 2)
        },
        "results": results
    }