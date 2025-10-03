import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timezone
from fastapi import HTTPException
from scipy.stats import entropy
import warnings

AGENT_VERSION = "1.1.0"

# --- Helper functions for profiling specific data types ---

def _profile_numeric(series: pd.Series):
    """Profiles a numeric series for stats and outliers."""
    stats = {}
    alerts = []
    
    if series.empty:
        return stats, alerts

    desc = series.describe()
    stats = {
        "min": desc.get("min", 0),
        "max": desc.get("max", 0),
        "mean": desc.get("mean", 0),
        "median": desc.get("50%", 0),
        "std_dev": desc.get("std", 0),
        "variance": series.var(),
        "p25": desc.get("25%", 0),
        "p75": desc.get("75%", 0),
    }

    # Real Outlier Detection (IQR Method)
    q1, q3 = stats['p25'], stats['p75']
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    stats["outlier_count"] = len(outliers)
    if not outliers.empty:
        stats["outlier_sample"] = outliers.head(5).tolist()
        if (len(outliers) / len(series.dropna())) > 0.1: # If more than 10% are outliers
            alerts.append({
                "level": "warning",
                "message": f"Column '{series.name}' has a high number of outliers ({len(outliers)} values)."
            })
            
    return {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in stats.items()}, alerts

def _profile_temporal(series: pd.Series):
    """Profiles a datetime series."""
    stats = {}
    alerts = []
    
    if series.empty:
        return stats, alerts
        
    series_dt = pd.to_datetime(series, errors='coerce').dropna()
    if not series_dt.empty:
        stats = {
            "earliest_date": str(series_dt.min().date()),
            "latest_date": str(series_dt.max().date()),
            "time_span_days": (series_dt.max() - series_dt.min()).days
        }
    return stats, alerts

def _profile_categorical(series: pd.Series):
    """Profiles a categorical series."""
    stats = {}
    alerts = []
    
    if series.empty:
        return stats, alerts
        
    value_counts = series.value_counts()
    stats = {
        "unique_values_count": series.nunique(),
        "top_values": [{"value": str(k), "count": int(v)} for k, v in value_counts.head(10).items()],
        "entropy": entropy(value_counts)
    }

    if stats["unique_values_count"] == 1 and len(series) > 1:
        alerts.append({
            "level": "info",
            "message": f"Column '{series.name}' contains only one unique value."
        })
        
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}, alerts

# --- Main agent logic ---

def _profile_dataframe(df: pd.DataFrame):
    """
    Generates the full agentic response for a single DataFrame.
    """
    if df.empty:
        return {"status": "success", "metadata": {"total_rows": 0}, "data": {"columns": {}}, "alerts": []}
        
    all_columns_profile = {}
    all_alerts = []

    for col_name in df.columns:
        series = df[col_name].dropna()
        col_profile = {
            "field_name": col_name,
            "data_type": str(df[col_name].dtype),
            "null_count": int(df[col_name].isnull().sum()),
            "null_percentage": round((df[col_name].isnull().sum() / len(df)) * 100, 2) if len(df) > 0 else 0
        }
        
        # Intelligent type-based profiling
        dtype = str(df[col_name].dtype)
        if pd.api.types.is_numeric_dtype(df[col_name]):
            col_profile["inferred_type"] = "Numeric"
            stats, alerts = _profile_numeric(series)
            col_profile.update(stats)
            all_alerts.extend(alerts)
        elif pd.api.types.is_datetime64_any_dtype(df[col_name]) or 'date' in col_name.lower():
            col_profile["inferred_type"] = "Temporal"
            stats, alerts = _profile_temporal(df[col_name]) # Pass full series to handle dates correctly
            col_profile.update(stats)
            all_alerts.extend(alerts)
        else: # Treat as categorical
            col_profile["inferred_type"] = "Categorical"
            stats, alerts = _profile_categorical(series)
            col_profile.update(stats)
            all_alerts.extend(alerts)
            
        all_columns_profile[col_name] = col_profile

    return {
        "status": "success",
        "metadata": {"total_rows": len(df)},
        "alerts": all_alerts,
        "data": {"columns": all_columns_profile}
    }

def profile_fields(file_contents: bytes, filename: str):
    """
    Main function for the Field Profiler agent.
    """
    start_time = time.time()
    file_extension = filename.split('.')[-1].lower()
    results = {}

    try:
        # File handling logic
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = _profile_dataframe(df)
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None, engine='openpyxl')
            for sheet_name, df in xls_sheets.items():
                results[sheet_name] = _profile_dataframe(df)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format for profiler: {file_extension}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time

    return {
        "source_file": filename,
        "agent": "FieldProfiler",
        "audit": {
            "profile_date": datetime.now(timezone.utc).isoformat(),
            "agent_version": AGENT_VERSION,
            "compute_time_seconds": round(compute_time, 2)
        },
        "results": results
    }