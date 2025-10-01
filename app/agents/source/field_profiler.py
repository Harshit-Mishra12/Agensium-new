import pandas as pd
import numpy as np
import io
from scipy.stats import entropy
from fastapi import HTTPException

def _profile_field_stats(df: pd.DataFrame):
    """
    Generates detailed field statistics for a single DataFrame.
    """
    if df.empty:
        return {
            "metadata": {"total_rows": 0, "message": "Sheet is empty."},
            "data": {"columns": {}}
        }

    columns_profile = {}
    for col in df.columns:
        stats = {}
        anomalies = {"missing_values": int(df[col].isnull().sum()), "outliers": []}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            stats = {
                "min": float(series.min()), "max": float(series.max()),
                "mean": float(series.mean()), "std_dev": float(series.std()),
                "unique_values": None, "entropy": float(entropy(series.value_counts()))
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            series = df[col].dropna()
            stats = {
                "min": str(series.min()), "max": str(series.max()),
                "temporal_spread_days": int((series.max() - series.min()).days),
                "unique_values": None, "entropy": None
            }
        else: # Categorical/Text
            series = df[col].dropna()
            stats = {
                "min": None, "max": None, "mean": None, "std_dev": None,
                "unique_values": int(series.nunique()), "entropy": float(entropy(series.value_counts()))
            }
        
        columns_profile[col] = {"stats": stats, "anomalies": anomalies}

    return {
        "metadata": {"total_rows": len(df)},
        "data": {"columns": columns_profile}
    }

def profile_fields(file_contents: bytes, filename: str):
    """
    Profiles detailed field stats from a file in a standardized format.
    """
    file_extension = filename.split('.')[-1].lower()
    results = {}

    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents), parse_dates=True)
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = {"status": "success", **_profile_field_stats(df)}
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                results[sheet_name] = {"status": "success", **_profile_field_stats(df)}

        return {
            "source_file": filename,
            "agent": "FieldProfiler",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")
