import pandas as pd
import numpy as np
import sqlite3
import io
import os
from scipy.stats import entropy


def _profile_dataframe(df: pd.DataFrame, baseline_schema: dict | None = None):
    """
    Profile a single dataframe (field stats + anomalies) with 2-decimal rounding.
    Returns the unified 'columns' format for frontend.
    """
    columns = {}

    for col in df.columns:
        col_data = df[col].dropna()

        # --- Stats ---
        if col_data.empty:
            stats = {}
        elif pd.api.types.is_numeric_dtype(col_data):
            stats = {
                "min": round(float(col_data.min()), 2),
                "max": round(float(col_data.max()), 2),
                "mean": round(float(col_data.mean()), 2),
                "std_dev": round(float(col_data.std()), 2),
                "unique_values": None,
                "entropy": round(float(entropy(pd.value_counts(col_data, normalize=True), base=2)), 2)
            }
        elif pd.api.types.is_datetime64_any_dtype(col_data) or "date" in col.lower():
            try:
                col_data = pd.to_datetime(col_data, errors="coerce").dropna()
                stats = {
                    "min": str(col_data.min().date()) if not col_data.empty else None,
                    "max": str(col_data.max().date()) if not col_data.empty else None,
                    "temporal_spread_days": int((col_data.max() - col_data.min()).days) if not col_data.empty else None,
                    "unique_values": None,
                    "entropy": None
                }
            except Exception:
                stats = {"note": "Invalid datetime format"}
        else:
            stats = {
                "min": None,
                "max": None,
                "mean": None,
                "std_dev": None,
                "unique_values": int(col_data.nunique()),
                "entropy": round(float(entropy(pd.value_counts(col_data, normalize=True), base=2)), 2)
            }

        # --- Anomalies ---
        anomalies = {
            "missing_values": round(df[col].isnull().mean() * 100, 2) if df[col].isnull().any() else 0,
            "outliers": [],
            "schema_change": None
        }

        if pd.api.types.is_numeric_dtype(col_data):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            high_threshold = q3 + 1.5 * iqr
            low_threshold = q1 - 1.5 * iqr
            outlier_values = df[(df[col] > high_threshold) | (df[col] < low_threshold)][col].tolist()
            anomalies["outliers"] = [round(float(x), 2) for x in outlier_values] if outlier_values else []

        columns[col] = {
            "stats": stats,
            "anomalies": anomalies
        }

    return {"columns": columns}


# --- File type handlers ---
def profile_csv(contents: bytes, filename: str):
    df = pd.read_csv(io.BytesIO(contents))
    file_key = os.path.splitext(filename)[0]
    return {file_key: {"main": _profile_dataframe(df)}}


def profile_excel(contents: bytes, filename: str):
    xls = pd.ExcelFile(io.BytesIO(contents))
    file_key = os.path.splitext(filename)[0]
    result = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(contents), sheet_name=sheet)
        result[sheet] = _profile_dataframe(df)
    return {file_key: result}


def profile_json(contents: bytes, filename: str):
    df = pd.read_json(io.BytesIO(contents))
    file_key = os.path.splitext(filename)[0]
    return {file_key: {"main": _profile_dataframe(df)}}


def profile_sql(contents: bytes, filename: str):
    sql_text = contents.decode("utf-8")
    conn = sqlite3.connect(":memory:")
    file_key = os.path.splitext(filename)[0]

    try:
        cursor = conn.cursor()
        cursor.executescript(sql_text)

        tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        result = {}
        for table in tables_df["name"].tolist():
            df = pd.read_sql(f"SELECT * FROM {table};", conn)
            result[table] = _profile_dataframe(df)

        return {file_key: result}
    finally:
        conn.close()


# --- Dispatcher ---
def profile_file(contents: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".csv"]:
        return profile_csv(contents, filename)
    elif ext in [".xls", ".xlsx"]:
        return profile_excel(contents, filename)
    elif ext in [".json"]:
        return profile_json(contents, filename)
    elif ext in [".sql"]:
        return profile_sql(contents, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
