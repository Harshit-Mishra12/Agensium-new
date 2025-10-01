import pandas as pd
import io
import sqlparse
import warnings
from fastapi import HTTPException

def _profile_dataframe(df: pd.DataFrame):
    """
    Helper function to generate the data and metadata for a single DataFrame.
    """
    if df.empty:
        return {
            "metadata": {"total_rows": 0, "message": "Sheet is empty."},
            "data": {"summary_table": []}
        }
        
    schema_summary = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        total_count = len(df)
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        distinct_count = df[col].nunique()
        top_values = df[col].value_counts().nlargest(3).index.tolist()
        top_values_str = ', '.join(map(str, top_values))

        col_type = str(df[col].dtype)
        data_type = col_type
        if 'object' in col_type:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(df[col].dropna().iloc[:100], errors='raise')
                data_type = 'Date'
            except (ValueError, TypeError, IndexError):
                data_type = 'Text'
        elif 'int' in col_type: data_type = 'Integer'
        elif 'float' in col_type: data_type = 'Float'
        elif 'datetime' in col_type: data_type = 'Date'

        schema_summary.append({
            "field": col, "data_type": data_type, "null": f"{null_percentage:.1f}%",
            "distinct_count": distinct_count, "top_values": top_values_str + ('…' if distinct_count > 3 else '')
        })

    return {
        "metadata": {"total_rows": len(df)},
        "data": {"summary_table": schema_summary}
    }

def scan_schema(file_contents: bytes, filename: str):
    """
    Scans a file and returns a schema profile in a standardized format.
    """
    file_extension = filename.split('.')[-1].lower()
    results = {}

    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = {"status": "success", **_profile_dataframe(df)}
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                results[sheet_name] = {"status": "success", **_profile_dataframe(df)}
        # Add other file types (json, sql) here if needed

        return {
            "source_file": filename,
            "agent": "SchemaScanner",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

