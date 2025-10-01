import pandas as pd
import io
import sqlparse
from fastapi import HTTPException

def _calculate_readiness_score(df: pd.DataFrame):
    """
    Calculates the readiness score for a single DataFrame.
    """
    if df.empty:
        return {
            "overall": 0, "completeness": 0, "consistency": 0, "schema_health": 0,
            "message": "Dataset is empty."
        }

    # 1. Completeness Score (based on nulls)
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    completeness_score = max(0, 100 - (null_cells / total_cells * 100)) if total_cells > 0 else 100

    # 2. Consistency Score (based on duplicate rows)
    duplicate_rows = df.duplicated().sum()
    total_rows = len(df)
    consistency_score = max(0, 100 - (duplicate_rows / total_rows * 100)) if total_rows > 0 else 100

    # 3. Schema Health Score (heuristic-based)
    schema_health_score = 100
    
    # Penalize for object columns that might contain mixed types
    for col in df.select_dtypes(include=['object']).columns:
        # Check if a sample of the column can be converted to numeric, suggesting mixed types
        if pd.to_numeric(df[col].dropna().iloc[:100], errors='coerce').notna().sum() > 0:
              schema_health_score -= 5
    
    # Penalize for columns with very low variance (e.g., all the same value)
    for col in df.columns:
        if df[col].nunique() == 1 and len(df) > 1:
            schema_health_score -= 10

    schema_health_score = max(0, schema_health_score)

    # 4. Overall Score (weighted average)
    overall_score = (completeness_score * 0.4) + (consistency_score * 0.4) + (schema_health_score * 0.2)

    return {
        "overall": round(overall_score),
        "completeness": round(completeness_score),
        "consistency": round(consistency_score),
        "schema_health": round(schema_health_score)
    }

def rate_readiness(file_contents: bytes, filename: str):
    """
    Main function to rate the readiness of a dataset from an uploaded file.
    Handles multiple sheets in Excel files.
    """
    file_extension = filename.split('.')[-1].lower()
    all_sheets_scores = {}

    try:
        if file_extension in ['csv', 'json', 'parquet']:
            if file_extension == 'csv':
                df = pd.read_csv(io.BytesIO(file_contents))
            elif file_extension == 'json':
                df = pd.read_json(io.BytesIO(file_contents))
            elif file_extension == 'parquet':
                df = pd.read_parquet(io.BytesIO(file_contents))
            
            sheet_name = filename.rsplit('.', 1)[0]
            all_sheets_scores[sheet_name] = {
                "readiness_score": _calculate_readiness_score(df),
                "total_rows_analyzed": len(df)
            }

        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            if not xls_sheets:
                raise ValueError("The Excel file has no sheets or is empty.")
            
            for sheet_name, df in xls_sheets.items():
                all_sheets_scores[sheet_name] = {
                    "readiness_score": _calculate_readiness_score(df),
                    "total_rows_analyzed": len(df)
                }
        
        elif file_extension == 'sql':
            # For SQL, we can only rate schema health
            sql_script = file_contents.decode('utf-8')
            schema_health_score = 100
            if 'create table' not in sql_script.lower():
                schema_health_score -= 50
            
            all_sheets_scores['sql_schema'] = {
                "readiness_score": {
                    "overall": round(schema_health_score * 0.2),
                    "completeness": 100,
                    "consistency": 100,
                    "schema_health": schema_health_score
                },
                "total_rows_analyzed": "N/A",
                "message": "Scored based on SQL schema definition; no data rows analyzed."
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

        return {
            "readiness_rater": all_sheets_scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the file '{filename}'. Error: {str(e)}")

