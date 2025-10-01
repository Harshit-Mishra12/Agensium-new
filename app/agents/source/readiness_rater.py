import pandas as pd
import io
from fastapi import HTTPException

def _calculate_readiness_score(df: pd.DataFrame):
    """
    Calculates the readiness score for a single DataFrame and formats the output.
    """
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows_analyzed": 0, "message": "Dataset is empty."},
            "data": {"readiness_score": {"overall": 0, "completeness": 0, "consistency": 0, "schema_health": 0}}
        }

    # 1. Completeness Score
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    completeness_score = max(0, 100 - (null_cells / total_cells * 100)) if total_cells > 0 else 100

    # 2. Consistency Score
    duplicate_rows = df.duplicated().sum()
    total_rows = len(df)
    consistency_score = max(0, 100 - (duplicate_rows / total_rows * 100)) if total_rows > 0 else 100

    # 3. Schema Health Score
    schema_health_score = 100
    for col in df.columns:
        if df[col].nunique() == 1 and len(df) > 1:
            schema_health_score -= 10 # Penalize for low variance
    schema_health_score = max(0, schema_health_score)

    # 4. Overall Score
    overall_score = (completeness_score * 0.4) + (consistency_score * 0.4) + (schema_health_score * 0.2)

    return {
        "status": "success",
        "metadata": {"total_rows_analyzed": len(df)},
        "data": {
            "readiness_score": {
                "overall": round(overall_score),
                "completeness": round(completeness_score),
                "consistency": round(consistency_score),
                "schema_health": round(schema_health_score)
            }
        }
    }

def rate_readiness(file_contents: bytes, filename: str):
    """
    Main function to rate the readiness of a dataset from an uploaded file.
    Returns data in the standardized API format.
    """
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            results[sheet_name] = _calculate_readiness_score(df)
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            if not xls_sheets:
                 # Handle empty Excel file by returning a single empty result
                results["sheet_1"] = _calculate_readiness_score(pd.DataFrame())
            else:
                for sheet_name, df in xls_sheets.items():
                    results[sheet_name] = _calculate_readiness_score(df)
        
        # Add other file formats like JSON or Parquet here if needed

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

        return {
            "source_file": filename,
            "agent": "ReadinessRater",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the file '{filename}'. Error: {str(e)}")

