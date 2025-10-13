from datetime import datetime
import io
import os
import time
import json
import base64  # 1. Import the base64 library
from typing import Dict, Any
import pandas as pd
import numpy as np
from fastapi import HTTPException

from app.config import AGENT_ROUTES


AGENT_VERSION = "1.1.4" # Version bump for the new fix


def _process_sheet(df: pd.DataFrame, source_system: str, sheet_name: str) -> Dict[str, Any]:
    """
    Process a single sheet/dataframe by adding the _source_system column.
    Returns the result structure for this sheet.
    """
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows": 0},
            "alerts": [{"level": "info", "message": f"Sheet '{sheet_name}' is empty."}],
            "routing": {
                "status": "Ready",
                "reason": "Sheet is empty but tagged.",
                "suggestion": "Run EntityResolver on non-empty sheets.",
                "suggested_agent_endpoint": AGENT_ROUTES.get("entity_resolver"),
            },
            "data": {
                "rows_tagged": 0,
                "source_system_added": source_system,
                "sample_data": [],
            }
        }
    
    # Add the _source_system column
    df["_source_system"] = source_system
    
    alerts = [
        {
            "level": "info",
            "message": f"Successfully tagged {len(df)} rows with source '{source_system}'.",
        }
    ]
    
    routing = {
        "status": "Ready",
        "reason": "Dataset has been tagged with source system for lineage.",
        "suggestion": "Run EntityResolver to identify potential entity duplicates.",
        "suggested_agent_endpoint": AGENT_ROUTES.get("entity_resolver"),
    }
    
    # Prepare a JSON-serializable sample of the data.
    sample_df = df.head(5)
    serializable_sample_json = sample_df.to_json(orient="records", default_handler=str)
    serializable_sample = json.loads(serializable_sample_json)
    
    data_payload = {
        "rows_tagged": int(len(df)),
        "source_system_added": source_system,
        "sample_data": serializable_sample,
    }
    
    return {
        "status": "success",
        "metadata": {
            "total_rows": int(len(df)),
            "columns": list(df.columns),
        },
        "alerts": alerts,
        "routing": routing,
        "data": data_payload,
    }


def tag_source(file_contents: bytes, filename: str, source_system: str) -> Dict[str, Any]:
    """
    Compute: Tag all rows with a `_source_system` column across CSV/Excel files.
    Explain: Info alert summarizing tagging.
    Decide: Always Ready; suggest Entity Resolver next.
    Returns: Stats + updated file content for orchestration workflows.
    """
    start = time.perf_counter()
    file_extension = filename.split('.')[-1].lower()
    results = {}
    updated_file_contents = None
    
    try:
        # Handle different file formats
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = os.path.splitext(os.path.basename(filename))[0]
            results[sheet_name] = _process_sheet(df, source_system, sheet_name)
            
            # Generate updated CSV file
            output_buffer = io.BytesIO()
            df.to_csv(output_buffer, index=False)
            updated_file_contents = output_buffer.getvalue()
            
        elif file_extension in ['xlsx', 'xls']:
            # Read all sheets
            excel_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)

            if not excel_sheets:
                raise ValueError("The provided Excel file contains no visible sheets.")
            
            # Process each sheet
            for sheet_name, df in excel_sheets.items():
                results[sheet_name] = _process_sheet(df, source_system, sheet_name)
            
            # Generate updated Excel file
            output_buffer = io.BytesIO()
            with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                for sheet_name, df_sheet in excel_sheets.items():
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            updated_file_contents = output_buffer.getvalue()
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported formats: csv, xls, xlsx"
            )
        
        compute_time = time.perf_counter() - start

        # --- FIX IMPLEMENTED HERE ---
        # 2. Encode the binary contents into a Base64 string for safe JSON transport.
        encoded_contents = base64.b64encode(updated_file_contents).decode('ascii')
        
        return {
            "source_file": os.path.basename(filename),
            "agent": "SourceTagger",
            "audit": {
                "profile_date": datetime.utcnow().isoformat() + "Z",
                "agent_version": AGENT_VERSION,
                "compute_time_seconds": round(compute_time, 6),
            },
            "results": results,
            # 3. Return the Base64 encoded string in the response.
            "updated_file": {
                "contents_b64": encoded_contents,
                "filename": filename,
                "encoding": "base64"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SourceTagger failed: {str(e)}")

