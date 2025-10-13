from datetime import datetime
import io
import os
import time
import json
import base64
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from fastapi import HTTPException

from app.config import AGENT_ROUTES


AGENT_VERSION = "1.1.0"  # Version bump for architecture alignment


def _process_sheet(df: pd.DataFrame, key_column: str, sheet_name: str) -> Dict[str, Any]:
    """
    Process a single sheet/dataframe by resolving entities based on key_column.
    Returns the result structure for this sheet.
    """
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows": 0},
            "alerts": [{"level": "info", "message": f"Sheet '{sheet_name}' is empty."}],
            "routing": {
                "status": "Ready",
                "reason": "Sheet is empty.",
                "suggestion": "Run DedupAgent on non-empty sheets.",
                "suggested_agent_endpoint": AGENT_ROUTES.get("deduplicate"),
            },
            "data": {
                "duplicate_entity_count": 0,
                "key_column": key_column,
                "duplicate_sample": [],
            }
        }
    
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in sheet '{sheet_name}'.")
    
    if len(df.columns) == 0:
        raise ValueError(f"Sheet '{sheet_name}' has no columns.")
    
    primary_key_col = df.columns[0]
    
    # Group by key_column and find groups with >1 unique primary key
    grouped = df.groupby(key_column, dropna=False)[primary_key_col].nunique()
    dup_keys = grouped[grouped > 1]
    duplicate_entity_count = int(dup_keys.shape[0])
    
    # Build a small sample of duplicate groups
    duplicate_sample: List[Dict[str, Any]] = []
    if duplicate_entity_count > 0:
        for k in dup_keys.head(5).index:
            subset = df[df[key_column] == k]
            duplicate_sample.append(
                {
                    key_column: k if pd.notna(k) else None,
                    "primary_keys": sorted(pd.unique(subset[primary_key_col].dropna().astype(str)))[:10],
                    "rows": int(len(subset)),
                }
            )
    
    if duplicate_entity_count > 0:
        alerts = [
            {
                "level": "warning",
                "message": f"Found {duplicate_entity_count} potential duplicate entities based on the key '{key_column}'. Manual review is recommended.",
            }
        ]
        routing = {
            "status": "Needs Review",
            "reason": "Multiple primary keys mapped to the same entity key.",
            "suggestion": "Run a cleaning or mastering tool to consolidate duplicates.",
            "suggested_agent_endpoint": AGENT_ROUTES.get("clean_data_tool"),
        }
    else:
        alerts = [
            {
                "level": "info",
                "message": "No duplicate entities found.",
            }
        ]
        routing = {
            "status": "Ready",
            "reason": "No entity-level duplicates detected for the chosen key.",
            "suggestion": "Proceed to DedupAgent to check for full-row duplicates.",
            "suggested_agent_endpoint": AGENT_ROUTES.get("deduplicate"),
        }
    
    data_payload = {
        "duplicate_entity_count": duplicate_entity_count,
        "key_column": key_column,
        "duplicate_sample": duplicate_sample,
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


def resolve_entities(file_contents: bytes, filename: str, key_column: str) -> Dict[str, Any]:
    """
    Compute: Group by key_column and find groups with >1 unique primary key (first column) across CSV/Excel files.
    Explain: Warn if duplicates found, else info.
    Decide: Needs Review -> clean_data_tool; otherwise Ready -> DedupAgent.
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
            results[sheet_name] = _process_sheet(df, key_column, sheet_name)
            
            # Generate updated CSV file (no modifications, just pass through)
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
                results[sheet_name] = _process_sheet(df, key_column, sheet_name)
            
            # Generate updated Excel file (no modifications, just pass through)
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

        # Encode the binary contents into a Base64 string for safe JSON transport
        encoded_contents = base64.b64encode(updated_file_contents).decode('ascii')
        
        return {
            "source_file": os.path.basename(filename),
            "agent": "EntityResolver",
            "audit": {
                "profile_date": datetime.utcnow().isoformat() + "Z",
                "agent_version": AGENT_VERSION,
                "compute_time_seconds": round(compute_time, 6),
            },
            "results": results,
            # Return the Base64 encoded string in the response
            "updated_file": {
                "contents_b64": encoded_contents,
                "filename": filename,
                "encoding": "base64"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"EntityResolver failed: {str(e)}")
