from datetime import datetime
import io
import os
import re
import time
import json
import base64
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from fastapi import HTTPException

from app.config import AGENT_ROUTES


AGENT_VERSION = "1.1.0"  # Version bump for architecture alignment


PII_PATTERNS = {
    "Email": re.compile(r"e-?mail", re.IGNORECASE),
    "PhoneNumber": re.compile(r"phone|contact.*num", re.IGNORECASE),
    "SSN": re.compile(r"ssn|social.*sec", re.IGNORECASE),
    "Address": re.compile(r"address|street", re.IGNORECASE),
    "DateOfBirth": re.compile(r"dob|birth.*date", re.IGNORECASE),
    "Name": re.compile(r"name|full.*name|user.*name", re.IGNORECASE),
}


def _process_sheet(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """
    Process a single sheet/dataframe by scanning headers for PII patterns.
    Returns the result structure for this sheet.
    """
    columns = list(df.columns)
    
    flagged: List[Dict[str, Any]] = []
    for col in columns:
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern.search(str(col)):
                flagged.append({
                    "field": col,
                    "suspected_pii_type": pii_type,
                })
                break  # first match is sufficient
    
    count = len(flagged)
    if count > 0:
        alerts = [
            {
                "level": "critical",
                "message": f"Found {count} field(s) suspected of containing sensitive PII data. Manual review is required.",
            }
        ]
        routing = {
            "status": "Needs Review",
            "reason": "Potential PII fields detected in schema.",
            "suggestion": "Run governance review and apply necessary protections.",
            "suggested_agent_endpoint": AGENT_ROUTES.get("govern_data_tool"),
        }
    else:
        alerts = [
            {
                "level": "info",
                "message": "No PII-suspect fields detected by header scan.",
            }
        ]
        routing = {
            "status": "Ready",
            "reason": "No sensitive columns suspected based on provided patterns.",
            "suggestion": "No further compliance action required.",
            "suggested_agent_endpoint": None,
        }
    
    data_payload = {
        "flagged_columns": flagged,
    }
    
    return {
        "status": "success",
        "metadata": {
            "total_rows": int(len(df)),
            "columns": columns,
        },
        "alerts": alerts,
        "routing": routing,
        "data": data_payload,
    }


def tag_compliance(file_contents: bytes, filename: str) -> Dict[str, Any]:
    """
    Compute: Read headers and flag columns suspected to contain PII based on name patterns across CSV/Excel files.
    Explain: Critical alert if PII suspected; info otherwise.
    Decide: If PII -> Needs Review (govern_data_tool); else Ready.
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
            results[sheet_name] = _process_sheet(df, sheet_name)
            
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
                results[sheet_name] = _process_sheet(df, sheet_name)
            
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
            "agent": "ComplianceTagger",
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
        raise HTTPException(status_code=400, detail=f"ComplianceTagger failed: {str(e)}")
