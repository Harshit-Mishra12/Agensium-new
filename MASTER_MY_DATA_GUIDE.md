# Master My Data Tool - Implementation Guide

## Overview

The **Master My Data** tool is a comprehensive data mastering workflow that establishes data lineage, resolves entities, removes duplicates, and checks compliance. The Source Tagger agent is the foundation of this workflow.

## Source Tagger Agent

### Purpose
Adds a `_source_system` column to every row in your dataset to establish data lineage. This is critical for:
- **Auditing**: Know where each record came from
- **Trust Building**: Establish data provenance
- **Multi-source Integration**: Track origin when combining datasets

### Features
✅ **Multi-format Support**: CSV, XLS, XLSX  
✅ **Multi-sheet Support**: Processes all sheets in Excel files  
✅ **Updated File Output**: Returns tagged file for downstream processing  
✅ **Orchestration Ready**: Designed for workflow integration  

### File Location
- **Agent**: `app/agents/source/source_tagger.py`
- **Version**: 1.1.0

---

## API Usage

### 1. Standalone Agent Endpoint

**Endpoint**: `POST /source-tagger`

**Parameters**:
- `file`: CSV/Excel file (UploadFile)
- `source_system`: Origin system name (Form string, e.g., "CRM_Data", "ERP_System")

**Example Request** (curl):
```bash
curl -X POST "http://localhost:8000/source-tagger" \
  -F "file=@customers.csv" \
  -F "source_system=CRM_Data"
```

**Response Structure**:
```json
{
  "source_file": "customers.csv",
  "agent": "SourceTagger",
  "audit": {
    "profile_date": "2025-10-11T10:35:46Z",
    "agent_version": "1.1.0",
    "compute_time_seconds": 0.123456
  },
  "results": {
    "customers": {
      "status": "success",
      "metadata": {
        "total_rows": 1000,
        "columns": ["id", "name", "email", "_source_system"]
      },
      "alerts": [
        {
          "level": "info",
          "message": "Successfully tagged 1000 rows with source 'CRM_Data'."
        }
      ],
      "routing": {
        "status": "Ready",
        "reason": "Dataset has been tagged with source system for lineage.",
        "suggestion": "Run EntityResolver to identify potential entity duplicates.",
        "suggested_agent_endpoint": "/entity-resolver"
      },
      "data": {
        "rows_tagged": 1000,
        "source_system_added": "CRM_Data",
        "sample_data": [
          {"id": 1, "name": "John", "email": "john@example.com", "_source_system": "CRM_Data"},
          {"id": 2, "name": "Jane", "email": "jane@example.com", "_source_system": "CRM_Data"}
        ]
      }
    }
  },
  "updated_file": {
    "contents": "<binary file contents>",
    "filename": "customers.csv"
  }
}
```

---

### 2. Master My Data Tool (Orchestrated Workflow)

**Endpoint**: `POST /run-tool/master-my-data`

**Parameters**:
- `current_file`: CSV/Excel file (UploadFile)
- `source_system`: Origin system name (Form string, required)
- `key_column`: Column for entity resolution (Form string, optional)

**Example Request** (curl):
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@customers.csv" \
  -F "source_system=CRM_Data" \
  -F "key_column=email"
```

**Workflow Steps**:
1. **Source Tagger**: Tags all rows with `_source_system` column
2. **Entity Resolver**: Identifies duplicate entities based on key column
3. **Deduplicator**: Finds exact row duplicates
4. **Compliance Tagger**: Scans for PII fields

**Response Structure**:
```json
{
  "source_file": "customers.csv",
  "tool": "MasterMyData",
  "audit": {
    "profile_date": "2025-10-11T10:35:46Z",
    "agent_version": "1.0.0",
    "compute_time_seconds": 1.234567,
    "workflow_status": "Completed"
  },
  "report": {
    "source_tagging": {
      "customers": {
        "rows_tagged": 1000,
        "source_system_added": "CRM_Data",
        "sample_data": [...]
      }
    },
    "entity_resolution": {
      "customers": {
        "duplicate_entity_count": 5,
        "key_column": "email",
        "duplicate_sample": [...]
      }
    },
    "deduplication": {
      "customers": {
        "original_row_count": 1000,
        "duplicate_row_count": 10,
        "final_row_count": 990
      }
    },
    "compliance": {
      "customers": {
        "pii_fields": ["email", "phone"],
        "compliance_score": 85
      }
    },
    "alerts": [...],
    "routing": {...}
  },
  "updated_file": {
    "contents": "<binary file contents>",
    "filename": "customers.csv"
  }
}
```

---

## Orchestration Architecture

### Workflow Definition
Located in: `app/tools/workflow_definitions.py`

```python
MASTER_MY_DATA_WORKFLOW = {
    "name": "MasterMyData",
    "steps": [
        {
            "agent_name": "source_tagger",
            "inputs": {"file": "current_file", "source_system": "source_system"},
            "passes_file": True  # Updated file passed to next agent
        },
        {
            "agent_name": "entity_resolver",
            "inputs": {"file": "updated_file", "key_column": "key_column"},
            "optional": False
        },
        {
            "agent_name": "deduplicator",
            "inputs": {"file": "current_file"}
        },
        {
            "agent_name": "compliance_tagger",
            "inputs": {"file": "current_file"}
        }
    ]
}
```

### Orchestrator Engine
Located in: `app/tools/master_my_data_orchestrator.py`

**Key Features**:
- **File Passing**: Agents can pass updated files to subsequent agents
- **Parameter Injection**: Dynamic parameter passing (source_system, key_column)
- **Stop Conditions**: Workflow can halt based on agent results
- **Result Aggregation**: Combines all agent outputs into comprehensive report

---

## Implementation Details

### Multi-Format Support

The Source Tagger handles different file formats:

```python
# CSV Files
if file_extension == 'csv':
    df = pd.read_csv(io.BytesIO(file_contents))
    df["_source_system"] = source_system
    output_buffer = io.BytesIO()
    df.to_csv(output_buffer, index=False)
    updated_file_contents = output_buffer.getvalue()

# Excel Files (XLS/XLSX)
elif file_extension in ['xlsx', 'xls']:
    excel_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
    for sheet_name, df in excel_sheets.items():
        df["_source_system"] = source_system
    
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        for sheet_name, df in excel_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    updated_file_contents = output_buffer.getvalue()
```

### File Passing in Orchestration

The orchestrator tracks the current file and passes updated versions:

```python
# Execute agent
result = agent_function(*inputs)
results_context[agent_name] = result

# If this agent passes an updated file, use it for the next agent
if step.get("passes_file", False) and "updated_file" in result:
    current_file = result["updated_file"]
```

---

## Agent Registry

The Source Tagger is registered in `app/agents/agent_registry.py`:

```python
AGENT_REGISTRY = {
    # ... other agents ...
    "source_tagger": source_tagger.tag_source,
    "entity_resolver": entity_resolver.resolve_entities,
    "compliance_tagger": compliance_tagger.tag_compliance,
    # ...
}
```

---

## Testing

### Test Standalone Agent
```bash
# Test with CSV
curl -X POST "http://localhost:8000/source-tagger" \
  -F "file=@test_data.csv" \
  -F "source_system=Test_System"

# Test with Excel
curl -X POST "http://localhost:8000/source-tagger" \
  -F "file=@test_data.xlsx" \
  -F "source_system=Test_System"
```

### Test Full Workflow
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.csv" \
  -F "source_system=Test_System" \
  -F "key_column=email"
```

---

## Dependencies

Required packages (already in `requirements.txt`):
- `pandas`: Data manipulation
- `openpyxl`: Excel file support
- `fastapi`: API framework
- `python-multipart`: File upload support

---

## Next Steps

1. **Extend Workflow**: Add more agents to the Master My Data workflow
2. **Custom Stop Conditions**: Implement conditional workflow branching
3. **File Download**: Add endpoint to download the updated file
4. **Batch Processing**: Support multiple files in one workflow run
5. **Validation Rules**: Add data quality checks after tagging

---

## Summary

The Source Tagger agent is now fully integrated into the Master My Data tool with:
- ✅ Multi-format support (CSV, XLS, XLSX)
- ✅ Updated file output for orchestration
- ✅ Complete workflow integration
- ✅ Proper error handling
- ✅ Comprehensive documentation

The agent can be used standalone or as part of the orchestrated workflow, making it flexible for different use cases.
