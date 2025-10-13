# Master My Data Tool - Implementation Summary

## Overview
The **Master My Data Tool** is a comprehensive data mastering orchestrator that chains together four specialized agents to establish data lineage, resolve entities, detect duplicates, and ensure compliance.

## Architecture

### Agents Involved (in execution order)

1. **Source Tagger** (`source_tagger.py`)
   - **Purpose**: Tags all rows with a `_source_system` column for data lineage tracking
   - **Input**: File contents, filename, source_system parameter
   - **Output**: Tagged dataset with `_source_system` column + base64-encoded updated file
   - **Version**: 1.1.4

2. **Entity Resolver** (`entity_resolver.py`)
   - **Purpose**: Identifies potential duplicate entities based on a key column
   - **Input**: File contents, filename, key_column parameter
   - **Output**: Analysis of duplicate entities + base64-encoded file
   - **Version**: 1.1.0

3. **Deduplication Agent** (`dedup_agent.py`)
   - **Purpose**: Detects exact full-row duplicates
   - **Input**: File contents, filename
   - **Output**: Duplicate row analysis + base64-encoded file
   - **Version**: 1.1.0

4. **Compliance Tagger** (`compliance_tagger.py`)
   - **Purpose**: Scans headers for potential PII fields
   - **Input**: File contents, filename
   - **Output**: PII field flagging + base64-encoded file
   - **Version**: 1.1.0

### Common Agent Architecture

All four agents follow the same standardized architecture:

```python
# Imports
import base64, json, pandas, numpy
from app.config import AGENT_ROUTES

# Helper function for processing individual sheets
def _process_sheet(df: pd.DataFrame, ...) -> Dict[str, Any]:
    """Process a single sheet/dataframe"""
    # Returns: status, metadata, alerts, routing, data

# Main agent function
def agent_function(file_contents: bytes, filename: str, ...) -> Dict[str, Any]:
    """
    Compute: [What it does]
    Explain: [Alert strategy]
    Decide: [Routing logic]
    Returns: Stats + updated file content for orchestration workflows.
    """
    # Handle CSV and Excel (multi-sheet)
    # Process each sheet
    # Return base64-encoded file
```

**Key Features:**
- Multi-format support (CSV, Excel with multiple sheets)
- Base64-encoded file output for safe JSON transport
- Consistent error handling with HTTPException
- JSON-serializable output
- Empty sheet handling
- Helper function pattern (`_process_sheet()`)

## Orchestrator

### Master My Data Orchestrator (`master_my_data_orchestrator.py`)

**Version**: 1.1.0

**Key Functions:**

1. **`run_master_workflow(workflow_def, files, params)`**
   - Executes the workflow sequentially
   - Handles file passing between agents
   - Decodes base64 contents when needed
   - Supports stop conditions
   - Returns comprehensive report

2. **`_prepare_agent_inputs()`**
   - Prepares inputs for each agent
   - Handles base64 decoding automatically
   - Injects parameters (source_system, key_column)

3. **`_assemble_master_report()`**
   - Aggregates results from all agents
   - Collects all alerts
   - Determines final routing

### Workflow Definition (`workflow_definitions.py`)

```python
MASTER_MY_DATA_WORKFLOW = {
    "name": "MasterMyData",
    "steps": [
        {
            "agent_name": "source_tagger",
            "passes_file": True  # Passes updated file to next agent
        },
        {
            "agent_name": "entity_resolver",
            "passes_file": True
        },
        {
            "agent_name": "deduplicator",
            "passes_file": True
        },
        {
            "agent_name": "compliance_tagger",
            "passes_file": False  # Last agent
        }
    ]
}
```

## API Endpoint

### `/run-tool/master-my-data` (POST)

**Parameters:**
- `current_file` (UploadFile, required): The data file to process (CSV or Excel)
- `source_system` (str, required): Source system identifier (e.g., "CRM_Data", "ERP_System")
- `key_column` (str, optional): Column name for entity resolution (e.g., "email", "customer_id")

**Response Structure:**
```json
{
  "source_file": "data.csv",
  "tool": "MasterMyData",
  "audit": {
    "profile_date": "2025-10-13T12:00:00Z",
    "agent_version": "1.1.0",
    "compute_time_seconds": 2.45,
    "workflow_status": "Completed"
  },
  "report": {
    "source_tagging": {
      "sheet_name": {
        "rows_tagged": 1000,
        "source_system_added": "CRM_Data"
      }
    },
    "entity_resolution": {
      "sheet_name": {
        "duplicate_entity_count": 5,
        "key_column": "email",
        "duplicate_sample": [...]
      }
    },
    "deduplication": {
      "sheet_name": {
        "original_row_count": 1000,
        "duplicate_row_count": 10,
        "final_row_count": 990
      }
    },
    "compliance": {
      "sheet_name": {
        "flagged_columns": [
          {"field": "email", "suspected_pii_type": "Email"},
          {"field": "ssn", "suspected_pii_type": "SSN"}
        ]
      }
    },
    "alerts": [...],
    "routing": {
      "status": "Needs Review",
      "reason": "...",
      "suggestion": "...",
      "suggested_agent_endpoint": "/run-tool/governance"
    }
  },
  "updated_file": {
    "contents": "base64_encoded_string",
    "filename": "data.csv"
  }
}
```

## File Flow

```
User Upload (CSV/Excel)
    ↓
Source Tagger (adds _source_system column)
    ↓ [base64-encoded file]
Entity Resolver (analyzes key_column duplicates)
    ↓ [base64-encoded file]
Deduplication Agent (finds exact row duplicates)
    ↓ [base64-encoded file]
Compliance Tagger (flags PII fields)
    ↓ [base64-encoded file]
Final Report + Updated File
```

## Agent Registry

All agents are registered in `agent_registry.py`:

```python
AGENT_REGISTRY = {
    "source_tagger": source_tagger.tag_source,
    "entity_resolver": entity_resolver.resolve_entities,
    "deduplicator": dedup_agent.deduplicate,
    "compliance_tagger": compliance_tagger.tag_compliance,
}
```

## Configuration

Endpoints defined in `config.py`:

```python
AGENT_ROUTES = {
    "source_tagger": "/source-tagger",
    "entity_resolver": "/entity-resolver",
    "deduplicate": "/deduplicate",
    "compliance_tagger": "/compliance-tagger",
    "master_my_data_tool": "/run-tool/master-my-data",
}
```

## Key Improvements Made

### 1. Architectural Alignment
- All agents now follow the same structure as `source_tagger`
- Consistent `_process_sheet()` helper pattern
- Uniform error handling

### 2. Multi-Format Support
- CSV support with single sheet handling
- Excel support with multi-sheet processing
- Proper sheet name extraction

### 3. Base64 File Encoding
- Safe JSON transport of binary file contents
- Automatic encoding/decoding in orchestrator
- Maintains file integrity through workflow

### 4. Enhanced Metadata
- All agents return column lists
- Proper row counts (not hardcoded)
- JSON-serializable sample data

### 5. File Passing
- Seamless file flow between agents
- Each agent can modify and pass the file
- Final updated file available in response

## Usage Example

```python
# Using curl
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@customer_data.csv" \
  -F "source_system=CRM_System" \
  -F "key_column=email"

# Using Python requests
import requests

with open('customer_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/run-tool/master-my-data',
        files={'current_file': f},
        data={
            'source_system': 'CRM_System',
            'key_column': 'email'
        }
    )

result = response.json()
print(f"Workflow Status: {result['audit']['workflow_status']}")
print(f"Alerts: {len(result['report']['alerts'])}")
```

## Benefits

1. **Data Lineage**: Track data origin with source system tagging
2. **Entity Resolution**: Identify and consolidate duplicate entities
3. **Data Quality**: Detect and flag duplicate rows
4. **Compliance**: Automatically identify PII fields for governance
5. **Orchestration**: Seamless multi-agent workflow execution
6. **Scalability**: Easy to add new agents or modify workflow
7. **Consistency**: All agents follow the same architecture pattern

## Future Enhancements

- Add stop conditions for early workflow termination
- Implement data cleaning/mastering capabilities
- Add more PII detection patterns
- Support for additional file formats (JSON, Parquet)
- Parallel agent execution where possible
- Workflow versioning and rollback
