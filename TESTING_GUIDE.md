# Master My Data Tool - Testing Guide

## Prerequisites

1. **Start the FastAPI server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. **Prepare test data**
   Create a sample CSV file: `test_data.csv`
   ```csv
   id,email,name,phone,address
   1,john@example.com,John Doe,555-1234,123 Main St
   2,jane@example.com,Jane Smith,555-5678,456 Oak Ave
   3,john@example.com,John Doe,555-1234,123 Main St
   4,bob@example.com,Bob Johnson,555-9012,789 Pine Rd
   ```

## Test Scenarios

### 1. Test Individual Agents

#### A. Source Tagger
```bash
curl -X POST "http://localhost:8000/source-tagger" \
  -F "file=@test_data.csv" \
  -F "source_system=CRM_System"
```

**Expected Output:**
- Status: 200 OK
- `results` contains sheet data
- `updated_file.contents_b64` contains base64-encoded file
- Alert: "Successfully tagged 4 rows with source 'CRM_System'"

#### B. Entity Resolver
```bash
curl -X POST "http://localhost:8000/entity-resolver" \
  -F "file=@test_data.csv" \
  -F "key_column=email"
```

**Expected Output:**
- Status: 200 OK
- `duplicate_entity_count`: 1 (john@example.com appears twice)
- Alert: Warning about duplicate entities
- Routing: "Needs Review" → clean_data_tool

#### C. Deduplication Agent
```bash
curl -X POST "http://localhost:8000/deduplicate" \
  -F "file=@test_data.csv"
```

**Expected Output:**
- Status: 200 OK
- `duplicate_row_count`: 1 (row 3 is duplicate of row 1)
- `final_row_count`: 3
- Alert: Warning about duplicate rows

#### D. Compliance Tagger
```bash
curl -X POST "http://localhost:8000/compliance-tagger" \
  -F "file=@test_data.csv"
```

**Expected Output:**
- Status: 200 OK
- `flagged_columns`: email, name, phone, address
- Alert: Critical - PII fields detected
- Routing: "Needs Review" → govern_data_tool

### 2. Test Master My Data Tool (Full Workflow)

#### Basic Test
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.csv" \
  -F "source_system=CRM_System" \
  -F "key_column=email"
```

**Expected Output:**
```json
{
  "source_file": "test_data.csv",
  "tool": "MasterMyData",
  "audit": {
    "profile_date": "2025-10-13T...",
    "agent_version": "1.1.0",
    "compute_time_seconds": 2.5,
    "workflow_status": "Completed"
  },
  "report": {
    "source_tagging": {
      "test_data": {
        "rows_tagged": 4,
        "source_system_added": "CRM_System"
      }
    },
    "entity_resolution": {
      "test_data": {
        "duplicate_entity_count": 1,
        "key_column": "email",
        "duplicate_sample": [...]
      }
    },
    "deduplication": {
      "test_data": {
        "original_row_count": 4,
        "duplicate_row_count": 1,
        "final_row_count": 3
      }
    },
    "compliance": {
      "test_data": {
        "flagged_columns": [
          {"field": "email", "suspected_pii_type": "Email"},
          {"field": "name", "suspected_pii_type": "Name"},
          {"field": "phone", "suspected_pii_type": "PhoneNumber"},
          {"field": "address", "suspected_pii_type": "Address"}
        ]
      }
    },
    "alerts": [
      {
        "level": "info",
        "message": "Successfully tagged 4 rows with source 'CRM_System'."
      },
      {
        "level": "warning",
        "message": "Found 1 potential duplicate entities..."
      },
      {
        "level": "warning",
        "message": "Detected 1 exact duplicate rows (25% of 4 records)."
      },
      {
        "level": "critical",
        "message": "Found 4 field(s) suspected of containing sensitive PII data..."
      }
    ],
    "routing": {
      "status": "Needs Review",
      "reason": "Potential PII fields detected in schema.",
      "suggestion": "Run governance review and apply necessary protections.",
      "suggested_agent_endpoint": "/run-tool/governance"
    }
  },
  "updated_file": {
    "contents": "base64_string_here",
    "filename": "test_data.csv"
  }
}
```

### 3. Test with Excel File

Create `test_data.xlsx` with multiple sheets:
- Sheet1: Sales_Q1
- Sheet2: Sales_Q2

```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.xlsx" \
  -F "source_system=ERP_System" \
  -F "key_column=customer_id"
```

**Expected Output:**
- Results for both sheets (Sales_Q1 and Sales_Q2)
- Each sheet processed independently
- Aggregated alerts from all sheets

### 4. Test Error Handling

#### Missing Required Parameter
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.csv"
```

**Expected Output:**
- Status: 400 Bad Request
- Error: "source_system parameter is required."

#### Invalid Key Column
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.csv" \
  -F "source_system=CRM_System" \
  -F "key_column=nonexistent_column"
```

**Expected Output:**
- Status: 400 Bad Request
- Error: "Key column 'nonexistent_column' not found..."

#### Unsupported File Format
```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@test_data.json" \
  -F "source_system=CRM_System"
```

**Expected Output:**
- Status: 400 Bad Request
- Error: "Unsupported file format: json. Supported formats: csv, xls, xlsx"

### 5. Test Empty File Handling

Create `empty.csv` with only headers:
```csv
id,email,name
```

```bash
curl -X POST "http://localhost:8000/run-tool/master-my-data" \
  -F "current_file=@empty.csv" \
  -F "source_system=CRM_System" \
  -F "key_column=email"
```

**Expected Output:**
- Status: 200 OK
- All agents handle empty sheets gracefully
- Alerts indicate empty sheets

## Python Testing Script

```python
import requests
import json

def test_master_my_data():
    """Test the Master My Data tool"""
    
    url = "http://localhost:8000/run-tool/master-my-data"
    
    with open('test_data.csv', 'rb') as f:
        files = {'current_file': f}
        data = {
            'source_system': 'CRM_System',
            'key_column': 'email'
        }
        
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nWorkflow Status: {result['audit']['workflow_status']}")
        print(f"Compute Time: {result['audit']['compute_time_seconds']}s")
        print(f"\nNumber of Alerts: {len(result['report']['alerts'])}")
        
        for alert in result['report']['alerts']:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
        
        print(f"\nFinal Routing: {result['report']['routing']['status']}")
        print(f"Suggestion: {result['report']['routing']['suggestion']}")
        
        # Check if updated file is present
        if 'updated_file' in result:
            print(f"\nUpdated file available: {result['updated_file']['filename']}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_master_my_data()
```

## Validation Checklist

### Agent-Level Validation
- [ ] Source Tagger adds `_source_system` column
- [ ] Entity Resolver detects duplicate entities correctly
- [ ] Dedup Agent finds exact row duplicates
- [ ] Compliance Tagger flags PII fields
- [ ] All agents return base64-encoded files
- [ ] All agents handle empty sheets
- [ ] All agents support CSV and Excel

### Orchestrator-Level Validation
- [ ] File passes correctly between agents
- [ ] Base64 encoding/decoding works
- [ ] Parameters injected correctly
- [ ] All agent results aggregated
- [ ] Alerts collected from all agents
- [ ] Final routing determined correctly
- [ ] Updated file included in response

### Integration Validation
- [ ] API endpoint responds correctly
- [ ] Form data parsed properly
- [ ] File upload works
- [ ] Error messages are clear
- [ ] Response structure is consistent
- [ ] Multi-sheet Excel files work

## Performance Testing

```python
import time
import requests

def performance_test(file_path, iterations=10):
    """Test performance of Master My Data tool"""
    
    url = "http://localhost:8000/run-tool/master-my-data"
    times = []
    
    for i in range(iterations):
        with open(file_path, 'rb') as f:
            files = {'current_file': f}
            data = {
                'source_system': 'CRM_System',
                'key_column': 'email'
            }
            
            start = time.time()
            response = requests.post(url, files=files, data=data)
            elapsed = time.time() - start
            
            times.append(elapsed)
            print(f"Iteration {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage Time: {avg_time:.2f}s")
    print(f"Min Time: {min(times):.2f}s")
    print(f"Max Time: {max(times):.2f}s")

if __name__ == "__main__":
    performance_test('test_data.csv')
```

## Troubleshooting

### Issue: "Agent not found in registry"
**Solution:** Check `agent_registry.py` - ensure all agents are imported and registered

### Issue: "Base64 decode error"
**Solution:** Verify that agents are returning `contents_b64` in the `updated_file` structure

### Issue: "Key column not found"
**Solution:** Ensure the key_column parameter matches an actual column name in the file

### Issue: "Workflow status: Halted"
**Solution:** Check stop conditions in workflow definition - may be intentional

### Issue: Slow performance
**Solution:** 
- Check file size
- Verify pandas/openpyxl versions
- Consider adding caching for large files

## Next Steps

After successful testing:

1. **Add more test cases** for edge cases
2. **Implement unit tests** for each agent
3. **Add integration tests** for the orchestrator
4. **Set up CI/CD pipeline** with automated testing
5. **Monitor performance** in production
6. **Collect user feedback** and iterate
