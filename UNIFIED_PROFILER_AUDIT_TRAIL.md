# Unified Profiler - Enhanced Audit Trail & Excel Export

## Overview

The Unified Profiler now includes a **comprehensive audit trail** that captures all agent activities and provides an **Excel export** for easy sharing and analysis.

## Enhanced Audit Trail Structure

### Complete Audit Object

```json
{
  "audit": {
    "agent_name": "UnifiedProfiler",
    "timestamp": "2025-10-22T09:37:15.123456+00:00",
    "profile_date": "2025-10-22T09:37:15.123456+00:00",
    "agent_version": "1.0.0",
    "compute_time_seconds": 1.23,
    
    "fields_scanned": [
      "customer_id",
      "age",
      "income",
      "city",
      "signup_date"
    ],
    
    "findings": [
      {
        "severity": "warning",
        "field": "age",
        "issue": "High null percentage: 25.5% of values are missing.",
        "category": "data_completeness",
        "data_type": "int64",
        "semantic_type": "Numeric",
        "null_percentage": 25.5
      },
      {
        "severity": "warning",
        "field": "income",
        "issue": "High outlier count: 45 outliers detected (4.5% of non-null values).",
        "category": "data_quality",
        "data_type": "float64",
        "semantic_type": "Numeric",
        "null_percentage": 2.1,
        "outlier_count": 45,
        "outlier_sample": [250000.0, 300000.0, 350000.0, 400000.0, 500000.0]
      }
    ],
    
    "actions": [
      "Profiled 5 columns across 1 sheet(s)",
      "Generated 2 alert(s)",
      "Performed semantic type inference on all fields",
      "Calculated statistical measures for numeric fields",
      "Detected outliers using IQR method",
      "Computed entropy for categorical/text fields"
    ],
    
    "scores": {
      "total_columns_profiled": 5,
      "total_sheets_analyzed": 1,
      "total_alerts_generated": 2,
      "critical_alerts": 0,
      "warning_alerts": 2,
      "info_alerts": 0
    },
    
    "overrides": {
      "null_alert_threshold": 15.0,
      "categorical_threshold": 30
    }
  }
}
```

## Audit Trail Components

### 1. **agent_name**
- **Type:** String
- **Description:** Name of the agent that ran
- **Value:** `"UnifiedProfiler"`

### 2. **timestamp**
- **Type:** ISO 8601 datetime string
- **Description:** Exact timestamp when the agent started execution
- **Example:** `"2025-10-22T09:37:15.123456+00:00"`

### 3. **fields_scanned**
- **Type:** Array of strings
- **Description:** List of all column names that were analyzed
- **Purpose:** Quick reference of which columns were involved in the profiling
- **Example:** `["customer_id", "age", "income", "city"]`

### 4. **findings**
- **Type:** Array of objects
- **Description:** Detailed list of all issues and problems detected by the agent
- **Structure:**
  ```json
  {
    "severity": "warning|critical|info",
    "field": "column_name",
    "issue": "Human-readable description",
    "category": "data_completeness|data_quality|data_variance|statistical_anomaly|data_availability",
    "data_type": "int64|float64|object|...",
    "semantic_type": "Numeric|Temporal|Categorical|Free Text",
    "null_percentage": 25.5,
    "outlier_count": 45,
    "outlier_sample": [250000.0, 300000.0]
  }
  ```

**Finding Categories:**
- `data_completeness`: Missing values, high null percentages
- `data_quality`: Outliers, data anomalies
- `data_variance`: Constant columns, low variance
- `statistical_anomaly`: Unusual statistical patterns
- `data_availability`: Empty datasets

### 5. **actions**
- **Type:** Array of strings
- **Description:** List of all operations performed by the agent
- **Purpose:** Provides transparency into what the agent did (crucial for transformation agents)
- **Examples:**
  - `"Profiled 5 columns across 1 sheet(s)"`
  - `"Generated 2 alert(s)"`
  - `"Performed semantic type inference on all fields"`
  - `"Detected outliers using IQR method"`

### 6. **scores**
- **Type:** Object with numeric values
- **Description:** Key metrics and quantitative measures from the profiling
- **Metrics:**
  - `total_columns_profiled`: Number of columns analyzed
  - `total_sheets_analyzed`: Number of sheets/datasets processed
  - `total_alerts_generated`: Total alerts raised
  - `critical_alerts`: Count of critical-level alerts
  - `warning_alerts`: Count of warning-level alerts
  - `info_alerts`: Count of info-level alerts

### 7. **overrides**
- **Type:** Object
- **Description:** Manual configurations/thresholds provided by the user
- **Purpose:** Captures human input and custom parameters
- **Example:**
  ```json
  {
    "null_alert_threshold": 15.0,
    "categorical_threshold": 30,
    "outlier_iqr_multiplier": 2.0
  }
  ```
- **Empty if no overrides:** `{}`

## Excel Export Feature

### Excel Export Object

```json
{
  "excel_export": {
    "filename": "data_profile_report.xlsx",
    "size_bytes": 45678,
    "format": "xlsx",
    "base64_data": "UEsDBBQAAAAIAH1p...",
    "sheets_included": [
      "Summary",
      "Fields Scanned",
      "Findings",
      "Actions",
      "Configuration",
      "User Overrides",
      "Profile_data"
    ],
    "download_ready": true
  }
}
```

### Excel File Structure

The generated Excel file contains multiple sheets:

#### **Sheet 1: Summary**
| Metric | Value |
|--------|-------|
| Source File | data.csv |
| Agent | UnifiedProfiler |
| Version | 1.0.0 |
| Timestamp | 2025-10-22T09:37:15... |
| Compute Time (seconds) | 1.23 |
| Total Sheets Analyzed | 1 |
| Total Columns Profiled | 5 |
| Total Alerts Generated | 2 |
| Critical Alerts | 0 |
| Warning Alerts | 2 |
| Info Alerts | 0 |

#### **Sheet 2: Fields Scanned**
| Field Name |
|------------|
| customer_id |
| age |
| income |
| city |
| signup_date |

#### **Sheet 3: Findings**
| severity | field | issue | category | data_type | semantic_type | null_percentage | outlier_count |
|----------|-------|-------|----------|-----------|---------------|-----------------|---------------|
| warning | age | High null percentage... | data_completeness | int64 | Numeric | 25.5 | - |
| warning | income | High outlier count... | data_quality | float64 | Numeric | 2.1 | 45 |

#### **Sheet 4: Actions**
| Action |
|--------|
| Profiled 5 columns across 1 sheet(s) |
| Generated 2 alert(s) |
| Performed semantic type inference on all fields |
| ... |

#### **Sheet 5: Configuration**
| Parameter | Value |
|-----------|-------|
| null_alert_threshold | 20.0 |
| categorical_threshold | 50 |
| categorical_ratio_threshold | 0.5 |
| ... |

#### **Sheet 6: User Overrides** (if any)
| Parameter | User Value |
|-----------|------------|
| null_alert_threshold | 15.0 |
| categorical_threshold | 30 |

#### **Sheet 7+: Profile_[SheetName]**
Detailed column profiles with type-specific statistics:

| Field Name | Data Type | Semantic Type | Null Count | Null % | Min | Max | Mean | Median | Std Dev | Outliers |
|------------|-----------|---------------|------------|--------|-----|-----|------|--------|---------|----------|
| age | int64 | Numeric | 255 | 25.5 | 18 | 95 | 42.5 | 41.0 | 15.2 | 12 |
| income | float64 | Numeric | 21 | 2.1 | 20000 | 500000 | 65000 | 60000 | 45000 | 45 |

## Using the Excel Export

### Download from API Response

```python
import requests
import base64

response = requests.post(
    'http://localhost:8000/unified-profiler',
    files={'file': open('data.csv', 'rb')}
)

result = response.json()

# Extract Excel export
excel_export = result['excel_export']

if excel_export['download_ready']:
    # Decode base64 data
    excel_bytes = base64.b64decode(excel_export['base64_data'])
    
    # Save to file
    with open(excel_export['filename'], 'wb') as f:
        f.write(excel_bytes)
    
    print(f"Downloaded: {excel_export['filename']}")
    print(f"Size: {excel_export['size_bytes']} bytes")
    print(f"Sheets: {excel_export['sheets_included']}")
```

### JavaScript/Frontend Example

```javascript
const response = await fetch('/unified-profiler', {
    method: 'POST',
    body: formData
});

const result = await response.json();
const excelExport = result.excel_export;

if (excelExport.download_ready) {
    // Create download link
    const blob = base64ToBlob(excelExport.base64_data, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = excelExport.filename;
    a.click();
    
    URL.revokeObjectURL(url);
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], {type: mimeType});
}
```

## Benefits

✓ **Complete Traceability:** Know exactly what the agent did and when  
✓ **Detailed Findings:** Structured information about all detected issues  
✓ **Human Input Tracking:** Captures manual overrides for reproducibility  
✓ **Quantitative Metrics:** Scores provide quick assessment of data quality  
✓ **Easy Sharing:** Excel export for stakeholders without API access  
✓ **Multi-Sheet Reports:** Organized information in separate tabs  
✓ **Audit Compliance:** Full audit trail for regulatory requirements

## Dependencies

Ensure `openpyxl` is installed for Excel generation:

```bash
pip install openpyxl
```

Add to `requirements.txt`:
```
openpyxl>=3.0.0
```
