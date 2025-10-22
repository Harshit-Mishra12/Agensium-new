# Unified Profiler API Documentation

## Endpoint
`POST /unified-profiler`

## Description
Comprehensive data profiling agent that combines schema scanning and field statistics. Supports optional threshold parameters that override default configuration.

## Parameters

### Required
- **file** (UploadFile): CSV or Excel file to profile

### Optional Thresholds (Form Data)
All threshold parameters are optional. The endpoint uses smart configuration loading:
- **All parameters provided**: Uses user values directly (no config.json loading)
- **Some/no parameters provided**: Loads defaults from `config.json` and overrides with user values where present

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `null_alert_threshold` | float | 20.0 | Percentage threshold for null value warnings (e.g., 20.0 = warn if >20% nulls) |
| `categorical_threshold` | int | 50 | Maximum unique values to classify as categorical |
| `categorical_ratio_threshold` | float | 0.5 | Ratio of unique/total rows to classify as categorical (0.5 = 50%) |
| `top_n_values` | int | 10 | Number of top values to return for categorical/text fields |
| `outlier_iqr_multiplier` | float | 1.5 | IQR multiplier for outlier detection (1.5 = standard, 3.0 = extreme) |
| `outlier_alert_threshold` | float | 0.05 | Percentage threshold for outlier warnings (0.05 = warn if >5% outliers) |

## Configuration Loading Behavior

The endpoint intelligently handles threshold configuration in three scenarios:

| Scenario | User Provides | Behavior | config.json Loaded? |
|----------|---------------|----------|---------------------|
| **1. All defaults** | No parameters | Uses all values from config.json | ✓ Yes |
| **2. Partial override** | Some parameters | Loads config.json, overrides specific values | ✓ Yes |
| **3. Full override** | All 6 parameters | Uses user values directly | ✗ No (optimized) |

## Usage Examples

### 1. Using Default Thresholds (from config.json)

**cURL:**
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv"
```

**Python:**
```python
import requests

with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/unified-profiler',
        files={'file': f}
    )
print(response.json())
```

### 2. Partial Override (Some Custom Thresholds)

**cURL:**
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=15.0" \
  -F "categorical_threshold=30"
```
*Loads config.json for missing parameters, overrides only the two specified*

**Python:**
```python
import requests

with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/unified-profiler',
        files={'file': f},
        data={
            'null_alert_threshold': 15.0,
            'categorical_threshold': 30
        }
    )
print(response.json())
```

### 3. Full Override (All Custom Thresholds)

**cURL:**
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=15.0" \
  -F "categorical_threshold=30" \
  -F "categorical_ratio_threshold=0.4" \
  -F "top_n_values=5" \
  -F "outlier_iqr_multiplier=2.0" \
  -F "outlier_alert_threshold=0.03"
```
*Uses all user values directly, config.json is NOT loaded (optimized)*

**Python:**
```python
import requests

with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/unified-profiler',
        files={'file': f},
        data={
            'null_alert_threshold': 15.0,
            'categorical_threshold': 30,
            'categorical_ratio_threshold': 0.4,
            'top_n_values': 5,
            'outlier_iqr_multiplier': 2.0,
            'outlier_alert_threshold': 0.03
        }
    )
print(response.json())
```

**JavaScript (FormData):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('null_alert_threshold', '15.0');
formData.append('categorical_threshold', '30');
formData.append('outlier_iqr_multiplier', '2.0');

fetch('http://localhost:8000/unified-profiler', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Structure

```json
{
  "source_file": "data.csv",
  "agent": "UnifiedProfiler",
  "audit": {
    "profile_date": "2025-10-22T08:15:30.123456+00:00",
    "agent_version": "1.0.0",
    "compute_time_seconds": 1.23
  },
  "results": {
    "data": {
      "status": "success",
      "metadata": {
        "total_rows": 1000
      },
      "alerts": [
        {
          "level": "warning",
          "field": "age",
          "message": "High null percentage: 25.5% of values are missing."
        }
      ],
      "routing": {
        "status": "Needs Review",
        "reason": "Data quality issues detected (high nulls, outliers, or empty dataset).",
        "suggestion": "Review the alerts and consider using the data cleaning tool to address issues.",
        "suggested_agent_endpoint": "/run-tool/cleaner"
      },
      "data": {
        "columns": {
          "age": {
            "field_name": "age",
            "data_type": "int64",
            "null_count": 255,
            "null_percentage": 25.5,
            "semantic_type": "Numeric",
            "min": 18.0,
            "max": 95.0,
            "mean": 42.5,
            "median": 41.0,
            "std_dev": 15.2,
            "variance": 231.04,
            "p25": 30.0,
            "p75": 55.0,
            "outlier_count": 12,
            "outlier_sample": [120.0, 150.0]
          }
        }
      }
    }
  }
}
```

## Threshold Tuning Guide

### Making Alerts More Strict
- **Lower** `null_alert_threshold` (e.g., 10.0) - warns at lower null percentages
- **Lower** `outlier_alert_threshold` (e.g., 0.02) - warns with fewer outliers
- **Lower** `outlier_iqr_multiplier` (e.g., 1.0) - detects more outliers

### Making Alerts More Lenient
- **Raise** `null_alert_threshold` (e.g., 30.0) - only warns at higher null percentages
- **Raise** `outlier_alert_threshold` (e.g., 0.10) - requires more outliers to warn
- **Raise** `outlier_iqr_multiplier` (e.g., 3.0) - detects fewer outliers

### Adjusting Categorical Detection
- **Lower** `categorical_threshold` (e.g., 20) - more fields classified as categorical
- **Raise** `categorical_threshold` (e.g., 100) - fewer fields classified as categorical
- Adjust `categorical_ratio_threshold` to fine-tune based on data cardinality

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Unsupported file format. Only CSV and Excel files are supported."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Configuration error: [Errno 2] No such file or directory: 'config.json'"
}
```
