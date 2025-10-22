# Unified Profiler - Complete Implementation Guide

## 🎯 Overview

The **Unified Profiler** is a comprehensive data profiling agent that combines schema scanning and field statistics into a single powerful tool. It now includes:

✅ **Comprehensive Audit Trail** - Tracks agent execution, findings, actions, and scores  
✅ **Excel Export** - Generates downloadable reports with base64-encoded data  
✅ **User Override Tracking** - Captures manual threshold configurations  
✅ **Smart Configuration Loading** - Optimized parameter handling  
✅ **Semantic Type Inference** - Automatic classification of data types  
✅ **Statistical Analysis** - Outlier detection, entropy calculation, and more

---

## 📋 Response Structure

### Complete JSON Response

```json
{
  "source_file": "data.csv",
  "agent": "UnifiedProfiler",
  
  "audit": {
    "agent_name": "UnifiedProfiler",
    "timestamp": "2025-10-22T09:37:15.123456+00:00",
    "profile_date": "2025-10-22T09:37:15.123456+00:00",
    "agent_version": "1.0.0",
    "compute_time_seconds": 1.23,
    
    "fields_scanned": ["customer_id", "age", "income", "city"],
    
    "findings": [
      {
        "severity": "warning",
        "field": "age",
        "issue": "High null percentage: 25.5% of values are missing.",
        "category": "data_completeness",
        "data_type": "int64",
        "semantic_type": "Numeric",
        "null_percentage": 25.5
      }
    ],
    
    "actions": [
      "Profiled 5 columns across 1 sheet(s)",
      "Generated 2 alert(s)",
      "Performed semantic type inference on all fields"
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
      "null_alert_threshold": 15.0
    }
  },
  
  "results": {
    "data": {
      "status": "success",
      "metadata": {"total_rows": 1000},
      "alerts": [...],
      "routing": {...},
      "data": {"columns": {...}}
    }
  },
  
  "excel_export": {
    "filename": "data_profile_report.xlsx",
    "size_bytes": 45678,
    "format": "xlsx",
    "base64_data": "UEsDBBQAAAAIAH1p...",
    "sheets_included": ["Summary", "Findings", "Profile_data"],
    "download_ready": true
  }
}
```

---

## 🔑 Audit Trail Components

### 1. **agent_name** ✓
- **Purpose:** Identifies which agent ran
- **Value:** `"UnifiedProfiler"`
- **Use Case:** Multi-agent systems tracking

### 2. **timestamp** ✓
- **Purpose:** Records when the agent executed
- **Format:** ISO 8601 with timezone
- **Use Case:** Audit logs, time-series analysis

### 3. **fields_scanned** ✓
- **Purpose:** Lists all columns analyzed
- **Type:** Array of strings
- **Use Case:** Quick reference, field tracking

### 4. **findings** ✓
- **Purpose:** Detailed problems/issues detected
- **Structure:** Array of finding objects
- **Categories:**
  - `data_completeness` - Missing values
  - `data_quality` - Outliers, anomalies
  - `data_variance` - Constant columns
  - `statistical_anomaly` - Unusual patterns
  - `data_availability` - Empty datasets

**Example Finding:**
```json
{
  "severity": "warning",
  "field": "income",
  "issue": "High outlier count: 45 outliers detected",
  "category": "data_quality",
  "data_type": "float64",
  "semantic_type": "Numeric",
  "null_percentage": 2.1,
  "outlier_count": 45,
  "outlier_sample": [250000.0, 300000.0]
}
```

### 5. **actions** ✓
- **Purpose:** What the agent did (crucial for transformation agents)
- **Type:** Array of action descriptions
- **Examples:**
  - `"Profiled 5 columns across 1 sheet(s)"`
  - `"Generated 2 alert(s)"`
  - `"Detected outliers using IQR method"`

### 6. **scores** ✓
- **Purpose:** Key metrics calculated by the agent
- **Metrics:**
  - `total_columns_profiled` - Number of columns analyzed
  - `total_sheets_analyzed` - Number of sheets processed
  - `total_alerts_generated` - Total alerts raised
  - `critical_alerts` - Critical-level alert count
  - `warning_alerts` - Warning-level alert count
  - `info_alerts` - Info-level alert count

### 7. **overrides** ✓
- **Purpose:** Captures manual configurations (human input)
- **Type:** Object with user-provided parameters
- **Empty if defaults used:** `{}`
- **Example:**
```json
{
  "null_alert_threshold": 15.0,
  "categorical_threshold": 30,
  "outlier_iqr_multiplier": 2.0
}
```

---

## 📊 Excel Export Feature

### Excel Blob Structure

```json
{
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
```

### Excel Sheets Generated

| Sheet Name | Content |
|------------|---------|
| **Summary** | Key metrics, agent info, compute time |
| **Fields Scanned** | List of all columns analyzed |
| **Findings** | Detailed issues with severity, category, field |
| **Actions** | All operations performed by the agent |
| **Configuration** | Threshold parameters used |
| **User Overrides** | Manual configurations (if any) |
| **Profile_[SheetName]** | Detailed column profiles with statistics |

### Download Excel File

**Python:**
```python
import base64

excel_export = result['excel_export']
excel_bytes = base64.b64decode(excel_export['base64_data'])

with open(excel_export['filename'], 'wb') as f:
    f.write(excel_bytes)
```

**JavaScript:**
```javascript
const blob = base64ToBlob(excelExport.base64_data, 
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
const url = URL.createObjectURL(blob);

const a = document.createElement('a');
a.href = url;
a.download = excelExport.filename;
a.click();
```

---

## 🚀 API Usage

### Endpoint
```
POST /unified-profiler
```

### Parameters

**Required:**
- `file` (UploadFile) - CSV or Excel file

**Optional Thresholds:**
- `null_alert_threshold` (float) - Default: 20.0
- `categorical_threshold` (int) - Default: 50
- `categorical_ratio_threshold` (float) - Default: 0.5
- `top_n_values` (int) - Default: 10
- `outlier_iqr_multiplier` (float) - Default: 1.5
- `outlier_alert_threshold` (float) - Default: 0.05

### Example 1: Default Configuration

```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv"
```

**Result:**
- Loads all thresholds from `config.json`
- `overrides` in audit trail: `{}`

### Example 2: Partial Override

```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@data.csv" \
  -F "null_alert_threshold=15.0" \
  -F "categorical_threshold=30"
```

**Result:**
- Loads defaults from `config.json`
- Overrides 2 parameters
- `overrides` in audit trail: `{"null_alert_threshold": 15.0, "categorical_threshold": 30}`

### Example 3: Full Override

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

**Result:**
- Uses all user values directly (config.json NOT loaded)
- `overrides` in audit trail: All 6 parameters

---

## 📁 Files Structure

```
Agensium-new/
├── app/
│   ├── agents/
│   │   └── unified_profiler.py          # Main agent implementation
│   ├── routes.py                         # FastAPI endpoint
│   └── config.py                         # AGENT_ROUTES configuration
├── config.json                           # Default thresholds
├── requirements.txt                      # Dependencies (includes openpyxl)
├── test_unified_profiler.py             # Basic tests
├── test_audit_trail_and_excel.py        # Audit trail & Excel tests
├── UNIFIED_PROFILER_API.md              # API documentation
├── UNIFIED_PROFILER_AUDIT_TRAIL.md      # Audit trail details
├── UNIFIED_PROFILER_SUMMARY.md          # Configuration logic
└── UNIFIED_PROFILER_COMPLETE_GUIDE.md   # This file
```

---

## 🧪 Testing

### Run Basic Test
```bash
python test_unified_profiler.py
```

### Run Audit Trail Test
```bash
python test_audit_trail_and_excel.py
```

### Manual cURL Test
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@sample_data.csv" \
  -F "null_alert_threshold=15.0" \
  | jq '.audit'
```

---

## ✅ Implementation Checklist

- [x] **agent_name** in audit trail
- [x] **timestamp** in audit trail
- [x] **fields_scanned** array in audit trail
- [x] **findings** with detailed issue information
- [x] **actions** list of operations performed
- [x] **scores** with key metrics
- [x] **overrides** capturing user input
- [x] **excel_export** blob with base64 data
- [x] Excel file with multiple sheets
- [x] User override tracking in endpoint
- [x] Smart configuration loading
- [x] Comprehensive documentation

---

## 🎓 Key Benefits

1. **Complete Traceability** - Know exactly what happened and when
2. **Audit Compliance** - Full audit trail for regulatory requirements
3. **Easy Sharing** - Excel reports for non-technical stakeholders
4. **Reproducibility** - User overrides captured for exact replication
5. **Quantitative Assessment** - Scores provide quick data quality metrics
6. **Detailed Diagnostics** - Findings include outlier samples and context
7. **Multi-Sheet Support** - Handles complex Excel workbooks
8. **Performance Optimized** - Smart config loading reduces I/O

---

## 📦 Dependencies

Required packages (already in `requirements.txt`):
- `fastapi` - Web framework
- `pandas` - Data manipulation
- `openpyxl` - Excel generation
- `scipy` - Entropy calculation
- `numpy` - Numeric operations
- `python-multipart` - File upload handling

---

## 🔧 Configuration

### Default Thresholds (`config.json`)

```json
{
  "UnifiedProfiler": {
    "null_alert_threshold": 20.0,
    "categorical_threshold": 50,
    "categorical_ratio_threshold": 0.5,
    "top_n_values": 10,
    "outlier_iqr_multiplier": 1.5,
    "outlier_alert_threshold": 0.05
  }
}
```

### Agent Routes (`app/config.py`)

```python
AGENT_ROUTES = {
    "unified_profiler": "/unified-profiler",
    "clean_data_tool": "/run-tool/cleaner",
    "readiness_rater": "/rate-readiness",
    # ...
}
```

---

## 🎯 Next Steps

1. **Start the server:** `uvicorn app.main:app --reload`
2. **Run tests:** `python test_audit_trail_and_excel.py`
3. **Integrate with UI:** Use the Excel export feature in your frontend
4. **Monitor audit trails:** Track agent performance over time
5. **Customize thresholds:** Adjust based on your data quality standards

---

## 📞 Support

For issues or questions:
- Check the API documentation: `UNIFIED_PROFILER_API.md`
- Review audit trail details: `UNIFIED_PROFILER_AUDIT_TRAIL.md`
- Test with provided scripts: `test_*.py`

**Version:** 1.0.0  
**Last Updated:** October 22, 2025
