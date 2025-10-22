# ReadinessRater - Enhanced with Audit Trail & Excel Export

## 🎯 Overview

The **ReadinessRater** agent has been enhanced to match the Unified Profiler architecture with comprehensive audit trail tracking and Excel export capabilities.

## ✨ What's New (v1.2.0)

### Enhanced Audit Trail
✅ **agent_name** - Identifies the agent: `"ReadinessRater"`  
✅ **timestamp** - ISO 8601 datetime of execution  
✅ **fields_scanned** - Array of all column names analyzed  
✅ **findings** - Detailed issues with scores and deductions  
✅ **actions** - List of operations performed  
✅ **scores** - Comprehensive metrics including average readiness scores  
✅ **overrides** - User-provided threshold parameters  

### Excel Export
✅ **Base64-encoded blob** - Ready for download  
✅ **Multi-sheet workbook** - Organized results  
✅ **Summary metrics** - Readiness scores at a glance  
✅ **Detailed breakdowns** - Scores, deductions, routing per sheet  

### Optional Parameters
✅ **ready_threshold** - Score threshold for "Ready" status (default: 85)  
✅ **needs_review_threshold** - Score threshold for "Needs Review" (default: 70)  
✅ **completeness_weight** - Weight for completeness score (default: 0.4)  
✅ **consistency_weight** - Weight for consistency score (default: 0.4)  
✅ **schema_health_weight** - Weight for schema health score (default: 0.2)  

---

## 📋 Complete Response Structure

```json
{
  "source_file": "data.csv",
  "agent": "ReadinessRater",
  
  "audit": {
    "agent_name": "ReadinessRater",
    "timestamp": "2025-10-22T10:15:30.123456+00:00",
    "profile_date": "2025-10-22T10:15:30.123456+00:00",
    "agent_version": "1.2.0",
    "compute_time_seconds": 0.85,
    
    "fields_scanned": ["customer_id", "age", "income", "city"],
    
    "findings": [
      {
        "severity": "warning",
        "sheet": "data",
        "issue": "Overall readiness score is 72. Manual review is recommended.",
        "category": "readiness_assessment",
        "readiness_score": 72,
        "completeness_score": 85,
        "consistency_score": 60,
        "schema_health_score": 70
      },
      {
        "severity": "info",
        "sheet": "data",
        "issue": "Completeness: Score reduced by 15.0 points due to 15.0% null values.",
        "category": "data_completeness"
      },
      {
        "severity": "info",
        "sheet": "data",
        "issue": "Consistency: Score reduced by 40.0 points due to 400 duplicate rows (40.0%).",
        "category": "data_consistency"
      }
    ],
    
    "actions": [
      "Analyzed 1000 rows across 1 sheet(s)",
      "Generated 3 alert(s)",
      "Calculated completeness score based on null values",
      "Calculated consistency score based on duplicate rows",
      "Calculated schema health score based on data variance",
      "Computed overall readiness score (weighted average)"
    ],
    
    "scores": {
      "total_sheets_analyzed": 1,
      "total_rows_analyzed": 1000,
      "total_alerts_generated": 3,
      "critical_alerts": 0,
      "warning_alerts": 1,
      "info_alerts": 2,
      "average_readiness_score": 72,
      "average_completeness_score": 85,
      "average_consistency_score": 60,
      "average_schema_health_score": 70
    },
    
    "overrides": {
      "ready_threshold": 90,
      "completeness_weight": 0.5
    }
  },
  
  "results": {
    "data": {
      "status": "success",
      "metadata": {
        "total_rows_analyzed": 1000
      },
      "routing": {
        "status": "Needs Review",
        "reason": "Dataset has moderate quality issues that should be reviewed.",
        "suggestion": "Run the 'Clean My Data' tool to address issues.",
        "suggested_agent_endpoint": "/run-tool/cleaner"
      },
      "alerts": [
        {
          "level": "warning",
          "message": "Overall readiness score is 72. Manual review is recommended."
        }
      ],
      "data": {
        "readiness_score": {
          "overall": 72,
          "completeness": 85,
          "consistency": 60,
          "schema_health": 70
        },
        "deductions": [
          "Completeness: Score reduced by 15.0 points due to 15.0% null values.",
          "Consistency: Score reduced by 40.0 points due to 400 duplicate rows (40.0%).",
          "Schema Health: Score reduced by 10 points because column 'status' has only one unique value."
        ]
      }
    }
  },
  
  "excel_export": {
    "filename": "data_readiness_report.xlsx",
    "size_bytes": 32456,
    "format": "xlsx",
    "base64_data": "UEsDBBQAAAAIAH1p...",
    "sheets_included": [
      "Summary",
      "Fields Scanned",
      "Findings",
      "Actions",
      "Configuration",
      "User Overrides",
      "data_Scores",
      "data_Route"
    ],
    "download_ready": true
  }
}
```

---

## 🚀 API Usage

### Endpoint
```
POST /rate-readiness
```

### Parameters

**Required:**
- `file` (UploadFile) - CSV or Excel file

**Optional Thresholds:**
- `ready_threshold` (int) - Score for "Ready" status (default: 85)
- `needs_review_threshold` (int) - Score for "Needs Review" (default: 70)
- `completeness_weight` (float) - Weight for completeness (default: 0.4)
- `consistency_weight` (float) - Weight for consistency (default: 0.4)
- `schema_health_weight` (float) - Weight for schema health (default: 0.2)

### Example 1: Default Configuration

```bash
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@data.csv"
```

**Result:**
- Uses all defaults from `config.json`
- `overrides: {}`

### Example 2: Custom Thresholds

```bash
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@data.csv" \
  -F "ready_threshold=90" \
  -F "completeness_weight=0.5" \
  -F "consistency_weight=0.3" \
  -F "schema_health_weight=0.2"
```

**Result:**
- Overrides 4 parameters
- `overrides: {"ready_threshold": 90, "completeness_weight": 0.5, ...}`

### Example 3: Python with Custom Weights

```python
import requests

with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/rate-readiness',
        files={'file': f},
        data={
            'ready_threshold': 90,
            'needs_review_threshold': 75,
            'completeness_weight': 0.5,
            'consistency_weight': 0.3,
            'schema_health_weight': 0.2
        }
    )

result = response.json()
print(f"Readiness Score: {result['audit']['scores']['average_readiness_score']}")
```

---

## 📊 Audit Trail Components

### 1. agent_name
- **Value:** `"ReadinessRater"`
- **Purpose:** Identifies which agent executed

### 2. timestamp
- **Format:** ISO 8601 with timezone
- **Example:** `"2025-10-22T10:15:30.123456+00:00"`
- **Purpose:** Records exact execution time

### 3. fields_scanned
- **Type:** Array of strings
- **Example:** `["customer_id", "age", "income", "city"]`
- **Purpose:** Lists all columns analyzed

### 4. findings
- **Type:** Array of finding objects
- **Structure:**
  ```json
  {
    "severity": "warning|critical|info",
    "sheet": "sheet_name",
    "issue": "Human-readable description",
    "category": "readiness_assessment|data_completeness|data_consistency|schema_health",
    "readiness_score": 72,
    "completeness_score": 85,
    "consistency_score": 60,
    "schema_health_score": 70
  }
  ```

**Finding Categories:**
- `readiness_assessment` - Overall readiness alerts
- `data_completeness` - Null value issues
- `data_consistency` - Duplicate row issues
- `schema_health` - Schema quality issues

### 5. actions
- **Type:** Array of strings
- **Examples:**
  - `"Analyzed 1000 rows across 1 sheet(s)"`
  - `"Calculated completeness score based on null values"`
  - `"Computed overall readiness score (weighted average)"`

### 6. scores
- **Type:** Object with metrics
- **Metrics:**
  - `total_sheets_analyzed` - Number of sheets processed
  - `total_rows_analyzed` - Total rows examined
  - `total_alerts_generated` - Total alerts raised
  - `critical_alerts`, `warning_alerts`, `info_alerts` - Alert counts by severity
  - `average_readiness_score` - Average overall score
  - `average_completeness_score` - Average completeness
  - `average_consistency_score` - Average consistency
  - `average_schema_health_score` - Average schema health

### 7. overrides
- **Type:** Object
- **Purpose:** Captures user-provided parameters
- **Example:**
  ```json
  {
    "ready_threshold": 90,
    "completeness_weight": 0.5
  }
  ```

---

## 📊 Excel Export Structure

### Sheet 1: Summary
| Metric | Value |
|--------|-------|
| Source File | data.csv |
| Agent | ReadinessRater |
| Version | 1.2.0 |
| Timestamp | 2025-10-22T... |
| Average Readiness Score | 72 |
| Average Completeness Score | 85 |
| Average Consistency Score | 60 |
| Average Schema Health Score | 70 |

### Sheet 2: Fields Scanned
| Field Name |
|------------|
| customer_id |
| age |
| income |
| city |

### Sheet 3: Findings
| severity | sheet | issue | category | readiness_score |
|----------|-------|-------|----------|-----------------|
| warning | data | Overall readiness score is 72... | readiness_assessment | 72 |
| info | data | Completeness: Score reduced... | data_completeness | - |

### Sheet 4: Actions
| Action |
|--------|
| Analyzed 1000 rows across 1 sheet(s) |
| Calculated completeness score based on null values |
| ... |

### Sheet 5: Configuration
| Parameter | Value |
|-----------|-------|
| ready_threshold | 85 |
| needs_review_threshold | 70 |
| completeness_weight | 0.4 |
| ... |

### Sheet 6: User Overrides (if any)
| Parameter | User Value |
|-----------|------------|
| ready_threshold | 90 |
| completeness_weight | 0.5 |

### Sheet 7+: Per-Sheet Results
- **[SheetName]_Scores** - Readiness scores breakdown
- **[SheetName]_Deduct** - Deductions list
- **[SheetName]_Route** - Routing decision

---

## 🎓 Configuration Guide

### Default Configuration (`config.json`)

```json
{
  "ReadinessRater": {
    "ready_threshold": 85,
    "needs_review_threshold": 70,
    "completeness_weight": 0.4,
    "consistency_weight": 0.4,
    "schema_health_weight": 0.2
  }
}
```

### Threshold Tuning

**More Strict (Higher Standards):**
```json
{
  "ready_threshold": 90,
  "needs_review_threshold": 80
}
```

**More Lenient (Lower Standards):**
```json
{
  "ready_threshold": 75,
  "needs_review_threshold": 60
}
```

### Weight Adjustments

**Prioritize Completeness:**
```json
{
  "completeness_weight": 0.6,
  "consistency_weight": 0.2,
  "schema_health_weight": 0.2
}
```

**Prioritize Consistency:**
```json
{
  "completeness_weight": 0.2,
  "consistency_weight": 0.6,
  "schema_health_weight": 0.2
}
```

**Equal Weights:**
```json
{
  "completeness_weight": 0.33,
  "consistency_weight": 0.33,
  "schema_health_weight": 0.34
}
```

---

## 🔄 Comparison: Before vs After

### Before (v1.1.0)
```json
{
  "audit": {
    "profile_date": "...",
    "agent_version": "1.1.0",
    "compute_time_seconds": 0.85
  }
}
```

### After (v1.2.0)
```json
{
  "audit": {
    "agent_name": "ReadinessRater",           ← NEW
    "timestamp": "...",                       ← NEW
    "profile_date": "...",
    "agent_version": "1.2.0",
    "compute_time_seconds": 0.85,
    "fields_scanned": [...],                  ← NEW
    "findings": [...],                        ← NEW
    "actions": [...],                         ← NEW
    "scores": {...},                          ← NEW (enhanced)
    "overrides": {...}                        ← NEW
  },
  "excel_export": {...}                       ← NEW
}
```

---

## ✅ Benefits

1. **Complete Traceability** - Know exactly what was analyzed and when
2. **Audit Compliance** - Full trail for regulatory requirements
3. **Reproducibility** - User overrides enable exact replication
4. **Easy Sharing** - Excel reports for stakeholders
5. **Flexible Scoring** - Adjustable weights and thresholds
6. **Detailed Diagnostics** - Findings include scores and deductions
7. **Multi-Sheet Support** - Handles complex Excel workbooks

---

## 📦 Dependencies

All required packages already in `requirements.txt`:
- `pandas` ✓
- `openpyxl` ✓
- `fastapi` ✓

---

## 🧪 Testing

```bash
# Test with default config
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@data.csv"

# Test with custom thresholds
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@data.csv" \
  -F "ready_threshold=90" \
  -F "completeness_weight=0.5"

# View audit trail
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@data.csv" \
  | jq '.audit'
```

---

**Version:** 1.2.0  
**Last Updated:** October 22, 2025  
**Compatible With:** Unified Profiler v1.0.0
