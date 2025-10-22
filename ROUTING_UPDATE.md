# Routing Logic Update - Master My Data Tool

## 🔄 Changes Made

Updated routing logic in three agents to return `master_my_data_tool` endpoint instead of `None` when data is clean/ready.

---

## ✅ Updated Agents

### 1. **Unified Profiler**

**Before:**
```python
"status": "Ready"
"suggestion": "Proceed to assess dataset readiness for analysis."
"suggested_agent_endpoint": AGENT_ROUTES.get("readiness_rater", ...)
```

**After:**
```python
"status": "Ready"
"suggestion": "Proceed to master data tool for entity resolution and deduplication."
"suggested_agent_endpoint": AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
```

**Trigger:** No significant data quality issues detected

---

### 2. **ReadinessRater**

**Before:**
```python
"status": "Ready"
"suggestion": "Proceed with downstream analytics or ML pipelines."
"suggested_agent_endpoint": None
```

**After:**
```python
"status": "Ready"
"suggestion": "Proceed to master data tool for entity resolution and deduplication."
"suggested_agent_endpoint": AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
```

**Trigger:** Overall readiness score >= ready_threshold (default: 85)

---

### 3. **RiskScorer**

**Before:**
```python
"status": "Low Risk"
"suggestion": "Continue monitoring data quality and governance"
"suggested_agent_endpoint": None
```

**After:**
```python
"status": "Low Risk"
"suggestion": "Proceed to master data tool for entity resolution and deduplication"
"suggested_agent_endpoint": AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
```

**Trigger:** Risk score < 50 AND no PII detected AND no governance gaps

---

## 🔀 Complete Routing Flow

### Unified Profiler
```
IF data quality issues detected:
  → "Needs Review" → clean_data_tool
ELSE:
  → "Ready" → master_my_data_tool ✓ (UPDATED)
```

### ReadinessRater
```
IF score >= 85:
  → "Ready" → master_my_data_tool ✓ (UPDATED)
ELIF score >= 70:
  → "Needs Review" → clean_data_tool
ELSE:
  → "Not Ready" → clean_data_tool
```

### RiskScorer
```
IF risk >= 75 OR pii_count > 0:
  → "High Risk" → govern_data_tool
ELIF risk >= 50 OR governance_gaps > 0:
  → "Medium Risk" → govern_data_tool
ELSE:
  → "Low Risk" → master_my_data_tool ✓ (UPDATED)
```

---

## 🎯 Workflow Integration

### Recommended Data Processing Pipeline

```
1. Upload Data
   ↓
2. Unified Profiler
   → Profiles fields, detects outliers
   ↓
   IF clean:
     → master_my_data_tool ✓
   ELSE:
     → clean_data_tool
   ↓
3. ReadinessRater
   → Assesses completeness, consistency, schema health
   ↓
   IF score >= 85:
     → master_my_data_tool ✓
   ELSE:
     → clean_data_tool
   ↓
4. RiskScorer
   → Detects PII, sensitive data, governance gaps
   ↓
   IF low risk:
     → master_my_data_tool ✓
   ELSE:
     → govern_data_tool
   ↓
5. Master My Data Tool
   → Entity resolution
   → Deduplication
   → Data mastering
```

---

## 📋 Master My Data Tool

The `master_my_data_tool` endpoint (`/run-tool/master-my-data`) typically handles:

- **Entity Resolution** - Match and merge duplicate entities
- **Deduplication** - Remove duplicate records
- **Data Standardization** - Normalize formats
- **Golden Record Creation** - Create master records
- **Relationship Mapping** - Link related entities

---

## ✅ Benefits

✓ **Clear Next Steps** - Always provides actionable routing  
✓ **Automated Workflow** - Seamless progression through pipeline  
✓ **No Dead Ends** - Every status has a suggested next action  
✓ **Logical Flow** - Clean data → Master data → Analytics  

---

## 🧪 Testing

### Test Unified Profiler
```bash
curl -X POST "http://localhost:8000/unified-profiler" \
  -F "file=@clean_data.csv" \
  | jq '.results[].routing'
```

**Expected:**
```json
{
  "status": "Ready",
  "suggested_agent_endpoint": "/run-tool/master-my-data"
}
```

### Test ReadinessRater
```bash
curl -X POST "http://localhost:8000/rate-readiness" \
  -F "file=@clean_data.csv" \
  | jq '.results[].routing'
```

**Expected:**
```json
{
  "status": "Ready",
  "suggested_agent_endpoint": "/run-tool/master-my-data"
}
```

### Test RiskScorer
```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@clean_data.csv" \
  | jq '.results[].routing'
```

**Expected:**
```json
{
  "status": "Low Risk",
  "suggested_agent_endpoint": "/run-tool/master-my-data"
}
```

---

**Status:** ✅ Complete  
**Date:** October 22, 2025  
**Agents Updated:** Unified Profiler, ReadinessRater, RiskScorer
