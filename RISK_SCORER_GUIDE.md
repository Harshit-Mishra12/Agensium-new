# RiskScorer - Complete Implementation Guide

## 🎯 Overview

The **RiskScorer** agent flags PII, sensitive data, governance gaps, and compliance risks with comprehensive audit trail and Excel export capabilities.

## ✨ Features

✅ **PII Detection** - Email, SSN, phone, credit card, IP address, passport  
✅ **Sensitive Field Detection** - Field name pattern matching  
✅ **Governance Gap Analysis** - Missing consent, classification, retention policies  
✅ **Risk Scoring** - Field-level and overall risk assessment  
✅ **Comprehensive Audit Trail** - All 7 components implemented  
✅ **Excel Export** - Downloadable risk reports  
✅ **Optional Parameters** - Configurable thresholds and detection settings  

---

## 📋 Response Structure

```json
{
  "source_file": "customer_data.csv",
  "agent": "RiskScorer",
  
  "audit": {
    "agent_name": "RiskScorer",
    "timestamp": "2025-10-22T16:30:00+00:00",
    "profile_date": "2025-10-22T16:30:00+00:00",
    "agent_version": "1.0.0",
    "compute_time_seconds": 1.45,
    
    "fields_scanned": ["customer_id", "email", "ssn", "phone"],
    
    "findings": [
      {
        "severity": "critical",
        "sheet": "customers",
        "field": "email",
        "issue": "PII detected: email",
        "category": "pii_detection",
        "risk_score": 30
      }
    ],
    
    "actions": [
      "Analyzed 1000 rows across 1 sheet(s)",
      "Detected 3 risk(s)",
      "Scanned for PII patterns (email, SSN, phone, credit card, etc.)",
      "Checked field names for sensitive data indicators",
      "Assessed governance and compliance gaps",
      "Calculated risk scores per field and overall"
    ],
    
    "scores": {
      "total_sheets_analyzed": 1,
      "total_rows_analyzed": 1000,
      "total_risks_detected": 3,
      "critical_risks": 2,
      "high_risks": 1,
      "medium_risks": 0,
      "low_risks": 0,
      "average_risk_score": 65,
      "pii_fields_detected": 2,
      "sensitive_fields_detected": 1,
      "governance_gaps": 2
    },
    
    "overrides": {}
  },
  
  "results": {...},
  "excel_export": {...}
}
```

---

## 🔍 PII Detection Patterns

| PII Type | Pattern | Example |
|----------|---------|---------|
| **Email** | Standard email regex | `user@example.com` |
| **SSN** | XXX-XX-XXXX | `123-45-6789` |
| **Phone** | Various formats | `(555) 123-4567` |
| **Credit Card** | 16 digits with separators | `1234-5678-9012-3456` |
| **IP Address** | IPv4 format | `192.168.1.1` |
| **Passport** | 1-2 letters + 6-9 digits | `AB1234567` |

---

## 🚨 Sensitive Field Names

The agent checks field names for these terms:
- **Identity:** ssn, social_security, tax_id, passport, license
- **Financial:** credit_card, salary, income, wage
- **Personal:** dob, birth_date, medical, health, diagnosis
- **Protected:** religion, ethnicity, race, political, sexual_orientation
- **Security:** password, secret, token, api_key

---

## 📊 Risk Scoring

### Risk Levels

| Risk Level | Score Range | Action Required |
|------------|-------------|-----------------|
| **Critical** | ≥ 75 | Immediate action |
| **High** | 50-74 | Review and remediate |
| **Medium** | 25-49 | Monitor and plan |
| **Low** | 0-24 | Continue monitoring |

### Risk Factors

| Factor | Points | Description |
|--------|--------|-------------|
| Sensitive field name | +20 | Field name contains sensitive term |
| PII detected | +30 | PII pattern found in data |
| Missing consent | +15 | No consent tracking for PII/sensitive field |
| Missing classification | +10 | No data classification metadata |

---

## 🚀 API Usage

### Endpoint
```
POST /score-risk
```

### Parameters

**Required:**
- `file` (UploadFile) - CSV or Excel file

**Optional:**
- `pii_sample_size` (int, default: 100) - Number of rows to sample for PII detection
- `high_risk_threshold` (int, default: 75) - Score threshold for critical risk
- `medium_risk_threshold` (int, default: 50) - Score threshold for high risk
- `pii_detection_enabled` (bool, default: true) - Enable PII pattern detection
- `sensitive_field_detection_enabled` (bool, default: true) - Enable field name checking
- `governance_check_enabled` (bool, default: true) - Enable governance gap analysis

### Example 1: Default Configuration

```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@customer_data.csv"
```

### Example 2: Custom Thresholds

```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@customer_data.csv" \
  -F "high_risk_threshold=80" \
  -F "pii_sample_size=200"
```

### Example 3: Disable Specific Checks

```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@customer_data.csv" \
  -F "governance_check_enabled=false"
```

### Example 4: Python

```python
import requests
import base64

response = requests.post(
    'http://localhost:8000/score-risk',
    files={'file': open('customer_data.csv', 'rb')},
    data={
        'high_risk_threshold': 80,
        'pii_sample_size': 200
    }
)

result = response.json()

# View risk summary
print(f"Average Risk Score: {result['audit']['scores']['average_risk_score']}")
print(f"PII Fields: {result['audit']['scores']['pii_fields_detected']}")
print(f"Governance Gaps: {result['audit']['scores']['governance_gaps']}")

# Download Excel report
excel_bytes = base64.b64decode(result['excel_export']['base64_data'])
with open(result['excel_export']['filename'], 'wb') as f:
    f.write(excel_bytes)
```

---

## 📊 Excel Export Structure

### Sheet 1: Summary
- Source file, agent info, timestamp
- Total risks detected by severity
- PII fields, sensitive fields, governance gaps

### Sheet 2: Fields Scanned
- List of all columns analyzed

### Sheet 3: Findings
- Detailed findings with severity, field, issue, category

### Sheet 4: Actions
- All operations performed by the agent

### Sheet 5: Configuration
- Thresholds and settings used

### Sheet 6+: Per-Sheet Risk Details
- Field-level risk scores
- PII types detected
- Risk summaries

---

## 🎯 Use Cases

### 1. **GDPR Compliance Check**
```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@eu_customer_data.csv" \
  -F "governance_check_enabled=true"
```

**Detects:**
- PII without consent tracking
- Missing data classification
- Sensitive fields without governance

### 2. **Pre-Production Security Scan**
```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@production_data.csv" \
  -F "high_risk_threshold=90"
```

**Detects:**
- Any PII in production datasets
- Unencrypted sensitive data
- Security token exposure

### 3. **Data Inventory Audit**
```bash
curl -X POST "http://localhost:8000/score-risk" \
  -F "file=@data_inventory.xlsx"
```

**Provides:**
- Complete PII inventory
- Risk scores per field
- Governance gap report

---

## 🔀 Routing Logic

```
IF risk_score >= 75 OR pii_count > 0:
  Status: "High Risk"
  Suggestion: Immediate action - implement governance
  Next Agent: govern_data_tool

ELIF risk_score >= 50 OR governance_gaps > 0:
  Status: "Medium Risk"
  Suggestion: Review and address gaps
  Next Agent: govern_data_tool

ELSE:
  Status: "Low Risk"
  Suggestion: Continue monitoring
  Next Agent: None
```

---

## 📋 Configuration

### Default Config (`config.json`)

```json
{
  "RiskScorer": {
    "pii_sample_size": 100,
    "high_risk_threshold": 75,
    "medium_risk_threshold": 50,
    "pii_detection_enabled": true,
    "sensitive_field_detection_enabled": true,
    "governance_check_enabled": true
  }
}
```

### Tuning Examples

**Stricter PII Detection:**
```json
{
  "pii_sample_size": 500,
  "high_risk_threshold": 60
}
```

**Focus on Governance:**
```json
{
  "governance_check_enabled": true,
  "pii_detection_enabled": false
}
```

---

## ✅ Implementation Checklist

- [x] PII detection with regex patterns
- [x] Sensitive field name checking
- [x] Governance gap analysis
- [x] Risk scoring algorithm
- [x] Comprehensive audit trail (7 components)
- [x] Excel export with base64 blob
- [x] Optional parameters support
- [x] Smart configuration loading
- [x] Routing logic
- [x] Multi-sheet Excel support

---

## 🎓 Benefits

✓ **Automated PII Discovery** - No manual data inspection  
✓ **Compliance Ready** - GDPR, CCPA, HIPAA alignment  
✓ **Risk Quantification** - Objective scoring system  
✓ **Governance Insights** - Identify missing controls  
✓ **Audit Trail** - Complete traceability  
✓ **Easy Reporting** - Excel export for stakeholders  

---

**Status:** ✅ Complete  
**Version:** 1.0.0  
**Date:** October 22, 2025
