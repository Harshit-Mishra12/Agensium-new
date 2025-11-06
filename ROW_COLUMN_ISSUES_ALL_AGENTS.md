# Row/Column-Level Issues - All Agents

## Overview

All four agents (UnifiedProfiler, DriftDetector, ReadinessRater, RiskScorer) now provide detailed row/column-level issue tracking with specific row indices and problematic values.

---

## 1. UnifiedProfiler

### Issue Types
- **Null Values** - Missing values with row indices
- **Outliers** - Statistical outliers with actual values
- **Constant Columns** - Columns with single unique value
- **Duplicate Rows** - Identical rows grouped together

### Example Response
```json
{
  "results": {
    "Sheet1": {
      "metadata": {
        "total_rows": 1000,
        "total_issues": 87
      },
      "data": {
        "row_level_issues": [
          {
            "row_index": 42,
            "column": "age",
            "issue_type": "outlier",
            "severity": "warning",
            "value": 150,
            "message": "Outlier detected in column 'age': 150",
            "bounds": {"lower": 18, "upper": 85}
          }
        ]
      },
      "issue_summary": {
        "total_issues": 87,
        "by_type": {
          "null_value": 45,
          "outlier": 12,
          "duplicate_row": 30
        }
      }
    }
  }
}
```

---

## 2. DriftDetector

### Issue Types
- **Numeric Drift** - Values significantly different from baseline (z-score > 2)
- **New Categories** - Categories not present in baseline
- **Schema Changes** - New/dropped columns

### Example Response
```json
{
  "results": {
    "dataset": {
      "metadata": {
        "baseline_rows": 1000,
        "current_rows": 1050,
        "total_issues": 23
      },
      "data": {
        "row_level_issues": [
          {
            "row_index": 15,
            "column": "salary",
            "issue_type": "numeric_drift",
            "severity": "warning",
            "value": 250000.0,
            "baseline_mean": 75000.0,
            "baseline_std": 15000.0,
            "z_score": 11.67,
            "message": "Value 250000.0 deviates significantly from baseline (z-score: 11.67)"
          },
          {
            "row_index": 42,
            "column": "department",
            "issue_type": "new_category",
            "severity": "warning",
            "value": "AI Research",
            "message": "New category 'AI Research' not present in baseline"
          }
        ]
      },
      "issue_summary": {
        "total_issues": 23,
        "by_type": {
          "numeric_drift": 8,
          "new_category": 15
        }
      }
    }
  }
}
```

---

## 3. ReadinessRater

### Issue Types
- **Null Values** - Missing values affecting completeness score
- **Duplicate Rows** - Affecting consistency score
- **Constant Columns** - Affecting schema health score

### Example Response
```json
{
  "results": {
    "Sheet1": {
      "metadata": {
        "total_rows_analyzed": 1000,
        "total_issues": 65
      },
      "data": {
        "readiness_score": {
          "overall": 78,
          "completeness": 85,
          "consistency": 90,
          "schema_health": 80
        },
        "row_level_issues": [
          {
            "row_index": 5,
            "column": "email",
            "issue_type": "null_value",
            "severity": "info",
            "value": null,
            "message": "Null value in column 'email'"
          },
          {
            "row_index": [23, 45, 67],
            "column": null,
            "issue_type": "duplicate_row",
            "severity": "warning",
            "value": null,
            "message": "Duplicate row found at indices [23, 45, 67]",
            "duplicate_count": 3
          }
        ]
      },
      "issue_summary": {
        "total_issues": 65,
        "by_type": {
          "null_value": 50,
          "duplicate_row": 15
        }
      }
    }
  }
}
```

---

## 4. RiskScorer

### Issue Types
- **PII Detected** - Personally Identifiable Information with masked values
- **Sensitive Field** - Fields containing sensitive data
- **Governance Gap** - Missing governance controls

### Example Response
```json
{
  "results": {
    "Sheet1": {
      "metadata": {
        "total_rows": 1000,
        "total_fields": 15,
        "pii_fields_count": 3,
        "total_issues": 42
      },
      "data": {
        "row_level_issues": [
          {
            "row_index": 5,
            "column": "email",
            "issue_type": "pii_detected",
            "pii_type": "email",
            "severity": "critical",
            "value": "joh***",
            "message": "PII detected (email) in column 'email'"
          },
          {
            "row_index": 12,
            "column": "ssn",
            "issue_type": "pii_detected",
            "pii_type": "ssn",
            "severity": "critical",
            "value": "123***",
            "message": "PII detected (ssn) in column 'ssn'"
          },
          {
            "row_index": null,
            "column": "salary",
            "issue_type": "sensitive_field",
            "severity": "high",
            "value": null,
            "message": "Column 'salary' contains sensitive data"
          },
          {
            "row_index": null,
            "column": "email",
            "issue_type": "governance_gap",
            "severity": "warning",
            "value": null,
            "message": "Field 'email' contains sensitive/PII data but no consent tracking detected"
          }
        ]
      },
      "issue_summary": {
        "total_issues": 42,
        "by_type": {
          "pii_detected": 25,
          "sensitive_field": 5,
          "governance_gap": 12
        },
        "by_pii_type": {
          "email": 15,
          "ssn": 8,
          "phone": 2
        }
      }
    }
  }
}
```

---

## Common Response Structure

All agents follow this pattern:

```json
{
  "results": {
    "sheet_name": {
      "metadata": {
        "total_rows": 1000,
        "total_issues": 87
      },
      "data": {
        "row_level_issues": [
          {
            "row_index": 42,
            "column": "column_name",
            "issue_type": "specific_type",
            "severity": "warning|info|critical|high",
            "value": "actual_value",
            "message": "Human-readable description"
          }
        ]
      },
      "issue_summary": {
        "total_issues": 87,
        "by_type": {},
        "by_severity": {},
        "by_column": {}
      }
    }
  }
}
```

---

## Issue Type Reference

| Agent | Issue Types |
|-------|-------------|
| **UnifiedProfiler** | null_value, outlier, constant_column, duplicate_row |
| **DriftDetector** | numeric_drift, new_category |
| **ReadinessRater** | null_value, duplicate_row, constant_column |
| **RiskScorer** | pii_detected, sensitive_field, governance_gap |

---

## Severity Levels

| Severity | Description | Use Case |
|----------|-------------|----------|
| **critical** | Immediate action required | PII exposure, schema changes |
| **high** | Important issue | Sensitive data without protection |
| **warning** | Should be reviewed | Outliers, duplicates, drift |
| **info** | Informational | Low null percentages, constant columns |

---

## Performance Limits

To maintain performance on large datasets:

| Agent | Limit |
|-------|-------|
| **UnifiedProfiler** | 20 nulls/column, 20 outliers/column, 10 duplicate groups |
| **DriftDetector** | 20 numeric drift rows, 5 rows per new category (10 categories max) |
| **ReadinessRater** | 20 nulls/column, 10 duplicate groups |
| **RiskScorer** | 10 PII matches/type, masked values for security |

All agents return max 100 issues in `row_level_issues` array. Full counts in `issue_summary`.

---

## Frontend Integration Example

```javascript
// Fetch data from any agent
const response = await fetch('/run-agent/profile-dataset', {
  method: 'POST',
  body: formData
});

const data = await response.json();

// Access issues
const sheet = data.results['Sheet1'];
const issues = sheet.data.row_level_issues;
const summary = sheet.issue_summary;

// Display summary
console.log(`Total Issues: ${summary.total_issues}`);
console.log(`By Type:`, summary.by_type);
console.log(`By Severity:`, summary.by_severity);

// Filter by type
const piiIssues = issues.filter(i => i.issue_type === 'pii_detected');
const outliers = issues.filter(i => i.issue_type === 'outlier');
const driftIssues = issues.filter(i => i.issue_type === 'numeric_drift');

// Group by column
const issuesByColumn = {};
issues.forEach(issue => {
  const col = issue.column || '(all)';
  if (!issuesByColumn[col]) issuesByColumn[col] = [];
  issuesByColumn[col].push(issue);
});

// Export to CSV for remediation
const csvData = issues.map(issue => ({
  row: issue.row_index,
  column: issue.column,
  type: issue.issue_type,
  severity: issue.severity,
  value: issue.value,
  message: issue.message
}));
```

---

## React Component Example

```jsx
function IssuesPanel({ agentResponse }) {
  const sheet = Object.values(agentResponse.results)[0];
  const { row_level_issues, issue_summary } = sheet.data;
  
  return (
    <div className="issues-panel">
      <h2>Data Quality Issues ({issue_summary.total_issues})</h2>
      
      {/* Summary Cards */}
      <div className="summary-grid">
        {Object.entries(issue_summary.by_type).map(([type, count]) => (
          <Card key={type}>
            <h4>{type.replace('_', ' ')}</h4>
            <span className="count">{count}</span>
          </Card>
        ))}
      </div>
      
      {/* Issue Table */}
      <table className="issues-table">
        <thead>
          <tr>
            <th>Row</th>
            <th>Column</th>
            <th>Type</th>
            <th>Severity</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {row_level_issues.map((issue, idx) => (
            <tr key={idx} className={`severity-${issue.severity}`}>
              <td>{Array.isArray(issue.row_index) 
                    ? issue.row_index.join(', ') 
                    : issue.row_index}</td>
              <td>{issue.column || '(all)'}</td>
              <td>{issue.issue_type}</td>
              <td><Badge severity={issue.severity}>{issue.severity}</Badge></td>
              <td>{issue.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {issue_summary.total_issues > row_level_issues.length && (
        <p className="truncation-notice">
          Showing first {row_level_issues.length} of {issue_summary.total_issues} issues
        </p>
      )}
    </div>
  );
}
```

---

## Security Considerations

### RiskScorer PII Masking
PII values are automatically masked in responses:
- **Email:** `joh***` (first 3 chars + ***)
- **SSN:** `123***`
- **Phone:** `555***`
- **Credit Card:** `4532***`

This prevents accidental PII exposure in logs or UI while still allowing row identification.

---

## Summary

All four agents now provide:
- ✅ **Precise row indices** for all issues
- ✅ **Actual values** (masked for PII)
- ✅ **Issue summaries** by type, severity, and column
- ✅ **Performance limits** to handle large datasets
- ✅ **Consistent structure** across all agents
- ✅ **Security-first** approach for sensitive data

This enables building powerful data quality dashboards with drill-down capabilities to specific problematic rows.
