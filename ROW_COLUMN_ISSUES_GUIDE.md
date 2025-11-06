# Row/Column-Level Issues in UnifiedProfiler

## Overview

The UnifiedProfiler now tracks and displays detailed row/column-level issues with specific row indices and problematic values. This enables precise identification and remediation of data quality problems.

---

## Issue Types Tracked

### 1. **Null Values**
- **Description:** Missing or null values in specific rows
- **Severity:** Warning (if > 20% nulls), Info (otherwise)
- **Information Provided:**
  - Row index where null occurs
  - Column name
  - Null percentage for the column

**Example:**
```json
{
  "row_index": 42,
  "column": "age",
  "issue_type": "null_value",
  "severity": "warning",
  "value": null,
  "message": "Null value in column 'age'"
}
```

### 2. **Outliers**
- **Description:** Values outside the normal range (IQR method)
- **Severity:** Warning
- **Information Provided:**
  - Row index of outlier
  - Column name
  - Actual outlier value
  - Lower and upper bounds

**Example:**
```json
{
  "row_index": 156,
  "column": "salary",
  "issue_type": "outlier",
  "severity": "warning",
  "value": 950000.0,
  "message": "Outlier detected in column 'salary': 950000.0",
  "bounds": {
    "lower": 30000.0,
    "upper": 120000.0
  }
}
```

### 3. **Constant Columns**
- **Description:** Columns with only one unique value
- **Severity:** Info
- **Information Provided:**
  - Column name
  - The single constant value
  - Affects all rows

**Example:**
```json
{
  "row_index": null,
  "column": "country",
  "issue_type": "constant_column",
  "severity": "info",
  "value": "USA",
  "message": "Column 'country' has only one unique value"
}
```

### 4. **Duplicate Rows**
- **Description:** Identical rows in the dataset
- **Severity:** Warning
- **Information Provided:**
  - List of all duplicate row indices
  - Count of duplicates
  - Affects entire row (all columns)

**Example:**
```json
{
  "row_index": [23, 45, 67],
  "column": null,
  "issue_type": "duplicate_row",
  "severity": "warning",
  "value": null,
  "message": "Duplicate row found at indices [23, 45, 67]",
  "duplicate_count": 3
}
```

---

## API Response Structure

### Complete Response Example

```json
{
  "source_file": "customers.csv",
  "agent": "UnifiedProfiler",
  "audit": { /* audit trail */ },
  "results": {
    "customers": {
      "status": "success",
      "metadata": {
        "total_rows": 1000,
        "total_issues": 87
      },
      "alerts": [ /* column-level alerts */ ],
      "routing": { /* routing info */ },
      "data": {
        "columns": { /* column profiles */ },
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
            "row_index": 42,
            "column": "age",
            "issue_type": "outlier",
            "severity": "warning",
            "value": 150,
            "message": "Outlier detected in column 'age': 150",
            "bounds": {"lower": 18, "upper": 85}
          }
          /* ... up to 100 issues shown */
        ]
      },
      "issue_summary": {
        "total_issues": 87,
        "by_type": {
          "null_value": 45,
          "outlier": 12,
          "duplicate_row": 30
        },
        "by_severity": {
          "warning": 42,
          "info": 45
        },
        "by_column": {
          "email": 20,
          "age": 15,
          "salary": 10
        }
      }
    }
  },
  "excel_export": { /* Excel export blob */ }
}
```

---

## Issue Summary Structure

The `issue_summary` provides aggregated statistics:

```json
{
  "total_issues": 87,
  "by_type": {
    "null_value": 45,
    "outlier": 12,
    "constant_column": 0,
    "duplicate_row": 30
  },
  "by_severity": {
    "warning": 42,
    "info": 45,
    "critical": 0
  },
  "by_column": {
    "email": 20,
    "age": 15,
    "salary": 10,
    "phone": 12
  }
}
```

---

## Performance Considerations

To maintain performance on large datasets:

1. **Null Values:** Limited to first 20 occurrences per column
2. **Outliers:** Limited to first 20 occurrences per column
3. **Duplicate Rows:** Limited to first 10 duplicate groups
4. **Total Issues Returned:** Limited to first 100 issues in response

The `issue_summary` always shows complete counts, even if not all individual issues are returned.

---

## UX Display Examples

### Issue List View

```
Data Quality Issues (87 total)

Row-Level Issues:
├─ Row 5, Column 'email': Null value ⚠
├─ Row 42, Column 'age': Outlier (150) - Expected: 18-85 ⚠
├─ Row 67, Column 'salary': Outlier (950000) ⚠
├─ Rows [23, 45, 67]: Duplicate rows (3 copies) ⚠
└─ Column 'country': Constant value 'USA' ℹ

Issue Summary:
├─ Null Values: 45
├─ Outliers: 12
└─ Duplicates: 30 rows
```

### Issue Table View

| Row | Column | Issue Type | Severity | Value | Details |
|-----|--------|------------|----------|-------|---------|
| 5 | email | Null Value | ⚠ Warning | null | Missing email |
| 42 | age | Outlier | ⚠ Warning | 150 | Outside range 18-85 |
| 67 | salary | Outlier | ⚠ Warning | 950000 | Outside range 30k-120k |
| 23,45,67 | (all) | Duplicate | ⚠ Warning | - | 3 identical rows |

### Issue Heatmap

```
Column-Level Issue Distribution:

email    ████████████████████ 20 issues
age      ███████████████ 15 issues
salary   ██████████ 10 issues
phone    ████████████ 12 issues
```

---

## Frontend Implementation

### Accessing Issues

```javascript
const response = await fetch('/run-agent/profile-dataset', {
  method: 'POST',
  body: formData
});

const data = await response.json();

// Access row-level issues
const issues = data.results['Sheet1'].data.row_level_issues;
const summary = data.results['Sheet1'].issue_summary;

console.log(`Total issues: ${summary.total_issues}`);
console.log(`Null values: ${summary.by_type.null_value}`);
console.log(`Outliers: ${summary.by_type.outlier}`);
```

### React Component Example

```jsx
function IssuesList({ issues, summary }) {
  return (
    <div className="issues-panel">
      <h3>Data Quality Issues ({summary.total_issues})</h3>
      
      {/* Summary Cards */}
      <div className="issue-summary">
        <Card>
          <h4>Null Values</h4>
          <span className="count">{summary.by_type.null_value || 0}</span>
        </Card>
        <Card>
          <h4>Outliers</h4>
          <span className="count">{summary.by_type.outlier || 0}</span>
        </Card>
        <Card>
          <h4>Duplicates</h4>
          <span className="count">{summary.by_type.duplicate_row || 0}</span>
        </Card>
      </div>
      
      {/* Detailed Issue List */}
      <table className="issues-table">
        <thead>
          <tr>
            <th>Row</th>
            <th>Column</th>
            <th>Issue Type</th>
            <th>Value</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          {issues.map((issue, idx) => (
            <tr key={idx} className={`severity-${issue.severity}`}>
              <td>{Array.isArray(issue.row_index) 
                    ? issue.row_index.join(', ') 
                    : issue.row_index}</td>
              <td>{issue.column || '(all)'}</td>
              <td>{issue.issue_type}</td>
              <td>{issue.value !== null ? issue.value : '-'}</td>
              <td>
                <Badge severity={issue.severity}>
                  {issue.severity}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {summary.total_issues > issues.length && (
        <p className="truncation-notice">
          Showing first {issues.length} of {summary.total_issues} issues
        </p>
      )}
    </div>
  );
}
```

### Filtering Issues

```javascript
// Filter by issue type
const nullIssues = issues.filter(i => i.issue_type === 'null_value');
const outlierIssues = issues.filter(i => i.issue_type === 'outlier');
const duplicateIssues = issues.filter(i => i.issue_type === 'duplicate_row');

// Filter by column
const emailIssues = issues.filter(i => i.column === 'email');

// Filter by severity
const warningIssues = issues.filter(i => i.severity === 'warning');

// Get issues for specific row
const row42Issues = issues.filter(i => i.row_index === 42);
```

---

## Use Cases

### 1. Data Cleaning Workflow

```
1. Upload file to UnifiedProfiler
2. Review issue_summary to understand problem areas
3. Filter issues by type (e.g., null_value)
4. Export row indices to CSV
5. Fix issues in source data
6. Re-upload and verify
```

### 2. Data Quality Dashboard

```
Display real-time metrics:
- Total issues count
- Issues by type (pie chart)
- Issues by column (bar chart)
- Severity distribution
- Trend over time
```

### 3. Automated Data Validation

```javascript
// Fail pipeline if too many critical issues
if (summary.by_severity.critical > 0) {
  throw new Error('Critical data quality issues detected');
}

// Warn if too many warnings
if (summary.by_severity.warning > 50) {
  console.warn('High number of data quality warnings');
}
```

### 4. Issue Export for Remediation

```javascript
// Export issues to CSV for data team
const csvData = issues.map(issue => ({
  row: issue.row_index,
  column: issue.column,
  type: issue.issue_type,
  value: issue.value,
  message: issue.message
}));

downloadCSV(csvData, 'data_quality_issues.csv');
```

---

## Configuration

Control issue detection thresholds in `config.json`:

```json
{
  "UnifiedProfiler": {
    "null_alert_threshold": 20.0,
    "outlier_iqr_multiplier": 1.5,
    "outlier_alert_threshold": 0.05
  }
}
```

- **null_alert_threshold:** Percentage of nulls to trigger warning severity (default: 20%)
- **outlier_iqr_multiplier:** IQR multiplier for outlier detection (default: 1.5)
- **outlier_alert_threshold:** Percentage of outliers to trigger alert (default: 5%)

---

## Benefits

✅ **Precise Issue Location** - Know exactly which rows have problems  
✅ **Actionable Insights** - See actual values and expected ranges  
✅ **Efficient Remediation** - Export row indices for targeted fixes  
✅ **Quality Metrics** - Track issues by type, severity, and column  
✅ **Performance Optimized** - Limits prevent overwhelming large datasets  
✅ **Comprehensive Coverage** - Detects nulls, outliers, constants, duplicates  

---

## Example Workflow

```bash
# 1. Upload file
curl -X POST http://localhost:8000/run-agent/profile-dataset \
  -F "file=@customers.csv"

# 2. Response includes row-level issues
{
  "results": {
    "customers": {
      "data": {
        "row_level_issues": [...],
      },
      "issue_summary": {
        "total_issues": 87,
        "by_type": {...}
      }
    }
  }
}

# 3. Frontend displays issues in table/list
# 4. User filters by column or type
# 5. User exports problematic rows
# 6. User fixes data and re-uploads
```

---

## Summary

The UnifiedProfiler now provides granular row/column-level issue tracking that enables:
- Precise identification of data quality problems
- Specific row indices for targeted remediation
- Aggregated summaries for quick assessment
- Performance-optimized for large datasets
- Rich UX possibilities for data quality dashboards

This enhancement transforms the UnifiedProfiler from a summary tool into a precise diagnostic instrument for data quality management.
