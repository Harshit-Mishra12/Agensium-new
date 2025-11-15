import pandas as pd
import numpy as np
import io
import time
import base64
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings
import re
from typing import Dict, List, Any, Optional

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

def _test_uniqueness(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Test uniqueness constraints on specified columns."""
    issues = []
    score = 100
    
    unique_columns = config.get('unique_columns', [])
    if not unique_columns:
        return {"score": 100, "issues": [], "tests_performed": []}
    
    tests_performed = []
    
    for col in unique_columns:
        if col not in df.columns:
            score -= 20
            issues.append({
                "type": "missing_unique_column",
                "severity": "critical",
                "message": f"Required unique column '{col}' not found",
                "column": col,
                "score_impact": -20
            })
            continue
            
        duplicates = df[col].duplicated().sum()
        total_rows = len(df)
        duplicate_percentage = (duplicates / total_rows * 100) if total_rows > 0 else 0
        
        test_result = {
            "column": col,
            "test_type": "uniqueness",
            "total_values": total_rows,
            "duplicate_count": duplicates,
            "duplicate_percentage": round(duplicate_percentage, 2),
            "passed": duplicates == 0
        }
        tests_performed.append(test_result)
        
        if duplicates > 0:
            deduction = min(25, duplicate_percentage)
            score -= deduction
            issues.append({
                "type": "uniqueness_violation",
                "severity": "critical" if duplicate_percentage > 10 else "warning",
                "message": f"Column '{col}' has {duplicates} duplicate values ({duplicate_percentage:.1f}%)",
                "column": col,
                "duplicate_count": duplicates,
                "score_impact": -deduction
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "tests_performed": tests_performed
    }

def _test_ranges(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Test range constraints on numeric columns."""
    issues = []
    score = 100
    
    range_tests = config.get('range_tests', {})
    if not range_tests:
        return {"score": 100, "issues": [], "tests_performed": []}
    
    tests_performed = []
    
    for col, constraints in range_tests.items():
        if col not in df.columns:
            score -= 15
            issues.append({
                "type": "missing_range_column",
                "severity": "warning",
                "message": f"Range test column '{col}' not found",
                "column": col,
                "score_impact": -15
            })
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            score -= 10
            issues.append({
                "type": "non_numeric_range_column",
                "severity": "warning",
                "message": f"Column '{col}' is not numeric, cannot perform range tests",
                "column": col,
                "score_impact": -10
            })
            continue
        
        min_val = constraints.get('min')
        max_val = constraints.get('max')
        
        col_data = df[col].dropna()
        violations = 0
        
        if min_val is not None:
            below_min = (col_data < min_val).sum()
            violations += below_min
            
        if max_val is not None:
            above_max = (col_data > max_val).sum()
            violations += above_max
        
        total_valid = len(col_data)
        violation_percentage = (violations / total_valid * 100) if total_valid > 0 else 0
        
        test_result = {
            "column": col,
            "test_type": "range",
            "min_constraint": min_val,
            "max_constraint": max_val,
            "total_values": total_valid,
            "violations": violations,
            "violation_percentage": round(violation_percentage, 2),
            "passed": violations == 0
        }
        tests_performed.append(test_result)
        
        if violations > 0:
            deduction = min(20, violation_percentage)
            score -= deduction
            issues.append({
                "type": "range_violation",
                "severity": "critical" if violation_percentage > 5 else "warning",
                "message": f"Column '{col}' has {violations} values outside range [{min_val}, {max_val}] ({violation_percentage:.1f}%)",
                "column": col,
                "violations": violations,
                "score_impact": -deduction
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "tests_performed": tests_performed
    }

def _test_formats(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Test format constraints using regex patterns."""
    issues = []
    score = 100
    
    format_tests = config.get('format_tests', {})
    if not format_tests:
        return {"score": 100, "issues": [], "tests_performed": []}
    
    tests_performed = []
    
    for col, pattern_info in format_tests.items():
        if col not in df.columns:
            score -= 15
            issues.append({
                "type": "missing_format_column",
                "severity": "warning",
                "message": f"Format test column '{col}' not found",
                "column": col,
                "score_impact": -15
            })
            continue
        
        pattern = pattern_info.get('pattern') if isinstance(pattern_info, dict) else pattern_info
        description = pattern_info.get('description', 'format') if isinstance(pattern_info, dict) else 'format'
        
        try:
            col_data = df[col].dropna().astype(str)
            matches = col_data.str.match(pattern, na=False)
            violations = (~matches).sum()
            total_valid = len(col_data)
            violation_percentage = (violations / total_valid * 100) if total_valid > 0 else 0
            
            test_result = {
                "column": col,
                "test_type": "format",
                "pattern": pattern,
                "description": description,
                "total_values": total_valid,
                "violations": violations,
                "violation_percentage": round(violation_percentage, 2),
                "passed": violations == 0
            }
            tests_performed.append(test_result)
            
            if violations > 0:
                deduction = min(15, violation_percentage)
                score -= deduction
                issues.append({
                    "type": "format_violation",
                    "severity": "warning",
                    "message": f"Column '{col}' has {violations} values not matching {description} pattern ({violation_percentage:.1f}%)",
                    "column": col,
                    "pattern": pattern,
                    "violations": violations,
                    "score_impact": -deduction
                })
                
        except re.error as e:
            score -= 5
            issues.append({
                "type": "invalid_regex_pattern",
                "severity": "warning",
                "message": f"Invalid regex pattern for column '{col}': {str(e)}",
                "column": col,
                "pattern": pattern,
                "score_impact": -5
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "tests_performed": tests_performed
    }

def _calculate_test_coverage_score(uniqueness_result: dict, range_result: dict, format_result: dict, config: dict) -> Dict[str, Any]:
    """Calculate overall test coverage score."""
    weights = {
        'uniqueness_weight': config.get('uniqueness_weight', 0.4),
        'range_weight': config.get('range_weight', 0.3),
        'format_weight': config.get('format_weight', 0.3)
    }
    
    overall_score = (
        uniqueness_result['score'] * weights['uniqueness_weight'] +
        range_result['score'] * weights['range_weight'] +
        format_result['score'] * weights['format_weight']
    )
    
    # Determine test coverage status
    excellent_threshold = config.get('excellent_threshold', 90)
    good_threshold = config.get('good_threshold', 75)
    
    if overall_score >= excellent_threshold:
        status = "excellent"
        status_color = "green"
    elif overall_score >= good_threshold:
        status = "good"
        status_color = "yellow"
    else:
        status = "needs_improvement"
        status_color = "red"
    
    return {
        "overall": round(overall_score, 1),
        "uniqueness": uniqueness_result['score'],
        "range": range_result['score'],
        "format": format_result['score'],
        "status": status,
        "status_color": status_color,
        "weights_used": weights
    }

def _generate_routing_info(test_score: dict, total_issues: int, critical_issues: int) -> Dict[str, Any]:
    """Generate routing information based on test coverage results."""
    overall_score = test_score.get('overall', 0)
    test_status = test_score.get('status', 'unknown')
    
    if test_status == 'excellent':
        return {
            "status": "Excellent Coverage",
            "reason": "Dataset has excellent test coverage with minimal issues.",
            "suggestion": "Test coverage is comprehensive. Data is ready for production use.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    elif test_status == 'good':
        return {
            "status": "Good Coverage",
            "reason": f"Dataset has good test coverage but could be improved. Score: {overall_score}/100.",
            "suggestion": "Consider adding more comprehensive tests or fixing minor test failures.",
            "suggested_agent_endpoint": "/run-tool/cleaner"
        }
    else:  # needs_improvement
       
        return {
            "status": "Poor Coverage",
            "reason": f"Dataset has {critical_issues} critical test failures that must be addressed. Score: {overall_score}/100.",
            "suggestion": "Address critical test failures and improve test coverage before using this data.",
            "suggested_agent_endpoint": "/run-tool/cleaner"
        }


def _detect_test_coverage_row_issues(df: pd.DataFrame, config: dict) -> List[Dict[str, Any]]:
    """Detect row-level test coverage issues in the DataFrame."""
    issues = []
    
    # Check for rows with missing values in critical columns
    critical_columns = config.get('critical_columns', [])
    for col in critical_columns:
        if col in df.columns:
            missing_rows = df[df[col].isnull()].index.tolist()
            for row_idx in missing_rows:
                issues.append({
                    "row_index": int(row_idx),
                    "column": col,
                    "issue_type": "missing_critical_value",
                    "description": f"Missing value in critical column '{col}'",
                    "severity": "high"
                })
    
    # Check for duplicate rows if uniqueness is required
    if config.get('check_duplicates', True):
        duplicates = df[df.duplicated(keep=False)]
        for idx in duplicates.index:
            issues.append({
                "row_index": int(idx),
                "column": "all",
                "issue_type": "duplicate_row",
                "description": "Duplicate row detected",
                "severity": "medium"
            })
    
    return issues

def _profile_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Profile a DataFrame for test coverage."""
    if df.empty:
        return {
            "status": "success",
            "metadata": {
                "total_rows": 0,
                "total_issues": 0
            },
            "routing": {
                "status": "No Data",
                "reason": "Dataset is empty, cannot perform test coverage analysis.",
                "suggestion": "Provide a dataset with data to analyze.",
                "suggested_agent_endpoint": "/run-tool/profile-my-data"
            },
            "data": {
                "test_coverage_score": {"overall": 0, "uniqueness": 0, "range": 0, "format": 0, "status": "needs_improvement"},
                "summary": "Dataset is empty, cannot perform test coverage analysis.",
                "row_level_issues": []
            },
            "alerts": [{"level": "critical", "message": "Dataset is empty"}],
            "issue_summary": {"total_issues": 0, "by_type": {}, "by_severity": {}, "by_column": {}}
        }
    
    # Perform test coverage checks
    uniqueness_result = _test_uniqueness(df, config)
    range_result = _test_ranges(df, config)
    format_result = _test_formats(df, config)
    
    # Detect row-level test coverage issues
    row_level_issues = _detect_test_coverage_row_issues(df, config)
    
    # Calculate overall test coverage score
    test_coverage_score = _calculate_test_coverage_score(uniqueness_result, range_result, format_result, config)
    
    # Compile all issues into alerts
    alerts = []
    all_issues = uniqueness_result['issues'] + range_result['issues'] + format_result['issues']
    
    for issue in all_issues:
        alert_level = "critical" if issue['severity'] == "critical" else "warning"
        alerts.append({
            "level": alert_level,
            "message": issue['message'],
            "type": issue['type'],
            "details": {k: v for k, v in issue.items() if k not in ['message', 'severity', 'type']}
        })
    
    # Generate summary
    total_issues = len(all_issues)
    critical_issues = len([i for i in all_issues if i['severity'] == 'critical'])
    
    total_tests = len(uniqueness_result['tests_performed']) + len(range_result['tests_performed']) + len(format_result['tests_performed'])
    passed_tests = sum(1 for test in uniqueness_result['tests_performed'] + range_result['tests_performed'] + format_result['tests_performed'] if test['passed'])
    
    summary = f"Test coverage analysis completed. Overall score: {test_coverage_score['overall']}/100 ({test_coverage_score['status']}). "
    summary += f"Executed {total_tests} tests, {passed_tests} passed, {total_issues} issues found ({critical_issues} critical)."
    
    # Generate routing information
    routing_info = _generate_routing_info(test_coverage_score, total_issues, critical_issues)
    
    return {
        "status": "success",
        "metadata": {
            "total_rows": len(df),
            "total_issues": len(row_level_issues)
        },
        "routing": routing_info,
        "data": {
            "test_coverage_score": test_coverage_score,
            "uniqueness_tests": uniqueness_result,
            "range_tests": range_result,
            "format_tests": format_result,
            "summary": summary,
            "total_records": len(df),
            "total_tests_executed": total_tests,
            "tests_passed": passed_tests,
            "fields_analyzed": list(df.columns),
            "row_level_issues": row_level_issues[:100]  # Limit to first 100 issues for performance
        },
        "alerts": alerts,
        "issue_summary": _summarize_test_issues(row_level_issues)
    }

def _summarize_test_issues(row_level_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize test coverage issues for reporting."""
    if not row_level_issues:
        return {
            "total_issues": 0,
            "critical_issues": 0,
            "issue_types": {},
            "severity_breakdown": {}
        }
    
    issue_types = {}
    severity_breakdown = {"high": 0, "medium": 0, "low": 0}
    critical_issues = 0
    
    for issue in row_level_issues:
        issue_type = issue.get('issue_type', 'unknown')
        severity = issue.get('severity', 'low')
        
        # Count issue types
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Count severity levels
        if severity in severity_breakdown:
            severity_breakdown[severity] += 1
        
        # Count critical issues (high severity)
        if severity == 'high':
            critical_issues += 1
    
    return {
        "total_issues": len(row_level_issues),
        "critical_issues": critical_issues,
        "issue_types": issue_types,
        "severity_breakdown": severity_breakdown
    }

def _generate_excel_export(response: dict) -> dict:
    """
    Generate Excel export blob with complete JSON response.
    
    Args:
        response: Complete JSON response containing source_file, agent, audit, results, and summary
        
    Returns:
        dict: Excel export metadata and base64-encoded blob
    """
    import json
    
    # Extract components from response
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "TestCoverageAgent")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        # Create Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Response Overview
            overview_data = {
                "Field": ["Source File", "Agent", "Summary"],
                "Value": [
                    filename,
                    agent_name,
                    summary[:500] + "..." if len(summary) > 500 else summary  # Truncate long summaries
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name="Response Overview", index=False)
            
            # Sheet 2: Test Coverage Summary
            if results:
                summary_data = []
                for sheet_name, result in results.items():
                    if 'data' in result and 'test_coverage_score' in result['data']:
                        score_data = result['data']['test_coverage_score']
                        summary_data.append({
                            'Sheet': sheet_name,
                            'Overall_Score': score_data.get('overall', 0),
                            'Uniqueness_Score': score_data.get('uniqueness', 0),
                            'Range_Score': score_data.get('range', 0),
                            'Format_Score': score_data.get('format', 0),
                            'Status': score_data.get('status', 'unknown'),
                            'Total_Records': result['data'].get('total_records', 0),
                            'Tests_Executed': result['data'].get('total_tests_executed', 0),
                            'Tests_Passed': result['data'].get('tests_passed', 0),
                            'Total_Issues': len(result.get('alerts', []))
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name="Test Coverage Summary", index=False)
            
            # Sheet 3: Audit Summary
            if audit_trail:
                audit_summary_data = {
                    "Metric": [
                        "Agent Name",
                        "Agent Version", 
                        "Timestamp",
                        "Compute Time (seconds)",
                        "Total Sheets Analyzed",
                        "Total Fields Scanned",
                        "Total Alerts Generated",
                        "Critical Alerts",
                        "Warning Alerts",
                        "Info Alerts"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", 0),
                        audit_trail.get("scores", {}).get("total_sheets_analyzed", 0),
                        len(audit_trail.get("fields_scanned", [])),
                        audit_trail.get("scores", {}).get("total_alerts_generated", 0),
                        audit_trail.get("scores", {}).get("critical_alerts", 0),
                        audit_trail.get("scores", {}).get("warning_alerts", 0),
                        audit_trail.get("scores", {}).get("info_alerts", 0)
                    ]
                }
                audit_summary_df = pd.DataFrame(audit_summary_data)
                audit_summary_df.to_excel(writer, sheet_name="Audit Summary", index=False)
            
            # Sheet 4: Fields Scanned
            if audit_trail.get("fields_scanned"):
                fields_df = pd.DataFrame({
                    "Field Name": audit_trail["fields_scanned"]
                })
                fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 5: Findings
            if audit_trail.get("findings"):
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 6: Actions
            if audit_trail.get("actions"):
                actions_df = pd.DataFrame({
                    "Action": audit_trail["actions"]
                })
                actions_df.to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 7: Configuration/Overrides
            if audit_trail.get("overrides"):
                config_data = {
                    "Parameter": list(audit_trail["overrides"].keys()),
                    "Value": list(audit_trail["overrides"].values())
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 8: Test Results Details
            test_details_data = []
            for sheet_name, sheet_results in results.items():
                if "data" in sheet_results:
                    data = sheet_results["data"]
                    
                    # Uniqueness tests
                    for test in data.get("uniqueness_tests", {}).get("tests_performed", []):
                        test_details_data.append({
                            "Sheet": sheet_name,
                            "Test_Type": "Uniqueness",
                            "Column": test.get("column", ""),
                            "Passed": test.get("passed", False),
                            "Total_Values": test.get("total_values", 0),
                            "Issues_Found": test.get("duplicate_count", 0),
                            "Issue_Percentage": test.get("duplicate_percentage", 0),
                            "Details": f"Duplicates: {test.get('duplicate_count', 0)}"
                        })
                    
                    # Range tests
                    for test in data.get("range_tests", {}).get("tests_performed", []):
                        test_details_data.append({
                            "Sheet": sheet_name,
                            "Test_Type": "Range",
                            "Column": test.get("column", ""),
                            "Passed": test.get("passed", False),
                            "Total_Values": test.get("total_values", 0),
                            "Issues_Found": test.get("violations", 0),
                            "Issue_Percentage": test.get("violation_percentage", 0),
                            "Details": f"Range: [{test.get('min_constraint', 'N/A')}, {test.get('max_constraint', 'N/A')}]"
                        })
                    
                    # Format tests
                    for test in data.get("format_tests", {}).get("tests_performed", []):
                        test_details_data.append({
                            "Sheet": sheet_name,
                            "Test_Type": "Format",
                            "Column": test.get("column", ""),
                            "Passed": test.get("passed", False),
                            "Total_Values": test.get("total_values", 0),
                            "Issues_Found": test.get("violations", 0),
                            "Issue_Percentage": test.get("violation_percentage", 0),
                            "Details": f"Pattern: {test.get('description', test.get('pattern', 'N/A'))}"
                        })
            
            if test_details_data:
                test_details_df = pd.DataFrame(test_details_data)
                test_details_df.to_excel(writer, sheet_name="Test Details", index=False)
            
            # Sheet: Complete JSON Response (for reference)
            json_data = {
                "Component": ["Complete JSON Response"],
                "JSON Data": [json.dumps(response, indent=2, default=str)]
            }
            json_df = pd.DataFrame(json_data)
            json_df.to_excel(writer, sheet_name="Raw JSON", index=False)
        
        # Get the Excel file as bytes
        output.seek(0)
        excel_bytes = output.read()
        
        # Encode to base64
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_test_coverage_report.xlsx",
            "format": "xlsx",
            "base64_data": excel_base64,
            "size_bytes": len(excel_bytes),
            "download_ready": True
        }
        
    except Exception as e:
        # If Excel generation fails, return error info
        warnings.warn(f"Excel export failed: {str(e)}")
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_test_coverage_report.xlsx",
            "format": "xlsx",
            "base64_data": "",
            "size_bytes": 0,
            "download_ready": False,
            "error": str(e)
        }

def check_test_coverage(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """Main function for the TestCoverageAgent."""
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    # Config should always be provided by routes.py from config.json
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not provided. This should be loaded from config.json by the route handler.")
    
    # Track audit trail data
    all_fields_scanned = []
    all_findings = []
    total_rows_analyzed = 0
    total_alerts_generated = 0

    try:
        # File handling logic
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_fields_scanned.extend(list(df.columns))
            total_rows_analyzed += len(df)
            sheet_result = _profile_dataframe(df, config)
            results[sheet_name] = sheet_result
            total_alerts_generated += len(sheet_result.get('alerts', []))
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                all_fields_scanned.extend(list(df.columns))
                total_rows_analyzed += len(df)
                sheet_result = _profile_dataframe(df, config)
                results[sheet_name] = sheet_result
                total_alerts_generated += len(sheet_result.get('alerts', []))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Extract findings from results for audit trail
    all_findings = []
    for sheet_name, result in results.items():
        for alert in result.get('alerts', []):
            finding = {
                "severity": alert.get("level", "info"),
                "sheet": sheet_name,
                "issue": alert.get("message", ""),
                "category": "test_coverage",
                "type": alert.get("type", "unknown")
            }
            
            # Add score context if available
            if 'data' in result and 'test_coverage_score' in result['data']:
                scores = result['data']['test_coverage_score']
                finding["overall_score"] = scores.get("overall", 0)
                finding["uniqueness_score"] = scores.get("uniqueness", 0)
                finding["range_score"] = scores.get("range", 0)
                finding["format_score"] = scores.get("format", 0)
                finding["test_status"] = scores.get("status", "unknown")
            
            all_findings.append(finding)
    
    # Populate overrides with actual configuration values being used
    effective_overrides = {
        "uniqueness_weight": config.get("uniqueness_weight"),
        "range_weight": config.get("range_weight"),
        "format_weight": config.get("format_weight"),
        "excellent_threshold": config.get("excellent_threshold"),
        "good_threshold": config.get("good_threshold")
    }
    
    # Build comprehensive audit trail
    audit_trail = {
        "agent_name": "TestCoverageAgent",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),  # Keep for backward compatibility
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),
        "findings": all_findings,
        "actions": [
            f"Analyzed {total_rows_analyzed} rows across {len(results)} sheet(s)",
            f"Generated {total_alerts_generated} alert(s)",
            "Executed uniqueness tests",
            "Executed range validation tests",
            "Executed format validation tests",
            "Computed overall test coverage score"
        ],
        "scores": {
            "total_sheets_analyzed": len(results),
            "total_rows_analyzed": total_rows_analyzed,
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info')
        },
        "overrides": effective_overrides,
        "lineage": {}
    }
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("TestCoverageAgent", results, audit_trail)

    # Build final response
    response = {
        "source_file": filename,
        "agent": "TestCoverageAgent",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob
    
    return response