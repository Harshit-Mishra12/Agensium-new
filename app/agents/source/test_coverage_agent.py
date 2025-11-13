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
        if critical_issues > 0:
            return {
                "status": "Poor Coverage",
                "reason": f"Dataset has {critical_issues} critical test failures that must be addressed. Score: {overall_score}/100.",
                "suggestion": "Address critical test failures and improve test coverage before using this data.",
                "suggested_agent_endpoint": "/run-tool/cleaner"
            }
        else:
            return {
                "status": "Needs Improvement",
                "reason": f"Dataset test coverage needs improvement. Score: {overall_score}/100.",
                "suggestion": "Add more comprehensive tests and fix existing test failures.",
                "suggested_agent_endpoint": "/run-tool/cleaner"
            }

def _profile_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Profile a DataFrame for test coverage."""
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows_analyzed": 0, "total_issues": 1},
            "routing": {
                "status": "No Data",
                "reason": "Dataset is empty, cannot perform test coverage analysis.",
                "suggestion": "Provide a dataset with data to analyze.",
                "suggested_agent_endpoint": "/run-tool/profile-my-data"
            },
            "data": {
                "test_coverage_score": {"overall": 0, "uniqueness": 0, "range": 0, "format": 0, "status": "needs_improvement"},
                "summary": "Dataset is empty, cannot perform test coverage analysis."
            },
            "alerts": [{"level": "critical", "message": "Dataset is empty"}]
        }
    
    # Perform test coverage checks
    uniqueness_result = _test_uniqueness(df, config)
    range_result = _test_ranges(df, config)
    format_result = _test_formats(df, config)
    
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
            "total_rows_analyzed": len(df),
            "total_issues": total_issues
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
            "fields_analyzed": list(df.columns)
        },
        "alerts": alerts
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
    
    # Build audit trail
    audit_trail = {
        "agent_name": "TestCoverageAgent",
        "timestamp": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),
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
            "total_alerts_generated": total_alerts_generated
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("TestCoverageAgent", results, audit_trail)

    return {
        "source_file": filename,
        "agent": "TestCoverageAgent",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "excel_export": ""
    }