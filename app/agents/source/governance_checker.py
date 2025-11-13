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

# Import the central route registry
from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

# Define the current version of the agent
AGENT_VERSION = "1.0.0"

def _validate_lineage(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """
    Validates data lineage requirements.
    """
    issues = []
    score = 100
    
    # Check for required lineage columns
    required_lineage_fields = config.get('required_lineage_fields', ['source_system', 'created_date', 'modified_date'])
    missing_fields = [field for field in required_lineage_fields if field not in df.columns]
    
    if missing_fields:
        deduction = len(missing_fields) * 20  # 20 points per missing field
        score -= deduction
        issues.append({
            "type": "missing_lineage_fields",
            "severity": "critical",
            "message": f"Missing required lineage fields: {', '.join(missing_fields)}",
            "fields": missing_fields,
            "score_impact": -deduction
        })
    
    # Check for empty lineage values
    for field in required_lineage_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                if null_percentage > config.get('lineage_null_threshold', 5):
                    deduction = min(15, null_percentage)  # Cap at 15 points
                    score -= deduction
                    issues.append({
                        "type": "incomplete_lineage_data",
                        "severity": "warning",
                        "message": f"Field '{field}' has {null_percentage:.1f}% null values",
                        "field": field,
                        "null_percentage": null_percentage,
                        "score_impact": -deduction
                    })
    
    # Validate source system format
    if 'source_system' in df.columns:
        invalid_sources = df[~df['source_system'].str.match(r'^[A-Z_]+$', na=False)]['source_system'].dropna().unique()
        if len(invalid_sources) > 0:
            score -= 10
            issues.append({
                "type": "invalid_source_format",
                "severity": "warning",
                "message": f"Invalid source system format: {', '.join(invalid_sources[:5])}",
                "invalid_sources": invalid_sources.tolist(),
                "score_impact": -10
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "fields_validated": [f for f in required_lineage_fields if f in df.columns]
    }

def _validate_consent(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """
    Validates consent and privacy requirements.
    """
    issues = []
    score = 100
    
    # Check for required consent fields
    required_consent_fields = config.get('required_consent_fields', ['consent_status', 'consent_date', 'data_subject_id'])
    missing_fields = [field for field in required_consent_fields if field not in df.columns]
    
    if missing_fields:
        deduction = len(missing_fields) * 25  # 25 points per missing field
        score -= deduction
        issues.append({
            "type": "missing_consent_fields",
            "severity": "critical",
            "message": f"Missing required consent fields: {', '.join(missing_fields)}",
            "fields": missing_fields,
            "score_impact": -deduction
        })
    
    # Validate consent status values
    if 'consent_status' in df.columns:
        valid_statuses = config.get('valid_consent_statuses', ['granted', 'denied', 'withdrawn', 'pending'])
        invalid_statuses = df[~df['consent_status'].isin(valid_statuses + [None])]['consent_status'].dropna().unique()
        
        if len(invalid_statuses) > 0:
            score -= 15
            issues.append({
                "type": "invalid_consent_status",
                "severity": "critical",
                "message": f"Invalid consent status values: {', '.join(invalid_statuses[:5])}",
                "invalid_statuses": invalid_statuses.tolist(),
                "valid_statuses": valid_statuses,
                "score_impact": -15
            })
        
        # Check for withdrawn consent without proper handling
        withdrawn_count = (df['consent_status'] == 'withdrawn').sum()
        if withdrawn_count > 0:
            issues.append({
                "type": "withdrawn_consent_detected",
                "severity": "warning",
                "message": f"Found {withdrawn_count} records with withdrawn consent - ensure proper data handling",
                "withdrawn_count": withdrawn_count,
                "score_impact": 0
            })
    
    # Check consent date validity
    if 'consent_date' in df.columns:
        try:
            consent_dates = pd.to_datetime(df['consent_date'], errors='coerce')
            invalid_dates = consent_dates.isnull().sum() - df['consent_date'].isnull().sum()
            if invalid_dates > 0:
                score -= 10
                issues.append({
                    "type": "invalid_consent_dates",
                    "severity": "warning",
                    "message": f"Found {invalid_dates} invalid consent date formats",
                    "invalid_count": invalid_dates,
                    "score_impact": -10
                })
        except Exception:
            score -= 10
            issues.append({
                "type": "consent_date_validation_error",
                "severity": "warning",
                "message": "Unable to validate consent dates",
                "score_impact": -10
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "fields_validated": [f for f in required_consent_fields if f in df.columns]
    }

def _validate_classification_tags(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """
    Validates data classification and tagging requirements.
    """
    issues = []
    score = 100
    
    # Check for required classification fields
    required_classification_fields = config.get('required_classification_fields', ['data_classification', 'sensitivity_level', 'retention_period'])
    missing_fields = [field for field in required_classification_fields if field not in df.columns]
    
    if missing_fields:
        deduction = len(missing_fields) * 20  # 20 points per missing field
        score -= deduction
        issues.append({
            "type": "missing_classification_fields",
            "severity": "critical",
            "message": f"Missing required classification fields: {', '.join(missing_fields)}",
            "fields": missing_fields,
            "score_impact": -deduction
        })
    
    # Validate data classification values
    if 'data_classification' in df.columns:
        valid_classifications = config.get('valid_data_classifications', ['public', 'internal', 'confidential', 'restricted'])
        invalid_classifications = df[~df['data_classification'].isin(valid_classifications + [None])]['data_classification'].dropna().unique()
        
        if len(invalid_classifications) > 0:
            score -= 15
            issues.append({
                "type": "invalid_data_classification",
                "severity": "critical",
                "message": f"Invalid data classification values: {', '.join(invalid_classifications[:5])}",
                "invalid_classifications": invalid_classifications.tolist(),
                "valid_classifications": valid_classifications,
                "score_impact": -15
            })
    
    # Validate sensitivity levels
    if 'sensitivity_level' in df.columns:
        valid_levels = config.get('valid_sensitivity_levels', ['low', 'medium', 'high', 'critical'])
        invalid_levels = df[~df['sensitivity_level'].isin(valid_levels + [None])]['sensitivity_level'].dropna().unique()
        
        if len(invalid_levels) > 0:
            score -= 10
            issues.append({
                "type": "invalid_sensitivity_level",
                "severity": "warning",
                "message": f"Invalid sensitivity level values: {', '.join(invalid_levels[:5])}",
                "invalid_levels": invalid_levels.tolist(),
                "valid_levels": valid_levels,
                "score_impact": -10
            })
    
    # Check for PII detection consistency
    pii_patterns = config.get('pii_patterns', {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
    })
    
    pii_detected = []
    for col in df.select_dtypes(include=['object']).columns:
        col_data = df[col].astype(str)
        for pii_type, pattern in pii_patterns.items():
            if col_data.str.contains(pattern, regex=True, na=False).any():
                pii_detected.append({"column": col, "pii_type": pii_type})
    
    if pii_detected and 'data_classification' in df.columns:
        # Check if PII data has appropriate classification
        public_classified_pii = []
        for pii_info in pii_detected:
            col = pii_info['column']
            if (df['data_classification'] == 'public').any():
                public_classified_pii.append(pii_info)
        
        if public_classified_pii:
            score -= 20
            issues.append({
                "type": "pii_classification_mismatch",
                "severity": "critical",
                "message": f"PII data found in columns classified as 'public': {[p['column'] for p in public_classified_pii]}",
                "pii_columns": public_classified_pii,
                "score_impact": -20
            })
    
    return {
        "score": max(0, score),
        "issues": issues,
        "pii_detected": pii_detected,
        "fields_validated": [f for f in required_classification_fields if f in df.columns]
    }

def _calculate_governance_score(lineage_result: dict, consent_result: dict, classification_result: dict, config: dict) -> Dict[str, Any]:
    """
    Calculates overall governance score based on weighted components.
    """
    weights = {
        'lineage_weight': config.get('lineage_weight', 0.3),
        'consent_weight': config.get('consent_weight', 0.4),
        'classification_weight': config.get('classification_weight', 0.3)
    }
    
    overall_score = (
        lineage_result['score'] * weights['lineage_weight'] +
        consent_result['score'] * weights['consent_weight'] +
        classification_result['score'] * weights['classification_weight']
    )
    
    # Determine compliance status
    compliance_threshold = config.get('compliance_threshold', 80)
    needs_review_threshold = config.get('needs_review_threshold', 60)
    
    if overall_score >= compliance_threshold:
        status = "compliant"
        status_color = "green"
    elif overall_score >= needs_review_threshold:
        status = "needs_review"
        status_color = "yellow"
    else:
        status = "non_compliant"
        status_color = "red"
    
    return {
        "overall": round(overall_score, 1),
        "lineage": lineage_result['score'],
        "consent": consent_result['score'],
        "classification": classification_result['score'],
        "status": status,
        "status_color": status_color,
        "weights_used": weights
    }

def _generate_routing_info(governance_score: dict, total_issues: int, critical_issues: int) -> Dict[str, Any]:
    """
    Generates routing information based on governance compliance score and issues.
    """
    overall_score = governance_score.get('overall', 0)
    compliance_status = governance_score.get('status', 'unknown')
    
    if compliance_status == 'compliant':
        return {
            "status": "Compliant",
            "reason": "Dataset meets governance compliance requirements.",
            "suggestion": "Data is ready for use. Continue with your data workflow.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    elif compliance_status == 'needs_review':
        return {
            "status": "Needs Review",
            "reason": f"Dataset has governance compliance issues that should be reviewed. Score: {overall_score}/100.",
            "suggestion": "Review governance issues and update data classification, consent, or lineage information.",
            "suggested_agent_endpoint": "/run-tool/governance"
        }
    else:  # non_compliant
        if critical_issues > 0:
            return {
                "status": "Non-Compliant",
                "reason": f"Dataset has {critical_issues} critical governance issues that must be addressed. Score: {overall_score}/100.",
                "suggestion": "Address critical governance compliance issues before using this data.",
                "suggested_agent_endpoint": "/run-tool/governance"
            }
        else:
            return {
                "status": "Non-Compliant",
                "reason": f"Dataset does not meet governance compliance requirements. Score: {overall_score}/100.",
                "suggestion": "Review and fix governance compliance issues before proceeding.",
                "suggested_agent_endpoint": "/run-tool/governance"
            }

def _profile_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """
    Profiles a single DataFrame for governance compliance.
    """
    if df.empty:
        return {
            "data": {
                "governance_score": {"overall": 0, "lineage": 0, "consent": 0, "classification": 0, "status": "non_compliant"},
                "summary": "Dataset is empty, cannot perform governance validation."
            },
            "alerts": [{"level": "critical", "message": "Dataset is empty"}]
        }
    
    # Perform validation checks
    lineage_result = _validate_lineage(df, config)
    consent_result = _validate_consent(df, config)
    classification_result = _validate_classification_tags(df, config)
    
    # Calculate overall governance score
    governance_score = _calculate_governance_score(lineage_result, consent_result, classification_result, config)
    
    # Compile all issues into alerts
    alerts = []
    all_issues = lineage_result['issues'] + consent_result['issues'] + classification_result['issues']
    
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
    
    summary = f"Governance validation completed. Overall score: {governance_score['overall']}/100 ({governance_score['status']}). "
    summary += f"Found {total_issues} issues ({critical_issues} critical)."
    
    # Generate routing information based on governance score
    routing_info = _generate_routing_info(governance_score, total_issues, critical_issues)
    
    return {
        "status": "success",
        "metadata": {
            "total_rows_analyzed": len(df),
            "total_issues": total_issues
        },
        "routing": routing_info,
        "data": {
            "governance_score": governance_score,
            "lineage_validation": lineage_result,
            "consent_validation": consent_result,
            "classification_validation": classification_result,
            "summary": summary,
            "total_records": len(df),
            "fields_analyzed": list(df.columns)
        },
        "alerts": alerts
    }

def _generate_excel_export(results: dict, filename: str, audit_trail: dict, config: dict) -> str:
    """
    Generates Excel export with governance validation results.
    """
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for sheet_name, result in results.items():
                if 'data' in result and 'governance_score' in result['data']:
                    score_data = result['data']['governance_score']
                    summary_data.append({
                        'Sheet': sheet_name,
                        'Overall_Score': score_data.get('overall', 0),
                        'Lineage_Score': score_data.get('lineage', 0),
                        'Consent_Score': score_data.get('consent', 0),
                        'Classification_Score': score_data.get('classification', 0),
                        'Status': score_data.get('status', 'unknown'),
                        'Total_Records': result['data'].get('total_records', 0),
                        'Total_Issues': len(result.get('alerts', []))
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Governance_Summary', index=False)
            
            # Issues detail sheet
            issues_data = []
            for sheet_name, result in results.items():
                for alert in result.get('alerts', []):
                    issues_data.append({
                        'Sheet': sheet_name,
                        'Severity': alert.get('level', 'info'),
                        'Type': alert.get('type', 'unknown'),
                        'Message': alert.get('message', ''),
                        'Details': str(alert.get('details', {}))
                    })
            
            if issues_data:
                issues_df = pd.DataFrame(issues_data)
                issues_df.to_excel(writer, sheet_name='Issues_Detail', index=False)
            
            # Configuration sheet
            config_data = [{'Parameter': k, 'Value': str(v)} for k, v in config.items()]
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        output.seek(0)
        return base64.b64encode(output.read()).decode('utf-8')
    
    except Exception as e:
        warnings.warn(f"Excel export failed: {str(e)}")
        return ""

def _extract_findings_from_result(sheet_result: dict, sheet_name: str) -> List[dict]:
    """
    Extract structured findings from sheet result for audit trail.
    """
    findings = []
    
    # Extract from alerts
    for alert in sheet_result.get('alerts', []):
        finding = {
            "severity": alert.get("level", "info"),
            "sheet": sheet_name,
            "issue": alert.get("message", ""),
            "category": "governance_validation",
            "type": alert.get("type", "unknown")
        }
        
        # Add score context if available
        if 'data' in sheet_result and 'governance_score' in sheet_result['data']:
            scores = sheet_result['data']['governance_score']
            finding["governance_score"] = scores.get("overall", 0)
            finding["lineage_score"] = scores.get("lineage", 0)
            finding["consent_score"] = scores.get("consent", 0)
            finding["classification_score"] = scores.get("classification", 0)
            finding["compliance_status"] = scores.get("status", "unknown")
        
        findings.append(finding)
    
    return findings

def _calculate_aggregate_scores(results: dict) -> dict:
    """
    Calculate aggregate scores across all sheets.
    """
    if not results:
        return {}
    
    scores = {
        'overall': [],
        'lineage': [],
        'consent': [],
        'classification': []
    }
    
    for result in results.values():
        if 'data' in result and 'governance_score' in result['data']:
            score_data = result['data']['governance_score']
            scores['overall'].append(score_data.get('overall', 0))
            scores['lineage'].append(score_data.get('lineage', 0))
            scores['consent'].append(score_data.get('consent', 0))
            scores['classification'].append(score_data.get('classification', 0))
    
    return {
        'average_overall': round(np.mean(scores['overall']) if scores['overall'] else 0, 1),
        'average_lineage': round(np.mean(scores['lineage']) if scores['lineage'] else 0, 1),
        'average_consent': round(np.mean(scores['consent']) if scores['consent'] else 0, 1),
        'average_classification': round(np.mean(scores['classification']) if scores['classification'] else 0, 1)
    }

def check_governance(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """
    Main function for the GovernanceChecker agent.
    
    Args:
        file_contents: Raw bytes of the uploaded file
        filename: Name of the uploaded file
        config: Configuration dictionary containing validation parameters (optional)
        user_overrides: User-provided parameter overrides for audit trail (optional)
    
    Returns:
        dict: Standardized JSON response with governance validation results, audit trail, and Excel export
    """
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    # Use default config if not provided
    if config is None:
        config = {
            'required_lineage_fields': ['source_system', 'created_date', 'modified_date'],
            'required_consent_fields': ['consent_status', 'consent_date', 'data_subject_id'],
            'required_classification_fields': ['data_classification', 'sensitivity_level', 'retention_period'],
            'valid_consent_statuses': ['granted', 'denied', 'withdrawn', 'pending'],
            'valid_data_classifications': ['public', 'internal', 'confidential', 'restricted'],
            'valid_sensitivity_levels': ['low', 'medium', 'high', 'critical'],
            'lineage_weight': 0.3,
            'consent_weight': 0.4,
            'classification_weight': 0.3,
            'compliance_threshold': 80,
            'needs_review_threshold': 60,
            'lineage_null_threshold': 5,
            'pii_patterns': {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
            }
        }
    
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
            
            # Extract findings
            findings = _extract_findings_from_result(sheet_result, sheet_name)
            all_findings.extend(findings)
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                all_fields_scanned.extend(list(df.columns))
                total_rows_analyzed += len(df)
                sheet_result = _profile_dataframe(df, config)
                results[sheet_name] = sheet_result
                total_alerts_generated += len(sheet_result.get('alerts', []))
                
                # Extract findings
                findings = _extract_findings_from_result(sheet_result, sheet_name)
                all_findings.extend(findings)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Calculate aggregate scores
    aggregate_scores = _calculate_aggregate_scores(results)
    
    # Build comprehensive audit trail
    audit_trail = {
        "agent_name": "GovernanceChecker",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),  # Keep for backward compatibility
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),  # Unique fields
        "findings": all_findings,
        "actions": [
            f"Analyzed {total_rows_analyzed} rows across {len(results)} sheet(s)",
            f"Generated {total_alerts_generated} alert(s)",
            "Validated data lineage requirements",
            "Validated consent and privacy compliance",
            "Validated data classification and tagging",
            "Computed overall governance compliance score"
        ],
        "scores": {
            "total_sheets_analyzed": len(results),
            "total_rows_analyzed": total_rows_analyzed,
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info'),
            "average_governance_score": aggregate_scores.get('average_overall', 0),
            "average_lineage_score": aggregate_scores.get('average_lineage', 0),
            "average_consent_score": aggregate_scores.get('average_consent', 0),
            "average_classification_score": aggregate_scores.get('average_classification', 0)
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Generate Excel export blob
    excel_blob = _generate_excel_export(results, filename, audit_trail, config)
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("GovernanceChecker", results, audit_trail)

    return {
        "source_file": filename,
        "agent": "GovernanceChecker",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "excel_export": excel_blob
    }