import pandas as pd
import numpy as np
import io
import time
import base64
import re
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

# PII Detection Patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'phone': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'passport': r'\b[A-Z]{1,2}\d{6,9}\b'
}

# Sensitive field name patterns
SENSITIVE_FIELD_NAMES = [
    'ssn', 'social_security', 'tax_id', 'passport', 'license', 'credit_card',
    'password', 'secret', 'token', 'api_key', 'salary', 'income', 'wage',
    'dob', 'birth_date', 'birthdate', 'medical', 'health', 'diagnosis',
    'religion', 'ethnicity', 'race', 'political', 'sexual_orientation'
]

# Compliance-related field names
COMPLIANCE_FIELDS = [
    'gdpr', 'consent', 'opt_in', 'opt_out', 'privacy', 'data_retention',
    'deletion_date', 'anonymized', 'encrypted', 'classification'
]


def score_risk(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """
    Main function for the RiskScorer agent.
    Flags PII, sensitive data, governance gaps, and compliance risks.
    """
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    if config is None:
        config = {
            'pii_sample_size': 100,
            'high_risk_threshold': 75,
            'medium_risk_threshold': 50,
            'pii_detection_enabled': True,
            'sensitive_field_detection_enabled': True,
            'governance_check_enabled': True
        }
    
    all_fields_scanned = []
    all_findings = []
    total_rows_analyzed = 0
    total_risks_detected = 0
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_fields_scanned.extend(list(df.columns))
            total_rows_analyzed += len(df)
            sheet_result = _assess_sheet_risk(df, sheet_name, config)
            results[sheet_name] = sheet_result
            total_risks_detected += len(sheet_result.get('alerts', []))
            
            findings = _extract_findings_from_result(sheet_result, sheet_name)
            all_findings.extend(findings)
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                all_fields_scanned.extend(list(df.columns))
                total_rows_analyzed += len(df)
                sheet_result = _assess_sheet_risk(df, sheet_name, config)
                results[sheet_name] = sheet_result
                total_risks_detected += len(sheet_result.get('alerts', []))
                
                findings = _extract_findings_from_result(sheet_result, sheet_name)
                all_findings.extend(findings)
        else:
            raise HTTPException(stsatus_code=400, detail=f"Unsupported file format: {file_extension}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")
    
    compute_time = round(time.time() - start_time, 2)
    
    aggregate_scores = _calculate_aggregate_scores(results)
    
    audit_trail = {
        "agent_name": "RiskScorer",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": compute_time,
        "fields_scanned": list(set(all_fields_scanned)),
        "findings": all_findings,
        "actions": [
            f"Analyzed {total_rows_analyzed} rows across {len(results)} sheet(s)",
            f"Detected {total_risks_detected} risk(s)",
            "Scanned for PII patterns (email, SSN, phone, credit card, etc.)",
            "Checked field names for sensitive data indicators",
            "Assessed governance and compliance gaps",
            "Calculated risk scores per field and overall"
        ],
        "scores": {
            "total_sheets_analyzed": len(results),
            "total_rows_analyzed": total_rows_analyzed,
            "total_risks_detected": total_risks_detected,
            "critical_risks": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "high_risks": sum(1 for f in all_findings if f.get('severity') == 'high'),
            "medium_risks": sum(1 for f in all_findings if f.get('severity') == 'medium'),
            "low_risks": sum(1 for f in all_findings if f.get('severity') == 'low'),
            "average_risk_score": aggregate_scores.get('average_risk_score', 0),
            "pii_fields_detected": aggregate_scores.get('pii_fields_count', 0),
            "sensitive_fields_detected": aggregate_scores.get('sensitive_fields_count', 0),
            "governance_gaps": aggregate_scores.get('governance_gaps', 0)
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    excel_blob = _generate_excel_export(results, filename, audit_trail, config)
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("RiskScorer", results, audit_trail)
    
    return {
        "source_file": filename,
        "agent": "RiskScorer",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "excel_export": excel_blob
    }


def _assess_sheet_risk(df: pd.DataFrame, sheet_name: str, config: dict) -> dict:
    """Assess risk for a single sheet/dataframe."""
    if df.empty:
        return {
            "status": "error",
            "metadata": {"total_rows": 0},
            "alerts": [{
                "level": "info",
                "message": "Empty dataset - no risk assessment possible"
            }],
            "routing": {"status": "No Data", "reason": "Dataset is empty"},
            "data": {"field_risks": {}, "overall_risk_score": 0}
        }
    
    field_risks = {}
    alerts = []
    pii_fields = []
    sensitive_fields = []
    governance_gaps = []
    row_level_issues = []
    
    for col_name in df.columns:
        risk_assessment, col_issues = _assess_field_risk_with_rows(df, col_name, config)
        field_risks[col_name] = risk_assessment
        row_level_issues.extend(col_issues)
        
        if risk_assessment['pii_detected']:
            pii_fields.append(col_name)
        if risk_assessment['is_sensitive']:
            sensitive_fields.append(col_name)
        if risk_assessment['governance_issues']:
            governance_gaps.extend(risk_assessment['governance_issues'])
        
        if risk_assessment['risk_level'] in ['critical', 'high']:
            alerts.append({
                "level": risk_assessment['risk_level'],
                "field": col_name,
                "message": risk_assessment['risk_summary']
            })
    
    overall_risk_score = _calculate_overall_risk_score(field_risks)
    
    routing = _determine_routing(overall_risk_score, len(pii_fields), len(governance_gaps))
    
    return {
        "status": "success",
        "metadata": {
            "total_rows": len(df),
            "total_fields": len(df.columns),
            "pii_fields_count": len(pii_fields),
            "sensitive_fields_count": len(sensitive_fields),
            "governance_gaps_count": len(governance_gaps),
            "total_issues": len(row_level_issues)
        },
        "routing": routing,
        "data": {
            "field_risks": field_risks,
            "overall_risk_score": overall_risk_score,
            "pii_fields": pii_fields,
            "sensitive_fields": sensitive_fields,
            "governance_gaps": governance_gaps,
            "row_level_issues": row_level_issues[:100]  # Limit to first 100
        },
        "issue_summary": _summarize_risk_issues(row_level_issues)
    }


def _assess_field_risk(series: pd.Series, col_name: str, config: dict) -> dict:
    """Assess risk for a single field/column."""
    risk_factors = []
    pii_types_detected = []
    is_sensitive = False
    governance_issues = []
    risk_score = 0
    
    col_lower = col_name.lower()
    for sensitive_term in SENSITIVE_FIELD_NAMES:
        if sensitive_term in col_lower:
            is_sensitive = True
            risk_factors.append(f"Sensitive field name: '{col_name}' contains '{sensitive_term}'")
            risk_score += 20
            break
    
    if config.get('pii_detection_enabled', True):
        sample_size = min(config.get('pii_sample_size', 100), len(series))
        sample_data = series.dropna().astype(str).head(sample_size)
        
        for pii_type, pattern in PII_PATTERNS.items():
            matches = sample_data.str.contains(pattern, regex=True, na=False)
            if matches.any():
                match_count = matches.sum()
                pii_types_detected.append(pii_type)
                risk_factors.append(f"PII detected: {pii_type} ({match_count} matches in sample)")
                risk_score += 30
    
    if config.get('governance_check_enabled', True):
        if pii_types_detected or is_sensitive:
            has_consent = any(term in col_lower for term in ['consent', 'opt_in', 'opt_out'])
            if not has_consent:
                governance_issues.append(f"Field '{col_name}' contains sensitive/PII data but no consent tracking detected")
                risk_score += 15
    
    if risk_score >= config.get('high_risk_threshold', 75):
        risk_level = 'critical'
    elif risk_score >= config.get('medium_risk_threshold', 50):
        risk_level = 'high'
    elif risk_score >= 25:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    if pii_types_detected:
        risk_summary = f"PII detected: {', '.join(pii_types_detected)}"
    elif is_sensitive:
        risk_summary = f"Sensitive field detected"
    elif governance_issues:
        risk_summary = f"Governance gaps: {len(governance_issues)} issue(s)"
    else:
        risk_summary = "No significant risks detected"
    
    return {
        "field_name": col_name,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "pii_detected": len(pii_types_detected) > 0,
        "pii_types": pii_types_detected,
        "is_sensitive": is_sensitive,
        "governance_issues": governance_issues,
        "risk_factors": risk_factors
    }


def _calculate_overall_risk_score(field_risks: dict) -> int:
    """Calculate overall risk score from field risks."""
    if not field_risks:
        return 0
    total_score = sum(risk['risk_score'] for risk in field_risks.values())
    return round(total_score / len(field_risks))


def _determine_routing(risk_score: int, pii_count: int, governance_gaps_count: int) -> dict:
    """Determine routing based on risk assessment."""
    if risk_score >= 75 or pii_count > 0:
        return {
            "status": "High Risk",
            "reason": f"Critical risks detected (score: {risk_score}, PII fields: {pii_count})",
            "suggestion": "Immediate action required: Review PII handling, implement data governance, ensure compliance",
            "suggested_agent_endpoint": AGENT_ROUTES.get("govern_data_tool", "/run-tool/governance")
        }
    elif risk_score >= 50 or governance_gaps_count > 0:
        return {
            "status": "Medium Risk",
            "reason": f"Moderate risks detected (score: {risk_score}, governance gaps: {governance_gaps_count})",
            "suggestion": "Review and address governance gaps, implement data classification",
            "suggested_agent_endpoint": AGENT_ROUTES.get("govern_data_tool", "/run-tool/governance")
        }
    else:
        return {
            "status": "Low Risk",
            "reason": "No significant risks detected",
            "suggestion": "Proceed to master data tool for entity resolution and deduplication",
            "suggested_agent_endpoint": AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
        }


def _extract_findings_from_result(sheet_result: dict, sheet_name: str) -> list:
    """Extract structured findings from sheet result for audit trail."""
    findings = []
    
    for alert in sheet_result.get('alerts', []):
        finding = {
            "severity": alert.get("level", "info"),
            "sheet": sheet_name,
            "field": alert.get("field", ""),
            "issue": alert.get("message", ""),
            "category": "risk_assessment"
        }
        findings.append(finding)
    
    if 'data' in sheet_result and 'field_risks' in sheet_result['data']:
        for field_name, risk_data in sheet_result['data']['field_risks'].items():
            if risk_data['pii_detected']:
                finding = {
                    "severity": risk_data['risk_level'],
                    "sheet": sheet_name,
                    "field": field_name,
                    "issue": f"PII detected: {', '.join(risk_data['pii_types'])}",
                    "category": "pii_detection",
                    "risk_score": risk_data['risk_score']
                }
                findings.append(finding)
    
    return findings


def _calculate_aggregate_scores(results: dict) -> dict:
    """Calculate aggregate scores across all sheets."""
    if not results:
        return {"average_risk_score": 0, "pii_fields_count": 0, "sensitive_fields_count": 0, "governance_gaps": 0}
    
    total_risk = 0
    total_pii = 0
    total_sensitive = 0
    total_governance = 0
    count = 0
    
    for sheet_result in results.values():
        if sheet_result.get('status') == 'success' and 'data' in sheet_result:
            total_risk += sheet_result['data'].get('overall_risk_score', 0)
            total_pii += len(sheet_result['data'].get('pii_fields', []))
            total_sensitive += len(sheet_result['data'].get('sensitive_fields', []))
            total_governance += len(sheet_result['data'].get('governance_gaps', []))
            count += 1
    
    return {
        "average_risk_score": round(total_risk / count) if count > 0 else 0,
        "pii_fields_count": total_pii,
        "sensitive_fields_count": total_sensitive,
        "governance_gaps": total_governance
    }


def _generate_excel_export(results: dict, filename: str, audit_trail: dict, config: dict) -> dict:
    """Generate Excel export blob with risk assessment results."""
    from io import BytesIO
    
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            summary_data = {
                "Metric": [
                    "Source File", "Agent", "Version", "Timestamp", "Compute Time (seconds)",
                    "Total Sheets Analyzed", "Total Rows Analyzed", "Total Risks Detected",
                    "Critical Risks", "High Risks", "Medium Risks", "Low Risks",
                    "Average Risk Score", "PII Fields Detected", "Sensitive Fields Detected", "Governance Gaps"
                ],
                "Value": [
                    filename, audit_trail["agent_name"], audit_trail["agent_version"], audit_trail["timestamp"],
                    audit_trail["compute_time_seconds"], audit_trail["scores"]["total_sheets_analyzed"],
                    audit_trail["scores"]["total_rows_analyzed"], audit_trail["scores"]["total_risks_detected"],
                    audit_trail["scores"]["critical_risks"], audit_trail["scores"]["high_risks"],
                    audit_trail["scores"]["medium_risks"], audit_trail["scores"]["low_risks"],
                    audit_trail["scores"]["average_risk_score"], audit_trail["scores"]["pii_fields_detected"],
                    audit_trail["scores"]["sensitive_fields_detected"], audit_trail["scores"]["governance_gaps"]
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            if audit_trail["fields_scanned"]:
                pd.DataFrame({"Field Name": audit_trail["fields_scanned"]}).to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            if audit_trail["findings"]:
                pd.DataFrame(audit_trail["findings"]).to_excel(writer, sheet_name="Findings", index=False)
            
            pd.DataFrame({"Action": audit_trail["actions"]}).to_excel(writer, sheet_name="Actions", index=False)
            
            pd.DataFrame({"Parameter": list(config.keys()), "Value": list(config.values())}).to_excel(writer, sheet_name="Configuration", index=False)
            
            for sheet_name, sheet_result in results.items():
                if sheet_result.get("status") == "success" and "data" in sheet_result:
                    data = sheet_result["data"]
                    
                    if data.get("field_risks"):
                        risks_data = []
                        for field, risk in data["field_risks"].items():
                            risks_data.append({
                                "Field": field,
                                "Risk Score": risk["risk_score"],
                                "Risk Level": risk["risk_level"],
                                "PII Detected": risk["pii_detected"],
                                "PII Types": ", ".join(risk["pii_types"]) if risk["pii_types"] else "",
                                "Sensitive": risk["is_sensitive"],
                                "Risk Summary": risk["risk_summary"]
                            })
                        safe_name = sheet_name[:25]
                        pd.DataFrame(risks_data).to_excel(writer, sheet_name=f"{safe_name}_Risks", index=False)
            
            if audit_trail["overrides"]:
                pd.DataFrame({"Parameter": list(audit_trail["overrides"].keys()), "User Value": list(audit_trail["overrides"].values())}).to_excel(writer, sheet_name="User Overrides", index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        sheets_list = ["Summary", "Fields Scanned" if audit_trail["fields_scanned"] else None, "Findings" if audit_trail["findings"] else None, "Actions", "Configuration", "User Overrides" if audit_trail["overrides"] else None]
        sheets_list = [s for s in sheets_list if s]
        sheets_list.extend([f"{sheet[:25]}_Risks" for sheet in results.keys()])
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_risk_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": sheets_list,
            "download_ready": True
        }
    except Exception as e:
        return {
            "filename": None,
            "size_bytes": 0,
            "format": "xlsx",
            "base64_data": None,
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }


def _assess_field_risk_with_rows(df: pd.DataFrame, col_name: str, config: dict) -> tuple:
    """
    Assess field risk and track row-level PII/sensitive data occurrences.
    
    Args:
        df: Full DataFrame
        col_name: Column name to assess
        config: Configuration dictionary
    
    Returns:
        tuple: (risk_assessment dict, list of row-level issues)
    """
    series = df[col_name]
    issues = []
    
    # Get the regular risk assessment
    risk_assessment = _assess_field_risk(series, col_name, config)
    
    # Track PII occurrences with row indices
    if risk_assessment['pii_detected'] and config.get('pii_detection_enabled', True):
        sample_size = min(config.get('pii_sample_size', 100), len(series))
        sample_data = series.dropna().astype(str).head(sample_size)
        
        for pii_type, pattern in PII_PATTERNS.items():
            matches = sample_data.str.contains(pattern, regex=True, na=False)
            if matches.any():
                matching_indices = sample_data.index[matches].tolist()
                
                # Limit to first 10 matches per PII type
                for idx in matching_indices[:10]:
                    value = str(series.iloc[idx]) if idx < len(series) else None
                    # Mask PII value for security
                    masked_value = value[:3] + "***" if value and len(value) > 3 else "***"
                    
                    issues.append({
                        "row_index": int(idx),
                        "column": col_name,
                        "issue_type": "pii_detected",
                        "pii_type": pii_type,
                        "severity": "critical",
                        "value": masked_value,  # Masked for security
                        "message": f"PII detected ({pii_type}) in column '{col_name}'"
                    })
    
    # Track sensitive field occurrences
    if risk_assessment['is_sensitive']:
        # For sensitive fields, just note the column (not specific rows)
        issues.append({
            "row_index": None,  # Affects all rows
            "column": col_name,
            "issue_type": "sensitive_field",
            "severity": "high",
            "value": None,
            "message": f"Column '{col_name}' contains sensitive data"
        })
    
    # Track governance gaps
    if risk_assessment['governance_issues']:
        for gap in risk_assessment['governance_issues']:
            issues.append({
                "row_index": None,  # Affects all rows
                "column": col_name,
                "issue_type": "governance_gap",
                "severity": "warning",
                "value": None,
                "message": gap
            })
    
    return risk_assessment, issues


def _summarize_risk_issues(issues: list) -> dict:
    """
    Summarize risk issues by type and severity.
    
    Args:
        issues: List of row-level risk issues
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_issues": len(issues),
        "by_type": {},
        "by_severity": {},
        "by_column": {},
        "by_pii_type": {}
    }
    
    for issue in issues:
        issue_type = issue.get("issue_type", "unknown")
        summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
        
        severity = issue.get("severity", "info")
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        column = issue.get("column")
        if column:
            summary["by_column"][column] = summary["by_column"].get(column, 0) + 1
        
        # Track PII types
        pii_type = issue.get("pii_type")
        if pii_type:
            summary["by_pii_type"][pii_type] = summary["by_pii_type"].get(pii_type, 0) + 1
    
    return summary
