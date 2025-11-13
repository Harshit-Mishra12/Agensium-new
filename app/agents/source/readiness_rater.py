import pandas as pd
import numpy as np
import io
import time
import base64
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings

# Import the central route registry
from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

# Define the current version of the agent
AGENT_VERSION = "1.2.0"

def _calculate_readiness_score(df: pd.DataFrame, config: dict):
    """
    Calculates the readiness score and generates explanations for a single DataFrame.
    """
    if df.empty:
        scores = {"overall": 0, "completeness": 0, "consistency": 0, "schema_health": 0}
        deductions = ["Dataset is empty, resulting in a score of 0."]
        return scores, deductions, []

    deductions = []
    row_level_issues = []
    
    # 1. Completeness Score (based on nulls)
    total_cells = df.size
    null_cells = df.isnull().sum().sum()
    null_percentage = (null_cells / total_cells * 100) if total_cells > 0 else 0
    completeness_score = 100 - null_percentage
    if null_cells > 0:
        deductions.append(f"Completeness: Score reduced by {null_percentage:.1f} points due to {null_percentage:.1f}% null values.")
        # Track null value rows
        null_issues = _detect_null_rows(df)
        row_level_issues.extend(null_issues)

    # 2. Consistency Score (based on duplicate rows)
    duplicate_rows = df.duplicated().sum()
    total_rows = len(df)
    duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
    consistency_score = 100 - duplicate_percentage
    if duplicate_rows > 0:
        deductions.append(f"Consistency: Score reduced by {duplicate_percentage:.1f} points due to {duplicate_rows} duplicate rows ({duplicate_percentage:.1f}%).")
        # Track duplicate rows
        dup_issues = _detect_duplicate_rows_readiness(df)
        row_level_issues.extend(dup_issues)

    # 3. Schema Health Score (heuristic-based)
    schema_health_score = 100
    # Penalize for columns with very low variance
    for col in df.columns:
        if df[col].nunique() == 1 and len(df) > 1:
            schema_health_score -= 10
            deductions.append(f"Schema Health: Score reduced by 10 points because column '{col}' has only one unique value.")
            # Track constant columns
            row_level_issues.append({
                "row_index": None,
                "column": col,
                "issue_type": "constant_column",
                "severity": "info",
                "value": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else None,
                "message": f"Column '{col}' has only one unique value"
            })
    
    # Penalize for potential mixed-type object columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for col in df.select_dtypes(include=['object']).columns:
            if pd.to_numeric(df[col].dropna().iloc[:100], errors='coerce').notna().any():
                schema_health_score -= 5
                deductions.append(f"Schema Health: Score reduced by 5 points for potential mixed data types in column '{col}'.")
    
    schema_health_score = max(0, schema_health_score)

    # 4. Overall Score (weighted average using config)
    completeness_weight = config.get('completeness_weight', 0.4)
    consistency_weight = config.get('consistency_weight', 0.4)
    schema_health_weight = config.get('schema_health_weight', 0.2)
    
    overall_score = (completeness_score * completeness_weight) + \
                    (consistency_score * consistency_weight) + \
                    (schema_health_score * schema_health_weight)

    scores = {
        "overall": round(overall_score),
        "completeness": round(completeness_score),
        "consistency": round(consistency_score),
        "schema_health": round(schema_health_score)
    }
    
    return scores, deductions, row_level_issues

def _profile_dataframe(df: pd.DataFrame, config: dict):
    """
    Generates the full agentic response for a single DataFrame.
    """
    scores, deductions, row_level_issues = _calculate_readiness_score(df, config)
    overall_score = scores['overall']
    alerts = []
    
    # Threshold-based routing and alerts (using config)
    ready_threshold = config.get('ready_threshold', 85)
    needs_review_threshold = config.get('needs_review_threshold', 70)
    
    if overall_score >= ready_threshold:
        routing_status = "Ready"
        reason = "Dataset meets readiness criteria."
        suggestion = "Proceed to master data tool for entity resolution and deduplication."
        endpoint = AGENT_ROUTES.get("master_my_data_tool", "/run-tool/master-my-data")
    elif overall_score >= needs_review_threshold:
        routing_status = "Needs Review"
        reason = "Dataset has moderate quality issues that should be reviewed."
        suggestion = "Run the 'Clean My Data' tool to address issues."
        endpoint = AGENT_ROUTES['clean_data_tool']
        alerts.append({"level": "warning", "message": f"Overall readiness score is {overall_score}. Manual review is recommended."})
    else:
        routing_status = "Not Ready"
        reason = "Dataset has significant quality issues and is not ready for use."
        suggestion = "Run the 'Clean My Data' tool to fix critical issues."
        endpoint = AGENT_ROUTES['clean_data_tool']
        alerts.append({"level": "critical", "message": f"Overall readiness score is {overall_score}. Data is not recommended for production use without cleaning."})

    routing = {
        "status": routing_status,
        "reason": reason,
        "suggestion": suggestion,
        "suggested_agent_endpoint": endpoint
    }
    
    return {
        "status": "success",
        "metadata": {
            "total_rows_analyzed": len(df),
            "total_issues": len(row_level_issues)
        },
        "routing": routing,
        "alerts": alerts,
        "data": {
            "readiness_score": scores,
            "deductions": deductions,
            "row_level_issues": row_level_issues[:100]  # Limit to first 100
        },
        "issue_summary": _summarize_readiness_issues(row_level_issues)
    }

def rate_readiness(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """
    Main function for the Readiness Rater agent.
    
    Args:
        file_contents: Raw bytes of the uploaded file
        filename: Name of the uploaded file
        config: Configuration dictionary containing thresholds (optional)
        user_overrides: User-provided threshold overrides for audit trail (optional)
    
    Returns:
        dict: Standardized JSON response with readiness results, audit trail, and Excel export
    """
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    
    # Use default config if not provided
    if config is None:
        config = {
            'ready_threshold': 85,
            'needs_review_threshold': 70,
            'completeness_weight': 0.4,
            'consistency_weight': 0.4,
            'schema_health_weight': 0.2
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
        # Add other file types (json, parquet) here if needed
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Calculate aggregate scores
    aggregate_scores = _calculate_aggregate_scores(results)
    
    # Populate overrides with actual configuration values being used
    effective_overrides = {
        "ready_threshold": config.get("ready_threshold"),
        "needs_review_threshold": config.get("needs_review_threshold"),
        "completeness_weight": config.get("completeness_weight"),
        "consistency_weight": config.get("consistency_weight"),
        "schema_health_weight": config.get("schema_health_weight")
    }
    
    # Build comprehensive audit trail
    audit_trail = {
        "agent_name": "ReadinessRater",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),  # Keep for backward compatibility
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),  # Unique fields
        "findings": all_findings,
        "actions": [
            f"Analyzed {total_rows_analyzed} rows across {len(results)} sheet(s)",
            f"Generated {total_alerts_generated} alert(s)",
            "Calculated completeness score based on null values",
            "Calculated consistency score based on duplicate rows",
            "Calculated schema health score based on data variance",
            f"Computed overall readiness score (weighted average)"
        ],
        "scores": {
            "total_sheets_analyzed": len(results),
            "total_rows_analyzed": total_rows_analyzed,
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info'),
            "average_readiness_score": aggregate_scores.get('average_overall', 0),
            "average_completeness_score": aggregate_scores.get('average_completeness', 0),
            "average_consistency_score": aggregate_scores.get('average_consistency', 0),
            "average_schema_health_score": aggregate_scores.get('average_schema_health', 0)
        },
        "overrides": effective_overrides,
        "lineage": {}
    }
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("ReadinessRater", results, audit_trail)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "ReadinessRater",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob

    return response


def _extract_findings_from_result(sheet_result: dict, sheet_name: str) -> list:
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
            "category": "readiness_assessment"
        }
        
        # Add score context if available
        if 'data' in sheet_result and 'readiness_score' in sheet_result['data']:
            scores = sheet_result['data']['readiness_score']
            finding["readiness_score"] = scores.get("overall", 0)
            finding["completeness_score"] = scores.get("completeness", 0)
            finding["consistency_score"] = scores.get("consistency", 0)
            finding["schema_health_score"] = scores.get("schema_health", 0)
        
        findings.append(finding)
    
    # Extract from deductions
    if 'data' in sheet_result and 'deductions' in sheet_result['data']:
        for deduction in sheet_result['data']['deductions']:
            # Categorize deduction
            category = "general"
            if "Completeness" in deduction:
                category = "data_completeness"
            elif "Consistency" in deduction:
                category = "data_consistency"
            elif "Schema Health" in deduction:
                category = "schema_health"
            
            finding = {
                "severity": "info",
                "sheet": sheet_name,
                "issue": deduction,
                "category": category
            }
            findings.append(finding)
    
    return findings


def _calculate_aggregate_scores(results: dict) -> dict:
    """
    Calculate aggregate scores across all sheets.
    """
    if not results:
        return {
            "average_overall": 0,
            "average_completeness": 0,
            "average_consistency": 0,
            "average_schema_health": 0
        }
    
    total_overall = 0
    total_completeness = 0
    total_consistency = 0
    total_schema_health = 0
    count = 0
    
    for sheet_result in results.values():
        if sheet_result.get('status') == 'success' and 'data' in sheet_result:
            scores = sheet_result['data'].get('readiness_score', {})
            total_overall += scores.get('overall', 0)
            total_completeness += scores.get('completeness', 0)
            total_consistency += scores.get('consistency', 0)
            total_schema_health += scores.get('schema_health', 0)
            count += 1
    
    if count == 0:
        return {
            "average_overall": 0,
            "average_completeness": 0,
            "average_consistency": 0,
            "average_schema_health": 0
        }
    
    return {
        "average_overall": round(total_overall / count),
        "average_completeness": round(total_completeness / count),
        "average_consistency": round(total_consistency / count),
        "average_schema_health": round(total_schema_health / count)
    }


def _generate_excel_export(response: dict) -> dict:
    """
    Generate Excel export blob with complete JSON response.
    """
    from io import BytesIO
    import json
    
    # Extract components from response
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "ReadinessRater")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        # Create Excel writer
        output = BytesIO()
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
            
            # Sheet 2: Readiness Summary
            if audit_trail:
                readiness_summary_data = {
                    "Metric": [
                        "Agent Name", "Agent Version", "Timestamp", "Compute Time (seconds)",
                        "Total Sheets Analyzed", "Total Rows Analyzed", "Total Alerts Generated",
                        "Critical Alerts", "Warning Alerts", "Info Alerts",
                        "Average Readiness Score", "Average Completeness Score",
                        "Average Consistency Score", "Average Schema Health Score"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", ""),
                        audit_trail.get("scores", {}).get("total_sheets_analyzed", ""),
                        audit_trail.get("scores", {}).get("total_rows_analyzed", ""),
                        audit_trail.get("scores", {}).get("total_alerts_generated", ""),
                        audit_trail.get("scores", {}).get("critical_alerts", ""),
                        audit_trail.get("scores", {}).get("warning_alerts", ""),
                        audit_trail.get("scores", {}).get("info_alerts", ""),
                        audit_trail.get("scores", {}).get("average_readiness_score", ""),
                        audit_trail.get("scores", {}).get("average_completeness_score", ""),
                        audit_trail.get("scores", {}).get("average_consistency_score", ""),
                        audit_trail.get("scores", {}).get("average_schema_health_score", "")
                    ]
                }
                readiness_summary_df = pd.DataFrame(readiness_summary_data)
                readiness_summary_df.to_excel(writer, sheet_name="Readiness Summary", index=False)
            
            # Sheet 3: Fields Scanned
            if audit_trail.get("fields_scanned"):
                fields_df = pd.DataFrame({
                    "Field Name": audit_trail["fields_scanned"]
                })
                fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            # Sheet 4: Findings
            if audit_trail.get("findings"):
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Sheet 5: Actions
            if audit_trail.get("actions"):
                actions_df = pd.DataFrame({
                    "Action": audit_trail["actions"]
                })
                actions_df.to_excel(writer, sheet_name="Actions", index=False)
            
            # Sheet 6: Configuration
            if audit_trail.get("overrides"):
                config_data = {
                    "Parameter": list(audit_trail["overrides"].keys()),
                    "Value": list(audit_trail["overrides"].values())
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name="Configuration", index=False)
            
            # Sheet 7: Routing Information
            routing_data = []
            for sheet_name, sheet_results in results.items():
                if "routing" in sheet_results:
                    routing_info = sheet_results["routing"]
                    routing_data.append({
                        "Sheet/Dataset": sheet_name,
                        "Status": routing_info.get("status", ""),
                        "Reason": routing_info.get("reason", ""),
                        "Suggestion": routing_info.get("suggestion", ""),
                        "Suggested Agent Endpoint": routing_info.get("suggested_agent_endpoint", "")
                    })
            
            if routing_data:
                routing_df = pd.DataFrame(routing_data)
                routing_df.to_excel(writer, sheet_name="Routing", index=False)
            
            # Sheet 8+: Detailed results per sheet
            for sheet_name, sheet_result in results.items():
                if sheet_result.get("status") == "success" and "data" in sheet_result:
                    data = sheet_result["data"]
                    
                    # Readiness scores
                    scores_data = {
                        "Score Type": ["Overall", "Completeness", "Consistency", "Schema Health"],
                        "Score": [
                            data["readiness_score"].get("overall", 0),
                            data["readiness_score"].get("completeness", 0),
                            data["readiness_score"].get("consistency", 0),
                            data["readiness_score"].get("schema_health", 0)
                        ]
                    }
                    scores_df = pd.DataFrame(scores_data)
                    
                    # Deductions
                    deductions_df = pd.DataFrame({
                        "Deduction": data.get("deductions", [])
                    })
                    
                    # Write to separate sheets
                    safe_sheet_name = sheet_name[:27]  # Excel limit with suffix
                    scores_df.to_excel(writer, sheet_name=f"{safe_sheet_name}_Scores", index=False)
                    if not deductions_df.empty:
                        deductions_df.to_excel(writer, sheet_name=f"{safe_sheet_name}_Deduct", index=False)
                        
                    # Add sheet-level metadata
                    if "metadata" in sheet_result:
                        metadata_data = {
                            "Metric": list(sheet_result["metadata"].keys()),
                            "Value": list(sheet_result["metadata"].values())
                        }
                        metadata_df = pd.DataFrame(metadata_data)
                        metadata_df.to_excel(writer, sheet_name=f"Meta_{safe_sheet_name}", index=False)
            
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
        
        # Build dynamic sheet list
        sheets_included = ["Response Overview"]
        if audit_trail:
            sheets_included.append("Readiness Summary")
        if audit_trail.get("fields_scanned"):
            sheets_included.append("Fields Scanned")
        if audit_trail.get("findings"):
            sheets_included.append("Findings")
        if audit_trail.get("actions"):
            sheets_included.append("Actions")
        if audit_trail.get("overrides"):
            sheets_included.append("Configuration")
        
        # Check if any routing data exists
        has_routing = any("routing" in sheet_results for sheet_results in results.values())
        if has_routing:
            sheets_included.append("Routing")
        
        # Add readiness assessment sheets
        for sheet_name in results.keys():
            safe_sheet_name = sheet_name[:27]
            sheets_included.extend([f"{safe_sheet_name}_Scores", f"Meta_{safe_sheet_name}"])
        
        sheets_included.append("Raw JSON")
        
        sheets_list = [sheet for sheet in sheets_included if sheet]  # Remove None values
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_readiness_report.xlsx",
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


def _detect_null_rows(df: pd.DataFrame) -> list:
    """
    Detect rows with null values.
    
    Args:
        df: DataFrame to check
    
    Returns:
        list: Row-level issues for null values
    """
    issues = []
    
    for col in df.columns:
        null_mask = df[col].isna()
        null_indices = df.index[null_mask].tolist()
        
        # Limit to first 20 null rows per column
        for idx in null_indices[:20]:
            issues.append({
                "row_index": int(idx),
                "column": col,
                "issue_type": "null_value",
                "severity": "warning" if len(null_indices) / len(df) > 0.2 else "info",
                "value": None,
                "message": f"Null value in column '{col}'"
            })
    
    return issues


def _detect_duplicate_rows_readiness(df: pd.DataFrame) -> list:
    """
    Detect duplicate rows.
    
    Args:
        df: DataFrame to check
    
    Returns:
        list: Row-level issues for duplicate rows
    """
    issues = []
    
    duplicate_mask = df.duplicated(keep=False)
    duplicate_indices = df.index[duplicate_mask].tolist()
    
    if len(duplicate_indices) > 0:
        # Group duplicates
        duplicate_groups = {}
        for idx in duplicate_indices:
            row_tuple = tuple(df.iloc[idx].values)
            if row_tuple not in duplicate_groups:
                duplicate_groups[row_tuple] = []
            duplicate_groups[row_tuple].append(int(idx))
        
        # Create issues for duplicate groups (limit to first 10)
        for group_idx, (row_values, indices) in enumerate(list(duplicate_groups.items())[:10]):
            issues.append({
                "row_index": indices,
                "column": None,
                "issue_type": "duplicate_row",
                "severity": "warning",
                "value": None,
                "message": f"Duplicate row found at indices {indices[:5]}{'...' if len(indices) > 5 else ''}",
                "duplicate_count": len(indices)
            })
    
    return issues


def _summarize_readiness_issues(issues: list) -> dict:
    """
    Summarize readiness issues by type and severity.
    
    Args:
        issues: List of row-level issues
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_issues": len(issues),
        "by_type": {},
        "by_severity": {},
        "by_column": {}
    }
    
    for issue in issues:
        issue_type = issue.get("issue_type", "unknown")
        summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
        
        severity = issue.get("severity", "info")
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        column = issue.get("column")
        if column:
            summary["by_column"][column] = summary["by_column"].get(column, 0) + 1
    
    return summary
