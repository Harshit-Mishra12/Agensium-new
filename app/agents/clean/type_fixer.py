import pandas as pd
import numpy as np
import io
import time
import base64
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings
from typing import Dict, List, Any, Optional, Union
import re

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return str(val)
        return val
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return str(obj)
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def _analyze_type_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data type inconsistencies in the dataset."""
    type_analysis = {
        "total_rows": len(df),
        "columns_with_issues": [],
        "type_summary": {},
        "recommendations": []
    }
    
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        current_dtype = str(df[col].dtype)
        issues = []
        suggested_type = current_dtype
        
        # Check for mixed types in object columns
        if current_dtype == 'object':
            numeric_count = 0
            date_count = 0
            string_count = 0
            
            sample_size = min(100, len(col_data))
            for val in col_data.head(sample_size):
                if _is_numeric_string(str(val)):
                    numeric_count += 1
                elif _is_date_string(str(val)):
                    date_count += 1
                else:
                    string_count += 1
            
            total_sampled = numeric_count + date_count + string_count
            if total_sampled > 0:
                numeric_pct = (numeric_count / total_sampled) * 100
                date_pct = (date_count / total_sampled) * 100
                
                if numeric_pct > 70:
                    issues.append("Should be numeric type")
                    suggested_type = "numeric"
                elif date_pct > 70:
                    issues.append("Should be datetime type")
                    suggested_type = "datetime"
        
        # Check for incorrectly typed numeric columns
        elif current_dtype in ['int64', 'float64']:
            if current_dtype == 'float64':
                try:
                    is_all_int = col_data.apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all()
                    if is_all_int:
                        issues.append("Float column contains only integer values")
                        suggested_type = "integer"
                except:
                    pass
        
        if issues:
            type_analysis["columns_with_issues"].append(str(col))
            type_analysis["type_summary"][str(col)] = {
                "current_type": current_dtype,
                "suggested_type": suggested_type,
                "issues": issues,
                "sample_values": [str(x) for x in col_data.head(5).tolist()]
            }
            
            priority = "high" if len(issues) > 1 else "medium"
            type_analysis["recommendations"].append({
                "column": str(col),
                "action": f"convert_to_{suggested_type}",
                "reason": "; ".join(issues),
                "priority": priority
            })
    
    return type_analysis

def _is_numeric_string(value: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def _is_date_string(value: str) -> bool:
    """Check if a string represents a date."""
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, str(value)):
            return True
    return False

def _apply_type_fixes(df: pd.DataFrame, fix_config: Dict[str, Any]) -> tuple:
    """Apply type fixes to the dataframe."""
    df_fixed = df.copy()
    fix_log = []
    
    column_fixes = fix_config.get('column_fixes', {})
    
    for col, target_type in column_fixes.items():
        if col not in df_fixed.columns:
            continue
            
        try:
            original_type = str(df_fixed[col].dtype)
            
            if target_type == 'numeric':
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                fix_log.append(f"Converted '{col}' from {original_type} to numeric")
                
            elif target_type == 'integer':
                df_fixed[col] = df_fixed[col].astype('Int64')
                fix_log.append(f"Converted '{col}' from {original_type} to integer")
                
            elif target_type == 'datetime':
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
                fix_log.append(f"Converted '{col}' from {original_type} to datetime")
                
            elif target_type == 'string':
                df_fixed[col] = df_fixed[col].astype(str)
                fix_log.append(f"Converted '{col}' from {original_type} to string")
                
            elif target_type == 'category':
                df_fixed[col] = df_fixed[col].astype('category')
                fix_log.append(f"Converted '{col}' from {original_type} to category")
                
        except Exception as e:
            fix_log.append(f"Error converting '{col}' to {target_type}: {str(e)}")
    
    return df_fixed, fix_log

def _calculate_fixing_score(original_df: pd.DataFrame, fixed_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Calculate type fixing effectiveness score."""
    original_issues = len(_analyze_type_issues(original_df)["columns_with_issues"])
    remaining_issues = len(_analyze_type_issues(fixed_df)["columns_with_issues"])
    
    # Calculate metrics
    issue_reduction_rate = ((original_issues - remaining_issues) / original_issues * 100) if original_issues > 0 else 100
    data_retention_rate = (len(fixed_df) / len(original_df) * 100) if len(original_df) > 0 else 0
    column_retention_rate = (len(fixed_df.columns) / len(original_df.columns) * 100) if len(original_df.columns) > 0 else 0
    
    # Calculate weighted score
    issue_weight = float(config.get('issue_reduction_weight', 0.6))
    data_weight = float(config.get('data_retention_weight', 0.3))
    column_weight = float(config.get('column_retention_weight', 0.1))
    
    overall_score = (
        float(issue_reduction_rate) * issue_weight +
        float(data_retention_rate) * data_weight +
        float(column_retention_rate) * column_weight
    )
    
    # Determine fixing quality
    excellent_threshold = float(config.get('excellent_threshold', 90))
    good_threshold = float(config.get('good_threshold', 75))
    
    if overall_score >= excellent_threshold:
        quality = "excellent"
        quality_color = "green"
    elif overall_score >= good_threshold:
        quality = "good"
        quality_color = "yellow"
    else:
        quality = "needs_improvement"
        quality_color = "red"
    
    return _convert_numpy_types({
        "overall_score": round(overall_score, 1),
        "quality": quality,
        "quality_color": quality_color,
        "metrics": {
            "issue_reduction_rate": round(issue_reduction_rate, 1),
            "data_retention_rate": round(data_retention_rate, 1),
            "column_retention_rate": round(column_retention_rate, 1),
            "original_issues": original_issues,
            "remaining_issues": remaining_issues,
            "original_rows": len(original_df),
            "fixed_rows": len(fixed_df),
            "original_columns": len(original_df.columns),
            "fixed_columns": len(fixed_df.columns)
        },
        "weights_used": {
            "issue_reduction_weight": issue_weight,
            "data_retention_weight": data_weight,
            "column_retention_weight": column_weight
        }
    })

def _detect_type_row_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect row-level type issues in the DataFrame."""
    issues = []
    
    for col in df.columns:
        if str(df[col].dtype) == 'object':
            # Check for mixed types in object columns
            for idx, value in df[col].head(100).items():  # Limit to first 100 rows
                if pd.notnull(value):
                    if _is_numeric_string(str(value)) and not isinstance(value, (int, float)):
                        issues.append({
                            "row_index": int(idx),
                            "column": str(col),
                            "issue_type": "mixed_type",
                            "description": f"Numeric value stored as text in column '{col}'",
                            "severity": "warning",
                            "value": str(value)
                        })
                        
                        if len(issues) >= 100:  # Limit for performance
                            break
    
    return issues

def _generate_routing_info(fixing_score: Dict[str, Any], remaining_issues: int) -> Dict[str, Any]:
    """Generate routing information based on type fixing results."""
    quality = fixing_score.get('quality', 'unknown')
    overall_score = fixing_score.get('overall_score', 0)
    
    if quality == 'excellent' and remaining_issues == 0:
        return {
            "status": "Types Fixed",
            "reason": f"Data types successfully fixed with score {float(overall_score)}/100. No type issues remain.",
            "suggestion": "Data types are properly aligned. Consider running quality checks or profiling.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    elif quality == 'excellent' or quality == 'good':
        return {
            "status": "Well Fixed",
            "reason": f"Data types fixed with {quality} quality (score: {float(overall_score)}/100). {int(remaining_issues)} issues remain.",
            "suggestion": "Consider additional type fixes if needed, or proceed with analysis.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    else:
        return {
            "status": "Needs More Fixing",
            "reason": f"Type fixing quality needs improvement (score: {float(overall_score)}/100). {int(remaining_issues)} issues remain.",
            "suggestion": "Review type conversion strategies and consider manual type specifications.",
            "suggested_agent_endpoint": "/run-tool/type-fixer"
        }

def _process_dataframe(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Process a single dataframe for type fixing."""
    if df.empty:
        return {
            "status": "success",
            "metadata": {"total_rows_processed": 0, "issues_fixed": 0},
            "routing": {
                "status": "No Data",
                "reason": "Dataset is empty, no type fixing needed.",
                "suggestion": "Provide a dataset with data to fix types.",
                "suggested_agent_endpoint": "/run-tool/profile-my-data"
            },
            "data": {
                "fixing_score": {"overall_score": 0, "quality": "no_data"},
                "summary": "Dataset is empty, no type fixing performed."
            },
            "alerts": [{"level": "warning", "message": "Dataset is empty"}]
        }
    
    # Analyze type issues
    type_analysis = _analyze_type_issues(df)
    
    # Track row-level issues before fixing
    row_level_issues = _detect_type_row_issues(df)
    
    # Apply type fixes
    df_fixed, fix_log = _apply_type_fixes(df, config)
    
    # Calculate fixing effectiveness
    fixing_score = _calculate_fixing_score(df, df_fixed, config)
    
    # Generate alerts
    alerts = []
    remaining_issues = fixing_score['metrics']['remaining_issues']
    
    if remaining_issues > 0:
        alerts.append({
            "level": "warning",
            "message": f"{int(remaining_issues)} type issues remain after fixing",
            "type": "remaining_issues",
            "details": {"remaining_issues": _convert_numpy_types(remaining_issues)}
        })
    
    if fixing_score['quality'] == 'needs_improvement':
        alerts.append({
            "level": "warning",
            "message": f"Type fixing quality is below expectations (score: {float(fixing_score['overall_score'])}/100)",
            "type": "low_quality",
            "details": fixing_score['metrics']
        })
    
    # Generate summary
    original_issues = fixing_score['metrics']['original_issues']
    issues_fixed = original_issues - remaining_issues
    
    summary = f"Type fixing completed. Quality: {fixing_score['quality']} (score: {float(fixing_score['overall_score'])}/100). "
    summary += f"Processed {int(len(df))} rows, fixed {int(issues_fixed)} type issues, {int(remaining_issues)} issues remain. "
    summary += f"Applied {int(len(fix_log))} type conversions."
    
    # Generate routing info
    routing_info = _generate_routing_info(fixing_score, remaining_issues)
    
    return {
        "status": "success",
        "metadata": _convert_numpy_types({
            "total_rows_processed": len(df_fixed),
            "issues_fixed": issues_fixed,
            "original_issues": original_issues,
            "remaining_issues": remaining_issues,
            "total_issues": len(row_level_issues)
        }),
        "routing": routing_info,
        "data": {
            "fixing_score": fixing_score,
            "type_analysis": type_analysis,
            "fix_log": fix_log,
            "summary": summary,
            "fixed_data_shape": list(df_fixed.shape),
            "original_data_shape": list(df.shape),
            "row_level_issues": row_level_issues[:100]  # Limit to first 100 issues for performance
        },
        "alerts": alerts,
        "fixed_dataframe": df_fixed
    }

def _generate_excel_export(response: dict) -> dict:
    """Generate Excel export blob with complete JSON response."""
    import json
    
    filename = response.get("source_file", "unknown")
    agent_name = response.get("agent", "TypeFixer")
    audit_trail = response.get("audit", {})
    results = response.get("results", {})
    summary = response.get("summary", "")
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Sheet 1: Response Overview
            overview_data = {
                "Field": ["Source File", "Agent", "Summary"],
                "Value": [
                    filename,
                    agent_name,
                    summary[:500] + "..." if len(summary) > 500 else summary
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name="Response Overview", index=False)
            
            # Sheet 2: Audit Summary
            if audit_trail:
                audit_summary_data = {
                    "Metric": [
                        "Agent Name", "Agent Version", "Timestamp", "Compute Time (seconds)",
                        "Total Sheets Processed", "Total Rows Processed", "Total Issues Fixed",
                        "Total Alerts Generated", "Critical Alerts", "Warning Alerts", "Info Alerts"
                    ],
                    "Value": [
                        audit_trail.get("agent_name", ""),
                        audit_trail.get("agent_version", ""),
                        audit_trail.get("timestamp", ""),
                        audit_trail.get("compute_time_seconds", 0),
                        audit_trail.get("scores", {}).get("total_sheets_processed", 0),
                        audit_trail.get("scores", {}).get("total_rows_processed", 0),
                        audit_trail.get("scores", {}).get("total_issues_fixed", 0),
                        audit_trail.get("scores", {}).get("total_alerts_generated", 0),
                        audit_trail.get("scores", {}).get("critical_alerts", 0),
                        audit_trail.get("scores", {}).get("warning_alerts", 0),
                        audit_trail.get("scores", {}).get("info_alerts", 0)
                    ]
                }
                audit_summary_df = pd.DataFrame(audit_summary_data)
                audit_summary_df.to_excel(writer, sheet_name="Audit Summary", index=False)
            
            # Additional sheets for findings, actions, etc. (similar to null_handler)
            if audit_trail.get("fields_scanned"):
                fields_df = pd.DataFrame({"Field Name": audit_trail["fields_scanned"]})
                fields_df.to_excel(writer, sheet_name="Fields Scanned", index=False)
            
            if audit_trail.get("findings"):
                findings_df = pd.DataFrame(audit_trail["findings"])
                findings_df.to_excel(writer, sheet_name="Findings", index=False)
            
            # Complete JSON Response
            json_data = {
                "Component": ["Complete JSON Response"],
                "JSON Data": [json.dumps(response, indent=2, default=str)]
            }
            json_df = pd.DataFrame(json_data)
            json_df.to_excel(writer, sheet_name="Raw JSON", index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_type_fixing_report.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64,
            "sheets_included": ["Response Overview", "Audit Summary", "Raw JSON"],
            "download_ready": True
        }
        
    except Exception as e:
        return {
            "filename": f"{filename.rsplit('.', 1)[0]}_type_fixing_report.xlsx",
            "error": f"Failed to generate Excel export: {str(e)}",
            "download_ready": False
        }

def fix_types(file_contents: bytes, filename: str, config: dict = None, user_overrides: dict = None):
    """Main function for the TypeFixer agent."""
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    file_extension = filename.split('.')[-1].lower()
    results = {}
    fixed_dataframes = {}  # Store fixed dataframes for Excel export
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not provided. This should be loaded from config.json by the route handler.")
    
    # Track processing data
    all_sheets_processed = []
    total_issues_fixed = 0
    total_rows_processed = 0
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.BytesIO(file_contents))
            sheet_name = filename.rsplit('.', 1)[0]
            all_sheets_processed.append(sheet_name)
            
            sheet_result = _process_dataframe(df, config)
            results[sheet_name] = sheet_result
            
            # Store fixed dataframe before it gets removed
            if 'fixed_dataframe' in sheet_result:
                fixed_dataframes[sheet_name] = sheet_result['fixed_dataframe']
            
            total_issues_fixed += sheet_result['metadata']['issues_fixed']
            total_rows_processed += sheet_result['metadata']['total_rows_processed']
            
        elif file_extension in ['xlsx', 'xls']:
            xls_sheets = pd.read_excel(io.BytesIO(file_contents), sheet_name=None)
            for sheet_name, df in xls_sheets.items():
                all_sheets_processed.append(sheet_name)
                
                sheet_result = _process_dataframe(df, config)
                results[sheet_name] = sheet_result
                
                # Store fixed dataframe before it gets removed
                if 'fixed_dataframe' in sheet_result:
                    fixed_dataframes[sheet_name] = sheet_result['fixed_dataframe']
                
                total_issues_fixed += sheet_result['metadata']['issues_fixed']
                total_rows_processed += sheet_result['metadata']['total_rows_processed']
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file '{filename}'. Error: {str(e)}")

    end_time = time.time()
    compute_time = end_time - start_time
    
    # Generate Excel file with all fixed data
    updated_excel_export = {}
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df_fixed in fixed_dataframes.items():
                safe_sheet_name = sheet_name[:31]
                df_fixed.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
        output.seek(0)
        excel_bytes = output.read()
        excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
        
        updated_excel_export = {
            "filename": f"{filename.rsplit('.', 1)[0]}_type_fixed_data.xlsx",
            "size_bytes": len(excel_bytes),
            "format": "xlsx",
            "base64_data": excel_base64
        }
        
    except Exception as e:
        updated_excel_export = {
            "filename": f"{filename.rsplit('.', 1)[0]}_type_fixed_data.xlsx",
            "size_bytes": 0,
            "format": "xlsx",
            "base64_data": "",
            "error": f"Failed to generate fixed data Excel: {str(e)}"
        }
    
    # Extract audit trail data
    all_fields_scanned = []
    all_findings = []
    total_alerts_generated = 0
    
    for sheet_name, sheet_result in results.items():
        if 'data' in sheet_result and 'type_analysis' in sheet_result['data']:
            type_analysis = sheet_result['data']['type_analysis']
            if 'columns_with_issues' in type_analysis:
                all_fields_scanned.extend(type_analysis['columns_with_issues'])
        
        alerts = sheet_result.get('alerts', [])
        total_alerts_generated += len(alerts)
        for alert in alerts:
            finding = {
                "severity": alert.get("level", "info"),
                "sheet": sheet_name,
                "issue": alert.get("message", ""),
                "category": "type_fixing",
                "type": alert.get("type", "unknown")
            }
            all_findings.append(finding)
    
    # Build audit trail
    effective_overrides = {
        "auto_convert_numeric": config.get("auto_convert_numeric"),
        "auto_convert_datetime": config.get("auto_convert_datetime"),
        "preserve_mixed_types": config.get("preserve_mixed_types"),
        "quality_threshold": config.get("quality_threshold")
    }
    
    audit_trail = {
        "agent_name": "TypeFixer",
        "timestamp": run_timestamp.isoformat(),
        "profile_date": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "fields_scanned": list(set(all_fields_scanned)),
        "findings": all_findings,
        "actions": [
            f"Processed {total_rows_processed} rows across {len(results)} sheet(s)",
            f"Fixed {total_issues_fixed} type issues",
            "Applied configured type conversion strategies",
            "Generated type-fixed dataset",
            "Created Excel export with fixed data"
        ],
        "scores": _convert_numpy_types({
            "total_sheets_processed": len(results),
            "total_rows_processed": total_rows_processed,
            "total_issues_fixed": total_issues_fixed,
            "total_alerts_generated": total_alerts_generated,
            "critical_alerts": sum(1 for f in all_findings if f.get('severity') == 'critical'),
            "warning_alerts": sum(1 for f in all_findings if f.get('severity') == 'warning'),
            "info_alerts": sum(1 for f in all_findings if f.get('severity') == 'info')
        }),
        "overrides": effective_overrides,
        "lineage": {}
    }
    
    # Remove fixed_dataframe from results before JSON serialization
    for sheet_name, sheet_result in results.items():
        if 'fixed_dataframe' in sheet_result:
            del sheet_result['fixed_dataframe']
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("TypeFixer", results, audit_trail)
    
    # Build final response
    response = {
        "source_file": filename,
        "agent": "TypeFixer",
        "audit": audit_trail,
        "results": results,
        "summary": llm_summary,
        "updated_excel_export": updated_excel_export
    }
    
    # Generate Excel export blob with complete response
    excel_blob = _generate_excel_export(response)
    response["excel_export"] = excel_blob
    
    return _convert_numpy_types(response)