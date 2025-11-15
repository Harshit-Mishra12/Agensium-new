import pandas as pd
import numpy as np
import io
import time
import base64
from datetime import datetime, timezone
from fastapi import HTTPException
import warnings
from typing import Dict, List, Any, Optional, Set

from app.config import AGENT_ROUTES
from app.agents.shared.chat_agent import generate_llm_summary

AGENT_VERSION = "1.0.0"

def _analyze_schema_changs(baseline_schema: Dict[str, Any], current_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze schema changes between baseline and current datasets."""
    changes = {
        "added_columns": [],
        "removed_columns": [],
        "type_changes": [],
        "nullable_changes": [],
        "constraint_changes": []
    }
    
    baseline_cols = set(baseline_schema.keys())
    current_cols = set(current_schema.keys())
    
    # Detect added and removed columns
    changes["added_columns"] = list(current_cols - baseline_cols)
    changes["removed_columns"] = list(baseline_cols - current_cols)
    
    # Analyze changes in common columns
    common_cols = baseline_cols & current_cols
    
    for col in common_cols:
        baseline_info = baseline_schema[col]
        current_info = current_schema[col]
        
        # Type changes
        if baseline_info['dtype'] != current_info['dtype']:
            changes["type_changes"].append({
                "column": col,
                "baseline_type": baseline_info['dtype'],
                "current_type": current_info['dtype'],
                "compatibility": _assess_type_compatibility(baseline_info['dtype'], current_info['dtype'])
            })
        
        # Nullable changes
        if baseline_info['nullable'] != current_info['nullable']:
            changes["nullable_changes"].append({
                "column": col,
                "baseline_nullable": baseline_info['nullable'],
                "current_nullable": current_info['nullable'],
                "impact": "breaking" if baseline_info['nullable'] and not current_info['nullable'] else "non-breaking"
            })
        
        # Constraint changes (unique values, ranges)
        baseline_constraints = baseline_info.get('constraints', {})
        current_constraints = current_info.get('constraints', {})
        
        if baseline_constraints != current_constraints:
            changes["constraint_changes"].append({
                "column": col,
                "baseline_constraints": baseline_constraints,
                "current_constraints": current_constraints
            })
    
    return changes

def _assess_type_compatibility(baseline_type: str, current_type: str) -> str:
    """Assess compatibility between data types."""
    # Define compatibility matrix
    compatible_upgrades = {
        'int64': ['float64', 'object'],
        'int32': ['int64', 'float64', 'object'],
        'float32': ['float64', 'object'],
        'bool': ['object'],
        'datetime64[ns]': ['object']
    }
    
    if baseline_type == current_type:
        return "identical"
    elif current_type in compatible_upgrades.get(baseline_type, []):
        return "compatible"
    elif baseline_type in compatible_upgrades.get(current_type, []):
        return "downgrade"
    else:
        return "incompatible"

def _extract_schema_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract comprehensive schema information from a DataFrame."""
    schema = {}
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'nullable': bool(df[col].isnull().any()),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': float((df[col].isnull().sum() / len(df) * 100) if len(df) > 0 else 0),
            'unique_count': int(df[col].nunique()),
            'unique_percentage': float((df[col].nunique() / len(df) * 100) if len(df) > 0 else 0)
        }
        
        # Add type-specific constraints
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min() if not df[col].empty else None
            max_val = df[col].max() if not df[col].empty else None
            mean_val = df[col].mean() if not df[col].empty else None
            
            col_info['constraints'] = {
                'min_value': float(min_val) if min_val is not None and not pd.isna(min_val) else None,
                'max_value': float(max_val) if max_val is not None and not pd.isna(max_val) else None,
                'mean': float(mean_val) if mean_val is not None and not pd.isna(mean_val) else None
            }
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            max_len = df[col].astype(str).str.len().max() if not df[col].empty else None
            min_len = df[col].astype(str).str.len().min() if not df[col].empty else None
            
            col_info['constraints'] = {
                'max_length': int(max_len) if max_len is not None and not pd.isna(max_len) else None,
                'min_length': int(min_len) if min_len is not None and not pd.isna(min_len) else None
            }
        
        schema[col] = col_info
    
    return schema

def _calculate_drift_score(changes: Dict[str, Any], config: dict) -> Dict[str, Any]:
    """Calculate overall schema drift score and impact assessment."""
    weights = {
        'added_columns_weight': config.get('added_columns_weight', 0.1),
        'removed_columns_weight': config.get('removed_columns_weight', 0.4),
        'type_changes_weight': config.get('type_changes_weight', 0.3),
        'nullable_changes_weight': config.get('nullable_changes_weight', 0.15),
        'constraint_changes_weight': config.get('constraint_changes_weight', 0.05)
    }
    
    # Calculate impact scores
    added_impact = len(changes['added_columns']) * 10
    removed_impact = len(changes['removed_columns']) * 40
    
    type_impact = 0
    for change in changes['type_changes']:
        if change['compatibility'] == 'incompatible':
            type_impact += 50
        elif change['compatibility'] == 'downgrade':
            type_impact += 30
        else:
            type_impact += 10
    
    nullable_impact = 0
    for change in changes['nullable_changes']:
        if change['impact'] == 'breaking':
            nullable_impact += 25
        else:
            nullable_impact += 5
    
    constraint_impact = len(changes['constraint_changes']) * 5
    
    # Calculate weighted drift score
    total_impact = (
        added_impact * weights['added_columns_weight'] +
        removed_impact * weights['removed_columns_weight'] +
        type_impact * weights['type_changes_weight'] +
        nullable_impact * weights['nullable_changes_weight'] +
        constraint_impact * weights['constraint_changes_weight']
    )
    
    # Determine drift level
    critical_threshold = config.get('critical_drift_threshold', 50)
    moderate_threshold = config.get('moderate_drift_threshold', 20)
    
    if total_impact >= critical_threshold:
        drift_level = "critical"
        drift_color = "red"
    elif total_impact >= moderate_threshold:
        drift_level = "moderate"
        drift_color = "yellow"
    else:
        drift_level = "minimal"
        drift_color = "green"
    
    return {
        "total_score": round(total_impact, 1),
        "drift_level": drift_level,
        "drift_color": drift_color,
        "impact_breakdown": {
            "added_columns": added_impact,
            "removed_columns": removed_impact,
            "type_changes": type_impact,
            "nullable_changes": nullable_impact,
            "constraint_changes": constraint_impact
        },
        "weights_used": weights
    }

def _generate_dev_recommendations(changes: Dict[str, Any], drift_score: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate developer recommendations based on schema changes."""
    recommendations = []
    
    # Critical issues first
    if changes['removed_columns']:
        recommendations.append({
            "priority": "critical",
            "category": "breaking_change",
            "title": "Removed Columns Detected",
            "description": f"Columns {changes['removed_columns']} have been removed. This is a breaking change.",
            "action": "Update dependent code, add migration scripts, or restore columns if removal was unintentional."
        })
    
    for change in changes['type_changes']:
        if change['compatibility'] == 'incompatible':
            recommendations.append({
                "priority": "critical",
                "category": "breaking_change",
                "title": f"Incompatible Type Change: {change['column']}",
                "description": f"Column '{change['column']}' changed from {change['baseline_type']} to {change['current_type']}.",
                "action": "Review data transformation logic and update type handling in consuming applications."
            })
    
    # Breaking nullable changes
    for change in changes['nullable_changes']:
        if change['impact'] == 'breaking':
            recommendations.append({
                "priority": "high",
                "category": "breaking_change",
                "title": f"Nullable Constraint Added: {change['column']}",
                "description": f"Column '{change['column']}' is no longer nullable.",
                "action": "Ensure all data sources provide values for this column or add default value handling."
            })
    
    # Non-breaking but notable changes
    if changes['added_columns']:
        recommendations.append({
            "priority": "medium",
            "category": "enhancement",
            "title": "New Columns Available",
            "description": f"New columns added: {changes['added_columns']}",
            "action": "Consider leveraging new data fields in analytics and reporting."
        })
    
    # Compatible type changes
    compatible_changes = [c for c in changes['type_changes'] if c['compatibility'] == 'compatible']
    if compatible_changes:
        recommendations.append({
            "priority": "low",
            "category": "optimization",
            "title": "Compatible Type Upgrades",
            "description": f"Columns with compatible type upgrades: {[c['column'] for c in compatible_changes]}",
            "action": "Update type annotations and consider optimizing for new data types."
        })
    
    return recommendations

def _generate_routing_info(drift_score: Dict[str, Any], total_changes: int) -> Dict[str, Any]:
    """Generate routing information based on schema drift analysis."""
    drift_level = drift_score.get('drift_level', 'unknown')
    total_score = drift_score.get('total_score', 0)
    
    if drift_level == 'critical':
        return {
            "status": "Critical Drift",
            "reason": f"Critical schema changes detected with impact score {total_score}. Immediate attention required.",
            "suggestion": "Review breaking changes, update dependent systems, and create migration plan.",
            "suggested_agent_endpoint": "/run-tool/governance"
        }
    elif drift_level == 'moderate':
        return {
            "status": "Moderate Drift",
            "reason": f"Moderate schema changes detected with impact score {total_score}. Review recommended.",
            "suggestion": "Assess impact on downstream systems and plan updates accordingly.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }
    else:
        return {
            "status": "Minimal Drift",
            "reason": f"Minimal schema changes detected with impact score {total_score}. Changes are manageable.",
            "suggestion": "Monitor changes and update documentation as needed.",
            "suggested_agent_endpoint": "/run-tool/profile-my-data"
        }

def _compare_datasets(baseline_df: pd.DataFrame, current_df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Compare two datasets and generate schema drift report."""
    # Extract schema information
    baseline_schema = _extract_schema_info(baseline_df)
    current_schema = _extract_schema_info(current_df)
    
    # Analyze changes
    changes = _analyze_schema_changes(baseline_schema, current_schema)
    
    # Calculate drift score
    drift_score = _calculate_drift_score(changes, config)
    
    # Generate recommendations
    recommendations = _generate_dev_recommendations(changes, drift_score)
    
    # Count total changes
    total_changes = (
        len(changes['added_columns']) +
        len(changes['removed_columns']) +
        len(changes['type_changes']) +
        len(changes['nullable_changes']) +
        len(changes['constraint_changes'])
    )
    
    # Generate alerts
    alerts = []
    
    # Critical alerts
    if changes['removed_columns']:
        alerts.append({
            "level": "critical",
            "message": f"Breaking change: {len(changes['removed_columns'])} columns removed",
            "type": "column_removal",
            "details": {"removed_columns": changes['removed_columns']}
        })
    
    for change in changes['type_changes']:
        if change['compatibility'] == 'incompatible':
            alerts.append({
                "level": "critical",
                "message": f"Incompatible type change in column '{change['column']}'",
                "type": "type_incompatibility",
                "details": change
            })
    
    # Warning alerts
    if changes['added_columns']:
        alerts.append({
            "level": "warning",
            "message": f"{len(changes['added_columns'])} new columns detected",
            "type": "column_addition",
            "details": {"added_columns": changes['added_columns']}
        })
    
    # Generate summary
    summary = f"Schema drift analysis completed. Drift level: {drift_score['drift_level']} (score: {drift_score['total_score']}). "
    summary += f"Found {total_changes} schema changes: {len(changes['added_columns'])} added, {len(changes['removed_columns'])} removed, "
    summary += f"{len(changes['type_changes'])} type changes. {len(recommendations)} recommendations generated."
    
    # Generate routing info
    routing_info = _generate_routing_info(drift_score, total_changes)
    
    return {
        "status": "success",
        "metadata": {
            "baseline_columns": len(baseline_schema),
            "current_columns": len(current_schema),
            "total_changes": total_changes
        },
        "routing": routing_info,
        "data": {
            "drift_score": drift_score,
            "schema_changes": changes,
            "baseline_schema": baseline_schema,
            "current_schema": current_schema,
            "recommendations": recommendations,
            "summary": summary,
            "baseline_rows": len(baseline_df),
            "current_rows": len(current_df)
        },
        "alerts": alerts
    }

def report_schema_drift(baseline_contents: bytes, current_contents: bytes, baseline_filename: str, current_filename: str, config: dict = None, user_overrides: dict = None):
    """Main function for the SchemaDriftReporter agent."""
    start_time = time.time()
    run_timestamp = datetime.now(timezone.utc)
    
    # Config should always be provided by routes.py from config.json
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not provided. This should be loaded from config.json by the route handler.")
    
    try:
        # Load baseline dataset
        baseline_ext = baseline_filename.split('.')[-1].lower()
        if baseline_ext == 'csv':
            baseline_df = pd.read_csv(io.BytesIO(baseline_contents))
        elif baseline_ext in ['xlsx', 'xls']:
            baseline_df = pd.read_excel(io.BytesIO(baseline_contents))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported baseline file format: {baseline_ext}")
        
        # Load current dataset
        current_ext = current_filename.split('.')[-1].lower()
        if current_ext == 'csv':
            current_df = pd.read_csv(io.BytesIO(current_contents))
        elif current_ext in ['xlsx', 'xls']:
            current_df = pd.read_excel(io.BytesIO(current_contents))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported current file format: {current_ext}")
        
        # Compare datasets
        result = _compare_datasets(baseline_df, current_df, config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files. Error: {str(e)}")
    
    end_time = time.time()
    compute_time = end_time - start_time
    
    # Build audit trail
    audit_trail = {
        "agent_name": "SchemaDriftReporter",
        "timestamp": run_timestamp.isoformat(),
        "agent_version": AGENT_VERSION,
        "compute_time_seconds": round(compute_time, 2),
        "baseline_file": baseline_filename,
        "current_file": current_filename,
        "actions": [
            f"Analyzed baseline schema ({len(baseline_df.columns)} columns, {len(baseline_df)} rows)",
            f"Analyzed current schema ({len(current_df.columns)} columns, {len(current_df)} rows)",
            "Detected schema changes and compatibility issues",
            "Generated developer recommendations",
            "Calculated schema drift impact score"
        ],
        "scores": {
            "drift_score": result['data']['drift_score']['total_score'],
            "drift_level": result['data']['drift_score']['drift_level'],
            "total_changes": result['metadata']['total_changes'],
            "recommendations_count": len(result['data']['recommendations'])
        },
        "overrides": user_overrides if user_overrides else {}
    }
    
    # Generate LLM summary
    llm_summary = generate_llm_summary("SchemaDriftReporter", {"comparison": result}, audit_trail)
    
    return {
        "source_file": f"{baseline_filename} vs {current_filename}",
        "agent": "SchemaDriftReporter",
        "audit": audit_trail,
        "results": {"comparison": result},
        "summary": llm_summary,
        "excel_export": ""
    }