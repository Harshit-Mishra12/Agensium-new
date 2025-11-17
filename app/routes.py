from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends,Body
from typing import Optional, List, Any
from app.config import AGENT_ROUTES
from app.agents.source import (
    readiness_rater,
    risk_scorer,
    drift_detector, 
    governance_checker,
    test_coverage_agent,
    schema_drift_reporter
)
from app.agents.clean import (
    null_handler,
    outlier_remover,
    type_fixer
)
from app.agents.shared import chat_agent
from app.agents.source import unified_profiler
import json
import os
from pathlib import Path

# --- New imports for the Smart Orchestration Engine ---
from app.tools.profile_my_data_orchestrator import run_workflow
from app.tools.master_my_data_orchestrator import run_master_workflow
from app.tools.workflow_definitions import PROFILE_MY_DATA_WORKFLOW, MASTER_MY_DATA_WORKFLOW


router = APIRouter()

# --- Tool Endpoints (Orchestrators) ---

@router.post(AGENT_ROUTES['profile_my_data_tool'])
async def profile_my_data_endpoint(
    current_file: UploadFile = File(...),
    # Use 'Any' and 'Form' to gracefully handle clients that send an empty string for optional files
    baseline_file: Any = Form(None)
):
    """
    Orchestrator endpoint for the 'Profile My Data' tool.
    It runs a full analysis by calling the smart workflow engine.
    """
    
    # 1. Prepare the files dictionary for the engine
    files = {
        "current_file": {"contents": await current_file.read(), "filename": current_file.filename}
    }

    # Check if baseline_file is a real UploadFile and not an empty string from the form
    if isinstance(baseline_file, UploadFile):
        # A small check to ensure it's not an empty upload with no filename
        if baseline_file.filename:
             files["baseline_file"] = {"contents": await baseline_file.read(), "filename": baseline_file.filename}

    # 2. Call the generic workflow engine with the specific workflow definition
    return run_workflow(
        workflow_def=PROFILE_MY_DATA_WORKFLOW,
        files=files
    )

# --- Individual Agent Endpoints (The "Expert's Toolbox") ---

@router.post(AGENT_ROUTES['rate_readiness'])
async def rate_readiness_endpoint(
    file: UploadFile = File(...),
    ready_threshold: Optional[str] = Form(None),
    needs_review_threshold: Optional[str] = Form(None),
    completeness_weight: Optional[str] = Form(None),
    consistency_weight: Optional[str] = Form(None),
    schema_health_weight: Optional[str] = Form(None)
):
    """
    Endpoint for the Readiness Rater agent.
    Accepts optional threshold parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'ready_threshold': parse_param(ready_threshold, int),
        'needs_review_threshold': parse_param(needs_review_threshold, int),
        'completeness_weight': parse_param(completeness_weight, float),
        'consistency_weight': parse_param(consistency_weight, float),
        'schema_health_weight': parse_param(schema_health_weight, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # If all parameters are provided, use them directly
    if all(value is not None for value in user_params.values()):
        config = user_params
    else:
        # Load defaults from config.json for missing parameters
        try:
            config_path = Path(__file__).parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)['ReadinessRater']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Override with user-provided parameters where present
        for key, value in user_params.items():
            if value is not None:
                config[key] = value
    
    return readiness_rater.rate_readiness(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['score_risk'])
async def score_risk_endpoint(
    file: UploadFile = File(...),
    pii_sample_size: Optional[str] = Form(None),
    high_risk_threshold: Optional[str] = Form(None),
    medium_risk_threshold: Optional[str] = Form(None),
    pii_detection_enabled: Optional[str] = Form(None),
    sensitive_field_detection_enabled: Optional[str] = Form(None),
    governance_check_enabled: Optional[str] = Form(None)
):
    """
    Endpoint for the RiskScorer agent. What are
    Accepts optional threshold parameters that override config.json defaults. This is to set the 
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            if param_type == bool:
                return value.lower() in ['true', '1', 'yes']
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'pii_sample_size': parse_param(pii_sample_size, int),
        'high_risk_threshold': parse_param(high_risk_threshold, int),
        'medium_risk_threshold': parse_param(medium_risk_threshold, int),
        'pii_detection_enabled': parse_param(pii_detection_enabled, bool),
        'sensitive_field_detection_enabled': parse_param(sensitive_field_detection_enabled, bool),
        'governance_check_enabled': parse_param(governance_check_enabled, bool)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # If all parameters are provided, use them directly
    if all(value is not None for value in user_params.values()):
        config = user_params
    else:
        # Load defaults from config.json for missing parameters
        try:
            config_path = Path(__file__).parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)['RiskScorer']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Override with user-provided parameters where present
        for key, value in user_params.items():
            if value is not None:
                config[key] = value
    
    return risk_scorer.score_risk(contents, file.filename, config, user_overrides)


@router.post(AGENT_ROUTES['unified_profiler'])
async def unified_profiler_endpoint(
    file: UploadFile = File(...),
    null_alert_threshold: Optional[str] = Form(None),
    categorical_threshold: Optional[str] = Form(None),
    categorical_ratio_threshold: Optional[str] = Form(None),
    top_n_values: Optional[str] = Form(None),
    outlier_iqr_multiplier: Optional[str] = Form(None),
    outlier_alert_threshold: Optional[str] = Form(None)
):
    """
    Endpoint for the Unified Profiler agent.
    Accepts optional threshold parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'null_alert_threshold': parse_param(null_alert_threshold, float),
        'categorical_threshold': parse_param(categorical_threshold, int),
        'categorical_ratio_threshold': parse_param(categorical_ratio_threshold, float),
        'top_n_values': parse_param(top_n_values, int),
        'outlier_iqr_multiplier': parse_param(outlier_iqr_multiplier, float),
        'outlier_alert_threshold': parse_param(outlier_alert_threshold, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # If all parameters are provided, use them directly
    if all(value is not None for value in user_params.values()):
        config = user_params
    else:
        # Load defaults from config.json for missing parameters
        try:
            config_path = Path(__file__).parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)['UnifiedProfiler']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Override with user-provided parameters where present
        for key, value in user_params.items():
            if value is not None:
                config[key] = value
    
    return unified_profiler.profile_dataset(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['detect_drift'])
async def detect_drift_endpoint(current_file: UploadFile = File(...),baseline_file: UploadFile = File(...)):
    """Endpoint for the Drift Detector agent."""
    current_contents = await current_file.read()
    baseline_contents = await baseline_file.read()
    return drift_detector.detect_drift(
        baseline_contents=baseline_contents,
        current_contents=current_contents,
        baseline_filename=baseline_file.filename,
        current_filename=current_file.filename
    )


@router.post(AGENT_ROUTES['governance_checker'])
async def governance_checker_endpoint(
    file: UploadFile = File(...),
    compliance_threshold: Optional[str] = Form(None),
    needs_review_threshold: Optional[str] = Form(None),
    lineage_weight: Optional[str] = Form(None),
    consent_weight: Optional[str] = Form(None),
    classification_weight: Optional[str] = Form(None),
    lineage_null_threshold: Optional[str] = Form(None)
):
    """
    Endpoint for the GovernanceChecker agent.
    Accepts optional threshold parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'compliance_threshold': parse_param(compliance_threshold, int),
        'needs_review_threshold': parse_param(needs_review_threshold, int),
        'lineage_weight': parse_param(lineage_weight, float),
        'consent_weight': parse_param(consent_weight, float),
        'classification_weight': parse_param(classification_weight, float),
        'lineage_null_threshold': parse_param(lineage_null_threshold, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # If all parameters are provided, use them directly
    if all(value is not None for value in user_params.values()):
        config = user_params
    else:
        # Load defaults from config.json for missing parameters
        try:
            config_path = Path(__file__).parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)['GovernanceChecker']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Override with user-provided parameters where present
        for key, value in user_params.items():
            if value is not None:
                config[key] = value
    
    return governance_checker.check_governance(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['test_coverage_agent'])
async def test_coverage_agent_endpoint(
    file: UploadFile = File(...),
    excellent_threshold: Optional[str] = Form(None),
    good_threshold: Optional[str] = Form(None),
    uniqueness_weight: Optional[str] = Form(None),
    range_weight: Optional[str] = Form(None),
    format_weight: Optional[str] = Form(None)
):
    """
    Endpoint for the TestCoverageAgent.
    Accepts optional threshold parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
          None
    
    # Parse and validate parameters
    user_params = {
        'excellent_threshold': parse_param(excellent_threshold, int),
        'good_threshold': parse_param(good_threshold, int),
        'uniqueness_weight': parse_param(uniqueness_weight, float),
        'range_weight': parse_param(range_weight, float),
        'format_weight': parse_param(format_weight, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # Load defaults from config.json and override with user parameters
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)['TestCoverageAgent']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    
    # Override with user-provided parameters where present
    for key, value in user_params.items():
        if value is not None:
            config[key] = value
    
    return test_coverage_agent.check_test_coverage(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['schema_drift_reporter'])
async def schema_drift_reporter_endpoint(
    baseline_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    critical_drift_threshold: Optional[str] = Form(None),
    moderate_drift_threshold: Optional[str] = Form(None),
    added_columns_weight: Optional[str] = Form(None),
    removed_columns_weight: Optional[str] = Form(None),
    type_changes_weight: Optional[str] = Form(None)
):
    """
    Endpoint for the SchemaDriftReporter agent.
    Compares two datasets to detect schema changes and drift.
    """
    baseline_contents = await baseline_file.read()
    current_contents = await current_file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'critical_drift_threshold': parse_param(critical_drift_threshold, int),
        'moderate_drift_threshold': parse_param(moderate_drift_threshold, int),
        'added_columns_weight': parse_param(added_columns_weight, float),
        'removed_columns_weight': parse_param(removed_columns_weight, float),
        'type_changes_weight': parse_param(type_changes_weight, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # Load defaults from config.json and override with user parameters
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)['SchemaDriftReporter']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    
    # Override with user-provided parameters where present
    for key, value in user_params.items():
        if value is not None:
            config[key] = value
    
    return schema_drift_reporter.report_schema_drift(
        baseline_contents, current_contents, 
        baseline_file.filename, current_file.filename, 
        config, user_overrides
    )
@router.post(AGENT_ROUTES['null_handler'])
async def null_handler_endpoint(
    file: UploadFile = File(...),
    global_strategy: Optional[str] = Form(None),
    knn_neighbors: Optional[str] = Form(None),
    null_reduction_weight: Optional[str] = Form(None),
    data_retention_weight: Optional[str] = Form(None),
    column_retention_weight: Optional[str] = Form(None),
    excellent_threshold: Optional[str] = Form(None),
    good_threshold: Optional[str] = Form(None)
):
    """
    Endpoint for the NullHandler agent.
    Accepts optional configuration parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'global_strategy': global_strategy if global_strategy and global_strategy != "" else None,
        'knn_neighbors': parse_param(knn_neighbors, int),
        'null_reduction_weight': parse_param(null_reduction_weight, float),
        'data_retention_weight': parse_param(data_retention_weight, float),
        'column_retention_weight': parse_param(column_retention_weight, float),
        'excellent_threshold': parse_param(excellent_threshold, int),
        'good_threshold': parse_param(good_threshold, int)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # Load defaults from config.json and override with user parameters
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)['NullHandler']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    
    # Override with user-provided parameters where present
    for key, value in user_params.items():
        if value is not None:
            config[key] = value
    
    return null_handler.handle_nulls(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['outlier_remover'])
async def outlier_remover_endpoint(
    file: UploadFile = File(...),
    detection_method: Optional[str] = Form(None),
    removal_strategy: Optional[str] = Form(None),
    z_threshold: Optional[str] = Form(None),
    iqr_multiplier: Optional[str] = Form(None),
    lower_percentile: Optional[str] = Form(None),
    upper_percentile: Optional[str] = Form(None),
    outlier_reduction_weight: Optional[str] = Form(None),
    data_retention_weight: Optional[str] = Form(None),
    column_retention_weight: Optional[str] = Form(None),
    excellent_threshold: Optional[str] = Form(None),
    good_threshold: Optional[str] = Form(None)
):
    """
    Endpoint for the OutlierRemover agent.
    Accepts optional parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters
    user_params = {
        'detection_method': detection_method if detection_method and detection_method in ['z_score', 'iqr', 'percentile'] else None,
        'removal_strategy': removal_strategy if removal_strategy and removal_strategy in ['remove', 'impute_mean', 'impute_median'] else None,
        'z_threshold': parse_param(z_threshold, float),
        'iqr_multiplier': parse_param(iqr_multiplier, float),
        'lower_percentile': parse_param(lower_percentile, float),
        'upper_percentile': parse_param(upper_percentile, float),
        'outlier_reduction_weight': parse_param(outlier_reduction_weight, float),
        'data_retention_weight': parse_param(data_retention_weight, float),
        'column_retention_weight': parse_param(column_retention_weight, float),
        'excellent_threshold': parse_param(excellent_threshold, float),
        'good_threshold': parse_param(good_threshold, float)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # If all parameters are provided, use them directly
    if all(value is not None for value in user_params.values()):
        config = user_params
    else:
        # Load defaults from config.json for missing parameters
        try:
            config_path = Path(__file__).parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)['OutlierRemover']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        
        # Override with user-provided parameters where present
        for key, value in user_params.items():
            if value is not None:
                config[key] = value
    
    return outlier_remover.handle_outliers(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['type_fixer'])
async def type_fixer_endpoint(
    file: UploadFile = File(...),
    auto_convert_numeric: Optional[str] = Form(None),
    auto_convert_datetime: Optional[str] = Form(None),
    auto_convert_boolean: Optional[str] = Form(None),
    preserve_mixed_types: Optional[str] = Form(None)
):
    """
    Endpoint for the TypeFixer agent.
    Accepts only the core type conversion parameters that override config.json defaults.
    """
    contents = await file.read()
    
    # Helper function to convert form values (handles empty strings)
    def parse_param(value, param_type):
        if value is None or value == "":
            return None
        try:
            if param_type == bool:
                return value.lower() in ['true', '1', 'yes']
            return param_type(value)
        except (ValueError, TypeError):
            return None
    
    # Parse and validate parameters - only accept the 4 core parameters
    user_params = {
        'auto_convert_numeric': parse_param(auto_convert_numeric, bool),
        'auto_convert_datetime': parse_param(auto_convert_datetime, bool),
        'auto_convert_boolean': parse_param(auto_convert_boolean, bool),
        'preserve_mixed_types': parse_param(preserve_mixed_types, bool)
    }
    
    # Track which parameters were overridden by user (for audit trail)
    user_overrides = {k: v for k, v in user_params.items() if v is not None}
    
    # Load defaults from config.json and override with user parameters
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)['TypeFixer']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    
    # Override with user-provided parameters where present
    for key, value in user_params.items():
        if value is not None:
            config[key] = value
    
    return type_fixer.fix_types(contents, file.filename, config, user_overrides)

@router.post(AGENT_ROUTES['chat_with_data'])
def chat_with_data_endpoint(agent_report: str = Form(...), user_question: str = Form(...),history: str = Form(None)):
    """Endpoint for the Chat With Data agent."""
    try:
        # Since form data comes in as strings, we need to parse the JSON report string back into a dictionary
        report_dict = json.loads(agent_report)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for agent_report.")

    return chat_agent.answer_question_on_report(agent_report=report_dict, user_question=user_question,history=history)