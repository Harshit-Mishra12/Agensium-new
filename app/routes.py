from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends,Body
from typing import Optional, List, Any
from app.config import AGENT_ROUTES
from app.agents.source import (
    schema_scanner, 
    readiness_rater,
    risk_scorer,
    field_profiler, 
    drift_detector, 
    dedup_agent,
    source_tagger,
    entity_resolver,
    compliance_tagger
)
from app.agents.shared import chat_agent
from app.agents import unified_profiler
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


@router.post(AGENT_ROUTES['master_my_data_tool'])
async def master_my_data_endpoint(
    current_file: UploadFile = File(...),
    source_system: str = Form(...),
    key_column: str = Form(None)
):
    """
    Orchestrator endpoint for the 'Master My Data' tool.
    Runs the full data mastering workflow: source tagging, entity resolution, 
    deduplication, and compliance checking.
    """
    # Input validation
    if not source_system or not source_system.strip():
        raise HTTPException(status_code=400, detail="source_system parameter is required.")
    
    # 1. Prepare the files dictionary
    files = {
        "current_file": {
            "contents": await current_file.read(), 
            "filename": current_file.filename
        }
    }
    
    # 2. Prepare parameters for agents
    params = {
        "source_system": source_system.strip(),
        "key_column": key_column.strip() if key_column else None
    }
    
    # 3. Call the Master My Data workflow engine
    return run_master_workflow(
        workflow_def=MASTER_MY_DATA_WORKFLOW,
        files=files,
        params=params
    )


# --- Individual Agent Endpoints (The "Expert's Toolbox") ---

@router.post(AGENT_ROUTES['scan_schema'])
async def scan_schema_endpoint(file: UploadFile = File(...)):
    """Endpoint for the Schema Scanner agent."""
    contents = await file.read()
    return schema_scanner.scan_schema(contents, file.filename)

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
    Endpoint for the RiskScorer agent.
    Accepts optional threshold parameters that override config.json defaults.
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

@router.post(AGENT_ROUTES['profile_fields'])
async def profile_fields_endpoint(file: UploadFile = File(...)):
    """Endpoint for the Field Profiler agent."""
    contents = await file.read()
    return field_profiler.profile_fields(contents, file.filename)

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

@router.post(AGENT_ROUTES['chat_with_data'])
def chat_with_data_endpoint(agent_report: str = Form(...), user_question: str = Form(...),history: str = Form(None)):
    """Endpoint for the Chat With Data agent."""
    try:
        # Since form data comes in as strings, we need to parse the JSON report string back into a dictionary
        report_dict = json.loads(agent_report)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for agent_report.")

    return chat_agent.answer_question_on_report(agent_report=report_dict, user_question=user_question,history=history)


# --- Master My Data Agents ---

@router.post(AGENT_ROUTES['source_tagger'])
async def source_tagger_endpoint(file: UploadFile = File(...),source_system: str = Form(...)):
    """
    Endpoint for the Source Tagger agent.
    Tags all rows with a source system identifier for data lineage.
    """
    # Input validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    
    if not source_system or not source_system.strip():
        raise HTTPException(status_code=400, detail="source_system parameter is required and cannot be empty.")
    
    contents = await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")
    
    return source_tagger.tag_source(contents, file.filename, source_system.strip())


@router.post(AGENT_ROUTES['entity_resolver'])
async def entity_resolver_endpoint(
    file: UploadFile = File(...),
    key_column: str = Form(...)
):
    """
    Endpoint for the Entity Resolver agent.
    Identifies potential duplicate entities based on a key column.
    """
    # Input validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    
    if not key_column or not key_column.strip():
        raise HTTPException(status_code=400, detail="key_column parameter is required and cannot be empty.")
    
    contents = await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")
    
    return entity_resolver.resolve_entities(contents, file.filename, key_column.strip())


@router.post(AGENT_ROUTES['deduplicate'])
async def deduplicate_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Deduplication agent.
    Identifies exact full-row duplicates in the dataset.
    """
    # Input validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    
    contents = await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")
    
    return dedup_agent.deduplicate(contents, file.filename)


@router.post(AGENT_ROUTES['compliance_tagger'])
async def compliance_tagger_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Compliance Tagger agent.
    Scans dataset schema to flag potential PII fields.
    """
    # Input validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    
    contents = await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")
    
    return compliance_tagger.tag_compliance(contents, file.filename)
