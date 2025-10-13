from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends,Body
from typing import Optional, List, Any
from app.config import AGENT_ROUTES
from app.agents.source import (
    schema_scanner, 
    readiness_rater, 
    field_profiler, 
    drift_detector, 
    dedup_agent,
    source_tagger,
    entity_resolver,
    compliance_tagger
)
from app.agents.shared import chat_agent
import json

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
async def rate_readiness_endpoint(file: UploadFile = File(...)):
    """Endpoint for the Readiness Rater agent."""
    contents = await file.read()
    return readiness_rater.rate_readiness(contents, file.filename)

@router.post(AGENT_ROUTES['profile_fields'])
async def profile_fields_endpoint(file: UploadFile = File(...)):
    """Endpoint for the Field Profiler agent."""
    contents = await file.read()
    return field_profiler.profile_fields(contents, file.filename)

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
