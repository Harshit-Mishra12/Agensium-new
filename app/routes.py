from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List, Any
from app.config import AGENT_ROUTES
from app.agents.source import schema_scanner, readiness_rater, field_profiler, drift_detector, dedup_agent

# --- New imports for the Smart Orchestration Engine ---
from app.tools.profile_my_data_orchestrator import run_workflow
from app.tools.workflow_definitions import PROFILE_MY_DATA_WORKFLOW


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
async def detect_drift_endpoint(
    current_file: UploadFile = File(...),
    baseline_file: UploadFile = File(...)
):
    """Endpoint for the Drift Detector agent."""
    current_contents = await current_file.read()
    baseline_contents = await baseline_file.read()
    return drift_detector.detect_drift(
        baseline_contents=baseline_contents,
        current_contents=current_contents,
        baseline_filename=baseline_file.filename,
        current_filename=current_file.filename
    )