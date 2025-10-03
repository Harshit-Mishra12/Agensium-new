from fastapi import APIRouter, UploadFile, File, HTTPException
from app.agents.source import schema_scanner, readiness_rater, field_profiler, drift_detector, dedup_agent
from app.config import AGENT_ROUTES

router = APIRouter()
SUPPORTED_FILE_EXTENSIONS = {"csv", "xlsx", "xls", "json", "sql"}

@router.post(AGENT_ROUTES['scan_schema'])
async def scan_schema_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Schema Scanner agent.
    Accepts CSV, Excel, JSON, and SQL files.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )
    contents = await file.read()
    return schema_scanner.scan_schema(contents, file.filename)

@router.post(AGENT_ROUTES['rate_readiness'])
async def rate_readiness_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Readiness Rater agent.
    Accepts CSV, Excel, JSON, and Parquet files.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )
    contents = await file.read()
    return readiness_rater.rate_readiness(contents, file.filename)

@router.post(AGENT_ROUTES['profile_fields'])
async def profile_fields_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Field Profiler agent.
    Accepts CSV and Excel files.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )
    contents = await file.read()
    return field_profiler.profile_fields(contents, file.filename)

@router.post(AGENT_ROUTES['detect_drift'])
async def detect_drift_endpoint(baseline_file: UploadFile = File(...), current_file: UploadFile = File(...)):
    """
    Endpoint for the Drift Detector agent. Requires two CSV file uploads.
    """
    baseline_file_extension = baseline_file.filename.split('.')[-1].lower()
    current_file_extension = current_file.filename.split('.')[-1].lower()
    if baseline_file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
    if current_file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
    
    baseline_contents = await baseline_file.read()
    current_contents = await current_file.read()
    # Using one filename as a representative name for the analysis
    return drift_detector.detect_drift(baseline_contents, current_contents, baseline_file.filename,current_file.filename)

