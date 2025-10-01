from fastapi import APIRouter, UploadFile, File, HTTPException
from app.agents.source import schema_scanner, readiness_rater, field_profiler, drift_detector, dedup_agent

router = APIRouter()

@router.post("/scan-schema")
async def scan_schema_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Schema Scanner agent.
    Accepts CSV, Excel, JSON, and SQL files.
    """
    supported_formats = ('.csv', '.xlsx', '.xls', '.json', '.sql')
    if not file.filename.endswith(supported_formats):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    contents = await file.read()
    return schema_scanner.scan_schema(contents, file.filename)

@router.post("/rate-readiness")
async def rate_readiness_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Readiness Rater agent.
    Accepts CSV, Excel, JSON, and Parquet files.
    """
    supported_formats = ('.csv', '.xlsx', '.xls', '.json', '.parquet', '.sql')
    if not file.filename.endswith(supported_formats):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    contents = await file.read()
    return readiness_rater.rate_readiness(contents, file.filename)

@router.post("/profile-fields")
async def profile_fields_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for the Field Profiler agent.
    Accepts CSV and Excel files.
    """
    supported_formats = ('.csv', '.xlsx', '.xls')
    if not file.filename.endswith(supported_formats):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    contents = await file.read()
    return field_profiler.profile_fields(contents, file.filename)

@router.post("/detect-drift")
async def detect_drift_endpoint(baseline_file: UploadFile = File(...), current_file: UploadFile = File(...)):
    """
    Endpoint for the Drift Detector agent. Requires two CSV file uploads.
    """
    if not baseline_file.filename.endswith('.csv') or not current_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Drift detection currently supports CSV files only.")
    
    baseline_contents = await baseline_file.read()
    current_contents = await current_file.read()
    # Using one filename as a representative name for the analysis
    return drift_detector.detect_drift(baseline_contents, current_contents, baseline_file.filename)

@router.post("/deduplicate")
def deduplicate_endpoint(items: list[str]):
    """
    Simple endpoint for the deduplication agent.
    """
    return dedup_agent.deduplicate(items)

