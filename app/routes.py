from fastapi import APIRouter, UploadFile, File, HTTPException
from app.agents.source import schema_scanner, dedup_agent, field_profiler, drift_detector, readiness_rater
from app.orchestrator import workflow
import os
import shutil
from tempfile import NamedTemporaryFile

router = APIRouter()
SUPPORTED_FILE_EXTENSIONS = {"csv", "xlsx", "xls", "json", "sql"}


# --- Schema scanner endpoint ---
@router.post("/scan-schema")
async def scan_schema(file: UploadFile = File(...)):
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types are: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )
    try:
        contents = await file.read()
        return schema_scanner.scan_schema(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# --- Field profiler endpoint (CSV, Excel, JSON, SQL) ---
@router.post("/field-profiler")
async def profile_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename

        if filename.lower().endswith(".csv"):
            return field_profiler.profile_csv(contents, filename)

        elif filename.lower().endswith((".xlsx", ".xls")):
            return field_profiler.profile_excel(contents, filename)

        elif filename.lower().endswith(".json"):
            return field_profiler.profile_json(contents, filename)

        elif filename.lower().endswith(".sql"):
            return field_profiler.profile_sql(contents, filename)

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload CSV, Excel, JSON, or SQL."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# --- Readiness rater endpoint ---
@router.post("/rate-readiness")
async def rate_readiness_endpoint(file: UploadFile = File(...)):
    """
    Calculates the readiness score for a dataset from an uploaded file.
    Supported file types: CSV, Excel, JSON, SQL.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format for readiness rating: {file_extension}"
        )

    try:
        contents = await file.read()
        return readiness_rater.rate_readiness(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rating readiness: {e}")


# --- Deduplicate endpoint ---
@router.post("/deduplicate")
def deduplicate(items: list[str]):
    return dedup_agent.deduplicate(items)


# --- Orchestrator endpoint ---
@router.post("/run-workflow")
def run_workflow(dataset: list[dict], items: list[str]):
    return workflow(dataset, items)


# --- Drift Detector endpoint (CSV, Excel, JSON, SQL) ---
@router.post("/detect-drift")
async def detect_drift(
    baseline_file: UploadFile = File(...),
    current_file: UploadFile = File(...)
):
    """
    Detect drift between baseline and current datasets (CSV, Excel, JSON, SQL).
    Returns JSON report with schema and data drift.
    """
    valid_ext = (".csv", ".xlsx", ".xls", ".json", ".sql")
    if not baseline_file.filename.lower().endswith(valid_ext) or not current_file.filename.lower().endswith(valid_ext):
        raise HTTPException(
            status_code=400,
            detail="Files must be of the same supported type (CSV, Excel, JSON, SQL)."
        )

    try:
        # Save baseline temp file
        base_suffix = os.path.splitext(baseline_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=base_suffix) as tmp_base:
            shutil.copyfileobj(baseline_file.file, tmp_base)
            baseline_path = tmp_base.name

        # Save current temp file
        curr_suffix = os.path.splitext(current_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=curr_suffix) as tmp_curr:
            shutil.copyfileobj(current_file.file, tmp_curr)
            current_path = tmp_curr.name

        # ✅ Pass original names so DriftDetector can use them
        report = drift_detector.DriftDetector.detect_drift(
            baseline_path, current_path,
            baseline_name=baseline_file.filename,
            current_name=current_file.filename
        )
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting drift: {e}")
    finally:
        # Cleanup temp files
        try:
            os.remove(baseline_path)
            os.remove(current_path)
        except Exception:
            pass
