from __future__ import annotations

import shutil
import uuid
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .models import JobCreateResponse, JobListResponse
from .runner import run_pipeline
from .store import (
    cleanup_old_runtime_data,
    create_job_record,
    ensure_runtime_dirs,
    list_jobs,
    load_job,
    outputs_dir,
    request_cancel,
    read_logs,
    uploads_dir,
)


app = FastAPI(title="FYP Event Extraction Backend", version="0.1.0")


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@app.on_event("startup")
def startup() -> None:
    ensure_runtime_dirs()
    retention_hours = int(os.getenv("BACKEND_RETENTION_HOURS", "72"))
    removed = cleanup_old_runtime_data(retention_hours=retention_hours)
    if removed:
        print(f"[startup] Cleaned {removed} old runtime folder(s) older than {retention_hours}h")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index():
    ui_path = ROOT_DIR / "backend_service" / "app" / "static" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(ui_path, media_type="text/html")


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)) -> JobCreateResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one evidence .txt file is required")

    invalid = [file.filename for file in files if not file.filename or not file.filename.lower().endswith(".txt")]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Only .txt files are supported for now. Invalid: {invalid}")

    job_id = str(uuid.uuid4())
    staging_dir = uploads_dir(job_id) / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for upload in files:
        target_path = staging_dir / upload.filename
        with target_path.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        saved_paths.append(target_path)

    create_job_record(job_id, [p.name for p in saved_paths])
    background_tasks.add_task(run_pipeline, job_id, saved_paths)

    return JobCreateResponse(job_id=job_id, status="queued")


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    if job.status in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": job.status, "message": "Job already finished"}

    updated = request_cancel(job_id)
    return {
        "job_id": updated.job_id,
        "status": updated.status,
        "cancel_requested": updated.cancel_requested,
        "message": updated.message,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error


@app.get("/jobs", response_model=JobListResponse)
def get_jobs() -> JobListResponse:
    return JobListResponse(jobs=list_jobs())


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        _ = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    return PlainTextResponse(read_logs(job_id))


@app.get("/jobs/{job_id}/artifacts")
def get_artifacts(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    base_output = outputs_dir(job_id)
    artifacts = []
    for artifact in job.artifacts:
        full_path = base_output / artifact.path
        artifacts.append(
            {
                "name": artifact.name,
                "path": artifact.path,
                "exists": full_path.exists(),
                "download_url": f"/jobs/{job_id}/artifacts/{artifact.path}" if full_path.exists() else None,
            }
        )
    return {"job_id": job_id, "artifacts": artifacts}


@app.get("/jobs/{job_id}/artifacts/{artifact_name:path}")
def download_artifact(job_id: str, artifact_name: str):
    target = outputs_dir(job_id) / artifact_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.get("/jobs/{job_id}/mindmap")
def get_mindmap(job_id: str):
    target = outputs_dir(job_id) / "evidence_mindmap.html"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Mindmap not ready")
    return FileResponse(target, media_type="text/html")
