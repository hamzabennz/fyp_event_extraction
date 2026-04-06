from __future__ import annotations

import shutil
import uuid
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .models import EventReviewSubmitRequest, EventTypeDefinition, EventTypeListResponse, EventTypeUpsertRequest, JobCreateResponse, JobListResponse
from .runner import run_pipeline
from .store import (
    cleanup_old_runtime_data,
    create_job_record,
    delete_event_type,
    delete_job,
    ensure_runtime_dirs,
    load_event_schema,
    list_jobs,
    load_job,
    outputs_dir,
    request_cancel,
    read_logs,
    submit_review_selection,
    upsert_event_type,
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


@app.delete("/jobs/{job_id}")
def remove_job(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    delete_job(job_id)
    return {"job_id": job_id, "deleted": True}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error


@app.get("/jobs", response_model=JobListResponse)
def get_jobs() -> JobListResponse:
    return JobListResponse(jobs=list_jobs())


@app.get("/event-types", response_model=EventTypeListResponse)
def get_event_types() -> EventTypeListResponse:
    return EventTypeListResponse(event_types=load_event_schema())


@app.post("/event-types", response_model=EventTypeListResponse)
def create_event_type(payload: EventTypeUpsertRequest) -> EventTypeListResponse:
    key = payload.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Event type key is required")
    schema = upsert_event_type(
        key,
        EventTypeDefinition(description=payload.description.strip(), specific_fields=payload.specific_fields),
    )
    return EventTypeListResponse(event_types=schema)


@app.put("/event-types/{event_key}", response_model=EventTypeListResponse)
def update_event_type(event_key: str, payload: EventTypeUpsertRequest) -> EventTypeListResponse:
    normalized_key = event_key.strip()
    if not normalized_key:
        raise HTTPException(status_code=400, detail="Event type key is required")
    if normalized_key != payload.key.strip():
        raise HTTPException(status_code=400, detail="Renaming event type keys is not supported from this screen")
    schema = upsert_event_type(
        normalized_key,
        EventTypeDefinition(description=payload.description.strip(), specific_fields=payload.specific_fields),
    )
    return EventTypeListResponse(event_types=schema)


@app.delete("/event-types/{event_key}", response_model=EventTypeListResponse)
def remove_event_type(event_key: str) -> EventTypeListResponse:
    try:
        schema = delete_event_type(event_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Event type not found") from error
    return EventTypeListResponse(event_types=schema)


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        _ = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    return PlainTextResponse(read_logs(job_id))


@app.get("/jobs/{job_id}/review-events")
def get_review_events(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    return {
        "job_id": job.job_id,
        "status": job.status,
        "review_required": job.review_required,
        "review_submitted": job.review_submitted,
        "events": [event.model_dump() for event in job.review_events],
    }


@app.post("/jobs/{job_id}/review-events")
def submit_review(job_id: str, payload: EventReviewSubmitRequest):
    try:
        updated = submit_review_selection(job_id, payload.selected_ids)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "job_id": updated.job_id,
        "status": updated.status,
        "review_required": updated.review_required,
        "review_submitted": updated.review_submitted,
        "message": updated.message,
    }


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


# ── Knowledge Graph routes ────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/graph-data")
def get_graph_data(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    from .graph_builder import _get_driver, get_graph_data_for_job
    driver = _get_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")
    return get_graph_data_for_job(job_id, driver=driver)


@app.get("/jobs/{job_id}/contradictions")
def get_contradictions(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    from .graph_builder import _get_driver
    driver = _get_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")

    with driver.session() as session:
        rows = session.run(
            """
            MATCH (e1:Event {job_id: $job_id})-[r:CONTRADICTS]->(e2:Event {job_id: $job_id})
            RETURN e1.id AS event1_id, e2.id AS event2_id,
                   r.type AS type, r.description AS description, r.severity AS severity
            """,
            job_id=job_id,
        ).data()

    return {"job_id": job_id, "contradictions": rows}


@app.post("/jobs/{job_id}/graph-data/refresh")
def refresh_graph(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    from .graph_builder import _get_driver, clear_job_graph, insert_job_events, reset_driver
    from .contradiction_detector import detect_and_store_contradictions
    from .store import get_selected_review_events

    reset_driver()  # force reconnect in case Neo4j restarted
    driver = _get_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j unavailable")

    clear_job_graph(job_id, driver=driver)
    events = get_selected_review_events(job_id)
    insert_job_events(job_id, events, driver=driver)
    contradictions = detect_and_store_contradictions(job_id)
    return {"job_id": job_id, "events_inserted": len(events), "contradictions_found": len(contradictions)}


@app.get("/graph/{job_id}", response_class=HTMLResponse)
def get_graph_ui(job_id: str):
    try:
        load_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found") from error

    graph_html_path = ROOT_DIR / "backend_service" / "app" / "static" / "graph.html"
    if not graph_html_path.exists():
        raise HTTPException(status_code=404, detail="Graph UI not found")
    return HTMLResponse(content=graph_html_path.read_text(encoding="utf-8"))
