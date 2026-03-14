from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List

from .models import Artifact, JobRecord, JobStatus, StepProgress


BASE_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = BASE_DIR / "backend_service" / "runtime"
JOBS_DIR = RUNTIME_DIR / "jobs"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"

_write_lock = Lock()


DEFAULT_STEPS = [
    ("validate_input", "Validate and store input files"),
    ("extract_events", "Extract events"),
    ("build_csv", "Build CSV artifacts"),
    ("lloom_scoring", "Run LLooM scoring"),
    ("synthesize_findings", "Synthesize findings"),
    ("build_mindmap", "Build mindmap HTML"),
]


def ensure_runtime_dirs() -> None:
    for directory in (RUNTIME_DIR, JOBS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def job_log_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.log"


def outputs_dir(job_id: str) -> Path:
    return OUTPUTS_DIR / job_id


def uploads_dir(job_id: str) -> Path:
    return UPLOADS_DIR / job_id


def create_job_record(job_id: str, input_files: List[str]) -> JobRecord:
    ensure_runtime_dirs()
    now = utc_now_iso()
    record = JobRecord(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        status=JobStatus.queued,
        input_files=input_files,
        steps=[
            StepProgress(key=key, label=label, order=index + 1)
            for index, (key, label) in enumerate(DEFAULT_STEPS)
        ],
        artifacts=[
            Artifact(name="EVENTS.json", path="EVENTS.json"),
            Artifact(name="events.csv", path="events.csv"),
            Artifact(name="events_enriched.csv", path="events_enriched.csv"),
            Artifact(name="score_results_combined.csv", path="score_results_combined.csv"),
            Artifact(name="findings.json", path="findings.json"),
            Artifact(name="evidence_mindmap.html", path="evidence_mindmap.html"),
        ],
    )
    persist_job(record)
    append_log(job_id, "Job created")
    return record


def persist_job(record: JobRecord) -> None:
    destination = job_json_path(record.job_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record.updated_at = utc_now_iso()
    with _write_lock:
        destination.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_job(job_id: str) -> JobRecord:
    path = job_json_path(job_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobRecord(**data)


def list_jobs() -> List[JobRecord]:
    ensure_runtime_dirs()
    records: List[JobRecord] = []
    for directory in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        file_path = directory / "job.json"
        if not file_path.exists():
            continue
        data = json.loads(file_path.read_text(encoding="utf-8"))
        records.append(JobRecord(**data))
    return records


def append_log(job_id: str, message: str) -> None:
    log_path = job_log_path(job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_iso()
    with _write_lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")


def read_logs(job_id: str) -> str:
    path = job_log_path(job_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def update_step(
    job_id: str,
    *,
    step_key: str,
    status: str,
    message: str,
    progress_percent: int,
) -> JobRecord:
    record = load_job(job_id)
    for step in record.steps:
        if step.key == step_key:
            step.status = status
            record.current_step = step.key
            record.step_index = step.order
            break
    record.progress_percent = progress_percent
    record.message = message
    if record.status == JobStatus.queued:
        record.status = JobStatus.running
    persist_job(record)
    append_log(job_id, message)
    return record


def request_cancel(job_id: str) -> JobRecord:
    record = load_job(job_id)
    record.cancel_requested = True
    if record.status == JobStatus.queued:
        record.status = JobStatus.cancelled
        record.message = "Job cancelled before execution"
    else:
        record.message = "Cancellation requested"
    persist_job(record)
    append_log(job_id, "Cancellation requested")
    return record


def is_cancel_requested(job_id: str) -> bool:
    record = load_job(job_id)
    return bool(record.cancel_requested)


def mark_cancelled(job_id: str, message: str = "Job cancelled") -> JobRecord:
    record = load_job(job_id)
    record.status = JobStatus.cancelled
    record.message = message
    record.current_step = "cancelled"
    persist_job(record)
    append_log(job_id, message)
    return record


def cleanup_old_runtime_data(retention_hours: int = 72) -> int:
    ensure_runtime_dirs()
    now = datetime.now(timezone.utc).timestamp()
    ttl_seconds = retention_hours * 3600
    removed = 0

    for folder in [JOBS_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
        for child in folder.iterdir():
            if not child.is_dir():
                continue
            age_seconds = now - child.stat().st_mtime
            if age_seconds > ttl_seconds:
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink(missing_ok=True)
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
                removed += 1

    return removed


def mark_completed(job_id: str, message: str) -> JobRecord:
    record = load_job(job_id)
    record.status = JobStatus.completed
    record.current_step = "done"
    record.step_index = record.total_steps
    record.progress_percent = 100
    record.message = message
    output_folder = outputs_dir(job_id)
    for artifact in record.artifacts:
        artifact.exists = (output_folder / artifact.path).exists()
    persist_job(record)
    append_log(job_id, message)
    return record


def mark_failed(job_id: str, error_message: str) -> JobRecord:
    record = load_job(job_id)
    record.status = JobStatus.failed
    record.error = error_message
    record.message = error_message
    persist_job(record)
    append_log(job_id, f"FAILED: {error_message}")
    return record
