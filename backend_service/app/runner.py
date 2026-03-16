from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable

from .pipeline_steps import (
    build_csv_from_events,
    extract_events_from_evidence,
    run_lloom_iterative,
    run_python_script,
    write_event_artifacts,
)
from .store import (
    append_log,
    get_selected_review_events,
    is_cancel_requested,
    load_job,
    mark_cancelled,
    mark_completed,
    mark_failed,
    mark_waiting_for_review,
    outputs_dir,
    save_review_events,
    update_step,
    uploads_dir,
)
from .settings import SETTINGS


class JobCancelledError(Exception):
    pass


def _copy_inputs(job_id: str, files: Iterable[Path]) -> list[Path]:
    destination = uploads_dir(job_id)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file_path in files:
        target = destination / file_path.name
        shutil.copy2(file_path, target)
        copied.append(target)
    return copied


def _prepare_output_evidence_dir(output_dir: Path, files: Iterable[Path]) -> list[Path]:
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file_path in files:
        target = evidence_dir / file_path.name
        shutil.copy2(file_path, target)
        copied.append(target)
    return copied


def _ensure_not_cancelled(job_id: str) -> None:
    if is_cancel_requested(job_id):
        mark_cancelled(job_id, "Job cancelled by user")
        raise JobCancelledError("Job cancelled by user")


def _run_with_retries(job_id: str, step_key: str, operation) -> None:
    retries = SETTINGS.step_retries.get(step_key, 1)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        _ensure_not_cancelled(job_id)
        try:
            operation()
            _ensure_not_cancelled(job_id)
            return
        except JobCancelledError:
            raise
        except Exception as error:
            last_error = error
            append_log(job_id, f"{step_key} attempt {attempt}/{retries} failed: {error}")
            if attempt < retries:
                time.sleep(2 * attempt)
    if last_error is not None:
        raise last_error


def _wait_for_review_submission(job_id: str) -> list[dict]:
    while True:
        _ensure_not_cancelled(job_id)
        record = load_job(job_id)
        if record.review_submitted:
            return get_selected_review_events(job_id)
        time.sleep(SETTINGS.review_poll_interval_seconds)


def run_pipeline(job_id: str, staged_input_files: list[Path]) -> None:
    try:
        _ensure_not_cancelled(job_id)
        root_dir = Path(__file__).resolve().parents[2]
        staged_files = _copy_inputs(job_id, staged_input_files)
        out_dir = outputs_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_files = _prepare_output_evidence_dir(out_dir, staged_files)

        update_step(
            job_id,
            step_key="validate_input",
            status="completed",
            message=f"Validated {len(input_files)} evidence file(s)",
            progress_percent=10,
        )

        update_step(
            job_id,
            step_key="extract_events",
            status="running",
            message="Extracting events from uploaded evidence",
            progress_percent=20,
        )
        extracted_events: list[dict] = []

        def do_extract() -> None:
            nonlocal extracted_events
            extracted_events = extract_events_from_evidence(
                root_dir=root_dir,
                evidence_files=input_files,
                output_dir=out_dir,
                log=lambda msg: append_log(job_id, msg),
            )

        _run_with_retries(job_id, "extract_events", do_extract)
        if not extracted_events:
            raise RuntimeError("No events were extracted from the uploaded evidence")
        save_review_events(job_id, extracted_events)
        update_step(
            job_id,
            step_key="extract_events",
            status="completed",
            message=f"Extracted {len(extracted_events)} event(s)",
            progress_percent=35,
        )

        update_step(
            job_id,
            step_key="review_events",
            status="running",
            message="Awaiting professor review of extracted events",
            progress_percent=42,
        )
        mark_waiting_for_review(job_id, "Awaiting professor review of extracted events")
        reviewed_events = _wait_for_review_submission(job_id)
        write_event_artifacts(out_dir, reviewed_events, log=lambda msg: append_log(job_id, msg))
        update_step(
            job_id,
            step_key="review_events",
            status="completed",
            message=f"Professor approved {len(reviewed_events)} extracted event(s)",
            progress_percent=48,
        )

        update_step(
            job_id,
            step_key="build_csv",
            status="running",
            message="Building evidence CSV files",
            progress_percent=55,
        )
        _run_with_retries(job_id, "build_csv", lambda: build_csv_from_events(out_dir, log=lambda msg: append_log(job_id, msg)))
        update_step(
            job_id,
            step_key="build_csv",
            status="completed",
            message="CSV artifacts generated",
            progress_percent=60,
        )

        update_step(
            job_id,
            step_key="lloom_scoring",
            status="running",
            message="Running LLooM scoring with iterative outlier reruns",
            progress_percent=70,
        )
        _run_with_retries(
            job_id,
            "lloom_scoring",
            lambda: run_lloom_iterative(
                root_dir=root_dir,
                output_dir=out_dir,
                log=lambda msg: append_log(job_id, msg),
                max_concepts=SETTINGS.lloom_max_concepts,
                max_iterations=SETTINGS.lloom_max_iterations,
                generic_coverage_threshold=SETTINGS.lloom_generic_coverage_threshold,
                mock_mode=SETTINGS.lloom_mock_mode,
            ),
        )
        update_step(
            job_id,
            step_key="lloom_scoring",
            status="completed",
            message="Scoring output generated",
            progress_percent=80,
        )

        update_step(
            job_id,
            step_key="synthesize_findings",
            status="running",
            message="Synthesizing findings",
            progress_percent=86,
        )
        _run_with_retries(
            job_id,
            "synthesize_findings",
            lambda: run_python_script(
                root_dir / "synthesize_findings.py",
                out_dir,
                log=lambda msg: append_log(job_id, msg),
                timeout_seconds=SETTINGS.script_timeouts_seconds["synthesize_findings"],
                cancel_check=lambda: is_cancel_requested(job_id),
            ),
        )
        update_step(
            job_id,
            step_key="synthesize_findings",
            status="completed",
            message="Findings generated",
            progress_percent=92,
        )

        update_step(
            job_id,
            step_key="build_mindmap",
            status="running",
            message="Building final mindmap HTML",
            progress_percent=97,
        )
        _run_with_retries(
            job_id,
            "build_mindmap",
            lambda: run_python_script(
                root_dir / "mindmap.py",
                out_dir,
                log=lambda msg: append_log(job_id, msg),
                timeout_seconds=SETTINGS.script_timeouts_seconds["build_mindmap"],
                cancel_check=lambda: is_cancel_requested(job_id),
            ),
        )
        update_step(
            job_id,
            step_key="build_mindmap",
            status="completed",
            message="Mindmap HTML generated",
            progress_percent=100,
        )

        mark_completed(job_id, "Pipeline completed successfully")
    except JobCancelledError:
        append_log(job_id, "Pipeline stopped due to cancellation")
    except Exception as error:
        append_log(job_id, f"Unhandled error: {error}")
        mark_failed(job_id, str(error))
