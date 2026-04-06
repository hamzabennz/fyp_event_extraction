from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _run_step(
    job_id: str,
    step_key: str,
    run_message: str,
    done_message: str,
    run_pct: int,
    done_pct: int,
    operation,
) -> None:
    update_step(job_id, step_key=step_key, status="running", message=run_message, progress_percent=run_pct)
    _run_with_retries(job_id, step_key, operation)
    update_step(job_id, step_key=step_key, status="completed", message=done_message, progress_percent=done_pct)


def _wait_for_review_submission(job_id: str) -> list[dict]:
    while True:
        _ensure_not_cancelled(job_id)
        record = load_job(job_id)
        if record.review_submitted:
            return get_selected_review_events(job_id)
        time.sleep(SETTINGS.review_poll_interval_seconds)


def _run_parallel_steps(
    job_id: str,
    out_dir: Path,
    reviewed_events: list[dict],
    root_dir: Path,
) -> None:
    """
    Run build_knowledge_graph and lloom_scoring in parallel using threads.

    Graph failures are NON-FATAL: log, mark step failed, continue.
    LLooM failures ARE fatal: re-raise after both threads complete.
    """
    from .graph_builder import insert_job_events
    from .contradiction_detector import detect_and_store_contradictions

    update_step(job_id, step_key="build_knowledge_graph", status="running",
                message="Building Neo4j knowledge graph", progress_percent=65)
    update_step(job_id, step_key="lloom_scoring", status="running",
                message="Running LLooM scoring with iterative outlier reruns", progress_percent=65)

    graph_error: Exception | None = None
    lloom_error: Exception | None = None

    def do_graph() -> None:
        nonlocal graph_error
        try:
            insert_job_events(job_id, reviewed_events)
            detect_and_store_contradictions(job_id)
            update_step(job_id, step_key="build_knowledge_graph", status="completed",
                        message="Knowledge graph built", progress_percent=75)
        except Exception as exc:
            graph_error = exc
            append_log(job_id, f"build_knowledge_graph failed (non-fatal): {exc}")
            update_step(job_id, step_key="build_knowledge_graph", status="failed",
                        message=f"Graph build failed (non-fatal): {exc}", progress_percent=75)

    def do_lloom() -> None:
        nonlocal lloom_error
        try:
            run_lloom_iterative(
                root_dir=root_dir,
                output_dir=out_dir,
                log=lambda msg: append_log(job_id, msg),
                max_concepts=SETTINGS.lloom_max_concepts,
                max_iterations=SETTINGS.lloom_max_iterations,
                generic_coverage_threshold=SETTINGS.lloom_generic_coverage_threshold,
                mock_mode=SETTINGS.lloom_mock_mode,
            )
            update_step(job_id, step_key="lloom_scoring", status="completed",
                        message="Scoring output generated", progress_percent=80)
        except Exception as exc:
            lloom_error = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(do_graph), pool.submit(do_lloom)]
        for future in as_completed(futures):
            future.result()  # surface unexpected thread exceptions

    if lloom_error is not None:
        raise lloom_error


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

        extracted_events: list[dict] = []

        def do_extract() -> None:
            nonlocal extracted_events
            extracted_events = extract_events_from_evidence(
                root_dir=root_dir,
                evidence_files=input_files,
                output_dir=out_dir,
                log=lambda msg: append_log(job_id, msg),
            )

        _run_step(
            job_id, "extract_events",
            "Extracting events from uploaded evidence",
            f"Extracted {len(extracted_events)} event(s)",
            20, 35,
            do_extract,
        )
        if not extracted_events:
            raise RuntimeError("No events were extracted from the uploaded evidence")
        save_review_events(job_id, extracted_events)

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

        _run_step(
            job_id, "build_csv",
            "Building evidence CSV files",
            "CSV artifacts generated",
            55, 60,
            lambda: build_csv_from_events(out_dir, log=lambda msg: append_log(job_id, msg)),
        )

        _ensure_not_cancelled(job_id)
        _run_parallel_steps(job_id, out_dir, reviewed_events, root_dir)

        _run_step(
            job_id, "synthesize_findings",
            "Synthesizing findings",
            "Findings generated",
            86, 92,
            lambda: run_python_script(
                root_dir / "synthesize_findings.py",
                out_dir,
                log=lambda msg: append_log(job_id, msg),
                timeout_seconds=SETTINGS.script_timeouts_seconds["synthesize_findings"],
                cancel_check=lambda: is_cancel_requested(job_id),
            ),
        )

        _run_step(
            job_id, "build_mindmap",
            "Building final mindmap HTML",
            "Mindmap HTML generated",
            97, 100,
            lambda: run_python_script(
                root_dir / "mindmap.py",
                out_dir,
                log=lambda msg: append_log(job_id, msg),
                timeout_seconds=SETTINGS.script_timeouts_seconds["build_mindmap"],
                cancel_check=lambda: is_cancel_requested(job_id),
            ),
        )

        mark_completed(job_id, "Pipeline completed successfully")
    except JobCancelledError:
        append_log(job_id, "Pipeline stopped due to cancellation")
    except Exception as error:
        append_log(job_id, f"Unhandled error: {error}")
        mark_failed(job_id, str(error))
