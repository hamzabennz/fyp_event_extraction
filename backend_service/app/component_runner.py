from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline_steps import (
    build_csv_from_events,
    extract_events_from_evidence,
    run_lloom_iterative,
    run_python_script,
)
from .settings import SETTINGS


def _log(message: str) -> None:
    print(message)


def run_component(component: str, root_dir: Path, output_dir: Path, evidence_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if component == "extract":
        if evidence_dir is None:
            raise ValueError("--evidence-dir is required for extract")
        evidence_files = sorted([p for p in evidence_dir.glob("*.txt") if p.is_file()])
        if not evidence_files:
            raise ValueError(f"No .txt files found in {evidence_dir}")
        local_evidence_dir = output_dir / "evidence"
        local_evidence_dir.mkdir(parents=True, exist_ok=True)
        staged_files = []
        for path in evidence_files:
            target = local_evidence_dir / path.name
            target.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            staged_files.append(target)
        extract_events_from_evidence(
            root_dir=root_dir,
            evidence_files=staged_files,
            output_dir=output_dir,
            log=_log,
        )
        return

    if component == "csv":
        build_csv_from_events(output_dir, log=_log)
        return

    if component == "lloom":
        run_lloom_iterative(
            root_dir=root_dir,
            output_dir=output_dir,
            log=_log,
            max_concepts=SETTINGS.lloom_max_concepts,
            max_iterations=SETTINGS.lloom_max_iterations,
            generic_coverage_threshold=SETTINGS.lloom_generic_coverage_threshold,
        )
        return

    if component == "synth":
        run_python_script(
            root_dir / "synthesize_findings.py",
            output_dir,
            log=_log,
            timeout_seconds=SETTINGS.script_timeouts_seconds["synthesize_findings"],
        )
        return

    if component == "mindmap":
        run_python_script(
            root_dir / "mindmap.py",
            output_dir,
            log=_log,
            timeout_seconds=SETTINGS.script_timeouts_seconds["build_mindmap"],
        )
        return

    if component == "all":
        if evidence_dir is None:
            raise ValueError("--evidence-dir is required for all")
        run_component("extract", root_dir, output_dir, evidence_dir)
        run_component("csv", root_dir, output_dir, None)
        run_component("lloom", root_dir, output_dir, None)
        run_component("synth", root_dir, output_dir, None)
        run_component("mindmap", root_dir, output_dir, None)
        return

    raise ValueError(f"Unknown component: {component}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend pipeline components in isolation")
    parser.add_argument(
        "--component",
        choices=["extract", "csv", "lloom", "synth", "mindmap", "all"],
        required=True,
        help="Which component to execute",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Job-like output directory where artifacts are read/written",
    )
    parser.add_argument(
        "--evidence-dir",
        required=False,
        help="Directory containing evidence .txt files (required for extract/all)",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else None

    run_component(args.component, root_dir=root_dir, output_dir=output_dir, evidence_dir=evidence_dir)


if __name__ == "__main__":
    main()
