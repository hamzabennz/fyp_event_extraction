from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineSettings:
    step_retries: dict[str, int] = field(
        default_factory=lambda: {
            "extract_events": 2,
            "build_csv": 2,
            "lloom_scoring": 2,
            "synthesize_findings": 2,
            "build_mindmap": 2,
        }
    )
    script_timeouts_seconds: dict[str, int] = field(
        default_factory=lambda: {
            "synthesize_findings": 900,
            "build_mindmap": 900,
        }
    )
    lloom_max_concepts: int = 5
    lloom_max_iterations: int = 3
    lloom_generic_coverage_threshold: float = 0.5


SETTINGS = PipelineSettings()
