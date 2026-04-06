"""
Thin adapter that maps the old SETTINGS.* attribute names used inside
backend_service/ to the canonical CONFIG values defined in config.py.

All tunable parameters live in config.py — edit them there.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so config.py can be found regardless of how
# the backend is started (uvicorn, uv run, direct python, tests, etc.)
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from config import CONFIG  # noqa: E402  (import after sys.path tweak)


class _SettingsAdapter:
    """
    Read-only view of CONFIG that preserves the SETTINGS.xyz attribute names
    already used throughout backend_service/app/.
    """

    # ── pipeline step retries ─────────────────────────────────────────────────
    @property
    def step_retries(self) -> dict[str, int]:
        n = CONFIG.pipeline.step_retry_count
        return {
            "extract_events":         n,
            "build_csv":              n,
            "build_knowledge_graph":  1,
            "lloom_scoring":          n,
            "synthesize_findings":    n,
            "build_mindmap":          n,
        }

    @property
    def script_timeouts_seconds(self) -> dict[str, int]:
        t = CONFIG.pipeline.subprocess_timeout_seconds
        return {
            "synthesize_findings": t,
            "build_mindmap":       t,
        }

    # ── LLooM ─────────────────────────────────────────────────────────────────
    @property
    def lloom_max_concepts(self) -> int:
        return CONFIG.lloom.max_concepts

    @property
    def lloom_max_iterations(self) -> int:
        return CONFIG.lloom.max_rerun_iterations

    @property
    def lloom_generic_coverage_threshold(self) -> float:
        return CONFIG.lloom.generic_concept_max_coverage_fraction

    @property
    def lloom_mock_mode(self) -> bool:
        return CONFIG.lloom.use_mock_scoring

    # ── job lifecycle ─────────────────────────────────────────────────────────
    @property
    def review_poll_interval_seconds(self) -> int:
        return CONFIG.pipeline.review_poll_interval_seconds

    # ── score thresholds ─────────────────────────────────────────────────────
    @property
    def synthesis_score_threshold(self) -> float:
        return CONFIG.scores.min_score_to_include_in_synthesis

    @property
    def visualization_score_threshold(self) -> float:
        return CONFIG.scores.min_score_to_show_in_mindmap


SETTINGS = _SettingsAdapter()
