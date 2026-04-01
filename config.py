"""
Pipeline configuration — single source of truth for every tunable parameter.

Usage
-----
    from config import CONFIG

    # read a value
    CONFIG.models.extraction_model
    CONFIG.lloom.max_concepts
    CONFIG.scores.min_score_to_include_in_synthesis
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Gemini model names
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeminiModels:
    # Model used in stage 1 to extract events from raw evidence text.
    extraction_model: str = "gemini-2.0-flash"

    # Model used by LLooM to distil bullet-point summaries from event narratives.
    lloom_distill_model: str = "gemini-2.0-flash"

    # Model used by LLooM to synthesise a concept label from a cluster of bullets.
    lloom_concept_synthesis_model: str = "gemini-2.0-flash"

    # Model used by LLooM to score each event against every concept.
    # Intentionally more capable than the distil/synth models.
    lloom_scoring_model: str = "gemini-2.5-flash"

    # Embedding model used by LLooM to cluster event narratives.
    lloom_embedding_model: str = "gemini-embedding-001"

    # Model used in stage 5 to write investigative findings from scored evidence.
    finding_synthesis_model: str = "gemini-2.0-flash"


# ──────────────────────────────────────────────────────────────────────────────
# Event extraction (stage 1)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    # Number of evidence .txt files sent to Gemini in a single prompt.
    # Larger batches = fewer API calls but higher risk of truncation.
    evidence_files_per_batch: int = 10

    # How many times to retry a failed Gemini extraction call before giving up.
    gemini_call_retry_limit: int = 3

    # Base wait time in seconds between retries (multiplied by attempt number).
    # e.g. attempt 1 → 2s, attempt 2 → 4s, attempt 3 → 6s
    gemini_retry_backoff_base_seconds: int = 2

    # Number of characters from an event's snippet used to match it back to its
    # source file when the model omits the source_file field.
    source_file_snippet_match_chars: int = 80


# ──────────────────────────────────────────────────────────────────────────────
# LLooM concept induction (stage 4)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLoomConfig:
    # Maximum number of concepts to keep after the auto-selection step.
    max_concepts: int = 5

    # Maximum number of rerun passes over uncovered ("outlier") events.
    max_rerun_iterations: int = 3

    # If the dataset has fewer rows than this, duplicate it until it reaches
    # this size. LLooM's clustering needs a minimum number of samples.
    min_rows_required_for_induction: int = 15

    # Number of events scored per batch during the scoring step.
    scoring_batch_size: int = 50

    # Number of texts sent per embedding API call.
    embedding_batch_size: int = 10

    # Output dimensionality of gemini-embedding-001.
    # Used to create zero-vectors for empty/invalid inputs.
    embedding_output_dimension: int = 3072

    # Token budget for each LLooM LLM call (distil, synth, score).
    llm_max_output_tokens: int = 65536

    # Context window size declared to the LLooM Model wrapper.
    llm_context_window_tokens: int = 32000

    # Temperature for LLooM generation steps (0 = fully deterministic).
    llm_temperature: float = 0.0

    # Rate limit (requests/min, tokens/min) for the distil model.
    # High limit because distil calls are short and frequent.
    distill_model_rate_limit_rpm_tpm: tuple[int, int] = (1000, 1000)

    # Rate limit for synth and score models (heavier calls, lower throughput).
    synth_and_score_model_rate_limit_rpm_tpm: tuple[int, int] = (60, 60)

    # How many times to retry a failed LLooM LLM call before returning None.
    llm_call_retry_limit: int = 5

    # How many times to retry a failed embedding API call before using zeros.
    embed_call_retry_limit: int = 3

    # Minimum score for an event to count as "covered" by a concept when
    # deciding whether that concept is too generic (covers everything).
    generic_concept_min_coverage_score: float = 0.75

    # If a concept covers at least this fraction of all events, it is flagged
    # as too generic and excluded from the outlier-detection pass.
    generic_concept_max_coverage_fraction: float = 0.5

    # Set to True to skip real LLooM and use deterministic fake scores instead.
    # Useful for testing the pipeline without making API calls.
    use_mock_scoring: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Score thresholds
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreThresholds:
    # Minimum LLooM score for an event to be included in stage-5 finding synthesis.
    # Must match min_score_to_show_in_mindmap so both outputs use the same evidence.
    min_score_to_include_in_synthesis: float = 0.80

    # Minimum LLooM score for an event node to appear in the mindmap HTML.
    min_score_to_show_in_mindmap: float = 0.80


# ──────────────────────────────────────────────────────────────────────────────
# Finding synthesis (stage 5)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthesisConfig:
    # Slight temperature to allow fluent narrative writing (not fully deterministic).
    gemini_temperature: float = 0.2

    # Token budget for each generated finding paragraph.
    gemini_max_output_tokens: int = 8192

    # How many times to retry a failed synthesis API call before writing a stub.
    gemini_call_retry_limit: int = 5

    # Base wait time in seconds between retries (multiplied by attempt number).
    # e.g. attempt 1 → 10s, attempt 2 → 20s  (longer because synthesis calls are heavy)
    gemini_retry_backoff_base_seconds: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline job lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Number of times each pipeline step is retried on failure before the job
    # is marked as failed. Applies to: extract, csv, lloom, synth, mindmap.
    step_retry_count: int = 2

    # How often (seconds) the runner checks whether the analyst has submitted
    # the event review before resuming the pipeline.
    review_poll_interval_seconds: int = 6

    # Maximum wall-clock time (seconds) allowed for synthesize_findings.py
    # and mindmap.py subprocesses before they are killed.
    subprocess_timeout_seconds: int = 900


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config object
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    models:    GeminiModels    = field(default_factory=GeminiModels)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    lloom:     LLoomConfig     = field(default_factory=LLoomConfig)
    scores:    ScoreThresholds = field(default_factory=ScoreThresholds)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    pipeline:  PipelineConfig  = field(default_factory=PipelineConfig)


CONFIG = Config()
