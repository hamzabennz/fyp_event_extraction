# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required Skills Before Coding

Before writing or modifying any code, you MUST invoke the following skills in order:

1. **`andrej-karpathy-skills:karpathy-guidelines`** — Guards against overcomplication, encourages surgical changes, surfaces assumptions, and defines verifiable success criteria.
2. **`superpowers:brainstorming`** — Explores user intent and requirements before implementation.
3. **`superpowers:writing-plans`** — Produces a step-by-step plan for multi-step tasks before touching code.
4. **`superpowers:test-driven-development`** — Guides implementation with tests first.

These are non-negotiable. Do not skip them, even for small changes.

## Project Overview

A 6-stage AI-powered digital forensics pipeline that extracts, reviews, analyzes, and visualizes events from raw evidence text files. Uses Google Gemini for LLM-based extraction and synthesis, LLooM for concept induction, and provides both a CLI and a FastAPI web backend.

## Environment Setup

Requires Python 3.12 (`.python-version`) and a `GOOGLE_API_KEY` environment variable (see `.env.example`).

Optional env vars: `MOCK_LLOOM=true` (skip LLooM in testing), `BACKEND_RETENTION_HOURS` (default 72).

## Commands

### Backend API (primary mode)
```bash
# Start backend (recommended — uses uv):
uv run --with-requirements backend_service/requirements.txt \
  uvicorn backend_service.app.main:app --reload --port 8000
```

### Standalone scripts (no backend)
```bash
python main.py                          # Single-file extraction (processes emails2.txt)
python main_batches.py --batch-size 10  # Batch extraction with options
python mindmap.py                       # Generate mindmap HTML
python synthesize_findings.py           # Phase 4 synthesis (reads score_results_combined.csv)
```

### Test individual pipeline components in isolation
```bash
uv run --with-requirements backend_service/requirements.txt \
  python -m backend_service.app.component_runner \
  --component {extract|csv|lloom|synth|mindmap|all} \
  --evidence-dir /path/to/evidence \
  --output-dir /path/to/output
```

## Architecture

### Pipeline Stages (in order)

1. **Event Extraction** (`pipeline_steps.py:extract_events_from_evidence`) — Gemini 2.0 Flash reads evidence `.txt` files in batches, extracts structured events using the schema in `event_types_db.json` (15 event types). Outputs `EVENTS.json` and `EVENTS_NARRATIVE.txt`.

2. **Human Review Gate** — Analyst reviews extracted events via the web UI. Job enters `awaiting_review` state until events are approved/rejected. Only approved events proceed.

3. **CSV Enrichment** (`pipeline_steps.py:build_csv_from_events`) — Flattens JSON events to `events.csv` / `events_enriched.csv`.

4. **LLooM Concept Scoring** (`pipeline_steps.py:run_lloom_iterative`) — Stanford LLooM induces high-level concepts from event narratives, then scores each event. Iterates up to 3× on uncovered rows (score == 0). Threshold: 0.5. Outputs `score_results_combined.csv`.

5. **Finding Synthesis** (`synthesize_findings.py`) — Gemini synthesizes high-confidence evidence (score ≥ 0.80) into investigative findings. Outputs `findings.json`.

6. **Mindmap Visualization** (`mindmap.py`) — Generates an interactive NotebookLM-style HTML tree (`evidence_mindmap.html`).

### Backend Service (`backend_service/`)

- **`app/main.py`** — FastAPI routes; job lifecycle: `queued → running → awaiting_review → completed` (or `failed`/`cancelled`)
- **`app/runner.py`** — `run_pipeline()` orchestrates all 7 steps with retry logic and cancellation support
- **`app/pipeline_steps.py`** — Concrete implementation of each step; `run_python_script()` executes synthesis/mindmap as subprocess with 900s timeout
- **`app/store.py`** — JSON file-based persistence under `backend_service/runtime/jobs/{job_id}/`
- **`app/models.py`** — Pydantic models for jobs, steps, events
- **`app/settings.py`** — Step retry count (2), timeouts, LLooM knobs (max 5 concepts, 3 iterations)
- **`app/static/index.html`** — Integrated supervisor UI (~53KB single-file)

### Standalone Scripts

- **`main.py`** — Uses `smolagents` framework with a custom `GeminiModel` wrapper and `SaveEventsToFileTool`
- **`main_batches.py`** — Batch extraction with progress tracking and incremental output appending

### Event Schema

`event_types_db.json` defines 15 forensic event types: `phone_call`, `bank_transaction`, `travel_movement`, `meeting`, `digital_communication`, `physical_surveillance`, `illicit_exchange`, `border_crossing`, `cyber_incident`, `financial_transaction`, `social_media_activity`, and 4 more. Each type has a description and type-specific fields extracted by the LLM.

### Job Artifacts Layout

```
backend_service/runtime/
├── jobs/{job_id}/
│   ├── job.json    # Status, steps, approved events
│   └── job.log
├── uploads/{job_id}/staging/   # Input .txt files
└── outputs/{job_id}/           # All generated artifacts
```

## Knowledge Graph Extension (`anis_graph` branch)

### Setup
```bash
docker compose up -d   # starts Neo4j 5.18 on bolt://localhost:7687 (default creds: neo4j/changeme)
```

### New Pipeline Step
After `build_csv`, a new `build_knowledge_graph` step runs **in parallel** with `lloom_scoring` using `ThreadPoolExecutor`. Graph failures are non-fatal (logged, pipeline continues).

### New Files
- **`app/graph_builder.py`** — Neo4j service: lazy driver, `insert_job_events()`, `get_graph_data_for_job()`, `clear_job_graph()`, `ensure_indexes()`
- **`app/contradiction_detector.py`** — `detect_and_store_contradictions()`: temporal overlap (Allen's Interval Algebra), border/domestic CSP rule, contact denial rule. Writes `[:CONTRADICTS]` edges to Neo4j.
- **`app/static/graph.html`** — D3.js v7 force-directed graph UI

### New API Routes
| Route | Purpose |
|---|---|
| `GET /jobs/{job_id}/graph-data` | D3.js {nodes, links} JSON |
| `GET /jobs/{job_id}/contradictions` | Contradiction list from Neo4j |
| `POST /jobs/{job_id}/graph-data/refresh` | Re-insert + re-detect |
| `GET /graph/{job_id}` | Serve graph.html |

### Graph Schema
```
(:Job {job_id})
(:Event {id, job_id, type, date_time, location, confidence_score, narrative, snippet, source_file, tsf_*})
(:Person {name})   — shared across jobs
(:Location {name}) — shared across jobs

(:Job)-[:HAS_EVENT]->(:Event)
(:Person)-[:PARTICIPANT_IN]->(:Event)
(:Event)-[:LOCATED_AT]->(:Location)
(:Event)-[:CONTRADICTS {type, description, severity}]->(:Event)
```

### Neo4j Config
Override defaults via env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. Set `neo4j.enabled=False` in `config.py` to disable entirely (e.g. in CI).

## Key Design Patterns

- **Retry with backoff**: `_run_with_retries()` in `runner.py` — exponential backoff (2×attempt seconds), configurable retry count in `settings.py`
- **Cancellation**: Jobs check a cancellation flag between steps; state is cleaned up on cancel
- **LLooM iterative coverage**: Reruns concept induction on zero-scored rows until convergence or max iterations
- **Component isolation**: `component_runner.py` lets you test any single stage without the full backend
- **Parallel steps**: `_run_parallel_steps()` in `runner.py` — `build_knowledge_graph` and `lloom_scoring` run concurrently; graph is non-fatal
