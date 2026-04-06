# Backend Service (JSON Job Store)

This directory contains a backend-first API that orchestrates your event pipeline as async jobs and reports step-by-step progress for a future UI progress bar.

## What it does now

- Accepts uploaded evidence `.txt` files
- Creates a `job_id`
- Runs a 6-step pipeline in background
- Persists status/logs/artifacts to JSON + plain files under `backend_service/runtime/`
- Exposes final `evidence_mindmap.html`

Current pipeline runs real steps:

1. Evidence event extraction (Gemini)
2. CSV generation (`events.csv`, `events_enriched.csv`)
3. LLooM scoring with automatic outlier loop iterations
4. Findings synthesis (`findings.json`)
5. Mindmap build (`evidence_mindmap.html`)

## Folder layout

- `app/main.py`: FastAPI routes
- `app/store.py`: JSON persistence + runtime paths
- `app/runner.py`: step orchestrator + progress updates
- `app/pipeline_steps.py`: extraction, LLooM, synthesis, and mindmap execution helpers
- `app/settings.py`: centralized retries/timeouts/LLooM iteration settings
- `app/component_runner.py`: run components in isolation (extract/csv/lloom/synth/mindmap/all)
- `app/models.py`: pydantic models
- `requirements.txt`: backend-specific Python dependencies
- `start_backend.sh`: `uv`-based launcher that ensures dependencies before startup
- `.env.example`: backend environment variable template
- `runtime/jobs/<job_id>/job.json`: job state
- `runtime/jobs/<job_id>/job.log`: text logs
- `runtime/uploads/<job_id>/`: uploaded files
- `runtime/outputs/<job_id>/`: generated artifacts

## API endpoints

- `GET /` (integrated supervisor UI)
- `GET /health`
- `POST /jobs` (multipart `.txt` files)
- `POST /jobs/{job_id}/cancel`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/logs`
- `GET /jobs/{job_id}/artifacts`
- `GET /jobs/{job_id}/artifacts/{artifact_name}`
- `GET /jobs/{job_id}/mindmap`

## Run



Equivalent direct command:

```bash
uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

## Deploy on Render (single service)

You can deploy either with the Render UI settings below, or by committing the root `render.yaml` blueprint.

- **Build Command**:

```bash
pip install uv
```

- **Start Command**:

```bash
uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --host 0.0.0.0 --port $PORT
```

- **Required Env Vars**:
	- `GOOGLE_API_KEY`
	- `BACKEND_RETENTION_HOURS` (optional, default `72`)

Copy `backend_service/.env.example` when testing locally.

## Notes

- LLooM loop behavior follows your notebook logic: when uncovered evidence rows remain (`score == 0`), the backend reruns concept induction on those rows, merges new concepts, and rescoring continues up to a bounded number of iterations.
- All paths are job-isolated under `backend_service/runtime/outputs/<job_id>/`.
- Phase C resilience is enabled:
	- per-step retries with backoff
	- cancellation request support (`cancel_requested` flag + cancelled terminal state)
	- script step timeouts for synthesis/mindmap
	- startup retention cleanup of old runtime artifacts (default `72h`, configurable with `BACKEND_RETENTION_HOURS`)

## Test Components in Isolation

Use the component runner to debug a single stage without running the full API job flow.

```bash
uv run --with-requirements backend_service/requirements.txt \
	python -m backend_service.app.component_runner \
	--component extract \
	--evidence-dir /absolute/path/to/evidence_txt_dir \
	--output-dir /absolute/path/to/workdir
```

Then run next stages independently against the same `--output-dir`:

```bash
uv run --with-requirements backend_service/requirements.txt python -m backend_service.app.component_runner --component csv --output-dir /absolute/path/to/workdir
uv run --with-requirements backend_service/requirements.txt python -m backend_service.app.component_runner --component lloom --output-dir /absolute/path/to/workdir
uv run --with-requirements backend_service/requirements.txt python -m backend_service.app.component_runner --component synth --output-dir /absolute/path/to/workdir
uv run --with-requirements backend_service/requirements.txt python -m backend_service.app.component_runner --component mindmap --output-dir /absolute/path/to/workdir
```

Or run all components in one command:

```bash
uv run --with-requirements backend_service/requirements.txt \
	python -m backend_service.app.component_runner \
	--component all \
	--evidence-dir /absolute/path/to/evidence_txt_dir \
	--output-dir /absolute/path/to/workdir
```

The backend runs using **FastAPI** coupled with basic local file-system tracking (no SQL database needed).

- `app/`: source code for the backend. `main.py` defines the routes, `runner` runs the long operations.
- `runtime/`: The local DB directory. All uploads, temporary state, and final JSON are written here per job UUID.

## Running

### Backend API (primary mode)

```bash
cd hamza_event_extraction
uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --reload --port 8000
```

### Standalone scripts (no backend)

```bash
cd hamza_event_extraction
# Local pipeline on emails.txt
python main.py
```
