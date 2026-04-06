# How This Repository Works

A digital forensics pipeline that takes raw evidence `.txt` files (emails,
intercepts, logs) and produces an interactive mindmap of investigative findings.

---

## Architecture

Two modes of operation:

| Mode | Entry point | When to use |
| --- | --- | --- |
| **Backend API** | `backend_service/app/main.py` (FastAPI) | Normal use — web UI, background jobs, review gate |
| **Standalone scripts** | `main.py`, `main_batches.py` | Quick local experiments without the API |

Start the backend with `uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --reload --port 8000`, then open
`http://localhost:8000`. Upload `.txt` files, watch live progress, review events,
and download the final mindmap.

---

## File Structure

```text
hamza_event_extraction/
├── config.py                    ← ALL tunable parameters (models, thresholds, retries…)
├── event_types_db.json          ← 16 forensic event type definitions
├── synthesize_findings.py       ← Stage 6, run as subprocess
├── mindmap.py                   ← Stage 7, run as subprocess
│
└── backend_service/app/
    ├── main.py                  ← FastAPI routes (upload, review, download)
    ├── runner.py                ← Orchestrates all 7 stages for one job
    ├── pipeline_steps.py        ← Implementation of stages 2, 4, 5
    ├── store.py                 ← Reads/writes job state (job.json, job.log)
    ├── models.py                ← Pydantic models (JobRecord, ReviewableEvent…)
    ├── settings.py              ← Thin adapter: SETTINGS.x → CONFIG.x
    └── static/index.html        ← Single-file web UI
```

Runtime data (not in git):

```text
backend_service/runtime/
├── jobs/{job_id}/job.json       ← Status, step progress, review events
├── uploads/{job_id}/staging/    ← Uploaded files
└── outputs/{job_id}/            ← All generated artifacts
```

---

## The Seven Pipeline Stages

### Stage 1 — Input validation (10%)

Uploaded `.txt` files are copied from staging into `outputs/{job_id}/evidence/`
to give every subsequent stage a stable path. No LLM call.

---

### Stage 2 — Event extraction (20% → 35%)

**Code:** `pipeline_steps.py` → `extract_events_from_evidence()`

- Loads `event_types_db.json` (16 event types: `phone_call`, `bank_transaction`,
  `meeting`, `illicit_exchange`, etc.) and builds a system prompt from it.
- Sends evidence files to Gemini in batches (`CONFIG.extraction.evidence_files_per_batch`,
  default 10 files/call). Each file is prefixed with `=== SOURCE: evidence/file.txt ===`
  so Gemini can tag which file each event came from.
- Parses the JSON response into event objects with: `type`, `snippet`, `justification`,
  `confidence_score` (High/Medium/Low), `date_time`, `location`, `parties`,
  `narrative`, `type_specific_fields`, `source_file`.
- Retries failed calls up to `CONFIG.extraction.gemini_call_retry_limit` times (default 3).
- Writes `EVENTS.json` and `EVENTS_NARRATIVE.txt`.

---

### Stage 3 — Human review gate (42% → 48%)

**Code:** `runner.py` → `_wait_for_review_submission()`

The pipeline **blocks** after extraction. The job moves to `awaiting_review`
and the runner polls `job.json` every `CONFIG.pipeline.review_poll_interval_seconds`
seconds (default 6s).

The analyst opens the web UI, sees a table of all extracted events, deselects
false positives, and clicks **Submit**. The API endpoint
`POST /jobs/{job_id}/review-events` records the approved IDs. The runner
detects `review_submitted = true`, overwrites `EVENTS.json` with only the
approved events, and continues.

---

### Stage 4 — CSV enrichment (55% → 60%)

**Code:** `pipeline_steps.py` → `build_csv_from_events()`

Flattens `EVENTS.json` into two CSVs:

- **`events.csv`** — `id, event` (the narrative text; fed to LLooM)
- **`events_enriched.csv`** — `id, event, source_file, snippet` (provenance for
  the mindmap)

---

### Stage 5 — LLooM concept scoring (70% → 80%)

**Code:** `pipeline_steps.py` → `run_lloom_iterative()`

LLooM (Stanford) induces abstract concepts from a set of text items, then scores
each item against every concept.

1. Loads `events.csv`. Duplicates the dataset if fewer than
   `CONFIG.lloom.min_rows_required_for_induction` rows (default 15) — LLooM's
   clustering needs a minimum sample size.
2. Initialises four Gemini-backed models:
   - **distill** (`gemini-2.0-flash`) — bullet-point summary per event
   - **cluster** (`gemini-embedding-001`) — embeds bullets to form clusters
   - **synth** (`gemini-2.0-flash`) — generates a concept label per cluster
   - **score** (`gemini-2.5-flash`) — scores each event against each concept (0–1)
3. Runs up to `CONFIG.lloom.max_rerun_iterations` outlier passes (default 3):
   events that score 0 on all concepts are fed to a second LLooM run so new
   concepts can be found to explain them.
4. Writes `score_results_combined.csv` — one row per (event × concept) pair.

---

### Stage 6 — Finding synthesis (86% → 92%)

**Code:** `synthesize_findings.py` (subprocess)

- Reads `score_results_combined.csv`, keeps rows with
  `score >= CONFIG.scores.min_score_to_include_in_synthesis` (default 0.80).
- For each concept sends the high-confidence evidence to Gemini with a forensic
  analyst prompt: identify the pattern, key actors, timeframe, significance, and
  evidence strength (STRONG / MODERATE / CIRCUMSTANTIAL).
- Writes `findings.json` — one entry per concept with `finding`, `strength`,
  `avg_score`, and `top_evidence`.

---

### Stage 7 — Mindmap generation (97% → 100%)

**Code:** `mindmap.py` (subprocess)

Reads all four artifacts (`score_results_combined.csv`, `events_enriched.csv`,
`EVENTS.json`, `findings.json`) and produces a self-contained
`evidence_mindmap.html` — an interactive tree:

- **Root** → Evidence Corpus
- **Level 2** → one node per concept (finding paragraph, strength, avg score)
- **Level 3** → one node per event (type, date, parties, source file, snippet, score)

No server needed — open the HTML directly in a browser.

---

## Data Flow

```text
Upload .txt files  →  POST /jobs  →  run_pipeline()
                                          │
    ┌─────────────────────────────────────┤
    │                                     │
    ▼                                     ▼
Stage 1: copy to evidence/          Stage 2: Gemini extraction
                                          │
                                    EVENTS.json
                                    EVENTS_NARRATIVE.txt
                                          │
                                    Stage 3: BLOCK (awaiting_review)
                                          │  analyst submits review
                                          ▼
                                    Stage 4: build CSVs
                                          │
                                    events.csv
                                    events_enriched.csv
                                          │
                                    Stage 5: LLooM scoring
                                          │
                                    score_results_combined.csv
                                          │
                                    Stage 6: synthesize_findings.py
                                          │
                                    findings.json
                                          │
                                    Stage 7: mindmap.py
                                          │
                                    evidence_mindmap.html  ← final output
```

---

## Worked Example

**Input — two email files:**

`email_001.txt`: Ahmed Hassan tells Khalid Musa that a package from Karachi
arrived and he transferred $50,000 to "the usual account".

`email_002.txt`: Khalid confirms and arranges a meeting at Café Milano, Brussels
at 18:00 Friday to hand over documents. "Come alone."

---

**Stage 2 output — Gemini returns 3 events (excerpt):**

```json
[
  {
    "type": "bank_transaction",
    "snippet": "I transferred $50,000 to the usual account (IBAN: GB29 NWBK …)",
    "confidence_score": "High",
    "date_time": "14 March 2024",
    "parties": ["Ahmed Hassan", "Khalid Musa"],
    "narrative": "Ahmed Hassan transferred $50,000 to a pre-established account. The phrase 'usual account' implies a prior pattern of transfers.",
    "type_specific_fields": { "amount": "$50,000", "transaction_direction": "outgoing" },
    "source_file": "evidence/email_001.txt",
    "id": 0
  },
  { "type": "meeting",      "id": 1, "source_file": "evidence/email_002.txt", "..." : "..." },
  { "type": "digital_communication", "id": 2, "source_file": "evidence/email_001.txt", "...": "..." }
]
```

---

**Stage 3 — Analyst approves all 3 events.** `EVENTS.json` is overwritten with
the approved set.

---

**Stage 5 — LLooM induces 2 concepts and scores all events:**

```text
doc_id  concept_name                            score
0       Financial and Logistical Coordination   0.94
0       Covert In-Person Communication          0.12
1       Financial and Logistical Coordination   0.21
1       Covert In-Person Communication          0.91
2       Financial and Logistical Coordination   0.87
2       Covert In-Person Communication          0.18
```

---

**Stage 6 — Gemini synthesises one finding per concept (score ≥ 0.80):**

`findings.json`:

```json
{
  "Financial and Logistical Coordination": {
    "finding": "Ahmed Hassan and Khalid Musa operate a delivery-against-payment structure linking a Karachi supply chain to high-value wire transfers via a pre-established account, indicating an organised cross-border logistics network.",
    "strength": "STRONG", "avg_score": 0.905, "n_items": 2
  },
  "Covert In-Person Communication": {
    "finding": "Khalid Musa directed a clandestine document handover at a Brussels café with explicit OPSEC instructions, suggesting awareness of surveillance.",
    "strength": "MODERATE", "avg_score": 0.91, "n_items": 1
  }
}
```

---

**Stage 7 — Final mindmap tree:**

```text
Evidence Corpus
├── Financial and Logistical Coordination  [STRONG · avg 0.91 · 2 events]
│   │  "Ahmed Hassan and Khalid Musa operate a delivery-against-payment structure…"
│   ├── Event 0 · bank_transaction · High · score 0.94 · email_001.txt
│   └── Event 2 · digital_communication · High · score 0.87 · email_001.txt
│
└── Covert In-Person Communication  [MODERATE · avg 0.91 · 1 event]
    │  "Khalid Musa directed a clandestine document handover…"
    └── Event 1 · meeting · High · score 0.91 · email_002.txt
```

Served at `GET /jobs/{id}/mindmap`. Job status: `completed`.

---

## Configuration Reference

All parameters live in `config.py`. Change a value once and every stage picks it up.

| Parameter | Default | Controls |
| --- | --- | --- |
| `CONFIG.models.extraction_model` | `gemini-2.0-flash` | Event extraction model |
| `CONFIG.models.lloom_scoring_model` | `gemini-2.5-flash` | LLooM scoring model (more capable) |
| `CONFIG.models.finding_synthesis_model` | `gemini-2.0-flash` | Finding synthesis model |
| `CONFIG.extraction.evidence_files_per_batch` | `10` | Files per Gemini extraction call |
| `CONFIG.extraction.gemini_call_retry_limit` | `3` | Extraction retries on failure |
| `CONFIG.lloom.max_concepts` | `5` | Max concepts induced by LLooM |
| `CONFIG.lloom.max_rerun_iterations` | `3` | Outlier rerun passes |
| `CONFIG.lloom.min_rows_required_for_induction` | `15` | Dataset duplicated below this size |
| `CONFIG.lloom.generic_concept_min_coverage_score` | `0.75` | Score to count an event as "covered" |
| `CONFIG.lloom.use_mock_scoring` | `False` | Skip real LLooM (for testing) |
| `CONFIG.scores.min_score_to_include_in_synthesis` | `0.80` | Min score for finding synthesis |
| `CONFIG.scores.min_score_to_show_in_mindmap` | `0.80` | Min score to appear in mindmap |
| `CONFIG.synthesis.gemini_temperature` | `0.2` | Creativity for finding writing |
| `CONFIG.pipeline.step_retry_count` | `2` | Retries per pipeline step on failure |
| `CONFIG.pipeline.subprocess_timeout_seconds` | `900` | Kill timeout for stages 6 and 7 |

To use mock scoring during development (no API calls for stages 5–7):

```bash
MOCK_LLOOM=true uv run --with-requirements backend_service/requirements.txt uvicorn backend_service.app.main:app --reload --port 8000
```

---

## Job Lifecycle

```text
queued → running → awaiting_review → running → completed
                                             → failed
                                             → cancelled (any time)
```

Cancel is checked between every stage. If requested the pipeline stops cleanly
and the job is marked `cancelled`.
