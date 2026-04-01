# How This Repository Works

A digital forensics pipeline that takes raw evidence text files (emails,
intercepts, surveillance logs) and produces an interactive mindmap of
investigative findings. Every step is explained below, followed by a
concrete worked example from file upload to final HTML.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [File structure](#2-file-structure)
3. [The seven pipeline stages](#3-the-seven-pipeline-stages)
   - [Stage 1 — Input validation](#stage-1--input-validation)
   - [Stage 2 — Event extraction](#stage-2--event-extraction)
   - [Stage 3 — Human review gate](#stage-3--human-review-gate)
   - [Stage 4 — CSV enrichment](#stage-4--csv-enrichment)
   - [Stage 5 — LLooM concept scoring](#stage-5--lloom-concept-scoring)
   - [Stage 6 — Finding synthesis](#stage-6--finding-synthesis)
   - [Stage 7 — Mindmap generation](#stage-7--mindmap-generation)
4. [Data flow diagram](#4-data-flow-diagram)
5. [Step-by-step worked example](#5-step-by-step-worked-example)
6. [Configuration reference](#6-configuration-reference)
7. [Job lifecycle states](#7-job-lifecycle-states)

---

## 1. Architecture overview

The system has two modes:

| Mode | Entry point | When to use |
|---|---|---|
| **Backend API** | `backend_service/app/main.py` (FastAPI) | Normal use — handles uploads, exposes a web UI, runs the pipeline in the background |
| **Standalone scripts** | `main.py`, `main_batches.py` | Quick experiments or processing a local folder without the API |

In normal use you start the backend with `zsh backend_service/start_backend.sh`,
open your browser, upload `.txt` evidence files, and the pipeline runs automatically.
The web UI at `http://localhost:8000` shows live progress, lets you review extracted
events, and serves the final mindmap.

---

## 2. File structure

```
hamza_event_extraction/
│
├── config.py                        ← ALL tunable parameters live here
├── event_types_db.json              ← The 16 forensic event type definitions
├── synthesize_findings.py           ← Stage 6 — run as a subprocess
├── mindmap.py                       ← Stage 7 — run as a subprocess
│
├── backend_service/
│   ├── start_backend.sh
│   └── app/
│       ├── main.py                  ← FastAPI routes (upload, review, download)
│       ├── runner.py                ← Orchestrates stages 1-7 for one job
│       ├── pipeline_steps.py        ← Concrete logic for stages 1, 4, 5
│       ├── store.py                 ← Reads/writes job state to disk
│       ├── models.py                ← Pydantic models (JobRecord, ReviewableEvent…)
│       ├── settings.py              ← Adapter: maps SETTINGS.x → CONFIG.x
│       ├── component_runner.py      ← CLI to run any single stage in isolation
│       └── static/index.html        ← Single-file web UI (~53 KB)
│
└── backend_service/runtime/         ← Created at startup, ignored by git
    ├── jobs/{job_id}/
    │   ├── job.json                 ← Job status, step progress, review events
    │   └── job.log                  ← Timestamped log lines
    ├── uploads/{job_id}/staging/    ← Uploaded .txt files land here
    └── outputs/{job_id}/            ← All generated artifacts land here
```

---

## 3. The seven pipeline stages

### Stage 1 — Input validation

**Code:** `runner.py` → `run_pipeline()` lines before `extract_events`

What happens:
- The uploaded `.txt` files are copied from staging into
  `outputs/{job_id}/evidence/` so every subsequent stage has a stable path.
- The step is marked `completed` immediately (no LLM call).
- Progress jumps to 10%.

Output: `outputs/{job_id}/evidence/*.txt`

---

### Stage 2 — Event extraction

**Code:** `pipeline_steps.py` → `extract_events_from_evidence()`

What happens:
1. `event_types_db.json` is loaded. It defines 16 forensic event types
   (`phone_call`, `bank_transaction`, `meeting`, `illicit_exchange`, etc.)
   with a description and a list of type-specific fields for each.
2. A system prompt is built from that schema. It instructs Gemini to act as a
   forensic analyst and return a JSON array of events it finds in the evidence.
3. The evidence files are split into batches
   (`CONFIG.extraction.evidence_files_per_batch`, default 10 files per call).
4. Each batch is sent to Gemini (`CONFIG.models.extraction_model`, default
   `gemini-2.0-flash`) in a single prompt.
5. The response is parsed as JSON. Each event object contains:
   - `type` — one of the 16 event types
   - `snippet` — the exact verbatim text from the evidence
   - `justification` — why this qualifies as that event type
   - `confidence_score` — High / Medium / Low
   - `date_time`, `location`, `parties`
   - `narrative` — a full forensic narrative paragraph
   - `type_specific_fields` — structured fields for that event type
   - `source_file` — which `.txt` file it came from
6. If Gemini fails, the call is retried up to
   `CONFIG.extraction.gemini_call_retry_limit` times (default 3) with
   exponential backoff.
7. All events are assigned sequential integer IDs.
8. Two artifacts are written:
   - `EVENTS.json` — the full structured event list
   - `EVENTS_NARRATIVE.txt` — one plain-text block per event

Progress: 20% → 35%

---

### Stage 3 — Human review gate

**Code:** `runner.py` → `_wait_for_review_submission()`, `store.py`

What happens:
1. The extracted events are saved to `job.json` as `review_events`.
2. The job status changes to `awaiting_review`. The pipeline thread **blocks**
   here — it polls `job.json` every
   `CONFIG.pipeline.review_poll_interval_seconds` seconds (default 6s).
3. The analyst opens the web UI, sees a table of all extracted events with their
   type, snippet, confidence, and narrative.
4. The analyst deselects any events that are wrong or irrelevant, then clicks
   **Submit review**.
5. The API endpoint `POST /jobs/{job_id}/review-events` records the approved
   event IDs in `job.json` and sets `review_submitted = true`.
6. The polling loop detects this, loads the approved events, overwrites
   `EVENTS.json` and `EVENTS_NARRATIVE.txt` with only the approved events, and
   resumes.

Why this gate exists: Gemini sometimes extracts false positives (events with
low confidence or that misread the evidence). The analyst removes them before
they propagate into scoring and synthesis.

Progress: 42% → 48%

---

### Stage 4 — CSV enrichment

**Code:** `pipeline_steps.py` → `build_csv_from_events()`

What happens:
- Reads the approved `EVENTS.json`.
- Writes two CSV files:

  **`events.csv`** (used by LLooM)
  ```
  id,event
  0,"On 14 March Ahmed called Khalid at 09:15..."
  1,"A wire transfer of $50,000 was made from..."
  ```

  **`events_enriched.csv`** (used by the mindmap for provenance)
  ```
  id,event,source_file,snippet
  0,"On 14 March...",evidence/email_001.txt,"Ahmed called Khalid..."
  ```

The `event` column is the narrative — a plain-English description that LLooM
can embed and reason about. The snippet and source_file columns are carried
along so the final mindmap can show exactly where in the raw evidence each
event came from.

Progress: 55% → 60%

---

### Stage 5 — LLooM concept scoring

**Code:** `pipeline_steps.py` → `run_lloom_iterative()`

This is the most complex stage. LLooM (Stanford) is a concept induction
library that finds abstract themes across a set of text items.

What happens:
1. `events.csv` is loaded. If it has fewer than
   `CONFIG.lloom.min_rows_required_for_induction` rows (default 15), the
   dataset is duplicated until it reaches that size — LLooM's clustering needs
   a minimum number of samples.
2. Four Gemini-backed LLooM models are initialised:
   - **distill** (`gemini-2.0-flash`) — summarises each event into a bullet
   - **cluster** (`gemini-embedding-001`) — embeds bullets to form clusters
   - **synth** (`gemini-2.0-flash`) — synthesises a concept label per cluster
   - **score** (`gemini-2.5-flash`) — scores every event against every concept
     on a 0–1 scale
3. `l.gen()` runs the distil → cluster → synthesise pipeline and produces up to
   `CONFIG.lloom.max_concepts` concepts (default 5).
4. `l.score()` scores every event against every concept.
5. **Iterative outlier rerun** (up to `CONFIG.lloom.max_rerun_iterations`,
   default 3):
   - Events with score = 0 across all non-generic concepts are "outliers" —
     nothing covers them yet.
   - A second LLooM instance runs on just those outlier events to find new
     concepts that explain them.
   - New concepts are merged back into the main concept set.
   - This repeats until no outliers remain or the iteration limit is reached.
6. The combined scoring table is written to `score_results_combined.csv`:

```
doc_id,text,concept_id,concept_name,concept_prompt,score,rationale,highlight,concept_seed
0,"On 14 March...",abc123,"Financial Coordination","Does this evidence...",0.91,"Strong match...",""
0,"On 14 March...",def456,"Communication Pattern","...",0.23,"Weak match...",""
```

Each event gets one row **per concept**. The `score` column (0–1) represents
how strongly the LLM judges the event to match the concept.

Progress: 70% → 80%

---

### Stage 6 — Finding synthesis

**Code:** `synthesize_findings.py` (run as a subprocess by `runner.py`)

What happens:
1. Reads `score_results_combined.csv`.
2. Filters to rows where `score >= CONFIG.scores.min_score_to_include_in_synthesis`
   (default 0.80). These are the events the model is highly confident about.
3. For each concept, deduplicates by `doc_id` (keeps highest-scoring row per
   event) and builds a numbered evidence block.
4. Sends that evidence block to Gemini (`CONFIG.models.finding_synthesis_model`,
   default `gemini-2.0-flash`) with a structured synthesis prompt:
   > "You are a senior forensic intelligence analyst. Write an investigative
   > finding — identify the pattern, key actors, timeframe, and significance.
   > End with: Strength: STRONG / MODERATE / CIRCUMSTANTIAL"
5. Parses the strength label from the response.
6. Writes `findings.json`:

```json
{
  "Financial Coordination": {
    "concept": "Financial Coordination",
    "n_items": 7,
    "finding": "A systematic pattern of cash transfers...",
    "strength": "STRONG",
    "avg_score": 0.88,
    "top_evidence": [...]
  }
}
```

If Gemini is safety-blocked or unreachable, a fallback stub is written so the
mindmap still has content.

Progress: 86% → 92%

---

### Stage 7 — Mindmap generation

**Code:** `mindmap.py` (run as a subprocess by `runner.py`)

What happens:
1. Reads `score_results_combined.csv`, filters to
   `score >= CONFIG.scores.min_score_to_show_in_mindmap` (default 0.80).
2. Reads `events_enriched.csv` to get `source_file` and `snippet` per event
   (provenance).
3. Reads `EVENTS.json` to get full metadata (event type, parties, location,
   date/time, confidence score).
4. Reads `findings.json` to get the investigative finding paragraph per concept.
5. Groups everything by concept:
   - concept name, average score, event count
   - investigative finding + strength rating
   - list of matching events, each with full metadata
6. Generates a self-contained HTML file (`evidence_mindmap.html`) with an
   interactive tree:
   - Root node: "Evidence Corpus"
   - Second level: one node per concept (shows finding, strength, count)
   - Third level: one node per event (shows type, date, parties, snippet,
     source file, score)

The HTML is entirely standalone — no server needed. You can open it directly
in a browser.

Progress: 97% → 100%

---

## 4. Data flow diagram

```
[Analyst uploads .txt files]
          │
          ▼
    POST /jobs  (main.py)
          │
          ▼
    run_pipeline()  (runner.py)
          │
          ├─► Stage 1: copy to outputs/{id}/evidence/
          │
          ├─► Stage 2: extract_events_from_evidence()
          │            │
          │            ├── event_types_db.json  ──► system prompt
          │            ├── Gemini API (batched)
          │            └── outputs/{id}/EVENTS.json
          │                outputs/{id}/EVENTS_NARRATIVE.txt
          │
          ├─► Stage 3: BLOCK — poll job.json until review_submitted=true
          │            │
          │            └── analyst selects/deselects via web UI
          │                POST /jobs/{id}/review-events
          │                overwrites EVENTS.json with approved events only
          │
          ├─► Stage 4: build_csv_from_events()
          │            │
          │            └── EVENTS.json ──► events.csv
          │                               events_enriched.csv
          │
          ├─► Stage 5: run_lloom_iterative()
          │            │
          │            ├── events.csv ──► LLooM (distil/cluster/synth/score)
          │            ├── iterative outlier reruns
          │            └── score_results_combined.csv
          │
          ├─► Stage 6: subprocess → synthesize_findings.py
          │            │
          │            ├── score_results_combined.csv (score ≥ 0.80)
          │            ├── Gemini API (one call per concept)
          │            └── findings.json
          │
          └─► Stage 7: subprocess → mindmap.py
                       │
                       ├── score_results_combined.csv (score ≥ 0.80)
                       ├── events_enriched.csv (provenance)
                       ├── EVENTS.json (metadata)
                       ├── findings.json (synthesis)
                       └── evidence_mindmap.html  ◄── final output
```

---

## 5. Step-by-step worked example

### Setup

Assume two evidence files:

**`email_001.txt`**
```
From: Ahmed Hassan <ahmed@secure.net>
To: Khalid Musa <khalid@proton.me>
Date: 14 March 2024, 09:15

Khalid, the package from Karachi arrived last night. I transferred
$50,000 to the usual account (IBAN: GB29 NWBK …). Confirm receipt.
```

**`email_002.txt`**
```
From: Khalid Musa <khalid@proton.me>
To: Ahmed Hassan <ahmed@secure.net>
Date: 14 March 2024, 11:42

Confirmed. Will meet you at Café Milano, Brussels at 18:00 on Friday
to hand over the documents. Come alone.
```

---

### Stage 1 — Files stored

Both files are copied to:
```
outputs/a3f9.../evidence/email_001.txt
outputs/a3f9.../evidence/email_002.txt
```
Job status: `running`, progress: 10%.

---

### Stage 2 — Gemini extraction

A single batch (2 files < batch size of 10) is sent to Gemini with the system
prompt. Gemini returns:

```json
[
  {
    "type": "bank_transaction",
    "justification": "An explicit transfer of $50,000 to an IBAN is mentioned",
    "snippet": "I transferred $50,000 to the usual account (IBAN: GB29 NWBK …)",
    "confidence_score": "High",
    "date_time": "14 March 2024",
    "location": "N/A",
    "parties": ["Ahmed Hassan", "Khalid Musa"],
    "narrative": "On 14 March 2024, Ahmed Hassan transferred $50,000 to Khalid Musa's account (IBAN: GB29 NWBK). The phrase 'usual account' implies a prior pattern of transfers.",
    "type_specific_fields": {
      "amount": "$50,000",
      "account_identifier": "IBAN: GB29 NWBK",
      "transaction_direction": "outgoing"
    },
    "source_file": "evidence/email_001.txt",
    "id": 0
  },
  {
    "type": "meeting",
    "justification": "An explicit in-person meeting at a named location is arranged",
    "snippet": "meet you at Café Milano, Brussels at 18:00 on Friday",
    "confidence_score": "High",
    "date_time": "Friday 18:00 (week of 14 March 2024)",
    "location": "Café Milano, Brussels",
    "parties": ["Ahmed Hassan", "Khalid Musa"],
    "narrative": "Khalid Musa arranged a face-to-face meeting with Ahmed Hassan at Café Milano, Brussels, at 18:00 on Friday to hand over unspecified documents. The instruction to 'come alone' suggests operational security awareness.",
    "type_specific_fields": {
      "venue": "Café Milano",
      "city": "Brussels",
      "purpose": "document handover"
    },
    "source_file": "evidence/email_002.txt",
    "id": 1
  },
  {
    "type": "digital_communication",
    "justification": "An email exchange confirming receipt of a physical package is recorded",
    "snippet": "the package from Karachi arrived last night",
    "confidence_score": "High",
    "date_time": "14 March 2024, 09:15",
    "location": "N/A",
    "parties": ["Ahmed Hassan", "Khalid Musa"],
    "narrative": "Ahmed Hassan informed Khalid Musa via encrypted email that a package originating from Karachi arrived the previous night, suggesting a supply chain connection to Pakistan.",
    "type_specific_fields": {
      "platform": "email (secure.net / proton.me)",
      "direction": "Ahmed → Khalid",
      "subject_matter": "package arrival confirmation"
    },
    "source_file": "evidence/email_001.txt",
    "id": 2
  }
]
```

Written to:
- `outputs/a3f9.../EVENTS.json` — the JSON above
- `outputs/a3f9.../EVENTS_NARRATIVE.txt` — one plain-text block per event

Job status: `awaiting_review`, progress: 35%.

---

### Stage 3 — Analyst review

The analyst opens the web UI. They see a table:

| # | Type | Confidence | Snippet | Action |
|---|---|---|---|---|
| 0 | bank_transaction | High | I transferred $50,000… | ✓ selected |
| 1 | meeting | High | meet you at Café Milano… | ✓ selected |
| 2 | digital_communication | High | the package from Karachi… | ✓ selected |

The analyst keeps all three and clicks **Submit**. The API records
`selected_ids: [0, 1, 2]`.

`EVENTS.json` is overwritten with the same 3 events (all approved).

Job status: `running`, progress: 48%.

---

### Stage 4 — CSV generation

**`events.csv`**
```
id,event
0,"On 14 March 2024, Ahmed Hassan transferred $50,000 to Khalid Musa's account (IBAN: GB29 NWBK). The phrase 'usual account' implies a prior pattern of transfers."
1,"Khalid Musa arranged a face-to-face meeting with Ahmed Hassan at Café Milano, Brussels, at 18:00 on Friday to hand over unspecified documents."
2,"Ahmed Hassan informed Khalid Musa via encrypted email that a package originating from Karachi arrived the previous night."
```

**`events_enriched.csv`**
```
id,event,source_file,snippet
0,"On 14 March 2024...",evidence/email_001.txt,"I transferred $50,000 to the usual account"
1,"Khalid Musa arranged...",evidence/email_002.txt,"meet you at Café Milano, Brussels at 18:00"
2,"Ahmed Hassan informed...",evidence/email_001.txt,"the package from Karachi arrived last night"
```

Progress: 60%.

---

### Stage 5 — LLooM scoring

LLooM processes the 3 narratives and induces (for example) these 2 concepts:

- **"Financial and Logistical Coordination"** — covers events 0 and 2 (the
  transfer and the package)
- **"Covert In-Person Communication"** — covers event 1 (the café meeting)

Scoring output (`score_results_combined.csv`, simplified):

```
doc_id,concept_name,score,rationale
0,Financial and Logistical Coordination,0.94,"Direct match: explicit bank transfer described"
0,Covert In-Person Communication,0.12,"Weak: financial in nature, not a meeting"
1,Financial and Logistical Coordination,0.21,"Indirect: meeting may relate to the transaction"
1,Covert In-Person Communication,0.91,"Strong: face-to-face handover with OPSEC instructions"
2,Financial and Logistical Coordination,0.87,"Match: package arrival tied to the financial transfer"
2,Covert In-Person Communication,0.18,"Weak: digital communication, not in-person"
```

All 3 events score above 0 on at least one concept — no outlier rerun needed.

Progress: 80%.

---

### Stage 6 — Finding synthesis

Two concepts have events scoring ≥ 0.80:

**Financial and Logistical Coordination** (events 0 and 2, avg score 0.905):

Gemini receives:
```
Concept: "Financial and Logistical Coordination"
Evidence:
1. [94%] On 14 March 2024, Ahmed Hassan transferred $50,000...
2. [87%] Ahmed Hassan informed Khalid Musa via encrypted email that a package...
```

Gemini responds:
```
Ahmed Hassan and Khalid Musa are engaged in a coordinated supply and payment
operation involving both physical goods sourced from Karachi and corresponding
high-value fund transfers to a pre-established account. The use of encrypted
email services and references to a 'usual account' indicate an established
operational relationship predating these communications. The concurrent timing
of the package arrival and the wire transfer on 14 March 2024 suggests a
delivery-against-payment structure consistent with organised criminal logistics.
The pattern implicates both actors in a cross-border financial-physical supply
chain.
Strength: STRONG
```

**`findings.json`** excerpt:
```json
{
  "Financial and Logistical Coordination": {
    "concept": "Financial and Logistical Coordination",
    "n_items": 2,
    "finding": "Ahmed Hassan and Khalid Musa are engaged in a coordinated supply...",
    "strength": "STRONG",
    "avg_score": 0.905
  },
  "Covert In-Person Communication": {
    "concept": "Covert In-Person Communication",
    "n_items": 1,
    "finding": "Khalid Musa directed Ahmed Hassan to a specific café in Brussels...",
    "strength": "MODERATE",
    "avg_score": 0.91
  }
}
```

Progress: 92%.

---

### Stage 7 — Mindmap generation

`mindmap.py` reads all four artifacts and builds an HTML tree:

```
Evidence Corpus
│
├── Financial and Logistical Coordination  [STRONG · avg 0.91 · 2 events]
│   │  Finding: "Ahmed Hassan and Khalid Musa are engaged in a coordinated
│   │            supply and payment operation…"
│   │
│   ├── Event 0 · bank_transaction · High confidence
│   │     Date: 14 March 2024
│   │     Parties: Ahmed Hassan, Khalid Musa
│   │     Source: evidence/email_001.txt
│   │     Score: 0.94
│   │     "I transferred $50,000 to the usual account (IBAN: GB29 NWBK …)"
│   │
│   └── Event 2 · digital_communication · High confidence
│         Date: 14 March 2024 09:15
│         Parties: Ahmed Hassan, Khalid Musa
│         Source: evidence/email_001.txt
│         Score: 0.87
│         "the package from Karachi arrived last night"
│
└── Covert In-Person Communication  [MODERATE · avg 0.91 · 1 event]
    │  Finding: "Khalid Musa directed Ahmed Hassan to a specific café…"
    │
    └── Event 1 · meeting · High confidence
          Date: Friday 18:00 (week of 14 March 2024)
          Location: Café Milano, Brussels
          Parties: Ahmed Hassan, Khalid Musa
          Source: evidence/email_002.txt
          Score: 0.91
          "meet you at Café Milano, Brussels at 18:00 on Friday to hand over the documents"
```

Written to `outputs/a3f9.../evidence_mindmap.html`.
Served at `GET /jobs/a3f9.../mindmap`.

Job status: `completed`, progress: 100%.

---

## 6. Configuration reference

All parameters are in `config.py` at the repo root. Change a value there and
every part of the pipeline that uses it will pick it up automatically.

| Parameter | Default | What it controls |
|---|---|---|
| `CONFIG.models.extraction_model` | `gemini-2.0-flash` | Model used to extract events from evidence |
| `CONFIG.models.lloom_scoring_model` | `gemini-2.5-flash` | More capable model used to score events against concepts |
| `CONFIG.models.finding_synthesis_model` | `gemini-2.0-flash` | Model used to write investigative findings |
| `CONFIG.extraction.evidence_files_per_batch` | `10` | Files per Gemini extraction call |
| `CONFIG.extraction.gemini_call_retry_limit` | `3` | Retries on extraction failure |
| `CONFIG.lloom.max_concepts` | `5` | Maximum concepts LLooM induces |
| `CONFIG.lloom.max_rerun_iterations` | `3` | Outlier rerun passes |
| `CONFIG.lloom.min_rows_required_for_induction` | `15` | Dataset is duplicated if fewer rows than this |
| `CONFIG.lloom.generic_concept_min_coverage_score` | `0.75` | Score threshold to count an event as "covered" |
| `CONFIG.lloom.generic_concept_max_coverage_fraction` | `0.5` | If a concept covers >50% of events it is considered generic |
| `CONFIG.lloom.use_mock_scoring` | `False` | Skip real LLooM, use fake scores (for testing) |
| `CONFIG.scores.min_score_to_include_in_synthesis` | `0.80` | Min score for finding synthesis |
| `CONFIG.scores.min_score_to_show_in_mindmap` | `0.80` | Min score to appear in the mindmap |
| `CONFIG.synthesis.gemini_temperature` | `0.2` | Creativity level for finding writing |
| `CONFIG.synthesis.gemini_call_retry_limit` | `5` | Retries for synthesis calls |
| `CONFIG.pipeline.step_retry_count` | `2` | Pipeline step retries on failure |
| `CONFIG.pipeline.review_poll_interval_seconds` | `6` | How often the runner checks for review submission |
| `CONFIG.pipeline.subprocess_timeout_seconds` | `900` | Kill timeout for stages 6 and 7 |

To enable mock LLooM mode (skips all real API calls for stages 5, 6, 7 —
useful for development):
```python
# config.py
CONFIG.lloom.use_mock_scoring = True
```

Or set the environment variable:
```bash
MOCK_LLOOM=true zsh backend_service/start_backend.sh
```

---

## 7. Job lifecycle states

```
queued
  │
  ▼
running  ←─────────────────────────────┐
  │                                    │  (review submitted)
  ▼                                    │
awaiting_review ───────────────────────┘
  │
  ▼
running  (continues through stages 4–7)
  │
  ├──► completed
  │
  ├──► failed  (unhandled error, retries exhausted)
  │
  └──► cancelled  (analyst clicked Cancel at any point)
```

At any point while the job is `running` or `awaiting_review`, the analyst can
call `POST /jobs/{id}/cancel`. The runner checks the cancel flag between every
stage. If it is set, the pipeline stops cleanly and the job is marked
`cancelled`.
