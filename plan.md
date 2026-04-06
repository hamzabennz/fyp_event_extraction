# Knowledge Graph & Visualization — Implementation Plan

## Overview

This plan extends the forensic event extraction pipeline with:
1. **Neo4j knowledge graph** — each job's approved events are stored as a graph after human review
2. **Parallel pipeline step** — graph construction runs alongside LLooM scoring after `build_csv`
3. **Contradiction detection** — graph-based algorithms detect temporal and logical inconsistencies between events
4. **D3.js visualization** — per-job force-directed graph at `/graph/{job_id}` with universal search, filters, and contradiction highlighting

Branch: `anis_graph`. Test before moving to next phase. Commit + push after each verified phase.

---

## Theoretical Background

### Knowledge Graph Model

A knowledge graph represents entities (persons, locations, events) as nodes and their relationships as edges. In this system:
- **Event nodes** encode all extracted forensic event data including type-specific fields
- **Person nodes** represent participants; shared across jobs (same person across multiple cases)
- **Location nodes** represent places; shared across jobs
- **Edges** encode participation (`PARTICIPANT_IN`), location (`LOCATED_AT`), and discovered contradictions (`CONTRADICTS`)

**Why Neo4j over relational databases:**
A relational model would require multi-table JOINs to answer questions like "Show all events involving person X and the other people they interacted with." In a property graph, this is a single multi-hop Cypher traversal. Graph databases are naturally suited to forensic link analysis where the connections between entities carry as much investigative meaning as the entities themselves.

### Contradiction Detection

**Definition:** A contradiction occurs when two or more events, when combined, violate a logical, temporal, or physical constraint about the world.

#### Algorithm 1 — Temporal Overlap Detection

**Theoretical basis:** Allen's Interval Algebra (1983) defines 13 possible temporal relations between time intervals. The "overlap," "during," and "equal" relations, when combined with different locations for the same person, create a physical impossibility (a person cannot be in two places simultaneously).

**Implementation:**
1. Cypher query retrieves pairs of events where the same person participates in events at different locations
2. Python post-processing parses the LLM-extracted `date_time` strings using `dateutil.parser.parse(fuzzy=True)` — chosen because LLM output timestamps are inconsistently formatted
3. Pairs within a 2-hour window at different locations are flagged as contradictions
4. Severity: **high** if same calendar day, **medium** if within 2 hours

**Complexity:** O(P × E²) where P = number of unique persons, E = average events per person. Practical for forensic datasets (typically < 1000 events per case).

#### Algorithm 2 — Constraint Satisfaction (CSP) Rules

**Theoretical basis:** Constraint Satisfaction Problems (CSP) encode domain knowledge as explicit rules over variables. Each event is a variable; constraints encode what cannot simultaneously be true.

**Rule A — Border/Domestic Conflict:** A person cannot be at a border crossing AND at a domestic location on the same day. This is encoded as a Cypher pattern match: `(p:Person)-[:PARTICIPANT_IN]->(bc:Event {type:'border_crossing'})` joined with a domestic event for the same person on the same day.

**Rule B — Contact Denial:** An event narrative containing "denied" or "deny" combined with a communication event (`phone_call`, `digital_communication`) for the same person and period constitutes a contradiction. Encoded as a keyword search on the `narrative` property joined with event type filtering.

**Rule categories for future extension:**
- Financial constraints: transaction amount exceeds declared income
- Sequential constraints: activity after declared departure from jurisdiction

### Universal Graph Search

**Definition:** Entity-centric search retrieves all graph nodes whose properties match a query, then expands to their immediate neighborhood to reveal relational context.

**Theoretical basis:** The "focus + context" visualization principle (Furnas, 1986 — Generalized Fisheye Views) argues that the most useful visualizations show selected items at full detail while preserving surrounding context at reduced detail. This prevents the analyst from losing orientation in the graph while investigating a specific entity.

**Implementation:**
- **Search scope:** node `label` (display name), `type` (event category), `location`, `narrative`, `snippet`, and all `type_specific_fields` values — a single query searches across persons, places, organizations, event types, and free-text keywords simultaneously
- **Highlighting:** matched nodes at full opacity → their direct neighbors at 70% → all other nodes dimmed to 15%
- **Example queries:**
  - `"John Doe"` → finds the Person node + all events they participated in
  - `"Paris"` → finds the Location node + all events that occurred there + all persons at those events
  - `"phone_call"` → finds all phone call events + their participants
  - `"suspicious"` → finds events whose narrative contains the word + their connected entities

---

## Implementation Phases

### Phase 0: Documentation (this file + progress.md + theoretical.md)
Write all planning and theoretical documents before coding begins.

### Phase 1: Neo4j Docker Setup

**Files:**
- `docker-compose.yml` (CREATE)
- `env.example` (MODIFY — append 3 lines)
- `config.py` (MODIFY — add `Neo4jConfig` dataclass)
- `backend_service/requirements.txt` (MODIFY — add `neo4j==5.26.0`)

**docker-compose.yml:**
```yaml
version: "3.9"
services:
  neo4j:
    image: neo4j:5.18
    container_name: hamza_neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: "${NEO4J_USER}/${NEO4J_PASSWORD}"
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
      NEO4J_server_memory_heap_initial__size: "512m"
      NEO4J_server_memory_heap_max__size: "1G"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_plugins:/plugins
volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_plugins:
```

**config.py Neo4jConfig:**
```python
@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "changeme"
    enabled: bool = True
    connection_timeout_seconds: int = 5
```

**Verification:**
```bash
docker compose up -d && sleep 25
python3 -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j','changeme')); d.verify_connectivity(); print('OK')"
```

---

### Phase 2: Graph Builder Service

**File:** `backend_service/app/graph_builder.py` (CREATE)

**Neo4j Schema:**
```
(:Job {job_id})
(:Event {id, job_id, type, date_time, location, confidence_score, narrative, snippet, source_file, tsf_*})
(:Person {name})
(:Location {name})

(:Job)-[:HAS_EVENT]->(:Event)
(:Person)-[:PARTICIPANT_IN]->(:Event)
(:Event)-[:LOCATED_AT]->(:Location)
(:Event)-[:CONTRADICTS {type, description, severity}]->(:Event)  ← added by Phase 4
```

**Key design decisions:**
- `type_specific_fields` flattened as `tsf_` prefixed properties (e.g., `tsf_amount`, `tsf_duration`)
- `Person` and `Location` nodes use MERGE (shared across jobs for cross-case analysis)
- Driver is lazy and returns `None` on connection failure (non-crashing)
- Party names normalized at insertion: `" ".join(name.strip().title().split())`

**Functions:**
```python
def _get_driver() -> Driver | None          # lazy, catches all connection errors
def ensure_indexes(driver) -> None          # create indexes on first call
def clear_job_graph(job_id, driver) -> None
def insert_job_events(job_id: str, events: list[dict]) -> None
def get_graph_data_for_job(job_id: str) -> dict  # {nodes, links} for D3.js
```

---

### Phase 3: Pipeline Integration

**Modified files:**
- `backend_service/app/store.py` — add `build_knowledge_graph` to `DEFAULT_STEPS` (before `lloom_scoring`)
- `backend_service/app/runner.py` — replace sequential `lloom_scoring` with `_run_parallel_steps()`
- `backend_service/app/settings.py` — add `"build_knowledge_graph": 1` to `step_retries`

**Parallel execution using `concurrent.futures.ThreadPoolExecutor`** (NOT asyncio — `run_pipeline()` is synchronous):
- Graph failures are NON-FATAL: log, mark step failed, continue pipeline
- LLooM failures ARE fatal: re-raise after both threads complete

**Progress %:** Both start at 65%; graph done at 75%; LLooM done at 80%

---

### Phase 4: Contradiction Detection

**File:** `backend_service/app/contradiction_detector.py` (CREATE)

**Main function:** `detect_and_store_contradictions(job_id: str) -> list[dict]`

**Return format:**
```python
{
  "event1_id": str, "event2_id": str,
  "type": "temporal_overlap" | "border_domestic" | "contact_denial",
  "description": str, "severity": "high" | "medium" | "low",
  "persons_involved": [str]
}
```

Writes `[:CONTRADICTS]` edges to Neo4j after detection.

---

### Phase 5: API Routes

Add to `backend_service/app/main.py`:

| Route | Purpose |
|---|---|
| `GET /jobs/{job_id}/graph-data` | D3.js nodes+links JSON |
| `GET /jobs/{job_id}/contradictions` | Contradiction list |
| `POST /jobs/{job_id}/graph-data/refresh` | Re-insert + re-detect |
| `GET /graph/{job_id}` | Serves graph.html |

---

### Phase 6: D3.js Graph Visualization

**File:** `backend_service/app/static/graph.html` (CREATE)

**Layout:** Topbar (title + job selector) → Toolbar (search + filters + contradiction toggle) → Main (sidebar detail panel + SVG canvas) → Footer status bar

**Node encoding:** Events=circles (16 colors by type), Persons=diamonds, Locations=triangles; size ∝ degree

**Edge encoding:** PARTICIPANT_IN=gray, LOCATED_AT=blue dashed, CONTRADICTS=red animated glow

**Universal search:** case-insensitive substring match across label, type, location, narrative, snippet, all tsf_ values → focus+context highlighting

**Job ID:** extracted from `window.location.pathname.split('/').pop()`

**Note:** Use `frontend-design` skill when writing this file.

---

### Phase 7: Final Documentation Pass

- Ensure all `progress.md` checkboxes ticked
- CLAUDE.md updated with new section
- Add completion timestamp to `progress.md`

---

## File Change Summary

| File | Action |
|---|---|
| `docker-compose.yml` | CREATE |
| `env.example` | MODIFY |
| `config.py` | MODIFY |
| `backend_service/requirements.txt` | MODIFY |
| `backend_service/app/graph_builder.py` | CREATE |
| `backend_service/app/contradiction_detector.py` | CREATE |
| `backend_service/app/runner.py` | MODIFY |
| `backend_service/app/store.py` | MODIFY |
| `backend_service/app/settings.py` | MODIFY |
| `backend_service/app/main.py` | MODIFY |
| `backend_service/app/static/graph.html` | CREATE |
| `plan.md` | CREATE |
| `progress.md` | CREATE |
| `theoretical.md` | CREATE |
| `CLAUDE.md` | MODIFY |

---

## End-to-End Verification

```bash
# Phase 1
docker compose up -d && sleep 25
python3 -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j','changeme')); d.verify_connectivity(); print('Neo4j OK')"

# Phase 2 (standalone test)
EVENTS_FILE=$(ls backend_service/runtime/outputs/*/EVENTS.json 2>/dev/null | head -1)
python3 -c "
import json, os, sys
os.environ.setdefault('NEO4J_URI','bolt://localhost:7687')
os.environ.setdefault('NEO4J_USER','neo4j')
os.environ.setdefault('NEO4J_PASSWORD','changeme')
sys.path.insert(0,'.')
from backend_service.app.graph_builder import insert_job_events
events = json.loads(open('$EVENTS_FILE').read())
insert_job_events('test-manual', events)
print(f'Inserted {len(events)} events OK')
"

# Phase 3: submit a job via UI, verify both steps run in parallel

# Phase 4
curl -s http://localhost:8000/jobs/{JOB_ID}/contradictions | python3 -m json.tool

# Phase 5
curl -s http://localhost:8000/jobs/{JOB_ID}/graph-data | python3 -m json.tool | head -40
curl -o /dev/null -w "%{http_code}" http://localhost:8000/graph/{JOB_ID}

# Phase 6: open http://localhost:8000/graph/{JOB_ID} in browser

# Resilience test
docker compose stop neo4j
# Submit new job — must complete; build_knowledge_graph shows "failed (non-fatal)"
docker compose start neo4j
```
