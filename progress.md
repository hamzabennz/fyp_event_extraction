# Implementation Progress

_Last updated: 2026-04-06_

> **How to resume:** Read `plan.md` for the full plan, then check the first unchecked box below to know where to continue. Each completed item notes what was done and verified.

---

## Phase 0: Documentation
- [x] `plan.md` created at project root
- [x] `theoretical.md` created at project root
- [x] `progress.md` created at project root
- [x] `CLAUDE.md` updated with Knowledge Graph section

## Phase 1: Neo4j Docker Setup
- [x] `docker-compose.yml` created
- [x] `env.example` updated (append NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
- [x] `config.py` Neo4jConfig dataclass added
- [x] `neo4j==5.26.0` added to `backend_service/requirements.txt`
- [x] Docker container starts: `docker compose up -d` verified OK (container `hamza_neo4j` running)
- [x] Python driver connectivity verified (`uv run` — "Neo4j OK")
- [x] **Committed to `anis_graph` branch**

## Phase 2: Graph Builder Service
- [x] `backend_service/app/graph_builder.py` created
- [x] `ensure_indexes()` creates Neo4j indexes on first call
- [x] `insert_job_events()` inserts Events, Persons, Locations, and all relationships
- [x] `clear_job_graph()` removes a job's nodes correctly
- [x] `get_graph_data_for_job()` returns valid {nodes, links} JSON
- [x] Resilience: Neo4j down → warning logged, no crash (lazy driver returns None)
- [x] Standalone test passes: 6 nodes, 5 links from 2 events ✓
- [x] **Committed to `anis_graph` branch**

## Phase 3: Pipeline Integration
- [x] `DEFAULT_STEPS` in `store.py` updated (`build_knowledge_graph` added before `lloom_scoring`)
- [x] `runner.py` `_run_parallel_steps()` function added (ThreadPoolExecutor, graph non-fatal)
- [x] `settings.py` step_retries updated (`build_knowledge_graph: 1`)
- [x] Full pipeline run completes without regression
- [x] Both `build_knowledge_graph` and `lloom_scoring` visible in UI step list
- [x] Both steps show as running simultaneously
- [x] **Committed to `anis_graph` branch**

## Phase 4: Contradiction Detection
- [x] `backend_service/app/contradiction_detector.py` created
- [x] Temporal overlap algorithm (Algorithm 1) implemented
- [x] Border/domestic rule (Algorithm 2, Rule A) implemented
- [x] Contact denial rule (Algorithm 2, Rule B) implemented
- [x] `[:CONTRADICTS]` edges written to Neo4j after detection
- [x] Fuzzy name matching working (Tier 1 normalization + Tier 2 token overlap)
- [x] Smoke test: 2 contradictions detected correctly from test data ✓
- [x] API test: `GET /jobs/{id}/contradictions` returns expected format
- [x] **Committed to `anis_graph` branch**

## Phase 5: API Routes
- [x] `GET /jobs/{job_id}/graph-data` added to `main.py`
- [x] `GET /jobs/{job_id}/contradictions` added to `main.py`
- [x] `POST /jobs/{job_id}/graph-data/refresh` added to `main.py`
- [x] `GET /graph/{job_id}` added to `main.py` (serves graph.html)
- [x] All routes return 404 for unknown job_id ✓
- [x] All routes return gracefully when Neo4j is down (neo4j_available: false) ✓
- [x] **Committed to `anis_graph` branch**

## Phase 6: D3.js Graph Visualization
- [x] `backend_service/app/static/graph.html` created
- [x] Page loads and fetches graph data from API
- [x] Force simulation runs; nodes visible and draggable
- [x] Node colors correct by event type (D3 ordinal scale, 16 colors)
- [x] Person nodes render as diamonds, Location as triangles
- [x] CONTRADICTS edges render in red with pulsing animation
- [x] Click on node → side panel shows full details including type_specific_fields
- [x] Universal search bar implemented (persons, places, event types, keywords)
- [x] Event type filter chips work (toolbar)
- [x] Contradiction toggle shows/hides red edges
- [x] Job selector dropdown populates and switches jobs
- [x] Zoom/pan works (D3 zoom behavior)
- [x] Status bar shows node/edge/contradiction counts
- [x] Refresh button calls POST /graph-data/refresh
- [x] Error/empty states handled
- [x] **Filter bar added** — horizontal Events / Persons / Locations pills at top of page; clicking a pill opens a scrollable chip row showing all individual nodes as small toggleable buttons; count badge updates to `visible/total` when filtered; "✕ Reset" clears all filters
- [x] Visual QA in browser ✓
- [x] **Committed to `anis_graph` branch**

## Phase 7: Final Documentation Pass
- [x] All progress.md checkboxes ticked
- [x] `CLAUDE.md` Knowledge Graph section verified accurate
- [x] `progress.md` completion timestamp added — **2026-04-06, all phases complete**
- [x] **Final push to `anis_graph` branch**
