"""
graph_builder.py — Neo4j knowledge graph service for per-job events.

Schema
------
(:Job {job_id})
(:Event {id, job_id, type, date_time, location, confidence_score,
         narrative, snippet, source_file, tsf_*})
(:Person {name})
(:Location {name})

(:Job)-[:HAS_EVENT]->(:Event)
(:Person)-[:PARTICIPANT_IN]->(:Event)
(:Event)-[:LOCATED_AT]->(:Location)
(:Event)-[:CONTRADICTS {type, description, severity}]->(:Event)  ← added by contradiction detector
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make repo root importable so config.py is always findable.
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from config import CONFIG  # noqa: E402

logger = logging.getLogger(__name__)

# ── lazy driver singleton ─────────────────────────────────────────────────────
# On failure we do NOT permanently cache None — we retry after a cooldown so
# that the backend can recover when Neo4j starts later.

_driver = None          # neo4j.Driver | None
_connected = False      # True only after a successful verify_connectivity()
_last_attempt: float = 0.0
_RETRY_COOLDOWN = 20    # seconds between reconnection attempts


def _get_driver():
    """Return a live Neo4j driver, or None if the database is unreachable."""
    global _driver, _connected, _last_attempt

    # Already have a working driver — return immediately.
    if _connected and _driver is not None:
        return _driver

    if not CONFIG.neo4j.enabled:
        return None

    # Throttle reconnection attempts so a down Neo4j doesn't block every request.
    now = time.monotonic()
    if now - _last_attempt < _RETRY_COOLDOWN:
        return None
    _last_attempt = now

    try:
        from neo4j import GraphDatabase  # local import — avoids hard dep at startup

        uri      = os.environ.get("NEO4J_URI",      CONFIG.neo4j.uri)
        user     = os.environ.get("NEO4J_USER",     CONFIG.neo4j.user)
        password = os.environ.get("NEO4J_PASSWORD", CONFIG.neo4j.password)

        # connection_timeout is not a valid kwarg in neo4j driver 5.x;
        # pass only uri + auth to stay compatible.
        drv = GraphDatabase.driver(uri, auth=(user, password))
        drv.verify_connectivity()
        logger.info("Neo4j connected at %s", uri)
        _driver = drv
        _connected = True
    except Exception as exc:
        logger.warning("Neo4j unavailable — will retry in %ds: %s", _RETRY_COOLDOWN, exc)
        _driver = None
        _connected = False

    return _driver


def reset_driver() -> None:
    """Force re-initialisation on next call (useful for tests and manual refresh)."""
    global _driver, _connected, _last_attempt
    if _driver is not None:
        try:
            _driver.close()
        except Exception:
            pass
    _driver = None
    _connected = False
    _last_attempt = 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Convert a value to float, returning 0.0 for non-numeric strings like 'High'."""
    try:
        return float(value or 0.0)
    except (ValueError, TypeError):
        mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
        return mapping.get(str(value).lower().strip(), 0.0)


def _norm_name(name: str) -> str:
    """Normalise a party name: title-case, collapse whitespace."""
    return " ".join(name.strip().title().split())


def _event_props(job_id: str, event: dict) -> dict[str, Any]:
    """Flatten an event dict into a Neo4j-friendly property map."""
    tsf = event.get("type_specific_fields") or {}
    props: dict[str, Any] = {
        "id": str(event.get("id", "")),
        "job_id": job_id,
        "type": str(event.get("type", "")),
        "date_time": str(event.get("date_time", "") or ""),
        "location": str(event.get("location", "") or ""),
        "confidence_score": _safe_float(event.get("confidence_score")),
        "narrative": str(event.get("narrative", "") or ""),
        "snippet": str(event.get("snippet", "") or ""),
        "source_file": str(event.get("source_file", "") or ""),
        "justification": str(event.get("justification", "") or ""),
    }
    for key, value in tsf.items():
        props[f"tsf_{key}"] = str(value) if value is not None else ""
    return props


# ── public API ────────────────────────────────────────────────────────────────

def ensure_indexes(driver=None) -> None:
    """Create indexes and constraints (idempotent, safe to call repeatedly)."""
    if driver is None:
        driver = _get_driver()
    if driver is None:
        return

    statements = [
        "CREATE INDEX event_id IF NOT EXISTS FOR (e:Event) ON (e.id)",
        "CREATE INDEX event_job_id IF NOT EXISTS FOR (e:Event) ON (e.job_id)",
        "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.type)",
        "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
        "CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name)",
        "CREATE INDEX job_id IF NOT EXISTS FOR (j:Job) ON (j.job_id)",
    ]
    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
            except Exception as exc:
                logger.warning("Index creation warning: %s", exc)


def clear_job_graph(job_id: str, driver=None) -> None:
    """Remove all nodes and relationships belonging to a specific job."""
    if driver is None:
        driver = _get_driver()
    if driver is None:
        return

    with driver.session() as session:
        # Detach-delete all Event nodes for this job (also removes their edges)
        session.run(
            "MATCH (e:Event {job_id: $job_id}) DETACH DELETE e",
            job_id=job_id,
        )
        # Remove the Job node itself
        session.run(
            "MATCH (j:Job {job_id: $job_id}) DETACH DELETE j",
            job_id=job_id,
        )


def insert_job_events(job_id: str, events: list[dict], driver=None) -> None:
    """
    Insert a job's approved events into the Neo4j graph.

    Creates:
      - (:Job) node
      - (:Event) nodes with all flattened properties
      - (:Person) nodes (MERGE — shared across jobs)
      - (:Location) nodes (MERGE — shared across jobs)
      - [:HAS_EVENT], [:PARTICIPANT_IN], [:LOCATED_AT] relationships
    """
    if driver is None:
        driver = _get_driver()
    if driver is None:
        logger.warning("insert_job_events: Neo4j unavailable, skipping")
        return

    ensure_indexes(driver)

    with driver.session() as session:
        # Upsert Job node
        session.run(
            "MERGE (j:Job {job_id: $job_id})",
            job_id=job_id,
        )

        for event in events:
            props = _event_props(job_id, event)
            event_id = props["id"]

            # Upsert Event node
            session.run(
                "MERGE (e:Event {id: $id, job_id: $job_id}) SET e += $props",
                id=event_id,
                job_id=job_id,
                props=props,
            )

            # Link Job → Event
            session.run(
                """
                MATCH (j:Job {job_id: $job_id})
                MATCH (e:Event {id: $event_id, job_id: $job_id})
                MERGE (j)-[:HAS_EVENT]->(e)
                """,
                job_id=job_id,
                event_id=event_id,
            )

            # Location node + Event → Location edge
            location = props["location"].strip()
            if location:
                session.run(
                    """
                    MERGE (l:Location {name: $name})
                    WITH l
                    MATCH (e:Event {id: $event_id, job_id: $job_id})
                    MERGE (e)-[:LOCATED_AT]->(l)
                    """,
                    name=location,
                    event_id=event_id,
                    job_id=job_id,
                )

            # Person nodes + Person → Event edges
            parties = event.get("parties") or []
            for raw_name in parties:
                name = _norm_name(str(raw_name))
                if not name:
                    continue
                session.run(
                    """
                    MERGE (p:Person {name: $name})
                    WITH p
                    MATCH (e:Event {id: $event_id, job_id: $job_id})
                    MERGE (p)-[:PARTICIPANT_IN]->(e)
                    """,
                    name=name,
                    event_id=event_id,
                    job_id=job_id,
                )


def get_graph_data_for_job(job_id: str, driver=None) -> dict:
    """
    Return {nodes: [...], links: [...]} suitable for D3.js force simulation.

    Node format:
      {id, label, group, type, job_id?, location?, narrative?, snippet?,
       confidence_score?, date_time?, source_file?, tsf_*}

    Link format:
      {source, target, rel_type}
    """
    if driver is None:
        driver = _get_driver()
    if driver is None:
        return {"nodes": [], "links": [], "neo4j_available": False}

    nodes: list[dict] = []
    links: list[dict] = []
    seen_nodes: set[str] = set()

    with driver.session() as session:
        # ── Event nodes ───────────────────────────────────────────────────────
        result = session.run(
            "MATCH (e:Event {job_id: $job_id}) RETURN e",
            job_id=job_id,
        )
        for record in result:
            e = dict(record["e"])
            node_id = f"event_{e['id']}"
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    **e,                            # spread first (includes id="0")
                    "id": node_id,                  # override to "event_0"
                    "label": e.get("type", "event"),
                    "group": "event",
                })

        # ── Person nodes ──────────────────────────────────────────────────────
        result = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id})
            RETURN DISTINCT p.name AS name
            """,
            job_id=job_id,
        )
        for record in result:
            node_id = f"person_{record['name']}"
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": record["name"],
                    "group": "person",
                    "name": record["name"],
                })

        # ── Location nodes ────────────────────────────────────────────────────
        result = session.run(
            """
            MATCH (e:Event {job_id: $job_id})-[:LOCATED_AT]->(l:Location)
            RETURN DISTINCT l.name AS name
            """,
            job_id=job_id,
        )
        for record in result:
            node_id = f"location_{record['name']}"
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": record["name"],
                    "group": "location",
                    "name": record["name"],
                })

        # ── PARTICIPANT_IN edges ──────────────────────────────────────────────
        result = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id})
            RETURN p.name AS person, e.id AS event_id
            """,
            job_id=job_id,
        )
        for record in result:
            links.append({
                "source": f"person_{record['person']}",
                "target": f"event_{record['event_id']}",
                "rel_type": "PARTICIPANT_IN",
            })

        # ── LOCATED_AT edges ──────────────────────────────────────────────────
        result = session.run(
            """
            MATCH (e:Event {job_id: $job_id})-[:LOCATED_AT]->(l:Location)
            RETURN e.id AS event_id, l.name AS location
            """,
            job_id=job_id,
        )
        for record in result:
            links.append({
                "source": f"event_{record['event_id']}",
                "target": f"location_{record['location']}",
                "rel_type": "LOCATED_AT",
            })

        # ── CONTRADICTS edges ─────────────────────────────────────────────────
        result = session.run(
            """
            MATCH (e1:Event {job_id: $job_id})-[r:CONTRADICTS]->(e2:Event {job_id: $job_id})
            RETURN e1.id AS id1, e2.id AS id2,
                   r.type AS type, r.description AS description, r.severity AS severity
            """,
            job_id=job_id,
        )
        for record in result:
            links.append({
                "source": f"event_{record['id1']}",
                "target": f"event_{record['id2']}",
                "rel_type": "CONTRADICTS",
                "contradiction_type": record["type"],
                "description": record["description"],
                "severity": record["severity"],
            })

    return {"nodes": nodes, "links": links, "neo4j_available": True}
