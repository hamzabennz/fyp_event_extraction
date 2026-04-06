"""
contradiction_detector.py — Graph-based contradiction detection.

Algorithms
----------
1. Temporal Overlap  — same person at different locations within a 2-hour window
2. Border/Domestic   — person at border crossing AND domestic location same day
3. Contact Denial    — narrative containing "denied/deny" + communication event same day

After detection, [:CONTRADICTS] edges are written to Neo4j.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── name matching helpers ──────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _names_match(a: str, b: str) -> bool:
    """Tier-1: exact normalised match. Tier-2: token overlap (≥2 tokens shared)."""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return False
    return len(tokens_a & tokens_b) >= 2


# ── date parsing ───────────────────────────────────────────────────────────────

def _parse_dt(raw: str) -> datetime | None:
    if not raw or raw.strip() in ("", "unknown", "Unknown", "N/A"):
        return None
    try:
        from dateutil import parser as dtparser
        return dtparser.parse(raw, fuzzy=True)
    except Exception:
        return None


# ── main entry point ──────────────────────────────────────────────────────────

def detect_and_store_contradictions(job_id: str) -> list[dict]:
    """
    Run all contradiction algorithms against a job's graph data.
    Writes [:CONTRADICTS] edges to Neo4j and returns a list of contradiction dicts.

    Return format per contradiction:
      {event1_id, event2_id, type, description, severity, persons_involved}
    """
    from .graph_builder import _get_driver, _norm_name

    driver = _get_driver()
    if driver is None:
        logger.warning("contradiction_detector: Neo4j unavailable, skipping")
        return []

    contradictions: list[dict] = []

    try:
        contradictions += _detect_temporal_overlap(job_id, driver)
        contradictions += _detect_border_domestic(job_id, driver)
        contradictions += _detect_contact_denial(job_id, driver)
        _write_contradicts_edges(job_id, contradictions, driver)
    except Exception as exc:
        logger.warning("contradiction_detector error: %s", exc)

    return contradictions


# ── Algorithm 1: Temporal Overlap ─────────────────────────────────────────────

def _detect_temporal_overlap(job_id: str, driver) -> list[dict]:
    """
    Flag pairs of events where the same person participates in events at
    different locations within a 2-hour window (Allen's Interval Algebra).
    """
    results: list[dict] = []

    with driver.session() as session:
        records = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id})
            WHERE e.location IS NOT NULL AND e.location <> ''
            RETURN p.name AS person, e.id AS event_id,
                   e.date_time AS date_time, e.location AS location
            ORDER BY p.name
            """,
            job_id=job_id,
        ).data()

    # Group by person
    from collections import defaultdict
    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_person[row["person"]].append(row)

    seen_pairs: set[tuple] = set()

    for person, events in by_person.items():
        for i, ev1 in enumerate(events):
            for ev2 in events[i + 1:]:
                if ev1["location"] == ev2["location"]:
                    continue
                dt1 = _parse_dt(ev1["date_time"])
                dt2 = _parse_dt(ev2["date_time"])
                if dt1 is None or dt2 is None:
                    continue
                diff = abs((dt1 - dt2).total_seconds())
                if diff > 7200:  # > 2 hours → not a conflict worth flagging
                    continue

                pair = tuple(sorted([ev1["event_id"], ev2["event_id"]]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                same_day = dt1.date() == dt2.date()
                severity = "high" if same_day else "medium"
                results.append({
                    "event1_id": ev1["event_id"],
                    "event2_id": ev2["event_id"],
                    "type": "temporal_overlap",
                    "description": (
                        f"{person} appears at '{ev1['location']}' and "
                        f"'{ev2['location']}' within {int(diff // 60)} minutes"
                    ),
                    "severity": severity,
                    "persons_involved": [person],
                })

    return results


# ── Algorithm 2a: Border/Domestic conflict ────────────────────────────────────

_DOMESTIC_TYPES = {
    "phone_call", "meeting", "digital_communication",
    "social_media_activity", "financial_transaction", "bank_transaction",
}

def _detect_border_domestic(job_id: str, driver) -> list[dict]:
    """
    Flag: person at a border_crossing AND at a domestic event on the same day.
    """
    results: list[dict] = []

    with driver.session() as session:
        border_rows = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id, type: 'border_crossing'})
            RETURN p.name AS person, e.id AS event_id, e.date_time AS date_time
            """,
            job_id=job_id,
        ).data()

        domestic_rows = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id})
            WHERE e.type IN $domestic_types
            RETURN p.name AS person, e.id AS event_id, e.date_time AS date_time, e.type AS type
            """,
            job_id=job_id,
            domestic_types=list(_DOMESTIC_TYPES),
        ).data()

    seen_pairs: set[tuple] = set()

    for br in border_rows:
        dt_b = _parse_dt(br["date_time"])
        if dt_b is None:
            continue
        for dr in domestic_rows:
            if not _names_match(br["person"], dr["person"]):
                continue
            dt_d = _parse_dt(dr["date_time"])
            if dt_d is None:
                continue
            if dt_b.date() != dt_d.date():
                continue

            pair = tuple(sorted([br["event_id"], dr["event_id"]]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            results.append({
                "event1_id": br["event_id"],
                "event2_id": dr["event_id"],
                "type": "border_domestic",
                "description": (
                    f"{br['person']} recorded at a border crossing and a "
                    f"domestic {dr['type']} event on the same day"
                ),
                "severity": "high",
                "persons_involved": [br["person"]],
            })

    return results


# ── Algorithm 2b: Contact Denial ──────────────────────────────────────────────

_CONTACT_TYPES = {"phone_call", "digital_communication", "meeting"}
_DENIAL_KEYWORDS = {"denied", "deny", "denies", "no contact", "never met"}

def _detect_contact_denial(job_id: str, driver) -> list[dict]:
    """
    Flag: event narrative contains denial language AND a contact event exists
    for the same person on the same day.
    """
    results: list[dict] = []

    with driver.session() as session:
        all_rows = session.run(
            """
            MATCH (p:Person)-[:PARTICIPANT_IN]->(e:Event {job_id: $job_id})
            RETURN p.name AS person, e.id AS event_id,
                   e.date_time AS date_time, e.type AS type,
                   e.narrative AS narrative
            """,
            job_id=job_id,
        ).data()

    denial_rows = [
        r for r in all_rows
        if r["narrative"] and any(kw in r["narrative"].lower() for kw in _DENIAL_KEYWORDS)
    ]
    contact_rows = [r for r in all_rows if r["type"] in _CONTACT_TYPES]

    seen_pairs: set[tuple] = set()

    for dr in denial_rows:
        dt_d = _parse_dt(dr["date_time"])
        for cr in contact_rows:
            if not _names_match(dr["person"], cr["person"]):
                continue
            if dr["event_id"] == cr["event_id"]:
                continue
            dt_c = _parse_dt(cr["date_time"])
            if dt_d is not None and dt_c is not None and dt_d.date() != dt_c.date():
                continue

            pair = tuple(sorted([dr["event_id"], cr["event_id"]]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            results.append({
                "event1_id": dr["event_id"],
                "event2_id": cr["event_id"],
                "type": "contact_denial",
                "description": (
                    f"{dr['person']} narrative denies contact but a "
                    f"{cr['type']} event is recorded for the same period"
                ),
                "severity": "medium",
                "persons_involved": [dr["person"]],
            })

    return results


# ── Write CONTRADICTS edges ───────────────────────────────────────────────────

def _write_contradicts_edges(job_id: str, contradictions: list[dict], driver) -> None:
    if not contradictions:
        return

    with driver.session() as session:
        for c in contradictions:
            try:
                session.run(
                    """
                    MATCH (e1:Event {id: $id1, job_id: $job_id})
                    MATCH (e2:Event {id: $id2, job_id: $job_id})
                    MERGE (e1)-[r:CONTRADICTS {type: $type}]->(e2)
                    SET r.description = $description,
                        r.severity    = $severity
                    """,
                    id1=c["event1_id"],
                    id2=c["event2_id"],
                    job_id=job_id,
                    type=c["type"],
                    description=c["description"],
                    severity=c["severity"],
                )
            except Exception as exc:
                logger.warning("Failed to write CONTRADICTS edge: %s", exc)
