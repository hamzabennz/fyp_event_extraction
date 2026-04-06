# Theoretical Background: Knowledge Graph, Contradiction Detection & Universal Search

_This document is intended for inclusion in the analyst's thesis. It provides academic-level explanations of the algorithms and design choices behind the graph intelligence layer of the forensic pipeline._

---

## 1. Knowledge Graph Model

### 1.1 Definition

A **knowledge graph** is a directed, labeled, property graph where:
- **Nodes** (vertices) represent entities in the domain
- **Edges** (relationships) represent typed connections between entities
- Both nodes and edges carry **properties** (key-value pairs) encoding additional attributes

In this forensic pipeline, the knowledge graph is stored in **Neo4j**, a native graph database that uses the **property graph model** and supports the **Cypher** declarative query language.

### 1.2 Schema

The graph schema for this system is:

```
(:Job {job_id})
    -[:HAS_EVENT]->
(:Event {
    id, job_id, type, date_time, location,
    confidence_score, narrative, snippet, source_file,
    tsf_* (type-specific fields, prefixed with "tsf_")
})
    -[:LOCATED_AT]->
(:Location {name})

(:Person {name})
    -[:PARTICIPANT_IN]->
(:Event)

(:Event)
    -[:CONTRADICTS {type, description, severity}]->
(:Event)
```

**Node types and their roles:**

| Node Label | Role | Scope |
|---|---|---|
| `Job` | Container for a pipeline run | Per job |
| `Event` | A single forensic event extracted from evidence | Per job |
| `Person` | An individual named in events | **Shared across jobs** |
| `Location` | A place named in events | **Shared across jobs** |

`Person` and `Location` nodes are shared across jobs via Cypher `MERGE` semantics. This enables cross-case analysis: the same individual appearing in multiple investigations is represented by a single node, with edges to events from different jobs. This is a core advantage of the graph model over per-job JSON artifacts.

### 1.3 Type-Specific Fields

The pipeline extracts 16 distinct event types, each with unique fields defined in `event_types_db.json`. These type-specific fields are stored as properties on `Event` nodes with a `tsf_` prefix to prevent key collisions with base properties:

| Event Type | Type-Specific Fields (stored as tsf_*) |
|---|---|
| `phone_call` | phone_number_sender, phone_number_receiver, duration, topic |
| `bank_transaction` | amount, currency, bank_name, transaction_method |
| `travel_movement` | transport_mode, ticket_number, purpose |
| `meeting` | agenda, outcome |
| `digital_communication` | sender_identifier, receiver_identifier, platform, content_summary, encryption_status |
| `physical_surveillance` | officer_or_source_id, observed_action, vehicle_details, clothing_description |
| `illicit_exchange` | item_description, quantity_or_weight, estimated_value, concealment_method |
| `border_crossing` | border_post_name, passport_number, travel_direction, luggage_inspection_result |
| `cyber_incident` | target_system_ip, attack_vector, data_impacted, timestamp_utc |
| `financial_transaction` | amount, currency, sender_entity, receiver_entity, platform, transaction_hash |
| `social_media_activity` | platform, account_username, post_content, engagement_count, visibility |
| `document_activity` | document_name, action, accessed_by, file_type, storage_location |
| `location_checkin` | gps_coordinates, accuracy_meters, checkin_service, device_id |
| `financial_record` | record_type, amount, payer, payee, purpose |
| `device_activity` | device_id, user_account, action_type, application, file_transferred |
| `relationship_establishment` | party_1, party_2, context, relationship_type |

### 1.4 Why a Graph Database?

A relational database could store this data, but answering forensic link-analysis queries would require complex multi-table JOINs. Consider the query: _"Find all persons connected within two degrees to a bank transaction over €50,000."_

- **Relational:** `SELECT ... FROM events JOIN event_parties ON ... JOIN events AS e2 JOIN event_parties AS ep2 ON ... WHERE amount > 50000` — exponential join cost as hop count increases
- **Neo4j Cypher:** `MATCH (e:Event {type:'bank_transaction'}) WHERE toFloat(e.tsf_amount) > 50000 MATCH (e)<-[:PARTICIPANT_IN*1..2]-(p:Person) RETURN p` — O(traversal) with index-backed starts

Graph databases offer **constant-time traversal** per hop (regardless of dataset size) because relationships are stored as direct pointers, not computed via JOIN operations. This is the core property that makes Neo4j ideal for forensic link analysis and contradiction detection.

---

## 2. Contradiction Detection

### 2.1 Definition

A **contradiction** in a forensic event graph occurs when two or more events, when combined, violate a logical, temporal, or physical constraint about the world.

Contradictions in forensic datasets may arise from:
- **Fabrication:** a witness or suspect constructs a false alibi
- **Error:** timestamps or locations were incorrectly recorded
- **Inference:** the investigator's timeline reconstruction contains gaps

The system detects contradictions automatically and presents them as red edges in the graph visualization, annotated with type and severity.

### 2.2 Algorithm 1 — Temporal Overlap Detection

#### Theoretical Basis

**Allen's Interval Algebra** (Allen, 1983) formalizes temporal reasoning by defining 13 mutually exclusive relations between two time intervals I₁ = [s₁, e₁] and I₂ = [s₂, e₂]:

| Relation | Symbol | Meaning |
|---|---|---|
| precedes | I₁ < I₂ | I₁ ends before I₂ starts |
| meets | I₁ m I₂ | I₁ ends exactly when I₂ starts |
| overlaps | I₁ o I₂ | I₁ starts before I₂ and they share a period |
| during | I₁ d I₂ | I₁ is entirely within I₂ |
| equals | I₁ = I₂ | Same start and end |
| ... and 8 inverse/symmetric variants | | |

For forensic contradiction detection, the relations **overlaps**, **during**, and **equals** — when combined with **different locations** for the same person — constitute a physical impossibility. A person cannot simultaneously participate in two events at geographically distinct locations.

#### Implementation

```
Step 1 (Cypher): For each job, retrieve all pairs of events (e1, e2) where:
   - The same Person node participates in both (via :PARTICIPANT_IN)
   - e1.location ≠ e2.location (both non-empty)
   - e1.id < e2.id (avoid duplicate pairs)

Step 2 (Python): For each candidate pair:
   - Parse e1.date_time and e2.date_time using dateutil.parser.parse(fuzzy=True)
   - If parsing fails for either event → skip (LLM timestamps are inconsistently formatted)
   - Compute |t1 - t2| in hours
   - If |t1 - t2| < 2 hours → contradiction detected

Step 3: Assign severity:
   - Same calendar day → HIGH
   - Within 2 hours → MEDIUM

Step 4 (Cypher): Create [:CONTRADICTS] edge between the two Event nodes
```

**Complexity:** O(P × E²) where P = number of unique persons and E = average events per person. For typical forensic datasets (< 500 events, < 100 persons), this is efficient. For larger datasets, add an index on `(Event.date_time)` and filter by date substring before parsing.

**Limitations:**
- Depends on the LLM correctly extracting `date_time` and `location` fields
- Does not model travel time between locations (a person 5 minutes apart by car is not a contradiction)
- Time zone handling is implicit (assumes all times are in the same zone)

#### Reference

Allen, J. F. (1983). "Maintaining knowledge about temporal intervals." *Communications of the ACM*, 26(11), 832–843.

---

### 2.3 Algorithm 2 — Constraint Satisfaction (CSP) Rules

#### Theoretical Basis

A **Constraint Satisfaction Problem (CSP)** is defined by:
- A set of **variables** X = {x₁, x₂, ..., xₙ}
- For each variable xᵢ, a **domain** Dᵢ of possible values
- A set of **constraints** C restricting which combinations of values are consistent

For forensic contradiction detection, each event is a variable; its domain is {true (occurred), false (did not occur), uncertain}. Constraints encode domain knowledge: rules that cannot all be simultaneously true.

Unlike Algorithm 1 (which is data-driven — it finds contradictions by comparing event properties), Algorithm 2 is **rule-driven**: an investigator encodes domain knowledge as explicit logical constraints, and the system checks whether the event data violates those constraints.

#### Rule A — Border Crossing + Simultaneous Domestic Event

**Constraint:** A person cannot be at an international border crossing AND at a domestic location on the same calendar day (excluding back-to-back crossings where the domestic event IS the destination).

**Cypher encoding:**
```cypher
MATCH (p:Person)-[:PARTICIPANT_IN]->(bc:Event {job_id: $job_id, type: 'border_crossing'})
MATCH (p)-[:PARTICIPANT_IN]->(other:Event {job_id: $job_id})
WHERE other.id <> bc.id
  AND other.type <> 'border_crossing'
  AND bc.location <> other.location
  AND substring(bc.date_time, 0, 10) = substring(other.date_time, 0, 10)
RETURN p.name, bc.id, other.id, bc.date_time, other.date_time
```

**Severity:** HIGH (physical impossibility if locations are geographically incompatible)

#### Rule B — Communication Denial vs. Communication Record

**Constraint:** An event whose narrative indicates denial of contact between two parties, combined with an existing communication event (`phone_call` or `digital_communication`) involving the same parties, constitutes a logical contradiction.

**Approach:** Since the LLM does not extract explicit "denial events" as a separate type, this rule uses keyword detection on the `narrative` property:

```cypher
MATCH (p:Person)-[:PARTICIPANT_IN]->(denial:Event {job_id: $job_id})
WHERE toLower(denial.narrative) CONTAINS 'denied' 
   OR toLower(denial.narrative) CONTAINS 'deny'
MATCH (p)-[:PARTICIPANT_IN]->(comm:Event {job_id: $job_id})
WHERE comm.type IN ['phone_call', 'digital_communication', 'meeting']
  AND comm.id <> denial.id
RETURN p.name, denial.id, comm.id
```

**Severity:** MEDIUM (depends on narrative interpretation accuracy)

#### Extensibility

The CSP framework is designed for easy extension. Future rules can be added by:
1. Writing a new Cypher query that captures the pattern
2. Adding it to the `_CSP_RULES` list in `contradiction_detector.py`
3. No other changes required

Planned future rules:
- **Financial:** transaction amount > declared income from financial records
- **Sequential:** activity after declared departure from jurisdiction
- **Communication frequency:** sudden spike in encrypted communications before an incident

#### References

Kumar, V. (1992). "Algorithms for constraint-satisfaction problems: A survey." *AI Magazine*, 13(1), 32–44.

---

### 2.4 Contradiction Severity Scale

| Severity | Definition | Action |
|---|---|---|
| HIGH | Physical impossibility (same person, two locations, same time) | Highlighted in bright red in graph; listed first in contradiction report |
| MEDIUM | Strong logical inconsistency (denial vs. communication record, border/domestic same day) | Orange highlight; listed in contradiction report |
| LOW | Weak inconsistency (possible but unusual timing or pattern) | Yellow highlight; listed but not prominently featured |

---

## 3. Universal Graph Search

### 3.1 Definition

**Universal graph search** (also called entity-centric or full-property search) retrieves all nodes in the knowledge graph whose properties match a user query, then expands the result to include the immediate neighbors of matched nodes.

This contrasts with **type-specific search** (e.g., "search only persons") by treating the graph as a unified semantic space where a single query can simultaneously surface a person named "Paris Hilton," an event located in "Paris," a financial record involving "Paris Bank," and a narrative mentioning "the Paris meeting."

### 3.2 Theoretical Basis: Focus + Context Visualization

**Furnas' Generalized Fisheye Views** (1986) established the principle that effective information displays should show:
- Items of direct interest at **full detail** (focus)
- Surrounding context at **reduced detail** (context)
- Items far from the focus at **minimal detail** or hidden

Applied to graph visualization, when an analyst searches for "John Doe":
- The matched `Person` node for John Doe: **full opacity (100%)**
- All `Event` nodes where John Doe participates: **high opacity (85%)**
- `Location` nodes connected to those events: **medium opacity (70%)**
- All other nodes: **dimmed (15%)**

This allows the analyst to see both the entity of interest and its immediate context, without losing the broader graph structure.

**Reference:** Furnas, G. W. (1986). "Generalized fisheye views." *Proceedings of CHI '86: Human Factors in Computing Systems*, 16–23.

### 3.3 Search Scope

The search covers the following node properties:

| Property | Node Types | Search Rationale |
|---|---|---|
| `label` (display name) | All | Direct name search |
| `type` | Event | Find all events of a specific type (e.g., "border_crossing") |
| `location` | Event | Find all events at a location (e.g., "Zurich") |
| `narrative` | Event | Free-text keyword search in the LLM-generated narrative |
| `snippet` | Event | Keyword search in the source evidence text |
| `tsf_*` (all type-specific fields) | Event | Search across specific fields (e.g., "bank_name", "platform") |

A single search query is matched as a **case-insensitive substring** against all of the above fields. This means:
- Searching "bank" will match events with `tsf_bank_name = "HSBC Bank"`, narratives mentioning "bank account," and event types like "bank_transaction"
- Searching "2024-03" will match all events in March 2024 via their `date_time` field
- Searching "encrypted" will match digital communication events whose `tsf_encryption_status = "encrypted"` as well as any narrative mentioning the word

### 3.4 Implementation

```javascript
function universalSearch(query) {
  const q = query.toLowerCase().trim();
  if (!q) { clearHighlight(); return; }

  // Match any node where any searchable property contains the query
  const matchedIds = new Set(
    graphData.nodes
      .filter(n => nodeMatchesQuery(n, q))
      .map(n => n.id)
  );

  // Expand: include direct neighbors of matched nodes
  const neighborIds = new Set();
  graphData.links.forEach(link => {
    const src = link.source.id || link.source;
    const tgt = link.target.id || link.target;
    if (matchedIds.has(src)) neighborIds.add(tgt);
    if (matchedIds.has(tgt)) neighborIds.add(src);
  });

  // Apply focus+context opacity
  nodeSelection.attr("opacity", d => {
    if (matchedIds.has(d.id)) return 1.0;        // focus
    if (neighborIds.has(d.id)) return 0.7;        // context
    return 0.15;                                   // background
  });
}

function nodeMatchesQuery(node, q) {
  const searchable = [
    node.label, node.type, node.location,
    node.narrative, node.snippet,
    ...Object.values(node.type_specific_fields || {})
  ];
  return searchable.some(v => v && String(v).toLowerCase().includes(q));
}
```

---

## 4. Graph Visualization Design

### 4.1 Force-Directed Layout

The graph uses a **force-directed layout** (Fruchterman & Reingold, 1991) implemented via D3.js v7's `forceSimulation`. Forces applied:

| Force | Parameters | Purpose |
|---|---|---|
| `forceLink` | distance=80 (PARTICIPANT_IN), 60 (CONTRADICTS), 100 (others) | Attract connected nodes |
| `forceManyBody` | strength=-300 | Repel all nodes from each other |
| `forceCenter` | canvas center | Prevent graph from drifting off-screen |
| `forceCollide` | radius = nodeSize + 8 | Prevent node overlap |

**Reference:** Fruchterman, T. M. J., & Reingold, E. M. (1991). "Graph drawing by force-directed placement." *Software: Practice and Experience*, 21(11), 1129–1164.

### 4.2 Visual Encoding

The system uses **pre-attentive visual features** (Ware, 2004) to encode node and edge types:

- **Shape:** distinguishes node type (circle=event, diamond=person, triangle=location) — processed before conscious attention
- **Color:** encodes event type for circles (16 distinct hues) — enables immediate categorical identification
- **Size:** encodes node degree (connection count) — larger nodes are more connected/more important
- **Stroke:** edge type encoded via line style (solid, dashed, animated) and color (gray, blue, red)

**Reference:** Ware, C. (2004). *Information Visualization: Perception for Design* (2nd ed.). Morgan Kaufmann.

### 4.3 Contradiction Visualization

Contradiction edges (`[:CONTRADICTS]`) are rendered with:
- **Red color** (#dc2626) — maximum salience
- **Animated stroke-dashoffset** — draws the eye to the edge, indicating an active alert
- **Glow SVG filter** (`feGaussianBlur` + `feMerge`) — creates a halo effect that makes the edge visible even when behind other elements
- **Increased stroke width** (2.5px vs 1px for normal edges) — additional visual weight

This multi-layered encoding ensures that contradictions are unmistakable even in dense graphs with many overlapping elements.
