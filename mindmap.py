"""
Evidence Corpus - NotebookLM Style Interactive Mindmap
Vertical tree: Root (left) → Concepts (center) → Events (right)
Each concept owns its vertical space. Smooth curved connectors.
Click concepts to expand/collapse. No overlap guaranteed.

Usage:
    python mindmap.py
"""

import pandas as pd
import json
import os
import re


def extract_date_from_text(text):
    if pd.isna(text):
        return ""
    date_match = re.search(
        r'(January|February|March|April|May|June|July|August|September|October'
        r'|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+\d{1,2},?\s+\d{4}', str(text)
    )
    return date_match.group(0) if date_match else ""


def main():
    print("📊 Evidence Corpus Analysis - NotebookLM Style Mindmap")
    print("=" * 70)

    csv_path = "score_results_combined.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df):,} total rows")

    df_filtered = df[df['score'] >= 0.80].copy()
    print(f"   ✓ Filtered to {len(df_filtered)} high-confidence rows (score ≥ 0.80)")

    if df_filtered.empty:
        print("⚠️  No data found!")
        return

    # Load provenance (source_file + snippet) from events_enriched.csv
    provenance = {}   # doc_id (str) → {source_file, snippet}
    enriched_path = "events_enriched.csv"
    if os.path.exists(enriched_path):
        enr = pd.read_csv(enriched_path)
        for _, r in enr.iterrows():
            provenance[str(int(r['id']))] = {
                'source_file': str(r['source_file']) if pd.notna(r.get('source_file')) else 'UNKNOWN',
                'snippet':     str(r['snippet'])     if pd.notna(r.get('snippet'))     else '',
            }
        print(f"   ✓ Loaded provenance for {len(enr)} events from {enriched_path}")
    else:
        print("   ℹ  events_enriched.csv not found — run events_to_csv.py to generate it")

    # Load EVENTS.json for rich metadata (type, parties, location, date_time)
    events_db = {}  # id (str) → full event dict
    events_json_path = "EVENTS.json"
    if os.path.exists(events_json_path):
        with open(events_json_path, 'r', encoding='utf-8') as f:
            events_list = json.load(f)
        for ev in events_list:
            events_db[str(ev.get('id', ''))] = ev
        print(f"   ✓ Loaded {len(events_db)} events from EVENTS.json")
    else:
        print("   ℹ  EVENTS.json not found — event metadata will be minimal")

    # Load findings from Phase 4 (optional — mindmap works without them)
    findings = {}
    findings_path = "findings.json"
    if os.path.exists(findings_path):
        with open(findings_path, 'r', encoding='utf-8') as f:
            findings = json.load(f)
        print(f"   ✓ Loaded findings for {len(findings)} concepts from findings.json")
    else:
        print("   ℹ  findings.json not found — run synthesize_findings.py for Phase 4 synthesis")

    concepts_data = {}
    for concept in sorted(df_filtered['concept_name'].dropna().unique()):
        rows = df_filtered[df_filtered['concept_name'] == concept]
        fd = findings.get(concept, {})
        concepts_data[concept] = {
            'count':    int(len(rows)),
            'avg_score': float(rows['score'].mean()),
            'finding':  fd.get('finding', ''),
            'strength': fd.get('strength', ''),
            'events':   []
        }
        seen = set()
        for _, row in rows.iterrows():
            doc_id = str(row['doc_id'])
            if doc_id not in seen:
                seen.add(doc_id)
                prov = provenance.get(doc_id, {})
                ev_meta = events_db.get(doc_id, {})
                raw_parties = ev_meta.get('parties', [])
                parties = [p for p in (raw_parties if isinstance(raw_parties, list) else []) if p and p != 'N/A']
                raw_tsf = ev_meta.get('type_specific_fields', {}) or {}
                type_specific_fields = {k: v for k, v in raw_tsf.items() if v and str(v) not in ('N/A', 'nan', 'null', '')}
                concepts_data[concept]['events'].append({
                    'doc_id':              doc_id,
                    'text':               str(row['text']) if pd.notna(row['text']) else "N/A",
                    'score':              float(row['score']),
                    'rationale':          str(row['rationale']) if pd.notna(row['rationale']) else "No analysis",
                    'date':               extract_date_from_text(row['text']),
                    'source_file':        prov.get('source_file', 'UNKNOWN'),
                    'snippet':            prov.get('snippet', ''),
                    'event_type':         ev_meta.get('type', '').replace('_', ' ').title(),
                    'narrative':          ev_meta.get('narrative', '') or '',
                    'parties':            parties,
                    'location':           ev_meta.get('location', '') or '',
                    'date_time':          ev_meta.get('date_time', '') or '',
                    'confidence_score':   ev_meta.get('confidence_score', '') or '',
                    'type_specific_fields': type_specific_fields,
                })

    print(f"   ✓ Organized {len(concepts_data)} concepts")
    print("\n🎨 Building interactive HTML...")

    source_files = _load_source_files(concepts_data)
    html_content = generate_html(concepts_data, source_files)
    output_path = "evidence_mindmap.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    total_ev = sum(len(c['events']) for c in concepts_data.values())
    print(f"\n✅ Done! Concepts: {len(concepts_data)}, Evidence items: {total_ev}")
    print(f"🌐 Open {output_path} in your browser.\n")


def _sanitize(text: str) -> str:
    """Remove non-UTF-8 surrogate/replacement characters that break JS string literals."""
    # Replace the Unicode replacement character and lone surrogates
    return text.replace('\ufffd', '?').encode('utf-8', errors='replace').decode('utf-8', errors='replace')


def _load_source_files(concepts_data: dict) -> dict:
    """Read every unique source_file referenced across all events into a dict."""
    paths = set()
    for c in concepts_data.values():
        for ev in c['events']:
            sf = ev.get('source_file', 'UNKNOWN')
            if sf and sf != 'UNKNOWN':
                paths.add(sf)
    loaded = {}
    missing = 0
    for path in sorted(paths):
        if os.path.exists(path):
            try:
                raw = open(path, encoding='utf-8', errors='replace').read()
                loaded[path] = _sanitize(raw)
            except Exception:
                missing += 1
        else:
            missing += 1
    print(f"   ✓ Embedded {len(loaded)} source files ({missing} missing/skipped)")
    return loaded


def generate_html(concepts_data, source_files=None):
    if source_files is None:
        source_files = {}
    total_events = sum(len(c['events']) for c in concepts_data.values())
    concepts_json = json.dumps(concepts_data, ensure_ascii=True)
    source_files_json = json.dumps(source_files, ensure_ascii=True)

    num_concepts = len(concepts_data)
    num_events   = total_events

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evidence Analysis – Interactive Mindmap</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f2f5;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.fullscreen-fab {
  position: fixed;
  top: 14px;
  right: 14px;
  z-index: 1200;
  background: rgba(49, 50, 68, 0.86);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 8px;
  padding: 7px 12px;
  font-size: 12px;
  cursor: pointer;
  backdrop-filter: blur(8px);
}
.fullscreen-fab:hover { background: rgba(69, 71, 90, 0.92); }

/* ── Header ─────────────────────────────────── */
.header {
  background: linear-gradient(135deg,#667eea,#764ba2);
  color: white;
  padding: 14px 40px;
  text-align: center;
  flex-shrink: 0;
  position: relative;
}
.header h1 { font-size:22px; font-weight:700; }
.header p  { font-size:12px; opacity:.9; margin-top:3px; }

.header-controls {
  position: absolute;
  top: 10px;
  right: 14px;
  display: flex;
  gap: 8px;
}

.ctrl-btn {
  background: rgba(255,255,255,0.2);
  color: white;
  border: 1px solid rgba(255,255,255,0.4);
  border-radius: 5px;
  padding: 4px 10px;
  font-size: 11px;
  cursor: pointer;
  transition: background .15s;
}
.ctrl-btn:hover { background: rgba(255,255,255,0.35); }

#header-collapsed {
  flex-shrink: 0;
}
.restore-btn {
  width: 100%;
  background: linear-gradient(135deg,#667eea,#764ba2);
  color: white;
  border: none;
  padding: 6px 20px;
  font-size: 12px;
  cursor: pointer;
  text-align: center;
}
.restore-btn:hover { opacity: .9; }

/* Fullscreen overrides */
:fullscreen #header { display: flex; flex-direction: column; align-items: center; }
:fullscreen .stats  { display: flex; }
:-webkit-full-screen #header { display: flex; flex-direction: column; align-items: center; }
:-webkit-full-screen .stats  { display: flex; }

/* ── Stats bar ───────────────────────────────── */
.stats {
  display: flex;
  justify-content: center;
  gap: 60px;
  padding: 12px 40px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  flex-shrink: 0;
}
.stat-box { text-align:center; }
.stat-number { font-size:22px; font-weight:700; color:#667eea; }
.stat-label  { font-size:10px; color:#999; text-transform:uppercase; letter-spacing:.5px; }

/* ── Canvas area ─────────────────────────────── */
#viewport {
  flex: 1;
  overflow: hidden;
  position: relative;
  cursor: grab;
  background-color: #f7f8fc;
  background-image:
    radial-gradient(circle, #d0d5e8 1px, transparent 1px);
  background-size: 28px 28px;
}
#viewport.grabbing { cursor: grabbing; }

#scene {
  position: absolute;
  top: 0; left: 0;
  transform-origin: 0 0;
}

/* ── SVG nodes ───────────────────────────────── */
.link {
  fill: none;
  stroke: #b0b8d0;
  stroke-width: 1.8;
}

.node { cursor: pointer; }
.node rect {
  rx: 6; ry: 6;
  stroke-width: 1.5;
  transition: filter .15s;
}
.node:hover rect { filter: drop-shadow(0 3px 8px rgba(0,0,0,.18)); }
.node text {
  pointer-events: none;
  user-select: none;
  dominant-baseline: middle;
  text-anchor: middle;
}

/* root */
.node-root rect  { fill:#667eea; stroke:#4a5fc1; }
.node-root text  { fill:white; font-size:13px; font-weight:700; }

/* concept – collapsed */
.node-concept rect         { fill:#7ec8e3; stroke:#3a9cbf; }
.node-concept text         { fill:white;  font-size:11px; font-weight:600; }

/* concept – expanded */
.node-concept.expanded rect { fill:#5dade2; stroke:#2874a6; }

/* event */
.node-event rect  { fill:#a8d5b5; stroke:#4caf50; }
.node-event text  { fill:#1a3a1a; font-size:10px; }
.node-event:hover rect { fill:#7ec89a; }

/* expand arrow on concept */
.expand-arrow {
  font-size: 10px;
  fill: rgba(255,255,255,.8);
  pointer-events: none;
  user-select: none;
  dominant-baseline: middle;
  text-anchor: start;
}

/* concept node that has a finding — gold top border */
.node-concept.has-finding rect { stroke: #f9a825; stroke-width: 2.5; }

/* finding badge dot on concept */
.finding-dot {
  pointer-events: none;
  user-select: none;
}

/* finding panel inside detail */
.d-finding {
  font-size:12px; line-height:1.7; color:#333;
  background: linear-gradient(135deg,#fffde7,#fff9c4);
  padding:12px 14px;
  border-radius:6px;
  border-left:4px solid #f9a825;
  margin-top:6px;
  white-space: pre-wrap;
}
.strength-badge {
  display:inline-block;
  padding:3px 10px; border-radius:14px;
  font-size:10px; font-weight:700;
  margin-bottom:8px;
}
.strength-STRONG       { background:#e8f5e9; color:#2e7d32; }
.strength-MODERATE     { background:#fff3e0; color:#e65100; }
.strength-CIRCUMSTANTIAL { background:#fce4ec; color:#c62828; }
.strength-UNKNOWN      { background:#f5f5f5; color:#757575; }

/* ── Hint bar ────────────────────────────────── */
.hint {
  background:#e8f0fe;
  border-top:1px solid #c5d2f6;
  padding:8px 20px;
  font-size:12px;
  color:#3c4a9e;
  flex-shrink:0;
  display:flex;
  justify-content:space-between;
  align-items:center;
}
.zoom-btns button {
  background:#667eea; color:white; border:none;
  border-radius:4px; padding:3px 10px;
  font-size:13px; cursor:pointer; margin-left:6px;
}
.zoom-btns button:hover { background:#764ba2; }

/* ── Detail panel ────────────────────────────── */
#detail {
  position:fixed; right:0; top:0;
  width:400px; height:100vh;
  background:white;
  border-left:1px solid #ddd;
  padding:24px 20px;
  overflow-y:auto;
  z-index:999;
  box-shadow:-4px 0 16px rgba(0,0,0,.12);
  display:none;
}
#detail.show { display:block; }
.d-close {
  position:absolute; top:12px; right:14px;
  background:none; border:none;
  font-size:22px; cursor:pointer; color:#aaa;
}
.d-close:hover { color:#333; }
.d-body { margin-top:36px; }
.d-badge {
  display:inline-block;
  padding:3px 9px; border-radius:14px;
  font-size:10px; font-weight:700;
  margin:0 4px 6px 0;
}
.badge-confirmed { background:#ffebee; color:#c62828; }
.badge-very-high { background:#fff3e0; color:#e65100; }
.badge-high      { background:#fffde7; color:#f57f17; }
.d-label { font-size:13px; font-weight:700; color:#444; margin:14px 0 6px; }
.d-text {
  font-size:12px; line-height:1.65; color:#555;
  background:#f8f9ff; padding:10px 12px;
  border-radius:5px; border-left:3px solid #667eea;
}
.d-rationale {
  font-size:11px; color:#777; line-height:1.55;
  padding-top:12px; border-top:1px solid #eee; margin-top:12px;
}
.d-field-table {
  width:100%; border-collapse:collapse; margin-top:2px;
  font-size:12px;
}
.d-field-table td {
  padding:5px 8px; vertical-align:top;
  border-bottom:1px solid #f0f0f0;
  line-height:1.5;
}
.d-field-table td:first-child {
  color:#888; font-weight:600; white-space:nowrap;
  width:38%; text-transform:capitalize;
}
.d-field-table td:last-child { color:#333; }
.d-party-pill {
  display:inline-block; background:#e8f5e9; color:#1a3a1a;
  border-radius:10px; padding:2px 8px; margin:2px 2px 0 0;
  font-size:11px;
}

/* ── Source viewer ───────────────────────────────── */
.src-btn {
  display:inline-block;
  margin-top:14px;
  padding:6px 14px;
  background:#667eea;
  color:white;
  border:none;
  border-radius:5px;
  font-size:11px;
  font-weight:600;
  cursor:pointer;
  transition:background .15s;
}
.src-btn:hover { background:#764ba2; }
.src-label {
  font-size:11px; color:#888;
  font-style:italic;
  margin-top:4px;
}
#source-viewer {
  position:fixed; left:0; top:0;
  width:calc(100vw - 400px);
  height:100vh;
  background:#1e1e2e;
  color:#cdd6f4;
  z-index:998;
  display:none;
  flex-direction:column;
  box-shadow: 4px 0 20px rgba(0,0,0,.3);
}
#source-viewer.show { display:flex; }
.sv-header {
  background:#313244;
  padding:10px 18px;
  display:flex;
  align-items:center;
  gap:12px;
  border-bottom:1px solid #45475a;
  flex-shrink:0;
}
.sv-filename {
  font-size:13px;
  font-weight:700;
  color:#cba6f7;
  font-family: monospace;
}
.sv-close {
  margin-left:auto;
  background:none;
  border:none;
  color:#a6adc8;
  font-size:20px;
  cursor:pointer;
}
.sv-close:hover { color:white; }
.sv-body {
  flex:1;
  overflow-y:auto;
  padding:20px 24px;
  font-family: 'Menlo', 'Consolas', monospace;
  font-size:12px;
  line-height:1.8;
  white-space:pre-wrap;
  word-break:break-word;
}
.sv-highlight {
  background: #ffd700;
  color: #1e1e2e;
  border-radius:2px;
  padding:1px 0;
  font-weight:700;
}
</style>
</head>
<body>

<button class="fullscreen-fab" onclick="toggleFullscreen()" title="Fullscreen">⛶ Fullscreen</button>

<div id="viewport">
  <svg id="scene">
    <g id="links-layer"></g>
    <g id="nodes-layer"></g>
  </svg>
</div>

<div class="hint">
  <span>🖱 Drag to pan &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Click <strong>concept</strong> to expand/view finding &nbsp;·&nbsp; Click <strong>event</strong> for details</span>
  <span class="zoom-btns">
    <button onclick="resetView()">⤢ Reset</button>
    <button onclick="changeZoom(1.2)">＋</button>
    <button onclick="changeZoom(0.8)">－</button>
  </span>
</div>

<div id="source-viewer">
  <div class="sv-header">
    <span>📄</span>
    <span class="sv-filename" id="sv-filename">email.txt</span>
    <button class="sv-close" onclick="closeSourceViewer()" title="Close">✕</button>
  </div>
  <div class="sv-body" id="sv-body"></div>
</div>

<div id="detail">
  <button class="d-close" onclick="closeDetail()">✕</button>
  <div class="d-body" id="d-body"></div>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────
const DATA = """ + concepts_json + """;

// ── Layout constants ──────────────────────────────────────────────────────
const PAD          = 40;   // outer padding
const ROW_GAP      = 16;   // vertical gap between event rows
const COL_GAP      = 80;   // horizontal gap between columns
const ROOT_W       = 150; const ROOT_H  = 44;
const CONC_W       = 170; const CONC_H  = 40;
const EVT_W        = 200; const EVT_H   = 34;
const COL_ROOT     = PAD;
const COL_CONC     = COL_ROOT + ROOT_W + COL_GAP;
const COL_EVT      = COL_CONC + CONC_W + COL_GAP;

// ── State ─────────────────────────────────────────────────────────────────
const expanded = new Set();      // which concepts are expanded
let tx = 60, ty = 40, sc = 1;   // pan / zoom

const svg        = document.getElementById('scene');
const linksLayer = document.getElementById('links-layer');
const nodesLayer = document.getElementById('nodes-layer');
const viewport   = document.getElementById('viewport');

// ── Compute layout (positions) ────────────────────────────────────────────
function computeLayout() {
  // For each concept, calculate the vertical block it occupies.
  // Collapsed concept → 1 row height (CONC_H)
  // Expanded concept  → max(CONC_H, n_events * (EVT_H + ROW_GAP) - ROW_GAP)
  // Add GAP between consecutive concept blocks.
  const CONCEPT_GAP = 24;  // gap between consecutive concept blocks

  const concepts = Object.keys(DATA);
  const layout = [];
  let y = PAD;

  concepts.forEach(name => {
    const nEv = expanded.has(name) ? DATA[name].events.length : 0;
    const evBlockH = nEv > 0
      ? nEv * EVT_H + (nEv - 1) * ROW_GAP
      : 0;
    const blockH = Math.max(CONC_H, evBlockH);
    // concept sits in middle of its block
    const concY = y + blockH / 2;

    const events = [];
    if (nEv > 0) {
      const totalEvH = evBlockH;
      const evStartY = y + (blockH - totalEvH) / 2;
      DATA[name].events.forEach((ev, i) => {
        events.push({
          ev,
          x: COL_EVT,
          y: evStartY + i * (EVT_H + ROW_GAP) + EVT_H / 2
        });
      });
    }

    layout.push({ name, concX: COL_CONC, concY, blockH, events });
    y += blockH + CONCEPT_GAP;
  });

  const totalH = y + PAD;
  const totalW = (expanded.size > 0 ? COL_EVT + EVT_W : COL_CONC + CONC_W) + PAD;
  const rootY  = totalH / 2;

  return { layout, totalW, totalH, rootY };
}

// ── SVG helpers ───────────────────────────────────────────────────────────
function ns(tag) {
  return document.createElementNS('http://www.w3.org/2000/svg', tag);
}

// Smooth horizontal S-curve path (like NotebookLM)
function hcurve(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

function makeRect(x, y, w, h, cls, onClick) {
  const g = ns('g');
  g.setAttribute('class', 'node ' + cls);
  g.setAttribute('transform', `translate(${x - w/2},${y - h/2})`);

  const r = ns('rect');
  r.setAttribute('width',  w);
  r.setAttribute('height', h);
  g.appendChild(r);

  if (onClick) g.addEventListener('click', e => { e.stopPropagation(); onClick(); });
  return g;
}

function addLabel(g, text, w, h, extraClass) {
  const t = ns('text');
  t.setAttribute('x', w / 2);
  t.setAttribute('y', h / 2);
  if (extraClass) t.setAttribute('class', extraClass);
  t.textContent = text;
  g.appendChild(t);
}

// ── Main render ───────────────────────────────────────────────────────────
function render() {
  linksLayer.innerHTML = '';
  nodesLayer.innerHTML = '';

  const { layout, totalW, totalH, rootY } = computeLayout();

  svg.setAttribute('width',  totalW);
  svg.setAttribute('height', totalH);

  // ── Root node ──────────────────────────────────────
  const rx = COL_ROOT + ROOT_W / 2;
  const ry = rootY;
  const rootG = makeRect(rx, ry, ROOT_W, ROOT_H, 'node-root', null);
  addLabel(rootG, 'EVIDENCE CORPUS', ROOT_W, ROOT_H);
  nodesLayer.appendChild(rootG);

  // ── Concepts + events ──────────────────────────────
  layout.forEach(({ name, concX, concY, events }) => {
    const cx = concX + CONC_W / 2;
    const cy = concY;
    const isExp = expanded.has(name);

    // Root → Concept curved link
    const lk = ns('path');
    lk.setAttribute('class', 'link');
    lk.setAttribute('d', hcurve(rx + ROOT_W/2, ry, cx - CONC_W/2, cy));
    linksLayer.appendChild(lk);

    // Concept node
    const hasFinding = DATA[name].finding && DATA[name].finding.length > 0;
    let cls = 'node-concept' + (isExp ? ' expanded' : '') + (hasFinding ? ' has-finding' : '');
    const cg = makeRect(cx, cy, CONC_W, CONC_H, cls, () => {
      if (hasFinding) {
        // single click: show finding panel
        showConceptDetail(name);
      }
      // always toggle expand
      if (expanded.has(name)) expanded.delete(name);
      else expanded.add(name);
      render();
    });

    // Label (truncate if needed)
    const shortName = name.length > 22 ? name.slice(0, 20) + '…' : name;
    addLabel(cg, shortName, CONC_W - 22, CONC_H);

    // Expand/collapse arrow
    const arrow = ns('text');
    arrow.setAttribute('class', 'expand-arrow');
    arrow.setAttribute('x', CONC_W - 16);
    arrow.setAttribute('y', CONC_H / 2);
    arrow.textContent = isExp ? '‹' : '›';
    cg.appendChild(arrow);

    // Gold dot indicator when finding exists
    if (hasFinding) {
      const dot = ns('circle');
      dot.setAttribute('class', 'finding-dot');
      dot.setAttribute('cx', 10);
      dot.setAttribute('cy', CONC_H / 2);
      dot.setAttribute('r', 5);
      dot.setAttribute('fill', '#f9a825');
      dot.setAttribute('stroke', 'white');
      dot.setAttribute('stroke-width', '1.5');
      cg.appendChild(dot);
    }

    nodesLayer.appendChild(cg);

    // Events
    events.forEach(({ ev, x, y }) => {
      const ex = x + EVT_W / 2;
      const ey = y;

      // Concept → Event link
      const el = ns('path');
      el.setAttribute('class', 'link');
      el.setAttribute('d', hcurve(cx + CONC_W/2, cy, ex - EVT_W/2, ey));
      linksLayer.appendChild(el);

      // Event node — label: type + up to 2 parties
      const eg = makeRect(ex, ey, EVT_W, EVT_H, 'node-event', () => showDetail(ev, name));
      const typeTag = ev.event_type || 'Event';
      const partyStr = (ev.parties || []).slice(0,2).join(' · ');
      const rawLabel = partyStr ? typeTag + ' — ' + partyStr : typeTag;
      const label = rawLabel.length > 32 ? rawLabel.slice(0, 30) + '\u2026' : rawLabel;
      addLabel(eg, label, EVT_W, EVT_H);
      nodesLayer.appendChild(eg);
    });
  });

  applyTransform();
}

// ── Pan / zoom ────────────────────────────────────────────────────────────
function applyTransform() {
  svg.style.transform = `translate(${tx}px,${ty}px) scale(${sc})`;
  svg.style.transformOrigin = '0 0';
}

let dragging = false, ox, oy;

viewport.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  dragging = true; ox = e.clientX - tx; oy = e.clientY - ty;
  viewport.classList.add('grabbing');
});
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  tx = e.clientX - ox; ty = e.clientY - oy;
  applyTransform();
});
window.addEventListener('mouseup', () => {
  dragging = false; viewport.classList.remove('grabbing');
});

viewport.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  const osc = sc;
  sc = Math.min(3, Math.max(0.2, sc * factor));
  const r = viewport.getBoundingClientRect();
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;
  tx = mx - (mx - tx) * sc / osc;
  ty = my - (my - ty) * sc / osc;
  applyTransform();
}, { passive: false });

function changeZoom(f) {
  const r = viewport.getBoundingClientRect();
  const mx = r.width / 2, my = r.height / 2;
  const osc = sc;
  sc = Math.min(3, Math.max(0.2, sc * f));
  tx = mx - (mx - tx) * sc / osc;
  ty = my - (my - ty) * sc / osc;
  applyTransform();
}

function resetView() { tx = 60; ty = 40; sc = 1; applyTransform(); }

// ── Detail panel ──────────────────────────────────────────────────────────
function showConceptDetail(name) {
  const c = DATA[name];

  let h = '<div class="d-label" style="font-size:15px;color:#764ba2">📌 ' + name + '</div>';
  h += '<div style="margin:8px 0 12px">';
  h += '<span class="d-badge" style="background:#f3e5f5;color:#6a1b9a">📊 ' + c.count + ' evidence items</span>';
  h += '</div>';

  if (c.finding) {
    h += '<div class="d-label">Overview</div>';
    h += '<div class="d-finding">' + c.finding.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
  } else {
    h += '<div class="d-label">Overview</div>';
    h += '<div class="d-text">No finding available. Run <code>synthesize_findings.py</code> to generate Phase 4 synthesis.</div>';
  }

  h += '<div class="d-label" style="margin-top:16px">Evidence items (' + c.count + ')</div>';
  h += '<div class="d-rationale">Click any event node in the graph to view individual evidence details.</div>';

  document.getElementById('d-body').innerHTML = h;
  document.getElementById('detail').classList.add('show');
}

function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function showDetail(ev, concept) {
  let h = '';

  // ── Title: event type + concept ──────────────────────────────
  h += '<div style="font-size:15px;font-weight:700;color:#333;margin-bottom:2px">' + esc(ev.event_type || 'Event') + '</div>';
  h += '<div style="font-size:11px;color:#888;margin-bottom:16px">Concept: <em>' + esc(concept) + '</em></div>';

  // ── Narrative ────────────────────────────────────────────────
  if (ev.narrative) {
    h += '<div class="d-label">Narrative</div>';
    h += '<div class="d-text">' + esc(ev.narrative) + '</div>';
  }

  // ── Core fields table ────────────────────────────────────────
  h += '<div class="d-label" style="margin-top:14px">Details</div>';
  h += '<table class="d-field-table">';

  const dt = ev.date_time || ev.date || '';
  if (dt && dt !== 'N/A') h += '<tr><td>Date / Time</td><td>' + esc(dt) + '</td></tr>';

  const loc = ev.location || '';
  if (loc && loc !== 'N/A') h += '<tr><td>Location</td><td>' + esc(loc) + '</td></tr>';

  if (ev.confidence_score && ev.confidence_score !== 'N/A') {
    h += '<tr><td>Confidence</td><td>' + esc(ev.confidence_score) + '</td></tr>';
  }

  if (ev.parties && ev.parties.length > 0) {
    const pills = ev.parties.map(p => '<span class="d-party-pill">' + esc(p) + '</span>').join('');
    h += '<tr><td>Parties</td><td>' + pills + '</td></tr>';
  }

  h += '</table>';

  // ── Type-specific fields ─────────────────────────────────────
  const tsf = ev.type_specific_fields || {};
  const tsfKeys = Object.keys(tsf);
  if (tsfKeys.length > 0) {
    h += '<div class="d-label" style="margin-top:14px">Type-specific fields</div>';
    h += '<table class="d-field-table">';
    tsfKeys.forEach(k => {
      const label = k.replace(/_/g, ' ');
      h += '<tr><td>' + esc(label) + '</td><td>' + esc(tsf[k]) + '</td></tr>';
    });
    h += '</table>';
  }

  // Source file button
  if (ev.source_file && ev.source_file !== 'UNKNOWN') {
    const sf = ev.source_file || '';
    // Store the current source args so the button can call openSourceViewer safely
    // without embedding raw JSON inside an onclick="" attribute (which breaks on quotes/HTML chars)
    window._pendingSrcFile = sf;
    window._pendingSrcSnippet = ev.snippet || '';
    h += '<div class="d-label" style="margin-top:16px">Evidence Source</div>';
    h += '<div class="src-label">' + sf.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
    h += '<button class="src-btn" id="src-open-btn">📄 View Evidence Item</button>';
  }

  document.getElementById('d-body').innerHTML = h;
  document.getElementById('detail').classList.add('show');
  // Attach click handler AFTER innerHTML is set (avoids onclick="" escaping issues)
  const openBtn = document.getElementById('src-open-btn');
  if (openBtn) {
    openBtn.addEventListener('click', function() {
      openSourceViewer(window._pendingSrcFile, window._pendingSrcSnippet);
    });
  }
}

// ── Source viewer ─────────────────────────────────────────────────────────
const SOURCE_FILES = """ + source_files_json + """;

function openSourceViewer(filePath, snippet) {
  const content = SOURCE_FILES[filePath];
  const filename = filePath.split('/').pop();
  document.getElementById('sv-filename').textContent = filename;

  if (content === undefined || content === null) {
    document.getElementById('sv-body').innerHTML =
      '<span style="color:#f38ba8">⚠️ File content not embedded. Re-run mindmap.py to include source files.</span>';
  } else {
    // Escape HTML
    let safe = content.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    // Highlight the snippet (exact text from the extraction step)
    if (snippet && snippet.length > 10) {
      const safeSnip = snippet.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      const escaped = safeSnip.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');
      try {
        const re = new RegExp(escaped, 'i');
        safe = safe.replace(re, m => '<mark class="sv-highlight">' + m + '</mark>');
      } catch(e) { /* skip invalid regex */ }
    }
    document.getElementById('sv-body').innerHTML = safe;
    // Scroll to the highlighted snippet
    setTimeout(() => {
      const mark = document.querySelector('#sv-body .sv-highlight');
      if (mark) mark.scrollIntoView({ behavior:'smooth', block:'center' });
    }, 80);
  }
  document.getElementById('source-viewer').classList.add('show');
}

function closeSourceViewer() {
  document.getElementById('source-viewer').classList.remove('show');
}

function closeDetail() {
  document.getElementById('detail').classList.remove('show');
  closeSourceViewer();
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

// ── Fullscreen control ───────────────────────────────────────────────────
function toggleFullscreen() {
  const el = document.documentElement;
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {
    (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
  } else {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
