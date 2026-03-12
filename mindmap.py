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

    concepts_data = {}
    for concept in sorted(df_filtered['concept_name'].dropna().unique()):
        rows = df_filtered[df_filtered['concept_name'] == concept]
        concepts_data[concept] = {
            'count': int(len(rows)),
            'avg_score': float(rows['score'].mean()),
            'events': []
        }
        seen = set()
        for _, row in rows.iterrows():
            doc_id = str(row['doc_id'])
            if doc_id not in seen:
                seen.add(doc_id)
                concepts_data[concept]['events'].append({
                    'doc_id': doc_id,
                    'text': str(row['text']) if pd.notna(row['text']) else "N/A",
                    'score': float(row['score']),
                    'rationale': str(row['rationale']) if pd.notna(row['rationale']) else "No analysis",
                    'date': extract_date_from_text(row['text']),
                })

    print(f"   ✓ Organized {len(concepts_data)} concepts")
    print("\n🎨 Building interactive HTML...")

    html_content = generate_html(concepts_data)
    output_path = "evidence_mindmap.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    total_ev = sum(len(c['events']) for c in concepts_data.values())
    print(f"\n✅ Done! Concepts: {len(concepts_data)}, Evidence items: {total_ev}")
    print(f"🌐 Open {output_path} in your browser.\n")


def generate_html(concepts_data):
    total_events = sum(len(c['events']) for c in concepts_data.values())
    concepts_json = json.dumps(concepts_data, ensure_ascii=False)

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
</style>
</head>
<body>

<div class="header" id="header">
  <div class="header-controls">
    <button class="ctrl-btn" onclick="toggleHeader()" title="Collapse header">▲ Hide</button>
    <button class="ctrl-btn" onclick="toggleFullscreen()" title="Fullscreen">⛶ Fullscreen</button>
  </div>
  <h1>📊 Evidence Analysis</h1>
  <p>Interactive mindmap – click concepts to expand / collapse evidence</p>
</div>
<div id="header-collapsed" style="display:none;">
  <button class="restore-btn" onclick="toggleHeader()">📊 Evidence Analysis ▼ Show</button>
</div>

<div class="stats">
  <div class="stat-box">
    <div class="stat-number">""" + str(num_concepts) + """</div>
    <div class="stat-label">Concepts</div>
  </div>
  <div class="stat-box">
    <div class="stat-number">""" + str(num_events) + """</div>
    <div class="stat-label">Evidence items</div>
  </div>
  <div class="stat-box">
    <div class="stat-number">≥ 0.80</div>
    <div class="stat-label">Confidence</div>
  </div>
</div>

<div id="viewport">
  <svg id="scene">
    <g id="links-layer"></g>
    <g id="nodes-layer"></g>
  </svg>
</div>

<div class="hint">
  <span>🖱 Drag to pan &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Click <strong>concept</strong> to expand &nbsp;·&nbsp; Click <strong>event</strong> for details</span>
  <span class="zoom-btns">
    <button onclick="resetView()">⤢ Reset</button>
    <button onclick="changeZoom(1.2)">＋</button>
    <button onclick="changeZoom(0.8)">－</button>
  </span>
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
    const cls = 'node-concept' + (isExp ? ' expanded' : '');
    const cg = makeRect(cx, cy, CONC_W, CONC_H, cls, () => {
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

      // Event node
      const eg = makeRect(ex, ey, EVT_W, EVT_H, 'node-event', () => showDetail(ev, name));
      const label = ev.text.length > 30 ? ev.text.slice(0, 28) + '…' : ev.text;
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
function showDetail(ev, concept) {
  const conf = ev.score >= 0.95 ? 'confirmed' : ev.score >= 0.85 ? 'very-high' : 'high';
  const label = conf === 'confirmed' ? '🔴 CONFIRMED' : conf === 'very-high' ? '🟠 VERY HIGH' : '🟡 HIGH';

  let h = `<div class="d-badge badge-${conf}">${label}</div>`;
  h    += `<div class="d-badge" style="background:#eee;color:#333">Score: ${(ev.score*100).toFixed(0)}%</div>`;
  if (ev.date) h += `<div class="d-badge" style="background:#e8f5e9;color:#2e7d32">📅 ${ev.date}</div>`;
  h += `<div class="d-label">Concept</div><div class="d-text" style="border-color:#5dade2">${concept}</div>`;
  h += `<div class="d-label">Evidence</div><div class="d-text">${ev.text.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
  h += `<div class="d-label">Analysis</div><div class="d-rationale">${ev.rationale.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;

  document.getElementById('d-body').innerHTML = h;
  document.getElementById('detail').classList.add('show');
}

function closeDetail() { document.getElementById('detail').classList.remove('show'); }

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

// ── Header / Fullscreen controls ─────────────────────────────────────────
function toggleHeader() {
  const h = document.getElementById('header');
  const s = document.querySelector('.stats');
  const c = document.getElementById('header-collapsed');
  const isHidden = h.style.display === 'none';
  h.style.display = isHidden ? '' : 'none';
  s.style.display = isHidden ? '' : 'none';
  c.style.display = isHidden ? 'none' : '';
}

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
