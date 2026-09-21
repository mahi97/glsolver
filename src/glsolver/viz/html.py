"""Self-contained interactive HTML viewer of a trace (one file, no CDN).

:func:`render_html` replays the trace in Python (:func:`glsolver.viz.render.replay`),
precomputes the per-step drawing state and embeds it as JSON together with
a small inline SVG renderer: a step slider with prev/next/play buttons, the
event list on the left (click to jump), the graph in the middle with the
same conventions as the Matplotlib frames (terminal squares, essential
sectors, witness rings, contracted vertices, blue matching / red secondary
arcs, grey dashed deleted arcs, green cycle in the reassignment inset,
translucent ``L``/``S``/``R`` hulls for a recorded tightest cut), the state
panel on the right, and keyboard navigation (← → Home End Space).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from glsolver.viz.layout import (
    ARC_COLOR,
    CUT_COLORS,
    CYCLE_COLOR,
    DELETED_COLOR,
    HIGHLIGHT_COLOR,
    MATCHING_COLOR,
    SECONDARY_COLOR,
    convex_hull,
    layout_bounds,
    node_radius,
    terminal_color,
    terminal_svg_hatch_angle,
)
from glsolver.viz.render import (
    StepState,
    TraceRenderer,
    _fmt,
    arc_geometry,
    cycle_arcs,
    terminal_label,
    terminal_position,
    vertex_label,
)

SVG_WIDTH = 820


def export_viewer_data(renderer: TraceRenderer) -> dict[str, Any]:
    """Everything the inline viewer needs, as plain JSON types."""
    inst = renderer.inst
    layout = renderer.layout
    tidx = inst.terminal_index()
    xmin, xmax, ymin, ymax = layout_bounds(layout, pad=0.8)
    scale = SVG_WIDTH / (xmax - xmin)
    height = int(round((ymax - ymin) * scale))
    r = node_radius(layout) * scale

    def P(v: int) -> tuple[float, float]:
        x, y = layout[v]
        return (round((x - xmin) * scale, 2), round((ymax - y) * scale, 2))

    orig = set(inst.arcs)
    pos = {v: P(v) for v in range(inst.n)}

    arc_geo: dict[str, dict[str, Any]] = {}

    def geo_key(u: int, h: int) -> str:
        key = f"{u}-{h}"
        if key not in arc_geo:
            start, c, end, lab = arc_geometry(pos[u], pos[h], (h, u) in orig, r, r + 3)
            d = f"M{_fmt(start[0])},{_fmt(start[1])} " + (f"Q{_fmt(c[0])},{_fmt(c[1])} " if c else "L") + f"{_fmt(end[0])},{_fmt(end[1])}"
            arc_geo[key] = {"d": d, "lx": round(lab[0], 1), "ly": round(lab[1], 1)}
        return key

    steps = [_export_step(renderer, st, tidx, geo_key, pos) for st in renderer.states]
    return {
        "name": inst.name or "instance",
        "n": inst.n,
        "k": inst.k,
        "weighted": inst.is_weighted,
        "width": SVG_WIDTH,
        "height": height,
        "r": round(r, 2),
        "pos": {str(v): list(p) for v, p in pos.items()},
        "labels": {str(v): vertex_label(inst, v, renderer.labels) for v in range(inst.n)},
        "tidx": {str(t): i for t, i in tidx.items()},
        "colors": [terminal_color(i) for i in range(inst.k)],
        "hatch": [terminal_svg_hatch_angle(i) for i in range(inst.k)],
        "style": {
            "arc": ARC_COLOR, "match": MATCHING_COLOR, "sec": SECONDARY_COLOR, "dead": DELETED_COLOR,
            "now": HIGHLIGHT_COLOR, "cycle": CYCLE_COLOR, "cut": CUT_COLORS,
        },
        "geo": arc_geo,
        "steps": steps,
    }


def _export_step(renderer: TraceRenderer, st: StepState, tidx: dict[int, int], geo_key, pos) -> dict[str, Any]:
    inst = renderer.inst

    def ti(t: int) -> int:  # colour index; an unknown terminal id is a malformed trace, not colour 0
        return terminal_position(inst, t, tidx)

    matching_draw = {st.draw_arc(p, t) for p, t in st.matching}
    secondary_draw = {st.draw_arc(p, q): i for i, (p, q) in enumerate(st.secondary)}
    highlight = set(st.highlight_arcs)
    tree = {(p, par) for p, par, _ in st.tree_arcs}
    arcs: list[list[Any]] = []
    for (u, head), _reason in st.dead_arcs.items():
        if u == head or (u, head) in tree:
            continue
        now = (u, head) in highlight and st.type == "delete_arc"
        faint = u not in st.live and head not in st.live
        arcs.append([geo_key(u, head), "deadnow" if now else ("deadfaint" if faint else "dead"), ""])
    for (u, _v), head in st.arcs.items():
        if u == head:
            continue
        key = (u, head)
        if key in secondary_draw:
            arcs.append([geo_key(u, head), "sec", f"e{secondary_draw[key] + 1}"])
        elif key in matching_draw:
            arcs.append([geo_key(u, head), "match", ""])
        else:
            arcs.append([geo_key(u, head), "arc", ""])
    trees = []
    for p, par, t in st.tree_arcs:
        now = st.type in ("contract", "dag_contract", "round_and_remove") and p in st.highlight_vertices
        trees.append([geo_key(p, par), ti(t), 1 if now else 0])
    nodes = []
    for v in range(inst.n):
        node: dict[str, Any] = {"v": v}
        if v in tidx:
            node["kind"] = "t"
            node["ti"] = tidx[v]
            node["removed"] = v not in st.live
            node["glow"] = v in st.highlight_terminals
            node["tip"] = f"{terminal_label(inst, v)} (vertex {v}): capacity {st.capacities.get(v, 'done')}, part {st.parts.get(v, [v])}"
        elif v in st.part_of:
            node["kind"] = "c"
            node["ti"] = ti(st.part_of[v])
            node["glow"] = v in st.highlight_vertices
            node["tip"] = f"{v}: in the part of {terminal_label(inst, st.part_of[v])}"
        elif v not in st.live:
            node["kind"] = "x"
        else:
            node["kind"] = "n"
            node["ess"] = [ti(t) for t in st.ess.get(v, [])]
            node["ring"] = [[ti(t), round(f, 4)] for t, f in st.ring_of(v)]
            node["old"] = [ti(a) for vv, a, _ in st.changes if vv == v]
            node["glow"] = v in st.highlight_vertices
            node["stale"] = st.ess_stale
            ess_txt = ", ".join(terminal_label(inst, t) for t in st.ess.get(v, [])) or "∅"
            phi_txt = terminal_label(inst, st.phi[v]) if v in st.phi else ", ".join(f"{terminal_label(inst, t)}×{u}" for (x, t), u in sorted(st.psi.items()) if x == v)
            node["tip"] = f"{v}: Ess = {{{ess_txt}}}, κ = {st.kappa.get(v, '?')}, assigned {phi_txt or '-'}"
            if st.weights and v in st.weights:
                node["w"] = st.weights[v]
        nodes.append(node)
    panel = _panel(renderer, st, tidx)
    reass = None
    if st.reassignment:
        cyc_arcs = set(cycle_arcs(st.reassignment))
        reass = {
            "nodes": [[ti(t), terminal_label(inst, t)] for t in st.terminals],
            "arcs": [[st.terminals.index(a), st.terminals.index(b), vertex_label(inst, v, renderer.labels), 1 if (a, b) in cyc_arcs else 0]
                     for a, b, v in st.reassignment["arcs"] if a in st.terminals and b in st.terminals],
        }
    cuts = None
    if st.cuts:
        cuts = {}
        for v, c in st.cuts.items():
            cuts[str(v)] = {side: [list(p) for p in convex_hull([pos[x] for x in c.get(side, []) if x in pos])] for side in ("L", "S", "R")}
    return {
        "i": st.index, "type": st.type, "title": st.title, "desc": st.description,
        "arcs": arcs, "tree": trees, "nodes": nodes, "panel": panel, "reass": reass, "cuts": cuts,
    }


def _panel(renderer: TraceRenderer, st: StepState, tidx: dict[int, int]) -> dict[str, Any]:
    inst = renderer.inst
    terms = []
    for t in inst.terminals:
        live = t in st.terminals
        terms.append([terminal_label(inst, t), t, (st.capacities.get(t, "-") if live else "done"), len(st.parts.get(t, [t])), live, tidx[t]])
    if st.phi:
        wit = "φ: " + ", ".join(f"{vertex_label(inst, v, renderer.labels)}→{terminal_label(inst, t)}" for v, t in sorted(st.phi.items()))
    elif st.psi:
        wit = "ψ: " + ", ".join(f"{vertex_label(inst, v, renderer.labels)}→{terminal_label(inst, t)}×{u}" for (v, t), u in sorted(st.psi.items()))
    else:
        wit = ""
    pot = None
    if st.potential_before is not None or st.potential_after is not None:
        pot = [st.potential_before, st.potential_after]
    elif st.crit and st.phi:
        pot = [None, st.potential()]
    crit_txt = ""
    if st.crit:
        crit_txt = "; ".join(
            f"e{i + 1}: " + ", ".join(f"({vertex_label(inst, v, renderer.labels)},{terminal_label(inst, t)})" for ii, v, t in st.crit if ii == i)
            for i in sorted({i for i, _, _ in st.crit})
        )
    return {
        "terminals": terms,
        "left": st.num_nonterminals(),
        "arcs": len(st.arcs),
        "witness": wit,
        "pot": pot,
        "potlabel": "Φ̄ (split potential)" if st.psi and not st.phi else "potential Φ",
        "crit": crit_txt,
        "counters": dict(st.counters),
        "stale": st.ess_stale,
        "cutinfo": {str(v): [len(c["L"]), len(c["S"]), len(c["R"])] for v, c in st.cuts.items()} if st.cuts else None,
    }


_CSS = """
:root { --bg:#fafafa; --panel:#ffffff; --line:#e0e0e0; --fg:#222; --muted:#666; --acc:#2b4bd6; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--fg); background:var(--bg); }
header { display:flex; align-items:center; gap:12px; padding:8px 14px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 12px 0 0; }
header button { font-size:14px; padding:3px 10px; cursor:pointer; }
header input[type=range] { flex:1; min-width:180px; }
#stepno { font-variant-numeric: tabular-nums; min-width: 90px; }
main { display:grid; grid-template-columns: 260px 1fr 330px; gap:0; height: calc(100vh - 50px); }
#events { overflow:auto; border-right:1px solid var(--line); background:var(--panel); margin:0; padding:6px 0; list-style:none; }
#events li { padding:4px 10px; cursor:pointer; font-size:12.5px; border-left:3px solid transparent; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#events li:hover { background:#f0f3ff; }
#events li.cur { background:#e6ebff; border-left-color:var(--acc); font-weight:600; }
#events li .n { color:var(--muted); display:inline-block; width:28px; }
#graphwrap { overflow:auto; padding:6px; display:flex; flex-direction:column; align-items:center; }
#title { font-size:14px; margin:4px 0 2px; align-self:flex-start; }
#desc { font-size:12.5px; color:#333; margin:0 0 6px; align-self:flex-start; max-width:100%; }
svg#graph { background:white; border:1px solid var(--line); max-width:100%; height:auto; }
#legend { font-size:11.5px; color:#444; margin-top:6px; display:flex; gap:14px; flex-wrap:wrap; align-self:flex-start; }
#legend span { display:inline-flex; align-items:center; gap:4px; }
#panel { overflow:auto; border-left:1px solid var(--line); background:var(--panel); padding:10px 12px; font-size:12.5px; }
#panel h2 { font-size:13px; margin:8px 0 4px; }
#panel table { border-collapse:collapse; width:100%; font-variant-numeric: tabular-nums; }
#panel td, #panel th { padding:2px 4px; text-align:right; border-bottom:1px solid #f0f0f0; }
#panel td:first-child, #panel th:first-child { text-align:left; }
#panel .sw { display:inline-block; width:12px; height:12px; border:1px solid #333; border-radius:2px; vertical-align:middle; margin-right:4px; }
#panel .dim { color:#999; }
#panel .mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; word-break:break-word; }
#panel .pot { font-weight:600; }
#panel .warn { color:#a00000; }
#reass { margin-top:8px; }
@media (max-width: 1100px) { main { grid-template-columns: 200px 1fr 260px; } }
"""

_JS = r"""
const D = window.__GL_DATA__;
const N = D.steps.length;
let cur = 0, timer = null;
const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function fillOf(ti) { return D.hatch[ti] === null ? D.colors[ti] : `url(#hatch${ti})`; }
function polar(cx, cy, r, deg) { const a = deg * Math.PI / 180; return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; }
function wedge(cx, cy, r, a1, a2) { // angles in degrees, counter-clockwise from +x, a2 > a1
  const [x1, y1] = polar(cx, cy, r, a1), [x2, y2] = polar(cx, cy, r, a2);
  const large = (a2 - a1) > 180 ? 1 : 0;
  if (a2 - a1 >= 359.99) return `M${cx},${cy - r} A${r},${r} 0 1 1 ${cx - 0.01},${cy - r} Z`;
  return `M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${large} 0 ${x2},${y2} Z`;
}
function ringArc(cx, cy, r, a1, a2) { // stroke-only arc path
  const [x1, y1] = polar(cx, cy, r, a1), [x2, y2] = polar(cx, cy, r, a2);
  const large = (a2 - a1) > 180 ? 1 : 0;
  if (a2 - a1 >= 359.99) return `M${cx},${cy - r} A${r},${r} 0 1 1 ${cx - 0.01},${cy - r}`;
  return `M${x1},${y1} A${r},${r} 0 ${large} 0 ${x2},${y2}`;
}
function buildDefs(svg) {
  const defs = el('defs', {}, svg);
  const cols = new Set([D.style.arc, D.style.match, D.style.sec, D.style.dead, D.style.now, D.style.cycle, '#666666', ...D.colors]);
  for (const c of cols) {
    const m = el('marker', {id: 'arr' + c.slice(1), viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto'}, defs);
    el('path', {d: 'M0,0 L10,5 L0,10 z', fill: c}, m);
  }
  D.hatch.forEach((ang, i) => {
    if (ang === null) return;
    const p = el('pattern', {id: 'hatch' + i, patternUnits: 'userSpaceOnUse', width: 6, height: 6, patternTransform: `rotate(${ang})`}, defs);
    el('rect', {width: 6, height: 6, fill: D.colors[i]}, p);
    el('line', {x1: 0, y1: 0, x2: 0, y2: 6, stroke: '#333', 'stroke-width': 1.2}, p);
  });
}
function renderGraph(step) {
  const svg = $('graph');
  while (svg.lastChild) svg.removeChild(svg.lastChild);
  buildDefs(svg);
  const r = D.r;
  const cutSel = $('cutsel');
  // cut hulls
  if (step.cuts) {
    const keys = Object.keys(step.cuts);
    cutSel.style.display = '';
    if (!keys.includes(cutSel.value)) {
      cutSel.textContent = '';
      for (const k of keys) { const o = document.createElement('option'); o.value = k; o.textContent = 'cut of ' + D.labels[k]; cutSel.appendChild(o); }
      cutSel.value = keys[0];
    }
    const cut = step.cuts[cutSel.value];
    for (const side of ['L', 'S', 'R']) {
      const pts = cut[side]; if (!pts.length) continue;
      const col = D.style.cut[side];
      if (pts.length === 1) el('circle', {cx: pts[0][0], cy: pts[0][1], r: r * 1.9, fill: col, opacity: 0.4}, svg);
      else el('polygon', {points: pts.map(p => p.join(',')).join(' '), fill: col, stroke: col, 'stroke-width': r * 3.6, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: 0.4}, svg);
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = Math.min(...pts.map(p => p[1])) - r * 1.7;
      el('text', {x: cx, y: cy, 'font-size': 12, fill: '#555', 'text-anchor': 'middle'}, svg, side);
    }
  } else { cutSel.style.display = 'none'; cutSel.textContent = ''; }
  // arcs
  for (const [key, cls, label] of step.arcs) {
    const g = D.geo[key];
    let stroke = D.style.arc, w = 1.2, dash = null, op = 1;
    if (cls === 'match') { stroke = D.style.match; w = 3; }
    else if (cls === 'sec') { stroke = D.style.sec; w = 3; }
    else if (cls === 'dead') { stroke = D.style.dead; w = 1; dash = '5,4'; }
    else if (cls === 'deadfaint') { stroke = D.style.dead; w = 1; dash = '5,4'; op = 0.45; }
    else if (cls === 'deadnow') { stroke = D.style.now; w = 2.6; dash = '6,4'; }
    el('path', {d: g.d, fill: 'none', stroke, 'stroke-width': w, 'stroke-dasharray': dash, opacity: op, 'marker-end': `url(#arr${stroke.slice(1)})`}, svg);
    if (label) {
      el('rect', {x: g.lx - 10, y: g.ly - 8, width: 20, height: 16, rx: 4, fill: 'white', opacity: 0.85}, svg);
      el('text', {x: g.lx, y: g.ly, 'font-size': 12, fill: D.style.sec, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-style': 'italic'}, svg, label);
    }
    if (cls === 'deadnow') el('text', {x: g.lx, y: g.ly, 'font-size': 16, fill: D.style.now, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-weight': 'bold'}, svg, '✕');
  }
  for (const [key, ti, now] of step.tree) {
    const g = D.geo[key];
    el('path', {d: g.d, fill: 'none', stroke: D.colors[ti], 'stroke-width': now ? 4.5 : 3.2, opacity: 0.9, 'marker-end': `url(#arr${D.colors[ti].slice(1)})`}, svg);
  }
  // nodes
  for (const nd of step.nodes) {
    const [x, y] = D.pos[nd.v];
    const lab = D.labels[nd.v];
    const g = el('g', {}, svg);
    if (nd.tip) el('title', {}, g, nd.tip);
    if (nd.kind === 't') {
      const s = 2.1 * r;
      if (nd.glow) el('circle', {cx: x, cy: y, r: r * 1.9, fill: D.colors[nd.ti], opacity: 0.25}, g);
      el('rect', {x: x - s / 2, y: y - s / 2, width: s, height: s, rx: r * 0.3, fill: fillOf(nd.ti), stroke: '#222', 'stroke-width': 1.3, 'stroke-dasharray': nd.removed ? '4,3' : null, opacity: nd.removed ? 0.45 : 1}, g);
      el('text', {x, y, 'font-size': r * 0.95, 'font-weight': 'bold', 'text-anchor': 'middle', 'dominant-baseline': 'central'}, g, lab);
    } else if (nd.kind === 'c') {
      if (nd.glow) el('circle', {cx: x, cy: y, r: r * 1.6, fill: D.colors[nd.ti], opacity: 0.3}, g);
      el('circle', {cx: x, cy: y, r: r * 0.72, fill: fillOf(nd.ti), stroke: '#333', 'stroke-width': 1.2, 'stroke-dasharray': '2,2'}, g);
      el('text', {x, y, 'font-size': r * 0.8, 'text-anchor': 'middle', 'dominant-baseline': 'central'}, g, lab);
    } else if (nd.kind === 'x') {
      el('circle', {cx: x, cy: y, r: r * 0.6, fill: 'white', stroke: '#999', 'stroke-dasharray': '2,2'}, g);
    } else {
      if (nd.glow) { const gc = nd.ring.length ? D.colors[nd.ring[0][0]] : D.style.now; el('circle', {cx: x, cy: y, r: r * 1.75, fill: gc, opacity: 0.25}, g); }
      el('circle', {cx: x, cy: y, r, fill: 'white', stroke: '#333', 'stroke-width': 1}, g);
      const n = nd.ess.length;
      nd.ess.forEach((ti, j) => {
        const a1 = 90 - 360 * (j + 1) / n, a2 = 90 - 360 * j / n;
        el('path', {d: wedge(x, y, r, a1, a2), fill: fillOf(ti), stroke: 'white', 'stroke-width': 0.6, opacity: nd.stale ? 0.55 : 1}, g);
      });
      el('circle', {cx: x, cy: y, r, fill: 'none', stroke: '#333', 'stroke-width': 1}, g);
      let start = 90;
      for (const [ti, frac] of nd.ring) {
        const span = 360 * frac;
        el('path', {d: ringArc(x, y, r * 1.22, start - span, start), fill: 'none', stroke: D.colors[ti], 'stroke-width': r * 0.24}, g);
        start -= span;
      }
      for (const ti of nd.old) el('circle', {cx: x, cy: y, r: r * 1.5, fill: 'none', stroke: D.colors[ti], 'stroke-width': 1.5, 'stroke-dasharray': '4,3'}, g);
      el('circle', {cx: x, cy: y, r: r * 0.46, fill: 'white', opacity: 0.85}, g);
      el('text', {x, y, 'font-size': r * 0.85, 'font-weight': 'bold', 'text-anchor': 'middle', 'dominant-baseline': 'central'}, g, lab);
      if (nd.w !== undefined) el('text', {x: x + r * 1.15, y: y + r * 1.45, 'font-size': r * 0.6, fill: '#444'}, g, 'w=' + nd.w);
    }
  }
}
function renderReass(step) {
  const box = $('reass');
  box.innerHTML = '';
  if (!step.reass) return;
  const W = 300, H = 210, cx = W / 2, cy = H / 2 + 8, R = 70;
  const svg = el('svg', {width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: 'background:white;border:1px solid #e0e0e0'}, box);
  buildDefs(svg);
  el('text', {x: cx, y: 14, 'font-size': 12, 'text-anchor': 'middle'}, svg, 'reassignment graph R (cycle in green)');
  const k = step.reass.nodes.length;
  const pos = step.reass.nodes.map((_, j) => polar(cx, cy, R, 90 + 360 * j / k));
  for (const [a, b, label, on] of step.reass.arcs) {
    const [x1, y1] = pos[a], [x2, y2] = pos[b];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1;
    const c = [mx + 0.25 * dy, my - 0.25 * dx];
    const u1 = [c[0] - x1, c[1] - y1], l1 = Math.hypot(...u1) || 1, u2 = [c[0] - x2, c[1] - y2], l2 = Math.hypot(...u2) || 1;
    const s = [x1 + u1[0] / l1 * 14, y1 + u1[1] / l1 * 14], e = [x2 + u2[0] / l2 * 16, y2 + u2[1] / l2 * 16];
    const col = on ? D.style.cycle : '#666666';
    el('path', {d: `M${s[0]},${s[1]} Q${c[0]},${c[1]} ${e[0]},${e[1]}`, fill: 'none', stroke: col, 'stroke-width': on ? 3 : 1.4, 'marker-end': `url(#arr${col.slice(1)})`}, svg);
    const lx = 0.25 * s[0] + 0.5 * c[0] + 0.25 * e[0], ly = 0.25 * s[1] + 0.5 * c[1] + 0.25 * e[1];
    el('rect', {x: lx - 9, y: ly - 7, width: 18, height: 14, rx: 3, fill: 'white', opacity: 0.9}, svg);
    el('text', {x: lx, y: ly, 'font-size': 11, fill: col, 'text-anchor': 'middle', 'dominant-baseline': 'central'}, svg, label);
  }
  step.reass.nodes.forEach(([ti, lab], j) => {
    const [x, y] = pos[j];
    el('rect', {x: x - 12, y: y - 12, width: 24, height: 24, rx: 4, fill: fillOf(ti), stroke: '#222'}, svg);
    el('text', {x, y, 'font-size': 11, 'font-weight': 'bold', 'text-anchor': 'middle', 'dominant-baseline': 'central'}, svg, lab);
  });
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
function renderPanel(step) {
  const p = step.panel;
  let h = `<h2>step ${step.i} / ${N - 1} &middot; ${esc(step.type)}</h2>`;
  h += '<table><tr><th>terminal</th><th>id</th><th>cap</th><th>part</th></tr>';
  for (const [lab, id, cap, size, live, ti] of p.terminals)
    h += `<tr class="${live ? '' : 'dim'}"><td><span class="sw" style="background:${D.colors[ti]}"></span>${lab}</td><td>${id}</td><td>${cap}</td><td>${size}</td></tr>`;
  h += '</table>';
  h += `<p>non-terminals left: <b>${p.left}</b> &middot; live arcs: <b>${p.arcs}</b></p>`;
  if (p.witness) h += `<p class="mono">${esc(p.witness)}</p>`;
  if (p.pot) h += `<p class="pot">${esc(p.potlabel)}: ${p.pot[0] === null ? '' : p.pot[0] + ' → '}${p.pot[1] === null ? '?' : p.pot[1]}</p>`;
  if (p.crit) h += `<p class="mono">critical: ${esc(p.crit)}</p>`;
  if (p.stale) h += '<p class="warn">essential sets not recomputed (stale)</p>';
  if (p.cutinfo) for (const v in p.cutinfo) h += `<p>tightest cut of ${esc(D.labels[v])}: |L|=${p.cutinfo[v][0]}, |S|=${p.cutinfo[v][1]}, |R|=${p.cutinfo[v][2]}</p>`;
  const c = p.counters;
  h += `<h2>counters</h2><table>` + Object.keys(c).map(k => `<tr><td>${k.replace(/_/g, ' ')}</td><td>${c[k]}</td></tr>`).join('') + '</table>';
  h += '<div id="reass"></div>';
  $('panel').innerHTML = h;
}
function goto(i) {
  cur = Math.max(0, Math.min(N - 1, i));
  const step = D.steps[cur];
  $('slider').value = cur;
  $('stepno').textContent = `step ${cur} / ${N - 1}`;
  $('title').textContent = `${D.name} — ${step.title}`;
  $('desc').textContent = step.desc;
  renderGraph(step);
  renderPanel(step);
  renderReass(step);
  document.querySelectorAll('#events li').forEach((li, j) => li.classList.toggle('cur', j === cur));
  const li = document.querySelector('#events li.cur'); if (li) li.scrollIntoView({block: 'nearest'});
}
function play() {
  if (timer) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
  $('play').textContent = '⏸ pause';
  timer = setInterval(() => { if (cur >= N - 1) { play(); return; } goto(cur + 1); }, 1000);
}
window.addEventListener('DOMContentLoaded', () => {
  const ul = $('events');
  D.steps.forEach((s, j) => { const li = document.createElement('li'); li.innerHTML = `<span class="n">${j}</span>${esc(s.title)}`; li.title = s.desc; li.onclick = () => goto(j); ul.appendChild(li); });
  $('slider').max = N - 1;
  $('slider').oninput = e => goto(+e.target.value);
  $('prev').onclick = () => goto(cur - 1);
  $('next').onclick = () => goto(cur + 1);
  $('play').onclick = play;
  $('cutsel').onchange = () => renderGraph(D.steps[cur]);
  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { goto(cur - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { goto(cur + 1); e.preventDefault(); }
    else if (e.key === 'Home') goto(0); else if (e.key === 'End') goto(N - 1);
    else if (e.key === ' ') { play(); e.preventDefault(); }
  });
  goto(0);
});
"""


def build_html(data: dict[str, Any]) -> str:
    """The complete HTML document for the viewer data of :func:`export_viewer_data`."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    name = html.escape(data["name"])
    c0, c1, c2 = terminal_color(0), terminal_color(1), terminal_color(2)
    legend = (
        f'<span><svg width="16" height="14"><rect x="1" y="1" width="12" height="12" rx="3" fill="{c0}" stroke="#222"/></svg>terminal</span>'
        f'<span><svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="white" stroke="#333"/>'
        f'<path d="M10,10 L10,4 A6,6 0 0 0 10,16 Z" fill="{c1}"/><path d="M10,10 L10,16 A6,6 0 0 0 10,4 Z" fill="{c2}"/>'
        f'<circle cx="10" cy="10" r="8.5" fill="none" stroke="{c2}" stroke-width="2"/></svg>sectors = Ess(v), ring = φ(v)</span>'
        f'<span><svg width="16" height="16"><circle cx="8" cy="8" r="5" fill="{c0}" stroke="#333" stroke-dasharray="2,2"/></svg>contracted</span>'
        f'<span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="{MATCHING_COLOR}" stroke-width="3"/></svg>matching</span>'
        f'<span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="{SECONDARY_COLOR}" stroke-width="3"/></svg>secondary e<sub>i</sub></span>'
        f'<span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="{DELETED_COLOR}" stroke-width="1.5" stroke-dasharray="4,3"/></svg>deleted</span>'
        f'<span><svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke="{c1}" stroke-width="3.5"/></svg>arborescence</span>'
        '<span>hover a vertex for Ess / κ / assignment</span>'
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — Győri–Lovász trace</title>
<meta name="gl-steps" content="{len(data['steps'])}">
<style>{_CSS}</style>
</head><body>
<header><h1>{name}</h1>
<button id="prev" title="previous (←)">◀</button><button id="next" title="next (→)">▶</button><button id="play" title="play/pause (space)">▶ play</button>
<span id="stepno"></span><input id="slider" type="range" min="0" max="0" value="0">
<select id="cutsel" style="display:none"></select>
<span style="color:#666;font-size:12px">keys: ← → Home End Space</span></header>
<main>
<ul id="events"></ul>
<div id="graphwrap"><div id="title"></div><div id="desc"></div>
<svg id="graph" xmlns="http://www.w3.org/2000/svg" width="{data['width']}" height="{data['height']}" viewBox="0 0 {data['width']} {data['height']}" font-family="Helvetica, Arial, sans-serif"></svg>
<div id="legend">{legend}</div></div>
<div id="panel"></div>
</main>
<script>window.__GL_DATA__ = {payload};</script>
<script>{_JS}</script>
</body></html>
"""


def render_html(inst: Any, trace: Any = None, path: str | Path = "trace.html", *, renderer: TraceRenderer | None = None, **kw: Any) -> Path:
    """Write ONE self-contained HTML file with the interactive step viewer.

    ``inst``/``trace`` are accepted as for :class:`TraceRenderer` (an
    ``Instance`` plus the event list, or a ``GLResult``); extra keyword
    arguments (``layout``, ``layout_method``, ``labels``, ...) go to the
    renderer.  Passing both ``renderer`` and such arguments is a
    ``TypeError`` (they would be silently ignored).  Returns the written path.
    """
    if renderer is not None and kw:
        raise TypeError(f"render_html: renderer options {sorted(kw)} cannot be combined with an explicit renderer")
    renderer = renderer or TraceRenderer(inst, trace, **kw)
    data = export_viewer_data(renderer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(data), encoding="utf-8")
    return path
