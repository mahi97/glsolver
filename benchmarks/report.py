"""Markdown summary tables and the self-contained verification dashboard.

* :func:`summary_markdown` -- per-config tables of medians grouped by
  ``(family, n, k, algorithm)`` plus an overview per algorithm (runs, VALID,
  INVALID, timeouts, errors).  Written to ``benchmarks/results/summary.md``.
* :func:`dashboard_html` -- ONE self-contained HTML file (inline CSS and plain
  JS, no external resources) listing every run: instance, n, m, k, density,
  connectivity (from generator metadata), capacities, algorithm, runtime, peak
  memory, primitive call counts, final part sizes, the INDEPENDENT VERIFIER
  verdict (green VALID / red INVALID), the baseline comparison against the
  brute-force / ILP oracles on the same instance, and for ``n <= 30`` an
  inline SVG of the graph coloured by the final partition
  (``glsolver.viz.partition_svg`` when importable, else a text listing).  The
  table is sortable (click a header) and filterable (text + selects).

Usage::

    python -m benchmarks.report [RESULTS.jsonl ...] [--html DASHBOARD] [--markdown SUMMARY]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.runner import RESULTS_DIR, load_results  # noqa: E402

DASHBOARD_PATH = RESULTS_DIR / "dashboard.html"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
SVG_MAX_N = 30
ORACLES = ("bruteforce", "ilp")
COUNTERS = (
    ("max_flow_calls", "flow"), ("matching_calls", "match"), ("contractions", "contract"),
    ("deletions", "delete"), ("cycle_shifts", "shift"), ("min_cost_flow_calls", "mcf"),
    ("terminal_removals", "t-rm"), ("roundings", "round"), ("heap_pops", "pop"),
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _fmt_s(x: Any) -> str:
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return "0"
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    if x < 100:
        return f"{x:.3f} s"
    return f"{x:.1f} s"


def _fmt_mb(x: Any) -> str:
    return "—" if x is None else f"{float(x):.0f} MB"


def _fmt_list(xs: Any, limit: int = 12) -> str:
    if not xs:
        return "—"
    xs = list(xs)
    if len(xs) <= limit:
        return ",".join(str(x) for x in xs)
    return ",".join(str(x) for x in xs[:limit]) + f",… ({len(xs)} parts)"


def _instance_name(row: dict[str, Any]) -> str:
    return str((row.get("instance") or {}).get("name") or "?")


def _label(row: dict[str, Any]) -> str:
    return str(row.get("algorithm_label") or row.get("algorithm") or "?")


def _connectivity_text(row: dict[str, Any]) -> str:
    conn = dict((row.get("instance") or {}).get("connectivity") or {})
    meta = (row.get("instance") or {}).get("meta") or {}
    parts: list[str] = []
    if conn.get("kappa") is not None:
        parts.append(f"κ={conn['kappa']}")
    if conn.get("vertex_connectivity") is not None:
        parts.append(f"κ(G)={conn['vertex_connectivity']}")
    if conn.get("claims_k_connected"):
        parts.append("claims k-conn" + ("✓" if conn.get("connectivity_verified") else "?"))
    if conn.get("claims_kT_connected"):
        parts.append("claims k-T-conn" + ("✓" if conn.get("connectivity_verified") else "?"))
    if meta.get("is_dag") or (row.get("instance") or {}).get("dag_kT_connected"):
        parts.append("DAG k-T")
    return " ".join(parts) if parts else "—"


def _counters_text(stats: dict[str, Any]) -> str:
    bits = []
    for key, short in COUNTERS:
        val = stats.get(key)
        if isinstance(val, (int, float)) and val:
            bits.append(f"{short}:{int(val)}")
    return " ".join(bits) if bits else "—"


def _agreement(solver: dict[str, Any], oracle: dict[str, Any]) -> tuple[str, str]:
    """Return ``(class, text)`` describing how a solver row compares with an oracle row."""
    ss, sv = solver.get("status"), solver.get("valid")
    os_, ov = oracle.get("status"), oracle.get("valid")
    if os_ == "timeout" or oracle.get("timed_out"):
        return "na", "oracle timeout"
    if os_ == "error":
        return "na", "oracle error"
    if ss == "ok" and sv is True and os_ == "ok" and ov is True:
        return "agree", "agree (both valid)"
    if ss == "ok" and sv is True and os_ == "infeasible":
        return "disagree", "DISAGREE: oracle says infeasible but solver found a verified partition"
    if ss == "precondition_failed" and os_ == "infeasible":
        return "agree", "consistent (no guarantee; oracle: infeasible)"
    if ss == "precondition_failed" and os_ == "ok":
        return "note", "oracle found a partition (precondition failed, no guarantee)"
    if sv is False:
        return "disagree", "solver INVALID"
    if ov is False:
        return "disagree", "oracle output INVALID"
    return "na", f"solver {ss} / oracle {os_}"


# ---------------------------------------------------------------------------
# markdown tables
# ---------------------------------------------------------------------------
def _median_or_none(vals: Sequence[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return statistics.median(xs) if xs else None


def overview_table(rows: Sequence[dict[str, Any]]) -> str:
    """Markdown table: one line per algorithm label with run/valid/invalid/timeout/error counts."""
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        a = agg[_label(r)]
        a["runs"] += 1
        if r.get("valid") is True:
            a["valid"] += 1
        if r.get("valid") is False:
            a["invalid"] += 1
        if r.get("timed_out"):
            a["timeouts"] += 1
        if r.get("status") == "error":
            a["errors"] += 1
        if r.get("status") == "precondition_failed":
            a["precondition_failed"] += 1
    lines = ["| algorithm | runs | VALID | INVALID | timeouts | errors | precondition failed |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label in sorted(agg):
        a = agg[label]
        lines.append(f"| {label} | {a['runs']} | {a['valid']} | {a['invalid']} | {a['timeouts']} | "
                     f"{a['errors']} | {a['precondition_failed']} |")
    return "\n".join(lines)


def median_table(rows: Sequence[dict[str, Any]]) -> str:
    """Markdown table of medians grouped by ``(family, n, k, algorithm)``."""
    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        inst = r.get("instance") or {}
        fam = str(inst.get("family") or "?")
        var = (inst.get("variant") or {}).get("type", "plain")
        if var != "plain":
            fam += f"+{var}"
        groups[(fam, int(r.get("n") or 0), int(r.get("k") or 0), _label(r))].append(r)
    lines = ["| family | n | k | algorithm | runs | median wall | min | max | verify | peak RSS | "
             "flow calls | VALID | timeouts |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(groups):
        fam, n, k, label = key
        rs = groups[key]
        walls = [(r.get("median") or {}).get("wall_time") for r in rs]
        walls_ok = [w for w in walls if w is not None]
        med = _median_or_none(walls)
        ver = _median_or_none([(r.get("median") or {}).get("verify_time") for r in rs])
        rss = _median_or_none([(r.get("child") or {}).get("peak_rss_mb") for r in rs])
        flows = _median_or_none([((r.get("result") or {}).get("stats") or {}).get("max_flow_calls") for r in rs])
        valid = sum(1 for r in rs if r.get("valid") is True)
        tos = sum(1 for r in rs if r.get("timed_out"))
        lines.append(
            f"| {fam} | {n} | {k} | {label} | {len(rs)} | {_fmt_s(med)} | "
            f"{_fmt_s(min(walls_ok) if walls_ok else None)} | {_fmt_s(max(walls_ok) if walls_ok else None)} | "
            f"{_fmt_s(ver)} | {_fmt_mb(rss)} | {'—' if flows is None else int(flows)} | "
            f"{valid}/{len(rs)} | {tos} |"
        )
    return "\n".join(lines)


def summary_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Complete markdown report (overview + one median table per config)."""
    if not rows:
        return "# Benchmark summary\n\n_No result rows found._\n"
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cfg[str(r.get("config") or "?")].append(r)
    machine = rows[-1].get("machine") or {}
    cpu = (machine.get("cpu") or {}).get("model")
    out = ["# Benchmark summary", "",
           f"Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} from {len(rows)} rows "
           f"({', '.join(sorted(by_cfg))}).", "",
           f"Machine: {cpu or '?'}, {(machine.get('cpu') or {}).get('cores_logical') or '?'} logical cores, "
           f"{(machine.get('memory') or {}).get('ram_gib') or '?'} GiB RAM, Python "
           f"{(machine.get('python') or {}).get('version') or '?'}, glsolver {rows[-1].get('glsolver_version')} "
           f"@ {str(rows[-1].get('git_commit') or '?')[:12]}.", "",
           "Timing = median over repeats of the wall time of one solve (verifier excluded, timed separately); "
           "each run is an isolated subprocess.", "",
           "## Overview", "", overview_table(rows), ""]
    for cfg in sorted(by_cfg):
        out += [f"## {cfg}", "", median_table(by_cfg[cfg]), ""]
    return "\n".join(out)


def write_summary_markdown(rows: Sequence[dict[str, Any]], path: Path | str = SUMMARY_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_markdown(rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# graph rendering for the dashboard
# ---------------------------------------------------------------------------
_svg_cache: dict[str, str] = {}


def _load_instance(row: dict[str, Any]) -> Any:
    path = (row.get("instance") or {}).get("path")
    if not path or not Path(path).exists():
        return None
    from glsolver.io import load_instance

    return load_instance(path)


def graph_html(row: dict[str, Any], max_n: int = SVG_MAX_N) -> str:
    """Inline SVG (via ``glsolver.viz.partition_svg``) or a text listing of the parts."""
    parts = (row.get("result") or {}).get("parts")
    n = int(row.get("n") or 0)
    if not parts or n > max_n:
        return ""
    key = f"{_instance_name(row)}::{_label(row)}"
    if key in _svg_cache:
        return _svg_cache[key]
    svg = ""
    try:
        try:
            from glsolver.viz import partition_svg  # type: ignore
        except ImportError:
            from glsolver.viz.render import partition_svg  # type: ignore

        inst = _load_instance(row)
        if inst is not None:
            out = partition_svg(inst, parts)
            svg = out if isinstance(out, str) else str(out)
            if "<svg" not in svg:
                svg = ""
    except Exception as exc:  # the viz module is optional: fall back to the text listing
        if os.environ.get("GLBENCH_DEBUG"):
            print(f"partition_svg failed for {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
        svg = ""
    if svg:
        body = f'<div class="svgwrap">{svg}</div>'
    else:
        items = "".join(f"<li>part {i}: {html.escape(_fmt_list(p, 40))}</li>" for i, p in enumerate(parts))
        body = f'<ul class="parts">{items}</ul>'
    out_html = f"<details><summary>graph</summary>{body}</details>"
    _svg_cache[key] = out_html
    return out_html


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--good:#0ca30c;--critical:#d03b3b;--warning:#fab219;--serious:#ec835a;--blue:#2a78d6;}
*{box-sizing:border-box}
body{margin:0;padding:16px 20px;font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px}
.sub{color:var(--ink2);margin-bottom:12px}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:10px 14px;min-width:120px}
.tile .v{font-size:22px;font-weight:600}.tile .l{color:var(--ink2);font-size:12px}
.tile.good .v{color:var(--good)}.tile.bad .v{color:var(--critical)}.tile.warn .v{color:#8a5a00}
.machine{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:8px 12px;color:var(--ink2);font-size:12px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0;padding:8px 10px;background:var(--surface);border:1px solid var(--grid);border-radius:8px}
.filters label{color:var(--ink2);font-size:12px}
.filters input,.filters select{font:inherit;padding:4px 6px;border:1px solid var(--grid);border-radius:6px;background:#fff}
.count{color:var(--ink2);margin-left:auto;font-size:12px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--grid);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none;font-weight:600;color:var(--ink2);font-size:12px}
th .arrow{color:var(--muted);font-size:10px;margin-left:3px}
td.num{text-align:right}
tr.invalid td{background:#fff3f3}tr.timeout td{background:#fff9ec}tr.error td{background:#fff3f3}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-weight:600;font-size:11px;color:#fff}
.badge.valid{background:var(--good)}.badge.invalid{background:var(--critical)}.badge.none{background:var(--muted)}
.badge.timeout{background:#b26b00}.badge.pf{background:var(--serious)}
.agree{color:#006300}.disagree{color:var(--critical);font-weight:600}.na{color:var(--muted)}.note{color:#8a5a00}
.caps{max-width:220px;overflow:hidden;text-overflow:ellipsis}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
details summary{cursor:pointer;color:var(--blue)}
.svgwrap svg{max-width:420px;height:auto;display:block;background:#fff;border:1px solid var(--grid);border-radius:6px}
ul.parts{margin:4px 0 0 0;padding-left:16px;white-space:normal;max-width:420px}
.foot{color:var(--muted);font-size:11px;margin-top:16px}
"""

_JS = """
(function(){
  const table=document.getElementById('runs');
  const tbody=table.tBodies[0];
  const rows=Array.from(tbody.rows);
  const q=document.getElementById('q');
  const selects=Array.from(document.querySelectorAll('select[data-col]'));
  const count=document.getElementById('count');
  function apply(){
    const text=q.value.trim().toLowerCase();
    let shown=0;
    for(const tr of rows){
      let ok=true;
      if(text && !tr.textContent.toLowerCase().includes(text)) ok=false;
      for(const s of selects){
        if(!ok) break;
        if(s.value && tr.dataset[s.dataset.col]!==s.value) ok=false;
      }
      tr.style.display=ok?'':'none';
      if(ok) shown++;
    }
    count.textContent=shown+' / '+rows.length+' runs shown';
  }
  q.addEventListener('input',apply);
  selects.forEach(s=>s.addEventListener('change',apply));
  let sortCol=-1,asc=true;
  Array.from(table.tHead.rows[0].cells).forEach((th,i)=>{
    th.addEventListener('click',()=>{
      asc=(sortCol===i)?!asc:true; sortCol=i;
      const numeric=th.dataset.type==='num';
      const key=tr=>{const c=tr.cells[i]; const v=c.dataset.sort!==undefined?c.dataset.sort:c.textContent.trim();
        return numeric?(v===''||v==='—'?(asc?Infinity:-Infinity):parseFloat(v)):v.toLowerCase();};
      rows.sort((a,b)=>{const x=key(a),y=key(b); if(x<y) return asc?-1:1; if(x>y) return asc?1:-1; return 0;});
      rows.forEach(tr=>tbody.appendChild(tr));
      Array.from(table.tHead.rows[0].cells).forEach(c=>{const a=c.querySelector('.arrow'); if(a) a.textContent='';});
      th.querySelector('.arrow').textContent=asc?'▲':'▼';
    });
  });
  apply();
})();
"""


def _badge(row: dict[str, Any]) -> str:
    status = row.get("status")
    if row.get("valid") is True:
        return '<span class="badge valid">✔ VALID</span>'
    if row.get("valid") is False:
        return '<span class="badge invalid">✘ INVALID</span>'
    if status == "timeout" or row.get("timed_out"):
        return '<span class="badge timeout">⏱ TIMEOUT</span>'
    if status == "precondition_failed":
        return '<span class="badge pf">precondition failed</span>'
    if status == "infeasible":
        return '<span class="badge none">infeasible (oracle)</span>'
    if status == "error":
        return '<span class="badge invalid">✘ ERROR</span>'
    return f'<span class="badge none">{html.escape(str(status))}</span>'


def _baseline_html(row: dict[str, Any], oracles: dict[str, dict[str, Any]]) -> str:
    if _label(row).split(":", 1)[0] in ORACLES:
        return '<span class="na">(is a baseline)</span>'
    if not oracles:
        return '<span class="na">—</span>'
    bits = []
    for olbl in sorted(oracles):
        orow = oracles[olbl]
        cls, text = _agreement(row, orow)
        t = (orow.get("median") or {}).get("wall_time")
        bits.append(f'<span class="{cls}" title="{html.escape(text)}">{html.escape(olbl)} {_fmt_s(t)} '
                    f'{"✔" if cls == "agree" else ("✘" if cls == "disagree" else "·")}</span>')
    return "<br>".join(bits)


def _row_html(idx: int, row: dict[str, Any], oracles: dict[str, dict[str, Any]], with_svg: bool) -> str:
    inst = row.get("instance") or {}
    res = row.get("result") or {}
    stats = res.get("stats") or {}
    med = row.get("median") or {}
    child = row.get("child") or {}
    caps = inst.get("capacities") or []
    verdict = "valid" if row.get("valid") is True else ("invalid" if row.get("valid") is False else "none")
    cls = "invalid" if row.get("valid") is False or row.get("status") == "error" else (
        "timeout" if row.get("timed_out") else "")
    fam = str(inst.get("family") or "?")
    variant = (inst.get("variant") or {}).get("type", "plain")
    wall = med.get("wall_time")
    rss = child.get("peak_rss_mb")
    msg = res.get("message") or row.get("error") or ""
    errs = res.get("verify_errors") or []
    title = html.escape("; ".join([msg] + list(errs))[:600])
    cells = [
        f'<td class="num">{idx}</td>',
        f'<td class="mono">{html.escape(_instance_name(row))}</td>',
        f"<td>{html.escape(fam)}</td>",
        f"<td>{html.escape(variant)}</td>",
        f'<td class="num">{row.get("n")}</td>',
        f'<td class="num">{row.get("m")}</td>',
        f'<td class="num">{row.get("k")}</td>',
        f'<td class="num" data-sort="{float(row.get("density") or 0):.6f}">{float(row.get("density") or 0):.3f}</td>',
        f"<td>{html.escape(_connectivity_text(row))}</td>",
        f'<td class="caps mono" title="{html.escape(_fmt_list(caps, 200))}">{html.escape(_fmt_list(caps))}</td>',
        f"<td>{html.escape(_label(row))}</td>",
        f'<td title="{title}">{html.escape(str(row.get("status")))}</td>',
        f'<td class="num" data-sort="{"" if wall is None else wall}">{_fmt_s(wall)}</td>',
        f'<td class="num" data-sort="{"" if med.get("verify_time") is None else med.get("verify_time")}">{_fmt_s(med.get("verify_time"))}</td>',
        f'<td class="num" data-sort="{"" if rss is None else rss}">{_fmt_mb(rss)}</td>',
        f'<td class="mono">{html.escape(_counters_text(stats))}</td>',
        f'<td class="mono">{html.escape(_fmt_list(res.get("part_sizes")))}</td>',
        f'<td title="{title}">{_badge(row)}</td>',
        f"<td>{_baseline_html(row, oracles)}</td>",
        f"<td>{graph_html(row) if with_svg else ''}</td>",
    ]
    data = (f'data-family="{html.escape(fam)}" data-algorithm="{html.escape(_label(row))}" '
            f'data-status="{html.escape(str(row.get("status")))}" data-verdict="{verdict}" '
            f'data-config="{html.escape(str(row.get("config")))}"')
    return f'<tr class="{cls}" {data}>{"".join(cells)}</tr>'


def dashboard_html(rows: Sequence[dict[str, Any]], title: str = "glsolver verification dashboard",
                   with_svg: bool = True) -> str:
    """Render the dashboard as one self-contained HTML string."""
    rows = list(rows)
    by_inst: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_inst[_instance_name(r)][_label(r)] = r
    n_valid = sum(1 for r in rows if r.get("valid") is True)
    n_invalid = sum(1 for r in rows if r.get("valid") is False)
    n_timeout = sum(1 for r in rows if r.get("timed_out"))
    n_error = sum(1 for r in rows if r.get("status") == "error")
    n_pf = sum(1 for r in rows if r.get("status") == "precondition_failed")
    disagreements = 0
    for r in rows:
        if _label(r).split(":", 1)[0] in ORACLES:
            continue
        for olbl, orow in by_inst[_instance_name(r)].items():
            if olbl.split(":", 1)[0] in ORACLES and _agreement(r, orow)[0] == "disagree":
                disagreements += 1
    machine = (rows[-1].get("machine") if rows else None) or {}
    cpu = machine.get("cpu") or {}
    configs = sorted({str(r.get("config")) for r in rows})
    families = sorted({str((r.get("instance") or {}).get("family") or "?") for r in rows})
    algos = sorted({_label(r) for r in rows})
    statuses = sorted({str(r.get("status")) for r in rows})

    def options(vals: Sequence[str]) -> str:
        return "".join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in vals)

    body_rows = []
    for i, r in enumerate(sorted(rows, key=lambda r: (int(r.get("n") or 0), _instance_name(r), _label(r))), 1):
        oracles = {lbl: orow for lbl, orow in by_inst[_instance_name(r)].items() if lbl.split(":", 1)[0] in ORACLES}
        body_rows.append(_row_html(i, r, oracles, with_svg))
    headers = [
        ("#", "num"), ("instance", "str"), ("family", "str"), ("variant", "str"), ("n", "num"), ("m", "num"),
        ("k", "num"), ("density", "num"), ("connectivity", "str"), ("capacities", "str"), ("algorithm", "str"),
        ("status", "str"), ("runtime (median)", "num"), ("verify time", "num"), ("peak RSS", "num"),
        ("primitive calls", "str"), ("part sizes", "str"), ("VERIFIER", "str"), ("baseline (oracle)", "str"),
        ("graph", "str"),
    ]
    thead = "".join(f'<th data-type="{t}">{html.escape(h)}<span class="arrow"></span></th>' for h, t in headers)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    tiles = [
        ("", len(rows), "runs"),
        ("good" if n_invalid == 0 else "", n_valid, "VALID (independent verifier)"),
        ("bad" if n_invalid else "", n_invalid, "INVALID"),
        ("warn" if n_timeout else "", n_timeout, "timeouts"),
        ("bad" if n_error else "", n_error, "errors"),
        ("", n_pf, "precondition failed"),
        ("bad" if disagreements else "good", disagreements, "oracle disagreements"),
    ]
    tiles_html = "".join(f'<div class="tile {cls}"><div class="v">{val}</div><div class="l">{lab}</div></div>'
                         for cls, val, lab in tiles)
    machine_html = html.escape(
        f"{cpu.get('model') or '?'} · {cpu.get('cores_logical') or '?'} logical cores · "
        f"{(machine.get('memory') or {}).get('ram_gib') or '?'} GiB · {machine.get('platform') or '?'} · "
        f"Python {(machine.get('python') or {}).get('version') or '?'} · glsolver "
        f"{rows[-1].get('glsolver_version') if rows else '?'} @ {str(rows[-1].get('git_commit') if rows else '?')[:12]}"
        f" · core built: {(machine.get('core') or {}).get('available')}"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class="sub">Every run of the benchmark configs {html.escape(', '.join(configs))}; generated {now}.
Each row is one (instance, algorithm) pair executed in an isolated subprocess; the VERIFIER column is the verdict of the
independent verifier (<code>glsolver.verify</code>, shares no code with the solvers) on the returned partition.</div>
<div class="tiles">{tiles_html}</div>
<div class="machine">{machine_html}</div>
<div class="filters">
  <label>search <input id="q" type="search" placeholder="instance, algorithm, status…"></label>
  <label>family <select data-col="family"><option value="">all</option>{options(families)}</select></label>
  <label>algorithm <select data-col="algorithm"><option value="">all</option>{options(algos)}</select></label>
  <label>status <select data-col="status"><option value="">all</option>{options(statuses)}</select></label>
  <label>verifier <select data-col="verdict"><option value="">all</option><option value="valid">VALID</option><option value="invalid">INVALID</option><option value="none">not verified</option></select></label>
  <label>config <select data-col="config"><option value="">all</option>{options(configs)}</select></label>
  <span class="count" id="count"></span>
</div>
<div class="tablewrap"><table id="runs"><thead><tr>{thead}</tr></thead><tbody>
{chr(10).join(body_rows)}
</tbody></table></div>
<div class="foot">Legend: VALID = the partition satisfies terminal membership, exact sizes / weight bounds and connectivity of every part
(docs/paper_notes.md §1); baseline ✔ = agreement with the exact oracle on the same instance; runtime = median wall time over repeats
(verifier excluded); primitive calls: flow = max-flow calls, match = matchings, contract/delete/shift = contractions, arc deletions,
cycle shifts, mcf = min-cost flows, pop = heap pops. Click a header to sort.</div>
<script>{_JS}</script>
</body></html>
"""


def write_dashboard(rows: Sequence[dict[str, Any]], path: Path | str = DASHBOARD_PATH, **kw: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dashboard_html(rows, **kw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmarks.report", description="Summary tables + verification dashboard")
    ap.add_argument("results", nargs="*", help="JSONL files or directories (default benchmarks/results)")
    ap.add_argument("--html", default=str(DASHBOARD_PATH), help="dashboard output path")
    ap.add_argument("--markdown", default=str(SUMMARY_PATH), help="markdown summary output path")
    ap.add_argument("--no-svg", action="store_true", help="skip the inline graph drawings")
    args = ap.parse_args(argv)
    paths: Any = None
    if args.results:
        paths = []
        for p in args.results:
            pp = Path(p)
            paths += sorted(pp.glob("*.jsonl")) if pp.is_dir() else [pp]
    rows = load_results(paths)
    md = write_summary_markdown(rows, args.markdown)
    dash = write_dashboard(rows, args.html, with_svg=not args.no_svg)
    print(md)
    print(dash)
    n_invalid = sum(1 for r in rows if r.get("valid") is False)
    print(f"{len(rows)} rows, {n_invalid} INVALID", file=sys.stderr)
    return 3 if n_invalid else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
