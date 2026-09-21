"""Publication-quality figures from raw benchmark JSONL rows.

Figures (all written to ``benchmarks/results/plots/`` by default):

* ``runtime_vs_n_<family>.png`` and ``runtime_vs_n_all.png`` -- log-log median
  runtime with an IQR band, one line per algorithm; the slope of
  ``log(runtime)`` vs ``log(n)`` fitted on the largest sizes is printed in the
  legend and next to the last point;
* ``runtime_vs_k.png`` -- runtime against ``k`` at the largest common ``n``;
* ``primitive_calls_vs_n.png`` -- flow / matching / contraction / deletion /
  cycle-shift / min-cost-flow / heap counters against ``n``;
* ``peak_rss_vs_n.png`` -- peak resident set size of the worker process;
* ``speedup_reference_vs_core.png`` -- reference-solver time divided by
  core-solver time on identical instances;
* ``oracle_vs_solver.png`` -- oracle (brute force / ILP) time against solver
  time on the small instances where both ran, agreement marked;
* ``dag_variants.png`` -- heap vs linear DAG variants (and policies).

Rules that keep the plots honest: every runtime axis is logarithmic with the
fitted exponent printed; no point is dropped for being slow; a run that timed
out is drawn as a hollow marker *at the timeout value* on a dotted "timeout"
line, and a run that crashed (status ``error``) as an ``x`` marker at the same
level with an "(errors)" legend entry, so a missing curve never hides bad
scaling or a crash.  In the oracle-vs-solver figure the classification of a
pair is the dashboard's (:func:`benchmarks.report._agreement`): only a genuine
disagreement is drawn red; an oracle that timed out or crashed is a grey
hollow marker at the timeout level, never a "disagreement".

Usage::

    python -m benchmarks.plot [RESULTS.jsonl ...] [--out DIR] [--metric wall_time]
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.runner import RESULTS_DIR, load_results  # noqa: E402

PLOTS_DIR = RESULTS_DIR / "plots"

# Categorical palette (validated order, light surface); colour follows the
# algorithm *identity*, never its position in a particular figure.
ALGO_ORDER = ["general", "weighted", "dag", "reference", "reference-weighted", "reference-dag",
              "bruteforce", "ilp", "auto"]
ALGO_COLOR = {
    "general": "#2a78d6", "weighted": "#eb6834", "dag": "#1baf7a", "reference": "#eda100",
    "reference-weighted": "#e87ba4", "reference-dag": "#008300", "bruteforce": "#4a3aa7",
    "ilp": "#e34948", "auto": "#52514e",
}
ALGO_MARKER = {
    "general": "o", "weighted": "s", "dag": "^", "reference": "D", "reference-weighted": "v",
    "reference-dag": "P", "bruteforce": "X", "ilp": "*", "auto": "h",
}
VARIANT_STYLES = ["-", "--", ":", "-."]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COUNTERS = ["max_flow_calls", "matching_calls", "contractions", "deletions", "cycle_shifts",
            "min_cost_flow_calls", "heap_pops"]
REFERENCE_OF_CORE = {"general": "reference", "weighted": "reference-weighted", "dag": "reference-dag"}


# ---------------------------------------------------------------------------
# data access helpers
# ---------------------------------------------------------------------------
def base_algorithm(label: str) -> str:
    return label.split(":", 1)[0]


def row_metric(row: dict[str, Any], metric: str = "wall_time") -> float | None:
    """The row's median value of ``metric`` (``wall_time``, ``solver_runtime``, ``cpu_time``,
    ``verify_time``) or the child's ``peak_rss_mb``; ``None`` when unavailable."""
    if metric == "peak_rss_mb":
        return (row.get("child") or {}).get("peak_rss_mb")
    return (row.get("median") or {}).get(metric)


def row_family(row: dict[str, Any]) -> str:
    inst = row.get("instance") or {}
    fam = str(inst.get("family") or "?")
    var = (inst.get("variant") or {}).get("type", "plain")
    return fam if var == "plain" else f"{fam}+{var}"


def _style(label: str, seen_variants: dict[str, list[str]]) -> dict[str, Any]:
    base = base_algorithm(label)
    variants = seen_variants.setdefault(base, [])
    if label not in variants:
        variants.append(label)
    idx = variants.index(label)
    return {
        "color": ALGO_COLOR.get(base, MUTED),
        "marker": ALGO_MARKER.get(base, "o"),
        "linestyle": VARIANT_STYLES[idx % len(VARIANT_STYLES)],
    }


def _sorted_labels(labels: Iterable[str]) -> list[str]:
    def key(lbl: str) -> tuple[int, str]:
        base = base_algorithm(lbl)
        return (ALGO_ORDER.index(base) if base in ALGO_ORDER else len(ALGO_ORDER), lbl)

    return sorted(set(labels), key=key)


def aggregate_by_x(rows: Sequence[dict[str, Any]], xkey: Callable[[dict[str, Any]], Any],
                   metric: str = "wall_time") -> dict[Any, dict[str, Any]]:
    """Per distinct ``x``: median / IQR of the successful rows' medians, plus timeout and error counts.

    ``timeout_value`` is the largest configured timeout among the timed-out
    rows and ``error_level`` the same among the crashed rows (the level at
    which their markers are drawn); both are ``None`` when there are none.
    """
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[xkey(r)].append(r)
    out: dict[Any, dict[str, Any]] = {}
    for x, rs in buckets.items():
        vals = [row_metric(r, metric) for r in rs if r.get("status") == "ok" and row_metric(r, metric) is not None]
        timeouts = [r for r in rs if r.get("timed_out")]
        errors = [r for r in rs if r.get("status") == "error" and not r.get("timed_out")]
        entry: dict[str, Any] = {"n_ok": len(vals), "n_timeout": len(timeouts), "n_error": len(errors),
                                 "n_rows": len(rs),
                                 "timeout_value": max((float(r.get("timeout") or 0) for r in timeouts), default=None),
                                 "error_level": max((float(r.get("timeout") or 0) for r in errors), default=None)}
        if vals:
            vals.sort()
            entry["median"] = statistics.median(vals)
            if len(vals) >= 2:
                q = statistics.quantiles(vals, n=4, method="inclusive")
                entry["q1"], entry["q3"] = q[0], q[2]
            else:
                entry["q1"] = entry["q3"] = vals[0]
        out[x] = entry
    return out


def fit_slope(xs: Sequence[float], ys: Sequence[float], top: int = 4) -> tuple[float, float] | None:
    """Least-squares slope/intercept of ``log y`` vs ``log x`` on the ``top`` largest distinct ``x``."""
    pts = sorted({(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y is not None and y > 0})
    pts = pts[-top:]
    if len(pts) < 2:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0:
        return None
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    return slope, my - slope * mx


# ---------------------------------------------------------------------------
# matplotlib setup
# ---------------------------------------------------------------------------
def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.8, "lines.markersize": 5.5, "axes.titlesize": 10, "axes.titleweight": "bold",
        "figure.dpi": 100,
    })
    return plt


def _finish(fig: Any, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _draw_failure_markers(ax: Any, agg: dict[Any, dict[str, Any]], xs: Sequence[Any], st: dict[str, Any],
                          label: str, has_curve: bool, timeout_levels: set[float],
                          pending_errors: list[tuple[Any, str, dict[str, Any]]]) -> bool:
    """Draw the timeout (hollow) and error (``x``) markers of one algorithm; return whether any was drawn.

    Timeouts sit at the configured timeout value.  Errors sit at the crashed
    rows' timeout value too; a crashed row without one is queued in
    ``pending_errors`` and placed at the top of the axis once the data range is
    known (:func:`_flush_pending_errors`).
    """
    drawn = False
    tos = [(x, agg[x]["timeout_value"]) for x in xs if agg[x]["n_timeout"] and agg[x]["timeout_value"]]
    if tos:
        ax.plot([t[0] for t in tos], [t[1] for t in tos], linestyle="none", marker=st["marker"],
                markerfacecolor="none", markeredgecolor=st["color"], markersize=8,
                label=None if has_curve else f"{label} (timeouts)")
        timeout_levels.update(t[1] for t in tos)
        drawn = True
    errs = [(x, agg[x]["error_level"]) for x in xs if agg[x]["n_error"]]
    placed = [(x, lvl) for x, lvl in errs if lvl]
    if placed:
        ax.plot([e[0] for e in placed], [e[1] for e in placed], linestyle="none", marker="x",
                markeredgecolor=st["color"], markeredgewidth=1.8, markersize=9, label=f"{label} (errors)")
        drawn = True
    for x, lvl in errs:
        if not lvl:
            pending_errors.append((x, label, st))
    return drawn


def _flush_pending_errors(ax: Any, pending: Sequence[tuple[Any, str, dict[str, Any]]]) -> None:
    """Place the error markers that had no timeout level at the top of the (now autoscaled) axis."""
    if not pending:
        return
    top = ax.get_ylim()[1]
    by_label: dict[str, list[Any]] = defaultdict(list)
    styles: dict[str, dict[str, Any]] = {}
    for x, label, st in pending:
        by_label[label].append(x)
        styles[label] = st
    for label, xs in by_label.items():
        ax.plot(xs, [top] * len(xs), linestyle="none", marker="x", markeredgecolor=styles[label]["color"],
                markeredgewidth=1.8, markersize=9, label=f"{label} (errors)")


def _draw_scaling_axis(ax: Any, rows: Sequence[dict[str, Any]], metric: str, ylabel: str,
                       seen_variants: dict[str, list[str]], title: str = "", slope_top: int = 4,
                       xkey: str = "n") -> bool:
    """Log-log ``metric`` vs ``n`` for every algorithm label present in ``rows``. Returns whether anything was drawn."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[str(r.get("algorithm_label") or r.get("algorithm"))].append(r)
    drawn = False
    timeout_levels: set[float] = set()
    pending_errors: list[tuple[Any, str, dict[str, Any]]] = []
    for label in _sorted_labels(by_label):
        agg = aggregate_by_x(by_label[label], lambda r: int(r.get(xkey) or 0), metric)
        xs = sorted(x for x in agg if x > 0)
        pts = [(x, agg[x]["median"]) for x in xs if "median" in agg[x]]
        st = _style(label, seen_variants)
        name = label
        fit = fit_slope([p[0] for p in pts], [p[1] for p in pts], top=slope_top) if len(pts) >= 2 else None
        if fit is not None:
            name = f"{label}  (slope {fit[0]:.2f})"
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, **st)
            lo = [agg[x]["q1"] for x, _ in pts]
            hi = [agg[x]["q3"] for x, _ in pts]
            ax.fill_between([p[0] for p in pts], lo, hi, color=st["color"], alpha=0.15, linewidth=0)
            if fit is not None:
                x_last, y_last = pts[-1]
                ax.annotate(f"{fit[0]:.2f}", (x_last, y_last), textcoords="offset points", xytext=(6, -2),
                            fontsize=7, color=st["color"])
            drawn = True
        if _draw_failure_markers(ax, agg, xs, st, label, bool(pts), timeout_levels, pending_errors):
            drawn = True
    for level in sorted(timeout_levels):
        ax.axhline(level, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(f"timeout {level:g}s", (ax.get_xlim()[0], level), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=MUTED)
    if drawn:
        ax.set_xscale("log")
        ax.set_yscale("log")
        _flush_pending_errors(ax, pending_errors)
        ax.set_xlabel("n (vertices)" if xkey == "n" else xkey)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(loc="best")
    return drawn


# ---------------------------------------------------------------------------
# individual figures
# ---------------------------------------------------------------------------
def plot_runtime_vs_n(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "wall_time",
                      dpi: int = 150) -> list[Path]:
    plt = _mpl()
    written: list[Path] = []
    fams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fams[row_family(r)].append(r)
    fams = {f: rs for f, rs in fams.items() if len({int(r.get("n") or 0) for r in rs}) >= 2}
    if not fams:
        return written
    seen: dict[str, list[str]] = {}
    ylabel = {"wall_time": "median wall time (s)", "solver_runtime": "median solver time (s)",
              "cpu_time": "median CPU time (s)"}.get(metric, metric)
    for fam, rs in sorted(fams.items()):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if _draw_scaling_axis(ax, rs, metric, ylabel, seen, title=f"runtime vs n — {fam}"):
            written.append(_finish(fig, out_dir / f"runtime_vs_n_{_safe(fam)}.png", dpi))
        else:
            plt.close(fig)
    if len(fams) > 1:
        cols = min(3, len(fams))
        nrows = math.ceil(len(fams) / cols)
        fig, axes = plt.subplots(nrows, cols, figsize=(5.2 * cols, 3.8 * nrows), squeeze=False)
        for ax, (fam, rs) in zip(axes.flat, sorted(fams.items())):
            _draw_scaling_axis(ax, rs, metric, ylabel, seen, title=fam)
        for ax in list(axes.flat)[len(fams):]:
            ax.set_visible(False)
        written.append(_finish(fig, out_dir / "runtime_vs_n_all.png", dpi))
    return written


def plot_runtime_vs_k(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "wall_time",
                      dpi: int = 150) -> list[Path]:
    plt = _mpl()
    fams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fams[row_family(r)].append(r)
    panels: list[tuple[str, int, list[dict[str, Any]]]] = []
    for fam, rs in sorted(fams.items()):
        by_n: dict[int, set[int]] = defaultdict(set)
        for r in rs:
            by_n[int(r.get("n") or 0)].add(int(r.get("k") or 0))
        cands = [n for n, ks in by_n.items() if len(ks) >= 2]
        if not cands:
            continue
        n_star = max(cands)
        panels.append((fam, n_star, [r for r in rs if int(r.get("n") or 0) == n_star]))
    if not panels:
        return []
    cols = min(3, len(panels))
    nrows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5.2 * cols, 3.8 * nrows), squeeze=False)
    seen: dict[str, list[str]] = {}
    for ax, (fam, n_star, rs) in zip(axes.flat, panels):
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            by_label[str(r.get("algorithm_label") or r.get("algorithm"))].append(r)
        timeout_levels: set[float] = set()
        pending_errors: list[tuple[Any, str, dict[str, Any]]] = []
        for label in _sorted_labels(by_label):
            agg = aggregate_by_x(by_label[label], lambda r: int(r.get("k") or 0), metric)
            ks = sorted(agg)
            pts = [(k, agg[k]["median"]) for k in ks if "median" in agg[k]]
            st = _style(label, seen)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=label, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, **st)
                ax.fill_between([p[0] for p in pts], [agg[k]["q1"] for k, _ in pts],
                                [agg[k]["q3"] for k, _ in pts], color=st["color"], alpha=0.15, linewidth=0)
            _draw_failure_markers(ax, agg, ks, st, label, bool(pts), timeout_levels, pending_errors)
        for level in sorted(timeout_levels):
            ax.axhline(level, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _flush_pending_errors(ax, pending_errors)
        ax.set_xlabel("k (terminals)")
        ax.set_ylabel("median wall time (s)")
        ax.set_title(f"{fam}, n = {n_star}")
        ax.legend(loc="best")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    return [_finish(fig, out_dir / "runtime_vs_k.png", dpi)]


def _scalable(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of grid families only (the fixed paper examples / counterexample have no size axis)."""
    return [r for r in rows if (r.get("instance") or {}).get("mode") != "fixed"]


def plot_primitive_calls(rows: Sequence[dict[str, Any]], out_dir: Path, dpi: int = 150) -> list[Path]:
    plt = _mpl()
    ok = [r for r in _scalable(rows) if r.get("status") == "ok"]
    present = [c for c in COUNTERS if any(((r.get("result") or {}).get("stats") or {}).get(c) for r in ok)]
    if not present or len({int(r.get("n") or 0) for r in ok}) < 2:
        return []
    cols = min(3, len(present))
    nrows = math.ceil(len(present) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5.2 * cols, 3.8 * nrows), squeeze=False)
    seen: dict[str, list[str]] = {}
    for ax, counter in zip(axes.flat, present):
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in ok:
            val = ((r.get("result") or {}).get("stats") or {}).get(counter)
            if isinstance(val, (int, float)) and val > 0:
                by_label[str(r.get("algorithm_label") or r.get("algorithm"))].append(
                    {"n": r.get("n"), "status": "ok", "median": {"x": float(val)}})
        for label in _sorted_labels(by_label):
            agg = aggregate_by_x(by_label[label], lambda r: int(r.get("n") or 0), "x")
            xs = sorted(x for x in agg if "median" in agg[x])
            if not xs:
                continue
            st = _style(label, seen)
            ys = [agg[x]["median"] for x in xs]
            fit = fit_slope(xs, ys)
            name = f"{label} (slope {fit[0]:.2f})" if fit else label
            ax.plot(xs, ys, label=name, markeredgecolor=SURFACE, markeredgewidth=0.8, **st)
            ax.fill_between(xs, [agg[x]["q1"] for x in xs], [agg[x]["q3"] for x in xs],
                            color=st["color"], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (vertices)")
        ax.set_ylabel(counter.replace("_", " "))
        ax.set_title(counter.replace("_", " "))
        ax.legend(loc="best")
    for ax in list(axes.flat)[len(present):]:
        ax.set_visible(False)
    return [_finish(fig, out_dir / "primitive_calls_vs_n.png", dpi)]


def plot_peak_rss(rows: Sequence[dict[str, Any]], out_dir: Path, dpi: int = 150) -> list[Path]:
    plt = _mpl()
    rs = [r for r in _scalable(rows) if (r.get("child") or {}).get("peak_rss_mb")]
    if len({int(r.get("n") or 0) for r in rs}) < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    seen: dict[str, list[str]] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rs:
        by_label[str(r.get("algorithm_label") or r.get("algorithm"))].append(
            {"n": r.get("n"), "status": "ok", "median": {"x": float(r["child"]["peak_rss_mb"])}})
    for label in _sorted_labels(by_label):
        agg = aggregate_by_x(by_label[label], lambda r: int(r.get("n") or 0), "x")
        xs = sorted(x for x in agg if "median" in agg[x])
        st = _style(label, seen)
        ys = [agg[x]["median"] for x in xs]
        fit = fit_slope(xs, ys)
        ax.plot(xs, ys, label=f"{label} (slope {fit[0]:.2f})" if fit else label,
                markeredgecolor=SURFACE, markeredgewidth=0.8, **st)
        ax.fill_between(xs, [agg[x]["q1"] for x in xs], [agg[x]["q3"] for x in xs],
                        color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("peak RSS of worker (MB)")
    ax.set_title("peak memory vs n")
    ax.legend(loc="best")
    return [_finish(fig, out_dir / "peak_rss_vs_n.png", dpi)]


def _by_instance(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        name = str((r.get("instance") or {}).get("name"))
        out[name][str(r.get("algorithm_label") or r.get("algorithm"))] = r
    return out


def plot_speedup(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "wall_time",
                 dpi: int = 150) -> list[Path]:
    plt = _mpl()
    per_inst = _by_instance(rows)
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for name, by_label in per_inst.items():
        for core, ref in REFERENCE_OF_CORE.items():
            core_rows = [r for lbl, r in by_label.items() if base_algorithm(lbl) == core]
            ref_row = by_label.get(ref)
            if ref_row is None or ref_row.get("status") != "ok":
                continue
            t_ref = row_metric(ref_row, metric)
            for cr in core_rows:
                t_core = row_metric(cr, metric)
                if cr.get("status") == "ok" and t_core and t_ref:
                    series[f"{ref} / {cr.get('algorithm_label')}"].append((int(cr.get("n") or 0), t_ref / t_core))
    if not series:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    seen: dict[str, list[str]] = {}
    for label in sorted(series):
        pts = sorted(series[label])
        core_label = label.split(" / ", 1)[1]
        st = _style(core_label, seen)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none", label=label, alpha=0.85,
                marker=st["marker"], color=st["color"], markeredgecolor=SURFACE, markeredgewidth=0.8,
                markersize=7)
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("speedup = t(reference) / t(core)")
    ax.set_title("reference vs core speedup (identical instances)")
    ax.legend(loc="best")
    return [_finish(fig, out_dir / "speedup_reference_vs_core.png", dpi)]


def classify_oracle_pair(solver_row: dict[str, Any], oracle_row: dict[str, Any]) -> str:
    """Plot class of a (solver, oracle) pair on one instance, consistent with the dashboard.

    Returns ``"agree"``, ``"disagree"``, ``"oracle timeout"``, ``"oracle
    error"`` or ``"na"`` -- derived from :func:`benchmarks.report._agreement`
    so that the figure and the dashboard never classify the same pair
    differently.
    """
    from benchmarks.report import _agreement  # local import: report imports this package's runner

    cls, text = _agreement(solver_row, oracle_row)
    if cls in ("agree", "disagree"):
        return cls
    if text == "oracle timeout":
        return "oracle timeout"
    if text == "oracle error":
        return "oracle error"
    return "na"


# marker style per class: (face colour or None for the algorithm colour, edge colour or None, legend tag)
_ORACLE_CLASS_STYLE = {
    "agree": (None, None, ""),
    "disagree": ("none", "#d03b3b", " (DISAGREE)"),
    "oracle timeout": ("none", MUTED, " (oracle timeout)"),
    "oracle error": ("none", MUTED, " (oracle error)"),
    "na": (MUTED, None, " (n/a)"),
}


def plot_oracle_vs_solver(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "wall_time",
                          dpi: int = 150) -> list[Path]:
    """Oracle time against solver time per instance; only a real disagreement is drawn red.

    A timed-out or crashed oracle has no time: it is drawn as a grey hollow
    marker at its timeout level (dotted line) so the slow instance stays
    visible without being presented as a correctness disagreement.
    """
    plt = _mpl()
    per_inst = _by_instance(rows)
    pts: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    timeout_levels: set[float] = set()
    for name, by_label in per_inst.items():
        oracles = {lbl: r for lbl, r in by_label.items() if base_algorithm(lbl) in ("bruteforce", "ilp")}
        solvers = {lbl: r for lbl, r in by_label.items() if base_algorithm(lbl) not in ("bruteforce", "ilp")}
        for olbl, orow in oracles.items():
            t_o = row_metric(orow, metric)
            for slbl, srow in solvers.items():
                t_s = row_metric(srow, metric)
                if t_s is None:
                    continue
                cls = classify_oracle_pair(srow, orow)
                y = t_o
                if cls in ("oracle timeout", "oracle error"):
                    y = float(orow.get("timeout") or 0) or None
                    if y is not None:
                        timeout_levels.add(y)
                if y is None:
                    continue
                pts[f"{olbl} vs {slbl}"].append((t_s, y, cls))
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    seen: dict[str, list[str]] = {}
    lo, hi = float("inf"), 0.0
    for label in sorted(pts):
        st = _style(label.split(" vs ", 1)[0], seen)
        for cls, (face, edge, tag) in _ORACLE_CLASS_STYLE.items():
            arr = [(x, y) for x, y, c in pts[label] if c == cls]
            if not arr:
                continue
            ax.plot([p[0] for p in arr], [p[1] for p in arr], linestyle="none", marker=st["marker"],
                    markerfacecolor=st["color"] if face is None else face,
                    markeredgecolor=SURFACE if edge is None else edge,
                    markeredgewidth=(0.8 if edge is None else 1.5), color=st["color"], label=label + tag,
                    alpha=0.9, markersize=7)
            lo = min(lo, *[min(p) for p in arr])
            hi = max(hi, *[max(p) for p in arr])
    if lo < hi:
        ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle=":", linewidth=1)
    for level in sorted(timeout_levels):
        ax.axhline(level, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(f"oracle timeout {level:g}s", (ax.get_xlim()[0], level), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("solver median wall time (s)")
    ax.set_ylabel("oracle median wall time (s)")
    ax.set_title("oracle vs solver on small instances (filled = agree)")
    ax.legend(loc="best")
    return [_finish(fig, out_dir / "oracle_vs_solver.png", dpi)]


def plot_dag_variants(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "wall_time",
                      dpi: int = 150) -> list[Path]:
    plt = _mpl()
    rs = [r for r in _scalable(rows)
          if base_algorithm(str(r.get("algorithm_label") or r.get("algorithm"))) in ("dag", "reference-dag")]
    if len({str(r.get("algorithm_label")) for r in rs}) < 1 or len({int(r.get("n") or 0) for r in rs}) < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    seen: dict[str, list[str]] = {}
    if not _draw_scaling_axis(ax, rs, metric, "median wall time (s)", seen, title="DAG solver variants"):
        plt.close(fig)
        return []
    return [_finish(fig, out_dir / "dag_variants.png", dpi)]


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def make_plots(rows_or_paths: Any = None, out_dir: Path | str = PLOTS_DIR, metric: str = "wall_time",
               dpi: int = 150) -> list[Path]:
    """Write every figure that has data. ``rows_or_paths`` is a list of rows, a path/list of JSONL
    files or ``None`` (all files in ``benchmarks/results``). Returns the written paths."""
    if rows_or_paths and isinstance(rows_or_paths, list) and isinstance(rows_or_paths[0], dict):
        rows = list(rows_or_paths)
    else:
        rows = load_results(rows_or_paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not rows:
        return written
    written += plot_runtime_vs_n(rows, out, metric, dpi)
    written += plot_runtime_vs_k(rows, out, metric, dpi)
    written += plot_primitive_calls(rows, out, dpi)
    written += plot_peak_rss(rows, out, dpi)
    written += plot_speedup(rows, out, metric, dpi)
    written += plot_oracle_vs_solver(rows, out, metric, dpi)
    written += plot_dag_variants(rows, out, metric, dpi)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmarks.plot", description="Plot benchmark JSONL results")
    ap.add_argument("results", nargs="*", help="JSONL files or directories (default benchmarks/results)")
    ap.add_argument("--out", default=str(PLOTS_DIR))
    ap.add_argument("--metric", default="wall_time", choices=["wall_time", "solver_runtime", "cpu_time"])
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)
    paths: Any = None
    if args.results:
        paths = []
        for p in args.results:
            pp = Path(p)
            paths += sorted(pp.glob("*.jsonl")) if pp.is_dir() else [pp]
    written = make_plots(paths, args.out, args.metric, args.dpi)
    for p in written:
        print(p)
    if not written:
        print("no figures written (no result rows found)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
