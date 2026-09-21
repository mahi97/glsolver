"""Abstracted figure of the official compact-connectivity counterexample
(docs/paper_notes.md §10, paper Lemma A.13 and Figure 20, authors' repo
``mahdi-jfri/Gyori-Lovasz-Codes``).

The full instance has ``n = 9 + 108 + 216·copies`` vertices and cannot be
drawn vertex by vertex; :func:`visualize_counterexample` draws its
*structure* instead:

* left: the nine terminals on a circle and, for every pair ``{i, j}``, the
  bundle ``P_{i,j} = {p^1, p^2, p^3}`` of pre-terminals on the chord
  ``t_i t_j`` (each pre-terminal split into the two colours of the terminals
  it is compact-connected to), plus the nine out-arcs of one exemplary
  forcing vertex ``v_e`` in orange;
* right (inset): that forcing gadget expanded exactly as in the paper's
  Figure 20 — the six displayed terminals ``a,b,c,d,x,y = t1..t6``, the nine
  out-neighbours ``P_{1,2} ∪ P_{1,3} ∪ {p^1_{4,5}, p^2_{4,5}, p^1_{5,6}}``
  and the targeted edge ``e = (p^1_{5,6}, t_6)`` highlighted;
* bottom: a text summary with the counts (also returned).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from glsolver.viz.layout import HIGHLIGHT_COLOR, terminal_color
from glsolver.viz.render import save_figure

K = 9


def counterexample_counts(copies: int = 1) -> dict[str, Any]:
    """Counts of the construction for ``copies`` forcing copies per arc (docs/paper_notes.md §10)."""
    pairs = K * (K - 1) // 2
    pre = 3 * pairs
    pre_arcs = 2 * pre
    forcing_kinds = pre_arcs
    forcing = forcing_kinds * copies
    return {
        "k": K,
        "copies": copies,
        "pairs": pairs,
        "pre_terminals": pre,
        "pre_terminal_arcs": pre_arcs,
        "forcing_kinds": forcing_kinds,
        "forcing_vertices": forcing,
        "forcing_arcs": forcing * K,
        "n": K + pre + forcing,
        "m": pre_arcs + forcing * K,
    }


def counterexample_summary(copies: int = 1, structure: dict[str, Any] | None = None) -> str:
    """Human-readable summary (counts and the claims) of the construction."""
    c = counterexample_counts(copies)
    caps = ""
    if structure is not None and "capacities" in structure:
        caps = "\n  capacities c = (" + ", ".join(str(x) for x in structure["capacities"]) + ")"
    return (
        f"Official counterexample [Lem A.13], copies = {copies}:\n"
        f"  k = {c['k']} terminals; {c['pairs']} pairs × 3 = {c['pre_terminals']} pre-terminals "
        f"with {c['pre_terminal_arcs']} arcs into T;\n"
        f"  one forcing vertex per pre-terminal arc × {copies} copies = {c['forcing_vertices']} forcing vertices, "
        f"9 out-arcs each ({c['forcing_arcs']} arcs);\n"
        f"  n = {c['n']}, m = {c['m']}{caps}\n"
        "  claims (authors' script, replicated by glref.counterexample.check_counterexample_claims):\n"
        "  compact connectivity holds; deleting ANY arc breaks it; contracting ANY pre-terminal breaks it."
    )


def _structure(copies: int) -> dict[str, Any]:
    from glref.counterexample import counterexample_structure

    return counterexample_structure(copies)


def _tname(i: int) -> str:
    return f"$t_{{{i + 1}}}$"


def _pname(r: int, i: int, j: int) -> str:
    return f"$p^{{{r}}}_{{{i + 1},{j + 1}}}$"


def _draw_split_node(ax, x: float, y: float, r: float, colors: list[str], label: str | None, *, zorder: float = 4, fontsize: float = 6.5, label_dy: float = 0.0) -> None:
    from matplotlib.patches import Circle, Wedge

    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="#333", lw=0.8, zorder=zorder))
    n = max(1, len(colors))
    for j, col in enumerate(colors):
        ax.add_patch(Wedge((x, y), r, 90 - 360 * (j + 1) / n, 90 - 360 * j / n, facecolor=col, edgecolor="white", lw=0.4, zorder=zorder + 0.1))
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor="#333", lw=0.8, zorder=zorder + 0.2))
    if label:
        ax.text(x, y + label_dy, label, fontsize=fontsize, ha="center", va="center", zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8) if label_dy == 0 else None)


def _draw_terminal(ax, x: float, y: float, s: float, i: int, *, zorder: float = 5, fontsize: float = 8) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={0.22 * s}", facecolor=terminal_color(i), edgecolor="#222", lw=1.0, zorder=zorder))
    ax.text(x, y, _tname(i), fontsize=fontsize, fontweight="bold", ha="center", va="center", zorder=zorder + 1)


def _arrow(ax, a, b, *, color, lw, shrink_a=0.0, shrink_b=0.0, alpha=1.0, zorder=2, ls="-"):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=4,head_width=2", shrinkA=shrink_a, shrinkB=shrink_b, color=color, lw=lw, alpha=alpha, zorder=zorder, linestyle=ls, mutation_scale=1.5))


def _exemplary_forcing(structure: dict[str, Any]) -> dict[str, Any]:
    """The paper's Figure 20 gadget: ``e = (p^1_{5,6}, t_6)``, ``(a,b,c,d,x,y) = (1..6)``."""
    for f in structure["forcing"]:
        if f["x"] == 4 and f["y"] == 5 and structure["pre_terminals"][f["edge"][0]]["r"] == 1:
            return f
    return structure["forcing"][0]


def counterexample_figure(copies: int = 1, *, structure: dict[str, Any] | None = None, figsize: tuple[float, float] = (14.0, 8.0), dpi: int = 100):
    """Matplotlib ``Figure`` of the abstracted construction (see the module docstring)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    structure = structure or _structure(copies)
    pre = structure["pre_terminals"]
    pair_to = {tuple(k) if isinstance(k, (list, tuple)) else k: v for k, v in structure["pair_to_pre_terminals"].items()}
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.01, 0.16, 0.50, 0.72])
    axg = fig.add_axes([0.53, 0.30, 0.46, 0.58])
    axt = fig.add_axes([0.02, 0.005, 0.96, 0.14])
    for a in (ax, axg, axt):
        a.axis("off")

    # ---------------- left: terminals on a circle, pre-terminal bundles on the chords
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.3, 1.34)
    ax.set_aspect("equal")
    R = 1.0
    tpos = {i: (R * math.cos(math.pi / 2 + 2 * math.pi * i / K), R * math.sin(math.pi / 2 + 2 * math.pi * i / K)) for i in range(K)}
    ppos: dict[int, tuple[float, float]] = {}
    delta = 0.05
    for (i, j), ps in sorted(pair_to.items()):
        a, b = tpos[i], tpos[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#c9c9c9", lw=0.7, zorder=1)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / d, dx / d
        for r_, p in enumerate(ps):
            off = (r_ - 1) * delta
            ppos[p] = (mx + nx * off, my + ny * off)
            _draw_split_node(ax, ppos[p][0], ppos[p][1], 0.02, [terminal_color(i), terminal_color(j)], None, zorder=3)
    ex = _exemplary_forcing(structure)
    vx, vy = -1.18, 0.95
    for tgt in ex["targets"]:
        _arrow(ax, (vx, vy), ppos[tgt], color=HIGHLIGHT_COLOR, lw=0.9, alpha=0.75, shrink_b=2, zorder=2)
    _draw_split_node(ax, vx, vy, 0.06, [terminal_color(ex["a"])], None, zorder=6)
    ax.text(vx, vy + 0.1, "$v_e$", fontsize=9, ha="center", va="bottom")
    p_e, y_e = ex["edge"]
    ax.plot([ppos[p_e][0], tpos[y_e][0]], [ppos[p_e][1], tpos[y_e][1]], color=HIGHLIGHT_COLOR, lw=2.4, zorder=2.5)
    for i in range(K):
        _draw_terminal(ax, tpos[i][0] * 1.13, tpos[i][1] * 1.13, 0.17, i)
    ax.set_title(
        f"structure: 9 terminals, 36 chords × 3 pre-terminals $p^r_{{i,j}}$ (halves = compact-connected terminals);\n"
        f"forcing vertices ({counterexample_counts(copies)['forcing_vertices']} = 216 × {copies}) omitted except $v_e$ for "
        f"$e = (p^1_{{5,6}}, t_6)$ (orange: its 9 out-arcs and the targeted edge)",
        fontsize=9, loc="left",
    )

    # ---------------- right: the forcing gadget (paper Figure 20)
    axg.set_xlim(-0.6, 9.6)
    axg.set_ylim(-0.9, 4.2)
    axg.set_aspect("equal")
    row = list(ex["targets"])  # P_{a,b} ∪ P_{a,c} ∪ {p^1_{d,x}, p^2_{d,x}, p^r_{x,y}}: nine pre-terminals
    xs = [j for j in range(len(row))]
    gpos = {p: (float(x), 1.9) for p, x in zip(row, xs)}
    shown_terms = sorted({ex["a"], ex["b"], ex["c"], ex["d"], ex["x"], ex["y"]})
    tx = {t: 0.4 + 8.2 * j / max(1, len(shown_terms) - 1) for j, t in enumerate(shown_terms)}
    gt = {t: (tx[t], 0.0) for t in shown_terms}
    vpos = (4.0, 3.7)
    for p in row:
        _arrow(axg, vpos, gpos[p], color="#555555", lw=0.9, shrink_a=8, shrink_b=7, zorder=2)
    for p in row:
        info = pre[p]
        i, j = info["pair"]
        for t in (i, j):
            is_target = (p, t) == tuple(ex["edge"])
            _arrow(axg, gpos[p], gt[t], color=HIGHLIGHT_COLOR if is_target else "#555555", lw=2.6 if is_target else 0.9, shrink_a=7, shrink_b=9, zorder=2.5 if is_target else 2)
    for p in row:
        info = pre[p]
        i, j = info["pair"]
        _draw_split_node(axg, gpos[p][0], gpos[p][1], 0.2, [terminal_color(i), terminal_color(j)], None, zorder=4)
        axg.text(gpos[p][0], gpos[p][1] + 0.32, _pname(info["r"], i, j), fontsize=7.5, ha="center", va="bottom")
    _draw_split_node(axg, vpos[0], vpos[1], 0.26, [terminal_color(ex["a"])], None, zorder=4)
    axg.text(vpos[0] + 0.4, vpos[1], "$v = v_e$", fontsize=9, ha="left", va="center")
    for t in shown_terms:
        _draw_terminal(axg, gt[t][0], gt[t][1], 0.42, t, fontsize=8)
    axg.text(9.3, 0.0, "$\\cdots\\ t_9$", fontsize=8, ha="right", va="center", color="#666")
    axg.set_title(
        "forcing gadget for $e = (p^1_{5,6}, t_6)$, $(a,b,c,d,x,y) = (1,2,3,4,5,6)$ (paper Fig. 20):\n"
        "$N^+(v_e) = P_{1,2} \\cup P_{1,3} \\cup \\{p^1_{4,5}, p^2_{4,5}, p^1_{5,6}\\}$; $v_e$ is compact-connected only to $t_1$\n"
        "and every witness of that connection uses the highlighted edge $e$",
        fontsize=9, loc="left",
    )

    # ---------------- bottom: summary text
    axt.text(0.0, 1.0, counterexample_summary(copies, structure), fontsize=8.3, family="monospace", va="top", ha="left")
    return fig


def visualize_counterexample(copies: int = 1, path: str | Path = "counterexample.svg", *, figsize: tuple[float, float] = (14.0, 8.0), dpi: int = 100) -> dict[str, Any]:
    """Write the abstracted counterexample figure to ``path`` (``.svg``/``.png``/``.pdf``).

    Returns ``{"path", "summary", "counts", "structure"}`` where ``summary``
    is the text block of the figure and ``counts`` the dictionary of
    :func:`counterexample_counts`.
    """
    structure = _structure(copies)
    fig = counterexample_figure(copies, structure=structure, figsize=figsize, dpi=dpi)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, dpi=dpi)  # byte-deterministic (no timestamp / random ids)
    return {
        "path": path,
        "summary": counterexample_summary(copies, structure),
        "counts": counterexample_counts(copies),
        "structure": structure,
    }
