"""Deterministic vertex layouts and the visual constants shared by every renderer.

Conventions follow the paper's figures (docs/paper_notes.md §7, paper Figure 4):
terminals sit on a bottom row, non-terminals above them.  Every layout is a
pure function of the instance and the seed, so all frames of a trace, the
HTML viewer and the animation reuse one and the same coordinates.

Methods of :func:`compute_layout`:

* ``"grid"`` (alias ``"layered"``): row ``d`` holds the vertices at directed
  distance ``d`` from the terminal set (reverse BFS along the arcs), ordered
  inside a row by the barycenter heuristic (three Sugiyama-style sweeps).
  This reproduces the paper's drawings of its own examples.
* ``"kamada"``: NetworkX ``kamada_kawai_layout`` on the undirected skeleton,
  started from the grid layout, then the terminals are pinned to the bottom
  row (ordered by their Kamada–Kawai abscissa) and the non-terminals are
  relaxed with a few ``spring_layout`` iterations (``fixed=terminals``,
  ``seed``).
* ``"spring"``: ``spring_layout`` with the terminals fixed on the bottom row.
* ``"auto"``: ``"grid"`` when the instance is small and shallow (``n <= 24``
  and at most five rows), else ``"kamada"``.

Coordinates are returned in a box of width :data:`BOX_WIDTH` with ``y = 0``
for the terminal row and ``y > 0`` above it.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence

from glsolver.instance import Instance

# --------------------------------------------------------------------------- style
#: Colour-blind safe terminal palette: Okabe–Ito (without black) followed by five
#: colours of Paul Tol's "muted" scheme.  Terminal ``i`` uses ``PALETTE[i % 12]``
#: and, from the 13th terminal on, an additional hatch pattern (:func:`terminal_hatch`).
PALETTE: tuple[str, ...] = (
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#D55E00",  # vermillion
    "#0072B2",  # blue
    "#44AA99",  # teal
    "#AA4499",  # purple
    "#999933",  # olive
    "#882255",  # wine
    "#88CCEE",  # cyan
)
HATCHES: tuple[str, ...] = ("//", "xx", "..", "\\\\", "++", "oo")
#: SVG-friendly names of the hatch patterns (same order as :data:`HATCHES`).
SVG_HATCH_ANGLES: tuple[int, ...] = (45, 90, 0, 135, 60, 30)

MATCHING_COLOR = "#2b4bd6"  # matching arcs (p_i, t_i): blue, thick
SECONDARY_COLOR = "#d1272e"  # secondary arcs e_i: red, thick, labelled
CYCLE_COLOR = "#1a9e3f"  # chosen cycle of the reassignment graph: green
DELETED_COLOR = "#bdbdbd"  # deleted arcs: light grey, dashed
ARC_COLOR = "#505050"  # ordinary arcs
HIGHLIGHT_COLOR = "#ff7f0e"  # the arc / vertex acted upon in the current event
TREE_ALPHA = 0.9
CUT_COLORS: dict[str, str] = {"L": "#8ecae6", "S": "#ffd166", "R": "#f4978e"}

#: Width of the layout box; heights depend on the number of rows.
BOX_WIDTH = 10.0
ROW_HEIGHT = 1.7
NODE_RADIUS = 0.34


def terminal_color(i: int) -> str:
    """Colour of terminal index ``i`` (cycles through :data:`PALETTE`)."""
    return PALETTE[i % len(PALETTE)]


def terminal_hatch(i: int) -> str | None:
    """Hatch pattern for terminal index ``i`` (``None`` for the first 12)."""
    if i < len(PALETTE):
        return None
    return HATCHES[(i // len(PALETTE) - 1) % len(HATCHES)]


def terminal_svg_hatch_angle(i: int) -> int | None:
    """Angle of the SVG hatch pattern for terminal index ``i`` (``None`` for the first 12)."""
    if i < len(PALETTE):
        return None
    return SVG_HATCH_ANGLES[(i // len(PALETTE) - 1) % len(SVG_HATCH_ANGLES)]


# --------------------------------------------------------------------------- helpers
Layout = dict[int, tuple[float, float]]

METHODS = ("auto", "grid", "layered", "kamada", "spring")


def distance_to_terminals(inst: Instance) -> list[int]:
    """Directed distance of every vertex to the nearest terminal (reverse BFS).

    Terminals have distance 0; vertices that reach no terminal get
    ``max_finite + 1`` so that they land on the top row.
    """
    n = inst.n
    in_adj = inst.in_adjacency()
    dist = [-1] * n
    dq: deque[int] = deque()
    for t in inst.terminals:
        dist[t] = 0
        dq.append(t)
    while dq:
        v = dq.popleft()
        for u in in_adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                dq.append(u)
    top = max(dist) + 1
    return [d if d >= 0 else top for d in dist]


def _rows(inst: Instance) -> list[list[int]]:
    dist = distance_to_terminals(inst)
    rows: list[list[int]] = [[] for _ in range(max(dist) + 1)]
    for v in range(inst.n):
        rows[dist[v]].append(v)
    rows[0] = list(inst.terminals)  # keep the terminal order of the instance
    return [r for r in rows if r]


def _barycenter_order(rows: list[list[int]], inst: Instance, sweeps: int = 3) -> list[list[int]]:
    """Order vertices within each row by the mean position of their neighbours
    in the adjacent row (down-sweep uses the row below, up-sweep the row above).
    Ties are broken by vertex id, so the result is deterministic."""
    out_adj = inst.out_adjacency()
    in_adj = inst.in_adjacency()
    pos: dict[int, float] = {}
    for row in rows:
        for j, v in enumerate(row):
            pos[v] = j
    row_of = {v: i for i, row in enumerate(rows) for v in row}

    def mean_pos(v: int, target_row: int, neigh: Iterable[int]) -> float | None:
        vals = [pos[u] for u in neigh if row_of[u] == target_row]
        return sum(vals) / len(vals) if vals else None

    for s in range(sweeps):
        order = range(1, len(rows)) if s % 2 == 0 else range(len(rows) - 2, 0, -1)
        for i in order:
            adj_row = i - 1 if s % 2 == 0 else i + 1
            if adj_row < 0 or adj_row >= len(rows):
                continue
            keyed = []
            for v in rows[i]:
                m = mean_pos(v, adj_row, list(out_adj[v]) + list(in_adj[v]))
                keyed.append((m if m is not None else pos[v], v))
            rows[i] = [v for _, v in sorted(keyed)]
            for j, v in enumerate(rows[i]):
                pos[v] = j
    return rows


def grid_layout(inst: Instance) -> Layout:
    """Layered ("grid") layout: terminals on the bottom row, one row per
    directed distance to ``T``; see the module docstring."""
    rows = _barycenter_order(_rows(inst), inst)
    layout: Layout = {}
    for i, row in enumerate(rows):
        m = len(row)
        for j, v in enumerate(row):
            x = (j + 0.5) * BOX_WIDTH / m
            layout[v] = (x, i * ROW_HEIGHT)
    return layout


def _skeleton(inst: Instance):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(inst.n))
    G.add_edges_from((u, v) for u, v in inst.arcs)
    return G


def _pin_terminals(inst: Instance, pos: Layout, order_by_x: bool = True) -> Layout:
    """Put the terminals on ``y = 0`` (evenly spaced, ordered by their current
    abscissa) and shift the non-terminals above the row."""
    terms = list(inst.terminals)
    if order_by_x:
        terms.sort(key=lambda t: (pos[t][0], t))
    k = len(terms)
    out: Layout = {}
    for j, t in enumerate(terms):
        out[t] = ((j + 0.5) * BOX_WIDTH / k, 0.0)
    tset = set(terms)
    nts = [v for v in range(inst.n) if v not in tset]
    if not nts:
        return out
    xs = [pos[v][0] for v in nts]
    ys = [pos[v][1] for v in nts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0
    height = max(ROW_HEIGHT, min(3.5 * ROW_HEIGHT, ROW_HEIGHT * span_y / span_x * 2.0))
    for v in nts:
        fx = (pos[v][0] - x0) / span_x
        fy = (pos[v][1] - y0) / span_y
        out[v] = (0.6 + fx * (BOX_WIDTH - 1.2), ROW_HEIGHT + fy * height)
    return out


def kamada_layout(inst: Instance, seed: int = 0) -> Layout:
    """Kamada–Kawai layout of the undirected skeleton with the terminals pinned
    to the bottom row (see the module docstring)."""
    import networkx as nx

    G = _skeleton(inst)
    init = {v: (x, y) for v, (x, y) in grid_layout(inst).items()}
    kk = nx.kamada_kawai_layout(G, pos=init, scale=BOX_WIDTH / 2)
    pinned = _pin_terminals(inst, {v: (float(p[0]), float(p[1])) for v, p in kk.items()})
    if inst.num_nonterminals == 0:
        return pinned
    relaxed = nx.spring_layout(
        G, pos={v: list(p) for v, p in pinned.items()}, fixed=list(inst.terminals),
        seed=seed, iterations=25, k=BOX_WIDTH / math.sqrt(max(2, inst.n)),
    )
    return _pin_terminals(
        inst, {v: (float(p[0]), float(p[1])) for v, p in relaxed.items()}, order_by_x=False
    )


def spring_layout(inst: Instance, seed: int = 0) -> Layout:
    """Fruchterman–Reingold layout with the terminals fixed on the bottom row."""
    import networkx as nx

    G = _skeleton(inst)
    init = grid_layout(inst)
    if inst.num_nonterminals == 0:
        return init
    pos = nx.spring_layout(
        G, pos={v: list(p) for v, p in init.items()}, fixed=list(inst.terminals), seed=seed,
        iterations=60, k=BOX_WIDTH / math.sqrt(max(2, inst.n)),
    )
    return _pin_terminals(inst, {v: (float(p[0]), float(p[1])) for v, p in pos.items()}, order_by_x=False)


def compute_layout(inst: Instance, seed: int = 0, method: str = "auto") -> Layout:
    """Deterministic ``vertex -> (x, y)`` layout (see the module docstring).

    ``method`` is one of ``"auto"``, ``"grid"`` (= ``"layered"``),
    ``"kamada"`` or ``"spring"``.  Terminals always lie on ``y = 0``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown layout method {method!r}; choose from {METHODS}")
    if method == "auto":
        rows = _rows(inst)
        method = "grid" if inst.n <= 24 and len(rows) <= 5 else "kamada"
    if method in ("grid", "layered"):
        return grid_layout(inst)
    if method == "kamada":
        return kamada_layout(inst, seed)
    return spring_layout(inst, seed)


def layout_bounds(layout: Layout, pad: float = 0.9) -> tuple[float, float, float, float]:
    """``(xmin, xmax, ymin, ymax)`` of the layout with a margin of ``pad`` units."""
    xs = [p[0] for p in layout.values()] or [0.0]
    ys = [p[1] for p in layout.values()] or [0.0]
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def node_radius(layout: Layout) -> float:
    """Node radius in layout units: :data:`NODE_RADIUS` shrunk when vertices are close."""
    pts = list(layout.values())
    if len(pts) < 2:
        return NODE_RADIUS
    dmin = min(
        math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1:]
    )
    return max(0.16, min(NODE_RADIUS, 0.38 * dmin))


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain; returns the hull counter-clockwise (2 points → segment)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
