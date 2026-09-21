"""Step-by-step rendering of a solver trace (docs/paper_notes.md §14).

Two layers live here:

1. :func:`replay` turns the ordered event list into one :class:`StepState`
   per event by replaying events ``0..i``: the current arc set (with the
   draw endpoint of arcs redirected by contractions), deleted arcs, live
   vertices and terminals, parts, capacities, the witness ``φ`` (or the split
   witness ``ψ``), essential sets, the matching / secondary arcs /
   criticality table of the running ShiftAssignment call, the reassignment
   graph and the chosen cycle, the potential before/after a shift, tightest
   cuts when the event carries them, and counters.  The replay is driven by
   the trace alone (no solver oracle is imported): the tracer emits an
   ``essential`` event after every recomputation of ``Ess``/``κ`` (arc
   deletion, terminal removal, rounding; paper_notes §14), and the frame of
   the mutation already shows the sets of the ``essential`` event that
   follows it (look-ahead).  A trace without such events (a backend that does
   not emit them) gets its sectors flagged *stale* (drawn faded, "stale" in
   the panel) from the mutation on until the next ``essential`` event.
   The frame of ``delete_arc`` keeps the matching / secondary arcs, the
   criticality table and the reassignment inset of the ShiftAssignment call
   it ends, so the reader can check that the deleted arc is non-critical;
   the next event clears them.

2. :class:`TraceRenderer` draws a state as a Matplotlib figure with the
   paper's conventions (paper Figure 4 and §"Visual conventions"): terminals
   are rounded squares with one colour each; a non-terminal is a circle whose
   interior is split into coloured sectors (its essential terminals) with a
   thick outer ring (its assigned terminal); contracted vertices are filled
   with their part's colour and dotted; deleted arcs stay visible as light
   grey dashes; matching arcs are blue, secondary arcs red and labelled
   ``e_i``; the reassignment graph is an inset with the chosen cycle in green;
   a side panel lists the step, the event, capacities, part sizes, the
   potential and the counters; a tightest min cut shades ``L``/``S``/``R``.

:func:`partition_svg` and :func:`final_figure` draw a finished partition.
"""
from __future__ import annotations

import copy
import html as _html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from glsolver.instance import Instance
from glsolver.viz.layout import (
    ARC_COLOR,
    CUT_COLORS,
    CYCLE_COLOR,
    DELETED_COLOR,
    HIGHLIGHT_COLOR,
    MATCHING_COLOR,
    SECONDARY_COLOR,
    Layout,
    compute_layout,
    convex_hull,
    layout_bounds,
    node_radius,
    terminal_color,
    terminal_hatch,
    terminal_svg_hatch_angle,
)

Arc = tuple[int, int]

MUTATIONS = ("delete_arc", "contract", "remove_terminal", "round_and_remove", "dag_contract")
COUNTER_KEYS = (
    "contractions", "deletions", "cycle_shifts", "terminal_removals", "roundings", "shift_calls",
)


# --------------------------------------------------------------------------- state
@dataclass
class StepState:
    """Full replayed state *after* event ``index`` (see the module docstring)."""

    index: int
    type: str
    event: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    description: str = ""
    arcs: dict[Arc, int] = field(default_factory=dict)  # live arc -> vertex where its head is drawn
    dead_arcs: dict[Arc, str] = field(default_factory=dict)  # (u, draw head) -> reason
    tree_arcs: list[tuple[int, int, int]] = field(default_factory=list)  # (p, parent, terminal)
    live: set[int] = field(default_factory=set)
    terminals: list[int] = field(default_factory=list)
    removed_terminals: list[int] = field(default_factory=list)
    part_of: dict[int, int] = field(default_factory=dict)  # finished non-terminal -> terminal
    parts: dict[int, list[int]] = field(default_factory=dict)
    capacities: dict[int, int] = field(default_factory=dict)
    weights: dict[int, int] | None = None
    phi: dict[int, int] = field(default_factory=dict)
    psi: dict[Arc, int] = field(default_factory=dict)
    ess: dict[int, list[int]] = field(default_factory=dict)
    kappa: dict[int, int] = field(default_factory=dict)
    ess_stale: bool = False  # Ess/κ outdated by a mutation and not (yet) re-emitted by the trace
    ess_recomputed: bool = False  # this ``essential`` event follows a recomputation
    ess_changes: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)  # v -> (Ess before, after)
    kappa_changes: dict[int, tuple[int | None, int]] = field(default_factory=dict)  # v -> (κ before, after)
    matching: list[Arc] = field(default_factory=list)
    secondary: list[Arc] = field(default_factory=list)
    crit: list[tuple[int, int, int]] = field(default_factory=list)
    reassignment: dict[str, Any] | None = None
    changes: list[tuple[int, int, int]] = field(default_factory=list)
    potential_before: int | None = None
    potential_after: int | None = None
    split_cost: int | None = None
    cuts: dict[int, dict[str, list[int]]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    highlight_arcs: list[Arc] = field(default_factory=list)  # draw endpoints
    highlight_vertices: list[int] = field(default_factory=list)
    highlight_terminals: list[int] = field(default_factory=list)

    # ---- derived helpers -------------------------------------------------
    def draw_arc(self, u: int, v: int) -> Arc:
        """Draw endpoints of the *current* arc ``(u, v)`` (head may be a contracted vertex)."""
        return (u, self.arcs.get((u, v), v))

    def ring_of(self, v: int) -> list[tuple[int, float]]:
        """``[(terminal, fraction)]`` for the outer ring of ``v``: the witness
        ``φ(v)`` (one full ring) or the split witness ``ψ(v, ·)`` (proportional)."""
        if v in self.phi:
            return [(self.phi[v], 1.0)]
        units = [(t, u) for (x, t), u in self.psi.items() if x == v and u > 0]
        total = sum(u for _, u in units)
        if not units or total <= 0:
            return []
        return [(t, u / total) for t, u in sorted(units)]

    def xi(self, v: int, t: int) -> int:
        """[Def 6.2] criticality cost ``ξ_v(t)`` from the current table."""
        return len({i for i, x, y in self.crit if x == v and y == t})

    def potential(self, phi: dict[int, int] | None = None) -> int:
        """[Def 6.3] ``Φ(φ) = Σ_v ξ_v(φ(v))`` for the current criticality table."""
        phi = self.phi if phi is None else phi
        return sum(self.xi(v, t) for v, t in phi.items())

    def num_nonterminals(self) -> int:
        tset = set(self.terminals)
        return sum(1 for v in self.live if v not in tset)


def _int_keys(d: Any) -> dict[int, Any]:
    if not isinstance(d, dict):
        return {}
    return {int(k): v for k, v in d.items()}


class _Replayer:
    """Mutable working state; ``snapshot`` freezes it into a :class:`StepState`."""

    def __init__(self, inst: Instance, trace: Sequence[dict[str, Any]]) -> None:
        self.inst = inst
        self.trace = trace
        self.tindex = inst.terminal_index()
        self._before: tuple[dict[int, list[int]], dict[int, int]] | None = None  # Ess/κ before a recomputation
        self.st = StepState(-1, "")
        st = self.st
        st.live = set(range(inst.n))
        st.terminals = list(inst.terminals)
        st.arcs = {(u, v): v for u, v in inst.arcs}
        st.capacities = {t: int(c) for t, c in zip(inst.terminals, inst.capacities)}
        st.parts = {t: [t] for t in inst.terminals}
        if inst.weights is not None:
            st.weights = {v: int(w) for v, w in enumerate(inst.weights) if v not in set(inst.terminals)}
        st.counters = {k: 0 for k in COUNTER_KEYS}
        self.clear_phase = False

    # ---- helpers ---------------------------------------------------------
    def _capacities_from(self, ev: dict[str, Any]) -> None:
        caps = ev.get("capacities")
        if isinstance(caps, dict):
            self.st.capacities = {int(t): int(c) for t, c in caps.items()}
        elif isinstance(caps, list) and len(caps) == len(self.st.terminals):
            self.st.capacities = {t: int(c) for t, c in zip(self.st.terminals, caps)}

    def _weight(self, v: int) -> int:
        return 1 if self.st.weights is None else self.st.weights.get(v, 1)

    def _drop_vertex_arcs(self, v: int, reason: str, keep_tree: int | None = None) -> None:
        st = self.st
        for (a, b), head in list(st.arcs.items()):
            if a == v:
                del st.arcs[(a, b)]
                if keep_tree is not None and head == keep_tree:
                    continue
                st.dead_arcs[(a, head)] = reason
            elif b == v:
                del st.arcs[(a, b)]
                st.dead_arcs[(a, head)] = reason

    def _contract(self, p: int, t: int, parent: int, reason: str) -> None:
        st = self.st
        # redirect incoming arcs (u, p) -> (u, t), merging duplicates [Def 2.1]
        for (u, x), head in list(st.arcs.items()):
            if x != p:
                continue
            del st.arcs[(u, x)]
            if (u, t) in st.arcs:
                st.dead_arcs[(u, head)] = "merged"
            else:
                st.arcs[(u, t)] = head
        # outgoing arcs of p are deleted; (p, parent) becomes an arborescence arc
        tree_done = False
        for (a, x), head in list(st.arcs.items()):
            if a != p:
                continue
            del st.arcs[(a, x)]
            if not tree_done and (head == parent or x == t):
                tree_done = True
                continue
            st.dead_arcs[(a, head)] = reason
        st.tree_arcs.append((p, parent, t))
        st.live.discard(p)
        st.part_of[p] = t
        st.parts.setdefault(t, [t]).append(p)
        st.phi.pop(p, None)
        st.ess.pop(p, None)
        st.kappa.pop(p, None)
        for key in [k for k in st.psi if k[0] == p]:
            del st.psi[key]

    def _remove_terminal(self, t: int, reason: str) -> None:
        st = self.st
        self._drop_vertex_arcs(t, reason)
        st.live.discard(t)
        if t in st.terminals:
            st.terminals.remove(t)
        st.removed_terminals.append(t)
        st.capacities.pop(t, None)
        for key in [k for k in st.psi if k[1] == t]:
            del st.psi[key]

    def _apply_essential(self, ev: dict[str, Any]) -> None:
        st = self.st
        st.ess = {v: sorted(int(t) for t in ts) for v, ts in _int_keys(ev.get("ess")).items()}
        st.kappa = {v: int(k) for v, k in _int_keys(ev.get("kappa")).items()}
        st.ess_stale = False

    def _invalidate_essential(self) -> None:
        """``Ess``/``κ`` are outdated by the current mutation (§13.3).

        The trace is expected to carry the recomputed sets in an ``essential``
        event right after the mutation; when it does, this frame already shows
        them (look-ahead) and the ``essential`` frame highlights what changed.
        Otherwise the sets are flagged stale until the next ``essential`` event.
        """
        st = self.st
        self._before = ({v: list(s) for v, s in st.ess.items()}, dict(st.kappa))
        nxt = self.trace[st.index + 1] if st.index + 1 < len(self.trace) else None
        if isinstance(nxt, dict) and nxt.get("type") == "essential":
            self._apply_essential(nxt)
        else:
            st.ess_stale = True

    # ---- event dispatch ---------------------------------------------------
    def apply(self, i: int, ev: dict[str, Any]) -> StepState:
        st = self.st
        typ = str(ev.get("type", "?"))
        st.index, st.type, st.event = i, typ, ev
        st.changes = []
        st.ess_recomputed, st.ess_changes, st.kappa_changes = False, {}, {}
        st.highlight_arcs, st.highlight_vertices, st.highlight_terminals = [], [], []
        st.potential_before = st.potential_after = None
        st.cuts = {}
        if self.clear_phase:  # the mutation frame kept the ShiftAssignment context; the next event drops it
            st.matching, st.secondary, st.crit, st.reassignment = [], [], [], None
            self.clear_phase = False
        handler = getattr(self, f"_ev_{typ}", None)
        if handler is not None:
            handler(ev)
        if isinstance(ev.get("cuts"), dict):
            st.cuts = {
                int(v): {side: [int(x) for x in c.get(side, [])] for side in ("L", "S", "R")}
                for v, c in ev["cuts"].items()
                if isinstance(c, dict)
            }
        if typ in MUTATIONS:
            self.clear_phase = True
        st.title, st.description = describe_event(self.inst, st)
        return self.snapshot()

    def snapshot(self) -> StepState:
        return copy.deepcopy(self.st)

    def _ev_init(self, ev: dict[str, Any]) -> None:
        st = self.st
        inst = self.inst
        if "n" in ev and int(ev["n"]) != inst.n:
            raise ValueError(f"trace was recorded on a different instance: n={ev['n']} in the trace, {inst.n} in the instance")
        if isinstance(ev.get("terminals"), list) and [int(t) for t in ev["terminals"]] != list(inst.terminals):
            raise ValueError(
                f"trace was recorded on a different instance: terminals {ev['terminals']} in the trace, "
                f"{list(inst.terminals)} in the instance"
            )
        if isinstance(ev.get("arcs"), list):
            st.arcs = {(int(u), int(v)): int(v) for u, v in ev["arcs"]}
        if isinstance(ev.get("terminals"), list):
            st.terminals = [int(t) for t in ev["terminals"]]
            st.parts = {t: [t] for t in st.terminals}
            st.capacities = {t: int(c) for t, c in zip(st.terminals, ev.get("capacities", []))} or st.capacities
        if isinstance(ev.get("weights"), list) and any(w > 1 for w in ev["weights"]):
            tset = set(st.terminals)
            st.weights = {v: int(w) for v, w in enumerate(ev["weights"]) if v not in tset}

    def _ev_essential(self, ev: dict[str, Any]) -> None:
        st = self.st
        before = self._before
        if before is None and st.ess:  # re-emitted without a preceding mutation: compare with the current sets
            before = ({v: list(s) for v, s in st.ess.items()}, dict(st.kappa))
        self._before = None
        self._apply_essential(ev)
        if before is None:  # the initial computation
            return
        b_ess, b_kappa = before
        st.ess_recomputed = True
        st.ess_changes = {v: (b_ess.get(v, []), s) for v, s in st.ess.items() if b_ess.get(v) != s}
        st.kappa_changes = {v: (b_kappa.get(v), k) for v, k in st.kappa.items() if b_kappa.get(v) != k}
        st.highlight_vertices = sorted(set(st.ess_changes) | set(st.kappa_changes))

    def _ev_witness(self, ev: dict[str, Any]) -> None:
        self.st.phi = {v: int(t) for v, t in _int_keys(ev.get("phi")).items()}
        self.st.psi = {}

    def _ev_matching(self, ev: dict[str, Any]) -> None:
        st = self.st
        st.matching = [(int(p), int(t)) for p, t in ev.get("pairs", [])]
        st.secondary = [(int(p), int(q)) for p, q in ev.get("secondary", [])]
        st.crit, st.reassignment = [], None
        st.counters["shift_calls"] += 1
        st.highlight_arcs = [st.draw_arc(p, t) for p, t in st.matching]

    def _ev_criticality(self, ev: dict[str, Any]) -> None:
        self.st.crit = [(int(i), int(v), int(t)) for i, v, t in ev.get("crit", [])]

    def _ev_reassignment_graph(self, ev: dict[str, Any]) -> None:
        self.st.reassignment = {
            "arcs": [(int(a), int(b), int(v)) for a, b, v in ev.get("arcs", [])],
            "cycle": [int(t) for t in ev.get("cycle", [])],
        }

    def _ev_cycle_shift(self, ev: dict[str, Any]) -> None:
        st = self.st
        st.changes = [(int(v), int(a), int(b)) for v, a, b in ev.get("changes", [])]
        before = st.potential() if st.crit else None
        for v, _old, new in st.changes:
            st.phi[v] = new
        if isinstance(ev.get("phi"), dict):
            st.phi = {v: int(t) for v, t in _int_keys(ev["phi"]).items()}
        after = st.potential() if st.crit else None
        st.potential_before = ev.get("potential_before") if ev.get("potential_before") is not None else before
        st.potential_after = ev.get("potential_after") if ev.get("potential_after") is not None else after
        st.counters["cycle_shifts"] += 1
        st.highlight_vertices = [v for v, _, _ in st.changes]

    def _ev_delete_arc(self, ev: dict[str, Any]) -> None:
        st = self.st
        u, v = int(ev["u"]), int(ev["v"])
        head = st.arcs.pop((u, v), v)
        st.dead_arcs[(u, head)] = "deleted"
        st.highlight_arcs = [(u, head)]
        st.counters["deletions"] += 1
        self._invalidate_essential()

    def _ev_contract(self, ev: dict[str, Any]) -> None:
        st = self.st
        p, t = int(ev["p"]), int(ev["t"])
        parent = int(ev.get("parent", t))
        w = self._weight(p)
        self._contract(p, t, parent, "contracted")
        if "capacities" in ev:
            self._capacities_from(ev)
        elif t in st.capacities:
            st.capacities[t] -= w
        st.counters["contractions"] += 1
        st.highlight_vertices = [p]
        st.highlight_terminals = [t]

    def _ev_dag_contract(self, ev: dict[str, Any]) -> None:
        st = self.st
        p, t = int(ev["p"]), int(ev["t"])
        parent = int(ev.get("parent", t))
        w = self._weight(p)
        self._contract(p, t, parent, "contracted")
        if ev.get("residual") is not None:
            st.capacities[t] = int(ev["residual"])
        elif t in st.capacities:
            st.capacities[t] -= w
        st.counters["contractions"] += 1
        st.highlight_vertices = [p]
        st.highlight_terminals = [t]

    def _ev_remove_terminal(self, ev: dict[str, Any]) -> None:
        st = self.st
        t = int(ev["t"])
        self._remove_terminal(t, "terminal_removed")
        if "capacities" in ev:
            self._capacities_from(ev)
        st.counters["terminal_removals"] += 1
        st.highlight_terminals = [t]
        self._invalidate_essential()

    def _ev_round_and_remove(self, ev: dict[str, Any]) -> None:
        st = self.st
        S = [int(t) for t in ev.get("S", [])]
        pairs = [(int(t), int(p)) for t, p in ev.get("pairs", [])]
        for t, p in pairs:
            head = st.arcs.get((p, t), t)
            st.tree_arcs.append((p, head, t))
            self._drop_vertex_arcs(p, "rounded", keep_tree=head)
            st.live.discard(p)
            st.part_of[p] = t
            st.parts.setdefault(t, [t]).append(p)
            st.ess.pop(p, None)
            st.kappa.pop(p, None)
            for key in [k for k in st.psi if k[0] == p]:
                del st.psi[key]
        for t in S:
            self._remove_terminal(t, "rounded")
        st.counters["roundings"] += 1
        st.highlight_vertices = [p for _, p in pairs]
        st.highlight_terminals = list(S)
        self._invalidate_essential()

    def _ev_min_cost_split(self, ev: dict[str, Any]) -> None:
        st = self.st
        st.psi = {(int(v), int(t)): int(u) for v, t, u in ev.get("psi", [])}
        st.phi = {}
        st.split_cost = ev.get("cost")
        st.potential_after = ev.get("cost")

    def _ev_done(self, ev: dict[str, Any]) -> None:
        st = self.st
        parts = _int_keys(ev.get("parts"))
        if parts:
            st.parts = {t: [int(v) for v in vs] for t, vs in parts.items()}
            for t, vs in st.parts.items():
                for v in vs:
                    if v != t:
                        st.part_of[v] = t


def replay(inst: Instance, trace: Sequence[dict[str, Any]]) -> list[StepState]:
    """Replay ``trace`` on ``inst`` and return the state after every event.

    Purely event-driven (see the module docstring); raises ``ValueError`` for
    an empty trace or one whose ``init`` event belongs to another instance.
    """
    if not trace:
        raise ValueError("empty trace: run the solver with trace=True (reference backends emit traces)")
    rp = _Replayer(inst, list(trace))
    return [rp.apply(i, ev) for i, ev in enumerate(trace)]


def cycle_arcs(reassignment: dict[str, Any] | None) -> list[tuple[int, int]]:
    """Arcs ``(a, b)`` of the chosen cycle of the reassignment graph, in arc direction.

    Every terminal has in-degree exactly one in ``R`` [§13.5], so the arcs of
    ``R`` with both endpoints on the cycle are exactly the cycle's arcs; this
    does not depend on whether the trace lists ``cycle`` in walk (backward)
    or forward order.  The list starts at ``cycle[0]`` and follows the arcs.
    """
    if not reassignment:
        return []
    cyc = [int(t) for t in reassignment.get("cycle", [])]
    cycset = set(cyc)
    out = {int(a): int(b) for a, b, _v in reassignment.get("arcs", []) if int(a) in cycset and int(b) in cycset}
    if not out:
        return []
    cur = cyc[0] if cyc and cyc[0] in out else next(iter(out))
    order: list[tuple[int, int]] = []
    seen: set[int] = set()
    while cur in out and cur not in seen:
        seen.add(cur)
        order.append((cur, out[cur]))
        cur = out[cur]
    return order


# --------------------------------------------------------------------------- labels
def terminal_position(inst: Instance, t: int, tidx: dict[int, int] | None = None) -> int:
    """Position of terminal ``t`` in ``inst.terminals`` (its colour index).

    A trace that names a vertex which is not a terminal of ``inst`` (a
    malformed or foreign trace) raises ``ValueError`` instead of silently
    getting the first terminal's colour.
    """
    tidx = inst.terminal_index() if tidx is None else tidx
    try:
        return tidx[t]
    except KeyError:
        raise ValueError(
            f"trace refers to {t} as a terminal but the terminals of {inst.name or 'the instance'} are {list(inst.terminals)}"
        ) from None


def terminal_label(inst: Instance, t: int) -> str:
    """``t1, t2, ...`` in the order of ``inst.terminals`` (paper numbering)."""
    idx = inst.terminal_index().get(t)
    return f"t{idx + 1}" if idx is not None else f"t?{t}"


def vertex_label(inst: Instance, v: int, labels: dict[int, str] | None = None) -> str:
    if labels and v in labels:
        return labels[v]
    if v in inst.terminals:
        return terminal_label(inst, v)
    return str(v)


def _arc_str(inst: Instance, u: int, v: int) -> str:
    return f"({vertex_label(inst, u)}→{vertex_label(inst, v)})"


def describe_event(inst: Instance, st: StepState) -> tuple[str, str]:
    """``(title, description)`` of the event of ``st`` with the paper's labels."""
    ev = st.event
    T = lambda t: terminal_label(inst, int(t))  # noqa: E731
    V = lambda v: vertex_label(inst, int(v))  # noqa: E731
    typ = st.type
    if typ == "init":
        caps = ",".join(str(st.capacities[t]) for t in st.terminals)
        kind = "weighted " if st.weights else ""
        return "input", (
            f"{kind}instance: n={ev.get('n', inst.n)}, k={ev.get('k', inst.k)}, "
            f"{len(st.arcs)} arcs, capacities c=({caps})"
        )
    if typ == "essential":
        if not st.ess_recomputed:
            return "essential terminals", "Ess(v) and κ(v) for every non-terminal: one max-flow each [Prop 4.2, Lem 4.1]"
        parts = [
            f"Ess({V(v)}) {{{','.join(T(t) for t in a)}}}→{{{','.join(T(t) for t in b)}}}" for v, (a, b) in sorted(st.ess_changes.items())
        ] + [f"κ({V(v)}) {a if a is not None else '?'}→{b}" for v, (a, b) in sorted(st.kappa_changes.items())]
        detail = "; ".join(parts) if parts else "unchanged for every live non-terminal"
        return "essential terminals recomputed", f"Ess and κ recomputed after the mutation [§13.3]: {detail}"
    if typ == "witness":
        return "initial witness", "witness φ of the Flow-Essential Assignment Condition [Def 5.1]"
    if typ == "matching":
        pairs = ", ".join(f"({V(p)},{T(t)})" for p, t in st.matching)
        sec = ", ".join(f"e{i + 1}={_arc_str(inst, p, q)}" for i, (p, q) in enumerate(st.secondary))
        return "matching + secondary arcs", f"ShiftAssignment: matching M={{{pairs}}} (blue) [Lem 7.8]; secondary arcs {sec} (red)"
    if typ == "criticality":
        n = len(st.crit)
        return "criticality table", f"{n} critical triple{'s' if n != 1 else ''} (e_i, v, t) [Def 6.1]; ξ_v(t) = #critical arcs"
    if typ == "reassignment_graph":
        arcs = cycle_arcs(st.reassignment)
        cyc = [a for a, _ in arcs] + [arcs[0][0]] if arcs else []
        return "reassignment graph", "R on T: arc φ(v_i)→t_i labelled v_i; cycle " + "→".join(T(t) for t in cyc) + " (green) [§13.5]"
    if typ == "cycle_shift":
        ch = ", ".join(f"{V(v)}: {T(a)}→{T(b)}" for v, a, b in st.changes)
        pot = ""
        if st.potential_before is not None and st.potential_after is not None:
            pot = f"; Φ: {st.potential_before} → {st.potential_after}"
        return "cycle shift", f"reassign {ch}{pot} [Lem 7.11]"
    if typ == "delete_arc":
        u, v = int(ev["u"]), int(ev["v"])
        idx = next((i for i, e in enumerate(st.secondary) if e == (u, v)), None)
        name = f"e{idx + 1} = {_arc_str(inst, u, v)}" if idx is not None else _arc_str(inst, u, v)
        if st.psi and not st.phi:
            why, ref = "critical for no pair (v,t) with ψ(v,t) > 0", "[Lem 8.4]"
        else:
            why, ref = "critical for no assignment pair (v, φ(v))", "[Lem 7.5]"
        return "delete non-critical arc", f"(iii) delete {name}: {why} {ref}; the call's matching/secondary arcs stay shown"
    if typ == "contract":
        p, t = int(ev["p"]), int(ev["t"])
        w = st.weights.get(p, 1) if st.weights else 1
        return "contract", f"(ii) contract {V(p)} into {T(t)} (parent {V(ev.get('parent', t))}); c_{T(t)} -= {w} → {st.capacities.get(t, '?')} [Lem 7.4]"
    if typ == "dag_contract":
        p, t = int(ev["p"]), int(ev["t"])
        return "DAG contraction", f"contract {V(p)} into {T(t)} (parent {V(ev.get('parent', t))}); residual c_{T(t)} = {ev.get('residual', '?')} [Alg 5]"
    if typ == "remove_terminal":
        t = int(ev["t"])
        return "remove terminal", f"(i) remove {T(t)}: capacity 0, part finished as {{{T(t)}}}; Ess recomputed [Lem 7.2, §13.3]"
    if typ == "round_and_remove":
        S = ", ".join(T(t) for t in ev.get("S", []))
        pairs = ", ".join(f"{T(t)}←{V(p)}" for t, p in ev.get("pairs", []))
        return "round and remove", f"(iv) minimal Hall-deficient S={{{S}}}, t_S={T(ev.get('t_S', -1))}; parts {pairs}; S ∪ PT(G,S) removed [Alg 4]"
    if typ == "min_cost_split":
        return "min-cost split assignment", f"split witness ψ by min-cost flow, potential Φ̄ = {ev.get('cost', '?')} [Prop 5.4]"
    if typ == "done":
        sizes = ", ".join(f"{T(t)}:{len(vs)}" for t, vs in st.parts.items())
        return "done", f"partition complete; part sizes {sizes}"
    return typ, ", ".join(f"{k}={v}" for k, v in ev.items() if k != "type")[:160]


# --------------------------------------------------------------------------- geometry
def bezier_control(p0: tuple[float, float], p1: tuple[float, float], rad: float) -> tuple[float, float]:
    """Control point of Matplotlib's ``arc3,rad=`` connection (same formula as SVG paths)."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return (mx + rad * dy, my - rad * dx)


def bezier_point(p0, c, p1, s: float = 0.5) -> tuple[float, float]:
    return (
        (1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * c[0] + s**2 * p1[0],
        (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * c[1] + s**2 * p1[1],
    )


def _unit(a, b) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def arc_geometry(
    p0: tuple[float, float], p1: tuple[float, float], curved: bool, r0: float, r1: float, rad: float = 0.18
) -> tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float], tuple[float, float]]:
    """``(start, control, end, label_point)`` of an arc between two nodes of
    radii ``r0``/``r1``, shortened to the node boundaries."""
    if curved:
        c = bezier_control(p0, p1, rad)
        u0 = _unit(p0, c)
        u1 = _unit(p1, c)
        start = (p0[0] + u0[0] * r0, p0[1] + u0[1] * r0)
        end = (p1[0] + u1[0] * r1, p1[1] + u1[1] * r1)
        c2 = bezier_control(start, end, rad)
        return start, c2, end, bezier_point(start, c2, end)
    u = _unit(p0, p1)
    start = (p0[0] + u[0] * r0, p0[1] + u[1] * r0)
    end = (p1[0] - u[0] * r1, p1[1] - u[1] * r1)
    return start, None, end, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")



def _fit_limits(ax, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[float, float]:
    """Set equal-aspect limits that fill the axes box exactly (no datalim
    juggling at draw time, so point conversions stay exact). Returns the x limits."""
    fig = ax.figure
    bbox = ax.get_position()
    box_aspect = (bbox.width * fig.get_figwidth()) / (bbox.height * fig.get_figheight())
    xr, yr = xmax - xmin, ymax - ymin
    if xr / yr < box_aspect:  # too narrow: widen x
        extra = yr * box_aspect - xr
        xmin, xmax = xmin - extra / 2, xmax + extra / 2
    else:  # too flat: heighten y
        extra = xr / box_aspect - yr
        ymin, ymax = ymin - extra / 2, ymax + extra / 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    return xmin, xmax

# --------------------------------------------------------------------------- matplotlib output
def save_figure(fig: Any, path: str | Path, *, dpi: int | None = None) -> Path:
    """``fig.savefig`` with byte-deterministic output.

    Matplotlib stamps SVG files with the current date (``<dc:date>``) and
    salts the ids of clip paths / markers with a random UUID unless
    ``rcParams['svg.hashsalt']`` is set; PDF files carry ``CreationDate``.
    Both are suppressed here so that two renders of the same state are
    byte-identical (PNG output has no such fields).
    """
    import matplotlib

    path = Path(path)
    suffix = path.suffix.lower()
    metadata: dict[str, Any] | None = None
    if suffix == ".svg":
        metadata = {"Date": None}
    elif suffix == ".pdf":
        metadata = {"CreationDate": None}
    with matplotlib.rc_context({"svg.hashsalt": "glsolver"}):
        fig.savefig(str(path), dpi=dpi, facecolor="white", metadata=metadata)
    return path


# --------------------------------------------------------------------------- matplotlib renderer
class TraceRenderer:
    """Render the frames of a trace (see the module docstring).

    ``inst`` may also be a :class:`glsolver.api.GLResult` (its instance and
    trace are used); ``trace`` may be a ``GLResult`` or an event list.
    ``layout`` defaults to :func:`compute_layout` with ``layout_method``.
    ``labels`` optionally overrides vertex labels.
    """

    def __init__(
        self,
        inst: Any,
        trace: Any = None,
        layout: Layout | None = None,
        *,
        labels: dict[int, str] | None = None,
        layout_method: str = "auto",
        seed: int = 0,
        figsize: tuple[float, float] = (12.0, 7.0),
        dpi: int = 100,
    ) -> None:
        self.inst, events = _coerce(inst, trace)
        self.trace = events
        self.layout = layout or compute_layout(self.inst, seed=seed, method=layout_method)
        self.labels = labels or {}
        self.figsize, self.dpi = figsize, dpi
        self.states = replay(self.inst, events)
        self.radius = node_radius(self.layout)
        self._orig = set(self.inst.arcs)
        self._tidx = self.inst.terminal_index()

    def __len__(self) -> int:
        return len(self.states)

    def state(self, i: int) -> StepState:
        return self.states[i]

    # ---- public API ---------------------------------------------------------
    def frame(self, i: int, cut_vertex: int | None = None):
        """Matplotlib ``Figure`` of the state after event ``i``."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        st = self.states[i]
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.01, 0.10, 0.66, 0.84])
        ax_leg = fig.add_axes([0.01, 0.005, 0.66, 0.085])
        ax_panel = fig.add_axes([0.685, 0.36, 0.305, 0.63])
        self._draw_graph(ax, st, cut_vertex)
        self._draw_legend(ax_leg)
        self._draw_panel(ax_panel, st)
        if st.reassignment:
            ax_r = fig.add_axes([0.70, 0.02, 0.28, 0.32])
            self._draw_reassignment(ax_r, st)
        name = self.inst.name or "instance"
        ax.set_title(f"{name} — step {i}/{len(self.states) - 1}: {st.title}", fontsize=11, loc="left")
        return fig

    def save_frame(self, i: int, path: str | Path, cut_vertex: int | None = None) -> Path:
        """Write frame ``i`` as ``.svg``/``.png``/``.pdf`` (by extension); the bytes are deterministic."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(self.frame(i, cut_vertex), path, dpi=self.dpi)
        return path

    def render_all(self, directory: str | Path, fmt: str = "svg", prefix: str = "step") -> list[Path]:
        """Write every frame as ``<directory>/<prefix>_<i:03d>.<fmt>``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        return [self.save_frame(i, d / f"{prefix}_{i:03d}.{fmt}") for i in range(len(self.states))]

    def frame_image(self, i: int):
        """Frame ``i`` as a ``PIL.Image`` (RGB) — used by the animation."""
        import numpy as np
        from PIL import Image

        fig = self.frame(i)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return Image.fromarray(buf[:, :, :3].copy())

    # ---- drawing ------------------------------------------------------------
    def _label(self, v: int) -> str:
        return vertex_label(self.inst, v, self.labels)

    def _color(self, t: int) -> str:
        return terminal_color(terminal_position(self.inst, t, self._tidx))

    def _hatch(self, t: int) -> str | None:
        return terminal_hatch(terminal_position(self.inst, t, self._tidx))

    def _pts_per_unit(self, ax) -> float:
        fig = ax.figure
        bbox = ax.get_position()
        width_in = bbox.width * fig.get_figwidth()
        x0, x1 = ax.get_xlim()
        return width_in * 72.0 / (x1 - x0)

    def _setup_axes(self, ax) -> None:
        xmin, xmax, ymin, ymax = layout_bounds(self.layout)
        ymin -= 0.2
        _fit_limits(ax, xmin, xmax, ymin, ymax)
        ax.axis("off")

    def _draw_arrow(self, ax, a, b, *, color, lw, ls="-", curved=False, alpha=1.0, zorder=2, head=True, r_a=None, r_b=None):
        from matplotlib.patches import FancyArrowPatch

        r = self.radius
        ppu = self._pts_per_unit(ax)
        r_a = r if r_a is None else r_a
        r_b = r if r_b is None else r_b
        style = "-|>,head_length=5,head_width=2.5" if head else "-"
        patch = FancyArrowPatch(
            a, b, arrowstyle=style, connectionstyle=f"arc3,rad={0.18 if curved else 0.0}",
            shrinkA=r_a * ppu, shrinkB=r_b * ppu + (1.5 if head else 0.0), color=color, lw=lw,
            linestyle=ls, alpha=alpha, zorder=zorder, mutation_scale=1.6,
        )
        ax.add_patch(patch)

    def _is_curved(self, u: int, v: int) -> bool:
        return (v, u) in self._orig

    def _draw_graph(self, ax, st: StepState, cut_vertex: int | None) -> None:
        from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

        self._setup_axes(ax)
        pos = self.layout
        r = self.radius
        inst = self.inst
        tidx = inst.terminal_index()

        # --- cut shading (L/S/R hulls)
        cut = self._pick_cut(st, cut_vertex)
        if cut:
            for side in ("L", "S", "R"):
                pts = [pos[v] for v in cut.get(side, []) if v in pos]
                if not pts:
                    continue
                hull = convex_hull(pts)
                if len(hull) == 1:
                    ax.add_patch(Circle(hull[0], r * 1.9, facecolor=CUT_COLORS[side], edgecolor=CUT_COLORS[side], alpha=0.35, zorder=0))
                elif len(hull) == 2:
                    ax.plot([hull[0][0], hull[1][0]], [hull[0][1], hull[1][1]], color=CUT_COLORS[side], lw=3.8 * r * self._pts_per_unit(ax), alpha=0.35, solid_capstyle="round", zorder=0)
                else:
                    ax.add_patch(Polygon(hull, closed=True, facecolor=CUT_COLORS[side], edgecolor=CUT_COLORS[side], lw=3.8 * r * self._pts_per_unit(ax), alpha=0.35, joinstyle="round", zorder=0))
                cx = sum(p[0] for p in pts) / len(pts)
                cy = max(p[1] for p in pts) + r * 1.6
                ax.text(cx, cy, side, fontsize=9, ha="center", va="bottom", color="#555555", zorder=0.5)

        # --- arcs
        matching_draw = {st.draw_arc(p, t) for p, t in st.matching}
        secondary_draw = {st.draw_arc(p, q): i for i, (p, q) in enumerate(st.secondary)}
        highlight = set(st.highlight_arcs)
        tree = {(p, par): t for p, par, t in st.tree_arcs}
        for (u, head), reason in st.dead_arcs.items():
            if u == head or (u, head) in tree:
                continue
            is_now = (u, head) in highlight and st.type == "delete_arc"
            both_dead = u not in st.live and head not in st.live
            self._draw_arrow(
                ax, pos[u], pos[head], color=HIGHLIGHT_COLOR if is_now else DELETED_COLOR,
                lw=2.4 if is_now else 0.9, ls="--", curved=self._is_curved(u, head),
                alpha=0.95 if is_now else (0.45 if both_dead else 0.8), zorder=1.5 if is_now else 1,
            )
            if is_now:
                geo = arc_geometry(pos[u], pos[head], self._is_curved(u, head), r, r)
                ax.text(geo[3][0], geo[3][1], "✕", color=HIGHLIGHT_COLOR, fontsize=13, ha="center", va="center", fontweight="bold", zorder=6)
        for (u, v), head in st.arcs.items():
            if u == head:
                continue
            key = (u, head)
            curved = self._is_curved(u, head)
            if key in secondary_draw:
                self._draw_arrow(ax, pos[u], pos[head], color=SECONDARY_COLOR, lw=2.6, curved=curved, zorder=3)
                geo = arc_geometry(pos[u], pos[head], curved, r, r)
                ax.text(geo[3][0], geo[3][1], f"$e_{{{secondary_draw[key] + 1}}}$", color=SECONDARY_COLOR, fontsize=9.5, ha="center", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            elif key in matching_draw:
                self._draw_arrow(ax, pos[u], pos[head], color=MATCHING_COLOR, lw=2.6, curved=curved, zorder=3)
            else:
                self._draw_arrow(ax, pos[u], pos[head], color=ARC_COLOR, lw=1.0, curved=curved, zorder=2)
        for p, par, t in st.tree_arcs:
            col = self._color(t)
            now = st.type in ("contract", "dag_contract", "round_and_remove") and p in st.highlight_vertices
            self._draw_arrow(ax, pos[p], pos[par], color=col, lw=4.0 if now else 3.0, curved=self._is_curved(p, par), alpha=0.9, zorder=2.5, r_a=r * 0.7)

        # --- vertices
        for v in range(inst.n):
            x, y = pos[v]
            is_t = v in tidx
            if is_t:
                col = self._color(v)
                removed = v not in st.live
                size = 2.1 * r
                if v in st.highlight_terminals:
                    ax.add_patch(Circle((x, y), r * 1.9, facecolor=col, edgecolor="none", alpha=0.25, zorder=3))
                ax.add_patch(FancyBboxPatch(
                    (x - size / 2, y - size / 2), size, size, boxstyle="round,pad=0,rounding_size=0.12",
                    facecolor=col, edgecolor="#222222", lw=1.2 if not removed else 1.0,
                    linestyle="--" if removed else "-", alpha=0.45 if removed else 1.0, hatch=self._hatch(v), zorder=4,
                ))
                ax.text(x, y, self._label(v), fontsize=9, fontweight="bold", ha="center", va="center", zorder=5, color="black")
                continue
            if v in st.part_of:  # contracted / rounded: solid fill with the part colour, dotted outline
                col = self._color(st.part_of[v])
                now = v in st.highlight_vertices
                if now:
                    ax.add_patch(Circle((x, y), r * 1.6, facecolor=col, edgecolor="none", alpha=0.3, zorder=3))
                ax.add_patch(Circle((x, y), r * 0.72, facecolor=col, edgecolor="#333333", lw=1.2, linestyle=":", hatch=self._hatch(st.part_of[v]), zorder=4))
                ax.text(x, y, self._label(v), fontsize=7.5, ha="center", va="center", zorder=5, color="black")
                continue
            if v not in st.live:  # vertex vanished without a part (should not happen)
                ax.add_patch(Circle((x, y), r * 0.6, facecolor="white", edgecolor="#999999", linestyle=":", zorder=4))
                continue
            if v in st.highlight_vertices:
                glow = self._color(st.phi[v]) if v in st.phi else HIGHLIGHT_COLOR
                ax.add_patch(Circle((x, y), r * 1.75, facecolor=glow, edgecolor="none", alpha=0.25, zorder=3))
            ess = st.ess.get(v, [])
            ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="#333333", lw=1.0, zorder=4))
            if ess:
                n = len(ess)
                for j, t in enumerate(ess):
                    th1 = 90 - 360 * (j + 1) / n
                    th2 = 90 - 360 * j / n
                    ax.add_patch(Wedge((x, y), r, th1, th2, facecolor=self._color(t), edgecolor="white", lw=0.6, hatch=self._hatch(t), alpha=0.55 if st.ess_stale else 1.0, zorder=4.2))
            ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor="#333333", lw=1.0, zorder=4.4))
            # outer ring: witness phi (full) or split witness psi (proportional)
            ring = st.ring_of(v)
            if ring:
                start = 90.0
                for t, frac in ring:
                    span = 360.0 * frac
                    ax.add_patch(Wedge((x, y), r * 1.32, start - span, start, width=r * 0.22, facecolor=self._color(t), edgecolor="none", zorder=4.5))
                    start -= span
            old = [a for vv, a, _ in st.changes if vv == v]
            if old:  # cycle shift: the previous ring colour lingers as a thin dashed outer ring
                ax.add_patch(Circle((x, y), r * 1.5, facecolor="none", edgecolor=self._color(old[0]), lw=1.4, linestyle="--", zorder=4.5))
            ax.text(x, y, self._label(v), fontsize=8, fontweight="bold", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none", alpha=0.85))
            if st.weights and v in st.weights:
                ax.text(x + r * 1.15, y - r * 1.15, f"w={st.weights[v]}", fontsize=6.5, ha="left", va="top", color="#444444", zorder=5)

    def _pick_cut(self, st: StepState, cut_vertex: int | None) -> dict[str, list[int]] | None:
        if not st.cuts:
            return None
        if cut_vertex is not None and cut_vertex in st.cuts:
            return st.cuts[cut_vertex]
        for key in ("v", "p", "u"):
            if key in st.event and int(st.event[key]) in st.cuts:
                return st.cuts[int(st.event[key])]
        return st.cuts[min(st.cuts)]

    def _draw_legend(self, ax) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyBboxPatch, Wedge

        ax.axis("off")
        ax.set_xlim(0, 112)
        ax.set_ylim(0, 10)
        ax.add_patch(FancyBboxPatch((1.5, 3), 3.4, 4, boxstyle="round,pad=0,rounding_size=0.5", facecolor=terminal_color(0), edgecolor="#222", lw=0.8))
        ax.text(6, 5, "terminal", fontsize=7.5, va="center")
        ax.add_patch(Circle((16, 5), 2.2, facecolor="white", edgecolor="#333", lw=0.8))
        ax.add_patch(Wedge((16, 5), 2.2, 90, 270, facecolor=terminal_color(1), edgecolor="white", lw=0.4))
        ax.add_patch(Wedge((16, 5), 2.2, -90, 90, facecolor=terminal_color(2), edgecolor="white", lw=0.4))
        ax.add_patch(Circle((16, 5), 2.9, facecolor="none", edgecolor=terminal_color(2), lw=2.2))
        ax.text(20, 5, "sectors = Ess(v), ring = φ(v)", fontsize=7.5, va="center")
        ax.add_patch(Circle((45, 5), 1.6, facecolor=terminal_color(0), edgecolor="#333", lw=0.8, linestyle=":"))
        ax.text(47.5, 5, "contracted", fontsize=7.5, va="center")
        items = [
            (MATCHING_COLOR, "-", 2.2, "matching"), (SECONDARY_COLOR, "-", 2.2, "secondary e_i"),
            (DELETED_COLOR, "--", 1.0, "deleted"), (terminal_color(1), "-", 3.0, "arborescence"),
        ]
        x = 58
        for col, ls, lw, text in items:
            ax.add_line(Line2D([x, x + 4], [5, 5], color=col, lw=lw, linestyle=ls))
            ax.text(x + 4.8, 5, text, fontsize=7.5, va="center")
            x += 5.5 + 1.15 * len(text)

    def _draw_panel(self, ax, st: StepState) -> None:
        from matplotlib.patches import Rectangle

        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        inst = self.inst
        y = 0.985
        dy = 0.043

        def put(text: str, *, bold=False, size=8.2, color="black", indent=0.0) -> float:
            nonlocal y
            ax.text(0.0 + indent, y, text, fontsize=size, fontweight="bold" if bold else "normal", va="top", ha="left", color=color, family="monospace")
            y -= dy * (size / 8.2)
            return y

        put(f"step {st.index} / {len(self.states) - 1}   event: {st.type}", bold=True, size=9)
        for chunk in _wrap(st.description, 44):
            put(chunk, size=7.6, color="#222222")
        y -= dy * 0.3
        put("terminal   id  cap  part", bold=True)
        for t in list(inst.terminals):
            live = t in st.terminals
            col = self._color(t)
            ax.add_patch(Rectangle((0.0, y - 0.032), 0.035, 0.03, facecolor=col, edgecolor="#222", lw=0.6, alpha=1.0 if live else 0.45, hatch=self._hatch(t)))
            cap = st.capacities.get(t, "-") if live else "done"
            size = len(st.parts.get(t, [t]))
            mark = "" if live else "  (removed)"
            put(f"    {terminal_label(inst, t):<5}{t:>4}{str(cap):>5}{size:>6}{mark}", indent=0.02, color="black" if live else "#777777")
        y -= dy * 0.3
        put(f"non-terminals left: {st.num_nonterminals()}   live arcs: {len(st.arcs)}")
        if st.phi:
            items = ", ".join(f"{self._label(v)}→{terminal_label(inst, t)}" for v, t in sorted(st.phi.items()))
            for j, chunk in enumerate(_wrap("φ: " + items, 44)[:4]):
                put(chunk, size=7.6)
        elif st.psi:
            items = ", ".join(f"{self._label(v)}→{terminal_label(inst, t)}×{u}" for (v, t), u in sorted(st.psi.items()))
            for chunk in _wrap("ψ: " + items, 44)[:4]:
                put(chunk, size=7.6)
        if st.potential_before is not None or st.potential_after is not None:
            b = "?" if st.potential_before is None else st.potential_before
            a = "?" if st.potential_after is None else st.potential_after
            label = "Φ̄ (split potential)" if st.psi and not st.phi else "potential Φ"
            put(f"{label}: {b} → {a}" if st.potential_before is not None else f"{label}: {a}", bold=True)
        elif st.crit and st.phi:
            put(f"potential Φ(φ) = {st.potential()}", bold=True)
        if st.ess_stale:
            put("(essential sets not recomputed: stale)", color="#a00000", size=7.4)
        y -= dy * 0.3
        c = st.counters
        put(f"contractions {c.get('contractions', 0)} · deletions {c.get('deletions', 0)} · cycle shifts {c.get('cycle_shifts', 0)}", size=7.6)
        put(f"terminal removals {c.get('terminal_removals', 0)} · roundings {c.get('roundings', 0)} · shift calls {c.get('shift_calls', 0)}", size=7.6)
        if st.cuts:
            v0 = min(st.cuts)
            cut = st.cuts[v0]
            put(f"tightest cut of {self._label(v0)}: |S|={len(cut['S'])}  L={len(cut['L'])}  R={len(cut['R'])}", size=7.6)

    def _draw_reassignment(self, ax, st: StepState) -> None:
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        ax.axis("off")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.35, 1.45)
        ax.set_aspect("equal")
        rg = st.reassignment or {"arcs": [], "cycle": []}
        terms = list(st.terminals)
        k = max(1, len(terms))
        posr = {t: (math.cos(math.pi / 2 + 2 * math.pi * j / k), math.sin(math.pi / 2 + 2 * math.pi * j / k)) for j, t in enumerate(terms)}
        cyc_arcs = set(cycle_arcs(rg))
        ax.text(0, 1.38, "reassignment graph R (cycle in green)", fontsize=8, ha="center", va="center")
        for a, b, v in rg["arcs"]:
            if a not in posr or b not in posr:
                continue
            on = (a, b) in cyc_arcs
            pa, pb = posr[a], posr[b]
            patch = FancyArrowPatch(pa, pb, arrowstyle="-|>,head_length=5,head_width=2.5", connectionstyle="arc3,rad=0.25",
                                    shrinkA=11, shrinkB=12, color=CYCLE_COLOR if on else "#666666", lw=2.6 if on else 1.2, mutation_scale=1.6, zorder=2)
            ax.add_patch(patch)
            c = bezier_control(pa, pb, 0.25)
            m = bezier_point(pa, c, pb)
            ax.text(m[0], m[1], self._label(v), fontsize=7.5, ha="center", va="center", color=CYCLE_COLOR if on else "#444444",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=3)
        for t, (x, yv) in posr.items():
            s = 0.3
            ax.add_patch(FancyBboxPatch((x - s / 2, yv - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.05", facecolor=self._color(t), edgecolor="#222", lw=1.0, hatch=self._hatch(t), zorder=4))
            ax.text(x, yv, terminal_label(self.inst, t), fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split(" ")
    out: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out or [""]


def _coerce(inst: Any, trace: Any) -> tuple[Instance, list[dict[str, Any]]]:
    """Accept ``(Instance, events)``, ``(GLResult, None)`` or ``(Instance, GLResult)``."""
    events = trace
    if not isinstance(inst, Instance):
        if hasattr(inst, "instance") and hasattr(inst, "trace"):
            if events is None:
                events = inst.trace
            inst = inst.instance
        else:
            raise TypeError("inst must be an Instance or a GLResult")
    if events is not None and hasattr(events, "trace") and not isinstance(events, list):
        events = events.trace
    if not events:
        raise ValueError("no trace events: run glpartition(..., trace=True) with a backend that emits traces")
    return inst, list(events)


# --------------------------------------------------------------------------- final partition
def _parts_map(inst: Instance, parts: Any) -> dict[int, int]:
    """``vertex -> terminal`` from ``parts`` (list aligned with terminals, or ``{t: [...]}``)."""
    part_of: dict[int, int] = {}
    if isinstance(parts, dict):
        for t, vs in parts.items():
            for v in vs:
                part_of[int(v)] = int(t)
    else:
        for i, vs in enumerate(parts):
            t = inst.terminals[i]
            for v in vs:
                part_of[int(v)] = t
    return part_of


def svg_hatch_defs(indices: Iterable[int]) -> str:
    """``<pattern>`` definitions for terminals beyond the 12-colour palette."""
    out = []
    for i in sorted(set(indices)):
        ang = terminal_svg_hatch_angle(i)
        if ang is None:
            continue
        out.append(
            f'<pattern id="hatch{i}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate({ang})">'
            f'<rect width="6" height="6" fill="{terminal_color(i)}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#333" stroke-width="1.2"/></pattern>'
        )
    return "".join(out)


def svg_fill(i: int) -> str:
    """SVG ``fill`` value for terminal index ``i`` (colour or hatch pattern)."""
    return f"url(#hatch{i})" if terminal_svg_hatch_angle(i) is not None else terminal_color(i)


def partition_svg(
    inst: Instance,
    parts: Any,
    width: int = 400,
    *,
    parents: dict[int, int] | None = None,
    layout: Layout | None = None,
    labels: dict[int, str] | None = None,
) -> str:
    """Standalone SVG (string, starts with ``<svg``) of a finished partition:
    vertices coloured by part, terminals as rounded squares, arcs inside a
    part in the part's colour, crossing arcs grey, ``parents`` (the
    in-arborescence certificate) drawn thick.  Used by the benchmark dashboard.
    """
    layout = layout or compute_layout(inst)
    part_of = _parts_map(inst, parts)
    tidx = inst.terminal_index()
    xmin, xmax, ymin, ymax = layout_bounds(layout, pad=0.7)
    scale = width / (xmax - xmin)
    height = int(round((ymax - ymin) * scale))
    r = node_radius(layout) * scale

    def P(v: int) -> tuple[float, float]:
        x, y = layout[v]
        return ((x - xmin) * scale, (ymax - y) * scale)

    orig = set(inst.arcs)
    colors_used: set[str] = set()
    body: list[str] = []
    tree = {(p, q) for p, q in (parents or {}).items()}
    for u, v in inst.arcs:
        same = part_of.get(u) == part_of.get(v) and u in part_of
        col = terminal_color(tidx[part_of[u]]) if same else "#c8c8c8"
        thick = (u, v) in tree
        colors_used.add(col)
        start, c, end, _ = arc_geometry(P(u), P(v), (v, u) in orig, r, r + 2)
        d = f"M{_fmt(start[0])},{_fmt(start[1])} " + (f"Q{_fmt(c[0])},{_fmt(c[1])} " if c else "L") + f"{_fmt(end[0])},{_fmt(end[1])}"
        body.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{3.0 if thick else 1.1}" marker-end="url(#arr{col[1:]})"/>')
    for v in range(inst.n):
        x, y = P(v)
        lab = _html.escape(vertex_label(inst, v, labels))
        if v in tidx:
            i = tidx[v]
            s = 2.1 * r
            body.append(f'<rect x="{_fmt(x - s / 2)}" y="{_fmt(y - s / 2)}" width="{_fmt(s)}" height="{_fmt(s)}" rx="{_fmt(r * 0.3)}" fill="{svg_fill(i)}" stroke="#222" stroke-width="1.2"/>')
            body.append(f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{_fmt(r * 0.95)}" font-weight="bold" text-anchor="middle" dominant-baseline="central">{lab}</text>')
        else:
            t = part_of.get(v)
            fill = svg_fill(tidx[t]) if t is not None and t in tidx else "white"
            body.append(f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(r)}" fill="{fill}" stroke="#333" stroke-width="1"/>')
            body.append(f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{_fmt(r * 0.9)}" text-anchor="middle" dominant-baseline="central">{lab}</text>')
    markers = "".join(
        f'<marker id="arr{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c in sorted(colors_used)
    )
    defs = markers + svg_hatch_defs(tidx.values())
    title = _html.escape(inst.name or "partition")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica, Arial, sans-serif"><title>{title}</title><defs>{defs}</defs>'
        f'<rect width="{width}" height="{height}" fill="white"/>' + "".join(body) + "</svg>"
    )


def final_figure(inst: Instance, parts: Any, *, parents: dict[int, int] | None = None, layout: Layout | None = None, labels: dict[int, str] | None = None, figsize: tuple[float, float] = (7.0, 5.0), dpi: int = 100):
    """Matplotlib figure of a finished partition (vertices coloured by part,
    in-arborescence ``parents`` thick when given)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    layout = layout or compute_layout(inst)
    part_of = _parts_map(inst, parts)
    tidx = inst.terminal_index()
    r = node_radius(layout)
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    xmin, xmax, ymin, ymax = layout_bounds(layout)
    xmin, xmax = _fit_limits(ax, xmin, xmax, ymin, ymax)
    ax.axis("off")
    ppu = ax.get_position().width * fig.get_figwidth() * 72.0 / (xmax - xmin)
    orig = set(inst.arcs)
    tree = {(p, q) for p, q in (parents or {}).items()}
    for u, v in inst.arcs:
        same = u in part_of and part_of.get(u) == part_of.get(v)
        col = terminal_color(tidx[part_of[u]]) if same else "#c8c8c8"
        lw = 3.0 if (u, v) in tree else 1.0
        ax.add_patch(FancyArrowPatch(layout[u], layout[v], arrowstyle="-|>,head_length=5,head_width=2.5", connectionstyle=f"arc3,rad={0.18 if (v, u) in orig else 0}", shrinkA=r * ppu, shrinkB=r * ppu + 1.5, color=col, lw=lw, mutation_scale=1.6, zorder=2))
    for v in range(inst.n):
        x, y = layout[v]
        if v in tidx:
            s = 2.1 * r
            ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.12", facecolor=terminal_color(tidx[v]), edgecolor="#222", lw=1.2, hatch=terminal_hatch(tidx[v]), zorder=4))
        else:
            t = part_of.get(v)
            col = terminal_color(tidx[t]) if t in tidx else "white"
            ax.add_patch(Circle((x, y), r, facecolor=col, edgecolor="#333", lw=1.0, hatch=terminal_hatch(tidx[t]) if t in tidx else None, zorder=4))
        ax.text(x, y, vertex_label(inst, v, labels), fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)
    sizes = ", ".join(f"{terminal_label(inst, t)}: {sum(1 for v, tt in part_of.items() if tt == t)}" for t in inst.terminals)
    ax.set_title(f"{inst.name or 'instance'} — final partition ({sizes})", fontsize=10, loc="left")
    return fig
