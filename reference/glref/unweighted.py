"""Reference implementation of the unweighted polynomial-time algorithm:
GLPartition [Alg 1] and ShiftAssignment [Alg 2] of arXiv 2608.30945.

This follows docs/paper_notes.md §7 *literally* (first-found choices, full
recomputation of essential sets, explicit criticality table). It is the
correctness reference against which the optimized core is checked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from glsolver.instance import Instance

from .assignment import find_witness, is_witness
from .critical import criticality_table, potential
from .essential import all_essential
from .graph import DiGraphState
from .matching import saturating_matching
from .trace import Tracer


@dataclass
class RefStats:
    max_flow_calls: int = 0
    matching_calls: int = 0
    contractions: int = 0
    deletions: int = 0
    cycle_shifts: int = 0
    terminal_removals: int = 0
    shift_calls: int = 0
    steps: int = 0
    time_essential: float = 0.0
    time_criticality: float = 0.0
    graph_size_over_time: list[tuple[int, int]] = field(default_factory=list)
    # counters used by the weighted [Alg 3/4] and DAG [Alg 5] reference solvers
    roundings: int = 0
    min_cost_flow_calls: int = 0
    time_min_cost_flow: float = 0.0
    heap_pushes: int = 0
    heap_pops: int = 0
    stale_pops: int = 0
    # debug-only re-verification work (FESAC re-tests, the A7 essential-set
    # comparison), booked apart from the algorithmic counters above so that
    # those do not depend on ``debug``
    debug_max_flow_calls: int = 0
    debug_min_cost_flow_calls: int = 0
    time_debug_checks: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_flow_calls": self.max_flow_calls,
            "matching_calls": self.matching_calls,
            "contractions": self.contractions,
            "deletions": self.deletions,
            "cycle_shifts": self.cycle_shifts,
            "terminal_removals": self.terminal_removals,
            "shift_calls": self.shift_calls,
            "steps": self.steps,
            "time_essential": self.time_essential,
            "time_criticality": self.time_criticality,
            "graph_size_over_time": list(self.graph_size_over_time),
            "roundings": self.roundings,
            "min_cost_flow_calls": self.min_cost_flow_calls,
            "time_min_cost_flow": self.time_min_cost_flow,
            "heap_pushes": self.heap_pushes,
            "heap_pops": self.heap_pops,
            "stale_pops": self.stale_pops,
            "debug_max_flow_calls": self.debug_max_flow_calls,
            "debug_min_cost_flow_calls": self.debug_min_cost_flow_calls,
            "time_debug_checks": self.time_debug_checks,
        }


@dataclass
class RefResult:
    status: str  # "ok" | "precondition_failed"
    parts: list[list[int]]  # aligned with inst.terminals
    parents: dict[int, int]  # in-arborescence: contracted vertex -> original out-neighbour in its part
    witness: dict[int, int]  # final (possibly empty) witness
    stats: RefStats
    message: str = ""
    trace: list[dict[str, Any]] | None = None


class InvariantError(AssertionError):
    """Raised when a debug assertion (paper_notes §7.2 A1–A8) fails."""


def _all_essential(g: DiGraphState, stats: RefStats) -> tuple[dict[int, int], dict[int, set[int]]]:
    t0 = time.perf_counter()
    kappa, ess = all_essential(g)
    stats.max_flow_calls += len(ess)
    stats.time_essential += time.perf_counter() - t0
    return kappa, ess


def trace_essential(tracer: Tracer, ess: dict[int, set[int]], kappa: dict[int, int]) -> None:
    """Record an ``essential`` event (paper_notes §14) carrying the current ``Ess``/``κ``.

    Emitted after the initial computation and after *every* recomputation
    (terminal removal, arc deletion, rounding; §13.3), so that a replay of
    the trace never needs the solver's own oracle. No-op when tracing is off.
    """
    if tracer.enabled:
        tracer.record("essential", ess={v: sorted(s) for v, s in ess.items()}, kappa=dict(kappa))


def shift_assignment(
    g: DiGraphState,
    cap: dict[int, int],
    phi: dict[int, int],
    ess: dict[int, set[int]],
    stats: RefStats,
    tracer: Tracer,
    debug: bool = False,
) -> tuple[dict[int, int], tuple[int, int]]:
    """[Alg 2] ShiftAssignment: update ``φ`` until a secondary arc is non-critical.

    Requires ``c_t > 0`` for all terminals and ``d^+(p) ≥ 2`` for all
    pre-terminals. Returns ``(φ', e_nc)`` with FEAC holding in ``G \\ e_nc``
    [Lem 7.5]. ``φ`` is modified in place and returned.
    """
    stats.shift_calls += 1
    if debug:
        if any(cap[t] <= 0 for t in g.terminals):
            raise InvariantError("ShiftAssignment requires positive capacities")
        if any(g.out_degree(p) < 2 for p in g.pre_terminals()):
            raise InvariantError("ShiftAssignment requires pre-terminal out-degree >= 2")

    # Line 1: matching from pre-terminals to terminals [Lem 7.8]
    stats.matching_calls += 1
    M = saturating_matching(g)
    if M is None:
        raise InvariantError("[Lem 7.8] no saturating matching although FEAC holds with c_t > 0")
    pairs = [(M[t], t) for t in g.terminals]  # (p_i, t_i)
    # Line 2: secondary arcs e_i = (p_i, q_i) ≠ (p_i, t_i)  (first-found)
    secondary: list[tuple[int, int]] = []
    for p, t in pairs:
        q = next(x for x in g.out[p] if x != t)
        secondary.append((p, q))
    tracer.record("matching", pairs=pairs, secondary=secondary)

    # criticality table for the k secondary arcs [Def 6.1]; graph is fixed inside the loop
    t0 = time.perf_counter()
    table = criticality_table(g, secondary, ess)
    stats.max_flow_calls += len(secondary) * len(ess)
    stats.time_criticality += time.perf_counter() - t0
    if tracer.enabled:
        tracer.record(
            "criticality",
            crit=[[i, v, t] for i, row in enumerate(table) for v, ts in row.items() for t in sorted(ts)],
        )
    if debug:
        for i, (p, t_i) in enumerate(pairs):
            for v, ts in table[i].items():
                if t_i in ts:  # [Lem 7.10]
                    raise InvariantError(f"[Lem 7.10] e_{i} critical for ({v}, t_{i}={t_i})")
                if ts and t_i not in ess[v]:  # [Lem 7.9]
                    raise InvariantError(f"[Lem 7.9] e_{i} critical for {v} but t_i={t_i} not essential")

    while True:
        # Line 4: a secondary arc that is critical for no assignment pair?
        for i, row in enumerate(table):
            if all(phi[v] not in row[v] for v in phi):
                return phi, secondary[i]
        # Lines 7–12: reassignment graph R: arc phi(v_i) -> t_i labelled v_i
        pred: dict[int, tuple[int, int]] = {}
        for i, (p, t_i) in enumerate(pairs):
            v_i = next(v for v in phi if phi[v] in table[i][v])
            if debug:
                if t_i not in ess[v_i]:
                    raise InvariantError(f"[Lem 7.9] t_i={t_i} not essential for v_i={v_i}")
                if phi[v_i] == t_i:
                    raise InvariantError("[Lem 7.10] self-loop in reassignment graph")
            pred[t_i] = (phi[v_i], v_i)
        # every terminal has in-degree one -> walk predecessors until a repeat [§13.5]
        start = g.terminals[0]
        order: list[int] = []
        pos: dict[int, int] = {}
        cur = start
        while cur not in pos:
            pos[cur] = len(order)
            order.append(cur)
            cur = pred[cur][0]
        cycle = order[pos[cur] :]  # terminals on the cycle
        changes = [(pred[t][1], pred[t][0], t) for t in cycle]  # (v, old terminal, new terminal)
        if debug:
            before = potential(table, phi)
        for v, old, new in changes:
            assert phi[v] == old
            phi[v] = new
        stats.cycle_shifts += 1
        if debug:
            after = potential(table, phi)
            if not after < before:  # [Lem 7.11]
                raise InvariantError(f"[Lem 7.11] potential did not decrease: {before} -> {after}")
            ok, why = is_witness(g, phi, ess, cap)
            if not ok:
                raise InvariantError(f"[Lem 7.11] shifted assignment is not a witness: {why}")
        tracer.record(
            "reassignment_graph",
            arcs=[[pred[t][0], t, pred[t][1]] for t in g.terminals],
            cycle=list(cycle),
        )
        tracer.record(
            "cycle_shift",
            changes=[list(c) for c in changes],
            potential_before=(before if debug else None),
            potential_after=(after if debug else None),
            phi=dict(phi),
        )


def gl_partition(
    inst: Instance,
    *,
    phi: dict[int, int] | None = None,
    tracer: Tracer | None = None,
    debug: bool = False,
) -> RefResult:
    """[Alg 1] GLPartition, iterative form (paper_notes §7.1).

    Computes an initial witness (or accepts ``phi``); if none exists the
    Flow-Essential Assignment Condition fails and the result has
    ``status == "precondition_failed"`` (paper_notes §13.9). Otherwise returns
    a partition with the in-arborescence parents as certificate.
    """
    if inst.weights is not None:
        raise ValueError("gl_partition handles unweighted instances; use glref.weighted for weights")
    tracer = tracer or Tracer(enabled=False)
    stats = RefStats()
    g = DiGraphState.from_instance(inst)
    cap = {t: c for t, c in zip(inst.terminals, inst.capacities)}
    parts: dict[int, list[int]] = {t: [t] for t in inst.terminals}
    parents: dict[int, int] = {}

    tracer.record(
        "init", n=inst.n, k=inst.k, terminals=list(inst.terminals),
        capacities=list(inst.capacities), arcs=[list(a) for a in inst.arcs],
    )
    kappa, ess = _all_essential(g, stats)
    trace_essential(tracer, ess, kappa)
    if phi is None:
        phi = find_witness(g, ess, cap)
        if phi is None:
            empty = [v for v in g.nonterminals() if not ess[v]]
            msg = "Flow-Essential Assignment Condition fails: no witness assignment exists"
            if empty:
                msg += f"; vertices with no essential terminal: {empty[:10]}"
            return RefResult("precondition_failed", [], {}, {}, stats, msg, tracer.events if tracer.enabled else None)
    else:
        ok, why = is_witness(g, phi, ess, cap)
        if not ok:
            raise ValueError(f"given phi is not a witness: {why}")
    tracer.record("witness", phi=dict(phi))

    def _check_witness(where: str) -> None:
        ok, why = is_witness(g, phi, ess, cap)
        if not ok:
            raise InvariantError(f"A1 violated after {where}: {why}")

    while g.num_nonterminals() > 0:
        stats.steps += 1
        stats.graph_size_over_time.append((g.num_nonterminals(), g.num_arcs()))
        # (i) remove a terminal with zero capacity [Lem 7.2]
        zero = [t for t in g.terminals if cap[t] == 0]
        if zero:
            t = zero[0]
            g.remove_terminal(t)
            stats.terminal_removals += 1
            tracer.record("remove_terminal", t=t, capacities={x: cap[x] for x in g.terminals})
            kappa, ess = _all_essential(g, stats)  # §13.3: new essential terminals may appear
            trace_essential(tracer, ess, kappa)
            if debug:
                _check_witness(f"terminal removal of {t}")
            continue
        # (ii) contract a pre-terminal of out-degree one [Lem 7.4]
        p_deg1 = next((p for p in g.pre_terminals() if g.out_degree(p) == 1), None)
        if p_deg1 is not None:
            p = p_deg1
            t = g.out_neighbors(p)[0]
            if debug and phi[p] != t:
                raise InvariantError(f"[Lem 7.4] phi({p})={phi[p]} but its only arc goes to {t}")
            parents[p] = g.contract(p, t)
            cap[t] -= 1
            del phi[p]
            del ess[p]
            del kappa[p]
            parts[t].append(p)
            stats.contractions += 1
            tracer.record("contract", p=p, t=t, parent=parents[p], capacities={x: cap[x] for x in g.terminals})
            if debug:  # §13.2: Ess unchanged for all remaining vertices
                kappa2, ess2 = _all_essential(g, stats)
                if ess2 != ess or kappa2 != kappa:
                    bad = [v for v in ess if ess2[v] != ess[v] or kappa2[v] != kappa[v]]
                    raise InvariantError(f"A7 violated: essential sets changed after contraction for {bad}")
                _check_witness(f"contraction of {p}")
            continue
        # (iii) delete a non-critical arc [Lem 7.5]
        phi, e_nc = shift_assignment(g, cap, phi, ess, stats, tracer, debug=debug)
        g.delete_arc(*e_nc)
        stats.deletions += 1
        tracer.record("delete_arc", u=e_nc[0], v=e_nc[1])
        kappa, ess = _all_essential(g, stats)
        trace_essential(tracer, ess, kappa)
        if debug:
            _check_witness(f"deletion of {e_nc}")  # A6

    parts_list = [sorted(parts[t]) for t in inst.terminals]
    tracer.record("done", parts={t: sorted(parts[t]) for t in inst.terminals}, parents=dict(parents))
    return RefResult("ok", parts_list, parents, phi, stats, "ok", tracer.events if tracer.enabled else None)
