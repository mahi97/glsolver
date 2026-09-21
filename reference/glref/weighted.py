"""Reference implementation of the weighted polynomial-time algorithm:
GLWeightedPartition [Alg 3], RoundAndRemove [Alg 4] and the min-cost split
assignment [Prop 5.4] of arXiv 2608.30945 (docs/paper_notes.md §5, §8).

Like :mod:`glref.unweighted` this follows the paper *literally*: first-found
choices, full recomputation of essential sets after every terminal removal,
arc deletion or rounding, the explicit criticality table, and the split
assignment witness recomputed from scratch by one min-cost flow in every step
(iii). It is the correctness reference for the optimized weighted solver.

Unit weights are allowed (``inst.weights is None`` means ``w ≡ 1``); with tight
capacities the algorithm is then an alternative exact unweighted algorithm and
rounding never triggers (paper_notes §13.6, checked by the tests).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from glsolver.instance import Instance

from .critical import criticality_cost, criticality_table
from .essential import all_essential
from .graph import DiGraphState
from .matching import minimal_hall_deficient_set_counted, saturating_matching
from .mincostflow import MinCostFlow
from .trace import Tracer
from .unweighted import InvariantError, RefResult, RefStats, trace_essential

CostFn = Callable[[int, int], int]
SplitAssignment = dict[tuple[int, int], int]

POLICIES = ("first", "last")


def zero_cost(v: int, t: int) -> int:
    """Cost function ``ξ ≡ 0`` (pure feasibility test of FESAC [Def 5.3])."""
    return 0


# ---------------------------------------------------------------------------
# [Prop 5.4] MinCostSplitAssignment
# ---------------------------------------------------------------------------


def min_cost_split_assignment(
    g: DiGraphState,
    ess: dict[int, set[int]],
    cap: dict[int, int],
    w: dict[int, int],
    xi: CostFn = zero_cost,
) -> SplitAssignment | None:
    """[Prop 5.4] / [Alg MinCostSplitAssignment]: a minimum-potential witness ``ψ``
    of the Flow-Essential Split-Assignment Condition [Def 5.3], or ``None``.

    Network: ``s→v`` (cap ``w_v``, cost 0); ``v→t`` (cap ``w_v``, cost
    ``ξ_v(t)``) for essential pairs only; ``t→z`` (cap ``c_t``, cost 0). A
    min-cost flow of value ``W = Σ w_v`` exists iff FESAC holds; its arc flows
    ``f(v,t)`` are ``ψ(v,t)``. Returns ``{(v, t): ψ(v, t)}`` containing only
    the positive entries. ``xi`` is a callable ``(v, t) -> int``.
    """
    nonterms = g.nonterminals()
    terms = list(g.terminals)
    W = sum(w[v] for v in nonterms)
    if W > sum(cap[t] for t in terms):
        return None
    vid = {v: i for i, v in enumerate(nonterms)}
    tid = {t: len(nonterms) + j for j, t in enumerate(terms)}
    s = len(nonterms) + len(terms)
    z = s + 1
    net = MinCostFlow(z + 1)
    arc_of: dict[tuple[int, int], int] = {}
    for v in nonterms:
        net.add_arc(s, vid[v], w[v], 0)
        for t in sorted(ess[v]):
            if t in tid:
                arc_of[(v, t)] = net.add_arc(vid[v], tid[t], w[v], int(xi(v, t)))
    for t in terms:
        net.add_arc(tid[t], z, cap[t], 0)
    value, _cost = net.min_cost_flow(s, z, W)
    if value != W:
        return None
    psi: SplitAssignment = {}
    for (v, t), a in arc_of.items():
        f = net.flow(a)
        if f > 0:
            psi[(v, t)] = f
    return psi


def is_split_witness(
    g: DiGraphState,
    psi: SplitAssignment,
    ess: dict[int, set[int]],
    cap: dict[int, int],
    w: dict[int, int],
) -> tuple[bool, str]:
    """Check [Def 5.3] for a split assignment ``ψ`` against precomputed ``Ess``.

    (0) ``ψ ≥ 0`` on live non-terminal × live terminal pairs; (1) ``ψ(v,t) > 0
    ⇒ t ∈ Ess(v)``; (2) ``Σ_t ψ(v,t) = w_v`` for every live non-terminal; (3)
    ``Σ_v ψ(v,t) ≤ c_t`` for every live terminal. Returns ``(ok, reason)``.
    """
    nonterms = g.nonterminals()
    live_nt = set(nonterms)
    live_t = set(g.terminals)
    sent = {v: 0 for v in nonterms}
    received = {t: 0 for t in g.terminals}
    for (v, t), units in psi.items():
        if units < 0:
            return False, f"psi({v},{t}) = {units} is negative"
        if units == 0:
            continue
        if v not in live_nt:
            return False, f"psi assigns weight from {v}, which is not a live non-terminal"
        if t not in live_t:
            return False, f"psi({v},{t}) > 0 but {t} is not a live terminal"
        if t not in ess[v]:
            return False, f"psi({v},{t}) = {units} but {t} is not essential for {v} (Ess = {sorted(ess[v])})"
        sent[v] += units
        received[t] += units
    for v in nonterms:
        if sent[v] != w[v]:
            return False, f"vertex {v} sends {sent[v]} units, weight is {w[v]}"
    for t in g.terminals:
        if received[t] > cap[t]:
            return False, f"terminal {t} receives {received[t]} units, capacity {cap[t]}"
    return True, "ok"


def split_potential(table: Sequence[dict[int, set[int]]], psi: SplitAssignment) -> int:
    """[Def weighted potential] ``Φ̄(ψ) = Σ_v Σ_t ψ(v,t) · ξ_v(t)`` with ``ξ`` from
    the criticality table [Def 6.2] (``ξ_v(t)`` = number of secondary arcs
    ``e_i`` critical for ``(v, t)``)."""
    return sum(units * criticality_cost(table, v, t) for (v, t), units in psi.items())


# ---------------------------------------------------------------------------
# [Alg 4] RoundAndRemove
# ---------------------------------------------------------------------------


def round_and_remove(
    g: DiGraphState,
    cap: dict[int, int],
    w: dict[int, int],
    stats: RefStats,
    tracer: Tracer,
    debug: bool = False,
    parents: dict[int, int] | None = None,
    policy: str = "first",
) -> dict[int, list[int]]:
    """[Alg 4] RoundAndRemove: complete the parts of a minimal Hall-deficient set.

    Requires: no matching from ``PT(G,T)`` to ``T`` saturating ``T``, all
    ``c_t > 0``, all pre-terminals of out-degree ``≥ 2`` (checked in debug
    mode). Finds an inclusion-minimal ``S ⊆ T`` with no saturating matching
    [Lem 7.6], picks ``t_S ∈ S``, a matching ``M'`` from ``S \\ {t_S}`` to
    ``PT(G,S)`` saturating both sides, sets ``parts[t_S] = {t_S}`` and
    ``parts[t] = {t, M'(t)}`` [Lem 8.6 rounded parts are valid], then deletes
    ``S ∪ PT(G,S)`` from ``g`` (``T := T \\ S``) [Lem 8.8 preservation after
    rounding]. Mutates ``g``; returns ``{t: [t, ...]}`` for ``t ∈ S``.

    If ``parents`` is given, ``parents[M'(t)]`` is set to the *original* head
    of the arc ``(M'(t), t)`` (``g.orig_head``, paper_notes §13.4): this is
    ``t`` itself unless the arc was created by an earlier contraction into
    ``t``, in which case it is a vertex already in the part of ``t`` — either
    way an original arc into the part, as the certificate requires.

    Every ``saturating_matching`` test performed — the ``≤ |T|^2 + 1`` of the
    minimal-set search plus the one computing ``M'`` — is counted in
    ``stats.matching_calls``.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if debug:
        if any(cap[t] <= 0 for t in g.terminals):
            raise InvariantError("RoundAndRemove requires positive capacities")
        if any(g.out_degree(p) < 2 for p in g.pre_terminals()):
            raise InvariantError("RoundAndRemove requires pre-terminal out-degree >= 2")
    S, tests = minimal_hall_deficient_set_counted(g)
    stats.matching_calls += tests  # ≤ |T|^2 + 1 saturating-matching tests (paper_notes §8)
    if S is None:
        raise InvariantError("RoundAndRemove called although T has a saturating matching")
    t_S = S[0] if policy == "first" else S[-1]
    rest = [t for t in S if t != t_S]
    stats.matching_calls += 1
    M = saturating_matching(g, rest)
    if M is None:
        raise InvariantError(f"[Lem 7.6] S \\ {{t_S}} = {rest} has no saturating matching")
    pt = set(g.pre_terminals(S))
    matched = set(M.values())
    if matched != pt:  # [Lem 7.6]: |PT(G,S)| = |S| - 1 and M' saturates both sides
        raise InvariantError(
            f"[Lem 7.6] matched pre-terminals {sorted(matched)} differ from PT(G,S) = {sorted(pt)}"
        )
    parts: dict[int, list[int]] = {t_S: [t_S]}
    for t in rest:
        p = M[t]
        parts[t] = [t, p]
        if parents is not None:
            parents[p] = g.orig_head[(p, t)]
    tracer.record(
        "round_and_remove",
        S=list(S),
        pairs=[[t, M[t]] for t in rest],
        t_S=t_S,
        capacities={t: cap[t] for t in S},
        weights={M[t]: w[M[t]] for t in rest},
    )
    for p in sorted(pt):
        g.remove_vertex(p)
    for t in S:
        g.remove_vertex(t)  # also drops t from g.terminals
    stats.roundings += 1
    return parts


# ---------------------------------------------------------------------------
# [Alg 3] GLWeightedPartition
# ---------------------------------------------------------------------------


def _all_essential(
    g: DiGraphState, stats: RefStats, *, debug_check: bool = False
) -> tuple[dict[int, int], dict[int, set[int]]]:
    """``κ`` and ``Ess`` of every live non-terminal [Prop 4.2] (``|V\\T|`` flows).

    ``debug_check=True`` marks a recomputation done only to verify an invariant
    (A7): its flows and time go to ``stats.debug_max_flow_calls`` /
    ``stats.time_debug_checks``, so ``stats.max_flow_calls`` counts algorithmic
    work only and is independent of ``debug``.
    """
    t0 = time.perf_counter()
    kappa, ess = all_essential(g)
    elapsed = time.perf_counter() - t0
    if debug_check:
        stats.debug_max_flow_calls += len(ess)
        stats.time_debug_checks += elapsed
    else:
        stats.max_flow_calls += len(ess)
        stats.time_essential += elapsed
    return kappa, ess


def _split_assignment(
    g: DiGraphState,
    ess: dict[int, set[int]],
    cap: dict[int, int],
    w: dict[int, int],
    xi: CostFn,
    stats: RefStats,
    *,
    debug_check: bool = False,
) -> SplitAssignment | None:
    """One min-cost split assignment [Prop 5.4]. ``debug_check=True`` books it
    as invariant verification (``stats.debug_min_cost_flow_calls`` /
    ``stats.time_debug_checks``) rather than as algorithmic work."""
    t0 = time.perf_counter()
    psi = min_cost_split_assignment(g, ess, cap, w, xi)
    elapsed = time.perf_counter() - t0
    if debug_check:
        stats.debug_min_cost_flow_calls += 1
        stats.time_debug_checks += elapsed
    else:
        stats.min_cost_flow_calls += 1
        stats.time_min_cost_flow += elapsed
    return psi


def gl_weighted_partition(
    inst: Instance,
    *,
    tracer: Tracer | None = None,
    debug: bool = False,
    policy: str = "first",
) -> RefResult:
    """[Alg 3] GLWeightedPartition, iterative form (paper_notes §8).

    Loop while non-terminals remain: (i) remove a zero-capacity terminal
    [Lem 8.1] and recompute ``Ess`` (§13.3); (ii) contract an out-degree-one
    pre-terminal ``p`` into its terminal ``t`` with ``c_t -= w_p`` [Lem 8.2]
    (``Ess`` unchanged, §13.2); (iii) if a saturating matching exists, choose
    secondary arcs, build the criticality table [Def 6.2], compute the
    minimum-potential split witness ``ψ`` [Prop 5.4] and delete the first
    secondary arc critical for no pair with ``ψ(v,t) > 0`` [Lem 8.4/8.5];
    (iv) otherwise RoundAndRemove [Alg 4].

    ``policy`` (``"first"`` | ``"last"``) fixes which of several valid
    candidates (zero-capacity terminal, out-degree-one pre-terminal, secondary
    arc, deletable arc, ``t_S``) is taken; both are valid (§13.8).

    Precondition: FESAC [Def 5.3], tested by a zero-cost min-cost flow. If it
    fails the result has ``status == "precondition_failed"`` (the theorem's
    guarantee does not apply; a partition may still exist, §13.9). Otherwise
    the parts are aligned with ``inst.terminals`` and satisfy
    ``Σ_{v ∈ V_t \\ T} w_v ≤ c_t + w_max - 1`` [Thm weighted-k-t-conn]; the
    in-arborescence ``parents`` (original arcs) certifies connectivity.

    Debug mode additionally verifies after every operation that FESAC still
    holds (zero-cost flow), that ``Ess`` is unchanged by contractions, that
    ``ψ`` is a witness, and that the chosen arc is non-critical for ``ψ``.
    The flows of these re-checks are booked in ``stats.debug_max_flow_calls``,
    ``stats.debug_min_cost_flow_calls`` and ``stats.time_debug_checks``; the
    algorithmic counters (``max_flow_calls``, ``min_cost_flow_calls``,
    ``matching_calls``, ...) are the same with and without ``debug``.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    tracer = tracer or Tracer(enabled=False)
    stats = RefStats()
    g = DiGraphState.from_instance(inst)
    tset = set(inst.terminals)
    cap: dict[int, int] = {t: c for t, c in zip(inst.terminals, inst.capacities)}
    if inst.weights is None:
        w: dict[int, int] = {v: 1 for v in range(inst.n) if v not in tset}
    else:
        w = {v: int(inst.weights[v]) for v in range(inst.n) if v not in tset}
    parts: dict[int, list[int]] = {t: [t] for t in inst.terminals}
    parents: dict[int, int] = {}

    def pick(seq: Sequence[Any]) -> Any:
        return seq[0] if policy == "first" else seq[-1]

    tracer.record(
        "init", n=inst.n, k=inst.k, terminals=list(inst.terminals),
        capacities=list(inst.capacities), arcs=[list(a) for a in inst.arcs],
        weights=[0 if v in tset else w[v] for v in range(inst.n)],
    )
    kappa, ess = _all_essential(g, stats)
    trace_essential(tracer, ess, kappa)

    # precondition: FESAC via a zero-cost min-cost flow [Def 5.3, Prop 5.4]
    psi = _split_assignment(g, ess, cap, w, zero_cost, stats)
    if psi is None:
        empty = [v for v in g.nonterminals() if not ess[v]]
        msg = "Flow-Essential Split-Assignment Condition fails: no split witness exists"
        if empty:
            msg += f"; vertices with no essential terminal: {empty[:10]}"
        return RefResult("precondition_failed", [], {}, {}, stats, msg, tracer.events if tracer.enabled else None)
    tracer.record("min_cost_split", psi=[[v, t, u] for (v, t), u in sorted(psi.items())], cost=0)

    def _check_fesac(where: str) -> None:
        if _split_assignment(g, ess, cap, w, zero_cost, stats, debug_check=True) is None:
            raise InvariantError(f"FESAC violated after {where}")

    while g.num_nonterminals() > 0:
        stats.steps += 1
        stats.graph_size_over_time.append((g.num_nonterminals(), g.num_arcs()))
        # (i) remove a terminal with zero capacity [Lem 8.1]
        zero = [t for t in g.terminals if cap[t] == 0]
        if zero:
            t = pick(zero)
            g.remove_terminal(t)
            stats.terminal_removals += 1
            tracer.record("remove_terminal", t=t, capacities={x: cap[x] for x in g.terminals})
            kappa, ess = _all_essential(g, stats)  # §13.3: new essential terminals may appear
            trace_essential(tracer, ess, kappa)
            if debug:
                _check_fesac(f"terminal removal of {t}")
            continue
        # (ii) contract a pre-terminal of out-degree one [Lem 8.2]
        deg1 = [p for p in g.pre_terminals() if g.out_degree(p) == 1]
        if deg1:
            p = pick(deg1)
            t = g.out_neighbors(p)[0]
            if cap[t] < w[p]:  # [Lem 8.2]: psi(p, t) = w_p <= c_t under FESAC
                raise InvariantError(f"[Lem 8.2] c_{t} = {cap[t]} < w_{p} = {w[p]} at contraction")
            parents[p] = g.contract(p, t)
            cap[t] -= w[p]
            parts[t].append(p)
            del ess[p]
            del kappa[p]
            w.pop(p)
            stats.contractions += 1
            tracer.record("contract", p=p, t=t, parent=parents[p], capacities={x: cap[x] for x in g.terminals})
            if debug:  # §13.2 / [Lem 7.3]: Ess unchanged for all remaining vertices
                kappa2, ess2 = _all_essential(g, stats, debug_check=True)
                if ess2 != ess or kappa2 != kappa:
                    bad = [v for v in ess if ess2[v] != ess[v] or kappa2[v] != kappa[v]]
                    raise InvariantError(f"A7 violated: essential sets changed after contraction for {bad}")
                _check_fesac(f"contraction of {p}")
            continue
        # (iii) a saturating matching exists: delete a non-critical secondary arc [Lem 8.5]
        stats.matching_calls += 1
        M = saturating_matching(g)
        if M is not None:
            pairs = [(M[t], t) for t in g.terminals]  # (p_i, t_i)
            secondary: list[tuple[int, int]] = []
            for p, t in pairs:
                others = [x for x in g.out[p] if x != t]
                secondary.append((p, pick(others)))
            tracer.record("matching", pairs=pairs, secondary=secondary)
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

            def xi(v: int, t: int, _table: list[dict[int, set[int]]] = table) -> int:
                return criticality_cost(_table, v, t)

            psi = _split_assignment(g, ess, cap, w, xi, stats)
            if psi is None:
                raise InvariantError("[Lem 8.5] min-cost split assignment infeasible although FESAC holds")
            if debug:
                ok, why = is_split_witness(g, psi, ess, cap, w)
                if not ok:
                    raise InvariantError(f"[Prop 5.4] min-cost flow output is not a split witness: {why}")
            phi_val = split_potential(table, psi)
            tracer.record(
                "min_cost_split", psi=[[v, t, u] for (v, t), u in sorted(psi.items())], cost=phi_val,
            )
            # e_nc: a secondary arc critical for no pair (v, t) with psi(v, t) > 0 [Lem 8.4]
            candidates = [
                i for i, row in enumerate(table) if not any(t in row[v] for (v, t) in psi)
            ]
            if not candidates:
                raise InvariantError(
                    "[Lem 8.4] every secondary arc is critical for a pair with positive split weight"
                )
            e_nc = secondary[pick(candidates)]
            g.delete_arc(*e_nc)
            stats.deletions += 1
            tracer.record("delete_arc", u=e_nc[0], v=e_nc[1])
            kappa, ess = _all_essential(g, stats)
            trace_essential(tracer, ess, kappa)
            if debug:  # the same psi witnesses FESAC in G \ e_nc [Lem 8.5]
                ok, why = is_split_witness(g, psi, ess, cap, w)
                if not ok:
                    raise InvariantError(f"[Lem 8.5] psi is not a witness after deleting {e_nc}: {why}")
            continue
        # (iv) no saturating matching: RoundAndRemove [Alg 4]
        rounded = round_and_remove(g, cap, w, stats, tracer, debug=debug, parents=parents, policy=policy)
        for t, members in rounded.items():
            for v in members:
                if v != t:
                    parts[t].append(v)
                    w.pop(v)
                    ess.pop(v, None)
                    kappa.pop(v, None)
        kappa, ess = _all_essential(g, stats)
        trace_essential(tracer, ess, kappa)
        if debug:
            _check_fesac(f"rounding of S = {sorted(rounded)}")

    parts_list = [sorted(parts[t]) for t in inst.terminals]
    tracer.record("done", parts={t: sorted(parts[t]) for t in inst.terminals}, parents=dict(parents))
    return RefResult("ok", parts_list, parents, {}, stats, "ok", tracer.events if tracer.enabled else None)
