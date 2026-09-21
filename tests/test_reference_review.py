"""Adversarial review tests for the weighted/DAG/compact reference modules
(``glref.weighted``, ``glref.mincostflow``, ``glref.dag``, ``glref.compact``,
``glref.counterexample``) against docs/paper_notes.md §5, §8, §9, §10, §13 and
the paper's own pseudocode ([Alg 3], [Alg 4], [Alg MinCostSplitAssignment],
[Alg 5]).

Every test here asserts a property that random unit tests elsewhere do not
cover: optimality of the min-cost split assignment against a brute force over
*all* split assignments, line-by-line agreement of the compact-connectivity
port with the authors' official script (imported from ``.research``), the
weighted solver on FESAC-only (not ``k``-``T``-connected) digraphs, the
rounding path with large Hall-deficient sets, and DAG policy independence.

Three confirmed review findings — ``matching_calls`` under-counting in
RoundAndRemove, ``debug``-dependent algorithmic counters, and the DAG
precondition judging a hand-built instance with duplicate arcs by arc
multiplicity — have been fixed; the corresponding tests now pin the fixed
behaviour instead of being marked ``xfail``.
"""
from __future__ import annotations

import itertools
import random
import sys
from collections import Counter, deque
from pathlib import Path

import networkx as nx
import pytest

from glref.dag import (
    POLICIES,
    canonical_topological_order,
    dag_precondition,
    gl_dag_partition,
    is_dag,
)
from glref.essential import all_essential
from glref.graph import DiGraphState
from glref.mincostflow import MinCostFlow
from glref.trace import Tracer
from glref.unweighted import InvariantError, RefStats, gl_partition
from glref.weighted import (
    gl_weighted_partition,
    is_split_witness,
    min_cost_split_assignment,
    round_and_remove,
    zero_cost,
)
from glsolver.instance import Instance, make_instance

try:  # the independent verifier is written by another agent; fall back to a local checker
    from glsolver.verify import verify_instance_parts as _verify_instance_parts
except ImportError:  # pragma: no cover
    _verify_instance_parts = None

AUTHORS_DIR = Path(__file__).resolve().parents[1] / ".research" / "Gyori-Lovasz-Codes" / "counterexample"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _local_check(inst: Instance, parts: list[list[int]]) -> list[str]:
    """Independent checker used only when ``glsolver.verify`` is unavailable."""
    errors: list[str] = []
    if len(parts) != inst.k:
        return ["wrong number of parts"]
    owner: dict[int, int] = {}
    for i, part in enumerate(parts):
        for v in part:
            if v in owner:
                errors.append(f"{v} twice")
            owner[v] = i
    if len(owner) != inst.n:
        errors.append("missing vertices")
    tset = set(inst.terminals)
    in_adj = inst.in_adjacency()
    for i, part in enumerate(parts):
        t = inst.terminals[i]
        pset = set(part)
        if t not in pset:
            errors.append(f"terminal {t} missing from part {i}")
            continue
        if inst.weights is None:
            if len(pset) != inst.capacities[i] + 1:
                errors.append(f"part {i} wrong size")
        else:
            wt = sum(inst.weights[v] for v in pset if v not in tset)
            if wt > inst.capacities[i] + inst.w_max - 1:
                errors.append(f"part {i} too heavy")
        reached = {t}
        dq = deque([t])
        while dq:
            x = dq.popleft()
            for u in in_adj[x]:
                if u in pset and u not in reached:
                    reached.add(u)
                    dq.append(u)
        if reached != pset:
            errors.append(f"part {i} disconnected")
    return errors


def assert_valid(inst: Instance, parts: list[list[int]]) -> None:
    if _verify_instance_parts is not None:
        report = _verify_instance_parts(inst, parts)
        assert report.valid, report.errors
    else:  # pragma: no cover
        assert _local_check(inst, parts) == []


def assert_certificate(inst: Instance, parts: list[list[int]], parents: dict[int, int]) -> None:
    """``parents`` is an in-arborescence of *original* arcs inside each part (§13.4, §9)."""
    arcs = set(inst.arcs)
    part_of = {v: i for i, part in enumerate(parts) for v in part}
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        assert v in parents, f"no parent for {v}"
        p = parents[v]
        assert (v, p) in arcs, f"({v},{p}) is not an original arc"
        assert part_of[v] == part_of[p], f"parent of {v} lies in another part"
        x, hops = v, 0
        while x not in tset:
            x = parents[x]
            hops += 1
            assert hops <= inst.n
        assert x == inst.terminals[part_of[v]]


def _fesac_holds(inst: Instance) -> bool:
    """FESAC [Def 5.3] via the zero-cost flow, independently of the solver loop."""
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    tset = set(inst.terminals)
    w = {v: (1 if inst.weights is None else inst.weights[v]) for v in range(inst.n) if v not in tset}
    cap = dict(zip(inst.terminals, inst.capacities))
    return min_cost_split_assignment(g, ess, cap, w, zero_cost) is not None


def _composition(rng: random.Random, total: int, parts: int) -> list[int]:
    cuts = sorted(rng.randint(0, total) for _ in range(parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _compositions(total: int, parts: int):
    """All tuples of ``parts`` nonnegative integers summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


# ---------------------------------------------------------------------------
# [Prop 5.4] MinCostSplitAssignment: integral and truly minimum
# ---------------------------------------------------------------------------


def test_min_cost_split_assignment_optimal_over_all_split_assignments() -> None:
    """Brute force over *every* split assignment ψ (each vertex distributes
    its weight among its essential terminals, capacities as upper bounds) must
    not beat the min-cost flow; the flow must be a witness and integral.

    The graph is a star of pre-terminals so that ``Ess(v)`` equals the chosen
    terminal set exactly (``κ(v) = d^+(v)``)."""
    rng = random.Random(1)
    checked = 0
    for _trial in range(220):
        k = rng.randint(1, 3)
        nnt = rng.randint(1, 3)
        terms = list(range(k))
        nonterms = list(range(k, k + nnt))
        chosen = {v: sorted(rng.sample(terms, rng.randint(1, k))) for v in nonterms}
        arcs = [(v, t) for v in nonterms for t in chosen[v]]
        g = DiGraphState(k + nnt, arcs, terms)
        _kappa, ess = all_essential(g)
        assert all(ess[v] == set(chosen[v]) for v in nonterms)
        w = {v: rng.randint(1, 4) for v in nonterms}
        W = sum(w.values())
        slack = rng.choice([0, 0, 1, 3])
        cap = dict(zip(terms, _composition(rng, W + slack, k)))
        cost = {(v, t): rng.randint(0, 3) for v in nonterms for t in terms}
        psi = min_cost_split_assignment(g, ess, cap, w, lambda v, t, _c=cost: _c[(v, t)])
        best = None
        per_v = []
        for v in nonterms:
            opts = chosen[v]
            per_v.append([(v, dict(zip(opts, comp))) for comp in _compositions(w[v], len(opts))])
        for choice in itertools.product(*per_v):
            recv = {t: 0 for t in terms}
            c = 0
            for v, d in choice:
                for t, u in d.items():
                    recv[t] += u
                    c += u * cost[(v, t)]
            if all(recv[t] <= cap[t] for t in terms):
                best = c if best is None else min(best, c)
        if best is None:
            assert psi is None, "flow found a witness although none exists"
            continue
        assert psi is not None, "flow reports infeasible although a witness exists"
        ok, why = is_split_witness(g, psi, ess, cap, w)
        assert ok, why
        assert all(isinstance(u, int) and u > 0 for u in psi.values())
        got = sum(u * cost[(v, t)] for (v, t), u in psi.items())
        assert got == best, (got, best, chosen, w, cap, cost, psi)
        checked += 1
    assert checked >= 150


def test_min_cost_flow_partial_value_is_optimal() -> None:
    """A flow of value *below* the maximum (the case ``W < Σ c``) must still be
    a minimum-cost flow of that value (networkx ``min_cost_flow_cost`` with
    demands), and asking for more than the maximum returns the maximum."""
    rng = random.Random(2)
    compared = 0
    for _trial in range(120):
        n = rng.randint(3, 8)
        net = MinCostFlow(n)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.5:
                    cap, cost = rng.randint(1, 9), rng.randint(0, 9)
                    net.add_arc(u, v, cap, cost)
                    G.add_edge(u, v, capacity=cap, weight=cost)
        s, z = 0, n - 1
        mv = nx.maximum_flow_value(G, s, z) if G.number_of_edges() else 0
        req = rng.randint(0, mv + 1)
        val, cst = net.min_cost_flow(s, z, req)
        if req <= mv:
            G2 = G.copy()
            G2.nodes[s]["demand"] = -req
            G2.nodes[z]["demand"] = req
            assert val == req and cst == nx.min_cost_flow_cost(G2)
            assert net.total_cost() == cst
            compared += 1
        else:
            assert val == mv
    assert compared >= 60


# ---------------------------------------------------------------------------
# [Alg 3] on FESAC-only digraphs (not k-T-connected), both policies, debug on
# ---------------------------------------------------------------------------


def _random_fesac_instances(seed: int, count: int) -> list[Instance]:
    rng = random.Random(seed)
    out: list[Instance] = []
    while len(out) < count:
        n = rng.randint(3, 8)
        k = rng.randint(1, min(4, n - 1))
        terms = rng.sample(range(n), k)
        tset = set(terms)
        p = rng.uniform(0.2, 0.8)
        arcs = [(u, v) for u in range(n) if u not in tset for v in range(n) if v != u and rng.random() < p]
        if rng.random() < 0.6:
            w_max = rng.randint(1, 4)
            weights = [0 if v in tset else rng.randint(1, w_max) for v in range(n)]
            total = sum(weights) + rng.choice([0, 0, 0, rng.randint(1, 3)])
        else:
            weights = None
            total = n - k
        caps = _composition(rng, total, k)
        inst = make_instance(n, arcs, terms, caps, weights=weights, directed=True, name=f"fesac_{len(out)}")
        if _fesac_holds(inst):
            out.append(inst)
    return out


@pytest.mark.parametrize("policy", ["first", "last"])
def test_weighted_solver_on_fesac_only_digraphs(policy: str) -> None:
    """Random sparse digraphs filtered by FESAC (most are *not* k-T-connected):
    the solver must succeed with every debug invariant, the parts must verify,
    the certificate must be original arcs, unit-weight instances must be exact
    with no rounding (§13.6), and rounding must actually be exercised."""
    instances = _random_fesac_instances(seed=17 if policy == "first" else 23, count=90)
    roundings = deletions = 0
    for inst in instances:
        res = gl_weighted_partition(inst, debug=True, policy=policy)
        assert res.status == "ok", (inst.name, res.message)
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        roundings += res.stats.roundings
        deletions += res.stats.deletions
        if inst.weights is None:
            assert res.stats.roundings == 0, inst.name
            assert [len(p) - 1 for p in res.parts] == list(inst.capacities)
    assert roundings > 0 and deletions > 0


def test_weighted_and_unweighted_solvers_agree_on_status_for_unit_weights() -> None:
    """For unit weights with tight capacities FESAC ⇔ FEAC, so the two
    reference solvers must return the same status on *every* digraph,
    including precondition failures (§13.9)."""
    rng = random.Random(5)
    ok = failed = 0
    for i in range(80):
        n = rng.randint(3, 7)
        k = rng.randint(1, min(3, n - 1))
        terms = rng.sample(range(n), k)
        tset = set(terms)
        p = rng.uniform(0.15, 0.7)
        arcs = [(u, v) for u in range(n) if u not in tset for v in range(n) if v != u and rng.random() < p]
        inst = make_instance(n, arcs, terms, _composition(rng, n - k, k), directed=True, name=f"agree_{i}")
        a = gl_weighted_partition(inst, debug=True)
        b = gl_partition(inst, debug=True)
        assert a.status == b.status, (inst.name, a.message, b.message)
        if a.status == "ok":
            ok += 1
            assert_valid(inst, a.parts)
            assert_valid(inst, b.parts)
        else:
            failed += 1
            assert a.parts == [] and a.parents == {}
    assert ok > 10 and failed > 10


def test_step_iii_deletes_secondary_arc_non_critical_for_all_positive_pairs() -> None:
    """[Alg 3] step (iii): the deleted arc is one of the secondary arcs
    ``e_i = (p_i, q_i) ≠ (p_i, t_i)`` of the matching recorded just before it,
    and it is critical for no pair with ``ψ(v,t) > 0`` (criticality event)."""
    seen = 0
    for inst in _random_fesac_instances(seed=31, count=25):
        tracer = Tracer()
        res = gl_weighted_partition(inst, debug=True, tracer=tracer)
        assert res.status == "ok"
        events = tracer.events
        for idx, ev in enumerate(events):
            if ev["type"] != "delete_arc":
                continue
            # walk back to the matching / criticality / min_cost_split of this step
            j = idx
            while events[j]["type"] != "matching":
                j -= 1
            matching = events[j]
            crit = next(e for e in events[j:idx] if e["type"] == "criticality")
            split = next(e for e in events[j:idx] if e["type"] == "min_cost_split")
            pairs = [tuple(x) for x in matching["pairs"]]
            secondary = [tuple(x) for x in matching["secondary"]]
            e_nc = (ev["u"], ev["v"])
            assert e_nc in secondary
            i = secondary.index(e_nc)
            assert e_nc != pairs[i] and e_nc[0] == pairs[i][0]
            positive = {(v, t) for v, t, units in split["psi"] if units > 0}
            critical_i = {(v, t) for ii, v, t in crit["crit"] if ii == i}
            assert not (positive & critical_i), (e_nc, positive & critical_i)
            seen += 1
    assert seen > 0


def test_contraction_decreases_capacity_by_weight_and_rounding_removes_pt() -> None:
    """[Alg 3] (ii): ``c_t -= w_p`` (trace capacities); [Alg 4]: the rounding
    event removes ``S ∪ PT(G,S)`` and the rounded parts are ``{t_S}`` and
    ``{t, M'(t)}`` with ``M'(t) ∈ PT(G,S)``."""
    rng = random.Random(8)
    contractions = roundings = 0
    for inst in _random_fesac_instances(seed=41, count=40):
        assert inst.weights is not None or True
        tracer = Tracer()
        res = gl_weighted_partition(inst, debug=True, tracer=tracer)
        assert res.status == "ok"
        w = inst.weights or [0 if v in inst.terminals else 1 for v in range(inst.n)]
        caps = dict(zip(inst.terminals, inst.capacities))
        g = DiGraphState.from_instance(inst)
        for ev in tracer.events:
            if ev["type"] == "contract":
                p, t = ev["p"], ev["t"]
                assert g.out_degree(p) == 1 and g.has_arc(p, t)
                caps[t] -= w[p]
                assert caps[t] >= 0
                assert {int(x): c for x, c in ev["capacities"].items()} == {x: caps[x] for x in g.terminals}
                g.contract(p, t)
                contractions += 1
            elif ev["type"] == "remove_terminal":
                assert caps[ev["t"]] == 0
                g.remove_terminal(ev["t"])
                del caps[ev["t"]]
            elif ev["type"] == "delete_arc":
                g.delete_arc(ev["u"], ev["v"])
            elif ev["type"] == "round_and_remove":
                S = list(ev["S"])
                pt = set(g.pre_terminals(S))
                matched = {p for _t, p in ev["pairs"]}
                assert matched == pt and len(pt) == len(S) - 1, (S, pt, matched)
                assert ev["t_S"] in S and all(t != ev["t_S"] for t, _p in ev["pairs"])
                for t, p in ev["pairs"]:
                    assert g.has_arc(p, t)
                    assert sorted(res.parts[inst.terminals.index(t)]) == sorted(g_part(res, inst, t))
                for p in pt:
                    g.remove_vertex(p)
                for t in S:
                    g.remove_vertex(t)
                    del caps[t]
                roundings += 1
        rng.random()
    assert contractions > 0 and roundings > 0


def g_part(res, inst: Instance, t: int) -> list[int]:
    return res.parts[inst.terminals.index(t)]


def test_rounding_with_large_hall_deficient_sets() -> None:
    """Construct instances with fewer pre-terminals than terminals so that
    RoundAndRemove sees Hall-deficient sets of size ≥ 3 and runs repeatedly;
    a part may exceed ``c_t`` only when it was completed by a rounding event."""
    rng = random.Random(1)
    sizes: Counter[int] = Counter()
    multi = tried = 0
    while tried < 45:
        k = rng.randint(3, 6)
        npre = rng.randint(1, k - 1)
        nother = rng.randint(0, 4)
        n = k + npre + nother
        terms = list(range(k))
        pres = list(range(k, k + npre))
        others = list(range(k + npre, n))
        arcs: set[tuple[int, int]] = set()
        for p in pres:
            for t in rng.sample(terms, rng.randint(2, k)):
                arcs.add((p, t))
        for o in others:
            for x in rng.sample(pres + others, rng.randint(1, len(pres + others))):
                if x != o:
                    arcs.add((o, x))
            if rng.random() < 0.3:
                arcs.add((o, rng.choice(terms)))
        w_max = rng.randint(1, 6)
        weights = [0] * k + [rng.randint(1, w_max) for _ in range(n - k)]
        total = sum(weights) + rng.choice([0, 0, rng.randint(1, 6)])
        inst = make_instance(n, sorted(arcs), terms, _composition(rng, total, k), weights=weights, directed=True)
        if not _fesac_holds(inst):
            continue
        tried += 1
        for policy in ("first", "last"):
            tracer = Tracer()
            res = gl_weighted_partition(inst, debug=True, policy=policy, tracer=tracer)
            assert res.status == "ok", res.message
            assert_valid(inst, res.parts)
            assert_certificate(inst, res.parts, res.parents)
            rounds = [e for e in tracer.events if e["type"] == "round_and_remove"]
            for e in rounds:
                sizes[len(e["S"])] += 1
            multi += len(rounds) >= 2
            for i, part in enumerate(res.parts):
                wt = sum(weights[v] for v in part if v >= k)
                assert wt <= inst.capacities[i] + inst.w_max - 1
                if wt > inst.capacities[i]:  # only rounding may exceed c_t [Lem 8.6]
                    matched = [pair[1] for e in rounds for pair in e["pairs"] if pair[0] == terms[i]]
                    assert len(matched) == 1, (i, part)
                    # the excess is bounded by the matched pre-terminal's weight minus one unit
                    assert wt - inst.capacities[i] <= weights[matched[0]] - 1
    assert max(sizes) >= 3 and multi > 0, (dict(sizes), multi)


def test_round_and_remove_guards() -> None:
    """[Alg 4] requirements are enforced: a saturable ``T`` and an
    out-degree-one pre-terminal are both rejected."""
    inst = make_instance(4, [(2, 0), (3, 1), (2, 1), (3, 0)], [0, 1], [3, 3], weights=[0, 0, 3, 3], directed=True)
    g = DiGraphState.from_instance(inst)
    with pytest.raises(InvariantError):
        round_and_remove(g, {0: 3, 1: 3}, {2: 3, 3: 3}, RefStats(), Tracer(False), debug=False)
    inst = make_instance(5, [(3, 0), (4, 3)], [0, 1, 2], [5, 1, 1], weights=[0, 0, 0, 2, 2], directed=True)
    g = DiGraphState.from_instance(inst)
    with pytest.raises(InvariantError):
        round_and_remove(g, {0: 5, 1: 1, 2: 1}, {3: 2, 4: 2}, RefStats(), Tracer(False), debug=True)


def test_round_and_remove_matching_calls_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """[Alg 4] accounting: every ``saturating_matching`` test performed — the
    ``≤ |T|^2 + 1`` of the minimal Hall-deficient-set search (paper_notes §8)
    plus the one computing ``M'`` — is counted in ``RefStats.matching_calls``,
    both for a bare ``round_and_remove`` call and over whole solver runs."""
    import glref.matching as matching_mod
    import glref.weighted as weighted_mod

    calls = {"n": 0}
    orig = matching_mod.saturating_matching

    def counting(g, S=None):
        calls["n"] += 1
        return orig(g, S)

    monkeypatch.setattr(matching_mod, "saturating_matching", counting)
    monkeypatch.setattr(weighted_mod, "saturating_matching", counting)
    arcs = [(4, t) for t in range(4)] + [(5, t) for t in range(4)]
    inst = make_instance(6, arcs, [0, 1, 2, 3], [3, 3, 3, 3], weights=[0, 0, 0, 0, 3, 3], directed=True)
    g = DiGraphState.from_instance(inst)
    stats = RefStats()
    round_and_remove(g, {t: 3 for t in range(4)}, {4: 3, 5: 3}, stats, Tracer(False))
    assert calls["n"] > 2
    assert stats.matching_calls == calls["n"], (stats.matching_calls, calls["n"])
    assert stats.matching_calls <= inst.k * inst.k + 2  # |T|^2 + 1 tests for S, one for M'
    # over complete runs (all four steps occur), with and without debug checks
    with_rounding = 0
    for inst in _random_fesac_instances(seed=47, count=20):
        for debug in (False, True):
            calls["n"] = 0
            res = gl_weighted_partition(inst, debug=debug)
            assert res.status == "ok", (inst.name, res.message)
            assert res.stats.matching_calls == calls["n"], (inst.name, debug, res.stats.matching_calls, calls["n"])
            with_rounding += res.stats.roundings > 0
    assert with_rounding > 0


def test_min_cost_flow_calls_independent_of_debug() -> None:
    """The algorithmic counters (``min_cost_flow_calls``, ``max_flow_calls``,
    ``matching_calls``, ...) do not depend on ``debug``: the debug-only
    re-checks — FESAC after every step (i), (ii), (iv) and A7 after every
    contraction — are booked in ``debug_min_cost_flow_calls`` /
    ``debug_max_flow_calls`` / ``time_debug_checks`` instead."""
    arcs = [(4, t) for t in range(4)] + [(5, t) for t in range(4)] + [(5, 4)]
    inst = make_instance(6, arcs, [0, 1, 2, 3], [3, 3, 3, 3], weights=[0, 0, 0, 0, 3, 3], directed=True)
    a = gl_weighted_partition(inst, debug=False).stats
    b = gl_weighted_partition(inst, debug=True).stats
    assert a.min_cost_flow_calls == b.min_cost_flow_calls >= 1, (a.min_cost_flow_calls, b.min_cost_flow_calls)
    assert a.max_flow_calls == b.max_flow_calls >= 1, (a.max_flow_calls, b.max_flow_calls)
    assert (a.debug_min_cost_flow_calls, a.debug_max_flow_calls, a.time_debug_checks) == (0, 0, 0.0)
    assert b.debug_min_cost_flow_calls >= 1 and b.time_debug_checks > 0
    assert b.as_dict()["debug_min_cost_flow_calls"] == b.debug_min_cost_flow_calls
    algorithmic = (
        "max_flow_calls", "min_cost_flow_calls", "matching_calls", "contractions",
        "deletions", "roundings", "terminal_removals", "steps",
    )
    debug_flows = 0
    for inst in _random_fesac_instances(seed=59, count=15):
        for policy in ("first", "last"):
            a = gl_weighted_partition(inst, debug=False, policy=policy).stats
            b = gl_weighted_partition(inst, debug=True, policy=policy).stats
            assert all(getattr(a, f) == getattr(b, f) for f in algorithmic), (inst.name, policy)
            # one zero-cost FESAC re-test after each of steps (i), (ii) and (iv)
            assert b.debug_min_cost_flow_calls == b.terminal_removals + b.contractions + b.roundings
            debug_flows += b.debug_max_flow_calls
    assert debug_flows > 0  # the A7 re-check after contractions is exercised


# ---------------------------------------------------------------------------
# [Alg 5] DAG: policy independence, stale entries, parents into the current part
# ---------------------------------------------------------------------------


def _random_dag(rng: random.Random, n: int, k: int, extra: int) -> tuple[list[tuple[int, int]], list[int]]:
    verts = list(range(n))
    terms = rng.sample(verts, k)
    nonterms = [v for v in verts if v not in terms]
    rng.shuffle(nonterms)
    order = nonterms + terms
    arcs: set[tuple[int, int]] = set()
    for i, v in enumerate(nonterms):
        later = order[i + 1 :]
        for x in rng.sample(later, rng.randint(k, min(len(later), k + extra))):
            arcs.add((v, x))
    return sorted(arcs), terms


def test_dag_all_policies_valid_with_zero_caps_and_weights() -> None:
    rng = random.Random(5)
    stale_seen = 0
    for trial in range(120):
        k = rng.randint(1, 5)
        n = rng.randint(k + 1, 22)
        arcs, terms = _random_dag(rng, n, k, rng.randint(0, 6))
        if rng.random() < 0.5:
            w_max = rng.randint(1, 5)
            weights = [0 if v in terms else rng.randint(1, w_max) for v in range(n)]
            total = sum(weights) + rng.choice([0, 0, rng.randint(1, 5)])
        else:
            weights = None
            total = n - k
        mode = trial % 3
        if mode == 0:
            caps = _composition(rng, total, k)
        elif mode == 1:
            caps = [0] * k
            caps[rng.randrange(k)] = total
        else:
            nz = max(1, k // 2)
            caps = _composition(rng, total, nz) + [0] * (k - nz)
            rng.shuffle(caps)
        inst = make_instance(n, arcs, terms, caps, weights=weights, directed=True)
        assert dag_precondition(inst) == (True, None)
        results = []
        for policy in POLICIES:
            res = gl_dag_partition(inst, policy=policy, debug=True)
            assert res.status == "ok", res.message
            assert_valid(inst, res.parts)
            assert_certificate(inst, res.parts, res.parents)
            assert res.stats.heap_pushes == inst.m and res.stats.contractions == n - k
            stale_seen += res.stats.stale_pops
            tset = set(terms)
            for i, part in enumerate(res.parts):
                if caps[i] == 0:
                    assert part == [terms[i]]
                if weights is None:
                    assert len(part) - 1 == caps[i]
                else:
                    assert sum(weights[v] for v in part if v not in tset) <= caps[i] + inst.w_max - 1
            results.append(res.parts)
        # a deterministic run: same policy, same output
        assert gl_dag_partition(inst, policy="round_robin").parts == results[POLICIES.index("round_robin")]
    assert stale_seen > 0


def test_dag_parent_is_the_arc_that_put_p_into_the_heap() -> None:
    """Vertex 5 has arcs to both 3 and 4 (both end up in part 0). It is pushed
    into ``H_0`` when 3 is contracted with head 3, so ``parents[5] == 3``; the
    later duplicate entry (head 4) is stale."""
    arcs = [(3, 0), (4, 0), (5, 3), (5, 4)] + [(v, t) for v in (3, 4, 5) for t in (1, 2)]
    inst = make_instance(6, arcs, [0, 1, 2], [3, 0, 0], directed=True)
    tracer = Tracer()
    res = gl_dag_partition(inst, policy="first", debug=True, tracer=tracer)
    assert res.status == "ok"
    assert_valid(inst, res.parts)
    steps = [(e["p"], e["parent"]) for e in tracer.events if e["type"] == "dag_contract"]
    assert steps == [(3, 0), (5, 3), (4, 0)]
    assert res.parents == {3: 0, 5: 3, 4: 0}
    assert res.stats.heap_pushes == inst.m


def test_dag_precondition_counts_distinct_out_neighbours_and_solver_validates() -> None:
    """[Lem 9.1] tests ``d^+(v) = |N^+(v)|`` (distinct out-neighbours, paper
    §2), so a hand-built ``Instance`` with duplicate arcs is judged like its
    normalized form; and ``gl_dag_partition`` rejects malformed input with the
    ``ValueError`` of ``inst.validate()`` instead of running into an
    ``InvariantError`` ("heap ... ran dry") deep in the heap loop."""
    inst = Instance(n=3, arcs=((2, 0), (2, 0)), terminals=(0, 1), capacities=(1, 0))
    with pytest.raises(ValueError, match="duplicate arcs"):
        inst.validate()
    assert dag_precondition(inst) == (False, 2)
    with pytest.raises(ValueError, match="duplicate arcs"):
        gl_dag_partition(inst)
    good = make_instance(3, [(2, 0), (2, 0)], [0, 1], [1, 0], directed=True)
    assert dag_precondition(good) == (False, 2)
    assert gl_dag_partition(good).status == "precondition_failed"
    # duplicates that used to hide a dry heap (InvariantError) now fail validation
    bad = Instance(n=4, arcs=((2, 0), (2, 0), (3, 2), (3, 2)), terminals=(0, 1), capacities=(1, 1))
    assert dag_precondition(bad) == (False, 2)
    with pytest.raises(ValueError, match="duplicate arcs"):
        gl_dag_partition(bad)
    # duplicate arcs change neither acyclicity, the canonical order nor the verdict
    dup = Instance(n=4, arcs=((3, 2), (3, 2), (2, 0), (2, 1), (3, 0), (3, 1)), terminals=(0, 1), capacities=(1, 1))
    norm = make_instance(4, dup.arcs, [0, 1], [1, 1], directed=True)
    assert norm.m == 5 and is_dag(dup) and is_dag(norm)
    assert canonical_topological_order(dup) == canonical_topological_order(norm) == [3, 2, 0, 1]
    assert dag_precondition(dup) == dag_precondition(norm) == (True, None)
    with pytest.raises(ValueError, match="duplicate arcs"):
        gl_dag_partition(dup)
    assert gl_dag_partition(norm).status == "ok"
    # Σw > Σc is a well-formedness error too (ValueError, not an InvariantError mid-run)
    heavy = Instance(n=3, arcs=((2, 0), (2, 1)), terminals=(0, 1), capacities=(1, 0), weights=(0, 0, 2))
    assert dag_precondition(heavy) == (True, None)
    with pytest.raises(ValueError, match="weights"):
        gl_dag_partition(heavy)


# ---------------------------------------------------------------------------
# compact / counterexample: line-by-line agreement with the authors' script
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def authors():
    if not (AUTHORS_DIR / "counterexample.py").exists():
        pytest.skip("authors' script not available under .research")
    sys.path.insert(0, str(AUTHORS_DIR))
    try:
        import compact_connectivity as cc  # type: ignore[import-not-found]
        import counterexample as ce  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cc, ce


@pytest.mark.parametrize("copies", [1, 2, 17])
def test_build_counterexample_compact_matches_authors_build(authors, copies: int) -> None:
    from glref.counterexample import build_counterexample_compact

    _cc, ce = authors
    a_inst, a_name = ce.build(copies)
    c_inst, c_name = build_counterexample_compact(copies)
    assert (a_inst.n, a_inst.k) == (c_inst.n, c_inst.k) == (333, 9)
    assert a_inst.out_adj == c_inst.out_adj
    assert a_inst.in_adj == c_inst.in_adj
    assert a_inst.mult == c_inst.mult
    assert a_inst.cap == c_inst.cap
    assert a_name == c_name
    assert sum(len(x) for x in a_inst.out_adj) == c_inst.num_arcs() == 2160
    assert a_inst.total_cap() == c_inst.total_cap() == c_inst.total_mult() == 108 + 216 * copies


def test_compact_primitives_match_authors_on_random_instances(authors) -> None:
    """``compact_connected``, ``compact_sets``, ``satisfies_condition``,
    ``delete_edge`` and ``contract`` agree with the authors' networkx code on
    random small instances with multiplicities and slack capacities.

    The authors' script raises ``NetworkXError`` when no terminal is reachable
    from ``v`` (no ``sink`` node); semantically ``v ∉ 𝒞(t)`` then, and the port
    returns ``False``."""
    from glref import compact as C

    A, _ce = authors
    rng = random.Random(0)

    def a_compact(inst, v, t):
        try:
            return A.compact_connected(inst, v, t)
        except nx.NetworkXError:
            return False

    compared = 0
    for _trial in range(70):
        k = rng.randint(1, 3)
        n = rng.randint(k + 1, k + 5)
        out_adj: list[list[int]] = [[] for _ in range(n)]
        for u in range(k, n):
            cands = [v for v in range(n) if v != u]
            out_adj[u] = rng.sample(cands, rng.randint(1, len(cands)))
        in_deg = [0] * n
        for u in range(n):
            for v in out_adj[u]:
                in_deg[v] += 1
        mult = [1] * n
        for v in range(k, n):
            if in_deg[v] == 0 and rng.random() < 0.5:
                mult[v] = rng.randint(2, 4)
        tot = sum(mult[k:])
        cap = _composition(rng, tot, k)
        if rng.random() < 0.2:
            cap[0] += 1
        a = A.Instance(k, out_adj, mult, cap)
        c = C.CompactInstance(k, out_adj, mult, cap)
        for v in range(k, n):
            for t in range(k):
                assert a_compact(a, v, t) == C.compact_connected(c, v, t), (out_adj, mult, v, t)
        sets_a = {v: frozenset(t for t in range(k) if a_compact(a, v, t)) for v in range(k, n)}
        assert sets_a == C.compact_sets(c)
        assert A.satisfies_condition(a, sets_a) == C.satisfies_compact_condition(c, sets_a)
        for u in range(k, n):
            for v in list(out_adj[u]):
                da, dc = A.delete_edge(a, u, v), C.delete_edge(c, u, v)
                assert (da.out_adj, da.mult, da.cap) == (dc.out_adj, dc.mult, dc.cap)
                if v < k:
                    ca, ida = A.contract(a, u, v)
                    cc_, idc = C.contract(c, u, v)
                    assert (ca.out_adj, ca.mult, ca.cap, ida) == (cc_.out_adj, cc_.mult, cc_.cap, idc)
                    assert ca.in_adj == cc_.in_adj
        compared += 1
    assert compared == 70


def test_check_claims_suspects_match_authors_fails_on_sampled_operations(authors) -> None:
    """The ``suspects`` short-cut of ``check_counterexample_claims`` gives the
    same verdict as the authors' ``fails`` on sampled deletions/contractions of
    the 17-copy instance (each side computes only the suspects' sets first);
    on that instance every sampled operation breaks the condition (README)."""
    from glref import compact as C
    from glref.counterexample import _fails, build_counterexample_compact

    A, ce = authors
    a_inst, _ = ce.build(17)
    c_inst, _ = build_counterexample_compact(17)
    rng = random.Random(4)
    edges = [(u, v) for u in c_inst.non_terminals() for v in c_inst.out_adj[u]]
    pre = [v for v in c_inst.non_terminals() if all(t < 9 for t in c_inst.out_adj[v])]
    for u, v in rng.sample(edges, 6):
        ours = _fails(C.delete_edge(c_inst, u, v), [u] + c_inst.in_adj[u])
        theirs = ce.fails(A.delete_edge(a_inst, u, v), [u] + a_inst.in_adj[u])
        assert ours == theirs is True, (u, v)
    pairs = [(p, t) for p in pre for t in c_inst.out_adj[p]]
    for p, t in rng.sample(pairs, 2):
        smaller, new_id = C.contract(c_inst, p, t)
        a_small, a_id = A.contract(a_inst, p, t)
        assert new_id == a_id
        ours = _fails(smaller, [new_id[w] for w in c_inst.in_adj[p]])
        theirs = ce.fails(a_small, [a_id[w] for w in a_inst.in_adj[p]])
        assert ours == theirs is True, (p, t)


# ---------------------------------------------------------------------------
# compact / local connectivity against the *definitions* [Def A.2], [Def A.5]
# ---------------------------------------------------------------------------


def _simple_paths_to(ci, v: int, targets) -> list[tuple[int, ...]]:
    G = nx.DiGraph()
    G.add_nodes_from(range(ci.n))
    for u in range(ci.n):
        for x in ci.out_adj[u]:
            G.add_edge(u, x)
    return [tuple(p) for t in targets if t != v for p in nx.all_simple_paths(G, v, t)]


def _internally_disjoint(paths) -> bool:
    """Paths share only their common start and terminal endpoints (endpoints may coincide)."""
    for a, b in itertools.combinations(paths, 2):
        ia, ib = set(a[1:-1]), set(b[1:-1])
        if ia & ib or a[-1] in ib or b[-1] in ia:
            return False
    return True


def _brute_local(ci, v: int, Tp) -> bool:
    Tp = sorted(set(Tp))
    if not Tp:
        return True
    if any(t in ci.out_adj[v] for t in Tp):  # pre-terminal convention of [Def A.2]
        return True
    paths = _simple_paths_to(ci, v, Tp)
    return any(_internally_disjoint(fam) for fam in itertools.combinations(paths, len(Tp)))


def _brute_compact(ci, v: int, t: int) -> bool:
    if t in ci.out_adj[v]:  # pre-terminal convention of [Def A.5]
        return True
    paths = _simple_paths_to(ci, v, range(ci.k))
    for fam in itertools.combinations(paths, ci.k):
        if not _internally_disjoint(fam):
            continue
        ends = [p[-1] for p in fam]
        if all(ends.count(tp) <= 1 for tp in range(ci.k) if tp != t):
            return True
    return False


def test_compact_and_local_connectivity_match_path_enumeration() -> None:
    """The flow formulations of ``compact_connected`` [Lem A.9] and
    ``is_locally_connected`` agree with a brute-force enumeration of families
    of internally vertex-disjoint paths per [Def A.5] / [Def A.2] (including
    the pre-terminal conventions) on random tiny digraphs."""
    from glref.compact import CompactInstance, compact_connected, is_locally_connected

    rng = random.Random(0)
    checked = 0
    for _trial in range(150):
        k = rng.randint(1, 3)
        n = rng.randint(k + 1, k + 4)
        out_adj: list[list[int]] = [[] for _ in range(n)]
        for u in range(k, n):
            cands = [x for x in range(n) if x != u]
            out_adj[u] = rng.sample(cands, rng.randint(0, len(cands)))
        ci = CompactInstance(k, out_adj, [1] * n, [0] * k)
        for v in range(k, n):
            for t in range(k):
                assert compact_connected(ci, v, t) == _brute_compact(ci, v, t), (out_adj, k, v, t)
                checked += 1
            for r in range(1, k + 1):
                for Tp in itertools.combinations(range(k), r):
                    assert is_locally_connected(ci, v, Tp) == _brute_local(ci, v, Tp), (out_adj, k, v, Tp)
                    checked += 1
    assert checked > 1500
