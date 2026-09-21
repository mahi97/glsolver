"""Tests for the reference weighted algorithm (glref.weighted, glref.mincostflow):
[Alg 3] GLWeightedPartition, [Alg 4] RoundAndRemove, [Prop 5.4] min-cost split
assignment; paper_notes §5, §8, §13.6, §13.12."""
from __future__ import annotations

import random
from collections import deque

import networkx as nx
import pytest

from glref.essential import all_essential
from glref.graph import DiGraphState
from glref.mincostflow import MinCostFlow
from glref.trace import Tracer
from glref.unweighted import InvariantError, gl_partition
from glref.weighted import (
    gl_weighted_partition,
    is_split_witness,
    min_cost_split_assignment,
    round_and_remove,
    split_potential,
)
from glsolver.instance import Instance, make_instance

try:  # the independent verifier is written by another agent; fall back to a local checker
    from glsolver.verify import verify_instance_parts as _verify_instance_parts
except ImportError:  # pragma: no cover
    _verify_instance_parts = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _local_check(inst: Instance, parts: list[list[int]]) -> list[str]:
    """Minimal independent checker (BFS on the original arcs) used only when
    glsolver.verify is unavailable."""
    errors: list[str] = []
    n, k = inst.n, inst.k
    if len(parts) != k:
        return [f"expected {k} parts, got {len(parts)}"]
    seen: dict[int, int] = {}
    for i, part in enumerate(parts):
        for v in part:
            if v in seen:
                errors.append(f"vertex {v} in parts {seen[v]} and {i}")
            seen[v] = i
    for v in range(n):
        if v not in seen:
            errors.append(f"vertex {v} missing")
    tset = set(inst.terminals)
    w = inst.weights
    w_max = inst.w_max
    in_adj = inst.in_adjacency()
    for i, part in enumerate(parts):
        t = inst.terminals[i]
        pset = set(part)
        if t not in pset:
            errors.append(f"terminal {t} not in part {i}")
            continue
        if any(x in tset and x != t for x in pset):
            errors.append(f"foreign terminal in part {i}")
        if w is None:
            if len(pset) != inst.capacities[i] + 1:
                errors.append(f"part {i} has size {len(pset)} != {inst.capacities[i] + 1}")
        else:
            wt = sum(w[v] for v in pset if v not in tset)
            if wt > inst.capacities[i] + w_max - 1:
                errors.append(f"part {i} weight {wt} exceeds bound")
        reached = {t}
        dq = deque([t])
        while dq:
            x = dq.popleft()
            for u in in_adj[x]:
                if u in pset and u not in reached:
                    reached.add(u)
                    dq.append(u)
        if reached != pset:
            errors.append(f"part {i} not connected to {t}: {sorted(pset - reached)}")
    return errors


def assert_valid(inst: Instance, parts: list[list[int]]) -> None:
    if _verify_instance_parts is not None:
        report = _verify_instance_parts(inst, parts)
        assert report.valid, report.errors
    else:  # pragma: no cover
        assert _local_check(inst, parts) == []


def assert_certificate(inst: Instance, parts: list[list[int]], parents: dict[int, int]) -> None:
    """The parents form an in-arborescence of original arcs inside each part."""
    arcs = set(inst.arcs)
    part_of = {v: i for i, part in enumerate(parts) for v in part}
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        assert v in parents, f"no parent recorded for non-terminal {v}"
        p = parents[v]
        assert (v, p) in arcs, f"parent arc ({v},{p}) is not an original arc"
        assert part_of[v] == part_of[p], f"parent of {v} lies in another part"
    for v in range(inst.n):
        if v in tset:
            continue
        x, hops = v, 0
        while x not in tset:
            x = parents[x]
            hops += 1
            assert hops <= inst.n, "parent pointers do not lead to the terminal"
        assert x == inst.terminals[part_of[v]]


def paper_running_example() -> Instance:
    """The paper's running example (figure complete-example): t1,t2,t3 = 0,1,2,
    v4..v9 = 3..8, capacities (2, 2, 2)."""
    arcs = [(7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1), (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2)]
    return make_instance(9, arcs, [0, 1, 2], [2, 2, 2], directed=True, name="paper_running_example")


def random_composition(rng: random.Random, total: int, parts: int) -> list[int]:
    """Nonnegative integers summing to ``total``."""
    cuts = sorted(rng.randint(0, total) for _ in range(parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def random_k_connected_undirected(rng: random.Random, n: int, k: int) -> nx.Graph:
    """Rejection sampling of a ``k``-vertex-connected undirected graph on ``n`` nodes."""
    while True:
        p = rng.uniform(0.35, 0.9)
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(1 << 30))
        if nx.is_connected(G) and nx.node_connectivity(G) >= k:
            return G


def random_k_t_connected_digraph(rng: random.Random, n: int, k: int, extra_back: int = 3) -> tuple[list[tuple[int, int]], list[int]]:
    """A ``k``-``T``-connected digraph: a random DAG in which every non-terminal
    has ``≥ k`` arcs to later vertices [Lem 9.1], plus random backward arcs
    (adding arcs never decreases ``κ``)."""
    verts = list(range(n))
    terminals = rng.sample(verts, k)
    nonterms = [v for v in verts if v not in terminals]
    rng.shuffle(nonterms)
    order = nonterms + terminals
    pos = {v: i for i, v in enumerate(order)}
    arcs: set[tuple[int, int]] = set()
    for i, v in enumerate(nonterms):
        later = order[i + 1 :]
        d = rng.randint(k, min(len(later), k + 2))
        for x in rng.sample(later, d):
            arcs.add((v, x))
    for _ in range(extra_back):
        u, v = rng.sample(nonterms, 2) if len(nonterms) >= 2 else (None, None)
        if u is not None and pos[u] > pos[v]:
            arcs.add((u, v))
    return sorted(arcs), terminals


def unweighted_instances(seed: int = 7, count: int = 40) -> list[Instance]:
    rng = random.Random(seed)
    out: list[Instance] = [paper_running_example()]
    for i in range(count):
        n = rng.randint(5, 12)
        k = rng.randint(1, min(4, n - 1))
        if i % 3 == 2:
            arcs, terminals = random_k_t_connected_digraph(rng, n, k)
            caps = random_composition(rng, n - k, k)
            out.append(make_instance(n, arcs, terminals, caps, directed=True, name=f"digraph_{i}"))
        else:
            G = random_k_connected_undirected(rng, n, k)
            terminals = rng.sample(range(n), k)
            caps = random_composition(rng, n - k, k)
            out.append(make_instance(n, list(G.edges()), terminals, caps, directed=False, name=f"undirected_{i}"))
    return out


def weighted_instances(seed: int = 11, count: int = 30) -> list[Instance]:
    rng = random.Random(seed)
    out: list[Instance] = []
    for i in range(count):
        n = rng.randint(5, 11)
        k = rng.randint(1, min(4, n - 1))
        w_max = rng.randint(2, 5)
        if i % 2:
            arcs, terminals = random_k_t_connected_digraph(rng, n, k)
            directed = True
        else:
            G = random_k_connected_undirected(rng, n, k)
            arcs = list(G.edges())
            terminals = rng.sample(range(n), k)
            directed = False
        weights = [0 if v in terminals else rng.randint(1, w_max) for v in range(n)]
        total = sum(weights)
        slack = rng.choice([0, 0, rng.randint(1, w_max)])
        caps = random_composition(rng, total + slack, k)
        out.append(
            make_instance(n, arcs, terminals, caps, weights=weights, directed=directed, name=f"weighted_{i}")
        )
    return out


# ---------------------------------------------------------------------------
# MinCostFlow [Prop 5.4 building block]
# ---------------------------------------------------------------------------


def test_min_cost_flow_matches_networkx_on_random_networks() -> None:
    rng = random.Random(3)
    for trial in range(40):
        n = rng.randint(4, 9)
        net = MinCostFlow(n)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        arcs = []
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.45:
                    cap, cost = rng.randint(1, 6), rng.randint(0, 5)
                    arcs.append((u, v, net.add_arc(u, v, cap, cost)))
                    G.add_edge(u, v, capacity=cap, weight=cost)
        s, z = 0, n - 1
        max_value = nx.maximum_flow_value(G, s, z) if G.number_of_edges() else 0
        value, cost = net.min_cost_flow(s, z, max_value)
        assert value == max_value
        if max_value > 0:
            ref = nx.max_flow_min_cost(G, s, z)
            assert cost == nx.cost_of_flow(G, ref), (trial, cost)
        # integrality, capacity and conservation of the returned flow
        balance = [0] * n
        for u, v, idx in arcs:
            f = net.flow(idx)
            assert 0 <= f <= G[u][v]["capacity"]
            balance[u] -= f
            balance[v] += f
        for x in range(n):
            if x not in (s, z):
                assert balance[x] == 0
        assert balance[z] == value == -balance[s]
        assert net.total_cost() == cost
        # asking for more than the max flow returns the max flow, not an error
        net2 = MinCostFlow(2)
        net2.add_arc(0, 1, 3, 2)
        assert net2.min_cost_flow(0, 1, 10) == (3, 6)


def test_min_cost_split_assignment_is_optimal_and_a_witness() -> None:
    """[Prop 5.4] on the running example: the flow is a split witness and its
    potential is minimal among all witnesses (brute force over unit assignments)."""
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = dict(zip(inst.terminals, inst.capacities))
    w = {v: 1 for v in g.nonterminals()}
    # cost table: pretend the arcs (3,1) and (5,6) are secondary and use random criticality sets
    rng = random.Random(5)
    table = [{v: {t for t in ess[v] if rng.random() < 0.5} for v in g.nonterminals()} for _ in range(3)]

    def xi(v: int, t: int) -> int:
        return sum(1 for row in table if t in row[v])

    psi = min_cost_split_assignment(g, ess, cap, w, xi)
    assert psi is not None
    ok, why = is_split_witness(g, psi, ess, cap, w)
    assert ok, why
    best = split_potential(table, psi)
    # brute force: every unit assignment phi with phi(v) in Ess(v) and exact capacities
    import itertools

    nonterms = g.nonterminals()
    options = [sorted(ess[v]) for v in nonterms]
    brute = None
    for choice in itertools.product(*options):
        counts = {t: 0 for t in inst.terminals}
        for t in choice:
            counts[t] += 1
        if any(counts[t] > cap[t] for t in counts):
            continue
        val = sum(xi(v, t) for v, t in zip(nonterms, choice))
        brute = val if brute is None else min(brute, val)
    assert brute is not None and best == brute
    # infeasible capacities -> None
    tight = dict(cap)
    tight[inst.terminals[0]] = 0
    assert min_cost_split_assignment(g, ess, tight, w) is None


def test_is_split_witness_rejects_bad_assignments() -> None:
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = dict(zip(inst.terminals, inst.capacities))
    w = {v: 1 for v in g.nonterminals()}
    psi = min_cost_split_assignment(g, ess, cap, w)
    assert psi is not None and is_split_witness(g, psi, ess, cap, w)[0]
    bad = dict(psi)
    (v, t), _ = next(iter(bad.items()))
    other = next(x for x in inst.terminals if x not in ess[v])
    del bad[(v, t)]
    bad[(v, other)] = 1
    ok, why = is_split_witness(g, bad, ess, cap, w)
    assert not ok and "not essential" in why
    short = dict(psi)
    del short[(v, t)]
    ok, why = is_split_witness(g, short, ess, cap, w)
    assert not ok and "sends" in why
    over = dict(psi)
    over[(v, t)] = 5
    ok, _ = is_split_witness(g, over, ess, cap, w)
    assert not ok


# ---------------------------------------------------------------------------
# (a) unweighted instances: exact partition, no rounding [§13.6]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["first", "last"])
def test_unweighted_instances_exact_and_no_rounding(policy: str) -> None:
    instances = unweighted_instances()
    assert len(instances) >= 40
    seen_deletions = 0
    for inst in instances:
        res = gl_weighted_partition(inst, debug=True, policy=policy)
        assert res.status == "ok", (inst.name, res.message)
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        assert res.stats.roundings == 0, f"{inst.name}: rounding on a unit-weight tight instance (§13.6)"
        assert res.stats.contractions == inst.num_nonterminals
        assert res.stats.min_cost_flow_calls >= 1
        seen_deletions += res.stats.deletions
    assert seen_deletions > 0, "step (iii) never exercised"


def test_running_example_matches_unweighted_solver_shape() -> None:
    inst = paper_running_example()
    tracer = Tracer()
    res = gl_weighted_partition(inst, debug=True, tracer=tracer)
    assert res.status == "ok"
    assert_valid(inst, res.parts)
    assert [len(p) for p in res.parts] == [3, 3, 3]
    types = [e["type"] for e in tracer.events]
    assert types[0] == "init" and types[-1] == "done"
    assert "min_cost_split" in types and "delete_arc" in types and "contract" in types
    # the unweighted reference agrees that a partition exists (both are exact here)
    ref = gl_partition(inst)
    assert ref.status == "ok"
    assert_valid(inst, ref.parts)


# ---------------------------------------------------------------------------
# (b) weighted instances: bound c_t + w_max - 1, debug clean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["first", "last"])
def test_weighted_random_instances_respect_bound(policy: str) -> None:
    for inst in weighted_instances():
        res = gl_weighted_partition(inst, debug=True, policy=policy)
        assert res.status == "ok", (inst.name, res.message)
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        w = inst.weights
        assert w is not None
        tset = set(inst.terminals)
        for i, part in enumerate(res.parts):
            wt = sum(w[v] for v in part if v not in tset)
            assert wt <= inst.capacities[i] + inst.w_max - 1, (inst.name, i, wt)
            assert inst.terminals[i] in part
        # every non-terminal is placed exactly once
        assert sorted(v for part in res.parts for v in part) == list(range(inst.n))


# ---------------------------------------------------------------------------
# (c) the paper's unavoidable-slack example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5])
def test_unavoidable_slack_example(k: int) -> None:
    """k terminals of capacity 1 and one vertex of weight k adjacent to all of
    them: some part must receive weight k = c_t + w_max - 1 (paper §5)."""
    n = k + 1
    v = k
    arcs = [(v, t) for t in range(k)]
    weights = [0] * k + [k]
    inst = make_instance(n, arcs, list(range(k)), [1] * k, weights=weights, directed=True)
    res = gl_weighted_partition(inst, debug=True)
    assert res.status == "ok"
    assert_valid(inst, res.parts)
    heavy = [i for i, part in enumerate(res.parts) if v in part]
    assert len(heavy) == 1
    i = heavy[0]
    assert sum(weights[x] for x in res.parts[i] if x != i) == k == inst.capacities[i] + inst.w_max - 1
    assert res.stats.roundings == 1  # no saturating matching from {v} to T for k >= 2
    assert res.parents[v] == i


# ---------------------------------------------------------------------------
# (d) rounding is exercised by a random search
# ---------------------------------------------------------------------------


def test_random_search_finds_rounding_instances() -> None:
    rng = random.Random(2024)
    roundings_seen = 0
    tried = 0
    for _ in range(120):
        n = rng.randint(4, 8)
        k = rng.randint(2, min(4, n - 1))
        arcs, terminals = random_k_t_connected_digraph(rng, n, k, extra_back=1)
        w_max = rng.randint(3, 6)
        weights = [0 if v in terminals else rng.randint(1, w_max) for v in range(n)]
        # one heavy vertex adjacent to every terminal so it must split its weight
        heavy = next(v for v in range(n) if v not in terminals)
        weights[heavy] = w_max
        arcs = sorted(set(arcs) | {(heavy, t) for t in terminals})
        caps = random_composition(rng, sum(weights), k)
        inst = make_instance(n, arcs, terminals, caps, weights=weights, directed=True)
        res = gl_weighted_partition(inst, debug=True)
        assert res.status == "ok", res.message
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        tried += 1
        roundings_seen += res.stats.roundings
    assert tried == 120
    assert roundings_seen > 0, "RoundAndRemove never triggered in the random search"


def test_round_and_remove_directly() -> None:
    """[Alg 4] on the slack example: S is a minimal Hall-deficient set, the
    matched pre-terminal is PT(G,S), and S ∪ PT(G,S) is deleted."""
    from glref.unweighted import RefStats

    k = 3
    v = k
    arcs = [(v, t) for t in range(k)]
    inst = make_instance(k + 1, arcs, list(range(k)), [1] * k, weights=[0] * k + [k], directed=True)
    g = DiGraphState.from_instance(inst)
    cap = dict(zip(inst.terminals, inst.capacities))
    w = {v: k}
    stats = RefStats()
    parents: dict[int, int] = {}
    parts = round_and_remove(g, cap, w, stats, Tracer(enabled=False), debug=True, parents=parents)
    assert len(parts) == 2  # a 2-element deficient set (|PT| = 1)
    S = sorted(parts)
    t_S = next(t for t in S if parts[t] == [t])
    other = next(t for t in S if t != t_S)
    assert parts[other] == [other, v] and parents[v] == other
    assert not g.live[v] and all(not g.live[t] for t in S)
    assert g.terminals == [t for t in range(k) if t not in S]
    assert stats.roundings == 1


# ---------------------------------------------------------------------------
# (e) FESAC failure
# ---------------------------------------------------------------------------


def test_fesac_failure_reports_precondition_failed() -> None:
    # v reaches only t1 (Ess(v) = {t1}) but c_{t1} = 0: no split witness
    inst = make_instance(3, [(2, 0)], [0, 1], [0, 1], directed=True)
    res = gl_weighted_partition(inst, debug=True)
    assert res.status == "precondition_failed"
    assert "Split-Assignment" in res.message
    assert res.parts == [] and res.parents == {}
    # a vertex that reaches no terminal at all (κ = 0, Ess = ∅) [§13.9]
    inst2 = make_instance(4, [(2, 0), (3, 2), (2, 3)], [0, 1], [1, 1], directed=True)
    res2 = gl_weighted_partition(inst2)
    assert res2.status == "precondition_failed"
    # weighted variant: total weight fits but only into a non-essential terminal
    inst3 = make_instance(3, [(2, 0)], [0, 1], [1, 5], weights=[0, 0, 3], directed=True)
    res3 = gl_weighted_partition(inst3)
    assert res3.status == "precondition_failed"
    # unit weights on a FEAC instance must agree with the unweighted solver's verdict
    inst4 = paper_running_example()
    assert gl_weighted_partition(inst4).status == gl_partition(inst4).status == "ok"


def test_invalid_policy_rejected() -> None:
    with pytest.raises(ValueError):
        gl_weighted_partition(paper_running_example(), policy="random")


def test_invariant_error_is_assertion_error() -> None:
    assert issubclass(InvariantError, AssertionError)
