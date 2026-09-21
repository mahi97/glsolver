"""Tests for the reference DAG algorithm (glref.dag): [Alg 5] GLDAGPartition,
[Lem 9.1] precondition, [Def 9.2] canonical order; paper_notes §9, §13.7."""
from __future__ import annotations

import random
from collections import deque

import pytest

from glref.dag import (
    POLICIES,
    canonical_topological_order,
    dag_precondition,
    gl_dag_partition,
    is_dag,
    is_k_t_connected_dag,
)
from glref.trace import Tracer
from glref.unweighted import InvariantError
from glsolver.instance import Instance, make_instance

try:
    from glsolver.verify import verify_instance_parts as _verify_instance_parts
except ImportError:  # pragma: no cover
    _verify_instance_parts = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _local_check(inst: Instance, parts: list[list[int]]) -> list[str]:
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
    arcs = set(inst.arcs)
    part_of = {v: i for i, part in enumerate(parts) for v in part}
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        p = parents[v]
        assert (v, p) in arcs and part_of[v] == part_of[p]
        x, hops = v, 0
        while x not in tset:
            x = parents[x]
            hops += 1
            assert hops <= inst.n
        assert x == inst.terminals[part_of[v]]


def random_k_t_connected_dag(rng: random.Random, n: int, k: int, extra: int = 3) -> tuple[list[tuple[int, int]], list[int]]:
    """Random order; each non-terminal picks >= k later vertices/terminals [Lem 9.1]."""
    verts = list(range(n))
    terminals = rng.sample(verts, k)
    nonterms = [v for v in verts if v not in terminals]
    rng.shuffle(nonterms)
    order = nonterms + terminals
    arcs: list[tuple[int, int]] = []
    for i, v in enumerate(nonterms):
        later = order[i + 1 :]
        d = rng.randint(k, min(len(later), k + extra))
        for x in rng.sample(later, d):
            arcs.append((v, x))
    return sorted(arcs), terminals


def capacities_for(rng: random.Random, mode: str, total: int, k: int) -> list[int]:
    if mode == "balanced":
        base, rem = divmod(total, k)
        return [base + (1 if i < rem else 0) for i in range(k)]
    if mode == "unbalanced":
        cuts = sorted(rng.randint(0, total) for _ in range(k - 1))
        return [b - a for a, b in zip([0] + cuts, cuts + [total])]
    if mode == "extreme":
        caps = [0] * k
        caps[rng.randrange(k)] = total
        return caps
    raise ValueError(mode)


def paper_dag_instance(caps: tuple[int, int, int]) -> Instance:
    """Figure k-conn-dag-1step: t1,t2,t3 = 0,1,2; v4..v8 = 3..7."""
    arcs = [(7, 6), (6, 5), (5, 4), (4, 3), (3, 0), (7, 5), (7, 4), (4, 0), (4, 1), (5, 2), (5, 3), (6, 3), (3, 1), (3, 2), (6, 1)]
    return make_instance(8, arcs, [0, 1, 2], list(caps), directed=True, name="k-conn-dag-1step")


# ---------------------------------------------------------------------------
# structure: order and precondition
# ---------------------------------------------------------------------------


def test_canonical_order_and_precondition_on_paper_instance() -> None:
    inst = paper_dag_instance((2, 2, 1))
    assert is_dag(inst)
    assert dag_precondition(inst) == (True, None)
    assert is_k_t_connected_dag(inst)
    order = canonical_topological_order(inst)
    assert order[-3:] == [0, 1, 2]  # terminals last, in inst.terminals order
    pos = {v: i for i, v in enumerate(order)}
    for u, v in inst.arcs:
        assert pos[u] < pos[v]
    assert set(order) == set(range(8))
    # the figure's horizontal order v8 ≺ v7 ≺ v6 ≺ v5 ≺ v4 is the unique one here
    assert order[:5] == [7, 6, 5, 4, 3]


def test_precondition_failures_detected() -> None:
    # a non-terminal with out-degree k - 1
    inst = make_instance(4, [(2, 0), (2, 1), (3, 2)], [0, 1], [1, 1], directed=True)
    assert is_dag(inst)
    ok, bad = dag_precondition(inst)
    assert not ok and bad == 3 and not is_k_t_connected_dag(inst)
    res = gl_dag_partition(inst)
    assert res.status == "precondition_failed" and "out-degree 1 < k = 2" in res.message and "3" in res.message
    # a cycle, all out-degrees >= k
    cyc = make_instance(
        5, [(2, 3), (3, 4), (4, 2), (2, 0), (3, 0), (4, 1), (2, 1), (3, 1), (4, 0)], [0, 1], [2, 1], directed=True
    )
    assert not is_dag(cyc)
    ok, bad = dag_precondition(cyc)
    assert not ok and bad in (2, 3, 4)
    with pytest.raises(ValueError):
        canonical_topological_order(cyc)
    res = gl_dag_partition(cyc, debug=True)
    assert res.status == "precondition_failed" and "acyclic" in res.message
    # policy validation
    with pytest.raises(ValueError):
        gl_dag_partition(paper_dag_instance((2, 2, 1)), policy="bogus")


# ---------------------------------------------------------------------------
# the paper's figure instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caps", [(2, 2, 1), (1, 2, 2), (5, 0, 0)])
@pytest.mark.parametrize("policy", POLICIES)
def test_paper_dag_instance_partitioned(caps: tuple[int, int, int], policy: str) -> None:
    inst = paper_dag_instance(caps)
    res = gl_dag_partition(inst, policy=policy, debug=True)
    assert res.status == "ok"
    assert_valid(inst, res.parts)
    assert_certificate(inst, res.parts, res.parents)
    assert [len(p) - 1 for p in res.parts] == list(caps)
    assert res.stats.contractions == 5
    assert res.stats.heap_pushes == inst.m  # every arc is pushed exactly once [Lem 9.6]
    assert res.stats.heap_pops <= res.stats.heap_pushes


def test_paper_figure_step_v7_into_t2() -> None:
    """Figure caption: H_2 = <v7, v5, v4>; with t2 active v7 is contracted into
    t2, then v8 is inserted and H_2 = <v8, v5, v4>."""
    inst = paper_dag_instance((0, 5, 0))
    tracer = Tracer()
    res = gl_dag_partition(inst, policy="first", tracer=tracer, debug=True)
    assert res.status == "ok"
    steps = [e for e in tracer.events if e["type"] == "dag_contract"]
    assert steps[0]["p"] == 6 and steps[0]["t"] == 1 and steps[0]["parent"] == 1  # v7 -> t2
    assert steps[1]["p"] == 7 and steps[1]["t"] == 1 and steps[1]["parent"] == 6  # v8 via arc (v8, v7)
    # then v5 (pos 3) whose in-neighbour v6 (pos 2) now precedes v4 (pos 4); finally v4
    assert [s["p"] for s in steps] == [6, 7, 4, 5, 3]
    assert res.parts[1] == [1, 6, 7, 4, 5, 3]
    assert res.stats.stale_pops >= 1  # v8 is re-pushed as an in-neighbour of v5/v6 after use


# ---------------------------------------------------------------------------
# random k-T-connected DAGs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_random_unweighted_dags(policy: str) -> None:
    rng = random.Random(100 + POLICIES.index(policy))
    for trial in range(60):
        k = rng.randint(1, 5)
        n = rng.randint(k + 1, 60)
        arcs, terminals = random_k_t_connected_dag(rng, n, k)
        mode = ("balanced", "unbalanced", "extreme")[trial % 3]
        caps = capacities_for(rng, mode, n - k, k)
        inst = make_instance(n, arcs, terminals, caps, directed=True, name=f"dag_{trial}")
        assert dag_precondition(inst) == (True, None)
        res = gl_dag_partition(inst, policy=policy, debug=True)
        assert res.status == "ok", res.message
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        assert [len(p) - 1 for p in res.parts] == caps
        assert res.stats.heap_pushes == inst.m
        assert res.stats.contractions == n - k


@pytest.mark.parametrize("policy", POLICIES)
def test_random_weighted_dags(policy: str) -> None:
    rng = random.Random(200 + POLICIES.index(policy))
    exceeded = 0
    for trial in range(60):
        k = rng.randint(1, 5)
        n = rng.randint(k + 1, 60)
        arcs, terminals = random_k_t_connected_dag(rng, n, k)
        w_max = rng.randint(1, 6)
        weights = [0 if v in terminals else rng.randint(1, w_max) for v in range(n)]
        total = sum(weights)
        slack = rng.choice([0, 0, rng.randint(1, 2 * w_max)])
        mode = ("balanced", "unbalanced", "extreme")[trial % 3]
        caps = capacities_for(rng, mode, total + slack, k)
        inst = make_instance(n, arcs, terminals, caps, weights=weights, directed=True, name=f"wdag_{trial}")
        res = gl_dag_partition(inst, policy=policy, debug=True)
        assert res.status == "ok", res.message
        assert_valid(inst, res.parts)
        assert_certificate(inst, res.parts, res.parents)
        tset = set(terminals)
        for i, part in enumerate(res.parts):
            wt = sum(weights[v] for v in part if v not in tset)
            assert wt <= caps[i] + inst.w_max - 1
            if wt > caps[i]:
                exceeded += 1
            if caps[i] == 0:
                assert part == [terminals[i]]
    assert exceeded > 0, "weighted overflow up to w_max - 1 never exercised"


def test_debug_invariants_pass_on_dense_dags() -> None:
    """Dense DAGs create many stale heap entries; the [Lem 9.4] check must hold."""
    rng = random.Random(9)
    for _ in range(10):
        k = rng.randint(2, 4)
        n = rng.randint(10, 30)
        arcs, terminals = random_k_t_connected_dag(rng, n, k, extra=12)
        caps = capacities_for(rng, "unbalanced", n - k, k)
        inst = make_instance(n, arcs, terminals, caps, directed=True)
        res = gl_dag_partition(inst, policy="max_residual", debug=True)
        assert res.status == "ok"
        assert_valid(inst, res.parts)
        assert res.stats.stale_pops > 0


def test_heap_never_runs_dry_and_policies_are_deterministic() -> None:
    rng = random.Random(77)
    arcs, terminals = random_k_t_connected_dag(rng, 25, 3)
    inst = make_instance(25, arcs, terminals, capacities_for(rng, "balanced", 22, 3), directed=True)
    for policy in POLICIES:
        a = gl_dag_partition(inst, policy=policy)
        b = gl_dag_partition(inst, policy=policy)
        assert a.parts == b.parts and a.parents == b.parents
    assert issubclass(InvariantError, AssertionError)
