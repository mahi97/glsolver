"""Tests for the C++ DAG solver ``glsolver._core.dag_partition`` ([Alg 5] GLDAGPartition,
docs/paper_notes.md §9; heap variant 0 and the linear stack variant 1 of RESEARCH_NOTES.md P1).

Properties asserted (every test names the paper item it checks):

* every output partition is accepted by the independent verifier
  ``glsolver.verify.verify_instance_parts`` and the ``parent`` array is an
  in-arborescence of original arcs inside each part (paper_notes §9);
* variant 0 (heaps) and variant 1 (stack) return *identical* assignments and
  parents for every policy (P1);
* the output equals the reference ``glref.dag.gl_dag_partition`` for the same
  policy (parts and parents);
* precondition failures [Lem 9.1] (out-degree < k, directed cycle) are reported
  with the offending vertex;
* the trace contains one ``dag_contract`` event per non-terminal, consistent
  with the returned parents.
"""
from __future__ import annotations

import random
import re
import time

import pytest

from glref.dag import gl_dag_partition
from glsolver import _core
from glsolver.generators import layered_dag, random_kT_connected_dag, weighted_variant
from glsolver.instance import Instance, make_instance
from glsolver.verify import verify_instance_parts

POLICY_NAMES = {0: "max_residual", 1: "round_robin", 2: "first"}
MODES = ("balanced", "unbalanced", "random", "extreme")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_core(inst: Instance, policy: int, variant: int, **kw) -> dict:
    return _core.dag_partition(
        inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities),
        None if inst.weights is None else list(inst.weights), policy, variant, **kw,
    )


def parts_of(inst: Instance, assignment: list[int]) -> list[list[int]]:
    parts: list[list[int]] = [[] for _ in range(inst.k)]
    for v, i in enumerate(assignment):
        assert 0 <= i < inst.k, f"vertex {v} unassigned ({i})"
        parts[i].append(v)
    return parts


def check_parents(inst: Instance, assignment: list[int], parent: list[int]) -> None:
    """The arcs (p, parent[p]) are original arcs into the same part and form an in-arborescence
    rooted at the terminal (every non-terminal reaches its terminal along parent pointers)."""
    arcs = set(inst.arcs)
    tset = set(inst.terminals)
    assert len(parent) == inst.n and len(assignment) == inst.n
    for v in range(inst.n):
        if v in tset:
            assert parent[v] == -1
            continue
        p = parent[v]
        assert p >= 0, f"non-terminal {v} has no parent"
        assert (v, p) in arcs, f"parent arc ({v},{p}) is not an original arc"
        assert assignment[p] == assignment[v], f"parent of {v} lies in another part"
    # acyclicity / reachability of the root along parent pointers
    for v in range(inst.n):
        x, steps = v, 0
        while x not in tset:
            x = parent[x]
            steps += 1
            assert steps <= inst.n, "parent pointers contain a cycle"
        assert x == inst.terminals[assignment[v]]


def assert_valid(inst: Instance, res: dict) -> list[list[int]]:
    assert res["status"] == "ok", res["message"]
    parts = parts_of(inst, res["assignment"])
    report = verify_instance_parts(inst, parts)
    assert report.valid, report.errors
    check_parents(inst, res["assignment"], res["parent"])
    return parts


def random_instance(seed: int, n_max: int = 60) -> Instance:
    rng = random.Random(seed)
    k = rng.randint(1, 8)
    mode = rng.choice(MODES)
    weighted = rng.random() < 0.5
    if rng.random() < 0.5:
        n = rng.randint(k + 1, n_max)
        return random_kT_connected_dag(
            n, k, seed, mode, extra_out=rng.randint(0, 3), weighted=weighted, w_max=rng.randint(1, 5)
        )
    layers, width = rng.randint(1, 6), rng.randint(1, max(1, n_max // 6))
    return layered_dag(layers, width, k, seed, mode, weighted=weighted, w_max=rng.randint(1, 4))


# ---------------------------------------------------------------------------
# [Alg 5] validity, P1 equivalence, agreement with the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_dag_partition_valid_and_variants_identical(seed: int) -> None:
    """[Alg 5]/[Lem 9.5]: both variants and all policies give verifier-accepted partitions
    with identical assignments and parents (P1), equal to glref.dag for the same policy."""
    inst = random_instance(seed)
    for policy in (0, 1, 2):
        ref = gl_dag_partition(inst, policy=POLICY_NAMES[policy])
        assert ref.status == "ok", ref.message
        results = [run_core(inst, policy, variant) for variant in (0, 1)]
        for res in results:
            parts = assert_valid(inst, res)
            assert [sorted(p) for p in parts] == [sorted(p) for p in ref.parts]
            parents = {v: p for v, p in enumerate(res["parent"]) if p >= 0}
            assert parents == ref.parents
            assert res["stats"]["contractions"] == inst.n - inst.k
            assert res["stats"]["time_seconds"]["dag"] >= 0.0
        assert results[0]["assignment"] == results[1]["assignment"]
        assert results[0]["parent"] == results[1]["parent"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("weighted", [False, True])
def test_dag_partition_all_capacity_modes(mode: str, weighted: bool) -> None:
    """All capacity modes, unit and random weights, larger n and k up to 8."""
    for seed in range(4):
        k = 2 + 2 * (seed % 4)
        inst = random_kT_connected_dag(300 + 100 * seed, k, seed, mode, weighted=weighted, w_max=6)
        for policy in (0, 1, 2):
            r0 = run_core(inst, policy, 0)
            r1 = run_core(inst, policy, 1)
            assert_valid(inst, r0)
            assert r0["assignment"] == r1["assignment"] and r0["parent"] == r1["parent"]


def test_dag_partition_medium_matches_reference() -> None:
    """n up to 2000: verifier-accepted, variants identical, equal to the reference (policy 0)."""
    for seed, (n, k) in enumerate([(2000, 8), (1500, 5), (1200, 3)]):
        inst = random_kT_connected_dag(n, k, 100 + seed, "random", extra_out=2)
        if seed == 1:
            inst = weighted_variant(inst, seed, 7, slack=3)
        ref = gl_dag_partition(inst, policy="max_residual")
        r0 = run_core(inst, 0, 0)
        r1 = run_core(inst, 0, 1)
        parts = assert_valid(inst, r0)
        assert [sorted(p) for p in parts] == [sorted(p) for p in ref.parts]
        assert {v: p for v, p in enumerate(r0["parent"]) if p >= 0} == ref.parents
        assert r0["assignment"] == r1["assignment"] and r0["parent"] == r1["parent"]


def test_dag_partition_layered_all_policies() -> None:
    for seed in range(6):
        inst = layered_dag(8, 25, 4 + seed % 5, seed, MODES[seed % 4], weighted=seed % 2 == 1)
        for policy in (0, 1, 2):
            ref = gl_dag_partition(inst, policy=POLICY_NAMES[policy])
            out = [run_core(inst, policy, v) for v in (0, 1)]
            parts = assert_valid(inst, out[0])
            assert [sorted(p) for p in parts] == [sorted(p) for p in ref.parts]
            assert out[0]["assignment"] == out[1]["assignment"]
            assert out[0]["parent"] == out[1]["parent"]


def test_k_equals_one_and_zero_capacities() -> None:
    """§13.10 k = 1; capacity-0 terminals stay singletons; weighted parts may exceed c_t by < w_max."""
    inst = random_kT_connected_dag(30, 1, 5)
    res = run_core(inst, 0, 1)
    parts = assert_valid(inst, res)
    assert sorted(parts[0]) == list(range(30))
    # extreme mode: all capacity on one terminal, the others have c_t = 0 and remain {t}
    inst = random_kT_connected_dag(40, 5, 6, "extreme")
    for variant in (0, 1):
        parts = assert_valid(inst, run_core(inst, 0, variant))
        for i, c in enumerate(inst.capacities):
            assert len(parts[i]) == c + 1
    # weighted: bound c_t + w_max - 1 [Thm weighted-k-t-conn]
    inst = random_kT_connected_dag(60, 4, 7, "balanced", weighted=True, w_max=5)
    parts = assert_valid(inst, run_core(inst, 1, 1))
    tset = set(inst.terminals)
    for i, part in enumerate(parts):
        wt = sum(inst.weights[v] for v in part if v not in tset)
        assert wt <= inst.capacities[i] + inst.w_max - 1


def test_policies_differ_but_all_valid() -> None:
    """§13.7: the policies are different rules (they can produce different partitions) and all valid."""
    inst = random_kT_connected_dag(80, 4, 11, "balanced", extra_out=1)
    outs = {policy: run_core(inst, policy, 1) for policy in (0, 1, 2)}
    for res in outs.values():
        assert_valid(inst, res)
    assignments = {tuple(r["assignment"]) for r in outs.values()}
    assert len(assignments) >= 2, "expected at least two policies to differ on this instance"


# ---------------------------------------------------------------------------
# [Lem 9.1] preconditions and input validation
# ---------------------------------------------------------------------------


def test_precondition_out_degree_reported() -> None:
    """[Lem 9.1]: a non-terminal with out-degree < k is reported by both variants (and skipped
    with check_precondition=False only to fail at run time with a clear error)."""
    inst = random_kT_connected_dag(30, 3, 3)
    out = inst.out_adjacency()
    bad = max((v for v in range(inst.n) if v not in inst.terminals), key=lambda v: len(out[v]))
    arcs = [a for a in inst.arcs if a[0] != bad] + [(bad, out[bad][0])]  # out-degree 1 < 3
    assert len(set(arcs)) == len(arcs)
    for variant in (0, 1):
        res = _core.dag_partition(inst.n, arcs, list(inst.terminals), list(inst.capacities), None, 0, variant)
        assert res["status"] == "precondition_failed"
        assert "out-degree" in res["message"]
        assert int(re.search(r"non-terminal (\d+)", res["message"]).group(1)) == bad
        assert res["assignment"] == [] and res["parent"] == []
    ref = gl_dag_partition(make_instance(inst.n, arcs, inst.terminals, inst.capacities), policy="first")
    assert ref.status == "precondition_failed"


def test_precondition_cycle_reported() -> None:
    """[Lem 9.1]/[Def 9.2]: a directed cycle among non-terminals is reported with a vertex on it."""
    inst = random_kT_connected_dag(25, 2, 4)
    # find two non-terminals u -> v and add v -> u (a 2-cycle); keep the arc list simple
    tset = set(inst.terminals)
    u, v = next((a, b) for a, b in inst.arcs if b not in tset)
    arcs = list(inst.arcs) + [(v, u)]
    assert (v, u) not in inst.arcs
    ref = gl_dag_partition(make_instance(inst.n, arcs, inst.terminals, inst.capacities), policy="first")
    assert ref.status == "precondition_failed" and "cycle" in ref.message
    ref_vertex = int(re.search(r"vertex (\d+)", ref.message).group(1))
    for variant in (0, 1):
        res = _core.dag_partition(inst.n, arcs, list(inst.terminals), list(inst.capacities), None, 0, variant)
        assert res["status"] == "precondition_failed"
        assert "cycle" in res["message"]
        reported = int(re.search(r"vertex (\d+)", res["message"]).group(1))
        assert reported == ref_vertex and reported not in tset
        # also with the out-degree check disabled: the canonical order does not exist
        res2 = _core.dag_partition(
            inst.n, arcs, list(inst.terminals), list(inst.capacities), None, 0, variant, check_precondition=False
        )
        assert res2["status"] == "precondition_failed" and "cycle" in res2["message"]


def test_precondition_total_weight_and_malformed_input() -> None:
    inst = random_kT_connected_dag(20, 2, 8, weighted=True, w_max=3)
    caps = list(inst.capacities)
    caps[0] -= 1  # Σ w > Σ c
    res = _core.dag_partition(inst.n, list(inst.arcs), list(inst.terminals), caps, list(inst.weights), 0, 0)
    assert res["status"] == "precondition_failed" and "weight" in res["message"]
    with pytest.raises(ValueError):
        _core.dag_partition(inst.n, list(inst.arcs), list(inst.terminals), caps[:1], list(inst.weights), 0, 0)
    with pytest.raises(ValueError):
        _core.dag_partition(inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities), None, 3, 0)
    with pytest.raises(ValueError):
        _core.dag_partition(inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities), None, 0, 2)
    with pytest.raises(ValueError):
        _core.dag_partition(inst.n, list(inst.arcs), [inst.terminals[0], inst.terminals[0]], caps, None, 0, 0)


def test_check_precondition_false_runs_dry_heap() -> None:
    """Without the [Lem 9.1] check a non-k-T-connected DAG makes a heap run dry [Lem 9.4] -> RuntimeError,
    or (if the deficient vertex is never needed) still returns a verified partition."""
    inst = random_kT_connected_dag(12, 3, 9)
    tset = set(inst.terminals)
    out = inst.out_adjacency()
    v0 = next(v for v in range(inst.n) if v not in tset)
    arcs = [a for a in inst.arcs if a[0] != v0] + [(v0, out[v0][0])]
    for variant in (0, 1):
        try:
            res = _core.dag_partition(
                inst.n, arcs, list(inst.terminals), list(inst.capacities), None, 2, variant, check_precondition=False
            )
        except RuntimeError as exc:
            assert "ran dry" in str(exc)
        else:
            assert res["status"] == "ok"
            report = verify_instance_parts(make_instance(inst.n, arcs, inst.terminals, inst.capacities), parts_of(inst, res["assignment"]))
            assert report.valid, report.errors


# ---------------------------------------------------------------------------
# trace (paper_notes §14)
# ---------------------------------------------------------------------------


def test_trace_events_consistent_with_parents() -> None:
    import json

    inst = random_kT_connected_dag(40, 3, 12, "random", weighted=True, w_max=3)
    for variant in (0, 1):
        res = run_core(inst, 0, variant, trace=True)
        assert_valid(inst, res)
        events = [(t, json.loads(j)) for t, j in res["trace"]]
        assert events[0][0] == "init" and events[-1][0] == "done"
        contracts = [e for t, e in events if t == "dag_contract"]
        assert len(contracts) == inst.n - inst.k
        seen = set()
        for e in contracts:
            p, t, parent = e["p"], e["t"], e["parent"]
            assert p not in seen
            seen.add(p)
            assert res["parent"][p] == parent
            assert inst.terminals[res["assignment"][p]] == t
        done = events[-1][1]["parts"]
        for i, t in enumerate(inst.terminals):
            assert sorted(done[str(t)]) == sorted(v for v in range(inst.n) if res["assignment"][v] == i)
    # trace disabled -> empty
    assert run_core(inst, 0, 0)["trace"] == []


# ---------------------------------------------------------------------------
# timing (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_timing_large_dag_both_variants() -> None:
    """n = 200k, m = 1.6M random k-T-connected DAG (k = 8): both variants finish, agree, and are
    verified. Prints core seconds (stats.time_seconds['dag']) and wall seconds per variant."""
    t0 = time.perf_counter()
    inst = random_kT_connected_dag(200_000, 8, 2024, "balanced")
    gen = time.perf_counter() - t0
    assert inst.m == 8 * (inst.n - inst.k)
    args = (inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities), None)
    outs = {}
    for variant in (0, 1):
        for policy in (0, 1, 2):
            t1 = time.perf_counter()
            res = _core.dag_partition(*args, policy, variant)
            wall = time.perf_counter() - t1
            assert res["status"] == "ok"
            outs[(variant, policy)] = res
            print(
                f"n={inst.n} m={inst.m} k={inst.k} variant={variant} policy={policy}: "
                f"core {res['stats']['time_seconds']['dag']:.3f}s, wall {wall:.3f}s (generation {gen:.1f}s)"
            )
    for policy in (0, 1, 2):
        assert outs[(0, policy)]["assignment"] == outs[(1, policy)]["assignment"]
        assert outs[(0, policy)]["parent"] == outs[(1, policy)]["parent"]
    report = verify_instance_parts(inst, parts_of(inst, outs[(1, 0)]["assignment"]))
    assert report.valid, report.errors[:5]
    check_parents(inst, outs[(1, 0)]["assignment"], outs[(1, 0)]["parent"])

