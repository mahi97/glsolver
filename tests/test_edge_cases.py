"""Edge cases of the API and of every registered solver (docs/verification.md).

Degenerate shapes (``k == n``, ``k == n-1``, ``k == 1``, ``n == 1``), zero
capacities, terminals without in-arcs, exactly-``k``-connected Harary graphs,
complete graphs for every ``k``, input normalization (self-loops, duplicate
edges, multigraphs, string node labels), the three worked examples of the
paper through every applicable backend, and the authors' official
counterexample (``copies = 1``) through the reference solver (slow) and the
C++ core (when built).
"""
from __future__ import annotations

import itertools

import networkx as nx
import pytest

from glref.counterexample import build_counterexample_instance
from glsolver.api import GLResult, core_available, glpartition, partition
from glsolver.generators import (
    complete_graph,
    harary_graph,
    paper_contract_counterexample,
    paper_essential_example,
    paper_running_example,
)
from glsolver.instance import Instance, make_instance
from glsolver.preconditions import (
    check_preconditions,
    is_k_T_connected,
    undirected_vertex_connectivity,
)
from glsolver.testing.registry import available_solvers, solvers_for
from glsolver.verify import verify_instance_parts, verify_partition

ORACLES = ("bruteforce", "ilp")
THEOREM_SOLVERS = ("reference", "reference-weighted", "reference-dag", "general", "weighted", "dag")


def _kw(name: str) -> dict:
    if name in ORACLES:
        return {"verify_preconditions": False, "time_limit": 60.0}
    if name.startswith("reference"):
        return {"debug": True}
    return {}


def solve_all(inst: Instance, *, include_oracles: bool = True) -> dict[str, GLResult]:
    """Run every applicable backend on ``inst`` and return ``name -> result``."""
    return {name: fn(inst, **_kw(name)) for name, fn in solvers_for(inst, include_oracles=include_oracles).items()}


def assert_all_ok(inst: Instance, results: dict[str, GLResult]) -> None:
    assert results, "no applicable solver"
    for name, res in results.items():
        assert res.status == "ok", f"{name}: {res.status}: {res.message}"
        assert res.valid is True, f"{name}: {res.verification.errors if res.verification else res.message}"
        assert verify_instance_parts(inst, res.parts).valid
        if inst.weights is None:
            assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities], name


# ---------------------------------------------------------------------------
# k == n, k == n-1, k == 1, n == 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_k_equals_n_every_part_is_a_singleton(n: int, directed: bool) -> None:
    edges = list(itertools.combinations(range(n), 2))
    inst = make_instance(n, edges, list(range(n)), [0] * n, directed=directed)
    assert inst.arcs == ()  # every arc leaves a terminal and is dropped (paper §2)
    results = solve_all(inst)
    assert set(results) >= {"reference", "reference-weighted", "reference-dag", "bruteforce"}
    assert_all_ok(inst, results)
    for name, res in results.items():
        assert res.parts == [[t] for t in range(n)], name
        assert res.assignment == list(range(n)), name
    # classical API with sizes all 1
    res = partition(nx.complete_graph(n), list(range(n)), sizes=[1] * n)
    assert res.status == "ok" and res.valid is True and res.parts == [[t] for t in range(n)]


def test_k_equals_n_minus_one_single_non_terminal() -> None:
    # undirected K_4, terminals 0,1,2, non-terminal 3: whichever terminal gets capacity 1 takes it
    inst_k4 = complete_graph(4, 3, seed=0, mode="extreme")
    for caps in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        inst = make_instance(4, inst_k4.undirected_edges, inst_k4.terminals, caps, directed=False)
        results = solve_all(inst)
        assert_all_ok(inst, results)
        (nt,) = [v for v in range(4) if v not in inst.terminals]
        i = caps.index(1)
        for name, res in results.items():
            assert sorted(res.parts[i]) == sorted([inst.terminals[i], nt]), name
    # directed: the non-terminal 3 has an arc to terminal 1 only
    inst = make_instance(4, [(3, 1)], [0, 1, 2], [0, 1, 0], directed=True)
    assert_all_ok(inst, solve_all(inst))
    bad = make_instance(4, [(3, 1)], [0, 1, 2], [1, 0, 0], directed=True)
    for name, res in solve_all(bad).items():
        if name in ORACLES:
            assert res.status == "infeasible", name  # truly no partition exists
        else:
            assert res.status == "precondition_failed", (name, res.status, res.message)


@pytest.mark.parametrize("builder", [
    lambda: nx.path_graph(7), lambda: nx.cycle_graph(6), lambda: nx.star_graph(5),
    lambda: nx.gnp_random_graph(9, 0.4, seed=3),
])
def test_k_equals_one_whole_graph(builder) -> None:
    graph = builder()
    if not nx.is_connected(graph):
        pytest.skip("generator produced a disconnected graph")
    n = graph.number_of_nodes()
    for t in range(n):
        inst = make_instance(n, list(graph.edges()), [t], [n - 1], directed=False)
        results = solve_all(inst, include_oracles=(t == 0))
        assert_all_ok(inst, results)
        for name, res in results.items():
            assert sorted(res.parts[0]) == list(range(n)), name
    # directed: k == 1 works iff every vertex reaches the terminal (§13.10)
    inst = make_instance(4, [(1, 0), (2, 1), (3, 2)], [0], [3], directed=True)
    assert_all_ok(inst, solve_all(inst))
    lonely = make_instance(4, [(1, 0), (2, 1)], [0], [3], directed=True)
    for name, res in solve_all(lonely).items():
        expected = "infeasible" if name in ORACLES else "precondition_failed"
        assert res.status == expected, (name, res.status)


def test_single_vertex_instance() -> None:
    inst = make_instance(1, [], [0], [0], directed=False)
    results = solve_all(inst)
    assert_all_ok(inst, results)
    assert all(res.parts == [[0]] for res in results.values())


# ---------------------------------------------------------------------------
# zero capacities and terminals without in-arcs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caps", [[2, 0, 0, 0], [0, 0, 2, 0], [1, 0, 1, 0], [0, 1, 0, 1]])
def test_capacities_with_several_zeros(caps: list[int]) -> None:
    base = complete_graph(6, 4, seed=1)
    inst = make_instance(6, base.undirected_edges, base.terminals, caps, directed=False)
    results = solve_all(inst)
    assert_all_ok(inst, results)
    for name, res in results.items():
        for i, c in enumerate(caps):
            if c == 0:
                assert res.parts[i] == [inst.terminals[i]], name
    # the reference solver removes each zero-capacity terminal (step (i), [Lem 7.2])
    ref = glpartition(inst, algorithm="reference", trace=True)
    assert ref.stats["terminal_removals"] >= caps.count(0) - 1
    assert sum(1 for ev in ref.trace if ev["type"] == "remove_terminal") == ref.stats["terminal_removals"]


def test_terminal_without_in_arcs() -> None:
    # terminal 2 has no in-arcs: fine with capacity 0 ...
    arcs = [(3, 0), (3, 1), (4, 0), (4, 1), (3, 4), (4, 3)]
    fine = make_instance(5, arcs, [0, 1, 2], [1, 1, 0], directed=True)
    pre = check_preconditions(fine, True)
    assert pre["k_T_connected"] is False and pre["feac"] is True
    assert_all_ok(fine, solve_all(fine))
    # ... but a precondition failure (no vertex has t_2 essential) with capacity > 0
    for caps in ([1, 0, 1], [0, 0, 2]):
        bad = make_instance(5, arcs, [0, 1, 2], caps, directed=True)
        assert check_preconditions(bad, True)["feac"] is False
        for name, res in solve_all(bad).items():
            if name in ORACLES:
                assert res.status == "infeasible", name
            else:
                assert res.status == "precondition_failed", (name, res.status)
                assert "infeasible" not in res.status


# ---------------------------------------------------------------------------
# exactly k-connected and complete graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (8, 4), (9, 3)])
@pytest.mark.parametrize("mode", ["balanced", "unbalanced", "extreme", "random"])
def test_harary_graph_is_exactly_k_connected_and_solved(n: int, k: int, mode: str) -> None:
    inst = harary_graph(n, k, seed=n * 10 + k, mode=mode)
    assert undirected_vertex_connectivity(inst) == k
    assert is_k_T_connected(inst) == (True, None)
    assert_all_ok(inst, solve_all(inst, include_oracles=(mode == "random")))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_complete_graph_every_k(n: int) -> None:
    edges = list(itertools.combinations(range(n), 2))
    for k in range(1, n + 1):
        for mode in ("balanced", "extreme", "unbalanced"):
            if k < n:
                inst = complete_graph(n, k, seed=k, mode=mode)
            else:  # the generator refuses k == n (K_n is (n-1)-connected); build it directly
                inst = make_instance(n, edges, list(range(n)), [0] * n, directed=False)
            assert undirected_vertex_connectivity(inst) == n - 1
            assert_all_ok(inst, solve_all(inst, include_oracles=(mode == "balanced")))


# ---------------------------------------------------------------------------
# input normalization
# ---------------------------------------------------------------------------


def test_self_loops_and_duplicate_edges_are_normalized() -> None:
    raw = [(0, 1), (1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (0, 3), (2, 0)]
    inst = make_instance(4, raw, [0, 2], [1, 1], directed=False)
    assert inst.undirected_edges == ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
    assert all(u != v for u, v in inst.arcs)
    assert len(set(inst.arcs)) == len(inst.arcs)
    assert not any(u in inst.terminals for u, _v in inst.arcs)
    assert_all_ok(inst, solve_all(inst))
    # directed input with duplicates, self-loops and arcs out of terminals
    draw = [(2, 0), (2, 0), (2, 2), (3, 2), (3, 1), (0, 3), (1, 2)]
    dinst = make_instance(4, draw, [0, 1], [1, 1], directed=True)
    assert dinst.arcs == ((2, 0), (3, 1), (3, 2))
    assert_all_ok(dinst, solve_all(dinst))
    # multigraph with a self-loop through the networkx front-end
    mg = nx.MultiGraph()
    mg.add_edges_from([(0, 1), (0, 1), (1, 1), (1, 2), (2, 0), (0, 2), (2, 3)])
    res = glpartition(mg, [0, 3], sizes=[2, 2])
    assert res.status == "ok" and res.valid is True
    assert res.instance.undirected_edges == ((0, 1), (0, 2), (1, 2), (2, 3))


def test_networkx_string_labels_round_trip() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c"), ("b", "e"), ("e", "c")])
    res = glpartition(graph, ["a", "d"], sizes=[3, 2])
    assert res.status == "ok" and res.valid is True
    assert res.labels == list(graph.nodes())
    labels = res.parts_labels
    assert "a" in labels[0] and "d" in labels[1]
    assert sorted(v for p in labels for v in p) == sorted(graph.nodes())
    assert verify_partition(graph, ["a", "d"], sizes=[3, 2], parts=labels).valid
    for name in ("reference-weighted", "bruteforce"):
        r = glpartition(graph, ["a", "d"], sizes=[3, 2], algorithm=name)
        assert r.status == "ok" and r.valid is True
        assert verify_partition(graph, ["a", "d"], sizes=[3, 2], parts=r.parts_labels).valid
    digraph = nx.DiGraph([("x", "t1"), ("x", "y"), ("y", "t2"), ("z", "y"), ("z", "t1")])
    r = glpartition(digraph, ["t1", "t2"], capacities=[2, 1])
    assert r.status == "ok" and r.valid is True
    assert verify_partition(digraph, ["t1", "t2"], capacities=[2, 1], parts=r.parts_labels).valid


# ---------------------------------------------------------------------------
# the paper's worked examples
# ---------------------------------------------------------------------------


def test_paper_running_example_is_feac_only_and_solved_by_all() -> None:
    inst = paper_running_example()
    pre = check_preconditions(inst, True)
    assert pre["k_T_connected"] is False and pre["feac"] is True and pre["violating_vertex"] == 3
    results = solve_all(inst)
    assert set(results) == set(available_solvers()) - {"reference-dag", "dag"} or core_available()
    assert_all_ok(inst, results)


def test_paper_contract_counterexample_solved_by_all() -> None:
    inst = paper_contract_counterexample()
    assert undirected_vertex_connectivity(inst) == 2 < inst.k == 3
    assert is_k_T_connected(inst) == (True, None)
    results = solve_all(inst)
    assert_all_ok(inst, results)
    assert all(len(p) == 3 for p in results["reference"].parts)


def test_paper_essential_example_solved_by_all() -> None:
    inst = paper_essential_example()
    pre = check_preconditions(inst, True)
    assert pre["feac"] is True and pre["k_T_connected"] is False
    results = solve_all(inst)
    assert_all_ok(inst, results)
    assert set(results) >= {"reference", "reference-weighted", "bruteforce"}


# ---------------------------------------------------------------------------
# the official counterexample (paper_notes §10)
# ---------------------------------------------------------------------------


def _check_counterexample(res: GLResult, inst: Instance) -> None:
    assert res.status == "ok", res.message
    assert res.valid is True, res.verification.errors if res.verification else res.message
    assert verify_instance_parts(inst, res.parts).valid
    assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities]
    assert res.stats.get("deletions", 0) + res.stats.get("contractions", 0) > 0


@pytest.mark.slow
def test_counterexample_copies1_reference() -> None:
    inst = build_counterexample_instance(1)
    assert (inst.n, inst.k, inst.m) == (333, 9, 2160)
    res = glpartition(inst, algorithm="reference", verify_preconditions=False)
    _check_counterexample(res, inst)


@pytest.mark.skipif(not core_available(), reason="C++ core not built")
def test_counterexample_copies1_core_general() -> None:
    inst = build_counterexample_instance(1)
    res = glpartition(inst, algorithm="general", verify_preconditions=False)
    _check_counterexample(res, inst)
