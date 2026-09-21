"""Review tests for ``glsolver.preconditions`` against the definitions of
docs/paper_notes.md §3-5, with hand-computed examples and independent
cross-checks (NetworkX ``node_connectivity`` with a super-sink, brute-force
enumeration of assignments, and the reference tightest-cut implementation of
``glref.essential`` [Prop 4.2] / [Lem 4.1]).
"""
from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from glsolver.instance import Instance, make_instance
from glsolver.preconditions import (
    check_preconditions,
    essential_sets_by_definition,
    essential_terminals_by_definition,
    feac_witness_nx,
    fesac_witness_nx,
    is_dag,
    is_k_T_connected,
    is_k_T_connected_dag,
    terminal_connectivity_nx,
    undirected_vertex_connectivity,
)

# ---------------------------------------------------------------------------
# helpers / independent oracles
# ---------------------------------------------------------------------------


def kappa_super_sink(inst: Instance, v: int) -> int:
    """Independent ``κ_G(v)`` [Def 3.2]: add a sink ``z`` with an arc from every
    terminal; the maximum number of internally vertex-disjoint ``v -> z``
    paths (``nx.node_connectivity``) equals the number of vertex-disjoint
    paths from ``v`` to *distinct* terminals, because each terminal is an
    internal vertex of such a path."""
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(inst.n))
    digraph.add_edges_from(inst.arcs)
    for t in inst.terminals:
        digraph.add_edge(t, "z")
    return int(nx.node_connectivity(digraph, v, "z"))


def random_directed_instance(rng: random.Random, n_max: int = 8, k_max: int = 4) -> Instance:
    n = rng.randint(2, n_max)
    k = rng.randint(1, min(k_max, n))
    terminals = rng.sample(range(n), k)
    p = rng.uniform(0.1, 0.8)
    arcs = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
    caps = [0] * k
    for v in range(n):
        if v not in terminals:
            caps[rng.randrange(k)] += 1
    return make_instance(n, arcs, terminals, caps, directed=True)


def witness_counts(inst: Instance, phi: dict[int, int]) -> dict[int, int]:
    counts = {t: 0 for t in inst.terminals}
    for t in phi.values():
        counts[t] += 1
    return counts


# ---------------------------------------------------------------------------
# κ counts paths to DISTINCT terminals [Def 3.2]
# ---------------------------------------------------------------------------


def test_kappa_counts_distinct_terminals_not_paths():
    # v = 3 has three vertex-disjoint paths to t_0 (direct, via 4, via 5) and none to t_1
    inst = make_instance(6, [(3, 0), (3, 4), (4, 0), (3, 5), (5, 0)], [0, 1], [3, 1], directed=True)
    assert terminal_connectivity_nx(inst, 3) == 1
    assert kappa_super_sink(inst, 3) == 1
    assert essential_terminals_by_definition(inst, 3) == {0}
    # give one of the routes a second terminal: κ becomes 2
    inst = make_instance(6, [(3, 0), (3, 4), (4, 0), (3, 5), (5, 0), (5, 1)], [0, 1], [3, 1], directed=True)
    assert terminal_connectivity_nx(inst, 3) == 2
    assert essential_terminals_by_definition(inst, 3) == {0, 1}
    # a pre-terminal with two arcs into the same terminal is impossible (simple graph); with arcs
    # to t_0 and t_0-only routes plus one arc to t_1, κ = 2 exactly even with high out-degree
    inst = make_instance(7, [(3, 0), (3, 4), (4, 0), (3, 5), (5, 0), (3, 6), (6, 1)], [0, 1], [4, 1], directed=True)
    assert terminal_connectivity_nx(inst, 3) == 2
    assert terminal_connectivity_nx(inst, 3) <= min(inst.k, 4)


def test_kappa_bounds_and_zero():
    # κ <= min(k, d^+(v)); κ = 0 iff v reaches no terminal (paper_notes §3 note, §13.9)
    inst = make_instance(5, [(2, 0), (2, 1), (3, 2), (4, 3), (3, 4)], [0, 1], [3, 0], directed=True)
    assert terminal_connectivity_nx(inst, 2) == 2  # d^+ = 2, k = 2
    assert terminal_connectivity_nx(inst, 3) == 1  # d^+ = 2 but every path goes through 2
    assert terminal_connectivity_nx(inst, 4) == 1
    assert is_k_T_connected(inst) == (False, 3)
    lonely = make_instance(4, [(2, 0), (2, 1), (2, 3)], [0, 1], [1, 1], directed=True)
    assert terminal_connectivity_nx(lonely, 3) == 0
    assert essential_terminals_by_definition(lonely, 3) == set()
    assert check_preconditions(lonely, mode=True)["violating_vertex"] == 3


def test_kappa_matches_independent_node_connectivity_on_random_instances():
    rng = random.Random(5)
    checked = 0
    for _ in range(120):
        inst = random_directed_instance(rng)
        for v in range(inst.n):
            if v in inst.terminals:
                continue
            kappa = terminal_connectivity_nx(inst, v)
            assert kappa == kappa_super_sink(inst, v), (inst, v)
            out_deg = sum(1 for u, _ in inst.arcs if u == v)
            assert 0 <= kappa <= min(inst.k, out_deg)
            checked += 1
    assert checked > 200


# ---------------------------------------------------------------------------
# essential terminals by VERTEX deletion of t [Def 4.1]
# ---------------------------------------------------------------------------


def test_essential_uses_vertex_deletion_not_arc_deletion():
    # v = 2: arcs v->t_0, v->x, x->t_0, v->t_1 (x = 3).  κ(v) = 2 ({v->t_0, v->t_1}).
    # Deleting the *vertex* t_0 removes both v->t_0 and x->t_0: κ drops to 1, so t_0 is essential.
    # Deleting only the arc (v, t_0) would leave v->x->t_0 and wrongly make t_0 non-essential.
    inst = make_instance(4, [(2, 0), (2, 3), (3, 0), (2, 1)], [0, 1], [2, 0], directed=True)
    assert terminal_connectivity_nx(inst, 2) == 2
    assert essential_terminals_by_definition(inst, 2) == {0, 1}
    # x = 3 itself: only route x->t_0, κ = 1, Ess = {t_0}
    assert essential_terminals_by_definition(inst, 3) == {0}


def test_essential_can_be_empty_with_positive_kappa():
    # v -> x, x -> t_0, x -> t_1: κ(v) = 1 (bottleneck x) but neither terminal is essential
    # (removing either leaves κ = 1), matching [Lem 4.1]: tightest cut S = {x}, T in L.
    inst = make_instance(4, [(2, 3), (3, 0), (3, 1)], [0, 1], [1, 1], directed=True)
    assert terminal_connectivity_nx(inst, 2) == 1
    assert essential_terminals_by_definition(inst, 2) == set()
    assert essential_terminals_by_definition(inst, 3) == {0, 1}
    # hence FEAC fails although every vertex reaches every terminal
    assert feac_witness_nx(inst) is None
    res = check_preconditions(inst, mode=True)
    assert res["feac"] is False and res["k_T_connected"] is False


def test_essential_hand_example_paper_notes_13_3_and_deletion_of_a_terminal():
    # v=3: v->t_0, v->x(4), x->t_1, x->t_2.  Ess(v) = {t_0}; after deleting t_1 both remaining are essential.
    inst = make_instance(5, [(3, 0), (3, 4), (4, 1), (4, 2)], [0, 1, 2], [2, 0, 0], directed=True)
    assert terminal_connectivity_nx(inst, 3) == 2
    assert essential_terminals_by_definition(inst, 3) == {0}
    assert essential_terminals_by_definition(inst, 4) == {1, 2}
    # emulate "G - t_1" literally: vertex 1 keeps its id but loses all arcs and its terminal status
    reduced = make_instance(5, [(3, 0), (3, 4), (4, 2)], [0, 2], [3, 0], directed=True)
    assert terminal_connectivity_nx(reduced, 3) == 2
    assert essential_terminals_by_definition(reduced, 3) == {0, 2}
    # the old essential terminal survives [Lem 7.1]
    assert {0} <= essential_terminals_by_definition(reduced, 3)


def test_essential_by_definition_matches_reference_tightest_cut_on_random_instances():
    """[Lem 4.1]: ``t`` essential iff ``t`` lies in the separator of the tightest
    min cut.  Cross-check the by-definition (k+1 flows) computation against the
    reference implementation of [Prop 4.2] in ``glref.essential``."""
    glref_essential = pytest.importorskip("glref.essential")
    glref_graph = pytest.importorskip("glref.graph")
    rng = random.Random(99)
    checked = 0
    for _ in range(120):
        inst = random_directed_instance(rng)
        state = glref_graph.DiGraphState(inst.n, inst.arcs, inst.terminals)
        for v in range(inst.n):
            if v in inst.terminals:
                continue
            kappa_ref, ess_ref = glref_essential.essential_terminals(state, v)
            kappa = terminal_connectivity_nx(inst, v)
            ess = essential_terminals_by_definition(inst, v)
            assert (kappa, ess) == (kappa_ref, set(ess_ref)), (inst, v)
            assert ess <= set(inst.terminals)
            assert len(ess) <= kappa
            if kappa == inst.k:
                assert ess == set(inst.terminals)
            checked += 1
    assert checked > 200


# ---------------------------------------------------------------------------
# FEAC witness respects EXACT capacities [Def 5.1]
# ---------------------------------------------------------------------------


def test_feac_witness_respects_exact_capacities_including_zero():
    arcs = [(2, 0), (2, 1), (3, 0), (3, 1)]  # both non-terminals see both terminals
    assert feac_witness_nx(make_instance(4, arcs, [0, 1], [2, 0])) == {2: 0, 3: 0}
    assert feac_witness_nx(make_instance(4, arcs, [0, 1], [0, 2])) == {2: 1, 3: 1}
    phi = feac_witness_nx(make_instance(4, arcs, [0, 1], [1, 1]))
    assert phi is not None and sorted(phi.values()) == [0, 1]
    # only t_0 is essential for both: capacities (1, 1) are infeasible, (2, 0) feasible
    arcs = [(2, 0), (3, 0)]
    assert feac_witness_nx(make_instance(4, arcs, [0, 1], [1, 1])) is None
    assert feac_witness_nx(make_instance(4, arcs, [0, 1], [2, 0])) == {2: 0, 3: 0}
    # a witness never uses a non-essential terminal, even when that would balance the load
    arcs = [(2, 3), (3, 0), (3, 1), (4, 0), (4, 1)]  # Ess(2) = {}, Ess(3) = Ess(4) = {0, 1}
    assert feac_witness_nx(make_instance(5, arcs, [0, 1], [2, 1])) is None


def test_feac_witness_agrees_with_brute_force_enumeration():
    rng = random.Random(21)
    feasible = infeasible = 0
    for _ in range(150):
        inst = random_directed_instance(rng, n_max=6, k_max=3)
        ess = essential_sets_by_definition(inst)
        nonterm = [v for v in range(inst.n) if v not in inst.terminals]
        exists = any(
            all(inst.terminals[a] in ess[v] for v, a in zip(nonterm, assign))
            and all(assign.count(i) == inst.capacities[i] for i in range(inst.k))
            for assign in itertools.product(range(inst.k), repeat=len(nonterm))
        )
        phi = feac_witness_nx(inst)
        assert (phi is not None) == exists, (inst, ess, phi)
        if phi is None:
            infeasible += 1
            continue
        feasible += 1
        assert set(phi) == set(nonterm)
        assert all(phi[v] in ess[v] for v in phi)
        assert witness_counts(inst, phi) == dict(zip(inst.terminals, inst.capacities))
    assert feasible > 20 and infeasible > 20


def test_check_preconditions_shortcut_witness_is_a_real_witness():
    """When k-T-connected, ``check_preconditions`` skips the k+1 flows and uses
    ``Ess(v) = T`` (paper §7.3); that must coincide with the definition and
    the witness must still be capacity-exact."""
    rng = random.Random(8)
    seen = 0
    for _ in range(200):
        inst = random_directed_instance(rng, n_max=7, k_max=3)
        res = check_preconditions(inst, mode=True)
        if not res["k_T_connected"]:
            assert terminal_connectivity_nx(inst, res["violating_vertex"]) < inst.k
            continue
        seen += 1
        ess = essential_sets_by_definition(inst)
        assert all(ess[v] == set(inst.terminals) for v in ess)
        assert res["feac"] is True
        phi = res["witness"]
        assert set(phi) == set(ess)
        assert witness_counts(inst, phi) == dict(zip(inst.terminals, inst.capacities))
    assert seen > 10


# ---------------------------------------------------------------------------
# FESAC witness [Def 5.3]
# ---------------------------------------------------------------------------


def test_fesac_witness_properties_and_infeasibility():
    rng = random.Random(11)
    found = failed = 0
    for _ in range(120):
        n = rng.randint(2, 7)
        k = rng.randint(1, min(3, n))
        terminals = rng.sample(range(n), k)
        p = rng.uniform(0.15, 0.8)
        arcs = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
        weights = [0 if v in terminals else rng.randint(1, 4) for v in range(n)]
        caps = [0] * k
        for _ in range(sum(weights) + rng.randint(0, 2)):
            caps[rng.randrange(k)] += 1
        inst = make_instance(n, arcs, terminals, caps, weights=weights, directed=True)
        ess = essential_sets_by_definition(inst)
        psi = fesac_witness_nx(inst)
        res = check_preconditions(inst, mode=True)
        assert res["condition"] == "FESAC"
        assert (psi is None) == (res["feac"] is False)
        if psi is None:
            failed += 1
            # Hall-type sanity: some set of vertices has more weight than its essential terminals can hold
            nonterm = [v for v in range(n) if v not in terminals]
            cap_of = dict(zip(terminals, caps))
            deficient = any(
                sum(weights[v] for v in subset) > sum(cap_of[t] for t in set().union(*(ess[v] for v in subset)))
                for r in range(1, len(nonterm) + 1)
                for subset in itertools.combinations(nonterm, r)
            )
            assert deficient, (inst, ess)
            continue
        found += 1
        loads = {t: 0 for t in terminals}
        for v, units in psi.items():
            assert v not in terminals
            assert sum(units.values()) == weights[v]
            assert set(units) <= ess[v]
            assert all(u > 0 for u in units.values())
            for t, u in units.items():
                loads[t] += u
        assert all(loads[t] <= c for t, c in zip(terminals, caps))
    assert found > 20 and failed > 20


def test_fesac_with_unit_weights_coincides_with_feac():
    rng = random.Random(13)
    for _ in range(60):
        inst = random_directed_instance(rng, n_max=7, k_max=3)
        unit = make_instance(
            inst.n, inst.arcs, inst.terminals, inst.capacities,
            weights=[0 if v in inst.terminals else 1 for v in range(inst.n)], directed=True,
        )
        phi = feac_witness_nx(inst)
        psi = fesac_witness_nx(unit)
        assert (phi is None) == (psi is None), inst
        if psi is not None:
            assert all(list(units.values()) == [1] for units in psi.values())


# ---------------------------------------------------------------------------
# k-T-connectivity, DAG criterion [Lem 9.1], classical connectivity [§1.1]
# ---------------------------------------------------------------------------


def test_k_T_connected_requires_kappa_equal_k_for_every_non_terminal():
    # complete bipartite {2,3} -> {0,1}: 2-T-connected
    inst = make_instance(4, [(2, 0), (2, 1), (3, 0), (3, 1)], [0, 1], [1, 1], directed=True)
    assert is_k_T_connected(inst) == (True, None)
    # add a vertex 4 that only reaches vertex 2: κ(4) = 1 < 2
    inst = make_instance(5, [(2, 0), (2, 1), (3, 0), (3, 1), (4, 2)], [0, 1], [2, 1], directed=True)
    assert is_k_T_connected(inst) == (False, 4)
    assert terminal_connectivity_nx(inst, 4) == 1
    # a vertex that reaches k terminals but only through one neighbour is NOT k-T-connected
    inst = make_instance(5, [(2, 0), (2, 1), (3, 0), (3, 1), (4, 2), (4, 3)], [0, 1], [2, 1], directed=True)
    assert is_k_T_connected(inst) == (True, None)  # 4 -> 2 -> t_0 and 4 -> 3 -> t_1 are disjoint


def test_dag_out_degree_criterion_hand_examples():
    # Lem 9.1 proof example: v = 4 has out-degree 2 to {2, 3}, both of out-degree 2: 2-T-connected
    inst = make_instance(5, [(4, 2), (4, 3), (2, 0), (2, 1), (3, 0), (3, 1)], [0, 1], [2, 1], directed=True)
    assert is_dag(inst)
    assert is_k_T_connected_dag(inst) == (True, None) == is_k_T_connected(inst)
    # out-degree 2 but one out-neighbour has out-degree 1: both criteria fail, possibly at different vertices
    inst = make_instance(5, [(4, 2), (4, 3), (2, 0), (2, 1), (3, 0)], [0, 1], [2, 1], directed=True)
    assert is_k_T_connected_dag(inst) == (False, 3)
    assert is_k_T_connected(inst) == (False, 3)
    # the criterion counts arcs to any vertex, terminal or not
    inst = make_instance(4, [(3, 2), (3, 0), (2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    assert is_k_T_connected_dag(inst) == (True, None)
    assert is_k_T_connected(inst) == (True, None)


def test_classically_k_connected_graphs_are_k_T_connected_for_every_T():
    """paper_notes §1.1: a k-vertex-connected undirected graph is k-T-connected
    for every terminal set of size k (after the symmetrisation / terminal
    out-arc dropping of make_instance)."""
    rng = random.Random(3)
    checked = 0
    for _ in range(80):
        n = rng.randint(3, 9)
        graph = nx.gnp_random_graph(n, rng.uniform(0.3, 0.9), seed=rng.randint(0, 10**6))
        conn = nx.node_connectivity(graph)
        if conn == 0:
            continue
        k = rng.randint(1, min(conn, n - 1))
        terminals = rng.sample(range(n), k)
        caps = [0] * k
        for v in range(n):
            if v not in terminals:
                caps[rng.randrange(k)] += 1
        inst = make_instance(n, list(graph.edges()), terminals, caps, directed=False)
        assert undirected_vertex_connectivity(inst) == conn
        ok, bad = is_k_T_connected(inst)
        assert ok, (inst, bad, conn, k)
        assert check_preconditions(inst, mode=True)["feac"] is True
        checked += 1
    assert checked > 40
    # the converse fails: paper Fig. 1 counterexample is 3-T-connected but only 2-connected
    edges = [(0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 8), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (3, 8)]
    inst = make_instance(9, edges, [0, 1, 2], sizes=[3, 3, 3], directed=False)
    assert undirected_vertex_connectivity(inst) == 2
    assert is_k_T_connected(inst) == (True, None)


def test_undirected_vertex_connectivity_counts_terminal_terminal_edges():
    # K_4 minus nothing: 3-connected; drop the terminal-terminal edge (0,1): connectivity 2
    complete = make_instance(4, [(u, v) for u in range(4) for v in range(u + 1, 4)], [0, 1], sizes=[2, 2], directed=False)
    assert undirected_vertex_connectivity(complete) == 3
    minus = make_instance(4, [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [0, 1], sizes=[2, 2], directed=False)
    assert undirected_vertex_connectivity(minus) == 2
    # both are 2-T-connected: the terminal-terminal edge never helps a non-terminal
    assert is_k_T_connected(complete) == (True, None) == is_k_T_connected(minus)
