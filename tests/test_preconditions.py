"""Tests for the NetworkX-based precondition cross-checks."""
from __future__ import annotations

import random

import networkx as nx
import pytest

from glsolver.instance import Instance, make_instance
from glsolver.preconditions import (
    AUTO_MAX_N,
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

# Paper Figure 1 running example: terminals t1,t2,t3 = 0,1,2; non-terminals v4..v9 = 3..8.
T1, T2, T3 = 0, 1, 2
V4, V5, V6, V7, V8, V9 = 3, 4, 5, 6, 7, 8
RUNNING_ARCS = [
    (V8, V4), (V8, V5), (V5, V4), (V4, T1), (V4, T2), (V5, T2),
    (V9, V6), (V9, V7), (V6, V7), (V6, T2), (V7, T2), (V7, T3),
]


def running_example() -> Instance:
    return make_instance(9, RUNNING_ARCS, [T1, T2, T3], [2, 2, 2], directed=True)


def contract_counter_example() -> Instance:
    """Paper Fig. 1 (k-conn-contract-counter): a 6-cycle v4..v9 with each terminal
    attached to two consecutive cycle vertices."""
    edges = [
        (T1, V4), (T1, V5), (T2, V6), (T2, V7), (T3, V8), (T3, V9),
        (V4, V5), (V5, V6), (V6, V7), (V7, V8), (V8, V9), (V4, V9),
    ]
    return make_instance(9, edges, [T1, T2, T3], sizes=[3, 3, 3], directed=False)


def is_feac_witness(inst: Instance, phi: dict[int, int], ess: dict[int, set[int]]) -> bool:
    tset = set(inst.terminals)
    if set(phi) != {v for v in range(inst.n) if v not in tset}:
        return False
    if any(phi[v] not in ess[v] for v in phi):
        return False
    counts = {t: 0 for t in inst.terminals}
    for t in phi.values():
        counts[t] += 1
    return all(counts[t] == inst.capacities[i] for i, t in enumerate(inst.terminals))


# ---------------------------------------------------------------------------
# running example (paper Figure 1 / §4 walk-through)
# ---------------------------------------------------------------------------


def test_running_example_terminal_connectivity():
    inst = running_example()
    for v in (V4, V5, V6, V7, V8, V9):
        assert terminal_connectivity_nx(inst, v) == 2
    with pytest.raises(ValueError):
        terminal_connectivity_nx(inst, T1)
    with pytest.raises(ValueError):
        terminal_connectivity_nx(inst, 9)


def test_running_example_essential_terminals():
    inst = running_example()
    expected = {
        V4: {T1, T2}, V5: {T1, T2}, V8: {T1, T2},
        V6: {T2, T3}, V7: {T2, T3}, V9: {T2, T3},
    }
    for v, ess in expected.items():
        assert essential_terminals_by_definition(inst, v) == ess
    assert essential_sets_by_definition(inst) == expected


def test_running_example_not_3_T_connected_but_feac_holds():
    inst = running_example()
    assert is_k_T_connected(inst) == (False, V4)
    ess = essential_sets_by_definition(inst)
    phi = feac_witness_nx(inst)
    assert phi is not None
    assert is_feac_witness(inst, phi, ess)
    # the paper's witness is one of the valid ones
    paper = {V4: T2, V5: T1, V6: T3, V7: T2, V8: T1, V9: T3}
    assert is_feac_witness(inst, paper, ess)
    # FESAC with unit weights coincides with FEAC (§13.6)
    psi = fesac_witness_nx(inst)
    assert psi is not None
    assert all(sum(units.values()) == 1 and set(units) <= ess[v] for v, units in psi.items())
    loads = {t: 0 for t in inst.terminals}
    for units in psi.values():
        for t, u in units.items():
            loads[t] += u
    assert all(loads[t] <= inst.capacities[i] for i, t in enumerate(inst.terminals))


def test_running_example_check_preconditions():
    inst = running_example()
    res = check_preconditions(inst, mode=True)
    assert res["k_T_connected"] is False
    assert res["violating_vertex"] == V4
    assert res["feac"] is True
    assert res["condition"] == "FEAC"
    assert is_feac_witness(inst, res["witness"], essential_sets_by_definition(inst))
    assert "not 3-T-connected" in res["message"]
    assert "FEAC holds" in res["message"]


def test_running_example_feac_fails_with_infeasible_capacities():
    # t3 is essential only for v6, v7, v9 -> capacity 4 for t3 cannot be met.
    inst = make_instance(9, RUNNING_ARCS, [T1, T2, T3], [1, 1, 4], directed=True)
    assert feac_witness_nx(inst) is None
    res = check_preconditions(inst, mode=True)
    assert res["feac"] is False
    assert res["witness"] is None
    assert "FEAC fails" in res["message"]


# ---------------------------------------------------------------------------
# paper Fig. 1 contraction counterexample (undirected)
# ---------------------------------------------------------------------------


def test_contract_counter_example_is_3_T_connected():
    inst = contract_counter_example()
    assert is_k_T_connected(inst) == (True, None)
    for v in (V4, V5, V6, V7, V8, V9):
        assert terminal_connectivity_nx(inst, v) == 3
        assert essential_terminals_by_definition(inst, v) == {T1, T2, T3}
    # Each terminal has exactly two neighbours (v4,v5 / v6,v7 / v8,v9), so the classical
    # vertex connectivity is 2 (< k = 3) although the graph is 3-T-connected: the paper's
    # caption says "3-connected to the set of terminals", not 3-vertex-connected.
    assert undirected_vertex_connectivity(inst) == 2
    res = check_preconditions(inst, mode="auto")
    assert res["k_T_connected"] is True
    assert res["feac"] is True
    assert is_feac_witness(inst, res["witness"], {v: {T1, T2, T3} for v in range(3, 9)})
    # contracting v4 into t1 (v4 removed, v5 keeps neighbours t1, v6) destroys the property
    contracted = make_instance(
        9,
        [(T1, V5), (T2, V6), (T2, V7), (T3, V8), (T3, V9),
         (V5, V6), (V6, V7), (V7, V8), (V8, V9), (T1, V9)],
        [T1, T2, T3], sizes=[3, 3, 3], directed=False,
    )
    # vertex 3 is now isolated (kappa 0) and v5 has only two neighbours
    assert is_k_T_connected(contracted) == (False, V4)
    assert terminal_connectivity_nx(contracted, V5) == 2


def test_undirected_vertex_connectivity_examples():
    complete = make_instance(5, [(u, v) for u in range(5) for v in range(u + 1, 5)], [0, 1], sizes=[2, 3], directed=False)
    assert undirected_vertex_connectivity(complete) == 4
    cycle = make_instance(6, [(i, (i + 1) % 6) for i in range(6)], [0, 3], sizes=[3, 3], directed=False)
    assert undirected_vertex_connectivity(cycle) == 2
    path = make_instance(4, [(0, 1), (1, 2), (2, 3)], [0, 3], sizes=[2, 2], directed=False)
    assert undirected_vertex_connectivity(path) == 1
    disconnected = make_instance(4, [(0, 1), (2, 3)], [0, 3], sizes=[2, 2], directed=False)
    assert undirected_vertex_connectivity(disconnected) == 0
    with pytest.raises(ValueError):
        undirected_vertex_connectivity(running_example())


# ---------------------------------------------------------------------------
# §13.3: terminal removal can create new essential terminals
# ---------------------------------------------------------------------------


def test_terminal_removal_creates_essential_terminals():
    # v = 3, x = 4; arcs v->t1, v->x, x->t2, x->t3
    inst = make_instance(5, [(3, 0), (3, 4), (4, 1), (4, 2)], [0, 1, 2], [2, 0, 0], directed=True)
    assert terminal_connectivity_nx(inst, 3) == 2
    assert essential_terminals_by_definition(inst, 3) == {0}
    assert essential_terminals_by_definition(inst, 4) == {1, 2}
    # G - t2 (vertex 1 deleted and dropped from T), relabelled: t1=0, t3=1, v=2, x=3
    reduced = make_instance(4, [(2, 0), (2, 3), (3, 1)], [0, 1], [2, 0], directed=True)
    assert terminal_connectivity_nx(reduced, 2) == 2
    assert essential_terminals_by_definition(reduced, 2) == {0, 1}


def test_kappa_zero_vertex_has_no_essential_terminals_and_breaks_feac():
    # vertex 3 reaches no terminal (§13.9)
    inst = make_instance(4, [(2, 0), (2, 1), (3, 2)], [0, 1], [1, 1], directed=True)
    assert terminal_connectivity_nx(inst, 2) == 2
    assert essential_terminals_by_definition(inst, 2) == {0, 1}
    lonely = make_instance(4, [(2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    assert terminal_connectivity_nx(lonely, 3) == 0
    assert essential_terminals_by_definition(lonely, 3) == set()
    assert feac_witness_nx(lonely) is None
    assert is_k_T_connected(lonely) == (False, 3)


def test_pre_terminal_with_single_arc():
    inst = make_instance(4, [(2, 0), (3, 2), (3, 1)], [0, 1], [1, 1], directed=True)
    assert essential_terminals_by_definition(inst, 2) == {0}
    assert essential_terminals_by_definition(inst, 3) == {0, 1}
    assert feac_witness_nx(inst) == {2: 0, 3: 1}


# ---------------------------------------------------------------------------
# DAG criterion [Lem 9.1] vs flow definition [Def 3.2]
# ---------------------------------------------------------------------------


def random_dag_instance(rng: random.Random) -> Instance:
    n = rng.randint(3, 12)
    k = rng.randint(1, min(3, n - 1))
    # non-terminals 0..n-k-1 in topological order, terminals n-k..n-1 are sinks
    terminals = list(range(n - k, n))
    arcs = []
    for u in range(n - k):
        candidates = list(range(u + 1, n))
        deg = rng.randint(0, len(candidates))
        for v in rng.sample(candidates, deg):
            arcs.append((u, v))
    caps = [0] * k
    for _ in range(n - k):
        caps[rng.randrange(k)] += 1
    return make_instance(n, arcs, terminals, caps, directed=True)


def test_dag_out_degree_criterion_agrees_with_flow_definition():
    rng = random.Random(2026)
    seen_true = seen_false = 0
    for _ in range(40):
        inst = random_dag_instance(rng)
        assert is_dag(inst)
        by_flow = is_k_T_connected(inst)
        by_degree = is_k_T_connected_dag(inst)
        assert by_flow[0] == by_degree[0], (inst, by_flow, by_degree)
        if by_flow[0]:
            assert by_flow == by_degree == (True, None)
            seen_true += 1
        else:
            # both name a genuine violator; the vertices may differ (see docstring)
            k = inst.k
            out_deg = [0] * inst.n
            for u, _ in inst.arcs:
                out_deg[u] += 1
            assert terminal_connectivity_nx(inst, by_flow[1]) < k
            assert out_deg[by_degree[1]] < k
            assert terminal_connectivity_nx(inst, by_degree[1]) < k
            seen_false += 1
    assert seen_true > 0 and seen_false > 0


def test_dag_criterion_rejects_cyclic_graphs():
    inst = make_instance(4, [(2, 3), (3, 2), (2, 0), (3, 1)], [0, 1], [1, 1], directed=True)
    assert not is_dag(inst)
    with pytest.raises(ValueError):
        is_k_T_connected_dag(inst)
    assert not is_dag(contract_counter_example())  # symmetric arcs form 2-cycles


def test_dag_criterion_hand_examples():
    # every non-terminal has out-degree >= 2 but 3 reaches terminals only through 2
    inst = make_instance(4, [(3, 2), (3, 0), (2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    assert is_k_T_connected_dag(inst) == (True, None)
    assert is_k_T_connected(inst) == (True, None)
    inst = make_instance(4, [(3, 2), (2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    assert is_k_T_connected_dag(inst) == (False, 3)
    assert is_k_T_connected(inst) == (False, 3)


# ---------------------------------------------------------------------------
# weighted FESAC [Def 5.3]
# ---------------------------------------------------------------------------


def test_fesac_weighted_witness():
    # terminals 0,1; v=2 (w 3) sees both, v=3 (w 2) sees only t1; capacities 3 and 2
    inst = make_instance(4, [(2, 0), (2, 1), (3, 0)], [0, 1], [3, 2], weights=[0, 0, 3, 2], directed=True)
    psi = fesac_witness_nx(inst)
    assert psi is not None
    assert psi[3] == {0: 2}
    assert sum(psi[2].values()) == 3
    assert set(psi[2]) <= {0, 1}
    loads = {0: 0, 1: 0}
    for units in psi.values():
        for t, u in units.items():
            loads[t] += u
    assert loads[0] <= 3 and loads[1] <= 2
    # v=2 must split its weight: 3 units cannot all go to t1 (only 1 unit left) nor to t2 (cap 2)
    assert len(psi[2]) == 2
    with pytest.raises(ValueError):
        feac_witness_nx(inst)
    res = check_preconditions(inst, mode=True)
    assert res["condition"] == "FESAC"
    assert res["feac"] is True
    assert res["k_T_connected"] is False  # vertex 3 has kappa 1


def test_fesac_fails_when_capacity_too_small():
    inst = make_instance(4, [(2, 0), (2, 1), (3, 0)], [0, 1], [1, 4], weights=[0, 0, 3, 2], directed=True)
    assert fesac_witness_nx(inst) is None  # vertex 3 (w 2) can only go to t1 (cap 1)
    res = check_preconditions(inst, mode=True)
    assert res["feac"] is False
    assert "FESAC fails" in res["message"]


# ---------------------------------------------------------------------------
# check_preconditions modes
# ---------------------------------------------------------------------------


def test_check_preconditions_modes():
    inst = running_example()
    off = check_preconditions(inst, mode=False)
    assert off["k_T_connected"] is None and off["feac"] is None
    assert off["message"]
    auto = check_preconditions(inst, mode="auto")
    assert auto["k_T_connected"] is False and auto["feac"] is True
    with pytest.raises(ValueError):
        check_preconditions(inst, mode="always")
    # large instance: 'auto' skips, True still runs
    n = AUTO_MAX_N + 1
    big = make_instance(n, [(v, 0) for v in range(1, n)], [0], [n - 1], directed=True)
    skipped = check_preconditions(big, mode="auto")
    assert skipped["k_T_connected"] is None and skipped["feac"] is None
    assert "skipped" in skipped["message"]
    small = make_instance(AUTO_MAX_N, [(v, 0) for v in range(1, AUTO_MAX_N)], [0], [AUTO_MAX_N - 1], directed=True)
    assert check_preconditions(small, mode="auto")["k_T_connected"] is True


def test_check_preconditions_witness_when_k_T_connected_is_a_real_witness():
    grid = nx.grid_2d_graph(3, 3)
    nodes = list(grid.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in grid.edges()]
    inst = make_instance(9, edges, [index[(0, 0)], index[(2, 2)]], sizes=[4, 5], directed=False)
    assert undirected_vertex_connectivity(inst) == 2
    res = check_preconditions(inst, mode=True)
    assert res["k_T_connected"] is True
    assert res["feac"] is True
    assert is_feac_witness(inst, res["witness"], essential_sets_by_definition(inst))
    assert feac_witness_nx(inst) is not None
