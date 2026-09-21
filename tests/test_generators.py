"""Graph-family generators: validity, meta, connectivity claims, determinism.

Runnable in isolation: ``pytest tests/test_generators.py``.
"""
from __future__ import annotations

import random

import networkx as nx
import pytest

from glsolver import generators as gen
from glsolver.instance import Instance, make_instance

UNDIRECTED_FAMILIES = {
    "complete": lambda seed: gen.complete_graph(8, 3, seed),
    "cycle": lambda seed: gen.cycle_graph(9, 2, seed),
    "wheel": lambda seed: gen.wheel_graph(9, 3, seed),
    "grid": lambda seed: gen.grid_graph(3, 4, 2, seed),
    "grid3d": lambda seed: gen.grid3d_graph(2, 3, 2, 3, seed),
    "harary": lambda seed: gen.harary_graph(11, 4, seed),
    "random_regular": lambda seed: gen.random_regular_graph(12, 4, 3, seed),
    "erdos_renyi": lambda seed: gen.erdos_renyi_graph(14, 0.5, 3, seed),
    "random_geometric": lambda seed: gen.random_geometric_graph(16, 0.55, 2, seed),
    "expander": lambda seed: gen.expander_graph(12, 4, 3, seed),
    "dense": lambda seed: gen.dense_graph(12, 4, seed),
    "sparse_k_connected": lambda seed: gen.sparse_k_connected(15, 3, seed),
    "adversarial_ladder": lambda seed: gen.adversarial_ladder(15, 3, seed),
}
DAG_FAMILIES = {
    "random_kT_dag": lambda seed: gen.random_kT_connected_dag(14, 3, seed, extra_out=1),
    "random_kT_dag_weighted": lambda seed: gen.random_kT_connected_dag(14, 3, seed, weighted=True, w_max=4),
    "layered_dag": lambda seed: gen.layered_dag(4, 3, 3, seed),
    "layered_dag_weighted": lambda seed: gen.layered_dag(3, 4, 2, seed, weighted=True),
}


def undirected_graph(inst: Instance) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(inst.n))
    G.add_edges_from(inst.undirected_edges or ())
    return G


def kappa_networkx(inst: Instance, v: int) -> int:
    """Independent [Def 3.2] check: max-flow on the vertex-split network (§3.1)."""
    K = inst.k + 1
    H = nx.DiGraph()
    for x in range(inst.n):
        H.add_edge((x, "in"), (x, "out"), capacity=K if x == v else 1)
    H.add_edge("s", (v, "in"), capacity=K)
    for x, y in inst.arcs:
        H.add_edge((x, "out"), (y, "in"), capacity=K)
    for t in inst.terminals:
        H.add_edge((t, "out"), "z", capacity=K)
    return int(nx.maximum_flow_value(H, "s", "z"))


def is_k_t_connected_networkx(inst: Instance) -> bool:
    return all(kappa_networkx(inst, v) >= inst.k for v in range(inst.n) if v not in inst.terminals)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def test_random_terminals():
    rng = random.Random(1)
    for _ in range(20):
        n = rng.randint(1, 15)
        k = rng.randint(1, n)
        ts = gen.random_terminals(n, k, rng)
        assert len(ts) == k == len(set(ts)) and all(0 <= t < n for t in ts)
        assert ts == sorted(ts)
    with pytest.raises(ValueError):
        gen.random_terminals(3, 4, rng)


@pytest.mark.parametrize("mode", gen._MODES)
def test_capacity_vector_modes(mode):
    rng = random.Random(7)
    for total, k in [(0, 1), (0, 3), (1, 3), (5, 5), (10, 4), (17, 1), (30, 6)]:
        if mode == "random_positive" and total < k:
            with pytest.raises(ValueError):
                gen.capacity_vector(total, k, rng, mode)
            continue
        caps = gen.capacity_vector(total, k, rng, mode)
        assert len(caps) == k and sum(caps) == total and min(caps) >= 0
        if mode == "balanced":
            assert max(caps) - min(caps) <= 1
        elif mode == "unbalanced" and total >= k - 1:
            assert sorted(caps) == [1] * (k - 1) + [total - k + 1]
        elif mode == "random_positive":
            assert min(caps) >= 1
        elif mode == "extreme":
            assert sorted(caps) == [0] * (k - 1) + [total]
    with pytest.raises(ValueError):
        gen.capacity_vector(4, 2, rng, "nonsense")


def test_capacity_vector_random_is_deterministic_and_varied():
    a = gen.capacity_vector(20, 4, random.Random(3), "random")
    b = gen.capacity_vector(20, 4, random.Random(3), "random")
    assert a == b
    seen = {tuple(gen.capacity_vector(20, 4, random.Random(s), "random")) for s in range(20)}
    assert len(seen) > 5


# ---------------------------------------------------------------------------
# undirected families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", sorted(UNDIRECTED_FAMILIES))
def test_undirected_family_is_valid_and_k_connected(family):
    inst = UNDIRECTED_FAMILIES[family](seed=3)
    inst.validate()
    assert not inst.directed and inst.undirected_edges
    assert inst.meta["family"] == family
    assert inst.meta["seed"] == 3
    assert inst.meta["claims_k_connected"] is True
    assert inst.meta["connectivity_verified"] is True
    G = undirected_graph(inst)
    assert nx.node_connectivity(G) >= inst.k
    # arcs are the symmetrized edges minus arcs leaving terminals
    tset = set(inst.terminals)
    expected = {(u, v) for u, v in G.edges() if u not in tset}
    expected |= {(v, u) for u, v in G.edges() if v not in tset}
    assert set(inst.arcs) == expected
    assert gen.is_k_t_connected(inst)


@pytest.mark.parametrize("family", sorted(UNDIRECTED_FAMILIES))
def test_undirected_family_is_deterministic(family):
    a = UNDIRECTED_FAMILIES[family](seed=11)
    b = UNDIRECTED_FAMILIES[family](seed=11)
    assert a == b and a.meta == b.meta


def test_random_families_vary_with_seed():
    for name in ("random_regular", "erdos_renyi", "random_geometric", "dense"):
        insts = [UNDIRECTED_FAMILIES[name](seed=s) for s in range(4)]
        assert len({(i.terminals, i.undirected_edges) for i in insts}) > 1, name


def test_structured_families_reject_impossible_k():
    with pytest.raises(ValueError):
        gen.cycle_graph(8, 3)
    with pytest.raises(ValueError):
        gen.wheel_graph(8, 4)
    with pytest.raises(ValueError):
        gen.grid_graph(3, 3, 3)
    with pytest.raises(ValueError):
        gen.grid3d_graph(2, 2, 2, 4)
    with pytest.raises(ValueError):
        gen.harary_graph(5, 5)
    with pytest.raises(ValueError):
        gen.adversarial_ladder(10, 3)


def test_harary_is_exactly_k_connected_and_sparse():
    for n, k in [(8, 2), (9, 3), (10, 4), (7, 5)]:
        inst = gen.harary_graph(n, k, 1)
        G = undirected_graph(inst)
        assert nx.node_connectivity(G) == k
        assert G.number_of_edges() == -(-k * n // 2)


def test_adversarial_ladder_connectivity():
    for n, k in [(6, 1), (12, 2), (12, 3), (16, 4)]:
        inst = gen.adversarial_ladder(n, k, 2)
        G = undirected_graph(inst)
        assert nx.node_connectivity(G) == k == inst.meta["connectivity"]
        assert inst.meta["cliques"] == n // k
        assert nx.diameter(G) >= n // k - 1
        closed = gen.adversarial_ladder(n, k, 2, closed=True)
        Gc = undirected_graph(closed)
        assert nx.node_connectivity(Gc) == closed.meta["connectivity"] == k + 1


def test_sparse_k_connected_adds_chords():
    inst = gen.sparse_k_connected(20, 3, 4, chords=5)
    base = gen.harary_graph(20, 3, 4)
    assert len(inst.undirected_edges) == len(base.undirected_edges) + 5
    assert inst.meta["chords"] == 5
    assert nx.node_connectivity(undirected_graph(inst)) >= 3


def test_erdos_renyi_without_requirement_reports_claim():
    inst = gen.erdos_renyi_graph(12, 0.05, 3, 1, require_k_connected=False)
    inst.validate()
    ok = nx.node_connectivity(undirected_graph(inst)) >= 3
    assert inst.meta["claims_k_connected"] is ok
    assert inst.meta["tries"] == 1


def test_erdos_renyi_rejection_gives_up():
    with pytest.raises(RuntimeError):
        gen.erdos_renyi_graph(12, 0.0, 2, 1)


def test_capacity_modes_propagate():
    for mode in gen._MODES:
        inst = gen.complete_graph(10, 3, 1, mode=mode)
        assert inst.meta["mode"] == mode
        assert sum(inst.capacities) == 7
    assert sorted(gen.complete_graph(10, 3, 1, mode="extreme").capacities) == [0, 0, 7]


# ---------------------------------------------------------------------------
# paper examples
# ---------------------------------------------------------------------------
def test_paper_running_example_shape():
    inst = gen.paper_running_example()
    assert (inst.n, inst.k, inst.m) == (9, 3, 12)
    assert inst.directed and inst.terminals == (0, 1, 2) and inst.capacities == (2, 2, 2)
    assert set(inst.arcs) == {(7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1),
                              (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2)}
    assert inst.meta["family"] == "paper_running_example"
    assert nx.is_directed_acyclic_graph(nx.DiGraph(inst.arcs))
    assert not is_k_t_connected_networkx(inst)  # v4 has out-degree 2 < k


def test_paper_contract_counterexample_shape():
    inst = gen.paper_contract_counterexample()
    assert (inst.n, inst.k) == (9, 3)
    assert not inst.directed
    assert len(inst.undirected_edges) == 12
    assert inst.m == 18  # 24 arcs minus the 6 leaving terminals
    assert inst.capacities == (2, 2, 2)
    assert nx.node_connectivity(undirected_graph(inst)) == 2
    assert inst.meta["claims_k_connected"] is False
    assert inst.meta["claims_kT_connected"] is True
    assert is_k_t_connected_networkx(inst)


def test_paper_essential_example_shape():
    inst = gen.paper_essential_example()
    assert (inst.n, inst.k, inst.m) == (13, 4, 20)
    assert inst.terminals == (0, 1, 2, 3) and inst.capacities == (2, 3, 2, 2)
    out = inst.out_adjacency()
    assert sorted(out[12]) == [4, 9, 10, 11]  # v13 -> v5, v10, v11, v12
    assert sorted(out[4]) == [0] and sorted(out[9]) == [3]  # v5 -> t1, v10 -> t4
    assert not nx.is_directed_acyclic_graph(nx.DiGraph(inst.arcs))  # v6..v9 cycle
    assert not is_k_t_connected_networkx(inst)


# ---------------------------------------------------------------------------
# directed families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", sorted(DAG_FAMILIES))
def test_dag_family_is_acyclic_and_kT_connected(family):
    inst = DAG_FAMILIES[family](seed=5)
    inst.validate()
    assert inst.directed
    assert inst.meta["dag"] is True and inst.meta["claims_kT_connected"] is True
    D = nx.DiGraph()
    D.add_nodes_from(range(inst.n))
    D.add_edges_from(inst.arcs)
    assert nx.is_directed_acyclic_graph(D)
    tset = set(inst.terminals)
    assert all(D.out_degree(v) >= inst.k for v in range(inst.n) if v not in tset)
    assert all(D.out_degree(t) == 0 for t in tset)
    assert is_k_t_connected_networkx(inst)  # [Lem 9.1]
    if "weighted" in family:
        assert inst.is_weighted
        assert all(1 <= inst.weights[v] <= 4 for v in range(inst.n) if v not in tset)
        assert sum(inst.capacities) == inst.total_weight
    assert DAG_FAMILIES[family](seed=5) == inst


def test_random_kT_dag_extra_out():
    plain = gen.random_kT_connected_dag(30, 3, 1)
    more = gen.random_kT_connected_dag(30, 3, 1, extra_out=2)
    assert more.m > plain.m
    out = more.out_adjacency()
    assert min(len(out[v]) for v in range(30) if v not in more.terminals) >= 3


def test_layered_dag_layers():
    inst = gen.layered_dag(5, 3, 2, 4)
    assert inst.n == 5 * 3 + 2 and inst.terminals == (15, 16)
    for u, v in inst.arcs:
        if v not in inst.terminals:
            assert 1 <= v // 3 - u // 3 <= 2


def test_random_kT_connected_digraph():
    for seed in range(3):
        inst = gen.random_kT_connected_digraph(12, 3, seed)
        inst.validate()
        assert inst.meta["family"] == "random_kT_digraph"
        assert inst.meta["connectivity_verified"] is True
        assert is_k_t_connected_networkx(inst)
        assert gen.random_kT_connected_digraph(12, 3, seed) == inst
    with pytest.raises(ValueError):
        gen.random_kT_connected_digraph(61, 2, 0)


def test_is_k_vertex_connected_matches_networkx():
    rng = random.Random(5)
    checked = 0
    for _ in range(120):
        n = rng.randint(2, 9)
        G = nx.gnp_random_graph(n, rng.random(), seed=rng.randrange(10**6))
        kappa = nx.node_connectivity(G)
        for k in range(0, n + 1):
            assert gen.is_k_vertex_connected(G, k) is (kappa >= k), (sorted(G.edges()), k)
            checked += 1
    assert checked > 500
    # structured cases: cycle (2), wheel (3), Petersen (3), complete (n-1), path (1)
    assert gen.is_k_vertex_connected(nx.cycle_graph(7), 2)
    assert not gen.is_k_vertex_connected(nx.cycle_graph(7), 3)
    assert gen.is_k_vertex_connected(nx.petersen_graph(), 3)
    assert not gen.is_k_vertex_connected(nx.petersen_graph(), 4)
    assert gen.is_k_vertex_connected(nx.complete_graph(6), 5)
    assert not gen.is_k_vertex_connected(nx.complete_graph(6), 6)
    assert gen.is_k_vertex_connected(nx.path_graph(4), 1)
    assert not gen.is_k_vertex_connected(nx.path_graph(4), 2)


def test_large_dense_graph_is_verified_by_degree_bound():
    inst = gen.dense_graph(120, 4, 3, density=0.8)
    assert inst.meta["claims_k_connected"] is True
    assert inst.meta["connectivity_verified"] is True


def test_terminal_connectivity_matches_networkx_flow():
    rng = random.Random(9)
    for _ in range(25):
        n = rng.randint(4, 11)
        k = rng.randint(1, min(4, n - 1))
        terminals = gen.random_terminals(n, k, rng)
        p = rng.random() * 0.6 + 0.1
        arcs = [
            (u, v) for u in range(n) if u not in terminals
            for v in range(n) if u != v and rng.random() < p
        ]
        thin = make_instance(
            n, arcs, terminals, gen.capacity_vector(n - k, k, rng, "random"), directed=True
        )
        for v in range(n):
            if v not in terminals:
                assert gen.terminal_connectivity(thin, v) == kappa_networkx(thin, v)
        assert gen.is_k_t_connected(thin) == is_k_t_connected_networkx(thin)
    with pytest.raises(ValueError):
        gen.terminal_connectivity(thin, thin.terminals[0])


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def test_weighted_variant():
    base = gen.harary_graph(12, 3, 2)
    w = gen.weighted_variant(base, 7, 4, slack=2)
    w.validate()
    assert w.is_weighted and w.arcs == base.arcs and w.terminals == base.terminals
    tset = set(w.terminals)
    assert all(1 <= w.weights[v] <= 4 for v in range(w.n) if v not in tset)
    assert all(w.weights[t] == 0 for t in tset)
    assert sum(w.capacities) == w.total_weight + 2
    assert w.meta["family"] == "weighted_variant" and w.meta["base_family"] == "harary"
    assert gen.weighted_variant(base, 7, 4, slack=2) == w
    directed = gen.weighted_variant(gen.paper_running_example(), 1, 3)
    assert directed.directed and directed.arcs == gen.paper_running_example().arcs


def test_directed_variant_keeps_kT_connectivity():
    base = gen.erdos_renyi_graph(18, 0.5, 3, 4)
    d = gen.directed_variant(base, 1, 0.6)
    d.validate()
    assert d.directed and d.terminals == base.terminals and d.capacities == base.capacities
    assert set(d.arcs) <= set(base.arcs)
    assert d.meta["dropped"] > 0
    assert d.meta["claims_kT_connected"] is True and d.meta["connectivity_verified"] is True
    assert is_k_t_connected_networkx(d)
    assert gen.directed_variant(base, 1, 0.6) == d
    # a degree-tight base (Harary) cannot lose any arc
    tight = gen.directed_variant(gen.harary_graph(10, 3, 1), 2, 0.9)
    assert tight.meta["dropped"] == 0
    assert is_k_t_connected_networkx(tight)
    with pytest.raises(ValueError):
        gen.directed_variant(d, 1, 0.5)


def test_directed_variant_unverified():
    base = gen.erdos_renyi_graph(18, 0.5, 3, 4)
    d = gen.directed_variant(base, 1, 0.6, verify=False)
    assert d.meta["claims_kT_connected"] is False
    assert d.meta["dropped"] == d.meta["proposed"] - sum(
        1 for a in _proposals(base, 1, 0.6) if a not in set(base.arcs)
    )


def _proposals(inst: Instance, seed: int, drop_fraction: float) -> list[tuple[int, int]]:
    """Replay of the proposal draw of ``directed_variant`` (same rng protocol)."""
    rng = random.Random(seed)
    edges = list(inst.undirected_edges or ())
    rng.shuffle(edges)
    out = []
    for u, v in edges:
        if rng.random() < drop_fraction:
            out.append((u, v) if rng.random() < 0.5 else (v, u))
    return out


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
def test_family_catalog_covers_all_families():
    cat = gen.family_catalog()
    expected = {
        "complete", "cycle", "wheel", "grid", "grid3d", "harary", "random_regular",
        "erdos_renyi", "random_geometric", "expander", "dense", "sparse_k_connected",
        "adversarial_ladder", "random_kT_dag", "layered_dag", "random_kT_digraph",
        "weighted_kT_dag", "directed_random", "paper_running_example",
        "paper_contract_counterexample", "paper_essential_example",
    }
    assert expected <= set(cat)
    for name, fn in cat.items():
        inst = fn(12, 2, 1)
        inst.validate()
        assert isinstance(inst, Instance)
        assert "family" in inst.meta and "seed" in inst.meta
        if name.startswith("paper_"):
            continue
        assert inst.k == 2
        assert inst.meta["seed"] == 1
        if inst.is_weighted:
            assert sum(inst.capacities) >= inst.total_weight
        else:
            assert max(inst.capacities) - min(inst.capacities) <= 1  # balanced
        if not inst.directed:
            assert nx.node_connectivity(undirected_graph(inst)) >= 2
        elif inst.meta.get("claims_kT_connected"):
            assert is_k_t_connected_networkx(inst)
        assert fn(12, 2, 1) == inst


def test_family_catalog_with_k3():
    cat = gen.family_catalog()
    for name in ("complete", "wheel", "grid3d", "harary", "random_regular", "expander",
                 "adversarial_ladder", "random_kT_dag", "layered_dag", "random_kT_digraph"):
        inst = cat[name](15, 3, 2)
        inst.validate()
        assert inst.k == 3
    with pytest.raises(ValueError):
        cat["cycle"](15, 3, 2)
