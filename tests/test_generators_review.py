"""Adversarial review of ``glsolver.generators``.

Every claim a generator makes (``meta["claims_k_connected"]``,
``meta["claims_kT_connected"]``, docstring connectivity values, capacity sums,
determinism, the paper examples) is re-checked here with *independent* tools:
``networkx.node_connectivity``, a NetworkX max-flow implementation of
``κ_G(v)`` [Def 3.2] on the vertex-split network of docs/paper_notes.md §3.1,
the ``k+1``-flow definition of essential terminals [Def 4.1], and the arc lists
transcribed from docs/algorithm.md / the generator docstrings.

Runnable in isolation: ``pytest tests/test_generators_review.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import pytest

from glsolver import generators as gen
from glsolver.instance import Instance, make_instance
from glsolver.oracle import bruteforce_partition
from glsolver.verify import verify_instance_parts

# ---------------------------------------------------------------------------
# independent checkers
# ---------------------------------------------------------------------------


def undirected_graph(inst: Instance) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(inst.n))
    G.add_edges_from(inst.undirected_edges or ())
    return G


def digraph(inst: Instance) -> nx.DiGraph:
    D = nx.DiGraph()
    D.add_nodes_from(range(inst.n))
    D.add_edges_from(inst.arcs)
    return D


def kappa_nx(n: int, arcs, terminals, v: int) -> int:
    """``κ_G(v)`` [Def 3.2] by NetworkX max-flow on the vertex-split network (§3.1)."""
    K = len(terminals) + 1
    H = nx.DiGraph()
    for x in range(n):
        H.add_edge((x, "in"), (x, "out"), capacity=K if x == v else 1)
    H.add_edge("s", (v, "in"), capacity=K)
    for x, y in arcs:
        H.add_edge((x, "out"), (y, "in"), capacity=K)
    for t in terminals:
        H.add_edge((t, "out"), "z", capacity=K)
    return int(nx.maximum_flow_value(H, "s", "z"))


def is_kT_nx(inst: Instance) -> bool:
    return all(
        kappa_nx(inst.n, inst.arcs, inst.terminals, v) >= inst.k
        for v in range(inst.n) if v not in inst.terminals
    )


def essential_nx(inst: Instance, v: int) -> tuple[int, frozenset[int]]:
    """``(κ_G(v), Ess_G(v))`` by the literal definition [Def 4.1]: ``t`` is
    essential iff deleting the vertex ``t`` lowers ``κ`` by one."""
    base = kappa_nx(inst.n, inst.arcs, inst.terminals, v)
    ess = set()
    for t in inst.terminals:
        arcs = [(a, b) for a, b in inst.arcs if a != t and b != t]
        terms = [x for x in inst.terminals if x != t]
        if kappa_nx(inst.n, arcs, terms, v) == base - 1:
            ess.add(t)
    return base, frozenset(ess)


def assert_claims_hold(inst: Instance) -> None:
    """Whatever the instance claims in ``meta`` must be true."""
    inst.validate()
    assert "family" in inst.meta and "seed" in inst.meta
    if not inst.directed:
        assert inst.undirected_edges is not None
        tset = set(inst.terminals)
        G = undirected_graph(inst)
        expected = {(u, v) for u, v in G.edges() if u not in tset}
        expected |= {(v, u) for u, v in G.edges() if v not in tset}
        assert set(inst.arcs) == expected
        if inst.meta.get("claims_k_connected"):
            assert nx.node_connectivity(G) >= inst.k, inst.name
    if inst.meta.get("claims_kT_connected"):
        assert is_kT_nx(inst), inst.name
        assert gen.is_k_t_connected(inst), inst.name
    if inst.meta.get("dag"):
        assert nx.is_directed_acyclic_graph(digraph(inst)), inst.name
    if inst.is_weighted:
        assert inst.weights is not None
        assert all(inst.weights[t] == 0 for t in inst.terminals)
        assert all(inst.weights[v] >= 1 for v in range(inst.n) if v not in inst.terminals)
        assert sum(inst.capacities) >= inst.total_weight
    else:
        assert sum(inst.capacities) == inst.n - inst.k


# ---------------------------------------------------------------------------
# builders under review (name -> seed -> Instance)
# ---------------------------------------------------------------------------
BUILDERS: dict[str, Callable[[int], Instance]] = {
    "complete": lambda s: gen.complete_graph(7, 3, s, "random"),
    "cycle": lambda s: gen.cycle_graph(9, 2, s, "unbalanced"),
    "wheel": lambda s: gen.wheel_graph(8, 3, s, "extreme"),
    "grid": lambda s: gen.grid_graph(3, 4, 2, s, "random_positive"),
    "grid3d": lambda s: gen.grid3d_graph(2, 2, 3, 3, s),
    "harary": lambda s: gen.harary_graph(10, 4, s, "random"),
    "random_regular": lambda s: gen.random_regular_graph(10, 4, 3, s, "random"),
    "erdos_renyi": lambda s: gen.erdos_renyi_graph(12, 0.5, 3, s, "random"),
    "random_geometric": lambda s: gen.random_geometric_graph(12, 0.6, 2, s, "random"),
    "expander": lambda s: gen.expander_graph(10, 4, 3, s),
    "dense": lambda s: gen.dense_graph(10, 4, s, density=0.7),
    "sparse_k_connected": lambda s: gen.sparse_k_connected(12, 3, s, chords=3),
    "adversarial_ladder": lambda s: gen.adversarial_ladder(12, 3, s, "random"),
    "adversarial_ladder_closed": lambda s: gen.adversarial_ladder(12, 3, s, closed=True),
    "random_kT_dag": lambda s: gen.random_kT_connected_dag(12, 3, s, "random", extra_out=1),
    "random_kT_dag_weighted": lambda s: gen.random_kT_connected_dag(
        12, 3, s, "random", weighted=True, w_max=4
    ),
    "layered_dag": lambda s: gen.layered_dag(3, 3, 3, s, "random"),
    "layered_dag_weighted": lambda s: gen.layered_dag(3, 3, 2, s, "random", weighted=True),
    "random_kT_digraph": lambda s: gen.random_kT_connected_digraph(10, 3, s, "random"),
    "weighted_variant": lambda s: gen.weighted_variant(gen.harary_graph(10, 3, s), s, 4, slack=2),
    "weighted_variant_dag": lambda s: gen.weighted_variant(
        gen.random_kT_connected_dag(10, 2, s), s, 3
    ),
    "directed_variant": lambda s: gen.directed_variant(gen.erdos_renyi_graph(12, 0.5, 3, s), s, 0.6),
    "directed_variant_full": lambda s: gen.directed_variant(gen.dense_graph(10, 3, s), s, 1.0),
    "paper_running_example": lambda s: gen.paper_running_example(),
    "paper_contract_counterexample": lambda s: gen.paper_contract_counterexample(),
    "paper_essential_example": lambda s: gen.paper_essential_example(),
}


def digest(inst: Instance) -> str:
    blob = json.dumps(
        [inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities), inst.weights,
         inst.directed, inst.undirected_edges, inst.name, inst.meta],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def digest_all(seed: int = 5) -> dict[str, str]:
    """Digest of every builder at ``seed`` (also run in a subprocess, see below)."""
    return {name: digest(fn(seed)) for name, fn in BUILDERS.items()}


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builder_is_deterministic_and_claims_hold(name: str):
    a = BUILDERS[name](3)
    b = BUILDERS[name](3)
    assert a == b and a.meta == b.meta
    assert digest(a) == digest(b)
    assert_claims_hold(a)


def test_random_builders_vary_with_seed():
    for name in ("random_regular", "erdos_renyi", "random_geometric", "dense", "random_kT_dag",
                 "layered_dag", "random_kT_digraph", "directed_variant", "weighted_variant",
                 "adversarial_ladder", "sparse_k_connected", "complete"):
        digests = {digest(BUILDERS[name](s)) for s in range(5)}
        assert len(digests) > 1, name


def test_builders_are_deterministic_across_processes():
    """Same seed, different interpreter and different ``PYTHONHASHSEED`` must give
    byte-identical instances (no dependence on set/dict iteration of hashed objects)."""
    here = Path(__file__).resolve()
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(here.parent)!r})\n"
        "import test_generators_review as m\n"
        "print(json.dumps(m.digest_all(5), sort_keys=True))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "987654"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=240, check=True)
    remote = json.loads(out.stdout.strip().splitlines()[-1])
    assert remote == digest_all(5)


# ---------------------------------------------------------------------------
# undirected families: exact connectivity values claimed in the docstrings
# ---------------------------------------------------------------------------
def test_structured_families_have_documented_exact_connectivity():
    cases = [
        (gen.complete_graph(6, 5, 1), 5),
        (gen.complete_graph(6, 2, 1), 5),
        (gen.cycle_graph(7, 2, 1), 2),
        (gen.cycle_graph(5, 1, 1), 2),
        (gen.wheel_graph(4, 3, 1), 3),
        (gen.wheel_graph(9, 3, 1), 3),
        (gen.grid_graph(1, 5, 1, 1), 1),
        (gen.grid_graph(2, 2, 2, 1), 2),
        (gen.grid_graph(3, 5, 2, 1), 2),
        (gen.grid3d_graph(1, 1, 4, 1, 1), 1),
        (gen.grid3d_graph(1, 3, 3, 2, 1), 2),
        (gen.grid3d_graph(2, 2, 2, 3, 1), 3),
        (gen.grid3d_graph(2, 3, 2, 3, 1), 3),
    ]
    for inst, expected in cases:
        assert nx.node_connectivity(undirected_graph(inst)) == expected, inst.name
        assert inst.meta["claims_k_connected"] is True
        assert inst.meta["connectivity_verified"] is True
        assert inst.k <= expected


@pytest.mark.parametrize("n,k", [(2, 1), (5, 1), (6, 2), (7, 3), (8, 5), (9, 7), (10, 9), (11, 4), (12, 5)])
def test_harary_is_exactly_k_connected(n: int, k: int):
    inst = gen.harary_graph(n, k, 2)
    G = undirected_graph(inst)
    assert nx.node_connectivity(G) == k
    # the classical ceil(kn/2) edge count holds for k >= 2; networkx's H_{1,n} is a path
    assert G.number_of_edges() == (-(-k * n // 2) if k >= 2 else n - 1)
    assert_claims_hold(inst)


def test_harary_docstring_edge_count_holds_for_k_equals_one():
    """Review finding: the docstring once claimed ceil(kn/2) edges for every k,
    but nx.hkn_harary_graph(1, n) is a path with n - 1 edges (n = 5: 4, not 3).
    The docstring now states the k = 1 exception; check both the count and
    that the exception is documented."""
    for n in (3, 5, 8):
        inst = gen.harary_graph(n, 1, 0)
        assert len(inst.undirected_edges) == n - 1
        assert nx.is_tree(undirected_graph(inst))
    doc = gen.harary_graph.__doc__ or ""
    assert "k >= 2" in doc and "n - 1" in doc


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_adversarial_ladder_exact_connectivity(k: int):
    for r in (2, 3, 4, 5):
        for seed in range(3):
            n = k * r
            open_ = gen.adversarial_ladder(n, k, seed)
            assert nx.node_connectivity(undirected_graph(open_)) == k == open_.meta["connectivity"]
            assert open_.meta["cliques"] == r and open_.meta["closed"] is False
            closed = gen.adversarial_ladder(n, k, seed, closed=True)
            expected = k + 1 if r >= 3 else k
            assert closed.meta["connectivity"] == expected
            assert nx.node_connectivity(undirected_graph(closed)) == expected
            # r cliques of size k joined by perfect matchings: |E| = r*C(k,2) + matchings*k
            matchings = (r - 1) + (1 if r >= 3 else 0)
            assert len(closed.undirected_edges) == r * k * (k - 1) // 2 + matchings * k
            assert len(open_.undirected_edges) == r * k * (k - 1) // 2 + (r - 1) * k
    with pytest.raises(ValueError):
        gen.adversarial_ladder(k, k, 0)  # a single clique
    if k > 1:
        with pytest.raises(ValueError):
            gen.adversarial_ladder(2 * k + 1, k, 0)  # not divisible


def test_sparse_k_connected_chords_and_connectivity():
    for n, k, chords in [(8, 2, 0), (10, 3, 4), (12, 4, 100), (6, 5, 3)]:
        inst = gen.sparse_k_connected(n, k, 1, chords=chords)
        base = gen.harary_graph(n, k, 1)
        possible = n * (n - 1) // 2 - len(base.undirected_edges)
        assert inst.meta["chords"] == min(chords, possible)
        assert len(inst.undirected_edges) == len(base.undirected_edges) + inst.meta["chords"]
        assert set(base.undirected_edges) <= set(inst.undirected_edges)
        assert nx.node_connectivity(undirected_graph(inst)) >= k
        assert_claims_hold(inst)


def test_expander_and_dense_are_relabelled_copies_of_their_base_family():
    e = gen.expander_graph(10, 4, 3, 7, "random")
    r = gen.random_regular_graph(10, 4, 3, 7, "random")
    assert (e.undirected_edges, e.terminals, e.capacities) == (r.undirected_edges, r.terminals, r.capacities)
    assert e.meta["family"] == "expander" and e.meta["expander_whp"] is True
    assert all(d == 4 for _, d in undirected_graph(e).degree())
    d = gen.dense_graph(10, 3, 7, "random", density=0.7)
    b = gen.erdos_renyi_graph(10, 0.7, 3, 7, "random")
    assert (d.undirected_edges, d.terminals, d.capacities) == (b.undirected_edges, b.terminals, b.capacities)
    assert d.meta["family"] == "dense" and d.meta["density"] == 0.7


def test_unrequired_sampling_reports_a_truthful_claim():
    """With ``require_k_connected=False`` the first sample is returned and the
    claim must equal the exact connectivity test."""
    seen_false = seen_true = 0
    for seed in range(12):
        for inst in (
            gen.erdos_renyi_graph(10, 0.25, 2, seed, require_k_connected=False),
            gen.random_geometric_graph(10, 0.4, 2, seed, require_k_connected=False),
            gen.random_regular_graph(10, 3, 3, seed, require_k_connected=False),
        ):
            actual = nx.node_connectivity(undirected_graph(inst)) >= inst.k
            assert inst.meta["claims_k_connected"] is actual, (inst.name, seed)
            assert inst.meta["connectivity_verified"] is True
            assert inst.meta["tries"] == 1
            seen_false += not actual
            seen_true += actual
    assert seen_false > 0 and seen_true > 0


def test_rejection_sampling_gives_up_with_runtime_error():
    with pytest.raises(RuntimeError):
        gen.erdos_renyi_graph(8, 0.0, 1, 0)
    with pytest.raises(RuntimeError):
        gen.random_geometric_graph(8, 0.0, 1, 0)
    with pytest.raises(RuntimeError):
        gen.random_kT_connected_digraph(6, 2, 0, p=0.0)


def test_structured_families_reject_impossible_parameters():
    bad = [
        lambda: gen.complete_graph(4, 4),
        lambda: gen.cycle_graph(2, 1),
        lambda: gen.cycle_graph(6, 3),
        lambda: gen.wheel_graph(3, 1),
        lambda: gen.wheel_graph(6, 4),
        lambda: gen.grid_graph(1, 4, 2),
        lambda: gen.grid_graph(3, 3, 3),
        lambda: gen.grid3d_graph(1, 1, 3, 2),
        lambda: gen.grid3d_graph(2, 2, 2, 4),
        lambda: gen.harary_graph(4, 4),
        lambda: gen.harary_graph(4, 0),
        lambda: gen.random_regular_graph(9, 3, 2),  # n*d odd
        lambda: gen.random_regular_graph(6, 2, 3),  # d < k
        lambda: gen.random_kT_connected_dag(3, 3),
        lambda: gen.random_kT_connected_digraph(3, 3),
        lambda: gen.random_kT_connected_digraph(61, 2),
        lambda: gen.layered_dag(0, 2, 1),
        lambda: gen.sparse_k_connected(5, 5),
        lambda: gen.random_terminals(3, 0, random.Random(0)),
        lambda: gen.capacity_vector(-1, 2, random.Random(0), "balanced"),
        lambda: gen.capacity_vector(3, 0, random.Random(0), "balanced"),
        lambda: gen.capacity_vector(3, 2, random.Random(0), "bogus"),
        lambda: gen.weighted_variant(gen.paper_running_example(), 0, 0),
        lambda: gen.weighted_variant(gen.paper_running_example(), 0, 2, slack=-1),
        lambda: gen.directed_variant(gen.paper_running_example(), 0, 0.5),
        lambda: gen.directed_variant(gen.harary_graph(6, 2, 0), 0, 1.5),
        lambda: gen.terminal_connectivity(gen.paper_running_example(), 0),
    ]
    for i, fn in enumerate(bad):
        with pytest.raises(ValueError):
            fn()
        _ = i


# ---------------------------------------------------------------------------
# DAG and digraph families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,k,extra_out", [(4, 3, 0), (5, 1, 2), (6, 5, 0), (9, 2, 5), (12, 4, 1), (16, 3, 0)])
def test_random_kT_dag_claims(n: int, k: int, extra_out: int):
    for seed in range(3):
        for weighted in (False, True):
            inst = gen.random_kT_connected_dag(
                n, k, seed, mode="random", extra_out=extra_out, weighted=weighted, w_max=3
            )
            assert_claims_hold(inst)
            D = digraph(inst)
            tset = set(inst.terminals)
            assert nx.is_directed_acyclic_graph(D)
            assert all(D.out_degree(t) == 0 for t in tset)
            degs = [D.out_degree(v) for v in range(n) if v not in tset]
            assert min(degs) >= k  # [Lem 9.1]
            assert max(degs) <= k + extra_out  # docstring: k + extra_out arcs, capped
            assert is_kT_nx(inst)
            assert inst.meta["claims_kT_connected"] is True and inst.meta["dag"] is True
            assert inst.meta["extra_out"] == extra_out and inst.meta["weighted"] is weighted
            if weighted:
                assert sum(inst.capacities) == inst.total_weight
                assert all(1 <= inst.weights[v] <= 3 for v in range(n) if v not in tset)
            else:
                assert inst.weights is None and sum(inst.capacities) == n - k
            # such an instance always admits a partition [Cor gl-dag-unweighted] / §1.4
            if n <= 9:
                st, parts = bruteforce_partition(inst)
                assert st == "ok" and verify_instance_parts(inst, parts).valid


@pytest.mark.parametrize("layers,width,k", [(1, 1, 1), (2, 1, 3), (3, 2, 2), (2, 5, 4), (4, 3, 1)])
def test_layered_dag_claims(layers: int, width: int, k: int):
    for seed in range(3):
        for weighted in (False, True):
            inst = gen.layered_dag(layers, width, k, seed, "random", weighted=weighted, w_max=3)
            assert_claims_hold(inst)
            nt = layers * width
            assert inst.n == nt + k and inst.terminals == tuple(range(nt, nt + k))
            D = digraph(inst)
            for u, v in inst.arcs:
                assert u < nt
                if v < nt:
                    assert 1 <= v // width - u // width <= 2
            degs = [D.out_degree(u) for u in range(nt)]
            assert min(degs) >= k and max(degs) <= k + 2
            assert nx.is_directed_acyclic_graph(D)
            assert is_kT_nx(inst)
            if weighted:
                assert sum(inst.capacities) == inst.total_weight
            else:
                assert sum(inst.capacities) == nt


def test_random_kT_digraph_claims():
    for seed in range(4):
        for n, k in [(5, 1), (6, 2), (9, 3), (12, 4)]:
            inst = gen.random_kT_connected_digraph(n, k, seed, "random")
            assert_claims_hold(inst)
            assert is_kT_nx(inst)
            assert inst.meta["tries"] >= 1 and inst.meta["p"] == min(1.0, max(0.3, 2.0 * k / (n - 1)))
            assert all(u not in inst.terminals for u, _ in inst.arcs)
    full = gen.random_kT_connected_digraph(6, 2, 0, p=1.0)
    assert full.m == (6 - 2) * 5  # every non-terminal points everywhere
    assert full.meta["tries"] == 1


# ---------------------------------------------------------------------------
# k-T-connectivity and k-connectivity tests of the module vs NetworkX
# ---------------------------------------------------------------------------
def test_is_k_vertex_connected_matches_networkx_including_degenerate_graphs():
    rng = random.Random(3)
    checked = 0
    for _ in range(150):
        n = rng.randint(1, 10)
        G = nx.gnp_random_graph(n, rng.random(), seed=rng.randrange(10**6))
        if n > 2 and rng.random() < 0.3:  # isolate a vertex
            G.remove_edges_from(list(G.edges(rng.randrange(n))))
        kappa = nx.node_connectivity(G) if n > 1 else 0
        for k in range(0, n + 2):
            assert gen.is_k_vertex_connected(G, k) is (kappa >= k), (sorted(G.edges()), n, k)
            checked += 1
    assert checked > 800
    # graphs below the 2*delta >= n + k - 2 shortcut but k-connected: Petersen, Q3, K_{3,3}
    for G, kappa in [(nx.petersen_graph(), 3), (nx.hypercube_graph(3), 3),
                     (nx.complete_bipartite_graph(3, 3), 3), (nx.complete_graph(1), 0),
                     (nx.empty_graph(3), 0), (nx.path_graph(2), 1)]:
        for k in range(0, 6):
            assert gen.is_k_vertex_connected(G, k) is (kappa >= k), (G, k)


def test_terminal_connectivity_matches_networkx_flow_on_random_digraphs():
    rng = random.Random(9)
    for _ in range(30):
        n = rng.randint(2, 11)
        k = rng.randint(1, min(4, n - 1))
        terminals = gen.random_terminals(n, k, rng)
        p = rng.random() * 0.7 + 0.05
        arcs = [
            (u, v) for u in range(n) if u not in terminals
            for v in range(n) if u != v and rng.random() < p
        ]
        inst = make_instance(n, arcs, terminals, gen.capacity_vector(n - k, k, rng, "random"),
                             directed=True)
        for v in range(n):
            if v not in terminals:
                assert gen.terminal_connectivity(inst, v) == kappa_nx(n, inst.arcs, terminals, v)
        assert gen.is_k_t_connected(inst) is is_kT_nx(inst)
    # k-connected undirected => k-T-connected for any T (paper_notes §1.1)
    for seed in range(4):
        inst = gen.harary_graph(9, 3, seed)
        assert gen.is_k_t_connected(inst) and is_kT_nx(inst)


# ---------------------------------------------------------------------------
# paper examples (arc lists transcribed independently from the docs)
# ---------------------------------------------------------------------------
def _v(j: int) -> int:  # paper's v_j -> internal id
    return j - 1


def _t(i: int) -> int:  # paper's t_i -> internal id
    return i - 1


# docs/algorithm.md "The running example": twelve arcs
DOC_RUNNING_ARCS = {
    (_v(8), _v(4)), (_v(8), _v(5)), (_v(5), _v(4)), (_v(4), _t(1)), (_v(4), _t(2)), (_v(5), _t(2)),
    (_v(9), _v(6)), (_v(9), _v(7)), (_v(6), _v(7)), (_v(6), _t(2)), (_v(7), _t(2)), (_v(7), _t(3)),
}
# docs/algorithm.md: Ess(v4)=Ess(v5)=Ess(v8)={t1,t2}, Ess(v6)=Ess(v7)=Ess(v9)={t2,t3}, kappa = 2
DOC_RUNNING_ESS = {
    _v(4): {_t(1), _t(2)}, _v(5): {_t(1), _t(2)}, _v(8): {_t(1), _t(2)},
    _v(6): {_t(2), _t(3)}, _v(7): {_t(2), _t(3)}, _v(9): {_t(2), _t(3)},
}
# generator docstring of paper_essential_example: v13 -> v5, v11, v12, v10; v11 -> v6, v7, v8;
# v12 -> v7, v8, v9; cycle v6 -> v7 -> v8 -> v9 -> v6; v5, v6 -> t1; v7 -> t2; v8 -> t3; v9, v10 -> t4
DOC_ESSENTIAL_ARCS = {
    (_v(13), _v(5)), (_v(13), _v(11)), (_v(13), _v(12)), (_v(13), _v(10)),
    (_v(11), _v(6)), (_v(11), _v(7)), (_v(11), _v(8)),
    (_v(12), _v(7)), (_v(12), _v(8)), (_v(12), _v(9)),
    (_v(6), _v(7)), (_v(7), _v(8)), (_v(8), _v(9)), (_v(9), _v(6)),
    (_v(5), _t(1)), (_v(6), _t(1)), (_v(7), _t(2)), (_v(8), _t(3)), (_v(9), _t(4)), (_v(10), _t(4)),
}
# tests/helpers_small_graphs.py (written from the paper by a different agent)
HELPERS_ESSENTIAL_EXPECTED = {
    4: {0}, 5: {0}, 6: {1}, 7: {2}, 8: {3}, 9: {3}, 10: {0, 1}, 11: {1, 2}, 12: {0, 1, 2, 3},
}
HELPERS_ESSENTIAL_WITNESS = {12: 2, 10: 1, 11: 1, 4: 0, 5: 0, 6: 1, 7: 2, 8: 3, 9: 3}
# generator docstring of paper_contract_counterexample
DOC_CONTRACT_EDGES = {
    (0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 8), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (3, 8),
}


def test_paper_running_example_matches_docs():
    inst = gen.paper_running_example()
    assert (inst.n, inst.k, inst.m) == (9, 3, 12)
    assert inst.directed and inst.terminals == (0, 1, 2) and inst.capacities == (2, 2, 2)
    assert set(inst.arcs) == DOC_RUNNING_ARCS
    assert inst.meta["family"] == "paper_running_example"
    assert inst.meta["claims_kT_connected"] is False
    assert nx.is_directed_acyclic_graph(digraph(inst))
    for v in range(3, 9):
        kappa, ess = essential_nx(inst, v)
        assert kappa == 2, v
        assert ess == DOC_RUNNING_ESS[v], v
    assert not is_kT_nx(inst) and not gen.is_k_t_connected(inst)
    # docs/algorithm.md witness phi(v4)=t2, phi(v5)=t1, phi(v6)=t3, phi(v7)=t2, phi(v8)=t1, phi(v9)=t3
    phi = {_v(4): _t(2), _v(5): _t(1), _v(6): _t(3), _v(7): _t(2), _v(8): _t(1), _v(9): _t(3)}
    counts = [0, 0, 0]
    for v, t in phi.items():
        assert t in DOC_RUNNING_ESS[v]
        counts[t] += 1
    assert tuple(counts) == inst.capacities
    st, parts = bruteforce_partition(inst)
    assert st == "ok" and verify_instance_parts(inst, parts).valid
    assert [set(p) for p in parts] == [{0, 3, 7}, {1, 4, 5}, {2, 6, 8}]  # unique partition


def test_paper_essential_example_matches_docstring_and_expected_essentials():
    inst = gen.paper_essential_example()
    assert (inst.n, inst.k, inst.m) == (13, 4, 20)
    assert inst.terminals == (0, 1, 2, 3) and inst.capacities == (2, 3, 2, 2)
    assert set(inst.arcs) == DOC_ESSENTIAL_ARCS
    assert inst.meta["claims_kT_connected"] is False
    assert not nx.is_directed_acyclic_graph(digraph(inst))
    assert nx.find_cycle(digraph(inst)) is not None
    for v, expected in HELPERS_ESSENTIAL_EXPECTED.items():
        kappa, ess = essential_nx(inst, v)
        assert ess == frozenset(expected), (v, sorted(ess))
        assert 1 <= kappa <= 4
    assert essential_nx(inst, 12)[0] == 4  # v13 is the only 4-T-connected vertex
    assert not is_kT_nx(inst)
    counts = [0] * 4
    for v, t in HELPERS_ESSENTIAL_WITNESS.items():
        assert t in HELPERS_ESSENTIAL_EXPECTED[v]
        counts[t] += 1
    assert tuple(counts) == inst.capacities  # the paper's witness respects c = (2,3,2,2)
    st, parts = bruteforce_partition(inst)
    assert st == "ok" and verify_instance_parts(inst, parts).valid


def test_paper_contract_counterexample_matches_docstring():
    inst = gen.paper_contract_counterexample()
    assert (inst.n, inst.k) == (9, 3) and not inst.directed
    assert set(inst.undirected_edges) == DOC_CONTRACT_EDGES and len(inst.undirected_edges) == 12
    assert inst.capacities == (2, 2, 2) and inst.sizes == (3, 3, 3)
    assert inst.m == 18  # 24 arcs minus the 6 leaving terminals
    G = undirected_graph(inst)
    assert nx.node_connectivity(G) == 2
    assert all(G.degree(t) == 2 for t in inst.terminals)
    assert all(G.degree(v) == 3 for v in range(3, 9))
    for v in range(3, 9):
        assert kappa_nx(inst.n, inst.arcs, inst.terminals, v) == 3
    assert is_kT_nx(inst) and gen.is_k_t_connected(inst)
    assert inst.meta["claims_k_connected"] is False
    assert inst.meta["claims_kT_connected"] is True
    assert inst.meta["connectivity_verified"] is True
    st, parts = bruteforce_partition(inst)
    assert st == "ok" and verify_instance_parts(inst, parts).valid


def test_paper_examples_ignore_catalog_arguments():
    cat = gen.family_catalog()
    for name in ("paper_running_example", "paper_contract_counterexample", "paper_essential_example"):
        assert cat[name](50, 7, 9) == cat[name](3, 1, 0) == getattr(gen, name)()


# ---------------------------------------------------------------------------
# capacity vectors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", gen._MODES)
def test_capacity_vector_sums_and_shapes(mode: str):
    rng = random.Random(11)
    for total in range(0, 9):
        for k in range(1, 6):
            if mode == "random_positive" and total < k:
                with pytest.raises(ValueError):
                    gen.capacity_vector(total, k, rng, mode)
                continue
            caps = gen.capacity_vector(total, k, rng, mode)
            assert len(caps) == k and sum(caps) == total and min(caps) >= 0, (mode, total, k, caps)
            assert all(isinstance(c, int) for c in caps)
            if mode == "balanced":
                assert max(caps) - min(caps) <= 1
            elif mode == "unbalanced":
                ones = min(k - 1, total)
                assert sorted(caps) == sorted([1] * ones + [0] * (k - 1 - ones) + [total - ones])
            elif mode == "random_positive":
                assert min(caps) >= 1
            elif mode == "extreme":
                assert sorted(caps) == [0] * (k - 1) + [total]


def test_capacity_vector_random_modes_cover_every_composition():
    weak = {tuple(gen.capacity_vector(3, 2, random.Random(s), "random")) for s in range(200)}
    assert weak == {(0, 3), (1, 2), (2, 1), (3, 0)}
    positive = {tuple(gen.capacity_vector(4, 2, random.Random(s), "random_positive")) for s in range(200)}
    assert positive == {(1, 3), (2, 2), (3, 1)}
    weak3 = {tuple(gen.capacity_vector(2, 3, random.Random(s), "random")) for s in range(300)}
    assert len(weak3) == 6  # C(4,2) weak compositions of 2 into 3 parts
    # the "big" entry of balanced/unbalanced/extreme is at a random position
    for mode in ("balanced", "unbalanced", "extreme"):
        positions = {gen.capacity_vector(7, 3, random.Random(s), mode).index(3 if mode == "balanced" else max(gen.capacity_vector(7, 3, random.Random(s), mode))) for s in range(40)}
        assert len(positions) == 3, mode


def test_capacity_modes_propagate_to_all_families():
    for mode in gen._MODES:
        for inst in (
            gen.complete_graph(9, 3, 2, mode),
            gen.random_kT_connected_dag(9, 3, 2, mode),
            gen.layered_dag(2, 3, 3, 2, mode),
            gen.random_kT_connected_digraph(9, 3, 2, mode),
            gen.random_kT_connected_dag(9, 3, 2, mode, weighted=True, w_max=2),
        ):
            assert inst.meta["mode"] == mode
            assert_claims_hold(inst)
            if mode == "balanced":
                assert max(inst.capacities) - min(inst.capacities) <= 1
            elif mode == "extreme":
                assert sorted(inst.capacities)[:-1] == [0, 0]
            elif mode == "random_positive":
                assert min(inst.capacities) >= 1


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def test_weighted_variant_produces_valid_solvable_instances():
    bases = [
        gen.harary_graph(8, 3, 1),
        gen.complete_graph(6, 2, 1, "extreme"),
        gen.random_kT_connected_dag(8, 3, 1),
        gen.random_kT_connected_dag(8, 2, 1, weighted=True, w_max=5),  # already weighted
        gen.random_kT_connected_digraph(7, 2, 1),
        gen.paper_contract_counterexample(),
        gen.paper_running_example(),
    ]
    for base in bases:
        for seed, w_max, slack in [(0, 1, 0), (1, 3, 0), (2, 4, 2), (3, 2, 7)]:
            w = gen.weighted_variant(base, seed, w_max, slack=slack)
            w.validate()
            assert w.is_weighted and w.directed == base.directed
            assert w.arcs == base.arcs and w.terminals == base.terminals
            assert w.undirected_edges == base.undirected_edges
            tset = set(w.terminals)
            assert all(1 <= w.weights[v] <= w_max for v in range(w.n) if v not in tset)
            assert all(w.weights[t] == 0 for t in tset)
            assert sum(w.capacities) == w.total_weight + slack
            assert w.meta["family"] == "weighted_variant"
            assert w.meta["base_family"] == base.meta["family"]
            assert w.meta["seed"] == seed and w.meta["w_max"] == w_max and w.meta["slack"] == slack
            assert w.name == f"{base.name}_weighted_s{seed}"
            assert gen.weighted_variant(base, seed, w_max, slack=slack) == w
            assert_claims_hold(w)
            if base.meta.get("claims_k_connected") or base.meta.get("claims_kT_connected"):
                # k-T-connected => a partition within c_t + w_max - 1 exists [Thm weighted-k-t-conn]
                st, parts = bruteforce_partition(w)
                assert st == "ok", (base.name, seed, w_max, slack)
                assert verify_instance_parts(w, parts).valid
        assert gen.weighted_variant(base, 0, 3) != gen.weighted_variant(base, 1, 3)


def test_directed_variant_keeps_kT_connectivity_for_every_seed():
    bases = [gen.erdos_renyi_graph(12, 0.5, 3, s) for s in range(3)]
    bases += [gen.dense_graph(9, 2, 4), gen.wheel_graph(8, 3, 1), gen.complete_graph(7, 4, 2)]
    dropped_total = 0
    for base in bases:
        for seed in range(3):
            for frac in (0.0, 0.4, 1.0):
                d = gen.directed_variant(base, seed, frac)
                d.validate()
                assert d.directed and d.terminals == base.terminals
                assert d.capacities == base.capacities and d.weights == base.weights
                assert set(d.arcs) <= set(base.arcs)
                assert d.meta["dropped"] == len(base.arcs) - len(d.arcs)
                assert d.meta["proposed"] >= d.meta["dropped"]
                assert d.meta["claims_kT_connected"] is True
                assert d.meta["connectivity_verified"] is True
                assert d.meta["base_family"] == base.meta["family"]
                assert is_kT_nx(d), (base.name, seed, frac)
                assert gen.is_k_t_connected(d)
                assert gen.directed_variant(base, seed, frac) == d
                if frac == 0.0:
                    assert d.arcs == base.arcs and d.meta["dropped"] == 0 == d.meta["proposed"]
                dropped_total += d.meta["dropped"]
    assert dropped_total > 0
    # a degree-tight base (Harary H_{k,n}) cannot lose any arc
    tight = gen.directed_variant(gen.harary_graph(10, 3, 1), 2, 1.0)
    assert tight.meta["dropped"] == 0 and tight.arcs == gen.harary_graph(10, 3, 1).arcs
    # a base that is not k-T-connected is refused when verifying
    weak = gen.erdos_renyi_graph(10, 0.15, 3, 2, require_k_connected=False)
    assert not weak.meta["claims_k_connected"]
    with pytest.raises(ValueError):
        gen.directed_variant(weak, 0, 0.5)
    unverified = gen.directed_variant(weak, 0, 0.5, verify=False)
    assert unverified.meta["claims_kT_connected"] is False
    assert unverified.meta["connectivity_verified"] is False


def test_directed_variant_unverified_applies_every_proposal():
    base = gen.erdos_renyi_graph(12, 0.5, 3, 4)
    rng = random.Random(1)  # replay of the documented rng protocol
    edges = list(base.undirected_edges)
    rng.shuffle(edges)
    proposals = []
    for u, v in edges:
        if rng.random() < 0.6:
            proposals.append((u, v) if rng.random() < 0.5 else (v, u))
    d = gen.directed_variant(base, 1, 0.6, verify=False)
    assert d.meta["proposed"] == len(proposals)
    assert set(d.arcs) == set(base.arcs) - set(proposals)
    assert d.meta["dropped"] == len(set(base.arcs) & set(proposals))
    # verification only ever keeps a superset of the unverified arc set
    v = gen.directed_variant(base, 1, 0.6)
    assert set(d.arcs) <= set(v.arcs) <= set(base.arcs)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(3, 1), (4, 1), (5, 2), (6, 3), (8, 4), (12, 2), (14, 3)])
def test_family_catalog_entries_are_valid_or_raise_value_error(n: int, k: int):
    cat = gen.family_catalog()
    built = 0
    for name, fn in cat.items():
        try:
            inst = fn(n, k, 1)
        except ValueError:
            continue
        built += 1
        assert_claims_hold(inst)
        assert fn(n, k, 1) == inst
        if name.startswith("paper_"):
            continue
        assert inst.k == k and inst.meta["seed"] == 1
        assert inst.meta["mode"] == "balanced"
        assert max(inst.capacities) - min(inst.capacities) <= 1
        if not inst.directed:
            assert inst.meta["claims_k_connected"] is True
            assert nx.node_connectivity(undirected_graph(inst)) >= k
        else:
            assert inst.meta["claims_kT_connected"] is True
        if name == "directed_random":
            assert inst.meta["dropped"] >= 0
    assert built >= len(cat) - 5


def test_single_vertex_claims_are_consistent_with_the_exact_test():
    """Review finding: complete_graph(1, 1), grid_graph(1, 1, 1) and
    grid3d_graph(1, 1, 1, 1) once claimed (verified) k-connectivity for the
    one-vertex graph although the module's own exact test
    is_k_vertex_connected(K_1, 1) and networkx.node_connectivity(K_1) = 0 say
    otherwise.  The families now refuse n <= k like harary_graph does."""
    assert gen.is_k_vertex_connected(nx.complete_graph(1), 1) is False
    assert nx.node_connectivity(nx.complete_graph(1)) == 0
    for build in (
        lambda: gen.complete_graph(1, 1),
        lambda: gen.complete_graph(1, 1, 3, "extreme"),
        lambda: gen.grid_graph(1, 1, 1),
        lambda: gen.grid3d_graph(1, 1, 1, 1),
    ):
        with pytest.raises(ValueError, match="0-connected"):
            build()
    # the smallest members of each family are still available and consistent
    for inst in (gen.complete_graph(2, 1), gen.grid_graph(1, 2, 1), gen.grid3d_graph(1, 1, 2, 1)):
        assert inst.n == 2 and inst.meta["claims_k_connected"] is True
        assert gen.is_k_vertex_connected(undirected_graph(inst), inst.k) is True
