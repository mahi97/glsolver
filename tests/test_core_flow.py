"""Differential tests of the C++ core (``glcore.Graph``, the tightest-cut flow engine and the
``EssentialOracle``) against the validated pure-Python reference ``glref``.

Every property asserted here is a statement of docs/paper_notes.md: [Prop 4.2] (kappa, tightest cut,
essential terminals), [Def 2.1] / §13.4 (contraction, parent), [Lem 4.3] / O1 (certified subsets after
deletions), O2 (warm-started criticality), O3 (contraction keeps kappa/Ess/cuts), O4 (terminal removal).
All randomness is seeded.  Runnable in isolation: ``pytest tests/test_core_flow.py``.
"""
from __future__ import annotations

import random
from typing import Iterable

import pytest

from glref.critical import is_critical
from glref.flow import tightest_min_cut
from glref.graph import DiGraphState
from glsolver import _core, generators
from glsolver.instance import Instance
from helpers_small_graphs import (
    paper_essential_example,
    paper_running_example,
    random_digraph,
    random_undirected_instance,
)

Graph = _core.Graph
EssentialOracle = _core.EssentialOracle


# ---------------------------------------------------------------------------
# instance pool (n <= 40, k <= 6, undirected + directed, paper examples)
# ---------------------------------------------------------------------------
def instance_pool() -> list[Instance]:
    pool: list[Instance] = [
        paper_running_example(),
        paper_essential_example(),
        generators.paper_contract_counterexample(),
        generators.paper_running_example(),
        generators.complete_graph(8, 4, seed=1),
        generators.cycle_graph(12, 2, seed=2),
        generators.wheel_graph(11, 3, seed=3),
        generators.grid_graph(4, 5, 2, seed=4),
        generators.harary_graph(18, 4, seed=5),
        generators.random_regular_graph(20, 4, 3, seed=6),
        generators.random_regular_graph(24, 6, 5, seed=7),
        generators.erdos_renyi_graph(22, 0.35, 3, seed=8),
        generators.sparse_k_connected(30, 3, seed=9),
        generators.random_kT_connected_digraph(18, 3, seed=10),
        generators.random_kT_connected_digraph(26, 4, seed=11),
        generators.directed_variant(generators.erdos_renyi_graph(24, 0.4, 3, seed=12), 12, 0.5),
        generators.directed_variant(generators.random_regular_graph(30, 6, 4, seed=13), 13, 0.4),
    ]
    rng = random.Random(2026)
    # sparse random digraphs: not k-T-connected, some vertices with kappa < k or kappa == 0 (§13.9)
    for n, k, p in [(12, 3, 0.15), (25, 4, 0.12), (40, 6, 0.08), (30, 5, 0.2), (16, 6, 0.3)]:
        pool.append(random_digraph(n, k, rng, p=p))
    pool.append(random_undirected_instance(16, 4, rng))
    pool.append(random_undirected_instance(28, 6, rng))
    return pool


POOL = instance_pool()
POOL_IDS = [f"{i}:{inst.name or inst.meta.get('family', 'anon')}" for i, inst in enumerate(POOL)]


def core_graph(inst: Instance) -> Graph:
    return Graph(inst.n, list(inst.arcs), list(inst.terminals))


def ref_graph(inst: Instance) -> DiGraphState:
    return DiGraphState.from_instance(inst)


def ref_kappa_ess(g: DiGraphState, v: int) -> tuple[int, frozenset[int]]:
    cut = tightest_min_cut(g, v)
    return cut.kappa, frozenset(cut.essential_terminals(g))


def core_ess_ids(inst: Instance, idx: Iterable[int]) -> frozenset[int]:
    """Original terminal indices -> terminal vertex ids."""
    return frozenset(inst.terminals[i] for i in idx)


def check_paths(inst: Instance, arcs: set[tuple[int, int]], terminals: Iterable[int], v: int, kappa: int, paths) -> None:
    """[Def 3.2]: kappa paths from v, pairwise vertex-disjoint except v, ending at distinct terminals, using
    only present arcs, and simple."""
    tset = set(terminals)
    assert len(paths) == kappa
    seen: set[int] = set()
    ends: set[int] = set()
    for p in paths:
        assert p[0] == v and len(p) >= 2
        assert p[-1] in tset
        assert len(set(p)) == len(p), "path is not simple"
        for a, b in zip(p, p[1:]):
            assert (a, b) in arcs, f"path uses a non-arc ({a},{b})"
        inner = set(p[1:])
        assert not (inner & seen), "paths share a vertex other than v"
        assert v not in inner
        seen |= inner
        assert p[-1] not in ends, "two paths end at the same terminal"
        ends.add(p[-1])


def same_graph(g: Graph, ref: DiGraphState) -> None:
    assert set(g.arcs()) == set(ref.arcs())
    assert g.num_live_arcs() == ref.num_arcs()
    assert g.pre_terminals() == ref.pre_terminals()
    assert list(g.terminals) == list(ref.terminals)
    assert [g.live(v) for v in range(g.n)] == list(ref.live)
    assert g.num_live_nonterminals() == ref.num_nonterminals()
    for v in ref.vertices():
        assert set(g.out_neighbors(v)) == set(ref.out_neighbors(v))
        assert set(g.in_neighbors(v)) == set(ref.in_neighbors(v))
        assert g.out_degree(v) == ref.out_degree(v)
    g.check_invariants()
    ref.check_invariants()


# ---------------------------------------------------------------------------
# (a) tightest cut: kappa / sides / ess identical to the reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inst", POOL, ids=POOL_IDS)
def test_graph_load_matches_reference(inst: Instance) -> None:
    g = core_graph(inst)
    ref = ref_graph(inst)
    same_graph(g, ref)
    assert g.k == inst.k and g.k0 == inst.k
    for i, t in enumerate(inst.terminals):
        assert g.terminal_index(t) == i and g.is_terminal(t)
    # orig_head == head for loaded arcs (§13.4)
    for u, v in inst.arcs:
        assert g.orig_head(u, v) == v


def test_graph_constructor_filters_like_reference() -> None:
    # self-loops, arcs leaving terminals and duplicates are dropped; first occurrence kept
    arcs = [(0, 1), (1, 2), (2, 2), (0, 1), (2, 0), (1, 0), (3, 1), (3, 1), (1, 3)]
    g = Graph(4, arcs, [2, 3])
    ref = DiGraphState(4, arcs, [2, 3])
    same_graph(g, ref)
    assert set(g.arcs()) == {(0, 1), (1, 2), (1, 0), (1, 3)}
    with pytest.raises((ValueError, RuntimeError)):
        Graph(3, [(0, 1)], [1, 1])
    with pytest.raises((ValueError, RuntimeError)):
        Graph(3, [(0, 5)], [1])


@pytest.mark.parametrize("inst", POOL, ids=POOL_IDS)
def test_tightest_cut_matches_reference(inst: Instance) -> None:
    g = core_graph(inst)
    ref = ref_graph(inst)
    arcs = set(inst.arcs)
    for v in ref.nonterminals():
        kappa, sides, ess, paths = _core.tightest_cut(g, v)
        cut = tightest_min_cut(ref, v)
        assert kappa == cut.kappa, (v, kappa, cut.kappa)
        assert sides == cut.side, f"sides differ for v={v}"  # the tightest cut is unique [Def 3.8]
        assert core_ess_ids(inst, ess) == frozenset(cut.essential_terminals(ref)), v
        assert sides[v] == "R" and sum(1 for s in sides.values() if s == "S") == kappa
        assert all(sides[t] != "R" for t in inst.terminals)
        check_paths(inst, arcs, inst.terminals, v, kappa, paths)


def test_paper_essential_example_expected_sets() -> None:
    from helpers_small_graphs import PAPER_ESSENTIAL_EXPECTED

    inst = paper_essential_example()
    g = core_graph(inst)
    o = EssentialOracle(g, 2, 0)
    o.compute_all()
    for v, expected in PAPER_ESSENTIAL_EXPECTED.items():
        assert core_ess_ids(inst, o.ess(v)) == expected


# ---------------------------------------------------------------------------
# (b) random operation sequences applied to Graph + oracle and to DiGraphState
# ---------------------------------------------------------------------------
def oracle_matches_reference(inst: Instance, g: Graph, ref: DiGraphState, o, exact: bool) -> None:
    """kappa always equals the reference; the stored Ess is a certified subset (O1/P2) and equals the
    reference whenever the oracle claims an exact cut (then the sides must be the unique tightest cut too);
    with ``exact`` every vertex must be exact."""
    for v in ref.nonterminals():
        rk, ress = ref_kappa_ess(ref, v)
        assert o.kappa(v) == rk, (v, o.kappa(v), rk)
        ess = core_ess_ids(inst, o.ess(v))
        if exact:
            assert ess == ress, (v, ess, ress)
        else:
            assert ess <= ress, (v, ess, ress)
        if o.cut_exact(v):
            assert ess == ress, (v, ess, ress)
            assert o.sides(v) == tightest_min_cut(ref, v).side, v
        check_paths(inst, set(ref.arcs()), ref.terminals, v, rk, o.paths(v))


def users_consistent(g: Graph, ref: DiGraphState, o) -> None:
    """(d) users_of_arc(a) == {v : a on a stored path of v}."""
    expected: dict[tuple[int, int], set[int]] = {}
    for v in ref.nonterminals():
        for p in o.paths(v):
            for a, b in zip(p, p[1:]):
                expected.setdefault((a, b), set()).add(v)
    for u, w in ref.arcs():
        assert set(o.users_of_arc(u, w)) == expected.get((u, w), set()), (u, w)


def run_random_ops(inst: Instance, seed: int, threads: int, exact: bool, steps: int, check_every: bool = True):
    """Apply a random sequence of delete_arc / contract / remove_terminal to both graphs; returns a
    transcript of (kappa, ess, paths) per step for determinism checks."""
    rng = random.Random(seed)
    g = core_graph(inst)
    ref = ref_graph(inst)
    o = EssentialOracle(g, threads, seed)
    o.compute_all()
    transcript = []
    if check_every:
        oracle_matches_reference(inst, g, ref, o, exact=True)
        users_consistent(g, ref, o)
    for _step in range(steps):
        ops = []
        if ref.num_arcs() > 0:
            ops.append("delete")
        deg1 = [p for p in g.pre_terminals() if g.out_degree(p) == 1]
        if deg1:
            ops += ["contract", "contract"]
        if g.k > 1:
            ops.append("remove_terminal")
        if not ops:
            break
        op = rng.choice(ops)
        if op == "delete":
            u, w = rng.choice(sorted(ref.arcs()))
            before = o.evaluate_deletion([(u, w)], exact)
            assert set(before) == set(o.users_of_arc(u, w))
            o.commit_deletion([(u, w)])
            ref.delete_arc(u, w)
            assert not g.has_arc(u, w)
            if exact:
                # unaffected vertices keep certified subsets (O1); exact sets need one reverse BFS each
                o.refresh_all_cuts()
        elif op == "contract":
            p = rng.choice(deg1)
            (t,) = g.out_neighbors(p)
            assert g.is_terminal(t)
            parent_core = g.contract(p, t)
            parent_ref = ref.contract(p, t)
            assert parent_core == parent_ref  # (f) §13.4
            o.after_contraction(p, t)
        else:
            t = rng.choice(list(ref.terminals))
            g.remove_terminal(t)
            ref.remove_terminal(t)
            o.after_terminal_removal(t)
        same_graph(g, ref)
        if check_every:
            oracle_matches_reference(inst, g, ref, o, exact=exact)
            users_consistent(g, ref, o)
        transcript.append(
            (op, sorted((v, o.kappa(v), tuple(o.ess(v)), tuple(map(tuple, o.paths(v)))) for v in ref.nonterminals()))
        )
    # exact refresh on warm flows must reproduce the reference exactly
    o.refresh_all_cuts()
    for v in ref.nonterminals():
        assert o.cut_exact(v)
    oracle_matches_reference(inst, g, ref, o, exact=True)
    return transcript, g, ref, o


@pytest.mark.parametrize("inst", POOL, ids=POOL_IDS)
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "subset"])
def test_random_operations_track_reference(inst: Instance, exact: bool) -> None:
    run_random_ops(inst, seed=7 + inst.n, threads=2, exact=exact, steps=14)


@pytest.mark.parametrize("seed", range(6))
def test_random_operations_many_seeds_small(seed: int) -> None:
    rng = random.Random(100 + seed)
    inst = random_digraph(14, 4, rng, p=0.3)
    run_random_ops(inst, seed=seed, threads=1, exact=bool(seed % 2), steps=25)


# ---------------------------------------------------------------------------
# (c) single-arc evaluate_deletion reproduces criticality [Def 6.1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inst", [POOL[0], POOL[1], POOL[2], POOL[9], POOL[13], POOL[17], POOL[19]],
                         ids=[POOL_IDS[i] for i in (0, 1, 2, 9, 13, 17, 19)])
def test_evaluate_deletion_reproduces_is_critical(inst: Instance) -> None:
    g = core_graph(inst)
    ref = ref_graph(inst)
    o = EssentialOracle(g, 3, 0)
    o.compute_all()
    ess_ref = {v: ref_kappa_ess(ref, v)[1] for v in ref.nonterminals()}
    rng = random.Random(inst.n)
    arcs = sorted(ref.arcs())
    for e in rng.sample(arcs, min(len(arcs), 12)):
        res = o.evaluate_deletion([e], True)
        h = ref.copy()
        h.delete_arc(*e)
        for v in ref.nonterminals():
            ess_g = ess_ref[v]
            if v in res:
                kappa_after, ess_after_idx = res[v]
                ess_after = core_ess_ids(inst, ess_after_idx)
                rk, ress = ref_kappa_ess(h, v)
                assert kappa_after == rk and ess_after == ress, (e, v)
                for t in inst.terminals:
                    assert (t in ess_g and t not in ess_after) == is_critical(ref, e, v, t, ess_g)
            else:
                # O1: flow avoids e -> kappa unchanged, Ess can only grow -> nothing is critical
                assert not any(is_critical(ref, e, v, t, ess_g) for t in inst.terminals), (e, v)
                rk, ress = ref_kappa_ess(h, v)
                assert rk == o.kappa(v) and ess_g <= ress


def test_evaluate_deletion_of_arc_set_matches_reference() -> None:
    """O5-style multi-arc deletion D = out(p) \\ {one arc}: exact flows/cuts in G \\ D."""
    for inst in (POOL[9], POOL[14], POOL[19], POOL[20]):
        g = core_graph(inst)
        ref = ref_graph(inst)
        o = EssentialOracle(g, 4, 0)
        o.compute_all()
        rng = random.Random(inst.n)
        for _ in range(3):
            cands = [p for p in ref.pre_terminals() if ref.out_degree(p) >= 2]
            if not cands:
                break
            p = rng.choice(cands)
            outs = ref.out_neighbors(p)
            keep = rng.choice([x for x in outs if ref.is_terminal(x)])
            D = [(p, x) for x in outs if x != keep]
            res = o.evaluate_deletion(D, True)
            h = ref.copy()
            for e in D:
                h.delete_arc(*e)
            for v in ref.nonterminals():
                rk, ress = ref_kappa_ess(h, v)
                if v in res:
                    assert res[v][0] == rk and core_ess_ids(inst, res[v][1]) == ress, (D, v)
                    check_paths(inst, set(h.arcs()), h.terminals, v, rk, o.evaluated_paths(v))
                else:
                    assert o.kappa(v) == rk and core_ess_ids(inst, o.ess(v)) <= ress, (D, v)
            o.commit_deletion(D)
            for e in D:
                ref.delete_arc(*e)
            same_graph(g, ref)
            oracle_matches_reference(inst, g, ref, o, exact=False)
            users_consistent(g, ref, o)
            o.refresh_all_cuts()
            oracle_matches_reference(inst, g, ref, o, exact=True)


# ---------------------------------------------------------------------------
# (e) determinism across thread counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("idx", [9, 11, 14, 17, 19])
def test_determinism_threads_1_vs_4(idx: int) -> None:
    inst = POOL[idx]
    t1, *_ = run_random_ops(inst, seed=5, threads=1, exact=False, steps=12, check_every=False)
    t4, *_ = run_random_ops(inst, seed=5, threads=4, exact=False, steps=12, check_every=False)
    assert t1 == t4


# ---------------------------------------------------------------------------
# (f) contraction parent and orig_head bookkeeping (§13.4)
# ---------------------------------------------------------------------------
def test_contraction_parent_and_merge() -> None:
    # 3 -> 2 -> t0, 3 -> t0 already exists (merge), 4 -> 2 (redirect creates (4,t0) with orig_head 2)
    arcs = [(2, 0), (3, 2), (3, 0), (4, 2), (4, 1), (2, 1)]
    g = Graph(5, arcs, [0, 1])
    ref = DiGraphState(5, arcs, [0, 1])
    g.delete_arc(2, 1)
    ref.delete_arc(2, 1)
    parent, created, deleted = g.contract_detailed(2, 0)
    assert parent == ref.contract(2, 0) == 0
    same_graph(g, ref)
    assert g.orig_head(3, 0) == 0  # merged: the existing arc keeps its orig_head
    assert g.orig_head(4, 0) == 2  # redirected: orig_head = p
    assert len(created) == 1 and g.arc(created[0])[:2] == (4, 0)
    assert sorted(g.arc(a)[:2] for a in deleted) == sorted([(3, 2), (4, 2), (2, 0)])
    # chain: contract 4 into t0 now; its parent must be 2 (the original head), as in the reference
    g.delete_arc(4, 1)
    ref.delete_arc(4, 1)
    assert g.contract(4, 0) == ref.contract(4, 0) == 2
    same_graph(g, ref)
    assert not g.live(2) and not g.live(4)
    with pytest.raises((ValueError, RuntimeError)):
        g.contract(3, 1)  # arc (3,1) absent


def test_remove_terminal_and_vertex() -> None:
    inst = POOL[9]
    g = core_graph(inst)
    ref = ref_graph(inst)
    t = inst.terminals[1]
    g.remove_terminal(t)
    ref.remove_terminal(t)
    same_graph(g, ref)
    assert g.terminal_index(t) == 1 and not g.is_terminal(t)
    v = ref.nonterminals()[0]
    g.remove_vertex(v)
    ref.remove_vertex(v)
    same_graph(g, ref)
    with pytest.raises((ValueError, RuntimeError)):
        g.remove_terminal(v)


def test_after_vertex_removal_keeps_certified_subsets() -> None:
    inst = POOL[10]
    g = core_graph(inst)
    ref = ref_graph(inst)
    o = EssentialOracle(g, 2, 0)
    o.compute_all()
    rng = random.Random(3)
    for _ in range(3):
        v = rng.choice(ref.nonterminals())
        g.remove_vertex(v)
        ref.remove_vertex(v)
        o.after_vertex_removal(v)
        same_graph(g, ref)
        oracle_matches_reference(inst, g, ref, o, exact=False)
        users_consistent(g, ref, o)
    o.refresh_all_cuts()
    oracle_matches_reference(inst, g, ref, o, exact=True)


def test_out_of_band_mutation_is_detected() -> None:
    """A graph mutation without the matching hook invalidates the oracle instead of serving stale data."""
    inst = POOL[9]
    g = core_graph(inst)
    ref = ref_graph(inst)
    o = EssentialOracle(g, 1, 0)
    o.compute_all()
    u, w = sorted(ref.arcs())[0]
    g.delete_arc(u, w)
    ref.delete_arc(u, w)
    oracle_matches_reference(inst, g, ref, o, exact=True)


def test_hooks_do_not_recompute_from_scratch() -> None:
    """O1/O3/O4 efficiency: deletions, contractions and terminal removals never trigger a from-scratch
    max-flow (only warm-started augmentations / reverse BFS); results still match the reference."""
    inst = generators.random_regular_graph(60, 6, 4, seed=21)
    g = core_graph(inst)
    ref = ref_graph(inst)
    o = EssentialOracle(g, 4, 0)
    o.compute_all()
    base = o.stats()["max_flow_calls"]
    assert base == ref.num_nonterminals()
    rng = random.Random(21)
    for _ in range(25):
        u, w = rng.choice(sorted(ref.arcs()))
        o.evaluate_deletion([(u, w)], False)
        o.commit_deletion([(u, w)])
        ref.delete_arc(u, w)
        deg1 = [p for p in g.pre_terminals() if g.out_degree(p) == 1]
        if deg1:
            p = deg1[0]
            (t,) = g.out_neighbors(p)
            assert g.contract(p, t) == ref.contract(p, t)
            o.after_contraction(p, t)
    t = ref.terminals[0]
    g.remove_terminal(t)
    ref.remove_terminal(t)
    o.after_terminal_removal(t)
    st = o.stats()
    assert st["max_flow_calls"] == base, st
    assert st["augment_calls"] > 0 and st["cut_calls"] > 0
    oracle_matches_reference(inst, g, ref, o, exact=False)
    users_consistent(g, ref, o)
    # every vertex is served without recomputation, and kappa still matches
    for v in ref.nonterminals():
        o.kappa(v)
    assert o.stats()["max_flow_calls"] == base


def test_invalid_uses_raise() -> None:
    inst = POOL[0]
    g = core_graph(inst)
    o = EssentialOracle(g, 1, 0)
    with pytest.raises((ValueError, RuntimeError)):
        o.kappa(inst.terminals[0])
    with pytest.raises((ValueError, RuntimeError)):
        _core.tightest_cut(g, inst.terminals[0])
    with pytest.raises((ValueError, RuntimeError)):
        o.evaluate_deletion([(0, 1)], True)  # arc out of a terminal: absent


@pytest.mark.slow
def test_larger_random_sequences() -> None:
    rng = random.Random(77)
    for i in range(4):
        inst = random_undirected_instance(36, 5, rng)
        run_random_ops(inst, seed=i, threads=4, exact=bool(i % 2), steps=30)
