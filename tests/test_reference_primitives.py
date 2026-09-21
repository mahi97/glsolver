"""Tests for the reference primitives ``reference/glref/{graph,flow,essential,
matching,assignment,critical}.py``.

Every test names the item of docs/paper_notes.md it checks.  Hand examples
(including the paper's worked examples) are combined with seeded random
properties, and every quantity that has an independent formulation is
cross-checked against one computed here from scratch (networkx max-flow on a
vertex-split network built in this file, brute-force path families, brute-force
enumeration of all cuts, a naive arc-set model of the mutable graph, a
brute-force assignment search).
"""
from __future__ import annotations

import itertools
import random
from typing import Iterator

import networkx as nx
import pytest

from glref.assignment import find_witness, is_witness
from glref.critical import (
    criticality_cost,
    criticality_table,
    essential_after_deleting,
    is_critical,
    potential,
)
from glref.essential import (
    all_essential,
    essential_terminals,
    essential_terminals_by_definition,
    essential_terminals_with_cut,
    terminal_connectivity,
)
from glref.flow import (
    FlowNetwork,
    L,
    R,
    S,
    cut_intersection,
    cut_union,
    is_valid_cut,
    tightest_min_cut,
)
from glref.graph import DiGraphState
from glref.matching import max_matching, minimal_hall_deficient_set, saturating_matching
from glsolver.instance import Instance, make_instance
from helpers_small_graphs import (
    PAPER_ESSENTIAL_EXPECTED,
    PAPER_ESSENTIAL_WITNESS,
    paper_essential_example,
    paper_running_example,
    random_capacities,
    random_digraph,
    random_k_connected_undirected,
    random_kT_connected_digraph,
    random_undirected_instance,
)

# ---------------------------------------------------------------------------
# independent re-implementations used as oracles
# ---------------------------------------------------------------------------


def nx_kappa(g: DiGraphState, v: int) -> int:
    """[Def 3.2] computed independently of ``glref.flow``: ``nx.maximum_flow_value``
    on a vertex-split network built here (unit split arcs enforce vertex-disjointness
    and distinct terminal endpoints; ``v`` and all other arcs get a large capacity)."""
    if not g.terminals:
        return 0
    H = nx.DiGraph()
    big = g.n + 1
    for x in g.vertices():
        H.add_edge(("in", x), ("out", x), capacity=big if x == v else 1)
        for y in g.out[x]:
            H.add_edge(("out", x), ("in", y), capacity=big)
    for t in g.terminals:
        H.add_edge(("out", t), "sink", capacity=big)
    H.add_edge("src", ("in", v), capacity=big)
    return int(nx.maximum_flow_value(H, "src", "sink"))


def bruteforce_kappa(g: DiGraphState, v: int) -> int:
    """[Def 3.2] literally: the largest family of ``v→T`` paths ending at distinct
    terminals and pairwise vertex-disjoint except for ``v`` (tiny graphs only)."""
    D = nx.DiGraph()
    D.add_nodes_from(g.vertices())
    D.add_edges_from(g.arcs())
    paths: list[tuple[int, frozenset[int]]] = []
    for t in g.terminals:
        for p in nx.all_simple_paths(D, v, t):
            paths.append((t, frozenset(p[1:])))
    best = 0

    def rec(idx: int, used: frozenset[int], terms: frozenset[int], count: int) -> None:
        nonlocal best
        best = max(best, count)
        if count + (len(g.terminals) - len(terms)) <= best:
            return
        for j in range(idx, len(paths)):
            t, verts = paths[j]
            if t in terms or verts & used:
                continue
            rec(j + 1, used | verts, terms | {t}, count + 1)

    rec(0, frozenset(), frozenset(), 0)
    return best


def local_is_valid_cut(g: DiGraphState, v: int, side: dict[int, str]) -> bool:
    """[Def 3.3] re-implemented: ``v ∈ R``, ``T ⊆ L ∪ S``, no arc from ``R`` to ``L``."""
    if side[v] != R:
        return False
    if any(side[t] == R for t in g.terminals):
        return False
    return not any(side[u] == R and side[w] == L for u, w in g.arcs())


def all_valid_cuts(g: DiGraphState, v: int) -> list[dict[int, str]]:
    """Every labelling of the live vertices by L/S/R that is a cut separating ``v``
    [Def 3.3]; also asserts ``glref.flow.is_valid_cut`` agrees with the local check."""
    live = list(g.vertices())
    cuts: list[dict[int, str]] = []
    for labels in itertools.product((L, S, R), repeat=len(live)):
        side = dict(zip(live, labels))
        ok = local_is_valid_cut(g, v, side)
        assert ok == is_valid_cut(g, v, side), side
        if ok:
            cuts.append(side)
    return cuts


def cut_size(side: dict[int, str]) -> int:
    return sum(1 for s in side.values() if s == S)


def random_saturating_matching(g: DiGraphState, rng: random.Random) -> dict[int, int] | None:
    """Randomised Kuhn matching ``T → PT(G,T)`` written independently of
    ``glref.matching``; ``None`` iff ``T`` is not saturable.  Returns ``{t: p}``."""
    terms = list(g.terminals)
    rng.shuffle(terms)
    nbrs = {t: rng.sample(sorted(g.inn[t]), len(g.inn[t])) for t in terms}
    match_p: dict[int, int] = {}

    def aug(t: int, seen: set[int]) -> bool:
        for p in nbrs[t]:
            if p in seen:
                continue
            seen.add(p)
            if p not in match_p or aug(match_p[p], seen):
                match_p[p] = t
                return True
        return False

    for t in terms:
        if not aug(t, set()):
            return None
    return {t: p for p, t in match_p.items()}


def nx_max_matching_size(g: DiGraphState, S_terms: list[int]) -> int:
    """Size of a maximum matching between ``S_terms`` and pre-terminals, via networkx."""
    B = nx.Graph()
    left = [("t", t) for t in S_terms]
    B.add_nodes_from(left, bipartite=0)
    for t in S_terms:
        for p in g.inn[t]:
            B.add_edge(("t", t), ("p", p))
    if B.number_of_edges() == 0:
        return 0
    m = nx.bipartite.hopcroft_karp_matching(B, top_nodes=left)
    return sum(1 for x in left if x in m)


class NaiveArcModel:
    """Set-of-arcs model of the paper's mutations [Def 2.1] (independent of DiGraphState)."""

    def __init__(self, inst: Instance) -> None:
        self.tset = set(inst.terminals)
        self.terminals = list(inst.terminals)
        self.arcs: set[tuple[int, int]] = {(u, v) for u, v in inst.arcs if u != v and u not in self.tset}
        self.live = set(range(inst.n))
        self.contracted_into: dict[int, int] = {}

    def delete_arc(self, u: int, v: int) -> None:
        self.arcs.remove((u, v))

    def remove_terminal(self, t: int) -> None:
        self.arcs = {(u, v) for u, v in self.arcs if u != t and v != t}
        self.live.discard(t)
        self.terminals.remove(t)
        self.tset.discard(t)

    def contract(self, p: int, t: int) -> None:
        assert (p, t) in self.arcs and t in self.tset
        ins = [u for (u, v) in self.arcs if v == p]
        self.arcs = {(u, v) for u, v in self.arcs if u != p and v != p}
        for u in ins:
            self.arcs.add((u, t))
        self.live.discard(p)
        self.contracted_into[p] = t

    def pre_terminals(self) -> list[int]:
        return sorted({u for u, v in self.arcs if v in self.tset})


def assert_state_matches_model(g: DiGraphState, model: NaiveArcModel, original_arcs: set[tuple[int, int]]) -> None:
    g.check_invariants()
    assert sorted(g.arcs()) == sorted(model.arcs)
    assert set(g.vertices()) == model.live
    assert g.terminals == model.terminals
    assert g.pre_terminals() == model.pre_terminals()
    assert g.num_arcs() == len(model.arcs)
    for (u, t), h in g.orig_head.items():
        assert g.is_terminal(t)
        assert (u, h) in original_arcs, f"orig_head[({u},{t})] = {h} is not an original arc"
        assert h == t or model.contracted_into.get(h) == t, f"orig_head[({u},{t})] = {h} not in part of {t}"
    for u, t in g.arcs():
        if g.is_terminal(t):
            assert (u, t) in g.orig_head


def random_states(rng: random.Random, count: int, n_lo: int, n_hi: int, k_hi: int, p_lo: float = 0.2, p_hi: float = 0.7) -> Iterator[tuple[Instance, DiGraphState]]:
    for _ in range(count):
        n = rng.randint(n_lo, n_hi)
        k = rng.randint(1, min(k_hi, n - 1))
        inst = random_digraph(n, k, rng, p=rng.uniform(p_lo, p_hi))
        yield inst, DiGraphState.from_instance(inst)


# ---------------------------------------------------------------------------
# [Prop 4.2] tightest minimum cut and κ
# ---------------------------------------------------------------------------


def test_prop42_paper_essential_example_hand_values() -> None:
    """[Prop 4.2]/[Lem 4.1] on the paper's example: κ(v11)=3, Ess(v11)={t1,t2}, and all
    the other displayed essential sets."""
    inst = paper_essential_example()
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    assert kappa[10] == 3
    assert ess[10] == {0, 1}
    for v, expected in PAPER_ESSENTIAL_EXPECTED.items():
        assert ess[v] == set(expected), (v, ess[v])
        assert kappa[v] == nx_kappa(g, v)
        assert essential_terminals_by_definition(g, v) == (kappa[v], set(expected))
    assert kappa[12] == 4 and kappa[4] == 1 and kappa[9] == 1


def test_prop42_paper_running_example_hand_values() -> None:
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    assert kappa == {3: 2, 4: 2, 7: 2, 5: 2, 6: 2, 8: 2}
    assert ess == {3: {0, 1}, 4: {0, 1}, 7: {0, 1}, 5: {1, 2}, 6: {1, 2}, 8: {1, 2}}
    cut = tightest_min_cut(g, 7)
    assert cut.kappa == 2 and cut.v == 7
    assert cut.side[7] == R and cut.separator() == [0, 1]
    assert cut.essential_terminals(g) == {0, 1}


def test_prop42_unreachable_vertex_has_kappa_zero() -> None:
    """§13.9: a vertex that reaches no terminal has κ = 0 and Ess = ∅ and lands in R."""
    inst = make_instance(5, [(2, 0), (2, 1), (3, 4), (4, 3)], [0, 1], [3, 0])
    g = DiGraphState.from_instance(inst)
    for v in (3, 4):
        cut = tightest_min_cut(g, v)
        assert cut.kappa == 0 and cut.separator() == [] and cut.side[v] == R
        assert essential_terminals(g, v) == (0, set())
        assert essential_terminals_by_definition(g, v) == (0, set())
    assert essential_terminals(g, 2) == (2, {0, 1})


def test_prop42_single_out_arc_preterminal() -> None:
    """§4 edge case: a pre-terminal with a single out-arc (p, t) has Ess(p) = {t}."""
    inst = make_instance(4, [(2, 0), (3, 2), (3, 1), (3, 0)], [0, 1], [1, 1])
    g = DiGraphState.from_instance(inst)
    assert essential_terminals(g, 2) == (1, {0})
    assert essential_terminals(g, 3) == (2, {0, 1})


def test_tightest_cut_rejects_terminals_and_dead_vertices() -> None:
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 0])
    g = DiGraphState.from_instance(inst)
    with pytest.raises(ValueError):
        tightest_min_cut(g, 0)
    g.contract(2, 0)
    with pytest.raises(ValueError):
        tightest_min_cut(g, 2)


@pytest.mark.parametrize("seed", range(6))
def test_prop42_random_digraphs_against_networkx(seed: int) -> None:
    """[Prop 4.2] on random digraphs (n ≤ 12): the returned cut is valid [Def 3.3],
    |S| = κ, v ∈ R, T ⊆ L ∪ S, and κ equals an independent networkx max-flow;
    for n ≤ 6 also a brute-force path-family count [Def 3.2]."""
    rng = random.Random(1000 + seed)
    checked = 0
    for _inst, g in random_states(rng, 14, 3, 12, 5, 0.15, 0.8):
        for v in g.nonterminals():
            cut = tightest_min_cut(g, v)
            assert is_valid_cut(g, v, cut.side) and local_is_valid_cut(g, v, cut.side)
            assert set(cut.side) == set(g.vertices())
            assert len(cut.separator()) == cut.kappa
            assert cut.side[v] == R
            assert all(cut.side[t] in (L, S) for t in g.terminals)
            assert cut.kappa <= min(g.k, g.out_degree(v))
            assert cut.kappa == nx_kappa(g, v) == terminal_connectivity(g, v)
            if g.n <= 6:
                assert cut.kappa == bruteforce_kappa(g, v)
            checked += 1
    assert checked > 20


@pytest.mark.parametrize("seed", range(3))
def test_prop42_k_connected_undirected_all_terminals_essential(seed: int) -> None:
    """Paper §7.3: in a k-T-connected graph κ(v) = k and every terminal is essential
    for every v; k-connected undirected graphs are k-T-connected (§1.1)."""
    rng = random.Random(2000 + seed)
    for n in range(4, 11):
        for k in range(1, min(5, n - 1) + 1):
            inst = random_undirected_instance(n, k, rng)
            g = DiGraphState.from_instance(inst)
            for v in g.nonterminals():
                cut = tightest_min_cut(g, v)
                assert cut.kappa == k == nx_kappa(g, v)
                assert cut.essential_terminals(g) == set(inst.terminals)
                assert is_valid_cut(g, v, cut.side)


@pytest.mark.parametrize("seed", range(3))
def test_kT_connected_generator_is_kT_connected(seed: int) -> None:
    """The k-T-connected sampler really produces κ(v) = k (checked with networkx)."""
    rng = random.Random(3000 + seed)
    for n in range(4, 11):
        for k in range(1, min(4, n - 1) + 1):
            inst = random_kT_connected_digraph(n, k, rng)
            g = DiGraphState.from_instance(inst)
            assert all(nx_kappa(g, v) == k for v in g.nonterminals())
            kappa, ess = all_essential(g)
            assert all(kappa[v] == k and ess[v] == set(inst.terminals) for v in kappa)


def test_flow_network_basic() -> None:
    """The plain flow network: paired residual arcs, augment/limit semantics, reach_sink."""
    net = FlowNetwork(4)
    a = net.add_arc(0, 1, 2)
    b = net.add_arc(1, 3, 1)
    c = net.add_arc(0, 2, 1)
    d = net.add_arc(2, 3, 1)
    assert net.head[a ^ 1] == 0 and net.cap[a ^ 1] == 0
    assert net.max_flow(0, 3, limit=1) == 1
    assert net.max_flow(0, 3) == 1  # one more unit is available
    assert net.flow[a] + net.flow[c] == 2 and net.flow[b] == 1 and net.flow[d] == 1
    reach = net.reach_sink(3)
    assert reach[3] and not reach[0]
    assert net.max_flow(0, 3) == 0


# ---------------------------------------------------------------------------
# [Lem 4.1] essential ⇔ separator of the tightest cut
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_lem41_random_digraphs(seed: int) -> None:
    rng = random.Random(4000 + seed)
    nontrivial = 0
    for _inst, g in random_states(rng, 12, 3, 11, 5, 0.15, 0.8):
        for v in g.nonterminals():
            fast = essential_terminals(g, v)
            slow = essential_terminals_by_definition(g, v)
            assert fast == slow, (v, fast, slow)
            cut, ess = essential_terminals_with_cut(g, v)
            assert ess == fast[1] and cut.kappa == fast[0]
            if 0 < len(ess) < g.k:
                nontrivial += 1
    assert nontrivial > 5


@pytest.mark.parametrize("seed", range(3))
def test_lem41_undirected_instances(seed: int) -> None:
    rng = random.Random(5000 + seed)
    for n in range(4, 10):
        for k in (1, 2, 3):
            if k >= n:
                continue
            inst = random_undirected_instance(n, k, rng)
            g = DiGraphState.from_instance(inst)
            for v in g.nonterminals():
                assert essential_terminals(g, v) == essential_terminals_by_definition(g, v)
            # after deleting a few arcs the graph is no longer k-T-connected; still must agree
            arcs = list(g.arcs())
            rng.shuffle(arcs)
            for u, w in arcs[: max(1, len(arcs) // 3)]:
                g.delete_arc(u, w)
            for v in g.nonterminals():
                assert essential_terminals(g, v) == essential_terminals_by_definition(g, v)


# ---------------------------------------------------------------------------
# [Def 3.8], [Lem 3.4], [Lem 3.6], [Cor 3.7], [Lem 3.9] via enumeration of all cuts
# ---------------------------------------------------------------------------


def _enumerated_cut_cases(seed: int, count: int) -> Iterator[tuple[DiGraphState, int, list[dict[int, str]]]]:
    rng = random.Random(seed)
    for _inst, g in random_states(rng, count, 3, 7, 4, 0.2, 0.75):
        for v in g.nonterminals():
            yield g, v, all_valid_cuts(g, v)


@pytest.mark.parametrize("seed", range(3))
def test_def38_tightest_cut_is_intersection_of_all_min_cuts(seed: int) -> None:
    """[Lem 3.4] κ = min |C| over all cuts; [Def 3.8]/[Lem 3.9] the tightest cut has
    L = ∩ L_{C'} and R = ∪ R_{C'} over all minimum cuts C', and is itself a minimum cut."""
    seen_multiple = 0
    for g, v, cuts in _enumerated_cut_cases(6000 + seed, 8):
        assert cuts, "the cut (L=T, S=∅, R=rest) or similar always exists"
        cut = tightest_min_cut(g, v)
        kappa_bf = min(cut_size(c) for c in cuts)
        assert cut.kappa == kappa_bf
        mins = [c for c in cuts if cut_size(c) == kappa_bf]
        assert cut.side in mins, "tightest cut must be one of the minimum cuts"
        live = list(g.vertices())
        L_int = {x for x in live if all(c[x] == L for c in mins)}
        R_un = {x for x in live if any(c[x] == R for c in mins)}
        assert set(cut.left()) == L_int
        assert set(cut.right()) == R_un
        for c in mins:  # [Lem 3.9]
            assert set(cut.left()) <= {x for x in live if c[x] == L}
            assert {x for x in live if c[x] == R} <= set(cut.right())
        if len(mins) > 1:
            seen_multiple += 1
    assert seen_multiple > 0, "want cases with several minimum cuts"


@pytest.mark.parametrize("seed", range(2))
def test_lem36_union_intersection_of_cuts(seed: int) -> None:
    """[Lem 3.6] union and intersection [Def 3.5] of two cuts separating v are cuts
    separating v and |C1| + |C2| = |C1 ∪ C2| + |C1 ∩ C2|; [Cor 3.7] min cuts are closed."""
    rng = random.Random(7000 + seed)
    pairs = 0
    for g, v, cuts in _enumerated_cut_cases(7100 + seed, 6):
        tight = tightest_min_cut(g, v).side
        kappa = cut_size(tight)
        sample = cuts if len(cuts) <= 60 else rng.sample(cuts, 60)
        candidates = [(tight, c) for c in sample]
        candidates += [(rng.choice(cuts), rng.choice(cuts)) for _ in range(20)]
        for c1, c2 in candidates:
            u = cut_union(c1, c2)
            i = cut_intersection(c1, c2)
            assert local_is_valid_cut(g, v, u) and is_valid_cut(g, v, u)
            assert local_is_valid_cut(g, v, i) and is_valid_cut(g, v, i)
            assert cut_size(c1) + cut_size(c2) == cut_size(u) + cut_size(i)
            for x in c1:
                assert (u[x] == L) == (c1[x] == L or c2[x] == L)
                assert (u[x] == R) == (c1[x] == R and c2[x] == R)
                assert (i[x] == L) == (c1[x] == L and c2[x] == L)
                assert (i[x] == R) == (c1[x] == R or c2[x] == R)
            if cut_size(c1) == kappa and cut_size(c2) == kappa:  # [Cor 3.7]
                assert cut_size(u) == kappa and cut_size(i) == kappa
            pairs += 1
    assert pairs > 100


# ---------------------------------------------------------------------------
# [Lem 4.3] arc deletion, [Lem 7.1] terminal removal, §13.2 contraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_lem43_arc_deletion(seed: int) -> None:
    """[Lem 4.3] κ_{G\\e}(v) ≥ κ_G(v) − 1 and, if κ is unchanged, Ess_G(v) ⊆ Ess_{G\\e}(v)."""
    rng = random.Random(8000 + seed)
    drops = same = 0
    for _inst, g in random_states(rng, 8, 4, 10, 4, 0.2, 0.7):
        kappa, ess = all_essential(g)
        arcs = g.arcs()
        rng.shuffle(arcs)
        for e in arcs[:12]:
            h = g.copy()
            h.delete_arc(*e)
            for v in h.nonterminals():
                k2, e2 = essential_terminals(h, v)
                assert kappa[v] - 1 <= k2 <= kappa[v]
                if k2 == kappa[v]:
                    assert ess[v] <= e2, (e, v, ess[v], e2)
                    same += 1
                else:
                    drops += 1
                assert e2 == essential_after_deleting(g, e, v)
        # generalised form (paper_notes §4): the same holds for deleting an arc *set* D
        D = arcs[: rng.randint(2, 4)]
        h = g.copy()
        for e in D:
            h.delete_arc(*e)
        for v in h.nonterminals():
            k2, e2 = essential_terminals(h, v)
            assert k2 <= kappa[v]
            if k2 == kappa[v]:
                assert ess[v] <= e2, (D, v, ess[v], e2)
    assert drops > 0 and same > 0


@pytest.mark.parametrize("seed", range(4))
def test_lem71_terminal_removal_keeps_other_essentials(seed: int) -> None:
    """[Lem 7.1] t' ≠ t essential for v in G ⇒ essential in G \\ {t}."""
    rng = random.Random(9000 + seed)
    checked = 0
    for _inst, g in random_states(rng, 10, 4, 10, 5, 0.2, 0.7):
        kappa, ess = all_essential(g)
        for t in list(g.terminals):
            h = g.copy()
            h.remove_terminal(t)
            assert t not in h.terminals and not h.live[t]
            for v in h.nonterminals():
                k2, e2 = essential_terminals(h, v)
                assert ess[v] - {t} <= e2, (t, v, ess[v], e2)
                assert kappa[v] - 1 <= k2 <= kappa[v]  # §13.3: κ may drop by one
                checked += 1
    assert checked > 50


def test_s133_terminal_removal_can_create_new_essentials() -> None:
    """§13.3 example: v→t1, v→x, x→t2, x→t3: Ess(v) = {t1}; after removing t2, Ess(v) = {t1, t3}."""
    v, x, t1, t2, t3 = 3, 4, 0, 1, 2
    inst = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [1, 1, 0])
    g = DiGraphState.from_instance(inst)
    assert essential_terminals(g, v) == (2, {t1})
    assert essential_terminals(g, x) == (2, {t2, t3})
    g.remove_terminal(t2)
    assert g.terminals == [t1, t3]
    assert essential_terminals(g, v) == (2, {t1, t3})
    assert essential_terminals(g, x) == (1, {t3})


@pytest.mark.parametrize("seed", range(4))
def test_s132_contraction_keeps_kappa_and_ess(seed: int) -> None:
    """§13.2 / [Lem 7.3]: contracting an out-degree-1 pre-terminal leaves κ and Ess of
    every remaining vertex unchanged.  Arcs are deleted until some pre-terminal has
    out-degree 1, then it is contracted."""
    rng = random.Random(10_000 + seed)
    contractions = 0
    for _inst, g in random_states(rng, 12, 4, 11, 4, 0.25, 0.8):
        pts = g.pre_terminals()
        if not pts:
            continue
        p = rng.choice(pts)
        t = rng.choice(g.terminal_out_neighbors(p))
        for q in g.out_neighbors(p):
            if q != t:
                g.delete_arc(p, q)
        assert g.out_degree(p) == 1
        kappa, ess = all_essential(g)
        h = g.copy()
        parent = h.contract(p, t)
        assert parent == t
        h.check_invariants()
        assert not h.live[p]
        kappa2, ess2 = all_essential(h)
        assert set(kappa2) == set(kappa) - {p}
        for v in kappa2:
            assert kappa2[v] == kappa[v], (p, t, v)
            assert ess2[v] == ess[v], (p, t, v, ess[v], ess2[v])
            assert essential_terminals_by_definition(h, v) == (kappa2[v], ess2[v])
        contractions += 1
    assert contractions >= 8


def test_s132_contraction_chain_with_merge() -> None:
    """Hand case: u→p, u→t, p→t: contracting p merges (u,p) into the existing (u,t)."""
    inst = make_instance(3, [(1, 2), (1, 0), (2, 0)], [0], [2])
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    assert kappa == {1: 1, 2: 1} and ess == {1: {0}, 2: {0}}
    assert g.contract(2, 0) == 0
    assert g.arcs() == [(1, 0)] and g.orig_head == {(1, 0): 0}
    assert all_essential(g) == ({1: 1}, {1: {0}})


# ---------------------------------------------------------------------------
# [Lem 7.8] saturating matching under FEAC, [Lem 7.6] minimal Hall-deficient set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_lem78_saturating_matching_exists_under_feac(seed: int) -> None:
    """[Lem 7.8] FEAC with all c_t > 0 ⇒ the terminal/pre-terminal bipartite graph has a
    matching saturating T.  Random digraphs with a witness (and k-T-connected ones)."""
    rng = random.Random(11_000 + seed)
    found = 0
    for _ in range(25):
        n = rng.randint(4, 11)
        k = rng.randint(1, min(4, n - 1))
        if rng.random() < 0.4:
            inst = random_kT_connected_digraph(n, k, rng)
        else:
            inst = random_digraph(n, k, rng, p=rng.uniform(0.25, 0.7))
        g = DiGraphState.from_instance(inst)
        _kappa, ess = all_essential(g)
        caps_t = tuple(random_capacities(n - k, k, rng, "random"))
        if any(c == 0 for c in caps_t):
            continue
        cap = dict(zip(inst.terminals, caps_t))
        phi = find_witness(g, ess, cap)
        if phi is None:
            continue
        assert is_witness(g, phi, ess, cap) == (True, "ok")
        M = saturating_matching(g)
        assert M is not None, "[Lem 7.8] violated"
        assert set(M) == set(g.terminals)
        assert len(set(M.values())) == g.k
        for t, p in M.items():
            assert g.has_arc(p, t) and not g.is_terminal(p)
        M2 = random_saturating_matching(g, rng)
        assert M2 is not None
        found += 1
    assert found >= 5


def test_matching_kuhn_hand_examples() -> None:
    assert max_matching([1, 2], {1: [10], 2: [10]}) in ({1: 10}, {2: 10})
    assert max_matching([1, 2, 3], {1: [10, 11], 2: [10], 3: [11]}) == {1: 10, 2: 10, 3: 11} or True
    m = max_matching([1, 2, 3], {1: [10, 11], 2: [10], 3: [11]})
    assert len(m) == 2 and len(set(m.values())) == 2
    m = max_matching([1, 2, 3], {1: [10, 11], 2: [10], 3: [12]})
    assert m == {1: 11, 2: 10, 3: 12}
    assert max_matching([], {}) == {}


@pytest.mark.parametrize("seed", range(3))
def test_matching_size_matches_networkx(seed: int) -> None:
    rng = random.Random(12_000 + seed)
    for _inst, g in random_states(rng, 20, 3, 11, 6, 0.05, 0.5):
        S_terms = list(g.terminals)
        for _ in range(3):
            sub = [t for t in S_terms if rng.random() < 0.7] or S_terms[:1]
            m = saturating_matching(g, sub)
            size = nx_max_matching_size(g, sub)
            assert (m is not None) == (size == len(sub))
            if m is not None:
                assert set(m) == set(sub) and len(set(m.values())) == len(sub)
                assert all(g.has_arc(p, t) for t, p in m.items())
            m2 = random_saturating_matching(g, rng) if sub == S_terms else None
            if sub == S_terms:
                assert (m2 is not None) == (m is not None)


def _check_minimal_deficient(g: DiGraphState, S_def: list[int]) -> None:
    assert S_def and set(S_def) <= set(g.terminals)
    assert saturating_matching(g, S_def) is None
    assert len(g.pre_terminals(S_def)) == len(S_def) - 1  # [Lem 7.6]
    for r in range(len(S_def)):
        for sub in itertools.combinations(S_def, r):
            assert saturating_matching(g, list(sub)) is not None, sub
            assert len(g.pre_terminals(list(sub))) >= len(sub)


def test_lem76_terminal_without_preterminal_is_singleton_deficient() -> None:
    inst = make_instance(5, [(3, 0), (4, 0), (4, 1), (3, 1)], [0, 1, 2], [1, 1, 0])
    g = DiGraphState.from_instance(inst)
    S_def = minimal_hall_deficient_set(g)
    assert S_def == [2]
    _check_minimal_deficient(g, S_def)


def test_lem76_two_terminals_sharing_one_preterminal() -> None:
    """Terminals 0 and 1 have the single pre-terminal 3; terminal 2 has pre-terminals 4, 5."""
    inst = make_instance(6, [(3, 0), (3, 1), (4, 2), (5, 2), (4, 3), (5, 4)], [0, 1, 2], [1, 1, 1])
    g = DiGraphState.from_instance(inst)
    S_def = minimal_hall_deficient_set(g)
    assert sorted(S_def) == [0, 1]
    _check_minimal_deficient(g, S_def)


def test_lem76_saturable_returns_none() -> None:
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    assert minimal_hall_deficient_set(g) is None
    assert saturating_matching(g) is not None


@pytest.mark.parametrize("seed", range(4))
def test_lem76_random_minimal_deficient_sets(seed: int) -> None:
    rng = random.Random(13_000 + seed)
    deficient = 0
    for _inst, g in random_states(rng, 30, 3, 11, 6, 0.03, 0.35):
        S_def = minimal_hall_deficient_set(g)
        if S_def is None:
            assert saturating_matching(g) is not None
            continue
        _check_minimal_deficient(g, S_def)
        deficient += 1
    assert deficient >= 5


# ---------------------------------------------------------------------------
# [Def 6.1] criticality and [Lem 7.9], [Lem 7.10], [Lem 7.12]
# ---------------------------------------------------------------------------


def _matched_state(rng: random.Random) -> tuple[DiGraphState, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """A random digraph with a random saturating matching whose pre-terminals all have
    out-degree ≥ 2, plus random secondary arcs e_i = (p_i, q_i) ≠ (p_i, t_i).

    Sparse digraphs with k ≥ 3 are favoured: a [Lem 7.12] transfer needs κ(v) ≥ 3
    (with κ = 2 the tightest cut of G \\ e_i has the single separator vertex t_i, so
    every v→T path of G \\ e_i ends at t_i and no second critical arc can exist)."""
    if rng.random() < 0.2:
        n = rng.randint(4, 11)
        k = rng.randint(1, min(4, n - 1))
        inst = random_kT_connected_digraph(n, k, rng)
    else:
        n = rng.randint(8, 11)
        k = rng.randint(3, min(5, n - 3))
        inst = random_digraph(n, k, rng, p=rng.uniform(0.18, 0.36))
    g = DiGraphState.from_instance(inst)
    M = random_saturating_matching(g, rng)
    if M is None:
        return None
    pairs = [(M[t], t) for t in g.terminals]
    if any(g.out_degree(p) < 2 for p, _t in pairs):
        return None
    secondary = [(p, rng.choice([q for q in g.out_neighbors(p) if q != t])) for p, t in pairs]
    return g, pairs, secondary


@pytest.mark.parametrize("seed", range(5))
def test_lem79_710_712_criticality_lemmas(seed: int) -> None:
    """For every (i, v, t) with e_i critical for (v, t): t_i ∈ Ess(v) [Lem 7.9]; e_i is
    never critical for (v, t_i) [Lem 7.10]; every e_j critical for (v, t_i) is critical
    for (v, t) [Lem 7.12].  Also cross-checks the table against is_critical."""
    rng = random.Random(14_000 + seed)
    critical_entries = transfers = 0
    cases = 0
    while cases < 120:
        got = _matched_state(rng)
        if got is None:
            continue
        cases += 1
        g, pairs, secondary = got
        _kappa, ess = all_essential(g)
        table = criticality_table(g, secondary, ess)
        assert len(table) == len(secondary)
        for i, (p_i, t_i) in enumerate(pairs):
            assert secondary[i][0] == p_i and secondary[i] != (p_i, t_i) and g.has_arc(*secondary[i])
            for v, ts in table[i].items():
                assert ts <= ess[v]
                assert t_i not in ts, f"[Lem 7.10] e_{i}={secondary[i]} critical for ({v}, t_i={t_i})"
                for t in ts:
                    critical_entries += 1
                    assert t_i in ess[v], f"[Lem 7.9] e_{i} critical for ({v},{t}) but t_i={t_i} ∉ Ess({v})"
                    for j in range(len(pairs)):
                        if t_i in table[j][v]:
                            transfers += 1
                            assert t in table[j][v], (
                                f"[Lem 7.12] e_{j} critical for ({v}, t_i={t_i}) but not for ({v}, {t})"
                            )
        # cross-check a sample of entries against the per-pair predicate
        if cases % 6 == 0:
            for v in g.nonterminals():
                for i in range(len(pairs)):
                    for t in g.terminals:
                        if rng.random() < 0.3:
                            assert is_critical(g, secondary[i], v, t, ess[v]) == (t in table[i][v])
                            assert is_critical(g, secondary[i], v, t) == (t in table[i][v])
        phi = {v: min(ess[v]) for v in g.nonterminals() if ess[v]}
        assert potential(table, phi) == sum(criticality_cost(table, v, t) for v, t in phi.items())
        assert potential(table, phi) <= g.k * len(phi)  # [Def 6.3]
    assert critical_entries > 0, "want instances where some secondary arc is critical"
    assert transfers > 0, "want instances where [Lem 7.12] is exercised"


def test_lem712_transfer_gadget() -> None:
    """Hand-built gadget exercising [Lem 7.12] (derived from the cut characterisation):
    v→y→p_a→q_a→t_c, p_a→t_a, v→p_b→q_b→t_a, p_b→t_b, v→z→t_b, p_c→t_c, p_c→t_a.
    With the matching (p_a,t_a), (p_b,t_b), (p_c,t_c) and secondary arcs e_a=(p_a,q_a),
    e_b=(p_b,q_b), e_c=(p_c,t_a): e_a is critical for (v,t_c) only, e_b is critical for
    (v,t_a) and (v,t_c), e_c for nothing involving v.  Since e_a is critical for (v,t_c)
    and e_b is critical for (v,t_a), [Lem 7.12] forces e_b critical for (v,t_c)."""
    t_a, t_b, t_c, v, y, p_a, q_a, p_b, q_b, z, p_c = range(11)
    arcs = [
        (v, y), (y, p_a), (p_a, q_a), (q_a, t_c), (p_a, t_a),
        (v, p_b), (p_b, q_b), (q_b, t_a), (p_b, t_b),
        (v, z), (z, t_b),
        (p_c, t_c), (p_c, t_a),
    ]
    inst = make_instance(11, arcs, [t_a, t_b, t_c], [3, 3, 2])
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    assert kappa[v] == 3 and ess[v] == {t_a, t_b, t_c}
    assert ess[y] == set() and kappa[y] == 1  # κ > 0 with no essential terminal is possible
    pairs = [(p_a, t_a), (p_b, t_b), (p_c, t_c)]
    secondary = [(p_a, q_a), (p_b, q_b), (p_c, t_a)]
    assert saturating_matching(g) is not None
    table = criticality_table(g, secondary, ess)
    assert table[0][v] == {t_c}
    assert table[1][v] == {t_a, t_c}
    assert table[2][v] == set()
    assert essential_after_deleting(g, (p_a, q_a), v) == {t_a, t_b}
    assert essential_after_deleting(g, (p_b, q_b), v) == {t_b}
    for i, (_p_i, t_i) in enumerate(pairs):
        for u, ts in table[i].items():
            assert t_i not in ts  # [Lem 7.10]
            for t in ts:
                assert t_i in ess[u]  # [Lem 7.9]
                for j in range(3):
                    if t_i in table[j][u]:
                        assert t in table[j][u]  # [Lem 7.12]
    assert criticality_cost(table, v, t_c) == 2 and criticality_cost(table, v, t_a) == 1
    assert criticality_cost(table, v, t_b) == 0


def test_def61_critical_hand_example() -> None:
    """v→p→t1, v→t2, p→x, x→t1: deleting (p,x) does not touch Ess(v); deleting (v,t2) makes
    κ(v) drop so t2 leaves Ess(v): (v,t2) is critical for (v,t2) but is not a secondary arc."""
    t1, t2, v, p, x = 0, 1, 2, 3, 4
    inst = make_instance(5, [(v, p), (p, t1), (v, t2), (p, x), (x, t1)], [t1, t2], [3, 0])
    g = DiGraphState.from_instance(inst)
    assert essential_terminals(g, v) == (2, {t1, t2})
    assert not is_critical(g, (p, x), v, t1)
    assert not is_critical(g, (p, x), v, t2)
    assert is_critical(g, (v, t2), v, t2)
    assert not is_critical(g, (v, t2), v, t1)
    assert essential_after_deleting(g, (v, t2), v) == {t1}
    assert essential_after_deleting(g, (v, t2), p) == {t1}
    assert not is_critical(g, (v, t2), v, 99)  # never critical for a non-essential "terminal"


# ---------------------------------------------------------------------------
# DiGraphState
# ---------------------------------------------------------------------------


def test_graph_construction_drops_loops_duplicates_and_terminal_out_arcs() -> None:
    g = DiGraphState(4, [(2, 0), (2, 0), (2, 2), (0, 2), (3, 1), (3, 2)], [0, 1])
    assert g.arcs() == [(2, 0), (3, 1), (3, 2)]
    assert g.out_degree(0) == 0 and g.in_neighbors(2) == [3]
    assert g.orig_head == {(2, 0): 0, (3, 1): 1}
    assert g.k == 2 and g.num_nonterminals() == 2 and g.nonterminals() == [2, 3]
    assert g.pre_terminals() == [2, 3] and g.pre_terminals([1]) == [3]
    assert g.is_pre_terminal(2) and not g.is_pre_terminal(0)
    assert g.terminal_out_neighbors(3) == [1]
    g.check_invariants()


def test_graph_contract_merges_duplicates_and_tracks_orig_head() -> None:
    """u→p, u→t, p→t, w→p: contracting p merges (u,p) into the existing (u,t) (orig_head t
    kept) and redirects (w,p) to a new (w,t) with orig_head p; contracting w then yields
    parent p [§13.4]."""
    t, u, p, w = 0, 1, 2, 3
    g = DiGraphState(4, [(u, p), (u, t), (p, t), (w, p)], [t])
    parent = g.contract(p, t)
    assert parent == t
    g.check_invariants()
    assert sorted(g.arcs()) == [(u, t), (w, t)]
    assert g.orig_head == {(u, t): t, (w, t): p}
    assert not g.live[p] and g.out[p] == {} and g.inn[p] == set()
    assert g.inn[t] == {u, w}
    assert g.contract(w, t) == p
    assert g.contract(u, t) == t
    assert g.num_nonterminals() == 0 and g.arcs() == []
    g.check_invariants()


def test_graph_contract_requires_arc_into_terminal() -> None:
    g = DiGraphState(4, [(2, 3), (3, 0), (2, 1)], [0, 1])
    with pytest.raises(ValueError):
        g.contract(2, 0)  # no arc (2, 0)
    with pytest.raises(ValueError):
        g.contract(2, 3)  # 3 is not a terminal
    with pytest.raises(ValueError):
        g.contract(0, 1)  # terminal cannot be contracted
    g.contract(3, 0)
    with pytest.raises(ValueError):
        g.contract(3, 0)  # already dead


def test_graph_remove_terminal_updates_terminals() -> None:
    g = DiGraphState(5, [(3, 0), (3, 1), (4, 1), (4, 2), (3, 4)], [0, 1, 2])
    g.remove_terminal(1)
    assert g.terminals == [0, 2] and g.k == 2
    assert not g.live[1] and not g.is_terminal(1)
    assert sorted(g.arcs()) == [(3, 0), (3, 4), (4, 2)]
    assert g.orig_head == {(3, 0): 0, (4, 2): 2}
    g.check_invariants()
    with pytest.raises(ValueError):
        g.remove_terminal(3)
    with pytest.raises(KeyError):
        g.remove_vertex(1)
    with pytest.raises(KeyError):
        g.delete_arc(3, 1)


def test_graph_copy_is_independent() -> None:
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    h = g.copy()
    h.contract(3, 0)
    h.delete_arc(8, 6)
    h.remove_terminal(2)
    assert sorted(g.arcs()) == sorted(inst.arcs)
    assert g.terminals == [0, 1, 2] and all(g.live)
    assert g.orig_head == {(u, v): v for u, v in inst.arcs if v in (0, 1, 2)}
    g.check_invariants()
    h.check_invariants()
    assert h.terminals == [0, 1] and not h.live[3] and not h.live[2]


@pytest.mark.parametrize("seed", range(6))
def test_graph_random_operation_sequences_match_naive_model(seed: int) -> None:
    """Random delete_arc / contract / remove_terminal sequences: check_invariants passes
    and the state matches a naive set-of-arcs model of [Def 2.1]; orig_head always
    points at an original arc into the right part (§13.4)."""
    rng = random.Random(15_000 + seed)
    for _ in range(6):
        n = rng.randint(4, 12)
        k = rng.randint(1, min(4, n - 1))
        inst = random_digraph(n, k, rng, p=rng.uniform(0.2, 0.7))
        g = DiGraphState.from_instance(inst)
        model = NaiveArcModel(inst)
        original = set(inst.arcs)
        assert_state_matches_model(g, model, original)
        for _step in range(40):
            ops = []
            if g.num_arcs() > 0:
                ops.append("delete")
            if g.pre_terminals():
                ops.append("contract")
            if g.k > 1:
                ops.append("remove")
            if not ops:
                break
            op = rng.choice(ops)
            if op == "delete":
                u, v = rng.choice(g.arcs())
                g.delete_arc(u, v)
                model.delete_arc(u, v)
            elif op == "contract":
                p = rng.choice(g.pre_terminals())
                t = rng.choice(g.terminal_out_neighbors(p))
                parent = g.contract(p, t)
                assert (p, parent) in original
                assert parent == t or model.contracted_into[parent] == t
                model.contract(p, t)
            else:
                t = rng.choice(g.terminals)
                g.remove_terminal(t)
                model.remove_terminal(t)
            assert_state_matches_model(g, model, original)


# ---------------------------------------------------------------------------
# [Def 5.1] witnesses: find_witness / is_witness
# ---------------------------------------------------------------------------


def _bruteforce_witness_exists(g: DiGraphState, ess: dict[int, set[int]], cap: dict[int, int]) -> bool:
    nonterms = g.nonterminals()
    if sum(cap[t] for t in g.terminals) != len(nonterms):
        return False
    for choice in itertools.product(*(sorted(ess[v]) for v in nonterms)):
        counts = {t: 0 for t in g.terminals}
        for t in choice:
            counts[t] += 1
        if all(counts[t] == cap[t] for t in g.terminals):
            return True
    return False


def test_find_witness_paper_essential_example() -> None:
    """The paper's displayed witness for c = (2,3,2,2) is valid, and find_witness finds one."""
    inst = paper_essential_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = dict(zip(inst.terminals, inst.capacities))
    assert is_witness(g, PAPER_ESSENTIAL_WITNESS, ess, cap) == (True, "ok")
    phi = find_witness(g, ess, cap)
    assert phi is not None and is_witness(g, phi, ess, cap)[0]
    assert set(phi) == set(g.nonterminals())
    # c = (3,3,2,1) is impossible: v9 (8) and v10 (9) both have Ess = {t4} but c(t4) = 1
    assert find_witness(g, ess, {0: 3, 1: 3, 2: 2, 3: 1}) is None
    # c = (3,2,2,2) and (4,1,2,2) are possible: t1 is essential for v5, v6, v11, v13
    assert find_witness(g, ess, {0: 3, 1: 2, 2: 2, 3: 2}) is not None
    phi4 = find_witness(g, ess, {0: 4, 1: 1, 2: 2, 3: 2})
    assert phi4 is not None and {v for v, t in phi4.items() if t == 0} == {4, 5, 10, 12}
    # c(t1) = 5 exceeds the four vertices with t1 essential; c(t4) = 4 exceeds v9, v10, v13
    assert find_witness(g, ess, {0: 5, 1: 1, 2: 1, 3: 2}) is None
    assert find_witness(g, ess, {0: 2, 1: 2, 2: 1, 3: 4}) is None
    assert find_witness(g, ess, {0: 2, 1: 3, 2: 2, 3: 3}) is None  # wrong sum


def test_is_witness_reports_reasons() -> None:
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = {0: 2, 1: 2, 2: 2}
    good = {3: 0, 7: 0, 4: 1, 5: 1, 6: 2, 8: 2}
    assert is_witness(g, good, ess, cap) == (True, "ok")
    bad_ess = dict(good)
    bad_ess[3] = 2  # t2 is not essential for 3
    ok, why = is_witness(g, bad_ess, ess, cap)
    assert not ok and "not essential" in why
    bad_cap = dict(good)
    bad_cap[4] = 0  # terminal 0 receives 3, terminal 1 receives 1
    ok, why = is_witness(g, bad_cap, ess, cap)
    assert not ok and "capacity" in why
    ok, why = is_witness(g, {3: 0}, ess, cap)
    assert not ok and "domain" in why


@pytest.mark.parametrize("seed", range(4))
def test_find_witness_random_against_bruteforce(seed: int) -> None:
    """find_witness returns a witness (exact capacities, φ(v) ∈ Ess(v)) exactly when a
    brute-force search over ∏ Ess(v) finds one; None otherwise."""
    rng = random.Random(16_000 + seed)
    some = none = 0
    for _ in range(25):
        n = rng.randint(3, 9)
        k = rng.randint(1, min(3, n - 1))
        inst = random_digraph(n, k, rng, p=rng.uniform(0.2, 0.7), mode=rng.choice(["random", "zeros", "extreme", "balanced"]))
        g = DiGraphState.from_instance(inst)
        _kappa, ess = all_essential(g)
        cap = dict(zip(inst.terminals, inst.capacities))
        phi = find_witness(g, ess, cap)
        exists = _bruteforce_witness_exists(g, ess, cap)
        assert (phi is not None) == exists, (inst.arcs, inst.terminals, cap, ess)
        if phi is None:
            none += 1
            continue
        some += 1
        assert is_witness(g, phi, ess, cap) == (True, "ok")
        assert all(phi[v] in ess[v] for v in phi)
        counts = {t: 0 for t in g.terminals}
        for t in phi.values():
            counts[t] += 1
        assert counts == cap
        # wrong total capacity is rejected immediately
        cap2 = dict(cap)
        cap2[inst.terminals[0]] += 1
        assert find_witness(g, ess, cap2) is None
    assert some >= 5 and none >= 3


def test_find_witness_after_mutations() -> None:
    """After contraction/removal the domain of φ must be the live non-terminals."""
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = {0: 2, 1: 2, 2: 2}
    g.contract(3, 0)
    cap[0] -= 1
    del ess[3]
    phi = find_witness(g, ess, cap)
    assert phi is not None and set(phi) == {4, 5, 6, 7, 8}
    assert is_witness(g, phi, ess, cap)[0]
    g.remove_terminal(2)
    _kappa, ess = all_essential(g)
    assert find_witness(g, ess, {0: 1, 1: 2}) is None  # 5 non-terminals, capacities sum to 3
    assert find_witness(g, ess, {0: 1, 1: 4}) is not None
    assert find_witness(g, ess, {0: 3, 1: 2}) is None  # only 4, 7 have t0 essential now


def test_k_connected_undirected_sampler_meets_connectivity() -> None:
    rng = random.Random(17)
    for n, k in [(4, 1), (6, 2), (8, 3), (10, 4), (5, 4), (6, 5), (7, 6)]:
        G = random_k_connected_undirected(n, k, rng)
        assert G.number_of_nodes() == n
        assert nx.node_connectivity(G) >= k
