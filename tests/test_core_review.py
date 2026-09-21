"""Adversarial differential fuzzing of the C++ core primitives (``glsolver._core``) against the
validated pure-Python reference ``glref`` and the independent verifier ``glsolver.verify``.

Thousands of *seeded* random small instances (n in 3..15, k in 1..5, directed and undirected,
including the degenerate shapes: k = 1, k = n - 1, isolated vertices, vertices with no path to T,
terminals without in-arcs, raw arc lists with self-loops / parallel arcs / arcs leaving terminals)
are pushed through every primitive and compared with the reference on exactly the properties that
docs/paper_notes.md and docs/optimizations.md promise:

* [Prop 4.2] ``tightest_cut``: kappa, the (unique, [Def 3.8]) L/S/R sides and Ess identical to
  ``glref.flow.tightest_min_cut``; the returned paths are a valid family [Def 3.2];
* ``Graph`` mutations ([Def 2.1], §13.4): arc-for-arc identical to ``glref.graph.DiGraphState``
  after random operation sequences (deletions, contractions of any out-degree, terminal removal,
  vertex removal), including the ``orig_head`` parent and the merged-duplicate case;
* ``EssentialOracle`` after random operation sequences (O1–O4, P2): kappa always equals the
  reference; the stored Ess is a certified subset, and equals the reference (sides included)
  whenever the oracle claims an exact cut; ``refresh_all_cuts`` reproduces the reference exactly;
  the arc-user index is exact;
* ``evaluate_deletion`` (O2/O5) against ``glref.critical.is_critical`` for every (v, t) and single
  arcs, and against the reference on ``G \\ D`` for arc sets (O5-style ``out(p) \\ {one arc}`` and
  random sets), including arcs whose tail is the evaluated vertex itself;
* matching [Lem 7.8] and the minimal Hall-deficient set [Lem 7.6]: saturability identical to the
  reference on random terminal subsets, validity of the matching, minimality and ``|PT| = |S| - 1``;
* min-cost flow [Prop 5.4]: (value, cost) identical to the reference on random networks with small
  and with 2^40-sized capacities/costs; returned flows feasible; int64 overflow is reported, never
  silently wrapped;
* ``dag_partition`` [Alg 5]: both variants (heaps / P1 stack) and all three policies accepted by the
  verifier, identical to each other and to ``glref.dag.gl_dag_partition`` on random DAGs with
  out-degree exactly k and larger (a vertex pushed by several parents), capacities with zeros and
  weights up to 2^40;
* thread safety / determinism: ``threads=8`` runs of ``compute_all``, ``evaluate_deletion``,
  ``after_terminal_removal`` and ``refresh_all_cuts`` repeated many times give bit-identical
  results to ``threads=1``.

Every assertion names the seed of the failing instance. Runnable in isolation:
``pytest tests/test_core_review.py`` (about 7 s single-process on the release build; roughly
20x slower under GLCORE_SANITIZE=ON, which is dominated by the C++ side).

``test_after_contraction_out_degree_ge_2_keeps_ess_contract`` documents a bug found by the core review
(``EssentialOracle::after_contraction`` kept serving stale "exact" cuts after contracting a pre-terminal
of out-degree >= 2); the hook now advances its exact-cut epoch in that case (see the test's comment block).
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

import pytest

from glref.critical import is_critical
from glref.dag import gl_dag_partition
from glref.flow import tightest_min_cut
from glref.graph import DiGraphState
from glref.matching import minimal_hall_deficient_set as ref_minimal_hall_deficient_set
from glref.matching import saturating_matching as ref_saturating_matching
from glref.mincostflow import MinCostFlow as RefMinCostFlow
from glsolver import _core
from glsolver.instance import Instance, make_instance
from glsolver.verify import verify_instance_parts

Graph = _core.Graph
EssentialOracle = _core.EssentialOracle

BIG = 1 << 40


# ---------------------------------------------------------------------------
# random raw instances (fed *unnormalized* to both implementations)
# ---------------------------------------------------------------------------


def random_raw_instance(rng: random.Random, n_max: int = 15) -> tuple[int, list[tuple[int, int]], list[int]]:
    """A raw ``(n, arcs, terminals)`` triple.  The arc list may contain self-loops, parallel arcs and
    arcs leaving terminals (the §2 conventions drop them at load time in both implementations);
    degenerate shapes are injected with fixed probabilities."""
    n = rng.randint(3, n_max)
    shape = rng.random()
    if shape < 0.12:
        k = 1
    elif shape < 0.22:
        k = n - 1
    elif shape < 0.25:
        k = 0  # no terminals at all: kappa == 0 everywhere
    else:
        k = rng.randint(1, min(5, n - 1))
    terminals = rng.sample(range(n), k)
    tset = set(terminals)
    directed = rng.random() < 0.6
    p = rng.choice([0.08, 0.15, 0.25, 0.4, 0.6, 0.9])
    arcs: list[tuple[int, int]] = []
    if directed:
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < p:
                    arcs.append((u, v))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    arcs.append((u, v))
                    arcs.append((v, u))
    if rng.random() < 0.3:  # isolated vertex
        iso = rng.randrange(n)
        arcs = [a for a in arcs if iso not in a]
    if terminals and rng.random() < 0.3:  # terminal without in-arcs
        t = rng.choice(terminals)
        arcs = [a for a in arcs if a[1] != t]
    if n - k > 0 and rng.random() < 0.3:  # non-terminal with no path to T (no out-arcs)
        x = rng.choice([v for v in range(n) if v not in tset])
        arcs = [a for a in arcs if a[0] != x]
    if terminals and n - k > 0 and rng.random() < 0.45:  # out-degree-1 pre-terminals (contraction fodder)
        for x in rng.sample([v for v in range(n) if v not in tset], min(2, n - k)):
            t = rng.choice(terminals)
            arcs = [a for a in arcs if a[0] != x] + [(x, t)]
    if rng.random() < 0.4:  # raw noise dropped by the §2 conventions
        x = rng.randrange(n)
        arcs.append((x, x))
        if arcs:
            arcs.append(rng.choice(arcs))
        if terminals:
            arcs.append((rng.choice(terminals), rng.randrange(n)))
    rng.shuffle(arcs)
    return n, arcs, terminals


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def same_graph(g: Graph, ref: DiGraphState, where: str) -> None:
    """The core graph and the reference graph are identical (arcs, live mask, terminals, degrees)."""
    assert set(g.arcs()) == set(ref.arcs()), where
    assert g.num_live_arcs() == ref.num_arcs(), where
    assert g.pre_terminals() == ref.pre_terminals(), where
    assert list(g.terminals) == list(ref.terminals), where
    assert [g.live(v) for v in range(g.n)] == list(ref.live), where
    assert g.num_live_nonterminals() == ref.num_nonterminals(), where
    for v in ref.vertices():
        assert set(g.out_neighbors(v)) == set(ref.out_neighbors(v)), (where, v)
        assert set(g.in_neighbors(v)) == set(ref.in_neighbors(v)), (where, v)
        assert g.out_degree(v) == ref.out_degree(v), (where, v)
        assert g.in_degree(v) == len(ref.inn[v]), (where, v)
        assert g.is_terminal(v) == ref.is_terminal(v), (where, v)
        assert g.is_pre_terminal(v) == ref.is_pre_terminal(v), (where, v)
    for u, v in ref.arcs():
        if ref.is_terminal(v):
            assert g.orig_head(u, v) == ref.orig_head[(u, v)], (where, u, v)
    g.check_invariants()
    ref.check_invariants()


def check_paths(arcs: set[tuple[int, int]], terminals: Iterable[int], v: int, kappa: int, paths, where) -> None:
    """[Def 3.2]: kappa simple paths from v to distinct terminals, vertex-disjoint except v, over
    present arcs only."""
    tset = set(terminals)
    assert len(paths) == kappa, (where, v, len(paths), kappa)
    seen: set[int] = set()
    ends: set[int] = set()
    for p in paths:
        assert p[0] == v and len(p) >= 2, (where, v, p)
        assert p[-1] in tset, (where, v, p)
        assert len(set(p)) == len(p), (where, v, p)
        for a, b in zip(p, p[1:]):
            assert (a, b) in arcs, (where, v, p, (a, b))
        inner = set(p[1:])
        assert not (inner & seen) and v not in inner, (where, v, p)
        seen |= inner
        assert p[-1] not in ends, (where, v, p)
        ends.add(p[-1])


def ref_cut(ref: DiGraphState, v: int):
    cut = tightest_min_cut(ref, v)
    return cut.kappa, cut.side, frozenset(cut.essential_terminals(ref))


class Harness:
    """Core Graph + EssentialOracle next to the reference DiGraphState on the same raw input."""

    def __init__(self, n: int, arcs: Sequence[tuple[int, int]], terminals: Sequence[int], threads: int, seed: int):
        self.terms0 = list(terminals)
        self.g = Graph(n, list(arcs), list(terminals))
        self.ref = DiGraphState(n, list(arcs), list(terminals))
        self.o = EssentialOracle(self.g, threads, seed)
        self.o.compute_all()

    def ess_ids(self, idx: Iterable[int]) -> frozenset[int]:
        return frozenset(self.terms0[i] for i in idx)

    def compare(self, exact: bool, where) -> None:
        """kappa == reference; Ess certified subset (== reference when exact is claimed, then the sides
        are the unique tightest cut too); paths valid."""
        arcs = set(self.ref.arcs())
        for v in self.ref.nonterminals():
            rk, rside, ress = ref_cut(self.ref, v)
            assert self.o.kappa(v) == rk, (where, v, self.o.kappa(v), rk)
            ess = self.ess_ids(self.o.ess(v))
            if exact:
                assert self.o.cut_exact(v), (where, v)
            if exact or self.o.cut_exact(v):
                assert ess == ress, (where, v, ess, ress)
                assert self.o.sides(v) == rside, (where, v)
            else:
                assert ess <= ress, (where, v, ess, ress)
                assert self.o.sides(v) == {}, (where, v)
            check_paths(arcs, self.ref.terminals, v, rk, self.o.paths(v), where)

    def users_consistent(self, where) -> None:
        expected: dict[tuple[int, int], set[int]] = {}
        for v in self.ref.nonterminals():
            for p in self.o.paths(v):
                for a, b in zip(p, p[1:]):
                    expected.setdefault((a, b), set()).add(v)
        for u, w in self.ref.arcs():
            assert set(self.o.users_of_arc(u, w)) == expected.get((u, w), set()), (where, u, w)

    def snapshot(self):
        return sorted(
            (v, self.o.kappa(v), tuple(self.o.ess(v)), self.o.cut_exact(v),
             tuple(sorted(self.o.sides(v).items())), tuple(map(tuple, self.o.paths(v))))
            for v in self.ref.nonterminals()
        )


def check_evaluate_deletion(h: Harness, D: list[tuple[int, int]], where, rng: random.Random) -> dict:
    """``evaluate_deletion(D, exact_cuts=True)`` against the reference on ``G \\ D``:
    affected vertices get the exact (kappa, Ess) of ``G \\ D`` and valid paths there; unaffected
    vertices satisfy O1 (kappa unchanged, stored Ess still a subset).  For a single arc the
    criticality [Def 6.1] of every (v, t) is compared with ``glref.critical.is_critical``."""
    res = h.o.evaluate_deletion(D, True)
    hh = h.ref.copy()
    for e in D:
        hh.delete_arc(*e)
    arcs_after = set(hh.arcs())
    users = set()
    for e in D:
        users |= set(h.o.users_of_arc(*e))
    assert set(res) == users, (where, D)
    for v in h.ref.nonterminals():
        rk_before, _s, ress_before = ref_cut(h.ref, v)
        rk, _side, ress = ref_cut(hh, v)
        ess_core_before = h.ess_ids(h.o.ess(v))
        assert ess_core_before <= ress_before, (where, D, v)
        if v in res:
            k_after, ess_after_idx = res[v]
            ess_after = h.ess_ids(ess_after_idx)
            assert k_after == rk, (where, D, v, k_after, rk)
            assert ess_after == ress, (where, D, v, ess_after, ress)
            check_paths(arcs_after, hh.terminals, v, rk, h.o.evaluated_paths(v), (where, D))
            if len(D) == 1 and h.o.cut_exact(v):
                # crit(e, v, t) = [t in Ess_G(v)] and [t not in Ess_{G\e}(v)]  [Def 6.1]
                crit_core = {t for t in h.terms0 if t in ess_core_before and t not in ess_after}
                crit_ref = ress_before - ress
                assert crit_core == crit_ref, (where, D, v, crit_core, crit_ref)
                # literal tie to glref.critical.is_critical on one random terminal (it recomputes the
                # flow of G \ e per call, so all k would multiply the reference work)
                live_terms = list(h.ref.terminals)
                if live_terms:
                    t = rng.choice(live_terms)
                    assert (t in crit_core) == is_critical(h.ref, D[0], v, t, ress_before), (where, D, v, t)
        else:
            # O1: the stored flow avoids D -> kappa unchanged, Ess can only grow, nothing critical
            assert rk == rk_before == h.o.kappa(v), (where, D, v)
            assert ess_core_before <= ress, (where, D, v)
            if len(D) == 1:
                t = rng.choice(h.terms0) if h.terms0 else None
                if t is not None and t in h.ref.terminals:
                    assert not is_critical(h.ref, D[0], v, t, ress_before), (where, D, v, t)
    return res


# ---------------------------------------------------------------------------
# (1) Graph load + tightest cut [Prop 4.2] on many raw instances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", range(60))
def test_fuzz_tightest_cut_matches_reference(chunk: int) -> None:
    for i in range(30):
        seed = 10_000 + chunk * 30 + i
        rng = random.Random(seed)
        n, arcs, terminals = random_raw_instance(rng)
        g = Graph(n, arcs, terminals)
        ref = DiGraphState(n, arcs, terminals)
        same_graph(g, ref, seed)
        assert g.k == len(terminals) and g.k0 == len(terminals)
        present = set(ref.arcs())
        for v in ref.nonterminals():
            kappa, sides, ess, paths = _core.tightest_cut(g, v)
            rk, rside, ress = ref_cut(ref, v)
            assert kappa == rk, (seed, v, kappa, rk)
            assert sides == rside, (seed, v)
            assert frozenset(terminals[j] for j in ess) == ress, (seed, v)
            assert sides[v] == "R" and sum(1 for s in sides.values() if s == "S") == kappa, (seed, v)
            assert all(sides[t] != "R" for t in ref.terminals), (seed, v)
            check_paths(present, ref.terminals, v, kappa, paths, seed)
            # [Def 3.3]: no arc from R to L
            for u, w in present:
                assert not (sides[u] == "R" and sides[w] == "L"), (seed, v, u, w)


# ---------------------------------------------------------------------------
# (2) Graph mutation semantics [Def 2.1] / §13.4 (no oracle, any out-degree)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", range(20))
def test_fuzz_graph_mutations_match_reference(chunk: int) -> None:
    for i in range(20):
        seed = 20_000 + chunk * 20 + i
        rng = random.Random(seed)
        n, arcs, terminals = random_raw_instance(rng)
        g = Graph(n, arcs, terminals)
        ref = DiGraphState(n, arcs, terminals)
        same_graph(g, ref, seed)
        for step in range(25):
            ops = []
            live_arcs = sorted(ref.arcs())
            if live_arcs:
                ops.append("delete")
            pts = ref.pre_terminals()
            if pts:
                ops += ["contract", "contract"]
            if ref.k >= 1:
                ops.append("remove_terminal")
            live = list(ref.vertices())
            if live:
                ops.append("remove_vertex")
            if not ops:
                break
            op = rng.choice(ops)
            where = (seed, step, op)
            if op == "delete":
                u, w = rng.choice(live_arcs)
                a = g.find_arc(u, w)
                assert a >= 0 and g.arc(a)[:3] == (u, w, True), where
                if rng.random() < 0.5:
                    g.delete_arc(u, w)
                else:
                    g.delete_arc_id(a)
                ref.delete_arc(u, w)
                assert not g.has_arc(u, w) and g.find_arc(u, w) == -1 and not g.arc(a)[2], where
            elif op == "contract":
                p = rng.choice(pts)
                t = rng.choice(ref.terminal_out_neighbors(p))
                v0 = g.version
                parent, created, deleted = g.contract_detailed(p, t)
                assert parent == ref.contract(p, t), where
                assert g.version == v0 + 1, where
                assert not g.live(p), where
                # every created arc is a redirected in-arc of p with orig_head == p; every deleted arc
                # was incident to p; ids are never reused
                for c in created:
                    tail, head, alive, oh = g.arc(c)
                    assert alive and head == t and oh == p and (tail, t) in ref.arcs(), where
                assert len(set(created) | set(deleted)) == len(created) + len(deleted), where
                for d in deleted:
                    tail, head, alive, _oh = g.arc(d)
                    assert not alive and p in (tail, head), where
            elif op == "remove_terminal":
                t = rng.choice(list(ref.terminals))
                g.remove_terminal(t)
                ref.remove_terminal(t)
                assert not g.is_terminal(t) and g.terminal_index(t) == terminals.index(t), where
            else:
                v = rng.choice(live)
                g.remove_vertex(v)
                ref.remove_vertex(v)
                assert not g.live(v), where
            same_graph(g, ref, where)


# ---------------------------------------------------------------------------
# (3) EssentialOracle after random operation sequences (exact_cuts=True), vs reference
# ---------------------------------------------------------------------------


def run_ops(n, arcs, terminals, seed: int, threads: int, steps: int, check_every: bool):
    rng = random.Random(seed)
    h = Harness(n, arcs, terminals, threads, seed)
    if check_every:
        h.compare(True, (seed, "init"))
        h.users_consistent((seed, "init"))
    transcript = []
    for step in range(steps):
        ops = []
        live_arcs = sorted(h.ref.arcs())
        if live_arcs:
            ops += ["delete", "delete", "delete_set"]
        deg1 = [p for p in h.g.pre_terminals() if h.g.out_degree(p) == 1]
        if deg1:
            ops += ["contract"] * 4
        if h.ref.k >= 1 and rng.random() < 0.5:  # down to k = 0 (kappa == 0 everywhere afterwards)
            ops.append("remove_terminal")
            if rng.random() < 0.3:
                ops.append("remove_terminal_as_vertex")  # generic vertex removal of a terminal
        nonterms = h.ref.nonterminals()
        if len(nonterms) >= 2 and rng.random() < 0.5:
            ops.append("remove_vertex")
        if check_every and live_arcs and rng.random() < 0.15:
            ops.append("oob_delete")  # mutation without its hook: the oracle must self-heal
        if check_every and nonterms and rng.random() < 0.3:
            ops.append("refresh_one")
        if not ops:
            break
        op = rng.choice(ops)
        where = (seed, step, op)
        if op == "delete":
            u, w = rng.choice(live_arcs)
            # prefer arcs used by some stored flow (the interesting O2 case), incl. arcs leaving v itself
            used = [(a, b) for (a, b) in live_arcs if h.o.users_of_arc(a, b)]
            if used and rng.random() < 0.8:
                u, w = rng.choice(used)
            if check_every:
                check_evaluate_deletion(h, [(u, w)], where, rng)
            else:
                h.o.evaluate_deletion([(u, w)], True)
            h.o.commit_deletion([(u, w)])
            h.ref.delete_arc(u, w)
        elif op == "delete_set":
            pts2 = [p for p in h.ref.pre_terminals() if h.ref.out_degree(p) >= 2]
            if pts2 and rng.random() < 0.7:
                p = rng.choice(pts2)
                outs = h.ref.out_neighbors(p)
                keep = rng.choice([x for x in outs if h.ref.is_terminal(x)])
                D = [(p, x) for x in outs if x != keep]  # O5: out(p) \ {(p, phi(p))}
            else:
                D = rng.sample(live_arcs, min(len(live_arcs), rng.randint(1, 3)))
            if check_every:
                check_evaluate_deletion(h, D, where, rng)
            else:
                h.o.evaluate_deletion(D, True)
            h.o.commit_deletion(D)
            for e in D:
                h.ref.delete_arc(*e)
        elif op == "contract":
            p = rng.choice(deg1)
            (t,) = h.g.out_neighbors(p)
            assert h.g.contract(p, t) == h.ref.contract(p, t), where
            h.o.after_contraction(p, t)
        elif op == "remove_terminal":
            t = rng.choice(list(h.ref.terminals))
            h.g.remove_terminal(t)
            h.ref.remove_terminal(t)
            h.o.after_terminal_removal(t)
        elif op == "remove_terminal_as_vertex":
            t = rng.choice(list(h.ref.terminals))
            h.g.remove_vertex(t)
            h.ref.remove_vertex(t)
            h.o.after_vertex_removal(t)
            assert not h.g.is_terminal(t) and t not in h.g.terminals, where
        elif op == "oob_delete":
            u, w = rng.choice(live_arcs)
            h.g.delete_arc(u, w)  # no evaluate/commit: the graph changed behind the oracle's back
            h.ref.delete_arc(u, w)
            h.compare(True, where)  # everything is recomputed from scratch (exact) on first access
        elif op == "refresh_one":
            v = rng.choice(nonterms)
            h.o.refresh_cut(v)
            rk, rside, ress = ref_cut(h.ref, v)
            assert h.o.cut_exact(v) and h.o.kappa(v) == rk, where
            assert h.ess_ids(h.o.ess(v)) == ress and h.o.sides(v) == rside, (where, v)
        else:
            v = rng.choice(nonterms)
            h.g.remove_vertex(v)
            h.ref.remove_vertex(v)
            h.o.after_vertex_removal(v)
        if check_every:
            same_graph(h.g, h.ref, where)
            h.compare(False, where)
            h.users_consistent(where)
            if step % 4 == 3:
                h.o.refresh_all_cuts()
                h.compare(True, (where, "refreshed"))
        transcript.append((op, h.snapshot()))
    h.o.refresh_all_cuts()
    if check_every:
        h.compare(True, (seed, "final"))
        h.users_consistent((seed, "final"))
    transcript.append(("final", h.snapshot()))
    return transcript, h


@pytest.mark.parametrize("chunk", range(60))
def test_fuzz_operation_sequences_track_reference(chunk: int) -> None:
    for i in range(10):
        seed = 30_000 + chunk * 10 + i
        rng = random.Random(seed)
        n, arcs, terminals = random_raw_instance(rng)
        run_ops(n, arcs, terminals, seed, threads=1 + (i % 3), steps=14, check_every=True)


# ---------------------------------------------------------------------------
# (3b) after_contraction for a pre-terminal of out-degree >= 2 (accepted by Graph::contract and by
#      the hook, which documents the case) must honour the ess()/cut_exact() contract:
#      "subset of the truth; exact if cut_exact".  O3 (no cut movement) only holds for d^+(p) = 1;
#      for d^+(p) >= 2 the contraction also deletes arcs, which can move tightest cuts of unaffected
#      vertices, so the hook must either advance the exact epoch or reject the call.
# ---------------------------------------------------------------------------


def test_after_contraction_out_degree_ge_2_keeps_ess_contract() -> None:
    # Deterministic witness (seed 37 of the fuzz below): contracting 7 into 8 makes terminal 8
    # essential for vertex 3 (5 loses its exits 7 -> 10 / 7 -> 12); the reference reports {8, 12}.
    n = 13
    arcs = [(0, 9), (0, 12), (2, 5), (2, 9), (3, 5), (3, 9), (3, 12), (5, 2), (5, 3), (5, 7), (5, 9), (5, 12), (6, 12),
            (7, 5), (7, 8), (7, 10), (7, 12), (8, 7), (9, 0), (9, 2), (9, 3), (9, 5), (9, 10), (9, 12), (10, 7), (10, 9),
            (11, 5), (11, 7), (11, 10), (11, 12), (12, 0), (12, 3), (12, 5), (12, 6), (12, 7), (12, 9)]
    T = [10, 11, 12, 8, 0]
    h = Harness(n, arcs, T, 1, 0)
    assert h.ref.out_degree(7) >= 2
    assert h.g.contract(7, 8) == h.ref.contract(7, 8)
    h.o.after_contraction(7, 8)
    rk, rside, ress = ref_cut(h.ref, 3)
    assert h.o.kappa(3) == rk
    assert h.ess_ids(h.o.ess(3)) <= ress
    if h.o.cut_exact(3):
        assert h.ess_ids(h.o.ess(3)) == ress, ("cut_exact claimed but Ess stale", h.ess_ids(h.o.ess(3)), ress)
        assert h.o.sides(3) == rside
    # and after an explicit refresh everything is exact again (the flows themselves are valid)
    h.o.refresh_all_cuts()
    h.compare(True, "refreshed")
    # fuzz: kappa equal, Ess subset, and exact whenever claimed exact
    for seed in range(60):
        rng = random.Random(95_000 + seed)
        n, arcs, T = random_raw_instance(rng)
        if not T:
            continue
        h = Harness(n, arcs, T, 1, 0)
        cands = [p for p in h.ref.pre_terminals() if h.ref.out_degree(p) >= 2]
        if not cands:
            continue
        p = rng.choice(cands)
        t = rng.choice(h.ref.terminal_out_neighbors(p))
        assert h.g.contract(p, t) == h.ref.contract(p, t), seed
        h.o.after_contraction(p, t)
        same_graph(h.g, h.ref, seed)
        h.compare(False, (seed, "contract", p, t))
        h.users_consistent((seed, "contract", p, t))
        h.o.refresh_all_cuts()
        h.compare(True, (seed, "refreshed"))


# ---------------------------------------------------------------------------
# (4) evaluate_deletion vs glref.critical for EVERY (arc, v, t) on fresh graphs, incl. after ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", range(30))
def test_fuzz_evaluate_deletion_every_arc_vs_critical(chunk: int) -> None:
    for i in range(8):
        seed = 40_000 + chunk * 8 + i
        rng = random.Random(seed)
        n, arcs, terminals = random_raw_instance(rng, n_max=12)
        h = Harness(n, arcs, terminals, threads=2, seed=seed)
        # a few warm-up mutations so that stored flows are warm-started, not fresh
        for _ in range(rng.randint(0, 4)):
            live = sorted(h.ref.arcs())
            if not live:
                break
            u, w = rng.choice(live)
            h.o.evaluate_deletion([(u, w)], rng.random() < 0.5)
            h.o.commit_deletion([(u, w)])
            h.ref.delete_arc(u, w)
        h.o.refresh_all_cuts()
        h.compare(True, (seed, "warm"))
        live = sorted(h.ref.arcs())
        for e in live[: min(len(live), 40)]:
            check_evaluate_deletion(h, [e], (seed, e), rng)


# ---------------------------------------------------------------------------
# (5) matching [Lem 7.8] and minimal Hall-deficient set [Lem 7.6]
# ---------------------------------------------------------------------------


def _check_matching(n, arcs, terminals, S, matched, where) -> None:
    present = set(arcs)
    tset = set(terminals)
    assert len(matched) == len(S), where
    assert len(set(matched)) == len(matched), where
    for t, p in zip(S, matched):
        assert 0 <= p < n and p not in tset and (p, t) in present, (where, t, p)


@pytest.mark.parametrize("chunk", range(20))
def test_fuzz_matching_and_hall_set(chunk: int) -> None:
    for i in range(30):
        seed = 50_000 + chunk * 30 + i
        rng = random.Random(seed)
        n, raw, terminals = random_raw_instance(rng)
        if not terminals:
            continue
        ref = DiGraphState(n, raw, terminals)
        arcs = ref.arcs()
        subsets = [list(terminals), []]
        for _ in range(3):
            subsets.append(rng.sample(terminals, rng.randint(0, len(terminals))))
        for S in subsets:
            r = ref_saturating_matching(ref, S)
            c = _core.saturating_matching(n, raw, terminals, S)
            assert (c is None) == (r is None), (seed, S)
            if c is not None:
                _check_matching(n, arcs, terminals, S, c, (seed, S))
        c_all = _core.saturating_matching(n, raw, terminals)
        assert (c_all is None) == (ref_saturating_matching(ref) is None), seed
        # Hall set
        hs = _core.minimal_hall_deficient_set(n, raw, terminals)
        hr = ref_minimal_hall_deficient_set(ref)
        assert (hs is None) == (hr is None), (seed, hs, hr)
        if hs is None:
            assert c_all is not None, seed
            continue
        S = list(hs)
        assert S and len(set(S)) == len(S) and set(S) <= set(terminals), (seed, S)
        assert ref_saturating_matching(ref, S) is None, (seed, S)
        assert len(ref.pre_terminals(S)) == len(S) - 1, (seed, S)
        # inclusion-minimal: every S \ {t} is saturable (hence every proper subset is)
        for t in S:
            assert ref_saturating_matching(ref, [x for x in S if x != t]) is not None, (seed, S, t)
        # The core runs the reference's deterministic scan (same terminal order, same saturability
        # answers, no singleton shortcut) and must return exactly the same set -- also when some
        # terminal has in-degree 0, where the paper's shortcut would pick a different minimal set.
        assert S == hr, (seed, S, hr)


# ---------------------------------------------------------------------------
# (6) min-cost flow [Prop 5.4]: reference agreement, feasibility, int64 overflow reporting
# ---------------------------------------------------------------------------


def _random_network(rng: random.Random):
    n = rng.randint(2, 9)
    big = rng.random() < 0.35
    arcs = []
    for _ in range(rng.randint(0, 22)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        cap = rng.randint(0, 5) if not big else rng.choice([0, 1, BIG, BIG + 7, rng.randint(1, BIG)])
        cost = rng.randint(0, 6) if not big else rng.choice([0, 1, BIG, rng.randint(0, BIG)])
        arcs.append((u, v, cap, cost))
    s, z = 0, n - 1
    if rng.random() < 0.2:
        s, z = rng.sample(range(n), 2)
    req = rng.randint(0, 12) if not big else rng.choice([1, BIG, 3 * BIG, rng.randint(0, 3 * BIG)])
    return n, arcs, s, z, req


def _check_flow(n, arcs, s, z, value, cost, flows) -> None:
    bal = [0] * n
    assert len(flows) == len(arcs)
    for (u, v, cap, _c), f in zip(arcs, flows):
        assert 0 <= f <= cap
        bal[u] -= f
        bal[v] += f
    assert bal[z] == value and bal[s] == -value
    assert all(bal[x] == 0 for x in range(n) if x not in (s, z))
    assert sum(f * c for (_u, _v, _cap, c), f in zip(arcs, flows)) == cost


@pytest.mark.parametrize("chunk", range(10))
def test_fuzz_min_cost_flow_matches_reference(chunk: int) -> None:
    for i in range(40):
        seed = 60_000 + chunk * 40 + i
        rng = random.Random(seed)
        n, arcs, s, z, req = _random_network(rng)
        ref = RefMinCostFlow(n)
        for a in arcs:
            ref.add_arc(*a)
        rv, rc = ref.min_cost_flow(s, z, req)  # Python ints: never overflow
        if rc < (1 << 63):
            value, cost, flows = _core.min_cost_flow(n, arcs, s, z, req)
            assert (value, cost) == (rv, rc), (seed, value, cost, rv, rc)
            _check_flow(n, arcs, s, z, value, cost, flows)
        else:
            with pytest.raises(OverflowError):
                _core.min_cost_flow(n, arcs, s, z, req)


def test_min_cost_flow_int64_overflow_is_reported() -> None:
    """A total cost above 2^63 - 1 must raise, never wrap silently."""
    c = 1 << 40
    with pytest.raises(OverflowError):
        _core.min_cost_flow(2, [(0, 1, 1 << 30, c)], 0, 1, 1 << 30)
    v, cost, _f = _core.min_cost_flow(2, [(0, 1, 1 << 22, c)], 0, 1, 1 << 22)
    assert (v, cost) == (1 << 22, (1 << 22) * c)


# ---------------------------------------------------------------------------
# (7) DAG solver [Alg 5]: both variants x all policies vs verifier and glref.dag
# ---------------------------------------------------------------------------

POLICY_NAMES = {0: "max_residual", 1: "round_robin", 2: "first"}


def _composition(total: int, k: int, rng: random.Random, zeros: bool) -> list[int]:
    if k == 1:
        return [total]
    if zeros and rng.random() < 0.5:
        nz = rng.randint(1, k)
        cuts = sorted(rng.randint(0, total) for _ in range(nz - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        parts += [0] * (k - nz)
        rng.shuffle(parts)
        return parts
    cuts = sorted(rng.randint(0, total) for _ in range(k - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def random_small_dag(rng: random.Random) -> Instance:
    """Random k-T-connected DAG [Lem 9.1]: every non-terminal has out-degree >= k (exactly k with
    probability 0.4, otherwise up to all later vertices so that a vertex is the in-neighbour of many
    part members: the 'pushed by several parents' case of P1).  Capacities may contain zeros; weights
    are unit, small, or up to 2^40."""
    n = rng.randint(3, 15)
    k = rng.randint(1, min(5, n - 1))
    perm = list(range(n))
    rng.shuffle(perm)
    nonterms, terminals = perm[: n - k], perm[n - k:]
    exact_k = rng.random() < 0.4
    arcs = []
    for i, u in enumerate(nonterms):
        cands = nonterms[i + 1:] + terminals
        d = k if exact_k else rng.randint(k, len(cands))
        arcs.extend((u, x) for x in rng.sample(cands, d))
    rng.shuffle(arcs)
    wmode = rng.choice(["unit", "small", "big"])
    if wmode == "unit":
        caps = _composition(n - k, k, rng, zeros=True)
        return make_instance(n, arcs, terminals, caps)
    w_max = 4 if wmode == "small" else BIG
    weights = [0] * n
    for u in nonterms:
        weights[u] = rng.randint(1, w_max)
    total = sum(weights)
    slack = rng.choice([0, 0, 1, rng.randint(0, w_max)])
    caps = _composition(total + slack, k, rng, zeros=True)
    return make_instance(n, arcs, terminals, caps, weights=weights)


def _parts_of(inst: Instance, assignment: list[int]) -> list[list[int]]:
    parts: list[list[int]] = [[] for _ in range(inst.k)]
    for v, i in enumerate(assignment):
        assert 0 <= i < inst.k, (v, i)
        parts[i].append(v)
    return parts


def _check_parents(inst: Instance, assignment: list[int], parent: list[int], where) -> None:
    present = set(inst.arcs)
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            assert parent[v] == -1, (where, v)
            continue
        p = parent[v]
        assert p >= 0 and (v, p) in present and assignment[p] == assignment[v], (where, v, p)
    for v in range(inst.n):
        x, steps = v, 0
        while x not in tset:
            x = parent[x]
            steps += 1
            assert steps <= inst.n, (where, "parent cycle")
        assert x == inst.terminals[assignment[v]], (where, v)


@pytest.mark.parametrize("chunk", range(30))
def test_fuzz_dag_partition_variants_policies_vs_reference(chunk: int) -> None:
    for i in range(30):
        seed = 70_000 + chunk * 30 + i
        rng = random.Random(seed)
        inst = random_small_dag(rng)
        args = (inst.n, list(inst.arcs), list(inst.terminals), list(inst.capacities),
                None if inst.weights is None else list(inst.weights))
        for policy in (0, 1, 2):
            ref = gl_dag_partition(inst, policy=POLICY_NAMES[policy])
            assert ref.status == "ok", (seed, ref.message)
            ref_parts = [sorted(p) for p in ref.parts]
            outs = []
            for variant in (0, 1):
                res = _core.dag_partition(*args, policy, variant)
                where = (seed, policy, variant)
                assert res["status"] == "ok", (where, res["message"])
                parts = _parts_of(inst, res["assignment"])
                report = verify_instance_parts(inst, parts)
                assert report.valid, (where, report.errors)
                _check_parents(inst, res["assignment"], res["parent"], where)
                assert [sorted(p) for p in parts] == ref_parts, where
                assert {v: p for v, p in enumerate(res["parent"]) if p >= 0} == ref.parents, where
                assert res["stats"]["contractions"] == inst.n - inst.k, where
                outs.append(res)
            assert outs[0]["assignment"] == outs[1]["assignment"], (seed, policy)
            assert outs[0]["parent"] == outs[1]["parent"], (seed, policy)
            # the [Lem 9.1] check is pure validation: skipping it changes nothing on a valid instance
            res_nc = _core.dag_partition(*args, policy, 1, check_precondition=False)
            assert res_nc["assignment"] == outs[1]["assignment"] and res_nc["parent"] == outs[1]["parent"], (seed, policy)
            # capacity-0 terminals never receive a vertex
            for j, c in enumerate(inst.capacities):
                if c == 0:
                    assert sorted(_parts_of(inst, outs[0]["assignment"])[j]) == [inst.terminals[j]], (seed, policy, j)


def test_dag_partition_degenerate_shapes() -> None:
    """k == n (no non-terminal at all), n == 1, and a single terminal with every vertex forced into it."""
    for terminals in ([0, 1, 2], [2, 0, 1], [0]):
        n = len(terminals)
        for variant in (0, 1):
            res = _core.dag_partition(n, [], terminals, [0] * n, None, 0, variant)
            assert res["status"] == "ok", res["message"]
            assert res["assignment"] == [terminals.index(v) for v in range(n)]
            assert res["parent"] == [-1] * n
            assert res["stats"]["contractions"] == 0
    inst = make_instance(6, [(1, 0), (2, 1), (3, 1), (4, 2), (5, 4), (5, 3)], [0], [5])
    ref = gl_dag_partition(inst, policy="first")
    for variant in (0, 1):
        res = _core.dag_partition(6, list(inst.arcs), [0], [5], None, 2, variant)
        assert res["status"] == "ok" and res["assignment"] == [0] * 6
        assert {v: p for v, p in enumerate(res["parent"]) if p >= 0} == ref.parents
        _check_parents(inst, res["assignment"], res["parent"], variant)
    # weighted, one terminal, weights 2^40 with slack: bound c + w_max - 1 holds trivially, parts exact
    w = [0, BIG, BIG - 1, 7, BIG, 1]
    inst = make_instance(6, [(1, 0), (2, 1), (3, 1), (4, 2), (5, 4), (5, 3)], [0], [sum(w)], weights=w)
    for variant in (0, 1):
        res = _core.dag_partition(6, list(inst.arcs), [0], [sum(w)], w, 0, variant)
        assert res["status"] == "ok" and res["assignment"] == [0] * 6
        assert verify_instance_parts(inst, _parts_of(inst, res["assignment"])).valid


@pytest.mark.parametrize("seed", range(120))
def test_fuzz_dag_precondition_out_degree_below_k(seed: int) -> None:
    """[Lem 9.1]: removing arcs of one non-terminal below out-degree k is reported (same vertex as the
    reference) by both variants; a cycle is reported with the same vertex."""
    rng = random.Random(80_000 + seed)
    inst = random_small_dag(rng)
    tset = set(inst.terminals)
    bad = rng.choice([v for v in range(inst.n) if v not in tset])
    out = inst.out_adjacency()
    keep = out[bad][: inst.k - 1]
    arcs = [a for a in inst.arcs if a[0] != bad] + [(bad, x) for x in keep]
    inst2 = make_instance(inst.n, arcs, inst.terminals, inst.capacities, weights=inst.weights)
    ref = gl_dag_partition(inst2, policy="first")
    assert ref.status == "precondition_failed" and f"non-terminal {bad} " in ref.message, seed
    for variant in (0, 1):
        res = _core.dag_partition(inst2.n, list(inst2.arcs), list(inst2.terminals), list(inst2.capacities),
                                  None if inst2.weights is None else list(inst2.weights), 0, variant)
        assert res["status"] == "precondition_failed", (seed, variant)
        assert f"non-terminal {bad} " in res["message"], (seed, variant, res["message"])
        assert res["assignment"] == [] and res["parent"] == [], (seed, variant)
    # duplicate arcs / self-loops in the raw list must not inflate the out-degree
    raw = list(inst2.arcs) + [(bad, keep[0])] * 3 + [(bad, bad)] if keep else list(inst2.arcs) + [(bad, bad)]
    for variant in (0, 1):
        res = _core.dag_partition(inst2.n, raw, list(inst2.terminals), list(inst2.capacities),
                                  None if inst2.weights is None else list(inst2.weights), 0, variant)
        assert res["status"] == "precondition_failed" and f"non-terminal {bad} " in res["message"], (seed, variant)
    # a 2-cycle among non-terminals (if there is an arc between two non-terminals)
    inner = [(u, v) for u, v in inst.arcs if v not in tset]
    if inner:
        u, v = rng.choice(inner)
        cyc = list(inst.arcs) + [(v, u)]
        ref = gl_dag_partition(make_instance(inst.n, cyc, inst.terminals, inst.capacities, weights=inst.weights), policy="first")
        assert ref.status == "precondition_failed" and "cycle" in ref.message, seed
        for variant in (0, 1):
            res = _core.dag_partition(inst.n, cyc, list(inst.terminals), list(inst.capacities),
                                      None if inst.weights is None else list(inst.weights), 0, variant)
            assert res["status"] == "precondition_failed" and "cycle" in res["message"], (seed, variant)
            assert res["message"] == ref.message, (seed, variant, res["message"], ref.message)


# ---------------------------------------------------------------------------
# (8) thread safety / determinism: threads=8 repeatedly vs threads=1
# ---------------------------------------------------------------------------


def _medium_raw_instance(rng: random.Random):
    n = rng.randint(36, 56)
    k = rng.randint(3, 7)
    terminals = rng.sample(range(n), k)
    tset = set(terminals)
    p = rng.choice([0.12, 0.2, 0.3])
    arcs = [(u, v) for u in range(n) if u not in tset for v in range(n) if u != v and rng.random() < p]
    return n, arcs, terminals


@pytest.mark.parametrize("seed", range(4))
def test_threads_8_compute_all_repeated_is_deterministic(seed: int) -> None:
    rng = random.Random(90_000 + seed)
    n, arcs, terminals = _medium_raw_instance(rng)
    base = Harness(n, arcs, terminals, threads=1, seed=seed)
    base.compare(True, (seed, "threads=1"))
    snap1 = base.snapshot()
    for rep in range(12):
        h = Harness(n, arcs, terminals, threads=8, seed=seed)
        assert h.snapshot() == snap1, (seed, rep)
        for _ in range(3):
            h.o.compute_all()
            assert h.snapshot() == snap1, (seed, rep)


@pytest.mark.parametrize("seed", range(4))
def test_threads_8_operation_sequence_is_deterministic(seed: int) -> None:
    rng = random.Random(91_000 + seed)
    n, arcs, terminals = _medium_raw_instance(rng)
    t1, h1 = run_ops(n, arcs, terminals, seed, threads=1, steps=18, check_every=False)
    h1.compare(True, (seed, "threads=1 final"))
    for rep in range(4):
        t8, _h8 = run_ops(n, arcs, terminals, seed, threads=8, steps=18, check_every=False)
        assert t8 == t1, (seed, rep)


def test_threads_more_than_vertices_is_deterministic() -> None:
    """threads=64 on tiny instances (workers capped by the work count) equals threads=1."""
    for seed in range(40):
        rng = random.Random(93_000 + seed)
        n, arcs, terminals = random_raw_instance(rng)
        h1 = Harness(n, arcs, terminals, threads=1, seed=seed)
        h64 = Harness(n, arcs, terminals, threads=64, seed=seed)
        assert h64.snapshot() == h1.snapshot(), seed
        live = sorted(h1.ref.arcs())
        if live:
            D = rng.sample(live, min(len(live), 2))
            r1 = h1.o.evaluate_deletion(D, True)
            r64 = h64.o.evaluate_deletion(D, True)
            assert sorted((v, kv[0], tuple(kv[1])) for v, kv in r1.items()) == sorted(
                (v, kv[0], tuple(kv[1])) for v, kv in r64.items()), seed
            h1.o.commit_deletion(D)
            h64.o.commit_deletion(D)
            h1.o.refresh_all_cuts()
            h64.o.refresh_all_cuts()
            assert h64.snapshot() == h1.snapshot(), seed


def test_threads_8_hooks_repeated() -> None:
    """after_terminal_removal / evaluate_deletion / refresh_all_cuts with 8 workers, many times, all
    bit-identical to the single-threaded oracle and to the reference."""
    rng = random.Random(92_000)
    n, arcs, terminals = _medium_raw_instance(rng)
    hs = [Harness(n, arcs, terminals, threads=th, seed=0) for th in (1, 8, 8, 8)]
    for step in range(10):
        live = sorted(hs[0].ref.arcs())
        D = rng.sample(live, min(len(live), 3))
        results = []
        for h in hs:
            res = h.o.evaluate_deletion(D, step % 2 == 0)
            results.append(sorted((v, kv[0], tuple(kv[1])) for v, kv in res.items()))
            h.o.commit_deletion(D)
            for e in D:
                h.ref.delete_arc(*e)
        assert all(r == results[0] for r in results), step
        if step % 3 == 2 and hs[0].ref.k >= 2:
            t = rng.choice(list(hs[0].ref.terminals))
            for h in hs:
                h.g.remove_terminal(t)
                h.ref.remove_terminal(t)
                h.o.after_terminal_removal(t)
        snaps = [h.snapshot() for h in hs]
        assert all(s == snaps[0] for s in snaps), step
        for h in hs:
            h.o.refresh_all_cuts()
        snaps = [h.snapshot() for h in hs]
        assert all(s == snaps[0] for s in snaps), step
    hs[0].compare(True, "final")
