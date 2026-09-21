"""Tests for the C++ matching / Hall-set / min-cost-flow primitives of ``glsolver._core``:
``saturating_matching`` [Lem 7.8], ``minimal_hall_deficient_set`` [Lem 7.6] and
``min_cost_flow`` [Prop 5.4] (docs/paper_notes.md §5, §7.2, §8).

Every core answer is checked against the pure-Python reference ``glref`` and, where
possible, against an independent brute force:

* matching: saturation agrees with ``glref.matching.saturating_matching`` on random
  digraphs and random terminal subsets; a returned matching is a valid matching
  (distinct pre-terminals, ``(p, t)`` an arc);
* minimal Hall-deficient set: ``None`` iff the reference finds ``T`` saturable;
  otherwise ``|PT(G,S)| = |S| - 1`` and every proper subset of ``S`` is saturable
  (checked with the reference matching on all proper subsets);
* min-cost flow: ``(value, cost)`` equals ``glref.mincostflow`` on random small
  networks, the returned flows are feasible with that value and cost, and on tiny
  split-assignment networks the cost equals a brute-force minimum over all
  split-assignments [Def 5.2].
"""
from __future__ import annotations

import itertools
import random

import pytest

from glref.graph import DiGraphState
from glref.matching import minimal_hall_deficient_set as ref_minimal_hall_deficient_set
from glref.matching import saturating_matching as ref_saturating_matching
from glref.mincostflow import MinCostFlow as RefMinCostFlow
from glsolver import _core
from glsolver.instance import Instance, make_instance
from helpers_small_graphs import random_digraph, random_kT_connected_digraph

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def core_matching(inst: Instance, S: list[int] | None = None):
    return _core.saturating_matching(inst.n, list(inst.arcs), list(inst.terminals), S)


def core_hall(inst: Instance):
    return _core.minimal_hall_deficient_set(inst.n, list(inst.arcs), list(inst.terminals))


def check_matching(inst: Instance, S: list[int], matched: list[int]) -> None:
    arcs = set(inst.arcs)
    tset = set(inst.terminals)
    assert len(matched) == len(S)
    assert len(set(matched)) == len(matched), "pre-terminals must be distinct"
    for t, p in zip(S, matched):
        assert p not in tset and 0 <= p < inst.n
        assert (p, t) in arcs, f"({p},{t}) is not an arc"


def sparse_instance(seed: int) -> Instance:
    """Random digraphs sparse enough that T is often *not* saturable."""
    rng = random.Random(seed)
    k = rng.randint(1, 7)
    n = rng.randint(k + 1, 18)
    p = rng.choice([0.05, 0.1, 0.2, 0.4, 0.7])
    return random_digraph(n, k, rng, p=p, mode="random")


# ---------------------------------------------------------------------------
# [Lem 7.8] saturating matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(150))
def test_saturating_matching_agrees_with_reference(seed: int) -> None:
    inst = sparse_instance(seed)
    g = DiGraphState.from_instance(inst)
    rng = random.Random(1000 + seed)
    subsets: list[list[int]] = [list(inst.terminals)]
    for _ in range(4):
        size = rng.randint(0, inst.k)
        subsets.append(rng.sample(list(inst.terminals), size))
    for S in subsets:
        ref = ref_saturating_matching(g, S)
        core = core_matching(inst, S if S != list(inst.terminals) else None)
        assert (core is None) == (ref is None), (seed, S)
        if core is not None:
            check_matching(inst, S, core)
            assert len(core) == len(ref)
    # default argument = all terminals
    assert (core_matching(inst) is None) == (ref_saturating_matching(g) is None)


def test_saturating_matching_kT_connected_always_exists() -> None:
    """[Lem 7.8]: under k-T-connectivity with c_t > 0 a saturating matching of T exists."""
    for seed in range(15):
        rng = random.Random(seed)
        k = rng.randint(1, 4)
        n = rng.randint(k + 1, 14)
        inst = random_kT_connected_digraph(n, k, rng)
        core = core_matching(inst)
        assert core is not None
        check_matching(inst, list(inst.terminals), core)


def test_saturating_matching_hand_examples() -> None:
    # two terminals sharing a single pre-terminal: not saturable
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 0])
    assert core_matching(inst) is None
    assert core_matching(inst, [0]) == [2] and core_matching(inst, [1]) == [2]
    assert core_matching(inst, []) == []
    # crossing structure with a unique perfect matching: 3->1 forced, then 2->0
    inst = make_instance(4, [(2, 0), (2, 1), (3, 1)], [0, 1], [1, 1])
    assert core_matching(inst) == [2, 3]
    # a non-terminal in S is rejected
    with pytest.raises(ValueError):
        core_matching(inst, [2])


def test_saturating_matching_large_bipartite() -> None:
    """Many terminals (deep augmenting paths; the DFS is iterative) against networkx."""
    import networkx as nx

    rng = random.Random(77)
    k, extra = 400, 500
    n = k + extra
    terminals = list(range(extra, n))
    arcs = []
    for p in range(extra):
        for t in rng.sample(terminals, 2):
            arcs.append((p, t))
    caps = [1] * k
    caps[0] = extra - (k - 1)
    inst = make_instance(n, arcs, terminals, caps)
    core = core_matching(inst)
    B = nx.Graph()
    B.add_nodes_from(terminals, bipartite=0)
    B.add_edges_from((t, ("p", p)) for p, t in inst.arcs)
    m = nx.bipartite.hopcroft_karp_matching(B, top_nodes=terminals)
    size = sum(1 for t in terminals if t in m)
    assert (core is not None) == (size == k)
    if core is not None:
        check_matching(inst, terminals, core)


# ---------------------------------------------------------------------------
# [Lem 7.6] minimal Hall-deficient set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(150))
def test_minimal_hall_deficient_set_properties(seed: int) -> None:
    inst = sparse_instance(seed)
    g = DiGraphState.from_instance(inst)
    ref = ref_minimal_hall_deficient_set(g)
    core = core_hall(inst)
    assert (core is None) == (ref is None), (seed, core, ref)
    if core is None:
        assert core_matching(inst) is not None
        return
    S = list(core)
    assert S and len(set(S)) == len(S) and set(S) <= set(inst.terminals)
    assert ref_saturating_matching(g, S) is None, "S must be unsaturable"
    assert len(g.pre_terminals(S)) == len(S) - 1, "[Lem 7.6] |PT(G,S)| = |S| - 1"
    for size in range(len(S)):
        for sub in itertools.combinations(S, size):
            assert ref_saturating_matching(g, list(sub)) is not None, (S, sub)


def test_minimal_hall_deficient_set_shortcut_and_hand_examples() -> None:
    # terminal 1 has no pre-terminal: singleton deficient set
    inst = make_instance(4, [(2, 0), (3, 0)], [0, 1], [2, 0])
    assert core_hall(inst) == [1]
    # {0,1} share the only pre-terminal 2; terminal 3 has its own pre-terminal 4
    inst = make_instance(5, [(2, 0), (2, 1), (4, 3)], [0, 1, 3], [1, 0, 1])
    S = core_hall(inst)
    assert sorted(S) == [0, 1]
    # saturable -> None
    inst = make_instance(4, [(2, 0), (3, 1), (2, 1)], [0, 1], [1, 1])
    assert core_hall(inst) is None


# ---------------------------------------------------------------------------
# [Prop 5.4] min-cost flow
# ---------------------------------------------------------------------------


def random_network(seed: int):
    rng = random.Random(seed)
    n = rng.randint(2, 10)
    m = rng.randint(0, 25)
    arcs = []
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        cap = rng.randint(0, 6) * (1 if rng.random() < 0.8 else 10**12)
        arcs.append((u, v, cap, rng.randint(0, 7)))
    return n, arcs, 0, n - 1, rng.randint(0, 15) * (1 if rng.random() < 0.9 else 10**12)


def check_flow(n, arcs, s, z, value, cost, flows) -> None:
    bal = [0] * n
    assert len(flows) == len(arcs)
    for (u, v, cap, _c), f in zip(arcs, flows):
        assert 0 <= f <= cap
        bal[u] -= f
        bal[v] += f
    assert bal[z] == value and bal[s] == -value
    assert all(bal[x] == 0 for x in range(n) if x not in (s, z))
    assert sum(f * c for (_u, _v, _cap, c), f in zip(arcs, flows)) == cost


@pytest.mark.parametrize("seed", range(300))
def test_min_cost_flow_agrees_with_reference(seed: int) -> None:
    n, arcs, s, z, req = random_network(seed)
    ref = RefMinCostFlow(n)
    for a in arcs:
        ref.add_arc(*a)
    rv, rc = ref.min_cost_flow(s, z, req)
    value, cost, flows = _core.min_cost_flow(n, arcs, s, z, req)
    assert (value, cost) == (rv, rc), seed
    check_flow(n, arcs, s, z, value, cost, flows)


def brute_force_split_assignment(w: list[int], caps: list[int], allowed: list[list[int]], xi) -> int | None:
    """Minimum Σ ψ(v,t) ξ_v(t) over all split-assignments ψ [Def 5.2] with ψ(v,t) > 0 only for
    allowed pairs and Σ_v ψ(v,t) <= c_t [Def 5.3]; None if none exists."""
    k = len(caps)

    def compositions(total: int, slots: list[int]):
        if not slots:
            if total == 0:
                yield {}
            return
        t = slots[0]
        for x in range(total + 1):
            for rest in compositions(total - x, slots[1:]):
                d = dict(rest)
                if x:
                    d[t] = x
                yield d

    best = None
    per_vertex = [list(compositions(w[v], allowed[v])) for v in range(len(w))]
    for choice in itertools.product(*per_vertex):
        load = [0] * k
        cost = 0
        for v, psi in enumerate(choice):
            for t, x in psi.items():
                load[t] += x
                cost += x * xi(v, t)
        if all(load[t] <= caps[t] for t in range(k)):
            best = cost if best is None else min(best, cost)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_min_cost_flow_split_assignment_brute_force(seed: int) -> None:
    """[Prop 5.4] on tiny instances: the min-cost flow of value W equals the brute-force minimum
    over all split-assignments, and is infeasible exactly when no split-assignment exists."""
    rng = random.Random(seed)
    nv, k = rng.randint(1, 3), rng.randint(1, 3)
    w = [rng.randint(1, 3) for _ in range(nv)]
    allowed = [sorted(rng.sample(range(k), rng.randint(1, k))) for _ in range(nv)]
    caps = [rng.randint(0, 4) for _ in range(k)]
    costs = {(v, t): rng.randint(0, 3) for v in range(nv) for t in allowed[v]}
    xi = lambda v, t: costs[(v, t)]  # noqa: E731
    W = sum(w)
    s, z = nv + k, nv + k + 1
    arcs = []
    for v in range(nv):
        arcs.append((s, v, w[v], 0))
        for t in allowed[v]:
            arcs.append((v, nv + t, w[v], xi(v, t)))
    for t in range(k):
        arcs.append((nv + t, z, caps[t], 0))
    value, cost, flows = _core.min_cost_flow(z + 1, arcs, s, z, W)
    best = brute_force_split_assignment(w, caps, allowed, xi)
    if best is None:
        assert value < W
    else:
        assert value == W and cost == best
        check_flow(z + 1, arcs, s, z, value, cost, flows)


def test_min_cost_flow_int64_and_errors() -> None:
    big = 3 * 10**12
    value, cost, flows = _core.min_cost_flow(3, [(0, 1, big, 5), (1, 2, big, 7), (0, 2, 10, 100)], 0, 2, 2 * big)
    assert value == big + 10 and cost == 12 * big + 1000
    assert flows == [big, big, 10]
    with pytest.raises(ValueError):
        _core.min_cost_flow(2, [(0, 1, -1, 0)], 0, 1, 1)
    with pytest.raises(ValueError):
        _core.min_cost_flow(2, [(0, 1, 1, -1)], 0, 1, 1)
    with pytest.raises(ValueError):
        _core.min_cost_flow(2, [], 0, 0, 1)
    assert _core.min_cost_flow(2, [], 0, 1, 5) == (0, 0, [])
