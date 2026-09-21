"""Tests for compact/local connectivity (glref.compact) and the authors' official
counterexample (glref.counterexample); paper_notes §10, [Def A.2], [Def A.5],
[Lem A.8], [Lem A.9], [Lem A.11], [Lem A.12], [Lem A.13]."""
from __future__ import annotations

import random

import networkx as nx
import pytest

from glref.assignment import find_witness, is_witness
from glref.compact import (
    CompactInstance,
    compact_connected,
    compact_sets,
    compact_witness,
    contract,
    delete_edge,
    is_locally_connected,
    local_set,
    satisfies_compact_condition,
    satisfies_local_condition,
)
from glref.counterexample import (
    K,
    build_counterexample_compact,
    build_counterexample_instance,
    check_counterexample_claims,
    counterexample_structure,
)
from glref.essential import all_essential
from glref.graph import DiGraphState
from glsolver.instance import make_instance

# ---------------------------------------------------------------------------
# CompactInstance basics
# ---------------------------------------------------------------------------


def test_compact_instance_validation_and_conversion() -> None:
    with pytest.raises(ValueError):
        CompactInstance(1, [[1], [0]], [1, 1], [1])  # terminal with an out-arc
    with pytest.raises(ValueError):
        CompactInstance(1, [[], [0], [1]], [1, 3, 1], [4])  # mult > 1 with in-degree > 0
    ci = CompactInstance(2, [[], [], [0, 1], [2, 0]], [1, 1, 1, 2], [2, 1])
    assert list(ci.terminals()) == [0, 1] and list(ci.non_terminals()) == [2, 3]
    assert ci.total_mult() == 3 == ci.total_cap()
    assert ci.in_adj[2] == [3] and ci.pre_terminals() == [2, 3]
    inst = ci.to_instance()
    assert inst.n == 5 and inst.k == 2 and inst.capacities == (2, 1)
    assert set(inst.arcs) == {(2, 0), (2, 1), (3, 2), (3, 0), (4, 2), (4, 0)}
    assert inst.meta["compact_vertices"][3] == [3, 4]
    # round trip for mult = 1
    inst1 = make_instance(5, [(0, 2), (2, 3), (3, 4), (4, 1), (0, 1)], [1, 4], [2, 1], directed=True)
    ci1, new_id = CompactInstance.from_instance(inst1)
    assert new_id[1] == 0 and new_id[4] == 1 and ci1.k == 2 and ci1.n == 5
    assert sorted(ci1.out_adj[new_id[0]]) == sorted([new_id[2], new_id[1]])
    assert ci1.cap == [2, 1]


# ---------------------------------------------------------------------------
# compact / local connectivity on small hand-made instances
# ---------------------------------------------------------------------------


def test_compact_and_local_on_tiny_instances() -> None:
    # k = 2: v -> t0, v -> t1 (pre-terminal of both); u -> v only.
    ci = CompactInstance(2, [[], [], [0, 1], [2]], [1, 1, 1, 1], [1, 1])
    assert compact_connected(ci, 2, 0) and compact_connected(ci, 2, 1)
    # u has one out-arc: only one path family of size 1 < k = 2 -> not compact-connected
    assert not compact_connected(ci, 3, 0) and not compact_connected(ci, 3, 1)
    assert compact_sets(ci) == {2: frozenset({0, 1}), 3: frozenset()}
    assert not satisfies_compact_condition(ci)
    assert compact_witness(ci) is None
    # locally connected to a single terminal needs one path only
    assert is_locally_connected(ci, 3, [0]) and is_locally_connected(ci, 3, [1])
    assert not is_locally_connected(ci, 3, [0, 1])
    assert local_set(ci, [0, 1]) == [2]
    assert not satisfies_local_condition(ci)  # |L({0,1})| = 1 < 2
    # give u a second route: u -> t1 directly. Now u has two internally disjoint paths.
    ci2 = CompactInstance(2, [[], [], [0, 1], [2, 1]], [1, 1, 1, 1], [1, 1])
    assert compact_connected(ci2, 3, 1)  # both paths may end at t1
    assert compact_connected(ci2, 3, 0)  # u->v->t0 and u->t1
    assert satisfies_compact_condition(ci2)
    assert satisfies_local_condition(ci2)
    w = compact_witness(ci2)
    assert w is not None and sum(w.values()) == 2
    # capacities not summing to the number of non-terminals -> False (as in the authors' script)
    ci3 = CompactInstance(2, [[], [], [0, 1], [2, 1]], [1, 1, 1, 1], [2, 1])
    assert not satisfies_compact_condition(ci3) and not satisfies_local_condition(ci3)
    with pytest.raises(ValueError):
        satisfies_local_condition(ci2, max_k=1)


def test_mutations_match_authors_semantics() -> None:
    ci = CompactInstance(2, [[], [], [0, 1], [2, 1], [2, 3]], [1, 1, 1, 1, 1], [2, 1])
    d = delete_edge(ci, 3, 1)
    assert d.out_adj[3] == [2] and ci.out_adj[3] == [2, 1]
    # contract 2 into 0: in-arcs of 2 redirected to 0 (merged), ids above 2 shift down
    c, new_id = contract(ci, 2, 0)
    assert new_id == {0: 0, 1: 1, 3: 2, 4: 3}
    assert c.n == 4 and c.cap == [1, 1]
    assert c.out_adj[2] == [0, 1]  # old 3: arcs to 2 (->0) and 1
    assert c.out_adj[3] == [0, 2]  # old 4: arcs to 2 (->0) and 3 (->2)
    with pytest.raises(ValueError):
        contract(ci, 3, 0)  # (3, 0) is not an arc


def test_hierarchy_on_random_small_instances() -> None:
    """[Lem A.8] k-T-connected ⇒ compact ⇒ local, and compact ⇒ FEAC [Lem A.12],
    on random k-connected undirected graphs with tight capacities."""
    rng = random.Random(41)
    checked = 0
    for _ in range(25):
        n = rng.randint(4, 8)
        k = rng.randint(1, 3)
        G = None
        for _attempt in range(200):
            H = nx.gnp_random_graph(n, rng.uniform(0.4, 0.9), seed=rng.randrange(1 << 30))
            if nx.is_connected(H) and nx.node_connectivity(H) >= k:
                G = H
                break
        if G is None:
            continue
        terminals = rng.sample(range(n), k)
        cuts = sorted(rng.randint(0, n - k) for _ in range(k - 1))
        caps = [b - a for a, b in zip([0] + cuts, cuts + [n - k])]
        inst = make_instance(n, list(G.edges()), terminals, caps, directed=False)
        ci, new_id = CompactInstance.from_instance(inst)
        sets = compact_sets(ci)
        # k-T-connected: every non-terminal is compact-connected to every terminal
        assert all(sets[v] == frozenset(range(k)) for v in ci.non_terminals())
        assert satisfies_compact_condition(ci, sets)
        assert satisfies_local_condition(ci)
        # compact ⇒ FEAC: compact-connected terminals are essential [Lem A.11]
        g = DiGraphState.from_instance(inst)
        _kappa, ess = all_essential(g)
        inv = {c: o for o, c in new_id.items()}
        for v in ci.non_terminals():
            assert {inv[t] for t in sets[v]} <= ess[inv[v]]
        phi = find_witness(g, ess, dict(zip(inst.terminals, inst.capacities)))
        assert phi is not None
        checked += 1
    assert checked >= 15


# ---------------------------------------------------------------------------
# the official counterexample
# ---------------------------------------------------------------------------


def _pre_terminal_arcs(ci: CompactInstance) -> list[tuple[int, int]]:
    return [(p, t) for p in ci.pre_terminals() for t in ci.out_adj[p]]


def test_counterexample_structure_counts() -> None:
    for copies in (1, 17):
        ci, names = build_counterexample_compact(copies)
        assert ci.k == K == 9 and ci.n == 9 + 108 + 216 and len(names) == ci.n
        pre = ci.pre_terminals()
        assert len(pre) == 108 and all(len(ci.out_adj[p]) == 2 for p in pre)
        forcing = [v for v in ci.non_terminals() if v not in set(pre)]
        assert len(forcing) == 216 and all(len(ci.out_adj[v]) == 9 for v in forcing)
        assert all(ci.mult[v] == copies and not ci.in_adj[v] for v in forcing)
        assert len(_pre_terminal_arcs(ci)) == 216 and ci.num_arcs() == 216 + 216 * 9
        assert ci.total_cap() == ci.total_mult() == 108 + 216 * copies
        assert names[9] == "p^1_{1,2}" and names[117] == "v(p^1_{1,2}->t1)"
        # capacities: p^1, p^3 -> t_j, p^2 -> t_i, forcing vertices -> t_a
        st = counterexample_structure(copies)
        cap = [0] * 9
        for p, info in st["pre_terminals"].items():
            i, j = info["pair"]
            assert info["assigned"] == (i if info["r"] == 2 else j)
            assert sorted(ci.out_adj[p]) == [i, j]
            cap[info["assigned"]] += 1
        for f in st["forcing"]:
            x, y = f["x"], f["y"]
            p, ty = f["edge"]
            assert ty == y and sorted(ci.out_adj[p]) == sorted([x, y])
            assert (f["a"], f["b"], f["c"], f["d"]) == tuple(i for i in range(9) if i not in (x, y))[:4]
            targets = set(f["targets"])
            expect = set(st["pair_to_pre_terminals"][tuple(sorted((f["a"], f["b"])))])
            expect |= set(st["pair_to_pre_terminals"][tuple(sorted((f["a"], f["c"])))])
            expect |= set(st["pair_to_pre_terminals"][tuple(sorted((f["d"], x)))][:2])
            expect.add(p)
            assert targets == expect and len(targets) == 9
            assert f["assigned"] == f["a"]
            cap[f["a"]] += copies
        assert cap == ci.cap == st["capacities"]
        assert len(st["pair_to_pre_terminals"]) == 36
        assert len(st["forcing"]) == 216

    inst = build_counterexample_instance(17)
    assert inst.n == 9 + 108 + 3672 and inst.k == 9 and inst.directed and inst.weights is None
    assert inst.m == 216 + 3672 * 9
    out = inst.out_adjacency()
    pre_arcs = [(u, v) for (u, v) in inst.arcs if v < 9 and len(out[u]) == 2]
    assert len(pre_arcs) == 216
    forcing = [v for v in range(9, inst.n) if len(out[v]) == 9]
    assert len(forcing) == 3672
    assert sum(inst.capacities) == inst.n - inst.k
    assert inst.meta["structure"]["copies"] == 17
    assert inst.meta["compact_vertices"][117] == list(range(117, 134))
    inst.validate()


def test_counterexample_copies1_compact_and_feac_hold() -> None:
    """[Lem A.13] instance (one copy): compact connectivity holds, hence FEAC
    holds [Lem A.12] — a witness is found from the essential sets."""
    ci, _ = build_counterexample_compact(1)
    sets = compact_sets(ci)
    assert satisfies_compact_condition(ci, sets)
    assert all(sets[v] for v in ci.non_terminals())
    inst = build_counterexample_instance(1)
    assert inst.n == ci.n  # ids coincide for copies = 1
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    pre = set(ci.pre_terminals())
    assert all(kappa[v] == 2 for v in pre)  # two arcs to distinct terminals
    assert all(kappa[v] >= 1 and len(ess[v]) >= 1 for v in ci.non_terminals())
    for v in ci.non_terminals():  # [Lem A.11] compact-connected ⇒ essential
        assert sets[v] <= ess[v], (v, sorted(sets[v]), sorted(ess[v]))
    cap = dict(zip(inst.terminals, inst.capacities))
    phi = find_witness(g, ess, cap)
    assert phi is not None
    ok, why = is_witness(g, phi, ess, cap)
    assert ok, why


def test_counterexample_sampled_deletions_break_compact_connectivity() -> None:
    ci, _ = build_counterexample_compact(1)
    edges = [(u, v) for u in ci.non_terminals() for v in ci.out_adj[u]]
    assert len(edges) == 2160
    rng = random.Random(0)
    sample = rng.sample(edges, 14) + rng.sample(_pre_terminal_arcs(ci), 6)
    for u, v in sample:
        d = delete_edge(ci, u, v)
        assert not satisfies_compact_condition(d, compact_sets(d)), (u, v)


def test_counterexample_sampled_contractions_break_compact_connectivity() -> None:
    ci, _ = build_counterexample_compact(17)
    pairs = _pre_terminal_arcs(ci)
    rng = random.Random(1)
    for p, t in rng.sample(pairs, 8):
        smaller, _new_id = contract(ci, p, t)
        assert smaller.total_cap() == smaller.total_mult()
        assert not satisfies_compact_condition(smaller, compact_sets(smaller)), (p, t)


def test_check_claims_small_copies_deletions_only() -> None:
    r = check_counterexample_claims(copies=1, skip_contractions=True)
    assert r["compact_holds"]
    assert r["deletions_break"] == r["deletions_total"] == 2160
    assert r["contractions_total"] == 0
    assert r["preserving_operations"] == []


@pytest.mark.slow
def test_full_counterexample_replication_copies17() -> None:
    """Replicates the authors' ``python counterexample/counterexample.py``
    (default 17 copies) end to end. Measured runtime: about 43 s in pure Python."""
    r = check_counterexample_claims(copies=17)
    assert r["compact_holds"]
    assert r["deletions_break"] == r["deletions_total"] == 2160
    assert r["contractions_break"] == r["contractions_total"] == 216
    assert r["preserving_operations"] == []
    print(f"copies=17 replication took {r['seconds']:.1f}s")
