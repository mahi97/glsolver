"""Exact oracles: brute force vs ILP agreement, completeness and soundness.

Runnable in isolation: ``pytest tests/test_oracles.py``.
"""
from __future__ import annotations

import itertools

import networkx as nx
import pytest

from glsolver.instance import Instance, make_instance
from glsolver.oracle import bruteforce_partition, enumerate_partitions, ilp_partition
from glsolver.oracle.ilp import VarIndex, build_model

try:  # the independent verifier may not exist yet; the local checker is authoritative here
    from glsolver.verify import verify_instance_parts as _verify_instance_parts
except Exception:  # pragma: no cover - depends on sibling module availability
    _verify_instance_parts = None


# ---------------------------------------------------------------------------
# Minimal local checker (docs/paper_notes.md §1.2 / §1.3, [Def connected-to])
# ---------------------------------------------------------------------------
def check_parts(inst: Instance, parts: list[list[int]]) -> list[str]:
    """Return a list of violations (empty iff ``parts`` is a valid partition)."""
    errors: list[str] = []
    if len(parts) != inst.k:
        return [f"expected {inst.k} parts, got {len(parts)}"]
    tset = set(inst.terminals)
    weights = list(inst.weights) if inst.weights is not None else [1] * inst.n
    slack = inst.w_max - 1 if inst.is_weighted else 0
    in_adj = inst.in_adjacency()
    seen: set[int] = set()
    for i, part in enumerate(parts):
        t = inst.terminals[i]
        pset = set(part)
        if len(pset) != len(part):
            errors.append(f"part {i} repeats a vertex")
        if t not in pset:
            errors.append(f"part {i} misses its terminal {t}")
        if pset & seen:
            errors.append(f"part {i} overlaps an earlier part")
        seen |= pset
        w = sum(weights[v] for v in pset if v not in tset)
        if inst.is_weighted:
            if w > inst.capacities[i] + slack:
                errors.append(f"part {i} weight {w} > {inst.capacities[i] + slack}")
        elif len(pset) != inst.capacities[i] + 1:
            errors.append(f"part {i} size {len(pset)} != {inst.capacities[i] + 1}")
        reach = {t}
        stack = [t]
        while stack:
            x = stack.pop()
            for y in in_adj[x]:
                if y in pset and y not in reach:
                    reach.add(y)
                    stack.append(y)
        if reach != pset:
            errors.append(f"part {i} is not connected to {t}")
    if seen != set(range(inst.n)):
        errors.append("not every vertex is assigned")
    return errors


def assert_valid(inst: Instance, parts: list[list[int]] | None) -> None:
    assert parts is not None
    assert check_parts(inst, parts) == []
    if _verify_instance_parts is not None:
        try:
            report = _verify_instance_parts(inst, parts)
        except TypeError:  # pragma: no cover - unknown signature of a sibling module
            return
        valid = getattr(report, "valid", report)
        if isinstance(valid, bool):
            assert valid is True


def naive_solutions(inst: Instance) -> set[frozenset[frozenset[int]]]:
    """All valid partitions by trying every assignment (tiny instances only)."""
    nonterm = [v for v in range(inst.n) if v not in inst.terminals]
    sols: set[frozenset[frozenset[int]]] = set()
    for asg in itertools.product(range(inst.k), repeat=len(nonterm)):
        parts: list[list[int]] = [[t] for t in inst.terminals]
        for v, i in zip(nonterm, asg):
            parts[i].append(v)
        if not check_parts(inst, parts):
            sols.add(frozenset(frozenset(p) for p in parts))
    return sols


def canon(parts: list[list[int]]) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(p) for p in parts)


# ---------------------------------------------------------------------------
# Fixed instance lists
# ---------------------------------------------------------------------------
def weak_compositions(total: int, k: int) -> list[tuple[int, ...]]:
    if k == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        out.extend((first, *rest) for rest in weak_compositions(total - first, k - 1))
    return out


def atlas_instances() -> list[Instance]:
    """Every connected graph on 2..5 vertices, k in {1,2,3} (first k vertices
    as terminals), every weak composition of the capacities (283 instances)."""
    insts: list[Instance] = []
    for G in nx.graph_atlas_g():
        n = G.number_of_nodes()
        if not 2 <= n <= 5 or not nx.is_connected(G):
            continue
        for k in (1, 2, 3):
            if k > n:
                continue
            for caps in weak_compositions(n - k, k):
                insts.append(
                    make_instance(
                        n, G.edges(), list(range(k)), list(caps), directed=False,
                        name=f"atlas{G.name}_k{k}_{caps}",
                    )
                )
    return insts


ATLAS = atlas_instances()


def path_infeasible() -> Instance:
    """Path 0-1-2, terminals (0,1), sizes (2,1): part of 0 would be {0,2}."""
    return make_instance(3, [(0, 1), (1, 2)], [0, 1], sizes=[2, 1], directed=False)


def paper_running_example() -> Instance:
    arcs = [(7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1),
            (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2)]
    return make_instance(9, arcs, [0, 1, 2], [2, 2, 2], directed=True)


def weighted_instances() -> list[Instance]:
    """Three weighted instances (§1.3): the paper's slack example (feasible only
    thanks to ``w_max - 1``), a weighted wheel, and an infeasible star."""
    k = 3
    slack_example = make_instance(
        k + 1, [(k, i) for i in range(k)], list(range(k)), [1] * k,
        weights=[0] * k + [k], directed=False, name="slack",
    )
    wheel = make_instance(
        7, nx.wheel_graph(7).edges(), [1, 2, 3], [4, 3, 2],
        weights=[2, 0, 0, 0, 3, 1, 2], directed=False, name="wheel_w",
    )
    # terminal 0 - {1, 2}, terminal 3 - {2}; w = (4, 4); c = (0, 8), w_max - 1 = 3:
    # vertex 1 only reaches terminal 0 whose bound is 3 < 4  -> infeasible
    star = make_instance(
        4, [(0, 1), (0, 2), (3, 2)], [0, 3], [0, 8],
        weights=[0, 4, 4, 0], directed=False, name="star_infeasible",
    )
    return [slack_example, wheel, star]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_atlas_instance_count():
    assert 250 <= len(ATLAS) <= 300


@pytest.mark.ilp
def test_bruteforce_ilp_agree_on_atlas():
    n_ok = n_inf = 0
    for inst in ATLAS:
        st_bf, parts_bf = bruteforce_partition(inst)
        st_ilp, parts_ilp = ilp_partition(inst)
        assert st_bf == st_ilp, inst.name
        if st_bf == "ok":
            n_ok += 1
            assert_valid(inst, parts_bf)
            assert_valid(inst, parts_ilp)
        else:
            n_inf += 1
            assert st_bf == "infeasible"
            assert parts_bf is None and parts_ilp is None
    # both outcomes must actually occur in the sample
    assert n_ok > 100 and n_inf > 20


def test_enumeration_matches_naive_on_atlas_sample():
    # every 5th atlas instance keeps this well under a second
    for inst in ATLAS[::5]:
        found = [canon(p) for p in enumerate_partitions(inst)]
        assert len(found) == len(set(found)), "duplicate solution"
        for parts in enumerate_partitions(inst):
            assert_valid(inst, parts)
        assert set(found) == naive_solutions(inst), inst.name


def test_infeasible_path():
    inst = path_infeasible()
    assert bruteforce_partition(inst) == ("infeasible", None)
    assert ilp_partition(inst) == ("infeasible", None)
    assert list(enumerate_partitions(inst)) == []
    # swapping roles makes it feasible: {0}, {1,2}
    feasible = make_instance(3, [(0, 1), (1, 2)], [0, 2], sizes=[1, 2], directed=False)
    st, parts = bruteforce_partition(feasible)
    assert st == "ok" and canon(parts) == canon([[0], [2, 1]])
    st, parts = ilp_partition(feasible)
    assert st == "ok" and canon(parts) == canon([[0], [2, 1]])


def test_disconnected_graph_is_handled():
    # two isolated edges; terminal 0 cannot collect vertex 3
    inst = make_instance(4, [(0, 1), (2, 3)], [0, 2], [2, 0], directed=False)
    assert bruteforce_partition(inst)[0] == "infeasible"
    assert ilp_partition(inst)[0] == "infeasible"
    ok = make_instance(4, [(0, 1), (2, 3)], [0, 2], [1, 1], directed=False)
    assert canon(bruteforce_partition(ok)[1]) == canon([[0, 1], [2, 3]])
    assert canon(ilp_partition(ok)[1]) == canon([[0, 1], [2, 3]])


def test_zero_capacity_and_single_terminal():
    G = nx.cycle_graph(5)
    inst = make_instance(5, G.edges(), [0, 2], [3, 0], directed=False)
    st, parts = bruteforce_partition(inst)
    assert st == "ok" and set(parts[1]) == {2}
    assert_valid(inst, parts)
    assert ilp_partition(inst)[0] == "ok"
    single = make_instance(5, G.edges(), [3], [4], directed=False)
    assert canon(bruteforce_partition(single)[1]) == canon([[0, 1, 2, 3, 4]])
    assert canon(ilp_partition(single)[1]) == canon([[0, 1, 2, 3, 4]])


def test_directed_paper_running_example():
    inst = paper_running_example()
    st, parts = bruteforce_partition(inst)
    assert st == "ok"
    assert_valid(inst, parts)
    st2, parts2 = ilp_partition(inst)
    assert st2 == "ok"
    assert_valid(inst, parts2)
    sols = [canon(p) for p in enumerate_partitions(inst)]
    assert len(sols) == 1 and sols[0] == canon(parts) == canon(parts2)
    assert set(naive_solutions(inst)) == set(sols)


def test_directed_reachability_direction_matters():
    # 1 -> 0 and 2 -> 1: part {0,1,2} is connected to 0 ; reversing arcs breaks it
    inst = make_instance(3, [(1, 0), (2, 1)], [0], [2], directed=True)
    assert bruteforce_partition(inst)[0] == "ok"
    assert ilp_partition(inst)[0] == "ok"
    # now 2 has no path to 0 inside the part (arc 1 -> 2 only)
    bad = make_instance(3, [(1, 0), (1, 2)], [0], [2], directed=True)
    assert bruteforce_partition(bad)[0] == "infeasible"
    assert ilp_partition(bad)[0] == "infeasible"


@pytest.mark.ilp
def test_weighted_instances_agree():
    expected = ["ok", "ok", "infeasible"]
    for inst, exp in zip(weighted_instances(), expected):
        assert inst.is_weighted
        st_bf, parts_bf = bruteforce_partition(inst)
        st_ilp, parts_ilp = ilp_partition(inst)
        assert st_bf == st_ilp == exp, inst.name
        if exp == "ok":
            assert_valid(inst, parts_bf)
            assert_valid(inst, parts_ilp)
    # the weighted wheel has many solutions; enumeration must match naive search
    wheel = weighted_instances()[1]
    found = [canon(p) for p in enumerate_partitions(wheel)]
    assert len(found) == len(set(found))
    assert set(found) == naive_solutions(wheel)
    assert len(found) > 1


def test_weighted_bound_uses_w_max_slack():
    # one vertex of weight 3, capacities (1,1,1): only the slack w_max-1 = 2 makes it fit
    inst = weighted_instances()[0]
    st, parts = bruteforce_partition(inst)
    assert st == "ok"
    heavy = [i for i, p in enumerate(parts) if len(p) == 2]
    assert len(heavy) == 1
    assert sum(len(p) for p in parts) == inst.n


def test_enumerate_limit_and_determinism():
    G = nx.complete_graph(6)
    inst = make_instance(6, G.edges(), [0, 1], [2, 2], directed=False)
    all_sols = [canon(p) for p in enumerate_partitions(inst)]
    assert len(all_sols) == 6  # choose 2 of the 4 non-terminals for part 0
    assert len(list(enumerate_partitions(inst, limit=2))) == 2
    assert list(enumerate_partitions(inst, limit=0)) == []
    assert bruteforce_partition(inst, seed=3) == bruteforce_partition(inst, seed=3)
    # different seeds may return different solutions, but always valid ones
    for seed in range(4):
        st, parts = bruteforce_partition(inst, seed=seed)
        assert st == "ok"
        assert_valid(inst, parts)


def test_bruteforce_time_limit():
    G = nx.grid_2d_graph(6, 6)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    inst = make_instance(36, G.edges(), [0, 35], [17, 17], directed=False)
    assert bruteforce_partition(inst, time_limit=0.0) == ("timeout", None)
    st, parts = bruteforce_partition(inst, time_limit=30.0)
    assert st == "ok"
    assert_valid(inst, parts)


@pytest.mark.ilp
def test_ilp_time_limit_is_accepted():
    G = nx.grid_2d_graph(5, 5)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    inst = make_instance(25, G.edges(), [0, 24, 12], [8, 8, 6], directed=False)
    st, parts = ilp_partition(inst, time_limit=30.0)
    assert st in ("ok", "timeout")
    if st == "ok":
        assert_valid(inst, parts)


def test_var_index_is_a_bijection():
    idx = VarIndex(n=5, k=2, m=7)
    cols = [idx.x(v, i) for v in range(5) for i in range(2)]
    cols += [idx.f(i, a) for i in range(2) for a in range(7)]
    assert sorted(cols) == list(range(idx.num_vars))
    assert idx.num_vars == 5 * 2 + 2 * 7
    with pytest.raises(IndexError):
        idx.x(5, 0)
    with pytest.raises(IndexError):
        idx.f(0, 7)


def test_build_model_shapes():
    inst = paper_running_example()
    idx, c, bounds, integrality, cons = build_model(inst)
    n, k, m = inst.n, inst.k, inst.m
    assert idx.num_vars == n * k + k * m == len(c) == len(integrality)
    assert integrality[: n * k].sum() == n * k and integrality[n * k:].sum() == 0
    # terminals are pinned through the bounds
    for i, t in enumerate(inst.terminals):
        for j in range(k):
            assert bounds.lb[idx.x(t, j)] == bounds.ub[idx.x(t, j)] == (1.0 if i == j else 0.0)
    assert cons is not None
    rows = (n - k) + k + 2 * k * m + k * (n - k)
    assert cons.A.shape == (rows, idx.num_vars)


def test_bruteforce_ilp_agree_on_random_dags():
    from glsolver.generators import random_kT_connected_dag

    for seed in range(6):
        inst = random_kT_connected_dag(9, 2 + seed % 2, seed, mode="random")
        st_bf, parts_bf = bruteforce_partition(inst)
        st_ilp, parts_ilp = ilp_partition(inst)
        # k-T-connected DAGs always admit a partition [Lem 9.1], [Thm k-t-conn]
        assert st_bf == st_ilp == "ok"
        assert_valid(inst, parts_bf)
        assert_valid(inst, parts_ilp)


def test_random_directed_weighted_agreement():
    """Random small digraphs, unweighted and weighted: oracles agree, enumeration
    matches naive search, and every returned partition is valid."""
    import random

    rng = random.Random(2024)
    n_ok = n_inf = 0
    for _ in range(40):
        n = rng.randint(3, 7)
        k = rng.randint(1, min(3, n - 1))
        terminals = sorted(rng.sample(range(n), k))
        p = rng.random() * 0.6 + 0.2
        arcs = [
            (u, v) for u in range(n) if u not in terminals
            for v in range(n) if u != v and rng.random() < p
        ]
        weighted = rng.random() < 0.5
        if weighted:
            weights = [0 if v in terminals else rng.randint(1, 3) for v in range(n)]
            total = sum(weights) + rng.randint(0, 2)
        else:
            weights = None
            total = n - k
        cuts = sorted(rng.randint(0, total) for _ in range(k - 1))
        caps = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        inst = make_instance(n, arcs, terminals, caps, weights=weights, directed=True)
        st_bf, parts_bf = bruteforce_partition(inst)
        st_ilp, parts_ilp = ilp_partition(inst)
        assert st_bf == st_ilp
        found = [canon(q) for q in enumerate_partitions(inst)]
        assert len(found) == len(set(found))
        assert set(found) == naive_solutions(inst)
        if st_bf == "ok":
            n_ok += 1
            assert_valid(inst, parts_bf)
            assert_valid(inst, parts_ilp)
            assert canon(parts_bf) in set(found)
        else:
            n_inf += 1
            assert found == []
    assert n_ok >= 5 and n_inf >= 5
