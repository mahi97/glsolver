"""Adversarial review of the exact oracles ``glsolver.oracle.bruteforce`` / ``.ilp``.

Everything the oracles return is checked against the *independent* verifier
``glsolver.verify.verify_instance_parts`` (docs/paper_notes.md §13.11/§13.12)
and, for completeness, against a naive enumerator written here that tries every
assignment of the non-terminals with ``itertools.product`` and keeps the ones
the verifier accepts (docs/paper_notes.md §1.2, §1.3, [Def connected-to]).

Runnable in isolation: ``pytest tests/test_oracles_review.py``.
"""
from __future__ import annotations

import itertools
import random
import time
from typing import Iterable

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import Bounds, milp

from glsolver import generators as gen
from glsolver.instance import Instance, make_instance
from glsolver.oracle import bruteforce_partition, enumerate_partitions, ilp_partition
from glsolver.oracle.ilp import _is_valid, build_model
from glsolver.verify import verify_instance_parts

Canon = frozenset[frozenset[int]]


# ---------------------------------------------------------------------------
# independent naive enumerator (the ground truth of this file)
# ---------------------------------------------------------------------------
def canon(parts: Iterable[Iterable[int]]) -> Canon:
    return frozenset(frozenset(p) for p in parts)


def naive_solutions(inst: Instance) -> set[Canon]:
    """Every valid partition of ``inst`` by exhaustive assignment (``k^(n-k)`` tries).

    A partition is valid iff the independent verifier accepts it; nothing from
    the oracle modules is used here.
    """
    nonterm = [v for v in range(inst.n) if v not in inst.terminals]
    sols: set[Canon] = set()
    for asg in itertools.product(range(inst.k), repeat=len(nonterm)):
        parts: list[list[int]] = [[t] for t in inst.terminals]
        for v, i in zip(nonterm, asg):
            parts[i].append(v)
        if verify_instance_parts(inst, parts).valid:
            sols.add(canon(parts))
    return sols


def assert_valid(inst: Instance, parts: list[list[int]] | None) -> None:
    assert parts is not None
    report = verify_instance_parts(inst, parts)
    assert report.valid, (inst.name, report.errors)
    # docstring contract of both oracles: parts[i] starts with terminals[i]
    assert [p[0] for p in parts] == list(inst.terminals)
    assert all(len(p) == len(set(p)) for p in parts)


def grid(rows: int, cols: int) -> nx.Graph:
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols), ordering="sorted")


# ---------------------------------------------------------------------------
# instance sources
# ---------------------------------------------------------------------------
def random_small_instance(rng: random.Random, n_max: int = 7, p_min: float = 0.0) -> Instance:
    """Random *structurally valid* instance, feasible or not: arbitrary density
    ``>= p_min`` (so disconnected graphs are frequent for ``p_min = 0``),
    undirected/directed, weighted or not, and now and then a deliberately
    lopsided capacity vector."""
    n = rng.randint(1, n_max)
    k = rng.randint(1, n)
    directed = rng.random() < 0.5
    terminals = sorted(rng.sample(range(n), k))
    p = p_min + rng.random() * (1.0 - p_min)
    if directed:
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
    else:
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    weighted = rng.random() < 0.4
    weights = None
    if weighted:
        weights = [0 if v in terminals else rng.randint(1, 4) for v in range(n)]
        total = sum(weights) + rng.choice([0, 0, 1, 3])
    else:
        total = n - k
    cuts = sorted(rng.randint(0, total) for _ in range(k - 1))
    caps = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    if k > 1 and rng.random() < 0.3:  # everything on one terminal (often a bad choice)
        caps = [0] * (k - 1) + [total]
        rng.shuffle(caps)
    return make_instance(
        n, edges, terminals, caps, weights=weights, directed=directed,
        name=f"rnd_n{n}_k{k}_{'d' if directed else 'u'}{'w' if weighted else ''}",
    )


def deliberately_bad_instances() -> list[Instance]:
    """Hand-made infeasible / borderline instances (all structurally valid)."""
    return [
        # path 0-1-2, terminals (0,1), sizes (2,1): part of 0 would be {0,2} (disconnected)
        make_instance(3, [(0, 1), (1, 2)], [0, 1], sizes=[2, 1], directed=False, name="path"),
        # star with centre 1 and leaf terminals 2, 3, c = (1, 1): whichever terminal
        # does not get the centre is stuck with the other leaf (not adjacent)
        make_instance(4, [(1, 0), (1, 2), (1, 3)], [2, 3], [1, 1], directed=False, name="star_leaf"),
        # terminal with capacity but no in-neighbour at all
        make_instance(4, [(0, 1), (1, 2)], [0, 3], [0, 2], directed=False, name="isolated_terminal"),
        # directed: everything points away from the terminal
        make_instance(3, [(0, 1), (1, 2)], [0], [2], directed=True, name="wrong_direction"),
        # two components, capacity split does not match the components
        make_instance(5, [(0, 1), (1, 2), (3, 4)], [0, 3], [1, 2], directed=False, name="two_comps"),
        # feasible twin of the above (control)
        make_instance(5, [(0, 1), (1, 2), (3, 4)], [0, 3], [2, 1], directed=False, name="two_comps_ok"),
        # weighted: two weight-3 vertices only reach terminal 0 with bound 3 + 2 = 5 < 6
        make_instance(
            4, [(2, 0), (3, 0)], [0, 1], [3, 3], weights=[0, 0, 3, 3], directed=True,
            name="weighted_over_bound",
        ),
        # weighted control: bound 4 + 2 = 6 fits both (exceeds c_0 by w_max - 1)
        make_instance(
            4, [(2, 0), (3, 0)], [0, 1], [4, 2], weights=[0, 0, 3, 3], directed=True,
            name="weighted_within_bound",
        ),
        # weighted with w_max = 1 (no slack): the only reachable terminal has c = 0
        make_instance(
            3, [(2, 1)], [0, 1], [1, 0], weights=[0, 0, 1], directed=True, name="weighted_exact",
        ),
        # no arcs at all but a non-terminal
        make_instance(2, [], [0], [1], directed=True, name="no_arcs"),
    ]


BAD_EXPECTED = {
    "path": "infeasible",
    "star_leaf": "infeasible",
    "isolated_terminal": "infeasible",
    "wrong_direction": "infeasible",
    "two_comps": "infeasible",
    "two_comps_ok": "ok",
    "weighted_over_bound": "infeasible",
    "weighted_within_bound": "ok",
    "weighted_exact": "infeasible",
    "no_arcs": "infeasible",
}


def small_generator_instances() -> list[Instance]:
    """>= 200 instances with n <= 8 drawn from the generator families, covering
    undirected/directed x unweighted/weighted and all capacity modes."""
    out: list[Instance] = []
    modes = list(gen._MODES)
    und_builders = [
        lambda s, m: gen.complete_graph(7, 3, s, m),
        lambda s, m: gen.cycle_graph(8, 2, s, m),
        lambda s, m: gen.wheel_graph(7, 3, s, m),
        lambda s, m: gen.grid_graph(2, 4, 2, s, m),
        lambda s, m: gen.grid3d_graph(2, 2, 2, 3, s, m),
        lambda s, m: gen.harary_graph(8, 3, s, m),
        lambda s, m: gen.random_regular_graph(8, 3, 3, s, m),
        lambda s, m: gen.erdos_renyi_graph(8, 0.6, 3, s, m),
        lambda s, m: gen.random_geometric_graph(8, 0.75, 2, s, m),
        lambda s, m: gen.adversarial_ladder(8, 2, s, m),
        lambda s, m: gen.sparse_k_connected(8, 2, s, m),
        lambda s, m: gen.expander_graph(8, 3, 3, s, m),
        lambda s, m: gen.dense_graph(7, 3, s, m),
    ]
    for j, build in enumerate(und_builders):
        for mi, mode in enumerate(modes):
            seed = 10 * j + mi
            base = build(seed, mode)
            assert base.n <= 8 and not base.directed
            weighted = gen.weighted_variant(base, seed, w_max=3, slack=mi % 2)
            out.append(base)
            out.append(weighted)
            if mi < 2:
                out.append(gen.directed_variant(base, seed, drop_fraction=0.5))
                out.append(gen.directed_variant(weighted, seed, drop_fraction=0.5))
    for mi, mode in enumerate(modes):
        for k in (2, 3):
            out.append(gen.random_kT_connected_dag(8, k, mi, mode))
            out.append(gen.random_kT_connected_dag(8, k, mi, mode, weighted=True, w_max=3))
        out.append(gen.layered_dag(2, 3, 2, mi, mode))
        out.append(gen.layered_dag(2, 2, 2, mi, mode, weighted=True))
        dig = gen.random_kT_connected_digraph(7, 2, mi, mode)
        out.append(dig)
        out.append(gen.weighted_variant(dig, mi, w_max=4, slack=1))
    assert all(i.n <= 8 for i in out)
    return out


GENERATOR_INSTANCES = small_generator_instances()


# ---------------------------------------------------------------------------
# soundness on generator instances (theorems guarantee feasibility)
# ---------------------------------------------------------------------------
def test_generator_sample_is_large_and_diverse():
    assert len(GENERATOR_INSTANCES) >= 200
    combos = {(i.directed, i.is_weighted) for i in GENERATOR_INSTANCES}
    assert combos == {(False, False), (False, True), (True, False), (True, True)}
    for combo in combos:
        assert sum(1 for i in GENERATOR_INSTANCES if (i.directed, i.is_weighted) == combo) >= 20
    modes = {i.meta.get("mode") for i in GENERATOR_INSTANCES}
    assert set(gen._MODES) <= modes


def test_bruteforce_sound_and_finds_partition_on_generator_instances():
    """All generator instances are k-T-connected (claimed and verified by the
    generators), so a partition exists [Thm k-t-conn] / [Thm weighted-k-t-conn]:
    the complete oracle must return "ok" and the verifier must accept the parts."""
    for inst in GENERATOR_INSTANCES:
        status, parts = bruteforce_partition(inst, seed=inst.n)
        assert status == "ok", (inst.name, status)
        assert_valid(inst, parts)


@pytest.mark.ilp
def test_ilp_sound_and_finds_partition_on_generator_instances():
    for inst in GENERATOR_INSTANCES:
        status, parts = ilp_partition(inst)
        assert status == "ok", (inst.name, status)
        assert_valid(inst, parts)


# ---------------------------------------------------------------------------
# completeness versus the naive enumerator
# ---------------------------------------------------------------------------
@pytest.mark.ilp
def test_statuses_agree_with_naive_enumerator_on_random_and_bad_instances():
    rng = random.Random(20260921)
    insts = [random_small_instance(rng) for _ in range(150)] + deliberately_bad_instances()
    n_ok = n_inf = 0
    for inst in insts:
        ref = naive_solutions(inst)
        st_bf, parts_bf = bruteforce_partition(inst, seed=rng.randrange(8))
        st_ilp, parts_ilp = ilp_partition(inst)
        expected = "ok" if ref else "infeasible"
        assert st_bf == expected, (inst.name, inst.arcs, inst.capacities, inst.weights)
        assert st_ilp == expected, (inst.name, inst.arcs, inst.capacities, inst.weights)
        if ref:
            n_ok += 1
            assert_valid(inst, parts_bf)
            assert_valid(inst, parts_ilp)
            assert canon(parts_bf) in ref
            assert canon(parts_ilp) in ref
        else:
            n_inf += 1
            assert parts_bf is None and parts_ilp is None
    assert n_ok >= 40 and n_inf >= 40, (n_ok, n_inf)


def test_enumeration_equals_naive_solution_set():
    rng = random.Random(77)
    insts = [random_small_instance(rng) for _ in range(60)]
    insts += [random_small_instance(rng, p_min=0.6) for _ in range(60)]
    insts += deliberately_bad_instances()
    nontrivial = 0
    for inst in insts:
        ref = naive_solutions(inst)
        found = [canon(p) for p in enumerate_partitions(inst, seed=rng.randrange(8))]
        assert len(found) == len(set(found)), ("duplicate solution", inst.name)
        assert set(found) == ref, (inst.name, inst.arcs, inst.capacities, inst.weights)
        if len(ref) > 1:
            nontrivial += 1
        for parts in enumerate_partitions(inst):
            assert_valid(inst, parts)
    assert nontrivial >= 10


@pytest.mark.ilp
def test_bad_instances_have_expected_status():
    for inst in deliberately_bad_instances():
        expected = BAD_EXPECTED[inst.name]
        assert bruteforce_partition(inst)[0] == expected, inst.name
        assert ilp_partition(inst)[0] == expected, inst.name
        assert bool(naive_solutions(inst)) == (expected == "ok"), inst.name


# ---------------------------------------------------------------------------
# ILP: the flow constraints must reject disconnected parts
# ---------------------------------------------------------------------------
def _pinned_status(inst: Instance, assignment: dict[int, int], *, relax: bool = False) -> int:
    """HiGHS status of the flow model with ``x`` fixed to ``assignment``
    (non-terminal -> part index): 0 = feasible, 2 = infeasible."""
    idx, c, bounds, integrality, cons = build_model(inst)
    lb, ub = bounds.lb.copy(), bounds.ub.copy()
    for v, i in assignment.items():
        for j in range(inst.k):
            lb[idx.x(v, j)] = ub[idx.x(v, j)] = 1.0 if i == j else 0.0
    if relax:
        integrality = np.zeros_like(integrality)
    res = milp(c, constraints=[cons], integrality=integrality, bounds=Bounds(lb, ub),
               options={"disp": False})
    return int(res.status)


@pytest.mark.ilp
def test_ilp_rejects_explicitly_disconnected_part():
    # Path 0-1-2-3-4 with terminals 1 and 3 (non-terminals 0, 2, 4).
    # c = (2, 1): the only connected choice is {1,0,2} | {3,4}.  Pinning 0 and 4
    # into part 0 puts two vertices with no arc between them and no path inside
    # {0,1,4} into one part -> the flow rows must be infeasible.
    edges = [(i, i + 1) for i in range(4)]
    p21 = make_instance(5, edges, [1, 3], [2, 1], directed=False, name="p5_c21")
    assert naive_solutions(p21) == {canon([[1, 0, 2], [3, 4]])}
    assert _pinned_status(p21, {0: 0, 2: 0, 4: 1}) == 0
    assert _pinned_status(p21, {0: 0, 4: 0, 2: 1}) == 2
    assert _pinned_status(p21, {0: 0, 4: 0, 2: 1}, relax=True) == 2  # not even fractionally
    assert _pinned_status(p21, {2: 0, 4: 0, 0: 1}) == 2
    # c = (3, 0): every size-respecting assignment has a disconnected part -> infeasible
    p30 = make_instance(5, edges, [1, 3], [3, 0], directed=False, name="p5_c30")
    assert naive_solutions(p30) == set()
    assert ilp_partition(p30) == ("infeasible", None)
    assert bruteforce_partition(p30) == ("infeasible", None)
    assert list(enumerate_partitions(p30)) == []
    # the "path example": path 0-1-2, terminals (0,1), sizes (2,1)
    path = make_instance(3, [(0, 1), (1, 2)], [0, 1], sizes=[2, 1], directed=False)
    assert ilp_partition(path) == ("infeasible", None)
    assert bruteforce_partition(path) == ("infeasible", None)
    assert _pinned_status(path, {2: 0}) == 2
    # directed twin: arcs 0->1, 2->3, 4->3 with c = (2, 1): 0 and 2 (or 4) in part 0
    # have no arc between them and no path inside the part
    dpath = make_instance(5, [(0, 1), (2, 3), (4, 3)], [1, 3], [2, 1], directed=True)
    assert naive_solutions(dpath) == set()
    assert ilp_partition(dpath) == ("infeasible", None)
    assert bruteforce_partition(dpath) == ("infeasible", None)
    assert _pinned_status(dpath, {0: 0, 2: 0, 4: 1}) == 2
    assert _pinned_status(dpath, {0: 0, 2: 0, 4: 1}, relax=True) == 2
    dok = make_instance(5, [(0, 1), (2, 3), (4, 3)], [1, 3], [1, 2], directed=True)
    assert _pinned_status(dok, {0: 0, 2: 1, 4: 1}) == 0


@pytest.mark.ilp
def test_ilp_rejects_part_whose_vertices_connect_only_through_other_parts():
    # C6 with terminals 0 and 3: part 0 = {0,2,4} is disconnected inside the part
    # although 2 and 4 reach 0 in G (through 1 resp. 5, which belong to part 1).
    c6 = make_instance(6, nx.cycle_graph(6).edges(), [0, 3], [2, 2], directed=False)
    assert _pinned_status(c6, {2: 0, 4: 0, 1: 1, 5: 1}) == 2
    assert _pinned_status(c6, {2: 0, 4: 0, 1: 1, 5: 1}, relax=True) == 2
    assert _pinned_status(c6, {1: 0, 5: 0, 2: 1, 4: 1}) == 0
    ref = naive_solutions(c6)
    assert len(ref) == 3
    st, parts = ilp_partition(c6)
    assert st == "ok"
    assert_valid(c6, parts)
    assert canon(parts) in ref


@pytest.mark.ilp
def test_ilp_flow_bound_suffices_for_long_chains():
    """The flow on the last arc of a chain equals the number of non-terminals of
    the part; ``B = |V \\ T|`` must therefore be enough for a single huge part."""
    n = 14
    path = make_instance(n, [(i, i + 1) for i in range(n - 1)], [0], [n - 1], directed=False)
    st, parts = ilp_partition(path)
    assert st == "ok"
    assert_valid(path, parts)
    chain = make_instance(n, [(i + 1, i) for i in range(n - 2)], [0, n - 1], [n - 2, 0], directed=True)
    st, parts = ilp_partition(chain)
    assert st == "ok"
    assert_valid(chain, parts)
    # reversing the chain makes every vertex unreachable
    rev = make_instance(n, [(i, i + 1) for i in range(n - 2)], [0, n - 1], [n - 2, 0], directed=True)
    assert ilp_partition(rev) == ("infeasible", None)
    assert bruteforce_partition(rev) == ("infeasible", None)


# ---------------------------------------------------------------------------
# weighted bound c_t + w_max - 1 [Thm weighted-k-t-conn]
# ---------------------------------------------------------------------------
@pytest.mark.ilp
def test_weighted_bound_is_c_plus_wmax_minus_one():
    # paper's slack example: k terminals with c = 1, one vertex of weight k adjacent to all
    for k in (2, 3, 4):
        inst = make_instance(
            k + 1, [(k, i) for i in range(k)], list(range(k)), [1] * k,
            weights=[0] * k + [k], directed=False, name=f"slack{k}",
        )
        for oracle in (bruteforce_partition, ilp_partition):
            st, parts = oracle(inst)
            assert st == "ok"
            report = verify_instance_parts(inst, parts)
            assert report.valid
            assert len(report.details["exceeds_capacity"]) == 1  # only the slack makes it fit
        assert len(naive_solutions(inst)) == k
    over = make_instance(
        4, [(2, 0), (3, 0)], [0, 1], [3, 3], weights=[0, 0, 3, 3], directed=True,
    )  # part 0 must take weight 6 > c_0 + w_max - 1 = 5
    within = make_instance(
        4, [(2, 0), (3, 0)], [0, 1], [4, 2], weights=[0, 0, 3, 3], directed=True,
    )  # bound 4 + 2 = 6: fits, exceeding c_0 by exactly w_max - 1 = 2
    assert bruteforce_partition(over) == ("infeasible", None)
    assert ilp_partition(over) == ("infeasible", None)
    for oracle in (bruteforce_partition, ilp_partition):
        st, parts = oracle(within)
        assert st == "ok"
        report = verify_instance_parts(within, parts)
        assert report.valid and report.details["exceeds_capacity"] == [0]
        assert report.details["part_weights"][0] == 6
    # unit weights with a tight total behave exactly like the unweighted problem
    exact = make_instance(3, [(2, 1)], [0, 1], [1, 0], weights=[0, 0, 1], directed=True)
    assert bruteforce_partition(exact) == ("infeasible", None)
    assert ilp_partition(exact) == ("infeasible", None)


def test_weighted_oracle_never_exceeds_bound_on_random_weighted_instances():
    rng = random.Random(5)
    seen_exceed = weighted_seen = 0
    for _ in range(150):
        inst = random_small_instance(rng, n_max=7)
        if not inst.is_weighted:
            continue
        weighted_seen += 1
        for parts in enumerate_partitions(inst):
            report = verify_instance_parts(inst, parts)
            assert report.valid, (inst.arcs, inst.capacities, inst.weights, parts, report.errors)
            seen_exceed += bool(report.details["exceeds_capacity"])
    assert weighted_seen >= 30
    assert seen_exceed > 0  # the slack is actually exercised


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
@pytest.mark.ilp
def test_edge_cases_no_nonterminals():
    cases = [
        make_instance(1, [], [0], [0], directed=True),
        make_instance(2, [(0, 1)], [0, 1], [0, 0], directed=False),
        make_instance(3, [(0, 1), (1, 2)], [0, 1, 2], [0, 0, 0], directed=True),
        make_instance(2, [(0, 1)], [0, 1], [0, 5], weights=[0, 0], directed=False),
        make_instance(3, [(0, 1), (1, 2)], [0, 1, 2], [7, 0, 9], weights=[0, 0, 0], directed=False),
    ]
    for inst in cases:
        expected = [[t] for t in inst.terminals]
        assert bruteforce_partition(inst) == ("ok", expected)
        assert ilp_partition(inst) == ("ok", expected)
        assert list(enumerate_partitions(inst)) == [expected]
        assert naive_solutions(inst) == {canon(expected)}


@pytest.mark.ilp
def test_zero_capacity_parts_and_k_equals_one():
    G = nx.cycle_graph(6)
    inst = make_instance(6, G.edges(), [0, 3], [4, 0], directed=False)
    for oracle in (bruteforce_partition, ilp_partition):
        st, parts = oracle(inst)
        assert st == "ok"
        assert_valid(inst, parts)
        assert parts[1] == [3]
    assert len(list(enumerate_partitions(inst))) == 1
    single = make_instance(6, G.edges(), [2], [5], directed=False)
    for oracle in (bruteforce_partition, ilp_partition):
        st, parts = oracle(single)
        assert st == "ok" and canon(parts) == canon([list(range(6))])
    # k = 1 but a vertex cannot reach the terminal
    lonely = make_instance(4, [(1, 0), (2, 0)], [0], [3], directed=True)
    assert bruteforce_partition(lonely) == ("infeasible", None)
    assert ilp_partition(lonely) == ("infeasible", None)


# ---------------------------------------------------------------------------
# determinism, seeds, time limits
# ---------------------------------------------------------------------------
def test_seed_changes_order_but_not_the_solution_set():
    inst = make_instance(7, nx.wheel_graph(7).edges(), [0, 3], [3, 2], directed=False)
    sets = [frozenset(canon(p) for p in enumerate_partitions(inst, seed=s)) for s in range(6)]
    assert all(s == sets[0] for s in sets)
    assert len(sets[0]) > 1
    firsts = {canon(bruteforce_partition(inst, seed=s)[1]) for s in range(6)}
    assert firsts <= sets[0]
    for s in range(3):
        assert bruteforce_partition(inst, seed=s) == bruteforce_partition(inst, seed=s)
        assert list(enumerate_partitions(inst, seed=s)) == list(enumerate_partitions(inst, seed=s))
    assert list(enumerate_partitions(inst, limit=2, seed=1)) == list(enumerate_partitions(inst, seed=1))[:2]
    assert list(enumerate_partitions(inst, limit=0)) == []
    assert list(enumerate_partitions(inst, limit=-3)) == []


@pytest.mark.perf
@pytest.mark.ilp
def test_time_limits_report_timeout_or_a_valid_partition():
    inst = make_instance(81, grid(9, 9).edges(), [0, 80, 40, 8], [20, 20, 19, 18], directed=False)
    t0 = time.perf_counter()
    assert bruteforce_partition(inst, time_limit=0.0) == ("timeout", None)
    st, parts = ilp_partition(inst, time_limit=0.0)
    assert time.perf_counter() - t0 < 5.0
    assert st in ("timeout", "ok")
    if st == "ok":
        assert_valid(inst, parts)
    else:
        assert parts is None
    # a generous limit on a small instance still solves it
    small = make_instance(9, grid(3, 3).edges(), [0, 8], [4, 3], directed=False)
    for oracle in (bruteforce_partition, ilp_partition):
        st, parts = oracle(small, time_limit=30.0)
        assert st == "ok"
        assert_valid(small, parts)


# ---------------------------------------------------------------------------
# the ILP's private salvage check ``_is_valid`` (used on time-limited incumbents)
# ---------------------------------------------------------------------------
def _random_parts(inst: Instance, rng: random.Random) -> list[list[int]]:
    parts: list[list[int]] = [[t] for t in inst.terminals]
    for v in range(inst.n):
        if v not in inst.terminals:
            parts[rng.randrange(inst.k)].append(v)
    return parts


def test_is_valid_agrees_with_verifier_on_duplicate_free_parts():
    rng = random.Random(9)
    agree_true = agree_false = 0
    for _ in range(200):
        inst = random_small_instance(rng)
        for _ in range(4):
            parts = _random_parts(inst, rng)
            if rng.random() < 0.3 and inst.n > inst.k:  # drop a vertex: cover violated
                v = rng.choice([v for v in range(inst.n) if v not in inst.terminals])
                parts = [[x for x in p if x != v] for p in parts]
            expected = verify_instance_parts(inst, parts).valid
            assert _is_valid(inst, parts) is expected, (inst.arcs, inst.capacities, inst.weights, parts)
            if expected:
                agree_true += 1
            else:
                agree_false += 1
    assert agree_true > 20 and agree_false > 20


def test_is_valid_rejects_duplicate_vertex_inside_a_part():
    """Regression for the review finding: ``_is_valid`` once worked on
    ``set(part)`` and accepted a part listing the same vertex twice."""
    inst = make_instance(
        4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)], [2, 3], [0, 4],
        weights=[3, 1, 0, 0], directed=False,
    )
    parts = [[2, 1, 1], [3, 0]]
    assert not verify_instance_parts(inst, parts).valid
    assert _is_valid(inst, parts) is False


@pytest.mark.ilp
def test_decode_never_repeats_or_drops_a_vertex():
    """The salvage path relies on ``decode`` producing an exact cover; check it
    on solved models (the only place duplicates could enter ``_is_valid``)."""
    rng = random.Random(11)
    checked = 0
    for _ in range(60):
        inst = random_small_instance(rng)
        idx, c, bounds, integ, cons = build_model(inst)
        res = milp(c, constraints=[cons] if cons is not None else None, integrality=integ,
                   bounds=bounds, options={"disp": False})
        if res.status != 0:
            continue
        parts = idx.decode(res.x, inst.terminals)
        flat = [v for p in parts for v in p]
        assert sorted(flat) == list(range(inst.n))
        assert [p[0] for p in parts] == list(inst.terminals)
        assert _is_valid(inst, parts) is verify_instance_parts(inst, parts).valid is True
        checked += 1
    assert checked >= 20
