"""Tests of the C++ weighted solver (``glsolver._core.solve_weighted`` via ``glpartition(...,
algorithm="weighted")``): GLWeightedPartition [Alg 3], RoundAndRemove [Alg 4] and the min-cost split assignment
[Prop 5.4] with the exact optimizations of docs/optimizations.md (paper_notes §5, §8, §13.6, §13.12).

Every returned partition is checked by the independent verifier ``glsolver.verify.verify_instance_parts``
(weight bound ``c_t + w_max - 1`` [Thm weighted-k-t-conn], exact sizes on unit weights, connectivity to the
terminal inside the part on the original input) and the in-arborescence certificate is checked against the
*original* arcs.  Statuses (``ok`` vs ``precondition_failed``) must agree with the literal reference
``glref.weighted`` (and, on unit weights, with the unweighted solvers); the reference is never asked for the
same *partition* (the choices of §13.8 are free).  All randomness is seeded.  Runnable in isolation:
``pytest tests/test_core_weighted.py``; the scaling smoke is ``@pytest.mark.slow`` (``--runslow``).
"""
from __future__ import annotations

import itertools
import json
import random
import time
from typing import Any

import pytest

from glsolver import _core, generators
from glsolver.api import GLResult, core_available, glpartition
from glsolver.generators import family_catalog
from glsolver.instance import Instance, make_instance
from glsolver.preconditions import check_preconditions, essential_sets_by_definition
from glsolver.verify import verify_instance_parts

pytestmark = pytest.mark.skipif(not core_available("weighted"), reason="C++ core (solve_weighted) not built")

CAPACITY_MODES = ("balanced", "unbalanced", "random", "extreme", "zeros")
TRACE_TYPES = {
    "init", "essential", "min_cost_split", "remove_terminal", "contract", "matching", "criticality",
    "delete_arc", "delete_arcs", "greedy_delete", "round_and_remove", "done",
}
LITERAL = {"greedy_contraction": False, "lazy_shift": False, "batch_unused_arcs": False}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def solve(inst: Instance, **kw: Any) -> GLResult:
    kw.setdefault("verify_preconditions", False)
    return glpartition(inst, algorithm="weighted", **kw)


def part_weight(inst: Instance, part: list[int]) -> int:
    tset = set(inst.terminals)
    if inst.weights is None:
        return sum(1 for v in part if v not in tset)
    return sum(inst.weights[v] for v in part if v not in tset)


def assert_valid(res: GLResult, inst: Instance, where: Any = "") -> None:
    """Independent verification plus the explicit weight bound of [Thm weighted-k-t-conn]."""
    assert res.status == "ok", (where, res.status, res.message)
    rep = verify_instance_parts(inst, res.parts)
    assert rep.valid, (where, rep.errors[:5])
    assert res.valid is True
    assert sorted(v for p in res.parts for v in p) == list(range(inst.n))
    for i, part in enumerate(res.parts):
        assert inst.terminals[i] in part
        assert part_weight(inst, part) <= inst.capacities[i] + inst.w_max - 1, (where, i)
    if inst.weights is None:
        assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities], where
        assert res.stats["roundings"] == 0, (where, "rounding on a unit-weight tight instance (§13.6)")
        assert rep.details["bound_used"] == "exact"
    elif res.stats["roundings"] == 0:
        # only RoundAndRemove can exceed c_t [Lem 8.6]
        assert rep.details["exceeds_capacity"] == [], where


def check_arborescence(inst: Instance, res: GLResult) -> None:
    """certificate['parents']: every non-terminal has a parent that is an ORIGINAL out-neighbour in the same
    part, and following parents reaches the part's terminal (§13.4, [Def connected-to])."""
    parents = res.certificate["parents"]
    arcs = set(inst.arcs)
    part_of = {v: i for i, p in enumerate(res.parts) for v in p}
    tset = set(inst.terminals)
    assert set(parents) == set(range(inst.n)) - tset
    for v, p in parents.items():
        assert (v, p) in arcs, (v, p)
        assert part_of[v] == part_of[p], (v, p)
    for v in parents:
        seen = {v}
        x = v
        while x not in tset:
            x = parents[x]
            assert x not in seen, "cycle in the parent pointers"
            seen.add(x)
        assert x == inst.terminals[part_of[v]]


def random_composition(rng: random.Random, total: int, parts: int) -> list[int]:
    cuts = sorted(rng.randint(0, total) for _ in range(parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def random_capacities(total: int, k: int, rng: random.Random, mode: str) -> list[int]:
    if mode == "zeros":
        nz = max(1, k // 2)
        caps = random_composition(rng, total, nz) + [0] * (k - nz)
        rng.shuffle(caps)
        return caps
    return generators.capacity_vector(total, k, rng, mode)


def with_capacities(inst: Instance, caps: list[int]) -> Instance:
    edges = inst.arcs if inst.directed else inst.undirected_edges
    return make_instance(inst.n, edges, inst.terminals, caps, weights=inst.weights, directed=inst.directed,
                         name=f"{inst.name}_caps", meta=dict(inst.meta))


def random_digraph(n: int, k: int, rng: random.Random, p: float, mode: str) -> Instance:
    terms = rng.sample(range(n), k)
    tset = set(terms)
    arcs = [(u, v) for u in range(n) if u not in tset for v in range(n) if v != u and rng.random() < p]
    return make_instance(n, arcs, terms, random_capacities(n - k, k, rng, mode), directed=True)


def claims_connected(inst: Instance) -> bool:
    return bool(inst.meta.get("claims_k_connected") or inst.meta.get("claims_kT_connected"))


def _family_grid() -> list[tuple[str, int, int, Instance]]:
    out = []
    for name, gen in family_catalog().items():
        for n in (10, 30, 60):
            for k in (1, 2, 3, 5):
                try:
                    inst = gen(n, k, 11 + n + k)
                except ValueError:
                    continue  # combination the family cannot realise
                out.append((name, n, k, inst))
    return out


FAMILY_GRID = _family_grid()
FAMILY_IDS = [f"{name}-n{n}-k{k}" for name, n, k, _ in FAMILY_GRID]


def feac_only_instances(count: int, seed: int, n_max: int = 12) -> list[Instance]:
    """Random digraphs satisfying FEAC but not k-T-connected (Theorem 2 beyond Theorem 1)."""
    rng = random.Random(seed)
    found: list[Instance] = []
    tries = 0
    while len(found) < count and tries < 60 * count:
        tries += 1
        n = rng.randint(4, n_max)
        k = rng.randint(2, min(4, n - 1))
        inst = random_digraph(n, k, rng, p=rng.choice([0.25, 0.35, 0.5]),
                              mode=rng.choice(["balanced", "random", "zeros", "unbalanced"]))
        pre = check_preconditions(inst, True)
        if pre["feac"] and not pre["k_T_connected"]:
            found.append(inst)
    assert len(found) == count
    return found


def weighted_bases(rng: random.Random, count: int) -> list[Instance]:
    """k-connected undirected graphs, k-T-connected directed variants and k-T-connected DAGs."""
    out: list[Instance] = []
    for i in range(count):
        n = rng.choice([10, 16, 24, 40])
        k = rng.choice([2, 3, 4])
        kind = i % 3
        if kind == 0:
            inst = generators.harary_graph(n, k, seed=rng.randint(0, 10**6))
        elif kind == 1:
            base = generators.erdos_renyi_graph(n, 0.4, k, seed=rng.randint(0, 10**6))
            inst = generators.directed_variant(base, rng.randint(0, 10**6), rng.choice([0.3, 0.6]))
        else:
            inst = generators.random_kT_connected_dag(n, k, seed=rng.randint(0, 10**6), extra_out=rng.choice([0, 1]))
        out.append(inst)
    return out


def small_pool(count: int, seed: int, n_max: int = 14, weighted_fraction: float = 0.5) -> list[Instance]:
    """Mixed small instances: generator families (unit or weighted variants), directed variants, FEAC-only."""
    rng = random.Random(seed)
    cat = family_catalog()
    names = [n for n in cat if not n.startswith("paper")]
    pool: list[Instance] = [generators.paper_running_example(), generators.paper_essential_example(),
                            generators.paper_contract_counterexample()]
    while len(pool) < count - count // 4:
        name = rng.choice(names)
        n = rng.randint(5, n_max)
        k = rng.randint(1, min(4, n - 1))
        try:
            inst = cat[name](n, k, rng.randint(0, 10**6))
        except ValueError:
            continue
        if inst.n > n_max:
            continue
        if not inst.directed and rng.random() < 0.3:
            inst = generators.directed_variant(inst, rng.randint(0, 10**6), 0.5)
        elif not inst.is_weighted and claims_connected(inst) and rng.random() < 0.5:
            caps = random_capacities(inst.n - inst.k, inst.k, rng, rng.choice(CAPACITY_MODES))
            inst = with_capacities(inst, list(caps))
        if not inst.is_weighted and rng.random() < weighted_fraction:
            inst = generators.weighted_variant(inst, rng.randint(0, 10**6), rng.choice([2, 3, 5]), rng.choice([0, 0, 1, 4]))
        pool.append(inst)
    pool += feac_only_instances(count - len(pool), seed + 1, n_max=min(n_max, 12))
    return pool


# ---------------------------------------------------------------------------
# 1. unit weights: paper examples, every generator family / capacity mode, FEAC-only, status agreement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [generators.paper_running_example, generators.paper_essential_example,
                                     generators.paper_contract_counterexample])
def test_paper_examples_exact(builder) -> None:
    inst = builder()
    res = solve(inst, debug=True)
    assert_valid(res, inst, inst.name)
    check_arborescence(inst, res)
    assert res.stats["backend"] == "core" and res.algorithm == "weighted"
    assert res.stats["k_T_connected"] == check_preconditions(inst, True)["k_T_connected"]
    assert res.stats["roundings"] == 0 and res.stats["contractions"] == inst.n - inst.k
    assert res.stats["min_cost_flow_calls"] >= 1


@pytest.mark.parametrize("name,n,k,inst", FAMILY_GRID, ids=FAMILY_IDS)
def test_family_grid_all_capacity_modes(name: str, n: int, k: int, inst: Instance) -> None:
    res = solve(inst, threads=2)
    assert_valid(res, inst, (name, n, k, "generator capacities"))
    check_arborescence(inst, res)
    if not claims_connected(inst) or inst.is_weighted:
        return  # paper examples are FEAC-only: other capacities may (correctly) fail; weighted DAGs: below
    rng = random.Random(1000 * n + k)
    for mode in CAPACITY_MODES:
        caps = random_capacities(inst.n - inst.k, inst.k, rng, mode)
        inst2 = with_capacities(inst, list(caps))
        res2 = solve(inst2, threads=2)
        assert_valid(res2, inst2, (name, n, k, mode, caps))
        if inst2.k > 1 and 0 in caps:
            assert res2.stats["terminal_removals"] >= 1


def test_feac_only_instances_exact() -> None:
    for inst in feac_only_instances(40, seed=2026):
        res = solve(inst, debug=True)
        assert_valid(res, inst, inst.name)
        assert res.stats["k_T_connected"] is False
        check_arborescence(inst, res)


def test_status_agrees_with_reference_and_general_on_random_unit_instances() -> None:
    """On arbitrary unit-weight digraphs the weighted solver answers ok (valid, exact) or precondition_failed
    exactly as the reference weighted solver and as the unweighted solvers (FESAC = FEAC on unit weights with
    tight capacities, §13.6; the condition is a property of the instance, never of the choices of §13.8)."""
    rng = random.Random(77)
    n_ok = n_fail = 0
    for i in range(200):
        n = rng.randint(3, 11)
        k = rng.randint(1, min(4, n - 1))
        inst = random_digraph(n, k, rng, p=rng.choice([0.15, 0.3, 0.45, 0.7]), mode=rng.choice(CAPACITY_MODES))
        core = solve(inst, debug=True)
        ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
        gen = glpartition(inst, algorithm="general", verify_preconditions=False)
        assert core.status == ref.status == gen.status, (i, core.status, ref.status, gen.status, core.message)
        assert core.status in ("ok", "precondition_failed"), core.message
        if core.status == "ok":
            assert_valid(core, inst, i)
            check_arborescence(inst, core)
            n_ok += 1
        else:
            assert "Flow-Essential Split-Assignment" in core.message
            assert core.parts == [] and core.valid is None
            n_fail += 1
    assert n_ok >= 20 and n_fail >= 20  # both outcomes are exercised


# ---------------------------------------------------------------------------
# 2. weighted variants: bound c_t + w_max - 1, status agreement with the reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w_max", [2, 3, 5, 20])
def test_weighted_variants_valid_and_agree_with_reference(w_max: int) -> None:
    rng = random.Random(100 + w_max)
    bases = weighted_bases(rng, 9)
    roundings = 0
    for i, base in enumerate(bases):
        for slack in (0, 1, 5):
            inst = generators.weighted_variant(base, 1000 * i + w_max + slack, w_max, slack)
            assert inst.is_weighted and inst.w_max <= w_max
            res = solve(inst, threads=2)
            assert_valid(res, inst, (base.name, w_max, slack))
            check_arborescence(inst, res)
            assert res.stats["k_T_connected"] is True
            roundings += res.stats["roundings"]
            if inst.n <= 24:
                ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
                assert ref.status == res.status == "ok", (base.name, w_max, slack, ref.message)
    assert roundings >= 0  # rounding is instance dependent; validity above is the property


def test_weighted_random_digraphs_status_agrees_with_reference() -> None:
    """Weighted variants of arbitrary digraphs: FESAC may fail; both solvers must say so identically."""
    rng = random.Random(4242)
    n_ok = n_fail = 0
    for i in range(150):
        n = rng.randint(3, 11)
        k = rng.randint(1, min(4, n - 1))
        base = random_digraph(n, k, rng, p=rng.choice([0.2, 0.35, 0.5, 0.8]), mode="random")
        inst = generators.weighted_variant(base, rng.randint(0, 10**6), rng.choice([2, 3, 5]), rng.choice([0, 0, 1, 5]))
        core = solve(inst, debug=True, threads=rng.choice([1, 3]))
        ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
        assert core.status == ref.status, (i, core.status, ref.status, core.message)
        if core.status == "ok":
            assert_valid(core, inst, i)
            check_arborescence(inst, core)
            pre = check_preconditions(inst, True)
            assert pre["feac"] is True and core.stats["k_T_connected"] == pre["k_T_connected"]
            n_ok += 1
        else:
            assert check_preconditions(inst, True)["feac"] is False
            n_fail += 1
    assert n_ok >= 20 and n_fail >= 20


# ---------------------------------------------------------------------------
# 3. rounding [Alg 4]
# ---------------------------------------------------------------------------
def random_k_t_connected_digraph(rng: random.Random, n: int, k: int, extra_back: int = 3):
    """As in tests/test_reference_weighted.py: a DAG with out-degree >= k [Lem 9.1] plus backward arcs."""
    verts = list(range(n))
    terminals = rng.sample(verts, k)
    nonterms = [v for v in verts if v not in terminals]
    rng.shuffle(nonterms)
    order = nonterms + terminals
    pos = {v: i for i, v in enumerate(order)}
    arcs: set[tuple[int, int]] = set()
    for i, v in enumerate(nonterms):
        later = order[i + 1:]
        d = rng.randint(k, min(len(later), k + 2))
        for x in rng.sample(later, d):
            arcs.add((v, x))
    for _ in range(extra_back):
        if len(nonterms) >= 2:
            u, v = rng.sample(nonterms, 2)
            if pos[u] > pos[v]:
                arcs.add((u, v))
    return sorted(arcs), terminals


def test_rounding_exercised_by_heavy_vertex_instances() -> None:
    """One vertex of weight w_max adjacent to every terminal must split its weight; RoundAndRemove is then
    unavoidable on some instances.  Every output is valid and the reference agrees on the status."""
    rng = random.Random(2024)
    roundings = 0
    for i in range(60):
        n = rng.randint(4, 8)
        k = rng.randint(2, min(4, n - 1))
        arcs, terminals = random_k_t_connected_digraph(rng, n, k, extra_back=1)
        w_max = rng.randint(3, 6)
        weights = [0 if v in terminals else rng.randint(1, w_max) for v in range(n)]
        heavy = next(v for v in range(n) if v not in terminals)
        weights[heavy] = w_max
        arcs = sorted(set(arcs) | {(heavy, t) for t in terminals})
        caps = random_composition(rng, sum(weights), k)
        inst = make_instance(n, arcs, terminals, caps, weights=weights, directed=True)
        res = solve(inst, debug=True, trace=True)
        assert_valid(res, inst, i)
        check_arborescence(inst, res)
        ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
        assert ref.status == "ok"
        events = [ev for ev in res.trace if ev["type"] == "round_and_remove"]
        assert len(events) == res.stats["roundings"]
        for ev in events:  # [Lem 7.6]: |pairs| = |S| - 1, t_S in S, matched pre-terminals distinct
            assert ev["t_S"] == ev["S"][0] and len(ev["pairs"]) == len(ev["S"]) - 1
            assert {t for t, _p in ev["pairs"]} == set(ev["S"]) - {ev["t_S"]}
            ps = [p for _t, p in ev["pairs"]]
            assert len(set(ps)) == len(ps)
            for t, p in ev["pairs"]:  # each rounded pre-terminal lands in the part of its matched terminal
                assert p in res.parts[inst.terminals.index(t)]
        roundings += res.stats["roundings"]
    assert roundings > 0, "RoundAndRemove never triggered in the random search"


@pytest.mark.parametrize("k", [2, 3, 5])
def test_unavoidable_slack_example(k: int) -> None:
    """k terminals of capacity 1 and one vertex of weight k adjacent to all of them: some part receives
    weight k = c_t + w_max - 1 (paper §5); RoundAndRemove runs exactly once."""
    n = k + 1
    v = k
    arcs = [(v, t) for t in range(k)]
    weights = [0] * k + [k]
    inst = make_instance(n, arcs, list(range(k)), [1] * k, weights=weights, directed=True)
    res = solve(inst, debug=True, trace=True)
    assert_valid(res, inst)
    heavy = [i for i, part in enumerate(res.parts) if v in part]
    assert len(heavy) == 1
    i = heavy[0]
    assert part_weight(inst, res.parts[i]) == k == inst.capacities[i] + inst.w_max - 1
    assert res.stats["roundings"] == 1
    assert res.certificate["parents"] == {v: i}
    rr = next(ev for ev in res.trace if ev["type"] == "round_and_remove")
    assert len(rr["S"]) == 2 and rr["pairs"] == [[i, v]] and rr["t_S"] != i
    rep = verify_instance_parts(inst, res.parts)
    assert rep.details["exceeds_capacity"] == [i]


def test_round_and_remove_with_isolated_terminal() -> None:
    """A terminal without pre-terminals is a singleton Hall-deficient set: rounded away with an empty
    matching, its capacity simply stays unused (Σ w <= Σ c is an inequality, §1.3)."""
    # 3 -> 0, 3 -> 4, 4 -> 0, 4 -> 1; terminal 2 has no in-arc; weights 2, 3; capacities (2, 3, 4)
    inst = make_instance(5, [(3, 0), (3, 4), (4, 0), (4, 1)], [0, 1, 2], [2, 3, 4], weights=[0, 0, 0, 2, 3], directed=True)
    res = solve(inst, debug=True, trace=True)
    assert_valid(res, inst)
    assert res.parts[2] == [2]
    ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
    assert ref.status == "ok" and sorted(map(sorted, ref.parts)) == sorted(map(sorted, res.parts))


# ---------------------------------------------------------------------------
# 4. large weights, debug assertions, determinism
# ---------------------------------------------------------------------------
def test_large_weights_2_pow_40() -> None:
    rng = random.Random(9)
    for seed in range(4):
        base = generators.harary_graph(30, 3, seed=seed) if seed % 2 == 0 else generators.random_kT_connected_dag(30, 3, seed=seed)
        w = [0 if v in base.terminals else rng.randint(1, 2**40) for v in range(base.n)]
        slack = rng.choice([0, 2**39])
        caps = random_composition(rng, sum(w) + slack, base.k)
        edges = base.undirected_edges if not base.directed else base.arcs
        inst = make_instance(base.n, edges, base.terminals, caps, weights=w, directed=base.directed)
        assert inst.w_max > 2**39
        res = solve(inst, debug=True, threads=2)
        assert_valid(res, inst, seed)
        check_arborescence(inst, res)
        for i, part in enumerate(res.parts):
            assert part_weight(inst, part) <= inst.capacities[i] + inst.w_max - 1
    # one vertex of weight 2^40 and two terminals of capacity 2^39 each: rounding with a huge slack
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [2**39, 2**39], weights=[0, 0, 2**40], directed=True)
    res = solve(inst, debug=True)
    assert_valid(res, inst)
    assert res.stats["roundings"] == 1


DEBUG_POOL = small_pool(120, seed=42, n_max=14)


def test_debug_asserts_run_clean_on_120_small_instances() -> None:
    """debug_asserts re-verify after every operation: psi is a split witness with exact essential sets, FESAC
    holds (zero-cost flow over exact sets), Ess/kappa unchanged by contractions (A7), [Lem 7.10] on secondary
    arcs, the primal-dual min-cost flow agrees with successive shortest paths."""
    assert len(DEBUG_POOL) == 120 and all(i.n <= 14 for i in DEBUG_POOL)
    weighted = feac_only = 0
    for inst in DEBUG_POOL:
        res = solve(inst, debug=True, threads=2)
        assert_valid(res, inst, inst.name)
        check_arborescence(inst, res)
        weighted += inst.is_weighted
        feac_only += not res.stats["k_T_connected"]
    assert weighted >= 30 and feac_only >= 25


@pytest.mark.parametrize("greedy,lazy,batch", list(itertools.product([True, False], repeat=3)), ids=lambda x: str(x))
def test_option_matrix(greedy: bool, lazy: bool, batch: bool) -> None:
    opts = {"greedy_contraction": greedy, "lazy_shift": lazy, "batch_unused_arcs": batch}
    for inst in DEBUG_POOL[:60]:
        res = solve(inst, options=opts, threads=2, debug=True)
        assert_valid(res, inst, (inst.name, opts))
        st = res.stats
        if not greedy:
            assert st["greedy_attempts"] == 0 and st["greedy_successes"] == 0
        if not greedy and not batch:
            assert st["batched_deletions"] == 0  # the literal [Alg 3]: single-arc deletions only
        assert st["greedy_successes"] <= st["greedy_attempts"]


DET_POOL = small_pool(24, seed=7, n_max=40)


def test_determinism_threads_1_vs_8() -> None:
    for inst in DET_POOL:
        r1 = solve(inst, threads=1)
        r8 = solve(inst, threads=8)
        assert_valid(r1, inst, inst.name)
        assert r1.parts == r8.parts, inst.name
        assert r1.certificate == r8.certificate, inst.name
        keys = ("steps", "contractions", "deletions", "roundings", "min_cost_flow_calls", "greedy_attempts",
                "greedy_successes", "batched_deletions", "max_flow_calls", "augment_calls", "cut_calls", "terminal_removals")
        assert {k: r1.stats[k] for k in keys} == {k: r8.stats[k] for k in keys}, inst.name
        r1b = solve(inst, threads=1, seed=12345)
        assert r1b.parts == r1.parts


# ---------------------------------------------------------------------------
# 5. precondition failures, binding errors, trace, stats
# ---------------------------------------------------------------------------
def test_precondition_failed_messages() -> None:
    # v reaches only t1 (Ess(v) = {t1}) but c_{t1} = 0: no split witness
    inst = make_instance(3, [(2, 0)], [0, 1], [0, 1], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed" and res.parts == [] and res.valid is None
    assert "Split-Assignment" in res.message and "guarantee does not apply" in res.message
    assert "no partition" not in res.message.lower().replace("a partition may still exist", "")
    # a vertex that reaches no terminal at all (kappa = 0, Ess = ∅) is named [§13.9]
    inst = make_instance(4, [(2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed"
    assert "no essential terminal" in res.message and "[3]" in res.message
    # total weight fits but only into a non-essential terminal
    inst = make_instance(3, [(2, 0)], [0, 1], [1, 5], weights=[0, 0, 3], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed" and "Split-Assignment" in res.message
    assert glpartition(inst, algorithm="reference-weighted").status == "precondition_failed"
    # the §13.3 example: FESAC fails although a partition exists; never claim "no partition exists"
    v, x, t1, t2, t3 = 3, 4, 0, 1, 2
    inst = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [0, 2, 0], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed"
    assert verify_instance_parts(inst, [[t1], [t2, v, x], [t3]]).valid
    inst2 = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [1, 1, 0], directed=True)
    res2 = solve(inst2, debug=True)
    assert_valid(res2, inst2)
    assert res2.parts == [[t1, v], [t2, x], [t3]]


def test_core_reports_errors_as_status_never_raises() -> None:
    out = _core.solve_weighted(3, [(0, 1), (1, 2)], [], [], [1, 1, 1], {})
    assert out["status"] == "error" and "terminal" in out["message"]
    out = _core.solve_weighted(3, [(0, 1), (1, 2)], [2], [1, 1], [1, 1, 0], {})
    assert out["status"] == "error" and "capacities" in out["message"]
    out = _core.solve_weighted(3, [(0, 1), (1, 2)], [2], [2], [1, 0, 0], {})
    assert out["status"] == "error" and "weight" in out["message"]
    out = _core.solve_weighted(3, [(0, 1), (1, 2)], [2], [1], [1, 1, 0], {})
    assert out["status"] == "precondition_failed" and "sum" in out["message"]
    out = _core.solve_weighted(3, [(0, 1), (1, 2)], [2], [2], [1, 1, 0], {"threads": 2})
    assert out["status"] == "ok" and out["assignment"] == [0, 0, 0]
    assert out["parent"] == [1, 2, -1] and out["witness"] == [-1, -1, -1] and out["k_T_connected"] is True
    assert set(out["stats"]) >= {"steps", "contractions", "deletions", "roundings", "min_cost_flow_calls",
                                 "time_seconds", "graph_size_over_time"}
    # n == k: nothing to do
    out = _core.solve_weighted(2, [], [0, 1], [3, 0], [0, 0], {})
    assert out["status"] == "ok" and out["assignment"] == [0, 1] and out["parent"] == [-1, -1]


def _split_witness_by_definition(inst: Instance, psi: list[list[int]]) -> None:
    """[Def 5.3] against the by-definition essential sets of the original instance."""
    ess = essential_sets_by_definition(inst)
    w = inst.weights or tuple(0 if v in inst.terminals else 1 for v in range(inst.n))
    sent = {v: 0 for v in ess}
    received = {t: 0 for t in inst.terminals}
    for v, t, units in psi:
        assert units > 0 and t in ess[v], (v, t, units, ess[v])
        sent[v] += units
        received[t] += units
    assert all(sent[v] == w[v] for v in ess)
    assert all(received[t] <= c for t, c in zip(inst.terminals, inst.capacities))


def test_trace_running_example_literal_mode() -> None:
    inst = generators.paper_running_example()
    res = solve(inst, trace=True, debug=True, options={**LITERAL, "record_cuts": True})
    assert_valid(res, inst)
    tr = res.trace
    assert tr is not None and tr[0]["type"] == "init" and tr[-1]["type"] == "done"
    types = [ev["type"] for ev in tr]
    assert set(types) <= TRACE_TYPES, set(types) - TRACE_TYPES
    assert {"init", "essential", "min_cost_split", "contract", "matching", "criticality", "delete_arc",
            "remove_terminal", "done"} <= set(types)
    assert "greedy_delete" not in types and "delete_arcs" not in types
    init = tr[0]
    assert init["n"] == inst.n and init["k"] == inst.k and init["terminals"] == list(inst.terminals)
    assert sorted(map(tuple, init["arcs"])) == sorted(inst.arcs) and init["capacities"] == list(inst.capacities)
    assert init["weights"] == [0 if v in inst.terminals else 1 for v in range(inst.n)]
    # first "essential" event: exact sets [Def 4.1] and tightest cuts [Prop 4.2]
    ess_ev = next(ev for ev in tr if ev["type"] == "essential")
    by_def = essential_sets_by_definition(inst)
    assert {int(v): set(ts) for v, ts in ess_ev["ess"].items()} == by_def
    for v, cut in ess_ev["cuts"].items():
        assert sorted(cut["L"] + cut["S"] + cut["R"]) == list(range(inst.n))
        assert int(v) in cut["R"] and len(cut["S"]) == ess_ev["kappa"][v]
    # the initial split witness [Def 5.3] with cost 0
    first_split = next(ev for ev in tr if ev["type"] == "min_cost_split")
    assert first_split["cost"] == 0
    _split_witness_by_definition(inst, first_split["psi"])
    # literal mode: every step (iii) is matching -> criticality -> min_cost_split -> delete_arc
    splits = sum(1 for t in types if t == "min_cost_split")
    assert splits == res.stats["min_cost_flow_calls"] == 1 + res.stats["deletions"]
    for i, ev in enumerate(tr):
        if ev["type"] == "matching":
            assert types[i + 1] == "criticality" and types[i + 2] == "min_cost_split" and types[i + 3] == "delete_arc"
            ps = [p for p, _t in ev["pairs"]]
            ts = [t for _p, t in ev["pairs"]]
            assert len(set(ps)) == len(ps) and len(set(ts)) == len(ts)
            assert [p for p, _q in ev["secondary"]] == ps
            assert all((p, q) != (p, t) for (p, q), (_p, t) in zip(ev["secondary"], ev["pairs"]))
            # the deleted arc is one of the secondary arcs; the potential is the number of critical entries
            d = tr[i + 3]
            assert [d["u"], d["v"]] in [list(s) for s in ev["secondary"]]
            crit = {(c[0], c[1], c[2]) for c in tr[i + 1]["crit"]}
            split = tr[i + 2]
            sec_index = {tuple(s): j for j, s in enumerate(ev["secondary"])}
            j = sec_index[(d["u"], d["v"])]
            assert not any((j, v, t) in crit for v, t, _u in split["psi"]), "deleted arc critical for a positive pair"
            assert split["cost"] == sum(units * sum(1 for jj in range(len(ev["pairs"])) if (jj, v, t) in crit)
                                        for v, t, units in split["psi"])
        if ev["type"] == "contract":
            assert (ev["p"], ev["parent"]) in set(inst.arcs)
            assert set(map(int, ev["capacities"])) <= set(inst.terminals)
    done = tr[-1]
    assert [sorted(done["parts"][str(t)]) for t in inst.terminals] == [sorted(p) for p in res.parts]
    assert {int(v): p for v, p in done["parents"].items()} == res.certificate["parents"]
    assert res.stats["contractions"] == sum(1 for t in types if t == "contract")
    assert res.stats["deletions"] == sum(1 for t in types if t == "delete_arc")


def test_trace_lazy_mode_events_and_payloads() -> None:
    inst = generators.paper_essential_example()
    out = _core.solve_weighted(inst.n, [list(a) for a in inst.arcs], list(inst.terminals), list(inst.capacities),
                               [1] * inst.n, {"trace": True})
    assert out["status"] == "ok"
    types = []
    for typ, payload in out["trace"]:
        assert typ in TRACE_TYPES
        d = json.loads(payload)
        assert isinstance(d, dict) and "type" not in d
        types.append(typ)
    assert types[0] == "init" and types[-1] == "done"
    assert sum(1 for t in types if t == "min_cost_split") == out["stats"]["min_cost_flow_calls"]
    assert out["stats"]["deletions"] == sum(1 if t == "delete_arc" else len(json.loads(p)["arcs"])
                                            for t, p in out["trace"] if t in ("delete_arc", "delete_arcs", "greedy_delete"))
    out2 = _core.solve_weighted(inst.n, [list(a) for a in inst.arcs], list(inst.terminals), list(inst.capacities), [1] * inst.n, {})
    assert out2["trace"] == [] and out2["assignment"] == out["assignment"]


def test_stats_and_time_keys() -> None:
    rng = random.Random(11)
    base = generators.random_regular_graph(60, 6, 4, seed=3)
    caps = random_capacities(base.n - base.k, base.k, rng, "zeros")
    inst = generators.weighted_variant(with_capacities(base, caps), 5, 4, 2)
    res = solve(inst, threads=4)
    assert_valid(res, inst)
    st = res.stats
    for key in ("roundings", "min_cost_flow_calls", "contractions", "deletions", "terminal_removals", "matching_calls",
                "max_flow_calls", "augment_calls", "cut_calls", "steps", "greedy_attempts", "greedy_successes",
                "batched_deletions", "graph_size_over_time", "time_seconds", "k_T_connected", "backend"):
        assert key in st, key
    times = st["time_seconds"]
    assert {"total", "compute_all", "min_cost_flow", "contract", "matching"} <= set(times), sorted(times)
    assert times["total"] > 0 and all(v >= 0 for v in times.values())
    assert st["min_cost_flow_calls"] >= 1 and st["contractions"] >= 1
    sizes = st["graph_size_over_time"]
    assert sizes[0] == (inst.n - inst.k, inst.m) and len(sizes) == st["steps"]
    assert all(a >= b and ma >= mb for (a, ma), (b, mb) in zip(sizes, sizes[1:]))
    # the heavy-vertex slack example exercises rounding / terminal removal timers
    inst2 = make_instance(4, [(3, 0), (3, 1), (3, 2)], [0, 1, 2], [1, 1, 1], weights=[0, 0, 0, 3], directed=True)
    res2 = solve(inst2)
    assert_valid(res2, inst2)
    assert "rounding" in res2.stats["time_seconds"] and res2.stats["roundings"] == 1
    inst3 = make_instance(4, [(2, 0), (3, 0), (3, 2)], [0, 1], [2, 0], directed=True)
    res3 = solve(inst3)
    assert_valid(res3, inst3)
    assert "terminal_removal" in res3.stats["time_seconds"] and res3.stats["terminal_removals"] == 1


# ---------------------------------------------------------------------------
# 6. scaling smoke (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("label,builder", [
    ("harary(2000,8) unit", lambda: generators.harary_graph(2000, 8, seed=1)),
    ("harary(2000,8) w_max=5", lambda: generators.weighted_variant(generators.harary_graph(2000, 8, seed=1), 3, 5, 0)),
])
def test_scaling_smoke(label: str, builder) -> None:
    inst = builder()
    t0 = time.perf_counter()
    res = solve(inst, threads=8)
    dt = time.perf_counter() - t0
    assert_valid(res, inst, label)
    check_arborescence(inst, res)
    keys = ("steps", "contractions", "deletions", "greedy_successes", "batched_deletions", "roundings",
            "min_cost_flow_calls", "max_flow_calls")
    print(f"\n{label}: n={inst.n} m={inst.m} {dt:.2f}s " + " ".join(f"{k}={res.stats[k]}" for k in keys))
    if not inst.is_weighted:
        assert res.stats["contractions"] == inst.n - inst.k
    sizes = res.stats["graph_size_over_time"]
    assert 0 < len(sizes) <= 10000 and sizes[0][0] == inst.n - inst.k
    assert all(a >= b for (a, _), (b, _) in zip(sizes, sizes[1:]))
    assert dt < 120.0, dt


def test_medium_scaling_fast() -> None:
    """A medium instance in the default test run: unit and weighted, with the arborescence checked."""
    base = generators.harary_graph(300, 6, seed=2)
    for inst in (base, generators.weighted_variant(base, 4, 5, 3)):
        t0 = time.perf_counter()
        res = solve(inst, threads=4)
        dt = time.perf_counter() - t0
        assert_valid(res, inst, inst.name)
        check_arborescence(inst, res)
        assert dt < 60.0, dt
