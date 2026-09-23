"""Tests of the C++ unweighted general solver (``glsolver._core.solve_general`` via
``glpartition(..., algorithm="general")``): GLPartition [Alg 1] + ShiftAssignment [Alg 2] with the exact
optimizations O1–O9 of docs/optimizations.md.

Every returned partition is checked by the independent verifier ``glsolver.verify.verify_instance_parts``
(exact part sizes ``c_i + 1``, connectivity to the terminal inside the part on the original input) and,
where the certificate is inspected, the in-arborescence ``parents`` is checked against the *original* arcs.
Statuses (``ok`` vs ``precondition_failed``) must agree with the literal reference ``glref.unweighted`` on
random small instances; the reference is never asked for the same *partition* (all choices of §13.8 are
free).  All randomness is seeded.  Runnable in isolation: ``pytest tests/test_core_solver.py``; the
counterexample with 17 copies and the scaling smoke are ``@pytest.mark.slow`` (``--runslow``).
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
from helpers_small_graphs import random_capacities, random_digraph

pytestmark = pytest.mark.skipif(not core_available("general"), reason="C++ core (solve_general) not built")

CAPACITY_MODES = ("balanced", "unbalanced", "random", "extreme", "zeros")
TRACE_TYPES = {
    "init", "essential", "witness", "remove_terminal", "contract", "delete_arc", "delete_arcs",
    "greedy_delete", "matching", "criticality", "reassignment_graph", "cycle_shift", "done",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def solve(inst: Instance, **kw: Any) -> GLResult:
    kw.setdefault("verify_preconditions", False)
    return glpartition(inst, algorithm="general", **kw)


def assert_valid(res: GLResult, inst: Instance, where: Any = "") -> None:
    assert res.status == "ok", (where, res.status, res.message)
    rep = verify_instance_parts(inst, res.parts)
    assert rep.valid, (where, rep.errors[:5])
    assert res.valid is True
    assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities]
    assert sorted(v for p in res.parts for v in p) == list(range(inst.n))


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


def with_capacities(inst: Instance, caps: list[int]) -> Instance:
    edges = inst.arcs if inst.directed else inst.undirected_edges
    return make_instance(inst.n, edges, inst.terminals, caps, directed=inst.directed,
                         name=f"{inst.name}_caps", meta=dict(inst.meta))


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
                if inst.is_weighted:
                    continue
                out.append((name, n, k, inst))
    return out


FAMILY_GRID = _family_grid()
FAMILY_IDS = [f"{name}-n{n}-k{k}" for name, n, k, _ in FAMILY_GRID]


def feac_only_instances(count: int, seed: int, n_max: int = 12) -> list[Instance]:
    """Random digraphs satisfying FEAC but not k-T-connected (Theorem 2 beyond Theorem 1), filtered with the
    NetworkX-based cross-check."""
    rng = random.Random(seed)
    found: list[Instance] = []
    tries = 0
    while len(found) < count and tries < 50 * count:
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


def small_pool(count: int, seed: int, n_max: int = 14) -> list[Instance]:
    """Mixed small instances: generator families, directed variants and FEAC-only digraphs."""
    rng = random.Random(seed)
    cat = family_catalog()
    names = [n for n in cat if not n.startswith("paper") and n != "weighted_kT_dag"]
    pool: list[Instance] = [generators.paper_running_example(), generators.paper_essential_example(),
                            generators.paper_contract_counterexample()]
    while len(pool) < count - count // 3:
        name = rng.choice(names)
        n = rng.randint(5, n_max)
        k = rng.randint(1, min(4, n - 1))
        try:
            inst = cat[name](n, k, rng.randint(0, 10**6))
        except ValueError:
            continue
        if inst.is_weighted or inst.n > n_max:
            continue
        if not inst.directed and rng.random() < 0.3:
            inst = generators.directed_variant(inst, rng.randint(0, 10**6), 0.5)
        elif claims_connected(inst) and rng.random() < 0.5:
            caps = random_capacities(inst.n - inst.k, inst.k, rng, rng.choice(CAPACITY_MODES))
            inst = with_capacities(inst, list(caps))
        pool.append(inst)
    pool += feac_only_instances(count - len(pool), seed + 1, n_max=min(n_max, 12))
    return pool


# ---------------------------------------------------------------------------
# 1. paper examples
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [generators.paper_running_example, generators.paper_essential_example,
                                     generators.paper_contract_counterexample])
def test_paper_examples_valid(builder) -> None:
    inst = builder()
    res = solve(inst, debug=True)
    assert_valid(res, inst, inst.name)
    check_arborescence(inst, res)
    assert res.stats["backend"] == "core" and res.algorithm == "general"
    assert res.stats["k_T_connected"] == check_preconditions(inst, True)["k_T_connected"]
    # the initial witness is a FEAC witness [Def 5.1]: essential terminal, exact capacity counts
    ess = essential_sets_by_definition(inst)
    witness = res.certificate["witness"]
    assert set(witness) == set(ess)
    counts = {t: 0 for t in inst.terminals}
    for v, t in witness.items():
        assert t in ess[v], (v, t, ess[v])
        counts[t] += 1
    assert [counts[t] for t in inst.terminals] == list(inst.capacities)


def test_paper_running_example_is_feac_only() -> None:
    inst = generators.paper_running_example()
    res = solve(inst)
    assert res.status == "ok" and res.stats["k_T_connected"] is False


# ---------------------------------------------------------------------------
# 2. validity on every generator family, capacity modes, directed variants, FEAC-only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,n,k,inst", FAMILY_GRID, ids=FAMILY_IDS)
def test_family_grid_all_capacity_modes(name: str, n: int, k: int, inst: Instance) -> None:
    res = solve(inst, threads=2)
    assert_valid(res, inst, (name, n, k, "generator capacities"))
    check_arborescence(inst, res)
    if not claims_connected(inst):
        return  # paper examples are FEAC-only: other capacities may (correctly) fail FEAC
    rng = random.Random(1000 * n + k)
    for mode in CAPACITY_MODES:
        caps = list(random_capacities(inst.n - inst.k, inst.k, rng, mode))
        inst2 = with_capacities(inst, caps)
        res2 = solve(inst2, threads=2)
        assert_valid(res2, inst2, (name, n, k, mode, caps))
        if inst2.k > 1 and 0 in caps:
            assert res2.stats["terminal_removals"] >= 1


@pytest.mark.parametrize("seed", range(6))
def test_directed_variants(seed: int) -> None:
    rng = random.Random(seed)
    base_gens = [generators.harary_graph, generators.random_regular_graph, generators.erdos_renyi_graph]
    for i in range(4):
        n = rng.choice([12, 24, 40])
        k = rng.choice([2, 3, 4])
        g = base_gens[(seed + i) % len(base_gens)]
        base = g(n, 6, k, seed=rng.randint(0, 10**6)) if g is generators.random_regular_graph else (
            g(n, 0.4, k, seed=rng.randint(0, 10**6)) if g is generators.erdos_renyi_graph else g(n, k, seed=rng.randint(0, 10**6)))
        inst = generators.directed_variant(base, rng.randint(0, 10**6), rng.choice([0.3, 0.6, 0.9]))
        assert inst.directed and inst.meta["claims_kT_connected"]
        for mode in ("balanced", "random", "zeros"):
            caps = list(random_capacities(inst.n - inst.k, inst.k, rng, mode))
            inst2 = with_capacities(inst, caps)
            res = solve(inst2, threads=3)
            assert_valid(res, inst2, (seed, i, mode))
            assert res.stats["k_T_connected"] is True


def test_feac_only_instances_valid() -> None:
    for inst in feac_only_instances(40, seed=2026):
        res = solve(inst, debug=True)
        assert_valid(res, inst, inst.name)
        assert res.stats["k_T_connected"] is False
        check_arborescence(inst, res)


def test_status_agrees_with_reference_on_random_small_instances() -> None:
    """On arbitrary digraphs the solver answers ok (valid) or precondition_failed, exactly as the reference
    (FEAC is a property of the instance; the paper's guarantee never depends on the choices of §13.8)."""
    rng = random.Random(77)
    n_ok = n_fail = 0
    for i in range(200):
        n = rng.randint(3, 11)
        k = rng.randint(1, min(4, n - 1))
        inst = random_digraph(n, k, rng, p=rng.choice([0.15, 0.3, 0.45, 0.7]), mode=rng.choice(CAPACITY_MODES))
        core = solve(inst, debug=True)
        ref = glpartition(inst, algorithm="reference", verify_preconditions=False)
        assert core.status == ref.status, (i, core.status, ref.status, core.message)
        assert core.status in ("ok", "precondition_failed"), core.message
        if core.status == "ok":
            assert_valid(core, inst, i)
            n_ok += 1
        else:
            assert "Flow-Essential" in core.message
            assert core.parts == [] and core.valid is None
            n_fail += 1
    assert n_ok >= 20 and n_fail >= 20  # both outcomes are exercised


def test_precondition_failed_messages() -> None:
    # vertex 3 cannot reach any terminal (§13.9): named in the message; never "no partition exists"
    inst = make_instance(4, [(2, 0), (2, 1), (3, 2)], [0, 1], [1, 1], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed"
    assert "no essential terminal" in res.message and "3" in res.message
    assert "guarantee does not apply" in res.message
    assert "no partition" not in res.message.lower().replace("a partition may still exist", "")
    # §13.3 example with c(t1) = 0: Ess(v) = {t1}, so no witness exists although {t1}, {t2, v, x}, {t3} is a
    # valid partition; the solver must say precondition_failed (never "no solution")
    v, x, t1, t2, t3 = 3, 4, 0, 1, 2
    inst = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [0, 2, 0], directed=True)
    res = solve(inst)
    assert res.status == "precondition_failed"
    assert "Flow-Essential" in res.message and "no essential terminal" not in res.message
    assert verify_instance_parts(inst, [[t1], [t2, v, x], [t3]]).valid
    # with c = (1, 1, 0) FEAC holds
    inst2 = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [1, 1, 0], directed=True)
    res2 = solve(inst2, debug=True)
    assert_valid(res2, inst2)
    assert res2.parts == [[t1, v], [t2, x], [t3]]


def test_core_reports_errors_as_status_never_raises() -> None:
    out = _core.solve_general(3, [(0, 1), (1, 2)], [], [], {})
    assert out["status"] == "error" and "terminal" in out["message"]
    out = _core.solve_general(3, [(0, 1), (1, 2)], [2], [1, 1], {})
    assert out["status"] == "error" and "capacities" in out["message"]
    out = _core.solve_general(3, [(0, 1), (1, 2)], [2], [1], {})
    assert out["status"] == "precondition_failed" and "sum" in out["message"]
    out = _core.solve_general(3, [(0, 1), (1, 2)], [2], [2], {"threads": 2})
    assert out["status"] == "ok" and out["assignment"] == [0, 0, 0]
    assert out["parent"] == [1, 2, -1] and out["witness"] == [0, 0, -1] and out["k_T_connected"] is True
    assert set(out["stats"]) >= {"steps", "contractions", "deletions", "time_seconds", "graph_size_over_time"}


# ---------------------------------------------------------------------------
# 3. option matrix
# ---------------------------------------------------------------------------
OPTION_POOL = small_pool(60, seed=31, n_max=24)


@pytest.mark.parametrize("greedy,lazy,batch", list(itertools.product([True, False], repeat=3)),
                         ids=lambda x: str(x))
def test_option_matrix(greedy: bool, lazy: bool, batch: bool) -> None:
    opts = {"greedy_contraction": greedy, "lazy_shift": lazy, "batch_unused_arcs": batch}
    for inst in OPTION_POOL:
        res = solve(inst, options=opts, threads=2)
        assert_valid(res, inst, (inst.name, opts))
        st = res.stats
        if not greedy:
            assert st["greedy_attempts"] == 0 and st["greedy_successes"] == 0
        if not greedy and not batch:
            # literal paper algorithm: only steps (i), (ii) and single-arc deletions of [Alg 2]
            assert st["batched_deletions"] == 0
            assert st["deletions"] == st["shift_calls"]
        assert st["greedy_successes"] <= st["greedy_attempts"]
        assert st["contractions"] == inst.n - inst.k


def test_literal_mode_matches_reference_counters_shape() -> None:
    """With all optimizations off every deletion is one ShiftAssignment call, every contraction removes one
    vertex; the reference's steps also equal removals + contractions + deletions."""
    inst = generators.paper_essential_example()
    res = solve(inst, options={"greedy_contraction": False, "lazy_shift": False, "batch_unused_arcs": False})
    assert_valid(res, inst)
    st = res.stats
    assert st["steps"] == st["contractions"] + st["deletions"] + st["terminal_removals"]
    assert st["shift_calls"] == st["deletions"]


# ---------------------------------------------------------------------------
# 4. debug assertions (A1–A8) on small instances
# ---------------------------------------------------------------------------
DEBUG_POOL = small_pool(150, seed=42, n_max=14)


def test_debug_asserts_run_clean() -> None:
    assert len(DEBUG_POOL) == 150 and all(i.n <= 14 for i in DEBUG_POOL)
    feac_only = 0
    for inst in DEBUG_POOL:
        res = solve(inst, debug=True, threads=2)
        assert_valid(res, inst, inst.name)
        check_arborescence(inst, res)
        feac_only += not res.stats["k_T_connected"]
    assert feac_only >= 40


# ---------------------------------------------------------------------------
# 5. determinism
# ---------------------------------------------------------------------------
DET_POOL = small_pool(30, seed=7, n_max=40)


def test_determinism_threads_1_vs_8() -> None:
    for inst in DET_POOL:
        r1 = solve(inst, threads=1)
        r8 = solve(inst, threads=8)
        assert_valid(r1, inst, inst.name)
        assert r1.parts == r8.parts, inst.name
        assert r1.certificate == r8.certificate, inst.name
        keys = ("steps", "contractions", "deletions", "cycle_shifts", "greedy_attempts", "greedy_successes",
                "batched_deletions", "shift_calls", "max_flow_calls", "augment_calls", "cut_calls")
        assert {k: r1.stats[k] for k in keys} == {k: r8.stats[k] for k in keys}, inst.name
        r1b = solve(inst, threads=1)
        assert r1b.parts == r1.parts


def test_seed_changes_never_change_validity() -> None:
    for inst in DET_POOL[:12]:
        for seed in (1, 12345, 2**31 - 1):
            res = solve(inst, seed=seed, threads=4)
            assert_valid(res, inst, (inst.name, seed))


# ---------------------------------------------------------------------------
# 6. trace
# ---------------------------------------------------------------------------
def test_trace_running_example() -> None:
    inst = generators.paper_running_example()
    res = solve(inst, trace=True, debug=True, options={"record_cuts": True})
    assert_valid(res, inst)
    tr = res.trace
    assert tr is not None and tr[0]["type"] == "init" and tr[-1]["type"] == "done"
    types = {ev["type"] for ev in tr}
    assert types <= TRACE_TYPES, types - TRACE_TYPES
    assert {"init", "essential", "witness", "contract", "done"} <= types
    init = tr[0]
    assert init["n"] == inst.n and init["k"] == inst.k and init["terminals"] == list(inst.terminals)
    assert sorted(map(tuple, init["arcs"])) == sorted(inst.arcs) and init["capacities"] == list(inst.capacities)
    # first "essential" event carries the exact sets [Def 4.1] (terminal vertex ids) and cuts
    ess_ev = next(ev for ev in tr if ev["type"] == "essential")
    by_def = essential_sets_by_definition(inst)
    assert {int(v): set(ts) for v, ts in ess_ev["ess"].items()} == by_def
    assert set(map(int, ess_ev["kappa"])) == set(by_def)
    assert set(map(int, ess_ev["cuts"])) == set(by_def)
    for v, cut in ess_ev["cuts"].items():
        assert sorted(cut["L"] + cut["S"] + cut["R"]) == list(range(inst.n))
        assert int(v) in cut["R"] and len(cut["S"]) == ess_ev["kappa"][v]
        assert set(inst.terminals) <= set(cut["L"] + cut["S"])
    # witness maps every non-terminal to an essential terminal
    wit = next(ev for ev in tr if ev["type"] == "witness")["phi"]
    assert all(t in by_def[int(v)] for v, t in wit.items()) and len(wit) == inst.n - inst.k
    # contract events carry parent + capacities keyed by terminal vertex; potentials decrease in shifts
    for ev in tr:
        if ev["type"] == "contract":
            assert (ev["p"], ev["parent"]) in set(inst.arcs) or ev["parent"] == ev["t"]
            assert (ev["p"], ev["parent"]) in set(inst.arcs)
            assert set(map(int, ev["capacities"])) <= set(inst.terminals)
        if ev["type"] == "cycle_shift":
            assert ev["potential_after"] < ev["potential_before"]
            assert all(old != new for _v, old, new in ev["changes"])
    done = tr[-1]
    parts = [sorted(done["parts"][str(t)]) for t in inst.terminals]
    assert parts == [sorted(p) for p in res.parts]
    assert {int(v): p for v, p in done["parents"].items()} == res.certificate["parents"]
    assert res.stats["contractions"] == sum(1 for ev in tr if ev["type"] == "contract")
    assert res.stats["deletions"] == sum(1 if ev["type"] == "delete_arc" else len(ev["arcs"])
                                         for ev in tr if ev["type"] in ("delete_arc", "delete_arcs", "greedy_delete"))


def test_trace_payloads_are_json_objects_without_type_key() -> None:
    inst = generators.paper_essential_example()
    out = _core.solve_general(inst.n, [list(a) for a in inst.arcs], list(inst.terminals), list(inst.capacities),
                              {"trace": True})
    assert out["status"] == "ok"
    for typ, payload in out["trace"]:
        assert typ in TRACE_TYPES
        d = json.loads(payload)
        assert isinstance(d, dict) and "type" not in d
    out2 = _core.solve_general(inst.n, [list(a) for a in inst.arcs], list(inst.terminals), list(inst.capacities), {})
    assert out2["trace"] == [] and out2["assignment"] == out["assignment"]


def test_trace_shift_events_are_consistent() -> None:
    """Force ShiftAssignment (optimizations off) and check the [Alg 2] events: pairs are arcs into distinct
    terminals, secondary arcs leave the matched pre-terminal, the reassignment cycle has length >= 2."""
    for inst in (generators.paper_running_example(), generators.paper_essential_example()):
        res = solve(inst, trace=True, debug=True,
                    options={"greedy_contraction": False, "lazy_shift": False, "batch_unused_arcs": False})
        assert_valid(res, inst)
        tr = res.trace
        assert any(ev["type"] == "matching" for ev in tr)
        for ev in tr:
            if ev["type"] == "matching":
                ps = [p for p, _t in ev["pairs"]]
                ts = [t for _p, t in ev["pairs"]]
                assert len(set(ps)) == len(ps) and len(set(ts)) == len(ts)
                assert [p for p, _q in ev["secondary"]] == ps
                assert all((p, q) != (p, t) for (p, q), (_p, t) in zip(ev["secondary"], ev["pairs"]))
            if ev["type"] == "reassignment_graph":
                assert len(ev["cycle"]) >= 2 and len(set(ev["cycle"])) == len(ev["cycle"])
                heads = [b for _a, b, _v in ev["arcs"]]
                assert len(set(heads)) == len(heads)
        assert res.stats["shift_calls"] == sum(1 for ev in tr if ev["type"] == "matching")


# ---------------------------------------------------------------------------
# 7. the official counterexample
# ---------------------------------------------------------------------------
def test_counterexample_copies1_under_20s() -> None:
    from glref.counterexample import build_counterexample_instance

    inst = build_counterexample_instance(1)
    assert (inst.n, inst.k) == (333, 9)
    t0 = time.perf_counter()
    res = solve(inst, threads=8)
    dt = time.perf_counter() - t0
    assert_valid(res, inst, "counterexample copies=1")
    check_arborescence(inst, res)
    assert dt < 20.0, dt
    assert res.stats["k_T_connected"] is False and res.stats["shift_calls"] >= 1


@pytest.mark.slow
def test_counterexample_copies17() -> None:
    from glref.counterexample import build_counterexample_instance

    inst = build_counterexample_instance(17)
    assert inst.n == 3789
    t0 = time.perf_counter()
    res = solve(inst, threads=8)
    dt = time.perf_counter() - t0
    assert_valid(res, inst, "counterexample copies=17")
    check_arborescence(inst, res)
    print(f"\ncounterexample copies=17: {dt:.2f}s stats={ {k: res.stats[k] for k in ('steps', 'contractions', 'deletions', 'cycle_shifts', 'greedy_successes', 'max_flow_calls')} }")
    assert dt < 300.0, dt


# ---------------------------------------------------------------------------
# 8. scaling smoke (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("label,builder", [
    ("harary(2000,8)", lambda: generators.harary_graph(2000, 8, seed=1)),
    ("random_regular(5000,d=8,k=8)", lambda: generators.random_regular_graph(5000, 8, 8, seed=2)),
    ("erdos_renyi(2000,k=6)", lambda: generators.erdos_renyi_graph(2000, generators._er_p(2000, 6), 6, seed=3)),
])
def test_scaling_smoke(label: str, builder) -> None:
    inst = builder()
    t0 = time.perf_counter()
    res = solve(inst, threads=8)
    dt = time.perf_counter() - t0
    assert_valid(res, inst, label)
    keys = ("contractions", "deletions", "cycle_shifts", "greedy_successes", "max_flow_calls", "steps")
    print(f"\n{label}: n={inst.n} m={inst.m} {dt:.2f}s " + " ".join(f"{k}={res.stats[k]}" for k in keys))
    assert res.stats["contractions"] == inst.n - inst.k
    sizes = res.stats["graph_size_over_time"]
    assert 0 < len(sizes) <= 10000 and sizes[0][0] == inst.n - inst.k
    assert all(a >= b for (a, _), (b, _) in zip(sizes, sizes[1:]))


def test_graph_size_series_is_monotone_and_bounded() -> None:
    inst = generators.random_regular_graph(120, 6, 4, seed=9)
    res = solve(inst, threads=4)
    assert_valid(res, inst)
    sizes = res.stats["graph_size_over_time"]
    assert sizes[0] == (inst.n - inst.k, inst.m) and len(sizes) == res.stats["steps"]
    assert all(a >= b and ma >= mb for (a, ma), (b, mb) in zip(sizes, sizes[1:]))
    times = res.stats["time_seconds"]
    assert {"total", "compute_all", "witness"} <= set(times) and times["total"] > 0


# ---------------------------------------------------------------------------
# 9. review fixes: malformed options, non-ok result shape, P2 subset commit after a cycle shift
# ---------------------------------------------------------------------------
BAD_OPTION_VALUES = [
    ("threads", "x"), ("threads", None), ("threads", 1.5), ("threads", 2**40), ("seed", [1]),
    ("trace", "x"), ("debug_asserts", object()), ("greedy_contraction", {}),
    ("routing", "nonsense"), ("routing", 5), ("routing", None),
]


def test_malformed_options_are_status_error_never_raised() -> None:
    """The binding's contract: a malformed `options` value comes back as status 'error' naming the key (like every
    other invalid input; never a Python exception), while well-typed coercions (bool as int, int as bool, None as
    False) and unknown keys keep working."""
    for key, value in BAD_OPTION_VALUES:
        out = _core.solve_general(3, [(2, 0), (2, 1)], [0, 1], [1, 0], {key: value})
        assert out["status"] == "error" and f"option '{key}'" in out["message"], (key, value, out["message"])
        assert out["assignment"] == [] and out["parent"] == [] and out["witness"] == []
    out = _core.solve_general(3, [(2, 0), (2, 1)], [0, 1], [1, 0],
                              {"threads": True, "trace": 1, "debug_asserts": None, "unknown_key": "ignored"})
    assert out["status"] == "ok" and out["assignment"] == [0, 1, 0] and out["trace"]


def test_non_ok_results_have_empty_vectors() -> None:
    """assignment / parent / witness are filled (size n) only on 'ok' and are empty lists on precondition_failed
    and error (the general and the weighted binding use the same convention)."""
    for n, arcs, terms, caps in [(3, [(2, 0)], [0, 1], [0, 1]), (3, [(2, 0)], [0, 1], [1, 1])]:
        out = _core.solve_general(n, arcs, terms, caps, {})
        assert out["status"] == "precondition_failed", out
        assert out["assignment"] == [] and out["parent"] == [] and out["witness"] == [], out
    out = _core.solve_general(3, [(0, 5)], [1], [2], {})
    assert out["status"] == "error" and out["assignment"] == [] and out["parent"] == [] and out["witness"] == []
    ok = _core.solve_general(3, [(2, 0)], [0, 1], [1, 0], {})
    assert ok["status"] == "ok" and ok["assignment"] == [0, 1, 0] and ok["parent"] == [-1, -1, 0] and ok["witness"] == [-1, -1, 0]


# Found by fuzzing (random digraph, n = 28, k = 6, unbalanced capacities): under greedy_contraction=True,
# lazy_shift=False, batch_unused_arcs=False the third cycle shift moves phi(21) onto a terminal outside the stale
# certified subset E(21) = {t_0}, and the deletion of a secondary arc whose evaluation had restored kappa(21)
# then committed that pre-shift subset.
P2_STALE_SUBSET_CASE = {
    "n": 28,
    "terminals": [20, 14, 0, 12, 22, 7],
    "capacities": [0, 2, 14, 5, 0, 1],
    "arcs": [
        (1, 10), (1, 12), (1, 14), (1, 15), (1, 23), (1, 24), (1, 25), (1, 26), (2, 7), (2, 8), (2, 9), (2, 12),
        (2, 17), (2, 20), (2, 21), (2, 25), (2, 26), (3, 2), (3, 5), (3, 6), (3, 8), (3, 9), (3, 10), (3, 13),
        (3, 14), (3, 15), (3, 16), (3, 19), (3, 21), (3, 22), (3, 25), (3, 26), (3, 27), (4, 2), (4, 10), (4, 13),
        (4, 15), (4, 22), (4, 25), (4, 27), (5, 0), (5, 2), (5, 4), (5, 6), (5, 7), (5, 14), (5, 15), (5, 16),
        (5, 18), (5, 20), (5, 23), (5, 25), (5, 27), (6, 1), (6, 4), (6, 7), (6, 11), (6, 12), (6, 13), (6, 16),
        (6, 17), (6, 22), (6, 24), (6, 26), (6, 27), (8, 5), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16), (8, 19),
        (8, 22), (8, 23), (8, 24), (9, 4), (9, 5), (9, 10), (9, 11), (9, 13), (9, 16), (9, 18), (9, 19), (9, 21),
        (9, 22), (9, 23), (9, 24), (9, 26), (10, 0), (10, 2), (10, 5), (10, 6), (10, 14), (10, 15), (10, 16), (10, 18),
        (10, 21), (10, 23), (10, 24), (10, 26), (11, 1), (11, 3), (11, 8), (11, 15), (11, 17), (11, 24), (11, 27), (13, 2),
        (13, 3), (13, 4), (13, 5), (13, 6), (13, 8), (13, 10), (13, 11), (13, 12), (13, 15), (13, 19), (13, 20), (13, 22),
        (13, 23), (13, 25), (13, 27), (15, 3), (15, 6), (15, 9), (15, 16), (15, 17), (15, 20), (15, 25), (15, 26), (15, 27),
        (16, 0), (16, 2), (16, 3), (16, 6), (16, 10), (16, 11), (16, 12), (16, 14), (16, 15), (16, 17), (16, 18), (16, 19),
        (17, 0), (17, 3), (17, 6), (17, 11), (17, 15), (17, 21), (17, 22), (17, 26), (17, 27), (18, 3), (18, 4), (18, 5),
        (18, 12), (18, 15), (18, 16), (18, 20), (18, 21), (18, 22), (19, 0), (19, 1), (19, 2), (19, 4), (19, 8), (19, 9),
        (19, 11), (19, 16), (19, 22), (19, 23), (19, 24), (19, 26), (19, 27), (21, 3), (21, 4), (21, 8), (21, 9), (21, 10),
        (21, 18), (23, 0), (23, 5), (23, 7), (23, 10), (23, 11), (23, 14), (23, 15), (23, 17), (23, 18), (23, 22), (23, 24),
        (23, 26), (23, 27), (24, 0), (24, 4), (24, 11), (24, 12), (24, 15), (24, 18), (24, 19), (24, 23), (25, 3), (25, 5),
        (25, 6), (25, 7), (25, 8), (25, 10), (25, 11), (25, 12), (25, 17), (25, 18), (25, 22), (25, 23), (26, 0), (26, 1),
        (26, 3), (26, 4), (26, 5), (26, 8), (26, 9), (26, 10), (26, 16), (26, 19), (26, 22), (27, 0), (27, 4), (27, 5),
        (27, 8), (27, 12), (27, 14), (27, 15), (27, 19), (27, 21),
    ],
}


def test_p2_subset_contains_phi_after_cycle_shift_then_deletion() -> None:
    """Regression for the P2 discipline (RESEARCH_NOTES.md P2, solver_general.cpp header): the certified subset
    stored for v must contain phi(v) after EVERY operation. A cycle shift phi(v) := t_i between the evaluation of
    the secondary arcs and the deletion of a kappa-restored one used to commit the PRE-shift copy of E(v), which
    need not contain t_i. The core now adopts the subset stored at commit time and asserts phi(v) ∈ E(v) after
    every deletion (status 'error' otherwise), so every option combination — with and without debug_asserts /
    trace, which would otherwise mask stale subsets by refreshing every cut — must answer ok with an
    independently verified partition."""
    c = P2_STALE_SUBSET_CASE
    inst = make_instance(c["n"], [tuple(a) for a in c["arcs"]], c["terminals"], c["capacities"], directed=True)
    shifts = 0
    for g, lz, b in itertools.product([True, False], repeat=3):
        for debug, trace in ((False, False), (True, True)):
            opts = {"greedy_contraction": g, "lazy_shift": lz, "batch_unused_arcs": b}
            res = solve(inst, threads=1, debug=debug, trace=trace, options=opts)
            assert_valid(res, inst, (g, lz, b, debug, trace))
            check_arborescence(inst, res)
            shifts += res.stats["cycle_shifts"]
    assert shifts > 0
    out = _core.solve_general(c["n"], c["arcs"], c["terminals"], c["capacities"],
                              {"threads": 1, "greedy_contraction": True, "lazy_shift": False, "batch_unused_arcs": False})
    assert out["status"] == "ok" and out["stats"]["cycle_shifts"] >= 3, out["message"]


# ---------------------------------------------------------------------------
# 10. C1 deletion-aware routing, options["routing"] = "avoid" (RESEARCH_NOTES E5)
# ---------------------------------------------------------------------------
ROUTING_POOL = small_pool(60, seed=915, n_max=22)
ROUTING_REF_POOL = small_pool(30, seed=916, n_max=13)


def test_routing_avoid_matches_bfs_on_kappa_ess_and_status() -> None:
    """``routing="avoid"`` routes the stored certificates around the arcs the algorithm is likely to delete.
    It only decides WHICH maximum flow is stored: the value kappa and the (unique, [Def 3.8]) tightest cut,
    hence Ess and every criticality decision, are unchanged.  So the status must agree with ``routing="bfs"``,
    the exact per-vertex kappa/Ess of the first ``essential`` trace event must be identical in both modes, and
    the partition must be valid (it may legitimately differ)."""
    differing = 0
    checked = 0
    for inst in ROUTING_POOL:
        ra = solve(inst, options={"routing": "avoid"}, threads=2, trace=True)
        rb = solve(inst, options={"routing": "bfs"}, threads=2, trace=True)
        assert ra.status == rb.status, (inst.name, ra.status, rb.status, ra.message)
        assert ra.stats["k_T_connected"] == rb.stats["k_T_connected"], inst.name
        ea = next(e for e in ra.trace if e["type"] == "essential")
        eb = next(e for e in rb.trace if e["type"] == "essential")
        assert ea["kappa"] == eb["kappa"], inst.name
        assert ea["ess"] == eb["ess"], inst.name
        checked += 1
        if ra.status != "ok":
            continue
        assert_valid(ra, inst, (inst.name, "avoid"))
        check_arborescence(inst, ra)
        differing += [sorted(p) for p in ra.parts] != [sorted(p) for p in rb.parts]
    assert checked == len(ROUTING_POOL)
    assert 0 <= differing <= len(ROUTING_POOL)


def test_routing_avoid_essential_sets_match_the_definition() -> None:
    """The stored sets of the penalized routing are the exact Ess_G(v) of [Def 4.1] (computed independently
    by ``glsolver.preconditions``), i.e. the option cannot make the oracle wrong."""
    for inst in ROUTING_REF_POOL:
        res = solve(inst, options={"routing": "avoid"}, threads=2, trace=True)
        by_def = essential_sets_by_definition(inst)
        ev = next(e for e in res.trace if e["type"] == "essential")
        assert {int(v): set(ts) for v, ts in ev["ess"].items()} == by_def, inst.name
        if res.status == "ok":
            assert_valid(res, inst, (inst.name, "avoid-ref"))


def test_routing_avoid_debug_asserts_and_determinism() -> None:
    """With ``debug_asserts`` the core also re-verifies the incremental maintenance of the per-arc penalty
    array against a full recomputation after every operation (A1-A8 plus the C1 check), and the result stays
    deterministic across thread counts."""
    for inst in DET_POOL[:15]:
        rd = solve(inst, options={"routing": "avoid"}, threads=2, debug=True)
        assert_valid(rd, inst, (inst.name, "debug"))
        # determinism across thread counts (debug mode refreshes cuts and therefore legitimately takes
        # different FEAC-preserving decisions, so it is compared against itself only)
        r1 = solve(inst, options={"routing": "avoid"}, threads=1)
        r8 = solve(inst, options={"routing": "avoid"}, threads=8)
        assert r1.status == "ok" and r1.parts == r8.parts, inst.name
        assert_valid(r1, inst, inst.name)
        keys = ("steps", "contractions", "deletions", "cycle_shifts", "augment_calls", "cut_calls", "flow_repairs")
        assert {k: r1.stats[k] for k in keys} == {k: r8.stats[k] for k in keys}, inst.name


def test_routing_diagnostics_are_reported_in_both_modes() -> None:
    """The two C1 diagnostics are measured in both modes (the comparison must be meaningful) and count what
    they claim: ``penalized_users_*`` is 0 exactly when no stored flow uses an arc a pre-terminal may lose,
    and ``flow_repairs`` is the number of stored flows re-validated by the deletion evaluations."""
    inst = generators.harary_graph(120, 4, seed=2)
    stats = {}
    for routing in ("bfs", "avoid"):
        res = solve(inst, options={"routing": routing}, threads=2)
        assert_valid(res, inst, routing)
        st = res.stats
        for key in ("flow_repairs", "penalized_users_initial", "penalized_users_witness"):
            assert isinstance(st[key], int) and st[key] >= 0, (routing, key)
        assert st["flow_repairs"] > 0  # Harary: every deletion re-validates many stored flows
        stats[routing] = st
    # the penalized routing never increases the number of users of penalized arcs it starts from
    assert stats["avoid"]["penalized_users_initial"] <= stats["bfs"]["penalized_users_initial"]
    # a line with no pre-terminal risk at all: a star has every non-terminal adjacent only to the terminal
    star = make_instance(5, [(v, 4) for v in range(4)], [4], [4], directed=True)
    res = solve(star, options={"routing": "avoid"})
    assert res.status == "ok" and res.stats["penalized_users_witness"] == 0
