"""Tests for the unweighted reference solver ``reference/glref/unweighted.py``
(GLPartition [Alg 1] and ShiftAssignment [Alg 2], docs/paper_notes.md §7).

Every solved instance is verified by the local checker of
``helpers_small_graphs`` and, when importable, by ``glsolver.verify`` (both
must agree); certificates (in-arborescence parents) and traces are checked
structurally; the solver's honesty on inputs violating FEAC is checked; and,
when ``glsolver.oracle.bruteforce`` is importable, its verdicts are compared
with an exhaustive search.
"""
from __future__ import annotations

import json
import random
from typing import Any

import pytest

from glref.assignment import find_witness
from glref.essential import all_essential
from glref.graph import DiGraphState
from glref.trace import Tracer
from glref.unweighted import InvariantError, RefResult, RefStats, gl_partition, shift_assignment
from glsolver.instance import Instance, make_instance
from helpers_small_graphs import (
    CAPACITY_MODES,
    HAVE_VERIFIER,
    assert_valid_partition,
    check_partition,
    paper_essential_example,
    paper_running_example,
    partition_is_valid,
    random_capacities,
    random_digraph,
    random_k_connected_undirected,
    random_kT_connected_digraph,
    random_undirected_instance,
)

try:
    from glsolver.oracle.bruteforce import bruteforce_partition
except Exception:  # pragma: no cover - depends on sibling work
    bruteforce_partition = None


# ---------------------------------------------------------------------------
# structural checks shared by the tests
# ---------------------------------------------------------------------------


def assert_arborescence_certificate(inst: Instance, res: RefResult) -> None:
    """``res.parents`` is an in-arborescence certificate (paper_notes §7.1, §13.4):
    every non-terminal has a parent, (v, parent[v]) is an *original* arc into the
    same part, and the parent chain reaches the part's terminal without cycles."""
    original = set(inst.arcs)
    tset = set(inst.terminals)
    part_of: dict[int, int] = {}
    for i, part in enumerate(res.parts):
        for v in part:
            part_of[v] = i
    assert set(res.parents) == set(range(inst.n)) - tset
    for v in range(inst.n):
        if v in tset:
            continue
        x = v
        seen = {v}
        while x not in tset:
            p = res.parents[x]
            assert (x, p) in original, f"certificate arc ({x},{p}) is not an original arc"
            assert part_of[p] == part_of[x], f"certificate arc ({x},{p}) leaves the part"
            assert p not in seen, f"parent chain of {v} cycles at {p}"
            seen.add(p)
            x = p
        assert x == inst.terminals[part_of[v]]


def assert_trace_well_formed(inst: Instance, res: RefResult, events: list[dict[str, Any]]) -> None:
    """Event order of paper_notes §14 for [Alg 1]/[Alg 2]: init, essential, witness first;
    each matching is followed by criticality, then (reassignment_graph, cycle_shift)*
    and exactly one delete_arc; every delete_arc and remove_terminal is followed by an
    essential event re-emitting Ess/κ of exactly the live non-terminals (§13.3); every
    contract records the certificate parent and decrements the target's capacity; done
    is last and carries the parts."""
    types = [e["type"] for e in events]
    assert types[:3] == ["init", "essential", "witness"]
    assert types[-1] == "done"
    allowed = {
        "init", "essential", "witness", "remove_terminal", "contract", "matching",
        "criticality", "reassignment_graph", "cycle_shift", "delete_arc", "done",
    }
    assert set(types) <= allowed, set(types) - allowed
    assert events[0]["n"] == inst.n and events[0]["k"] == inst.k
    assert events[0]["terminals"] == list(inst.terminals)
    assert events[0]["capacities"] == list(inst.capacities)
    assert sorted(tuple(a) for a in events[0]["arcs"]) == sorted(inst.arcs)

    g0 = DiGraphState.from_instance(inst)
    kappa0, ess0 = all_essential(g0)
    assert events[1]["ess"] == {str(v): sorted(s) for v, s in ess0.items()}
    assert events[1]["kappa"] == {str(v): kappa0[v] for v in kappa0}
    phi0 = {int(v): t for v, t in events[2]["phi"].items()}
    assert all(phi0[v] in ess0[v] for v in phi0)

    cap = dict(zip(inst.terminals, inst.capacities))
    live_terminals = list(inst.terminals)
    live_nonterminals = {v for v in range(inst.n) if v not in inst.terminals}
    in_shift = False
    shifts_in_block = 0
    i = 3
    while i < len(events) - 1:
        ev = events[i]
        t_ = ev["type"]
        if t_ == "matching":
            assert not in_shift
            in_shift = True
            shifts_in_block = 0
            assert events[i + 1]["type"] == "criticality"
            pairs = [tuple(p) for p in ev["pairs"]]
            secondary = [tuple(s) for s in ev["secondary"]]
            assert sorted(t for _p, t in pairs) == sorted(live_terminals)
            assert len({p for p, _t in pairs}) == len(pairs)
            for (p, t), (p2, q) in zip(pairs, secondary):
                assert p == p2 and q != t
            i += 2
            continue
        if t_ == "reassignment_graph":
            assert in_shift
            assert events[i + 1]["type"] == "cycle_shift"
            cycle = ev["cycle"]
            assert len(cycle) >= 2 and len(set(cycle)) == len(cycle)  # [Lem 7.10]: no self-loops
            arcs = [tuple(a) for a in ev["arcs"]]
            assert sorted(a[1] for a in arcs) == sorted(live_terminals)  # in-degree one everywhere
            changes = [tuple(c) for c in events[i + 1]["changes"]]
            assert len(changes) == len(cycle)
            assert all(old != new for _v, old, new in changes)
            if events[i + 1]["potential_before"] is not None:
                assert events[i + 1]["potential_after"] < events[i + 1]["potential_before"]  # [Lem 7.11]
            shifts_in_block += 1
            i += 2
            continue
        if t_ == "delete_arc":
            assert in_shift, "delete_arc must come from a ShiftAssignment block"
            in_shift = False
            i += 1
            continue
        assert not in_shift, f"{t_} inside a ShiftAssignment block"
        if t_ == "essential":  # re-emitted right after every recomputation of Ess (§13.3, §14)
            assert events[i - 1]["type"] in ("delete_arc", "remove_terminal"), events[i - 1]["type"]
            assert {int(v) for v in ev["ess"]} == live_nonterminals == {int(v) for v in ev["kappa"]}
            assert all(set(ts) <= set(live_terminals) and ts == sorted(ts) for ts in ev["ess"].values())
        elif t_ == "contract":
            p, t, parent = ev["p"], ev["t"], ev["parent"]
            assert res.parents[p] == parent
            assert (p, parent) in set(inst.arcs)
            assert events[i + 1]["type"] != "essential"  # contractions leave Ess unchanged (§13.2)
            live_nonterminals.remove(p)
            cap[t] -= 1
            assert ev["capacities"] == {str(x): cap[x] for x in live_terminals}
        elif t_ == "remove_terminal":
            t = ev["t"]
            assert cap[t] == 0
            live_terminals.remove(t)
            assert ev["capacities"] == {str(x): cap[x] for x in live_terminals}
        else:  # pragma: no cover - guarded by the `allowed` check
            raise AssertionError(t_)
        i += 1
    assert not in_shift
    assert types.count("contract") == inst.n - inst.k == res.stats.contractions
    assert types.count("delete_arc") == res.stats.deletions == types.count("matching")
    assert types.count("matching") == res.stats.shift_calls
    assert types.count("cycle_shift") == res.stats.cycle_shifts
    assert types.count("remove_terminal") == res.stats.terminal_removals
    assert types.count("essential") == 1 + res.stats.deletions + res.stats.terminal_removals
    assert events[-1]["parts"] == {str(t): sorted(p) for t, p in zip(inst.terminals, res.parts)}
    assert events[-1]["parents"] == {str(v): p for v, p in res.parents.items()}


def solve_and_check(inst: Instance, *, debug: bool = True, trace: bool = False) -> RefResult:
    tracer = Tracer(enabled=trace)
    res = gl_partition(inst, tracer=tracer, debug=debug)
    assert res.status == "ok", (res.message, inst.name, inst.terminals, inst.capacities, inst.arcs)
    assert res.message == "ok"
    assert len(res.parts) == inst.k
    assert_valid_partition(inst, res.parts)
    assert_arborescence_certificate(inst, res)
    assert res.witness == {}  # every non-terminal has been contracted
    assert res.stats.contractions == inst.n - inst.k
    assert res.stats.steps == res.stats.contractions + res.stats.deletions + res.stats.terminal_removals
    zeros = sum(1 for c in inst.capacities if c == 0)
    if inst.n > inst.k:
        # step (i) fires for the initial zeros and for every terminal whose capacity reaches 0
        # while non-terminals remain; the last terminal is never removed (Σ c = |V\T| > 0).
        assert zeros <= res.stats.terminal_removals <= inst.k - 1
    else:
        assert res.stats.terminal_removals == 0 and res.stats.steps == 0
    assert res.stats.shift_calls == res.stats.deletions
    assert res.stats.max_flow_calls >= inst.n - inst.k
    if trace:
        assert res.trace is not None and res.trace is tracer.events
        assert_trace_well_formed(inst, res, res.trace)
    else:
        assert res.trace is None
    return res


# ---------------------------------------------------------------------------
# paper examples
# ---------------------------------------------------------------------------


def test_paper_running_example_debug_trace() -> None:
    inst = paper_running_example()
    res = solve_and_check(inst, debug=True, trace=True)
    assert [sorted(p) for p in res.parts] == [[0, 3, 7], [1, 4, 5], [2, 6, 8]] or partition_is_valid(inst, res.parts)
    types = [e["type"] for e in res.trace or []]
    assert "contract" in types and "delete_arc" in types and "cycle_shift" in types
    assert res.stats.cycle_shifts >= 1 and res.stats.deletions >= 1
    assert res.stats.time_essential >= 0 and res.stats.time_criticality >= 0
    d = res.stats.as_dict()
    assert d["contractions"] == 6 and len(d["graph_size_over_time"]) == res.stats.steps
    assert d["graph_size_over_time"][0] == (6, 12)


def test_paper_running_example_no_debug_matches_debug() -> None:
    inst = paper_running_example()
    a = solve_and_check(inst, debug=False, trace=True)
    b = solve_and_check(inst, debug=True, trace=False)
    assert a.parts == b.parts and a.parents == b.parents
    assert a.stats.max_flow_calls < b.stats.max_flow_calls  # debug re-verifies A7 with extra flows
    assert a.stats.deletions == b.stats.deletions and a.stats.cycle_shifts == b.stats.cycle_shifts


def test_paper_essential_example_solved_and_with_given_witness() -> None:
    inst = paper_essential_example()
    solve_and_check(inst, trace=True)
    # the paper's displayed witness may be passed in as phi
    phi = {12: 2, 10: 1, 11: 1, 4: 0, 5: 0, 6: 1, 7: 2, 8: 3, 9: 3}
    res = gl_partition(inst, phi=dict(phi), debug=True)
    assert res.status == "ok"
    assert_valid_partition(inst, res.parts)
    assert_arborescence_certificate(inst, res)
    # a non-witness phi is rejected (t1 is not essential for v7 = 6)
    bad = dict(phi)
    bad[6], bad[4] = 0, 1
    with pytest.raises(ValueError, match="not a witness"):
        gl_partition(inst, phi=bad)


# ---------------------------------------------------------------------------
# random families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", CAPACITY_MODES)
def test_random_k_connected_undirected(mode: str) -> None:
    """Classical GL on random k-connected undirected graphs, n 4..14, k 1..5, debug=True."""
    rng = random.Random(hash(mode) % 1000 + 100)
    solved = 0
    for n in range(4, 15):
        for k in range(1, 6):
            if k > n:
                continue
            if k == n:
                inst = make_instance(n, [], list(range(n)), [0] * n, name=f"K_{n}-all-terminals")
            else:
                inst = random_undirected_instance(n, k, rng, mode, name=f"und-{mode}-{n}-{k}")
            res = solve_and_check(inst, debug=True, trace=(n <= 9))
            solved += 1
            if mode == "extreme" and n > k:
                big = max(range(k), key=lambda i: inst.capacities[i])
                assert len(res.parts[big]) == n - k + 1
    assert solved >= 50


@pytest.mark.parametrize("seed", range(3))
def test_random_kT_connected_digraphs(seed: int) -> None:
    rng = random.Random(200 + seed)
    for n in range(4, 13):
        for k in range(1, min(5, n) + 1):
            mode = rng.choice(CAPACITY_MODES)
            inst = random_kT_connected_digraph(n, k, rng, mode, name=f"kT-{n}-{k}-{mode}")
            solve_and_check(inst, debug=True, trace=(n <= 8))


def test_k_equals_one() -> None:
    """§13.10: k = 1, c_1 = |V\\T|; the algorithm degenerates to contractions along a BFS tree."""
    rng = random.Random(300)
    for n in range(2, 12):
        G = random_k_connected_undirected(n, 1, rng)
        t = rng.randrange(n)
        inst = make_instance(n, list(G.edges()), [t], [n - 1], directed=False)
        res = solve_and_check(inst, trace=True)
        # with one terminal the reassignment graph has one node: a ShiftAssignment call
        # returns its secondary arc immediately [Lem 7.10], so no cycle shift ever happens
        # (the literal [Alg 1] may still delete arcs of pre-terminals with out-degree ≥ 2).
        assert res.stats.cycle_shifts == 0 and res.stats.terminal_removals == 0
        assert sorted(res.parts[0]) == list(range(n))
    # directed: a random digraph in which every vertex reaches the terminal
    for n in range(3, 10):
        inst = random_kT_connected_digraph(n, 1, rng, "balanced")
        res = solve_and_check(inst, trace=True)
        assert res.stats.cycle_shifts == 0 and res.stats.terminal_removals == 0


def test_k_equals_n_minus_one_and_n() -> None:
    rng = random.Random(400)
    for n in range(2, 9):
        # k = n - 1: complete graph, one non-terminal, exactly one terminal gets capacity 1
        caps = [0] * (n - 1)
        caps[rng.randrange(n - 1)] = 1
        terminals = rng.sample(range(n), n - 1)
        G = random_k_connected_undirected(n, n - 1, rng)
        inst = make_instance(n, list(G.edges()), terminals, caps, directed=False)
        res = solve_and_check(inst, trace=True)
        big = caps.index(1)
        (v,) = set(range(n)) - set(terminals)
        assert sorted(res.parts[big]) == sorted([terminals[big], v])
        assert res.parents == {v: terminals[big]}
        # k = n: every vertex is a terminal, nothing to do
        inst = make_instance(n, list(G.edges()), list(range(n)), [0] * n, directed=False)
        res = solve_and_check(inst, trace=True)
        assert res.parts == [[t] for t in range(n)] and res.parents == {}
        assert res.stats.steps == 0
    # k = n on a directed instance without arcs
    res = solve_and_check(make_instance(1, [], [0], [0]), trace=True)
    assert res.parts == [[0]]


def test_many_zero_capacities() -> None:
    """Several c_i = 0 [Lem 7.2]: those terminals are removed first (§13.3 recomputation)."""
    rng = random.Random(500)
    for n in range(6, 13):
        for k in (3, 4, 5):
            caps = [0] * k
            caps[rng.randrange(k)] = n - k
            if k >= 3:
                caps = [0] * k
                nz = rng.sample(range(k), 2)
                a = rng.randint(0, n - k)
                caps[nz[0]], caps[nz[1]] = a, n - k - a
            inst_u = random_undirected_instance(n, k, rng, "random")
            inst = make_instance(n, inst_u.undirected_edges or [], inst_u.terminals, caps, directed=False)
            res = solve_and_check(inst, trace=(n <= 9))
            assert res.stats.terminal_removals >= caps.count(0)
            for i, c in enumerate(caps):
                if c == 0:
                    assert res.parts[i] == [inst.terminals[i]]
            inst_d = random_kT_connected_digraph(n, k, rng, "zeros")
            solve_and_check(inst_d, trace=(n <= 9))


@pytest.mark.parametrize("seed", range(3))
def test_random_digraphs_with_witness_are_solved(seed: int) -> None:
    """Random (not necessarily k-T-connected) digraphs: whenever FEAC holds the solver must
    succeed [Thm ess-assign-cond]; otherwise it must report precondition_failed."""
    rng = random.Random(600 + seed)
    ok = failed = 0
    for _ in range(40):
        n = rng.randint(3, 11)
        k = rng.randint(1, min(4, n - 1))
        inst = random_digraph(n, k, rng, p=rng.uniform(0.2, 0.7), mode=rng.choice(CAPACITY_MODES))
        g = DiGraphState.from_instance(inst)
        _kappa, ess = all_essential(g)
        phi = find_witness(g, ess, dict(zip(inst.terminals, inst.capacities)))
        res = gl_partition(inst, debug=True, tracer=Tracer(enabled=True))
        if phi is None:
            failed += 1
            assert res.status == "precondition_failed"
            assert res.parts == [] and res.parents == {} and res.witness == {}
            assert "Flow-Essential Assignment Condition fails" in res.message
            assert [e["type"] for e in res.trace or []] == ["init", "essential"]
        else:
            ok += 1
            assert res.status == "ok", res.message
            assert_valid_partition(inst, res.parts)
            assert_arborescence_certificate(inst, res)
            assert res.trace is not None
            assert_trace_well_formed(inst, res, res.trace)
    assert ok >= 8 and failed >= 5, (ok, failed)


# ---------------------------------------------------------------------------
# precondition failures (§13.9, §7.3)
# ---------------------------------------------------------------------------


def test_precondition_failed_vertex_cannot_reach_terminals() -> None:
    inst = make_instance(5, [(2, 0), (2, 1), (3, 4), (4, 3), (2, 3)], [0, 1], [2, 1])
    res = gl_partition(inst, debug=True)
    assert res.status == "precondition_failed"
    assert "no essential terminal" in res.message and "[3, 4]" in res.message
    assert res.parts == [] and res.parents == {}
    assert res.stats.max_flow_calls == 3 and res.stats.steps == 0


def test_precondition_failed_hall_deficient_capacities() -> None:
    """§13.3 example with c(t1) = 0: Ess(v) = {t1} so no witness exists although the
    partition {t1}, {t2, v, x}, {t3} is valid.  The solver must say precondition_failed
    (never 'no solution'); the brute-force oracle confirms a partition exists."""
    v, x, t1, t2, t3 = 3, 4, 0, 1, 2
    inst = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [0, 2, 0])
    res = gl_partition(inst, debug=True)
    assert res.status == "precondition_failed"
    assert "no witness" in res.message and "no essential terminal" not in res.message
    assert partition_is_valid(inst, [[t1], [t2, v, x], [t3]])
    if bruteforce_partition is not None:
        status, parts = bruteforce_partition(inst, time_limit=5)
        assert status == "ok" and parts is not None and partition_is_valid(inst, parts)
    # with c = (1, 1, 0) FEAC holds and the solver succeeds
    inst2 = make_instance(5, [(v, t1), (v, x), (x, t2), (x, t3)], [t1, t2, t3], [1, 1, 0])
    res2 = solve_and_check(inst2, trace=True)
    assert res2.parts == [[t1, v], [t2, x], [t3]]


def test_precondition_failed_only_essential_terminal_has_zero_capacity() -> None:
    inst = make_instance(4, [(2, 0), (3, 0), (3, 1)], [0, 1], [0, 2])
    res = gl_partition(inst, debug=True)
    assert res.status == "precondition_failed"
    res = gl_partition(make_instance(4, [(2, 0), (3, 0), (3, 1)], [0, 1], [1, 1]), debug=True)
    assert res.status == "ok" and res.parts == [[0, 2], [1, 3]]


def test_weighted_instance_rejected() -> None:
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 1], weights=[0, 0, 1])
    with pytest.raises(ValueError, match="unweighted"):
        gl_partition(inst)


def test_shift_assignment_debug_preconditions() -> None:
    """[Alg 2] requires c_t > 0 and d^+(p) ≥ 2 for pre-terminals; debug mode raises."""
    inst = paper_running_example()
    g = DiGraphState.from_instance(inst)
    _kappa, ess = all_essential(g)
    cap = {0: 2, 1: 2, 2: 2}
    phi = find_witness(g, ess, cap)
    assert phi is not None
    with pytest.raises(InvariantError, match="positive capacities"):
        shift_assignment(g, {0: 0, 1: 4, 2: 2}, dict(phi), ess, RefStats(), Tracer(False), debug=True)
    g2 = g.copy()
    g2.delete_arc(3, 1)  # 3 now has the single arc (3, 0)
    with pytest.raises(InvariantError, match="out-degree"):
        shift_assignment(g2, cap, dict(phi), ess, RefStats(), Tracer(False), debug=True)
    # a proper call returns a non-critical secondary arc and keeps phi a witness in G \ e
    stats = RefStats()
    phi2, e_nc = shift_assignment(g, cap, dict(phi), ess, stats, Tracer(False), debug=True)
    assert g.has_arc(*e_nc) and not g.is_terminal(e_nc[0])
    h = g.copy()
    h.delete_arc(*e_nc)
    _k2, ess2 = all_essential(h)
    assert all(phi2[v] in ess2[v] for v in phi2)  # A6 / [Lem 7.5]
    assert stats.shift_calls == 1 and stats.matching_calls == 1


# ---------------------------------------------------------------------------
# agreement with the exhaustive oracle (guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bruteforce_partition is None, reason="glsolver.oracle.bruteforce not available")
@pytest.mark.parametrize("seed", range(2))
def test_agreement_with_bruteforce(seed: int) -> None:
    """Whenever the reference solver returns ok, its partition is valid and the exhaustive
    search agrees that a partition exists; whenever the exhaustive search proves
    infeasibility the solver must have reported precondition_failed."""
    assert bruteforce_partition is not None
    rng = random.Random(700 + seed)
    ok = infeasible = 0
    for _ in range(30):
        n = rng.randint(3, 8)
        k = rng.randint(1, min(3, n - 1))
        if rng.random() < 0.3:
            inst = random_undirected_instance(n, k, rng, rng.choice(CAPACITY_MODES))
        else:
            inst = random_digraph(n, k, rng, p=rng.uniform(0.25, 0.65), mode=rng.choice(CAPACITY_MODES))
        res = gl_partition(inst, debug=True)
        status, parts = bruteforce_partition(inst, time_limit=10)
        assert status != "timeout"
        if res.status == "ok":
            ok += 1
            assert_valid_partition(inst, res.parts)
            assert status == "ok" and parts is not None
            assert_valid_partition(inst, parts)
        else:
            assert res.status == "precondition_failed"
            if status == "infeasible":
                infeasible += 1
    assert ok >= 10 and infeasible >= 1, (ok, infeasible)


def test_local_checker_and_verifier_reject_bad_partitions() -> None:
    """The checkers used above really reject wrong answers (and agree with each other)."""
    inst = paper_running_example()
    good = [[0, 3, 7], [1, 4, 5], [2, 6, 8]]
    assert partition_is_valid(inst, good)
    assert not partition_is_valid(inst, [[0, 3, 4], [1, 7, 5], [2, 6, 8]])  # 7 cannot reach 1 inside its part
    assert not partition_is_valid(inst, [[0, 3, 7, 4], [1, 5], [2, 6, 8]])  # sizes
    assert not partition_is_valid(inst, [[0, 3, 7], [1, 4, 5], [2, 6]])  # 8 missing
    assert not partition_is_valid(inst, [[0, 3, 7], [1, 4, 5], [2, 6, 8, 3]])  # duplicate
    assert not partition_is_valid(inst, [[0, 3, 7], [1, 4, 5, 2], [6, 8]])  # foreign terminal
    assert check_partition(inst, [[0, 3, 7], [1, 4, 5]])  # wrong number of parts
    if HAVE_VERIFIER:
        from glsolver.verify import verify_instance_parts

        assert not verify_instance_parts(inst, [[0, 3, 7], [1, 4, 5]]).valid


def test_deterministic_and_trace_dumps_as_json(tmp_path: Any) -> None:
    """Two runs give identical parts/parents/stats; the trace is JSON-serialisable and
    ``Tracer.dump`` writes it (paper_notes §14)."""
    rng = random.Random(800)
    inst = random_undirected_instance(12, 4, rng, "random")
    tr1, tr2 = Tracer(enabled=True), Tracer(enabled=True)
    r1 = gl_partition(inst, tracer=tr1, debug=True)
    r2 = gl_partition(inst, tracer=tr2, debug=False)
    assert r1.status == r2.status == "ok"
    assert r1.parts == r2.parts and r1.parents == r2.parents
    assert (r1.stats.deletions, r1.stats.cycle_shifts, r1.stats.contractions) == (
        r2.stats.deletions, r2.stats.cycle_shifts, r2.stats.contractions
    )
    non_debug_types = [e["type"] for e in tr2.events]
    assert [e["type"] for e in tr1.events] == non_debug_types
    path = tmp_path / "trace.json"
    tr1.dump(str(path))
    loaded = json.loads(path.read_text())
    assert loaded == json.loads(json.dumps(tr1.events)) == tr1.events
    assert loaded[-1]["type"] == "done"


def test_larger_sparse_instances_with_long_reassignment_cycles() -> None:
    """Larger sparse k-connected graphs exercise many arc deletions and reassignment
    cycles of length ≥ 3 (all debug assertions A1–A8 on)."""
    rng = random.Random(9000)
    cycle_lengths: set[int] = set()
    deletions = 0
    for n, k in [(15, 4), (16, 5), (17, 6), (18, 5), (16, 3), (18, 6)]:
        inst = random_undirected_instance(n, k, rng, rng.choice(CAPACITY_MODES), name=f"und-{n}-{k}")
        res = solve_and_check(inst, debug=True, trace=True)
        deletions += res.stats.deletions
        for ev in res.trace or []:
            if ev["type"] == "reassignment_graph":
                cycle_lengths.add(len(ev["cycle"]))
    assert deletions > 100
    assert cycle_lengths and max(cycle_lengths) >= 3, cycle_lengths


@pytest.mark.slow
def test_stress_many_seeds() -> None:
    rng = random.Random(9100)
    for _ in range(3):
        for n in (14, 16, 18):
            for k in (3, 5, 6):
                inst = random_undirected_instance(n, k, rng, rng.choice(CAPACITY_MODES))
                solve_and_check(inst, debug=True, trace=False)
        for n in (12, 14):
            for k in (4, 6):
                inst = random_kT_connected_digraph(n, k, rng, rng.choice(CAPACITY_MODES))
                solve_and_check(inst, debug=True, trace=True)


def test_capacity_generator_modes() -> None:
    rng = random.Random(1)
    for mode in CAPACITY_MODES:
        for total in (0, 1, 5, 12):
            for k in (1, 2, 5):
                caps = random_capacities(total, k, rng, mode)
                assert len(caps) == k and sum(caps) == total and min(caps) >= 0
                if mode == "extreme":
                    assert max(caps) == total
                if mode == "balanced":
                    assert max(caps) - min(caps) <= 1
                if mode == "zeros" and k >= 2:
                    assert caps.count(0) >= k - max(1, k // 2)
    with pytest.raises(ValueError):
        random_capacities(3, 2, rng, "nope")
