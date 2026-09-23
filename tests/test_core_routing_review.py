"""Adversarial review of the C1 deletion-aware routing (``options["routing"] = "avoid"``).

The claim under review (RESEARCH_NOTES P4/E5): *changing which maximum flow is stored cannot change*
``kappa``, ``Ess``, criticality *or any decision of the algorithm*.  These tests attack it from four sides.

1. **Differential fuzz** (:func:`test_routing_fuzz_status_and_validity`): >2000 seeded instances of every
   kind the review harness generates (undirected ``k``-connected, directed ``k``-``T``-connected, FEAC-only,
   DAG, random digraph, and the weighted variants of all of them; ``n`` 3..14, ``k`` 1..5) solved with
   ``routing="avoid"`` and ``routing="bfs"``.  Both must agree on the status and on ``k_T_connected``, and
   every ``ok`` partition must be accepted by the independent verifier.
2. **Oracle vs. the literal definition** (:func:`test_routing_avoid_oracle_matches_reference_essential`):
   the per-vertex ``kappa`` / ``Ess`` the core reports at every ``essential`` trace event -- i.e. after each
   deletion / contraction / terminal removal of a random operation sequence -- compared against
   ``glref.essential.all_essential`` on the replayed reference graph.
3. **Full trace replay** (:func:`test_routing_avoid_trace_replay`): the FEAC/FESAC replay of
   ``tests/test_core_solver_review.py`` re-run with ``routing="avoid"`` forced on every core call.
4. **The 0-1 search itself** (:func:`test_routing_probe_cxx`): the pybind11 surface exposes no way to install
   a penalty array, so the penalized search can only be reached from Python end-to-end through the solvers.
   ``tests/cxx/routing_probe.cpp`` links ``flow.cpp`` directly and checks, on 200 000 seeded residual states,
   that the penalized maximum flow has the same ``kappa`` / ``Ess`` / L-S-R sides as the BFS one, that every
   stored path family is vertex-disjoint and simple (the invariant a 0-1 deque bug breaks), that
   ``augment_once`` finds an augmenting path exactly when the plain BFS does from the same state, and that
   the first augmenting path really has minimum total penalty (independent Dijkstra).

Plus: the incremental maintenance of the penalty array against a full recomputation
(:func:`test_routing_avoid_penalty_maintenance`, via ``debug_asserts``, with measured coverage of cycle
shifts / terminal removals / contractions) and the otherwise-unreachable ``GLCORE_C1_REROUTE=1`` pass
(:func:`test_routing_reroute_pass_is_exact`).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from glref.essential import all_essential
from glref.graph import DiGraphState
from glsolver import _core
from glsolver.api import GLResult, core_available, glpartition
from glsolver.instance import Instance
from glsolver.verify import verify_instance_parts
from test_core_solver_review import KINDS, algorithms_for, cases, replay

pytestmark = pytest.mark.skipif(
    not (core_available("general") and core_available("weighted")), reason="C++ core solvers not built"
)

ROOT = Path(__file__).resolve().parents[1]
OPTION_COMBOS = [
    {"greedy_contraction": True, "lazy_shift": True, "batch_unused_arcs": True},
    {"greedy_contraction": False, "lazy_shift": False, "batch_unused_arcs": False},
    {"greedy_contraction": True, "lazy_shift": False, "batch_unused_arcs": True},
]


def solve(inst: Instance, algo: str, routing: str, **kw: Any) -> GLResult:
    opts = dict(kw.pop("options", {}))
    opts["routing"] = routing
    return glpartition(inst, algorithm=algo, options=opts, verify_preconditions=False,
                       threads=kw.pop("threads", 1), **kw)


def assert_partition_ok(res: GLResult, inst: Instance, where: Any) -> None:
    assert res.status == "ok", (where, res.status, res.message)
    rep = verify_instance_parts(inst, res.parts)
    assert rep.valid, (where, rep.errors[:3])
    assert sorted(v for p in res.parts for v in p) == list(range(inst.n)), where


# =====================================================================================================
# 1. differential fuzz: the routing may not change the status, and both partitions must verify
# =====================================================================================================
def test_routing_fuzz_status_and_validity() -> None:
    n_inst = 0
    n_runs = 0
    n_ok = 0
    n_precondition_failed = 0
    n_parts_differ = 0
    for kind in KINDS:
        for _seed, inst in cases(kind, range(240)):
            assert 3 <= inst.n <= 14 and 1 <= inst.k <= 5, inst.name
            n_inst += 1
            for algo in algorithms_for(inst):
                a = solve(inst, algo, "avoid")
                b = solve(inst, algo, "bfs")
                n_runs += 2
                where = (kind, inst.name, algo)
                assert a.status != "error", (where, a.message)
                assert b.status != "error", (where, b.message)
                # P4: the status is a property of the instance, not of the stored certificates
                assert a.status == b.status, (where, a.status, b.status, a.message, b.message)
                assert a.stats["k_T_connected"] == b.stats["k_T_connected"], where
                if a.status == "ok":
                    n_ok += 1
                    assert_partition_ok(a, inst, where + ("avoid",))
                    assert_partition_ok(b, inst, where + ("bfs",))
                    if [sorted(p) for p in a.parts] != [sorted(p) for p in b.parts]:
                        n_parts_differ += 1
                else:
                    n_precondition_failed += 1
    # the fuzz really covered what it claims to cover
    print(f"[fuzz] instances={n_inst} solver_runs={n_runs} ok={n_ok} precondition_failed={n_precondition_failed} partitions_differ={n_parts_differ}")
    assert n_inst >= 2000, n_inst
    assert n_runs >= 4000, n_runs
    assert n_ok >= 1500 and n_precondition_failed >= 100, (n_ok, n_precondition_failed)
    # the option genuinely changes the stored certificates on this pool (otherwise the test is vacuous)
    assert n_parts_differ >= 1, n_parts_differ


# =====================================================================================================
# 2. the oracle's kappa / Ess under "avoid" are the exact ones, at every step of the operation sequence
# =====================================================================================================
def _essential_events_are_exact(inst: Instance, res: GLResult, where: Any) -> int:
    """Replay the trace on the reference graph and compare EVERY ``essential`` event (which the core emits
    after terminal removals and, in trace mode, with every cut refreshed) with ``all_essential``."""
    g = DiGraphState.from_instance(inst)
    seen = 0
    for idx, ev in enumerate(res.trace):
        typ = ev["type"]
        if typ == "essential":
            kappa, ess = all_essential(g)
            got_ess = {int(v): set(ts) for v, ts in ev["ess"].items()}
            got_kappa = {int(v): int(x) for v, x in ev["kappa"].items()}
            assert got_ess == ess, (where, f"event {idx}", sorted(set(got_ess) ^ set(ess)) or
                                    {v: (sorted(got_ess[v]), sorted(ess[v])) for v in got_ess if got_ess[v] != ess[v]})
            assert got_kappa == kappa, (where, f"event {idx}", got_kappa, kappa)
            seen += 1
        elif typ == "remove_terminal":
            g.remove_terminal(ev["t"])
        elif typ == "contract":
            g.contract(ev["p"], ev["t"])
        elif typ == "delete_arc":
            g.delete_arc(ev["u"], ev["v"])
        elif typ in ("delete_arcs", "greedy_delete"):
            for u, v in ev["arcs"]:
                g.delete_arc(u, v)
        elif typ == "round_and_remove":
            for _t, p in (tuple(x) for x in ev["pairs"]):
                g.remove_vertex(p)
            for t in ev["S"]:
                g.remove_vertex(t)
    return seen


def test_routing_avoid_oracle_matches_reference_essential() -> None:
    checked = 0
    events = 0
    for kind in KINDS:
        for _seed, inst in cases(kind, range(35)):
            for algo in algorithms_for(inst):
                res = solve(inst, algo, "avoid", trace=True)
                if res.status != "ok":
                    continue
                events += _essential_events_are_exact(inst, res, (kind, inst.name, algo))
                checked += 1
    print(f"[oracle-vs-reference] traces={checked} essential_events={events}")
    assert checked >= 250, checked
    assert events >= 250, events


# =====================================================================================================
# 3. the FEAC / FESAC trace replay of test_core_solver_review.py, under routing = "avoid"
# =====================================================================================================
def test_routing_avoid_trace_replay() -> None:
    """``replay_general`` / ``replay_weighted``: after every state change the carried witness must be a FEAC
    (FESAC) witness w.r.t. the EXACT essential sets of the reference primitives."""
    totals: dict[str, int] = {}
    replayed = 0
    for kind in KINDS:
        for _seed, inst in cases(kind, range(25)):
            for algo in algorithms_for(inst):
                for opts in OPTION_COMBOS:
                    res = solve(inst, algo, "avoid", options=dict(opts), trace=True, threads=2)
                    assert res.status != "error", (kind, inst.name, algo, opts, res.message)
                    if res.status != "ok":
                        continue
                    for key, val in replay(inst, algo, res).items():
                        totals[key] = totals.get(key, 0) + val
                    replayed += 1
    print(f"[replay] runs={replayed} counts={totals}")
    assert replayed >= 300, replayed
    assert totals["checks"] >= 1000 and totals["contract"] >= 300, totals
    assert totals["shifts"] >= 1 and totals["greedy"] >= 50 and totals["remove"] >= 20, totals


# =====================================================================================================
# 4. incremental maintenance of the per-arc penalty array (debug_asserts recomputes it in full after
#    every operation and throws on any difference on an alive arc)
# =====================================================================================================
def test_routing_avoid_penalty_maintenance() -> None:
    cover = {"cycle_shifts": 0, "terminal_removals": 0, "contractions": 0, "deletions": 0,
             "batched_deletions": 0, "greedy_successes": 0, "roundings": 0, "runs": 0}
    for kind in KINDS:
        for _seed, inst in cases(kind, range(30)):
            for algo in algorithms_for(inst):
                for opts in OPTION_COMBOS:
                    res = solve(inst, algo, "avoid", options=dict(opts), debug=True, threads=2)
                    assert res.status != "error", (kind, inst.name, algo, opts, res.message)
                    if res.status != "ok":
                        continue
                    assert_partition_ok(res, inst, (kind, inst.name, algo, opts))
                    cover["runs"] += 1
                    for key in cover:
                        if key != "runs":
                            cover[key] += int(res.stats.get(key, 0))
    # the sequences really contained the operations whose penalty updates are incremental
    print(f"[penalty-maintenance] {cover}")
    assert cover["runs"] >= 300, cover
    assert cover["cycle_shifts"] >= 1, cover
    assert cover["terminal_removals"] >= 20, cover
    assert cover["contractions"] >= 500, cover
    assert cover["deletions"] >= 200, cover
    assert cover["roundings"] >= 1, cover


# =====================================================================================================
# 5. the 0-1 search itself (C++ probe; see the module docstring)
# =====================================================================================================
CXX_SOURCES = ["tests/cxx/routing_probe.cpp", "src/glcore/flow.cpp", "src/glcore/graph.cpp"]


@pytest.fixture(scope="module")
def routing_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler available")
    for src in CXX_SOURCES:
        if not (ROOT / src).is_file():
            pytest.skip(f"missing {src}")
    exe = tmp_path_factory.mktemp("routing_probe") / "routing_probe"
    cmd = [cxx, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic",
           "-I", str(ROOT / "src" / "glcore")] + [str(ROOT / s) for s in CXX_SOURCES] + ["-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"cannot build the probe: {proc.stderr[-800:]}")
    assert proc.stderr.strip() == "", proc.stderr  # -Wall -Wextra -Wpedantic must stay clean
    return exe


@pytest.mark.parametrize("lo,hi", [(0, 20000)])
def test_routing_probe_cxx(routing_probe: Path, lo: int, hi: int) -> None:
    proc = subprocess.run([str(routing_probe), str(lo), str(hi)], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout.strip()
    assert out.startswith("OK "), out
    fields = dict(kv.split("=") for kv in out.split()[1:])
    # the probe must really have exercised the penalized code path, not just the all-zero-penalty one
    print(f"[cxx-probe] {out}")
    assert int(fields["paths_differ"]) >= 1000, out
    assert int(fields["positive_penalty_paths"]) >= 1000, out
    assert int(fields["warm_starts"]) >= 10000, out


# =====================================================================================================
# 6. the optional re-routing pass (GLCORE_C1_REROUTE=1) is otherwise dead code: exercise it
# =====================================================================================================
REROUTE_SCRIPT = r"""
import json, sys
from glsolver import generators
from glsolver.api import glpartition
from glsolver.verify import verify_instance_parts
out = []
for name, inst in (("harary", generators.harary_graph(60, 4, seed=2)),
                   ("sparse", generators.sparse_k_connected(120, 3, seed=1)),
                   ("rr", generators.random_regular_graph(80, 4, 4, seed=1))):
    for algo in ("general", "weighted"):
        r = glpartition(inst, algorithm=algo, threads=2, verify=False, verify_preconditions=False,
                        options={"routing": "avoid"}, debug=True)
        rep = verify_instance_parts(inst, r.parts)
        out.append({"name": name, "algo": algo, "status": r.status, "valid": bool(rep.valid),
                    "reroutes": r.stats["reroutes"], "parts": [sorted(p) for p in r.parts]})
print(json.dumps(out))
"""


def _run_reroute(env_value: str | None) -> list[dict[str, Any]]:
    env = dict(os.environ)
    env.pop("GLCORE_C1_REROUTE", None)
    if env_value is not None:
        env["GLCORE_C1_REROUTE"] = env_value
    proc = subprocess.run([sys.executable, "-c", REROUTE_SCRIPT], capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_routing_reroute_pass_is_exact() -> None:
    """``GLCORE_C1_REROUTE=1`` re-computes, once, every flow that uses a penalized arc.  It is off by
    default and reachable only through the environment, so nothing else covers it: it must run, it must
    actually re-route flows, and the partitions must stay valid."""
    off = _run_reroute(None)
    on = _run_reroute("1")
    assert [r["reroutes"] for r in off] == [0] * len(off), off
    print(f"[reroute] off={[r['reroutes'] for r in off]} on={[r['reroutes'] for r in on]}")
    assert sum(r["reroutes"] for r in on) > 0, on
    for a, b in zip(off, on):
        assert a["status"] == b["status"] == "ok", (a, b)
        assert a["valid"] and b["valid"], (a, b)


# =====================================================================================================
# 7. the penalized 0-1 search through its own pybind11 surface
#
# Until this review the search had no unit-test surface at all: neither ``_core.tightest_cut`` nor
# ``_core.EssentialOracle`` could install a penalty array, so every test of it went through a solver, where
# the only observables are kappa / Ess / the partition.  ``tightest_cut(graph, v, penalties)`` and
# ``EssentialOracle.set_penalties`` close that gap; the invariants below are the path-level ones (the C++
# probe of section 5 checks the same properties on residual states a solver would be needed to reach).
# =====================================================================================================
def _random_digraph_with_terminals(rng: random.Random, n: int, k: int, extra: float) -> tuple[int, list, list]:
    """A random digraph on ``n`` vertices with ``k`` terminals: a random out-tree towards the terminals
    (so most vertices reach one) plus random extra arcs.  Returns (n, arcs, terminals)."""
    terminals = rng.sample(range(n), k)
    tset = set(terminals)
    arcs: set[tuple[int, int]] = set()
    order = [v for v in range(n) if v not in tset]
    rng.shuffle(order)
    reached = list(terminals)
    for v in order:  # every non-terminal gets an arc towards something already connected
        arcs.add((v, rng.choice(reached)))
        reached.append(v)
    want = int(extra * n)
    for _ in range(want):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and u not in tset:
            arcs.add((u, v))
    return n, sorted(arcs), sorted(terminals)


def _penalty_array(rng: random.Random, g: Any, mode: str) -> list[int]:
    """0/1 per arc id.  ``mode`` picks the shape: the solver's rule (out-arcs of a pre-terminal other than
    one kept arc), a random half, all ones, or all zeros."""
    pen = [0] * g.num_arc_ids()
    if mode == "zeros":
        return pen
    if mode == "ones":
        return [1] * g.num_arc_ids()
    if mode == "random":
        return [rng.randrange(2) for _ in range(g.num_arc_ids())]
    for p in g.pre_terminals():  # "solver": every out-arc of a pre-terminal except the first into a terminal
        outs = [(x, g.find_arc(p, x)) for x in g.out_neighbors(p)]
        keep = next((a for x, a in outs if g.is_terminal(x)), -1)
        for _x, a in outs:
            if a != keep:
                pen[a] = 1
    return pen


def _check_path_family(g: Any, v: int, paths: list[list[int]], kappa: int, where: Any) -> None:
    """A stored family must be ``kappa`` vertex-disjoint simple paths from ``v`` to distinct terminals."""
    assert len(paths) == kappa, (where, len(paths), kappa)
    seen: set[int] = set()
    ends: set[int] = set()
    for path in paths:
        assert path and path[0] == v, (where, path)
        assert len(set(path)) == len(path), (where, "path repeats a vertex", path)
        for x, y in zip(path, path[1:]):
            assert g.has_arc(x, y), (where, "not an arc", (x, y))
            assert g.live(x) and g.live(y), (where, "dead vertex on a path", (x, y))
        for x in path[1:]:  # the source is shared, every other vertex is used by at most one path
            assert x not in seen, (where, "paths share the vertex", x)
            seen.add(x)
        assert g.is_terminal(path[-1]), (where, "path does not end at a terminal", path)
        for x in path[1:-1]:
            assert not g.is_terminal(x), (where, "path passes through a terminal", path)
        ends.add(path[-1])
    assert len(ends) == kappa, (where, "two paths end at the same terminal", ends)


def _min_penalty_to_a_terminal(n: int, arcs: list, terminals: list, pen: list[int], v: int) -> int | None:
    """Independent 0-1 BFS: the minimum total penalty of a ``v``-to-terminal path that does not pass through
    another terminal (the rule the engine's first augmenting search follows).  None if no terminal is
    reachable."""
    from collections import deque
    out: dict[int, list[tuple[int, int]]] = {}
    for i, (x, y) in enumerate(arcs):
        out.setdefault(x, []).append((y, pen[i]))
    tset = set(terminals)
    INF = float("inf")
    dist = [INF] * n
    dist[v] = 0
    dq: deque[int] = deque([v])
    while dq:
        x = dq.popleft()
        if x in tset:
            continue  # a terminal ends the search; it is never an interior vertex
        for y, w in out.get(x, ()):
            if y == v:
                continue  # the engine never routes into the source
            if dist[x] + w < dist[y]:
                dist[y] = dist[x] + w
                (dq.appendleft if w == 0 else dq.append)(y)
    best = min((dist[t] for t in terminals), default=INF)
    return None if best == INF else int(best)


def test_routing_penalized_tightest_cut_matches_bfs() -> None:
    """P4 on the flow primitive itself: with any 0/1 penalty array the maximum flow of every vertex has the
    same kappa, the same L/S/R sides and the same Ess as the plain BFS one, and its paths are a valid
    vertex-disjoint family; an all-zero array reproduces the BFS paths exactly."""
    rng = random.Random(20260923)
    checked = 0
    differ = 0
    zero_identical = 0
    for trial in range(120):
        n = rng.randrange(6, 26)
        k = rng.randrange(1, min(4, n - 1) + 1)
        n, arcs, terminals = _random_digraph_with_terminals(rng, n, k, rng.choice([1.0, 2.0, 3.0]))
        g = _core.Graph(n, arcs, terminals)
        for mode in ("solver", "random", "ones", "zeros"):
            pen = _penalty_array(rng, g, mode)
            for v in g.live_nonterminals():
                base_k, base_sides, base_ess, base_paths = _core.tightest_cut(g, v)
                got_k, got_sides, got_ess, got_paths = _core.tightest_cut(g, v, pen)
                where = (trial, mode, v)
                assert got_k == base_k, (where, got_k, base_k)
                assert got_sides == base_sides, where
                assert got_ess == base_ess, (where, got_ess, base_ess)
                _check_path_family(g, v, got_paths, got_k, where)
                _check_path_family(g, v, base_paths, base_k, where)
                if mode == "zeros":
                    # E5 §2: with all penalties zero the 0-1 search is the plain BFS, path for path
                    assert got_paths == base_paths, (where, got_paths, base_paths)
                    zero_identical += 1
                elif got_paths != base_paths:
                    differ += 1
                checked += 1
    print(f"[tightest-cut-penalties] searches={checked} penalized_paths_differ={differ} zero_identical={zero_identical}")
    assert checked >= 4000, checked
    assert differ >= 100, differ  # the penalties really steer the search (otherwise the test is vacuous)
    assert zero_identical >= 500, zero_identical


def test_routing_penalized_path_has_minimum_total_penalty() -> None:
    """With a single terminal ``compute_max_flow`` performs exactly one augmenting search, so the stored
    path IS the search's path: its total penalty must equal the minimum over all admissible paths, computed
    by an independent 0-1 BFS.  This is the property a deque-based 0-1 search gets wrong."""
    rng = random.Random(4242)
    checked = 0
    positive = 0
    for trial in range(200):
        n = rng.randrange(5, 30)
        n, arcs, terminals = _random_digraph_with_terminals(rng, n, 1, rng.choice([1.0, 2.0, 4.0]))
        g = _core.Graph(n, arcs, terminals)
        pen = [rng.randrange(2) for _ in range(g.num_arc_ids())]
        for v in g.live_nonterminals():
            kappa, _sides, _ess, paths = _core.tightest_cut(g, v, pen)
            want = _min_penalty_to_a_terminal(n, arcs, terminals, pen, v)
            if kappa == 0:
                assert want is None, (trial, v, want)
                continue
            assert kappa == 1 and want is not None, (trial, v, kappa, want)
            arc_pen = [pen[g.find_arc(x, y)] for x, y in zip(paths[0], paths[0][1:])]
            assert sum(arc_pen) == want, (trial, v, paths[0], arc_pen, want)
            positive += 1 if want > 0 else 0
            checked += 1
    print(f"[min-penalty] paths={checked} of which minimum penalty > 0: {positive}")
    assert checked >= 1000, checked
    assert positive >= 50, positive  # instances where every path must cross a penalized arc


def test_routing_oracle_set_penalties_and_warm_start() -> None:
    """``EssentialOracle.set_penalties`` on the stored certificates: kappa / Ess of every vertex and of every
    warm-started re-validation after a deletion are those of the plain BFS oracle (P4), the stored families
    stay valid path families, and ``set_penalties(None)`` returns to the BFS routing exactly."""
    rng = random.Random(77)
    checked = 0
    differ = 0
    evaluated = 0
    for trial in range(40):
        n = rng.randrange(10, 30)
        k = rng.randrange(2, 4)
        n, arcs, terminals = _random_digraph_with_terminals(rng, n, k, rng.choice([2.0, 3.0]))
        ga, gb = _core.Graph(n, arcs, terminals), _core.Graph(n, arcs, terminals)
        oa, ob = _core.EssentialOracle(ga, 1, 0), _core.EssentialOracle(gb, 1, 0)
        pen = _penalty_array(rng, ga, "solver")
        oa.set_penalties(pen)
        oa.compute_all()
        ob.compute_all()
        bfs_paths = {v: ob.paths(v) for v in gb.live_nonterminals()}  # before any deletion below
        for v in ga.live_nonterminals():
            assert oa.kappa(v) == ob.kappa(v), (trial, v)
            assert oa.ess(v) == ob.ess(v), (trial, v)
            assert oa.sides(v) == ob.sides(v), (trial, v)
            _check_path_family(ga, v, oa.paths(v), oa.kappa(v), (trial, v, "avoid"))
            if oa.paths(v) != ob.paths(v):
                differ += 1
            checked += 1
        # A warm-started re-validation of the stored flows (the O2 path, where the penalties are read again).
        # WHICH flows are affected legitimately differs (that is the whole point of the routing); what the
        # two runs must agree on is the post-deletion kappa / Ess of every vertex, exactly (P4).
        live = [a for a in ga.arcs() if not ga.is_terminal(a[0])]
        if live:
            D = [rng.choice(live)]
            ra, rb = oa.evaluate_deletion(D, True), ob.evaluate_deletion(D, True)
            for v in set(ra) & set(rb):
                assert ra[v] == rb[v], (trial, D, v, ra[v], rb[v])  # (kappa, Ess) after the deletion
            for v in ra:
                _check_path_family(ga, v, oa.evaluated_paths(v), ra[v][0], (trial, D, v, "warm"))
            oa.commit_deletion(D)
            ob.commit_deletion(D)
            oa.refresh_all_cuts()
            ob.refresh_all_cuts()
            for v in ga.live_nonterminals():
                assert oa.kappa(v) == ob.kappa(v), (trial, D, v)
                assert oa.ess(v) == ob.ess(v), (trial, D, v)
                _check_path_family(ga, v, oa.paths(v), oa.kappa(v), (trial, D, v, "after commit"))
                evaluated += 1
        # back to the plain BFS: a fresh computation on the ORIGINAL graph must equal the BFS oracle's
        # families path for path (set_penalties(None) really restores the plain BFS, E5 §3)
        ga2 = _core.Graph(n, arcs, terminals)
        oc = _core.EssentialOracle(ga2, 1, 0)
        oc.set_penalties(_penalty_array(rng, ga2, "solver"))
        oc.set_penalties(None)
        oc.compute_all()
        for v in ga2.live_nonterminals():
            assert oc.paths(v) == bfs_paths[v], (trial, v)
    print(f"[oracle-penalties] vertices={checked} families_differ={differ} warm_started={evaluated}")
    assert checked >= 400, checked
    assert differ >= 20, differ
    assert evaluated >= 20, evaluated


def test_routing_reroute_pass_toggles_within_one_process() -> None:
    """The re-routing pass reads ``GLCORE_C1_REROUTE`` once per solver (not into a function-local static),
    so both settings can be measured in one process — which is what makes it testable at all.  Same
    instance, same options, two solves: ``reroutes`` must be 0 with the switch off and > 0 with it on, and
    both partitions must verify."""
    from glsolver import generators
    inst = generators.sparse_k_connected(120, 3, seed=1)
    old = os.environ.get("GLCORE_C1_REROUTE")
    seen = {}
    try:
        for value in (None, "1", None):
            if value is None:
                os.environ.pop("GLCORE_C1_REROUTE", None)
            else:
                os.environ["GLCORE_C1_REROUTE"] = value
            res = glpartition(inst, algorithm="general", threads=2, verify=False, verify_preconditions=False,
                              options={"routing": "avoid"}, debug=True)
            assert res.status == "ok", res.message
            assert verify_instance_parts(inst, res.parts).valid
            seen.setdefault(str(value), []).append(int(res.stats["reroutes"]))
    finally:
        os.environ.pop("GLCORE_C1_REROUTE", None)
        if old is not None:
            os.environ["GLCORE_C1_REROUTE"] = old
    print(f"[reroute-in-process] {seen}")
    assert seen["None"] == [0, 0], seen          # off before AND after the on-run: nothing is cached
    assert seen["1"][0] > 0, seen
