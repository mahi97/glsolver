"""Adversarial review of the C++ general and weighted solvers (``glsolver._core.solve_general`` /
``solve_weighted``, sources ``src/glcore/solver_general.cpp``, ``solver_weighted.cpp``, ``solver_common.hpp``,
``bindings_solver*.cpp``) against docs/paper_notes.md §7–§8, docs/optimizations.md O1–O9 and the certified-subset
discipline of RESEARCH_NOTES.md P2.

What is checked (every assertion is a real property; nothing here trusts a solver's own bookkeeping):

* thousands of seeded small instances (``n`` 3..14, ``k`` 1..5): undirected ``k``-connected, directed
  ``k``-``T``-connected, FEAC-only (confirmed with ``glsolver.preconditions.check_preconditions``), DAGs, arbitrary
  digraphs, weighted variants (``w_max <= 5``, slack) — capacities balanced / unbalanced / random / with zeros /
  all-in-one — through ``algorithm="general"`` and ``"weighted"`` with EVERY combination of ``greedy_contraction``,
  ``lazy_shift``, ``batch_unused_arcs`` and ``debug_asserts`` on. Every partition is verified by the independent
  verifier ``glsolver.verify.verify_instance_parts`` and its in-arborescence certificate is checked against the
  ORIGINAL arcs; statuses are compared with the literal reference solvers ``glref`` and, for ``n <= 8``, with the
  brute-force oracle (the core never answers ``precondition_failed`` where the reference answers ``ok`` or vice
  versa; the oracle finds a partition whenever the core does; the core never returns ``error``).
* exactness of the greedy (O5) / batched (O1) paths: instances on which they fired are re-solved with them off,
  and every trace (``trace=True``) is REPLAYED with the exact reference primitives (``glref.graph.DiGraphState``,
  ``glref.essential.all_essential``): after every delete / contract / remove / shift event the witness ``phi``
  carried by the trace satisfies ``phi[v] ∈ Ess_G(v)`` with EXACT essential sets and exact capacity counts
  (FEAC maintenance, independent of the core's certified subsets); weighted traces are checked for FESAC
  feasibility with ``glref.weighted.min_cost_split_assignment`` (zero costs) and the carried split witness.
* degenerate inputs (``k == n``, ``k == 1``, single-capacity vectors, isolated vertices, terminals without in-arcs,
  raw arc lists with self-loops / parallel arcs / arcs out of terminals straight into ``_core``, huge weights,
  ``n == 1``, random junk), thread determinism (``threads=16`` vs ``1``), vertex-relabelling invariance, ``k > 64``.
* targeted checks for the review's watch-list: terminal index vs vertex id confusion (relabelling, trace payloads
  validated against the reference graph), stale ``phi`` after terminal removal (replay), capacity off-by-one
  (replay counts), secondary arc equal to the matching arc (replay), cycle detection with ``k = 1`` (no shift can
  ever happen, [Lem 7.10]) and ``k = 2`` (every cycle has length 2), rounding parents being original arcs.

All randomness is seeded; ``make_case(kind, seed)`` is a pure function of its arguments, so any failure is
reproducible from the ids in the assertion message.  Runnable in isolation:
``pytest tests/test_core_solver_review.py`` (``-n 8`` recommended); ``--runslow`` adds the 17-copy counterexample.
"""
from __future__ import annotations

import functools
import itertools
import json
import random
import time
from typing import Any

import pytest

from glref.assignment import find_witness
from glref.essential import all_essential
from glref.graph import DiGraphState
from glref.weighted import is_split_witness, min_cost_split_assignment, zero_cost
from glsolver import _core, generators
from glsolver.api import GLResult, core_available, glpartition
from glsolver.instance import Instance, make_instance
from glsolver.oracle.bruteforce import bruteforce_partition
from glsolver.preconditions import check_preconditions
from glsolver.verify import verify_instance_parts
from helpers_small_graphs import (
    random_capacities,
    random_digraph,
    random_kT_connected_digraph,
    random_undirected_instance,
)

pytestmark = pytest.mark.skipif(
    not (core_available("general") and core_available("weighted")), reason="C++ core solvers not built"
)

OPTION_COMBOS = [
    {"greedy_contraction": g, "lazy_shift": lz, "batch_unused_arcs": b}
    for g, lz, b in itertools.product([True, False], repeat=3)
]
LITERAL = {"greedy_contraction": False, "lazy_shift": False, "batch_unused_arcs": False}
CAP_MODES = ("balanced", "unbalanced", "extreme", "random", "zeros")
KINDS = (
    "undirected", "directed_kT", "feac_only", "dag", "random",
    "weighted_und", "weighted_dir", "weighted_dag", "weighted_random",
)
WEIGHTED_BASE = {"weighted_und": "undirected", "weighted_dir": "directed_kT", "weighted_dag": "dag",
                 "weighted_random": "random"}
GENERAL_TRACE_TYPES = {
    "init", "essential", "witness", "remove_terminal", "contract", "delete_arc", "delete_arcs", "greedy_delete",
    "matching", "criticality", "reassignment_graph", "cycle_shift", "done",
}
WEIGHTED_TRACE_TYPES = {
    "init", "essential", "min_cost_split", "remove_terminal", "contract", "matching", "criticality", "delete_arc",
    "delete_arcs", "greedy_delete", "round_and_remove", "done",
}
SEEDS_PER_BLOCK = 40
BLOCKS = 5


# =====================================================================================================
# instance generation (pure functions of (kind, seed))
# =====================================================================================================
def exact_feac_status(inst: Instance) -> tuple[bool, bool]:
    """``(FEAC holds, k-T-connected)`` with the reference primitives (fast pre-filter; the FEAC-only kind is
    confirmed with ``check_preconditions`` afterwards)."""
    g = DiGraphState.from_instance(inst)
    kappa, ess = all_essential(g)
    cap = {t: c for t, c in zip(inst.terminals, inst.capacities)}
    return find_witness(g, ess, cap) is not None, all(x == inst.k for x in kappa.values())


@functools.cache
def make_case(kind: str, seed: int, n_min: int = 3, n_max: int = 14, k_max: int = 5) -> Instance | None:
    """Seeded instance of the given kind (``None`` when the FEAC-only search finds nothing for this seed)."""
    rng = random.Random(seed * 7919 + KINDS.index(kind))
    n = rng.randint(n_min, n_max)
    k = rng.randint(1, min(k_max, n - 1))
    mode = rng.choice(CAP_MODES)
    if kind == "undirected":
        return random_undirected_instance(n, k, rng, mode, name=f"und-{seed}")
    if kind == "directed_kT":
        return random_kT_connected_digraph(n, k, rng, mode, name=f"dkt-{seed}")
    if kind == "feac_only":
        for _ in range(60):
            inst = random_digraph(n, k, rng, p=rng.choice([0.25, 0.35, 0.5, 0.7]), mode=mode, name=f"feac-{seed}")
            feac, kt = exact_feac_status(inst)
            if feac and not kt:
                pre = check_preconditions(inst, True)  # the NetworkX cross-check confirms the classification
                assert pre["feac"] is True and pre["k_T_connected"] is False, (seed, pre["message"])
                return inst
        return None
    if kind == "dag":
        return generators.random_kT_connected_dag(
            n, k, seed, rng.choice(("balanced", "unbalanced", "random", "extreme")), extra_out=rng.choice([0, 1, 2])
        )
    if kind == "random":
        return random_digraph(n, k, rng, p=rng.choice([0.15, 0.3, 0.45, 0.7]), mode=mode, name=f"rnd-{seed}")
    if kind in WEIGHTED_BASE:
        base = make_case(WEIGHTED_BASE[kind], seed, n_min, n_max, k_max)
        if base is None:
            return None
        return generators.weighted_variant(base, seed, rng.choice([1, 2, 3, 5]), rng.choice([0, 0, 1, 2, 4]))
    raise ValueError(kind)


def cases(kind: str, seeds: range) -> list[tuple[int, Instance]]:
    out = []
    for s in seeds:
        inst = make_case(kind, s)
        if inst is not None:
            out.append((s, inst))
    return out


def algorithms_for(inst: Instance) -> list[str]:
    return ["weighted"] if inst.is_weighted else ["general", "weighted"]


def reference_algorithm(inst: Instance) -> str:
    return "reference-weighted" if inst.is_weighted else "reference"


def solve(inst: Instance, algo: str, **kw: Any) -> GLResult:
    kw.setdefault("verify_preconditions", False)
    kw.setdefault("threads", 1)
    return glpartition(inst, algorithm=algo, **kw)


# =====================================================================================================
# independent checks
# =====================================================================================================
def assert_valid(res: GLResult, inst: Instance, where: Any) -> None:
    assert res.status == "ok", (where, res.status, res.message)
    rep = verify_instance_parts(inst, res.parts)
    assert rep.valid, (where, rep.errors[:5])
    assert res.valid is True, where
    assert sorted(v for p in res.parts for v in p) == list(range(inst.n)), where
    if not inst.is_weighted:
        assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities], where
        assert res.stats["roundings"] == 0, (where, "rounding on a unit-weight tight instance (§13.6)")
    check_arborescence(inst, res, where)


def check_arborescence(inst: Instance, res: GLResult, where: Any) -> None:
    """certificate['parents']: an ORIGINAL out-neighbour in the same part for every non-terminal; following the
    parents reaches the part's terminal (§13.4, [Def connected-to])."""
    parents = res.certificate["parents"]
    arcs = set(inst.arcs)
    part_of = {v: i for i, p in enumerate(res.parts) for v in p}
    tset = set(inst.terminals)
    assert set(parents) == set(range(inst.n)) - tset, where
    for v, p in parents.items():
        assert (v, p) in arcs, (where, "parent is not an original arc", v, p)
        assert part_of[v] == part_of[p], (where, "parent crosses parts", v, p)
    for v in parents:
        seen = {v}
        x = v
        while x not in tset:
            x = parents[x]
            assert x not in seen, (where, "cycle in the parent pointers")
            seen.add(x)
        assert x == inst.terminals[part_of[v]], where


class ReplayError(AssertionError):
    pass


def _essential_event_is_exact(g: DiGraphState, ev: dict[str, Any], where: str) -> None:
    kappa, ess = all_essential(g)
    got_ess = {int(v): set(ts) for v, ts in ev["ess"].items()}
    got_kappa = {int(v): int(x) for v, x in ev["kappa"].items()}
    if got_ess != ess or got_kappa != kappa:
        bad = {v: (sorted(got_ess.get(v, ())), sorted(ess.get(v, ()))) for v in set(got_ess) | set(ess)
               if got_ess.get(v) != ess.get(v)}
        raise ReplayError(f"{where}: 'essential' event differs from the exact sets: {bad}; kappa {got_kappa} vs {kappa}")


def _check_matching_event(g: DiGraphState, ev: dict[str, Any], where: str) -> None:
    pairs = [tuple(x) for x in ev["pairs"]]
    ps = [p for p, _ in pairs]
    ts = [t for _, t in pairs]
    if len(set(ps)) != len(ps) or sorted(ts) != sorted(g.terminals):
        raise ReplayError(f"{where}: matching does not saturate T with distinct pre-terminals: {pairs}")
    for p, t in pairs:
        if g.is_terminal(p) or not g.has_arc(p, t):
            raise ReplayError(f"{where}: matched pair ({p},{t}) is not an arc from a non-terminal")
        if g.out_degree(p) < 2:
            raise ReplayError(f"{where}: matched pre-terminal {p} has out-degree {g.out_degree(p)} ([Alg 2] needs >= 2)")
    for (p, q), (p2, t) in zip(map(tuple, ev["secondary"]), pairs):
        if p != p2 or q == t or not g.has_arc(p, q):
            raise ReplayError(f"{where}: secondary arc ({p},{q}) for the pair ({p2},{t}) is not a distinct arc of p")


def replay_general(inst: Instance, trace: list[dict[str, Any]]) -> dict[str, int]:
    """Replay a general-solver trace on ``DiGraphState``; after every state change the last known ``phi`` must be
    a FEAC witness [Def 5.1] w.r.t. EXACT essential sets and exact capacity counts."""
    g = DiGraphState.from_instance(inst)
    cap = {t: c for t, c in zip(inst.terminals, inst.capacities)}
    phi: dict[int, int] | None = None
    counts = {"checks": 0, "shifts": 0, "greedy": 0, "batch": 0, "contract": 0, "remove": 0, "delete": 0}

    def check(where: str) -> None:
        assert phi is not None, where
        kappa, ess = all_essential(g)
        live = set(g.nonterminals())
        if set(phi) != live:
            raise ReplayError(f"{where}: phi domain {sorted(phi)} != live non-terminals {sorted(live)}")
        for v in live:
            if phi[v] not in ess[v]:
                raise ReplayError(f"{where}: phi({v}) = {phi[v]} not in exact Ess = {sorted(ess[v])} (kappa {kappa[v]})")
        cnt = {t: 0 for t in g.terminals}
        for v in live:
            if phi[v] not in cnt:
                raise ReplayError(f"{where}: phi({v}) = {phi[v]} is not a live terminal")
            cnt[phi[v]] += 1
        for t in g.terminals:
            if cnt[t] != cap[t]:
                raise ReplayError(f"{where}: terminal {t} receives {cnt[t]} vertices, capacity {cap[t]}")
        counts["checks"] += 1

    def caps_match(ev: dict[str, Any], where: str) -> None:
        if {int(x): c for x, c in ev["capacities"].items()} != {t: cap[t] for t in g.terminals}:
            raise ReplayError(f"{where}: capacities payload {ev['capacities']} differs from {cap}")

    for idx, ev in enumerate(trace):
        typ = ev["type"]
        where = f"event {idx} ({typ})"
        if typ not in GENERAL_TRACE_TYPES:
            raise ReplayError(f"{where}: unknown event type")
        if typ == "init":
            assert ev["n"] == inst.n and ev["k"] == inst.k and ev["terminals"] == list(inst.terminals), where
            assert sorted(map(tuple, ev["arcs"])) == sorted(inst.arcs) and ev["capacities"] == list(inst.capacities), where
        elif typ == "essential":
            _essential_event_is_exact(g, ev, where)
        elif typ == "witness":
            phi = {int(v): int(t) for v, t in ev["phi"].items()}
            check(where)
        elif typ == "remove_terminal":
            t = ev["t"]
            if cap[t] != 0:
                raise ReplayError(f"{where}: removing terminal {t} with capacity {cap[t]}")
            g.remove_terminal(t)
            counts["remove"] += 1
            caps_match(ev, where)
            check(where)
        elif typ == "contract":
            p, t = ev["p"], ev["t"]
            if g.out_neighbors(p) != [t] or not g.is_terminal(t):
                raise ReplayError(f"{where}: contracting {p} (out {g.out_neighbors(p)}) into {t}")
            assert phi is not None
            if phi.get(p) != t:
                raise ReplayError(f"{where}: phi({p}) = {phi.get(p)} but contracted into {t}")
            parent = g.contract(p, t)
            if parent != ev["parent"] or (p, parent) not in set(inst.arcs):
                raise ReplayError(f"{where}: parent {ev['parent']} (reference {parent}) is not an original arc of {p}")
            cap[t] -= 1
            del phi[p]
            counts["contract"] += 1
            caps_match(ev, where)
            check(where)
        elif typ == "delete_arc":
            g.delete_arc(ev["u"], ev["v"])
            counts["delete"] += 1
            check(where)
        elif typ in ("delete_arcs", "greedy_delete"):
            for u, v in ev["arcs"]:
                g.delete_arc(u, v)
            counts["delete"] += len(ev["arcs"])
            if typ == "greedy_delete":
                p, t = ev["p"], ev["t"]
                assert phi is not None
                if g.out_neighbors(p) != [t] or phi[p] != t:
                    raise ReplayError(f"{where}: greedy target {t}, out({p}) = {g.out_neighbors(p)}, phi({p}) = {phi[p]}")
                counts["greedy"] += 1
            else:
                counts["batch"] += 1
            check(where)
        elif typ == "matching":
            _check_matching_event(g, ev, where)
        elif typ == "cycle_shift":
            assert phi is not None
            new_phi = {int(v): int(t) for v, t in ev["phi"].items()}
            changed = {v for v, _o, _n in ev["changes"]}
            for v, old, new in ev["changes"]:
                if phi.get(v) != old or new_phi.get(v) != new or old == new:
                    raise ReplayError(f"{where}: change ({v},{old},{new}) inconsistent with phi")
            for v in set(phi) | set(new_phi):
                if v not in changed and phi.get(v) != new_phi.get(v):
                    raise ReplayError(f"{where}: phi({v}) changed outside the cycle")
            if len(changed) < 2 or len(changed) > g.k:
                raise ReplayError(f"{where}: reassignment cycle of length {len(changed)} with k = {g.k}")
            phi = new_phi
            counts["shifts"] += 1
            check(where)
        elif typ == "done":
            parts = {int(t): sorted(vs) for t, vs in ev["parts"].items()}
            if sorted(v for vs in parts.values() for v in vs) != list(range(inst.n)) or g.num_nonterminals() != 0:
                raise ReplayError(f"{where}: parts do not cover V or non-terminals remain")
    if trace[-1]["type"] != "done" or counts["checks"] == 0:
        raise ReplayError("trace does not end with 'done' or performed no check")
    return counts


def replay_weighted(inst: Instance, trace: list[dict[str, Any]]) -> dict[str, int]:
    """Replay a weighted-solver trace: FESAC [Def 5.3] must hold (zero-cost min-cost split assignment over the
    EXACT essential sets) after every state change, and the carried split witness psi (reconstructed from the
    ``min_cost_split`` / ``greedy_delete`` / ``contract`` / ``round_and_remove`` events) must be a witness."""
    g = DiGraphState.from_instance(inst)
    tset = set(inst.terminals)
    cap = {t: c for t, c in zip(inst.terminals, inst.capacities)}
    w = {v: (1 if inst.weights is None else int(inst.weights[v])) for v in range(inst.n) if v not in tset}
    psi: dict[tuple[int, int], int] | None = None
    counts = {"checks": 0, "greedy": 0, "batch": 0, "contract": 0, "remove": 0, "delete": 0, "round": 0, "split": 0}

    def check(where: str) -> None:
        kappa, ess = all_essential(g)
        if min_cost_split_assignment(g, ess, cap, w, zero_cost) is None:
            raise ReplayError(f"{where}: FESAC fails over the exact essential sets")
        if psi is not None:
            ok, why = is_split_witness(g, psi, ess, cap, w)
            if not ok:
                raise ReplayError(f"{where}: carried psi is not a split witness: {why}")
        counts["checks"] += 1

    def caps_match(ev: dict[str, Any], where: str) -> None:
        if {int(x): c for x, c in ev["capacities"].items()} != {t: cap[t] for t in g.terminals}:
            raise ReplayError(f"{where}: capacities payload {ev['capacities']} differs from {cap}")

    def drop_psi(v: int) -> None:
        assert psi is not None
        for key in [key for key in psi if key[0] == v]:
            del psi[key]

    for idx, ev in enumerate(trace):
        typ = ev["type"]
        where = f"event {idx} ({typ})"
        if typ not in WEIGHTED_TRACE_TYPES:
            raise ReplayError(f"{where}: unknown event type")
        if typ == "init":
            assert ev["n"] == inst.n and ev["k"] == inst.k and ev["terminals"] == list(inst.terminals), where
            assert sorted(map(tuple, ev["arcs"])) == sorted(inst.arcs) and ev["capacities"] == list(inst.capacities), where
            assert ev["weights"] == [0 if v in tset else w[v] for v in range(inst.n)], where
            check(where)
        elif typ == "essential":
            _essential_event_is_exact(g, ev, where)
        elif typ == "min_cost_split":
            psi = {}
            for v, t, units in ev["psi"]:
                if (v, t) in psi or units <= 0:
                    raise ReplayError(f"{where}: bad psi entry ({v},{t},{units})")
                psi[(v, t)] = units
            counts["split"] += 1
            check(where)
        elif typ == "remove_terminal":
            t = ev["t"]
            if cap[t] != 0:
                raise ReplayError(f"{where}: removing terminal {t} with capacity {cap[t]}")
            if psi is not None and any(tt == t for (_v, tt) in psi):
                raise ReplayError(f"{where}: psi sends weight to the removed terminal {t}")
            g.remove_terminal(t)
            counts["remove"] += 1
            caps_match(ev, where)
            check(where)
        elif typ == "contract":
            p, t = ev["p"], ev["t"]
            if g.out_neighbors(p) != [t] or not g.is_terminal(t):
                raise ReplayError(f"{where}: contracting {p} (out {g.out_neighbors(p)}) into {t}")
            assert psi is not None
            mine = {tt: u for (v, tt), u in psi.items() if v == p}
            if mine != {t: w[p]}:
                raise ReplayError(f"{where}: psi({p}) = {mine} but {p} is contracted into {t} with weight {w[p]}")
            if cap[t] < w[p]:
                raise ReplayError(f"{where}: c_{t} = {cap[t]} < w_{p} = {w[p]} [Lem 8.2]")
            parent = g.contract(p, t)
            if parent != ev["parent"] or (p, parent) not in set(inst.arcs):
                raise ReplayError(f"{where}: parent {ev['parent']} (reference {parent}) is not an original arc of {p}")
            cap[t] -= w[p]
            drop_psi(p)
            w.pop(p)
            counts["contract"] += 1
            caps_match(ev, where)
            check(where)
        elif typ == "delete_arc":
            g.delete_arc(ev["u"], ev["v"])
            counts["delete"] += 1
            check(where)
        elif typ in ("delete_arcs", "greedy_delete"):
            for u, v in ev["arcs"]:
                g.delete_arc(u, v)
            counts["delete"] += len(ev["arcs"])
            if typ == "greedy_delete":
                p, t = ev["p"], ev["t"]
                if g.out_neighbors(p) != [t]:
                    raise ReplayError(f"{where}: after the greedy deletion out({p}) = {g.out_neighbors(p)} != [{t}]")
                assert psi is not None
                drop_psi(p)
                psi[(p, t)] = w[p]
                counts["greedy"] += 1
            else:
                counts["batch"] += 1
            check(where)
        elif typ == "matching":
            _check_matching_event(g, ev, where)
        elif typ == "round_and_remove":
            S = list(ev["S"])
            pairs = [tuple(x) for x in ev["pairs"]]
            t_S = ev["t_S"]
            if t_S != S[0] or len(pairs) != len(S) - 1 or {t for t, _ in pairs} != set(S) - {t_S}:
                raise ReplayError(f"{where}: rounding payload inconsistent: S={S} pairs={pairs} t_S={t_S}")
            if any(cap[t] <= 0 for t in S):
                raise ReplayError(f"{where}: rounding with a zero-capacity terminal in S")
            pt = sorted(g.pre_terminals(S))
            if sorted(p for _, p in pairs) != pt:
                raise ReplayError(f"{where}: matched pre-terminals {sorted(p for _, p in pairs)} != PT(G,S) = {pt} [Lem 7.6]")
            for t, p in pairs:
                if not g.has_arc(p, t) or w[p] > cap[t] + inst.w_max - 1:
                    raise ReplayError(f"{where}: rounded pair ({t},{p}) is not an arc or exceeds the bound [Lem 8.6]")
            if psi is not None:
                survivors = set(g.nonterminals()) - set(pt)
                if any(v in survivors and t in S for (v, t) in psi):
                    raise ReplayError(f"{where}: a survivor sends weight into the deficient set [Lem 7.7]")
                for p in pt:
                    drop_psi(p)
            for p in pt:
                g.remove_vertex(p)
                w.pop(p)
            for t in S:
                g.remove_vertex(t)
            counts["round"] += 1
            check(where)
        elif typ == "done":
            parts = {int(t): sorted(vs) for t, vs in ev["parts"].items()}
            if sorted(v for vs in parts.values() for v in vs) != list(range(inst.n)) or g.num_nonterminals() != 0:
                raise ReplayError(f"{where}: parts do not cover V or non-terminals remain")
    if trace[-1]["type"] != "done" or counts["checks"] == 0:
        raise ReplayError("trace does not end with 'done' or performed no check")
    return counts


def replay(inst: Instance, algo: str, res: GLResult) -> dict[str, int]:
    assert res.trace is not None and res.trace[0]["type"] == "init"
    return replay_weighted(inst, res.trace) if algo == "weighted" else replay_general(inst, res.trace)


# =====================================================================================================
# 1. differential fuzz: every kind x every option combination x debug asserts, reference + oracle statuses
# =====================================================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("block", range(BLOCKS))
def test_differential_fuzz(kind: str, block: int) -> None:
    seeds = range(block * SEEDS_PER_BLOCK, (block + 1) * SEEDS_PER_BLOCK)
    pool = cases(kind, seeds)
    assert len(pool) >= SEEDS_PER_BLOCK // 2, (kind, block, len(pool))
    statuses: dict[str, int] = {}
    for seed, inst in pool:
        assert 3 <= inst.n <= 14 and 1 <= inst.k <= 5
        ref = glpartition(inst, algorithm=reference_algorithm(inst), verify_preconditions=False)
        assert ref.status in ("ok", "precondition_failed"), (kind, seed, ref.status, ref.message)
        if ref.status == "ok":
            assert ref.valid is True, (kind, seed, "reference produced an invalid partition", ref.message)
        oracle = bruteforce_partition(inst, time_limit=30)[0] if inst.n <= 8 else None
        statuses[ref.status] = statuses.get(ref.status, 0) + 1
        for algo in algorithms_for(inst):
            for opts in OPTION_COMBOS:
                where = (kind, seed, algo, opts)
                res = solve(inst, algo, options=dict(opts), debug=True)
                assert res.status != "error", (where, res.message)
                assert res.status == ref.status, (where, res.status, ref.status, res.message)
                if res.status == "ok":
                    assert_valid(res, inst, where)
                    if oracle is not None:
                        assert oracle == "ok", (where, "core found a partition but the exhaustive oracle says", oracle)
                    st = res.stats
                    assert st["greedy_successes"] <= st["greedy_attempts"], where
                    if not opts["greedy_contraction"]:
                        assert st["greedy_attempts"] == 0 == st["greedy_successes"], where
                    if not opts["batch_unused_arcs"] and not opts["greedy_contraction"]:
                        assert st["batched_deletions"] == 0, where
                    if not inst.is_weighted:
                        assert st["contractions"] == inst.n - inst.k, where
                else:
                    assert "Flow-Essential" in res.message and res.parts == [] and res.valid is None, where
                    assert "no partition" not in res.message.lower().replace("a partition may still exist", ""), where
    assert statuses.get("ok", 0) > 0, (kind, block, statuses)
    if kind in ("random", "weighted_random"):
        assert statuses.get("precondition_failed", 0) > 0, (kind, block, statuses)


def test_mechanisms_are_exercised_by_the_fuzz_pools() -> None:
    """The differential fuzz is only meaningful if the optimized paths actually run: count them on the pools
    (default options)."""
    tot = {"cycle_shifts": 0, "greedy_successes": 0, "batched_deletions": 0, "terminal_removals": 0, "roundings": 0,
           "shift_calls": 0, "feac_only": 0, "weighted": 0, "zero_caps": 0}
    for kind in KINDS:
        for seed, inst in cases(kind, range(60)):
            tot["weighted"] += inst.is_weighted
            tot["zero_caps"] += 0 in inst.capacities
            for algo in algorithms_for(inst):
                res = solve(inst, algo)
                if res.status != "ok":
                    continue
                for key in ("cycle_shifts", "greedy_successes", "batched_deletions", "terminal_removals", "roundings", "shift_calls"):
                    tot[key] += res.stats[key]
                tot["feac_only"] += not res.stats["k_T_connected"]
    assert tot["cycle_shifts"] >= 30 and tot["shift_calls"] >= 50, tot
    assert tot["greedy_successes"] >= 500 and tot["batched_deletions"] >= 100, tot
    assert tot["terminal_removals"] >= 100 and tot["roundings"] >= 5, tot
    assert tot["feac_only"] >= 100 and tot["weighted"] >= 150 and tot["zero_caps"] >= 100, tot


# =====================================================================================================
# 2. greedy / batched paths are exact; trace replay with the exact reference primitives
# =====================================================================================================
@pytest.mark.parametrize("kind", KINDS)
def test_greedy_and_batch_paths_rerun_without_them(kind: str) -> None:
    fired = 0
    for seed, inst in cases(kind, range(80)):
        for algo in algorithms_for(inst):
            res = solve(inst, algo, debug=True)
            if res.status != "ok":
                continue
            st = res.stats
            if st["greedy_successes"] == 0 and st["batched_deletions"] == 0:
                continue
            fired += 1
            assert_valid(res, inst, (kind, seed, algo, "optimized"))
            plain = solve(inst, algo, options=LITERAL, debug=True)
            assert plain.status == "ok", (kind, seed, algo, plain.message)
            assert_valid(plain, inst, (kind, seed, algo, "literal"))
            assert plain.stats["greedy_successes"] == 0 == plain.stats["batched_deletions"], (kind, seed, algo)
            if algo == "general":  # literal [Alg 1]: every deletion is one ShiftAssignment call
                assert plain.stats["deletions"] == plain.stats["shift_calls"], (kind, seed, algo)
    assert fired >= 20, (kind, fired)


@pytest.mark.parametrize("kind", KINDS)
def test_trace_replay_with_exact_reference_primitives(kind: str) -> None:
    """FEAC / FESAC maintenance checked independently of the core's certified subsets (P2)."""
    totals: dict[str, int] = {}
    replayed = 0
    for seed, inst in cases(kind, range(25)):
        for algo in algorithms_for(inst):
            for opts in OPTION_COMBOS:
                for debug in ((False, True) if opts["greedy_contraction"] else (False,)):
                    res = solve(inst, algo, options=dict(opts), debug=debug, trace=True, threads=2)
                    assert res.status != "error", (kind, seed, algo, opts, debug, res.message)
                    if res.status != "ok":
                        assert res.trace is not None and res.trace[0]["type"] == "init"
                        continue
                    try:
                        c = replay(inst, algo, res)
                    except ReplayError as e:
                        pytest.fail(f"{kind} seed {seed} {algo} {opts} debug={debug}: {e}")
                    replayed += 1
                    for key, val in c.items():
                        totals[key] = totals.get(key, 0) + val
                    # the trace agrees with the counters the binding reports
                    tr = res.trace
                    st = res.stats
                    assert st["contractions"] == sum(1 for ev in tr if ev["type"] == "contract"), (kind, seed, algo, opts)
                    assert st["deletions"] == sum(1 if ev["type"] == "delete_arc" else len(ev["arcs"]) for ev in tr
                                                  if ev["type"] in ("delete_arc", "delete_arcs", "greedy_delete")), (kind, seed, algo, opts)
                    assert st["terminal_removals"] == sum(1 for ev in tr if ev["type"] == "remove_terminal") + \
                        sum(len(ev["S"]) for ev in tr if ev["type"] == "round_and_remove"), (kind, seed, algo, opts)
                    if algo == "general":
                        assert st["cycle_shifts"] == c["shifts"] and st["greedy_successes"] == c["greedy"], (kind, seed, opts)
                    else:
                        assert st["roundings"] == c["round"] and st["greedy_successes"] == c["greedy"], (kind, seed, opts)
                    # the oracle books every committed deletion of >= 2 arcs (O1 batch or O5 greedy) as "batched"
                    assert st["batched_deletions"] == sum(1 for ev in tr if ev["type"] in ("delete_arcs", "greedy_delete")
                                                          and len(ev["arcs"]) > 1), (kind, seed, algo, opts)
    assert replayed >= 40, (kind, replayed)
    assert totals["checks"] >= 500 and totals["contract"] >= 100, (kind, totals)


def test_replay_covers_shifts_greedy_batch_and_rounding() -> None:
    tot: dict[str, int] = {}
    for kind in ("feac_only", "random", "weighted_random", "weighted_und"):
        for seed, inst in cases(kind, range(40)):
            for algo in algorithms_for(inst):
                for opts in (OPTION_COMBOS[0], LITERAL):
                    res = solve(inst, algo, options=dict(opts), trace=True)
                    if res.status == "ok":
                        for key, val in replay(inst, algo, res).items():
                            tot[key] = tot.get(key, 0) + val
    assert tot["shifts"] >= 20 and tot["greedy"] >= 200 and tot["batch"] >= 50 and tot["remove"] >= 50, tot
    assert tot["round"] >= 3, tot


# =====================================================================================================
# 3. the review watch-list: k = 1 / k = 2 cycles, relabelling, initial witness, rounding parents
# =====================================================================================================
def test_k1_never_shifts_and_k2_cycles_have_length_two() -> None:
    """[Lem 7.10]: e_i is never critical for (v, t_i); with k = 1 every secondary arc is therefore non-critical
    and no cycle shift can occur; with k = 2 the reassignment cycle is always t_1 -> t_2 -> t_1."""
    rng = random.Random(21)
    seen = {1: 0, 2: 0}
    shifts2 = 0
    for i in range(160):
        k = 1 + (i % 2)
        n = rng.randint(3, 12)
        if rng.random() < 0.5:
            inst = random_digraph(n, k, rng, p=rng.choice([0.2, 0.4, 0.7]), mode=rng.choice(CAP_MODES))
        else:
            inst = random_undirected_instance(n, k, rng, rng.choice(CAP_MODES))
        for algo in ("general", "weighted"):
            for opts in (OPTION_COMBOS[0], LITERAL):
                res = solve(inst, algo, options=dict(opts), debug=True, trace=True)
                assert res.status != "error", (i, algo, opts, res.message)
                if res.status != "ok":
                    continue
                assert_valid(res, inst, (i, algo, opts))
                replay(inst, algo, res)
                seen[k] += 1
                if k == 1:
                    assert res.stats["cycle_shifts"] == 0, (i, algo, opts)
                    assert not any(ev["type"] == "cycle_shift" for ev in res.trace), (i, algo, opts)
                elif algo == "general":
                    for ev in res.trace:
                        if ev["type"] == "reassignment_graph":
                            assert sorted(ev["cycle"]) == sorted(inst.terminals) and len(ev["cycle"]) == 2, (i, opts, ev)
                        if ev["type"] == "cycle_shift":
                            assert len(ev["changes"]) == 2, (i, opts, ev)
                            shifts2 += 1
    assert seen[1] >= 60 and seen[2] >= 60, seen
    assert shifts2 >= 5, "no cycle shift with k = 2 was exercised"


def _relabel(inst: Instance, perm: list[int], term_order: list[int]) -> Instance:
    """Vertex ``v`` becomes ``perm[v]``; the terminals are listed in the order ``term_order`` (a permutation of
    the terminal indices) with their capacities / weights carried along."""
    edges = inst.arcs if inst.directed else inst.undirected_edges
    new_edges = [(perm[u], perm[v]) for u, v in edges]
    terms = [perm[inst.terminals[i]] for i in term_order]
    caps = [inst.capacities[i] for i in term_order]
    weights = None
    if inst.weights is not None:
        weights = [0] * inst.n
        for v in range(inst.n):
            weights[perm[v]] = inst.weights[v]
    return make_instance(inst.n, new_edges, terms, caps, weights=weights, directed=inst.directed, name=inst.name + "_perm")


def test_vertex_and_terminal_relabelling_invariance() -> None:
    """Terminal index vs vertex id confusion would show up as a different status or an invalid answer once the
    vertices are permuted and the terminals are listed in another order (FEAC / FESAC are label-independent)."""
    rng = random.Random(99)
    n_ok = 0
    for kind in KINDS:
        for seed, inst in cases(kind, range(12)):
            perm = list(range(inst.n))
            rng.shuffle(perm)
            term_order = list(range(inst.k))
            rng.shuffle(term_order)
            inst2 = _relabel(inst, perm, term_order)
            assert inst2.k == inst.k and inst2.m == inst.m
            for algo in algorithms_for(inst):
                r1 = solve(inst, algo, debug=True)
                r2 = solve(inst2, algo, debug=True)
                assert r1.status == r2.status, (kind, seed, algo, r1.status, r2.status, r2.message)
                if r1.status == "ok":
                    assert_valid(r1, inst, (kind, seed, algo, "original"))
                    assert_valid(r2, inst2, (kind, seed, algo, "relabelled"))
                    n_ok += 1
                    if algo == "general":
                        # the reported witness is indexed by ORIGINAL terminal index and vertex id
                        wit2 = r2.certificate["witness"]
                        assert set(wit2) == set(range(inst2.n)) - set(inst2.terminals)
                        assert all(t in inst2.terminals for t in wit2.values())
    assert n_ok >= 60


def test_initial_witness_is_an_exact_feac_witness() -> None:
    """certificate['witness'] of the general solver (paper_notes §7.3): phi(v) essential for v (exact sets of the
    ORIGINAL instance) and |phi^{-1}(t)| = c_t; the binding reports -1 for terminals."""
    checked = 0
    for kind in ("undirected", "directed_kT", "feac_only", "dag", "random"):
        for seed, inst in cases(kind, range(25)):
            res = solve(inst, "general", debug=True)
            raw = _core.solve_general(inst.n, [list(a) for a in inst.arcs], list(inst.terminals), list(inst.capacities), {})
            assert raw["status"] == res.status
            if res.status != "ok":
                continue
            g = DiGraphState.from_instance(inst)
            _kappa, ess = all_essential(g)
            wit = res.certificate["witness"]
            assert set(wit) == set(ess), (kind, seed)
            counts = {t: 0 for t in inst.terminals}
            for v, t in wit.items():
                assert t in ess[v], (kind, seed, v, t, sorted(ess[v]))
                counts[t] += 1
            assert [counts[t] for t in inst.terminals] == list(inst.capacities), (kind, seed)
            assert len(raw["witness"]) == inst.n and all(raw["witness"][t] == -1 for t in inst.terminals)
            assert all(inst.terminals[raw["witness"][v]] == wit[v] for v in wit), (kind, seed)
            assert raw["parent"] == [res.certificate["parents"].get(v, -1) for v in range(inst.n)]
            checked += 1
    assert checked >= 80


def _heavy_vertex_instance(rng: random.Random) -> Instance:
    """A weight-w_max vertex adjacent to every terminal on a k-T-connected DAG-plus-back-arcs base: RoundAndRemove
    is unavoidable on some of these (as in tests/test_core_weighted.py)."""
    n = rng.randint(4, 9)
    k = rng.randint(2, min(4, n - 1))
    terminals = rng.sample(range(n), k)
    nonterms = [v for v in range(n) if v not in terminals]
    rng.shuffle(nonterms)
    order = nonterms + terminals
    pos = {v: i for i, v in enumerate(order)}
    arcs: set[tuple[int, int]] = set()
    for i, v in enumerate(nonterms):
        later = order[i + 1:]
        for x in rng.sample(later, rng.randint(k, min(len(later), k + 2))):
            arcs.add((v, x))
    if len(nonterms) >= 2:
        u, v = rng.sample(nonterms, 2)
        if pos[u] > pos[v]:
            arcs.add((u, v))
    w_max = rng.randint(3, 5)
    weights = [0 if v in terminals else rng.randint(1, w_max) for v in range(n)]
    heavy = nonterms[0]
    weights[heavy] = w_max
    arcs |= {(heavy, t) for t in terminals}
    total = sum(weights)
    cuts = sorted(rng.randint(0, total) for _ in range(k - 1))
    caps = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    return make_instance(n, sorted(arcs), terminals, caps, weights=weights, directed=True)


def test_rounding_parents_are_original_arcs_and_bound_holds() -> None:
    rng = random.Random(2025)
    rounded = 0
    for i in range(150):
        inst = _heavy_vertex_instance(rng)
        ref = glpartition(inst, algorithm="reference-weighted", verify_preconditions=False)
        for opts in OPTION_COMBOS:
            res = solve(inst, "weighted", options=dict(opts), debug=True, trace=True)
            assert res.status == ref.status, (i, opts, res.status, ref.status, res.message)
            if res.status != "ok":
                continue
            assert_valid(res, inst, (i, opts))
            c = replay(inst, "weighted", res)
            rep = verify_instance_parts(inst, res.parts)
            if res.stats["roundings"] == 0:
                assert rep.details["exceeds_capacity"] == [], (i, opts)  # only rounding can exceed c_t [Lem 8.6]
            else:
                rounded += 1
                assert c["round"] == res.stats["roundings"]
                for ev in res.trace:
                    if ev["type"] == "round_and_remove":
                        for t, p in ev["pairs"]:
                            assert p in res.parts[inst.terminals.index(t)], (i, opts, ev)
                            assert (p, res.certificate["parents"][p]) in set(inst.arcs)
                            assert res.certificate["parents"][p] in res.parts[inst.terminals.index(t)]
    assert rounded >= 10, rounded


# =====================================================================================================
# 4. degenerate inputs and the raw binding
# =====================================================================================================
def _parts_from(out: dict[str, Any], n: int, k: int) -> list[list[int]]:
    parts: list[list[int]] = [[] for _ in range(k)]
    for v, i in enumerate(out["assignment"]):
        parts[i].append(v)
    return parts


def test_degenerate_inputs_through_the_api() -> None:
    # n == 1, k == 1
    inst = make_instance(1, [], [0], [0], directed=True)
    for algo in ("general", "weighted", "reference", "reference-weighted"):
        res = glpartition(inst, algorithm=algo, debug=True)
        assert res.status == "ok" and res.parts == [[0]], algo
    # k == n (nothing to do): every capacity 0; and with a weighted slack
    inst = make_instance(4, [], [2, 0, 3, 1], [0, 0, 0, 0], directed=True)
    for algo in ("general", "weighted"):
        res = solve(inst, algo, debug=True)
        assert res.status == "ok" and res.parts == [[2], [0], [3], [1]] and res.certificate["parents"] == {}, algo
    inst = make_instance(3, [], [0, 1, 2], [4, 0, 1], weights=[0, 0, 0], directed=True)
    res = solve(inst, "weighted", debug=True)
    assert res.status == "ok" and res.parts == [[0], [1], [2]]
    # k == 1: contractions along a BFS tree (§13.10); all capacity in the single terminal; an unreachable vertex
    for seed in range(20):
        rng = random.Random(seed)
        n = rng.randint(2, 12)
        arcs = [(v, rng.randrange(v)) for v in range(1, n)] + [(rng.randrange(1, n), rng.randrange(n)) for _ in range(n)]
        arcs = [(u, v) for u, v in arcs if u != v]
        inst = make_instance(n, arcs, [0], [n - 1], directed=True)
        for algo in ("general", "weighted"):
            res = solve(inst, algo, debug=True, trace=True)
            assert_valid(res, inst, (seed, algo))
            assert res.stats["cycle_shifts"] == 0 and res.stats["contractions"] == n - 1
            replay(inst, algo, res)
    inst = make_instance(5, [(1, 0), (2, 0), (3, 1)], [0], [4], directed=True)  # vertex 4 reaches nothing (§13.9)
    for algo in ("general", "weighted"):
        res = solve(inst, algo)
        assert res.status == "precondition_failed" and "[4]" in res.message and "no essential terminal" in res.message
        assert glpartition(inst, algorithm=reference_algorithm(inst)).status == "precondition_failed"
    # all capacity in one terminal (all-in-one) on a 3-connected undirected graph, other capacities 0
    inst = generators.harary_graph(9, 3, seed=1)
    inst = make_instance(inst.n, inst.undirected_edges, inst.terminals, [6, 0, 0], directed=False)
    for algo in ("general", "weighted"):
        res = solve(inst, algo, debug=True, trace=True)
        assert_valid(res, inst, algo)
        assert res.stats["terminal_removals"] == 2 and res.parts[1] == [inst.terminals[1]] and res.parts[2] == [inst.terminals[2]]
        replay(inst, algo, res)
    # a terminal without in-arcs: fine with capacity 0, FEAC fails with capacity > 0
    inst = make_instance(4, [(2, 0), (3, 0), (3, 2)], [0, 1], [2, 0], directed=True)
    for algo in ("general", "weighted"):
        res = solve(inst, algo, debug=True)
        assert_valid(res, inst, algo)
        assert res.parts == [[0, 2, 3], [1]], algo
    inst = make_instance(4, [(2, 0), (3, 0), (3, 2)], [0, 1], [1, 1], directed=True)
    for algo in ("general", "weighted"):
        assert solve(inst, algo).status == "precondition_failed"
    # isolated vertex
    inst = make_instance(4, [(2, 0), (2, 1)], [0, 1], [1, 1], directed=True)
    for algo in ("general", "weighted"):
        res = solve(inst, algo)
        assert res.status == "precondition_failed" and "[3]" in res.message
    # huge weights (2^40) and a huge slack; a sum that overflows int64 is reported as an error, never a crash
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [2**40, 2**40], weights=[0, 0, 2**41], directed=True)
    res = solve(inst, "weighted", debug=True)
    assert_valid(res, inst, "2^41")
    assert res.stats["roundings"] == 1
    out = _core.solve_weighted(4, [(2, 0), (3, 1)], [0, 1], [2**62, 2**62], [0, 0, 2**62, 2**62], {})
    assert out["status"] == "error" and "overflow" in out["message"]
    out = _core.solve_weighted(3, [(2, 0), (2, 1)], [0, 1], [2**62, 2**62], [0, 0, 5], {})
    assert out["status"] == "error" and "overflow" in out["message"]


def test_raw_binding_normalizes_and_never_crashes() -> None:
    # self-loops, parallel arcs and arcs out of terminals are dropped by the core exactly like make_instance
    raw = [(0, 0), (2, 0), (2, 0), (0, 2), (1, 2), (2, 1), (1, 1), (0, 1)]
    out = _core.solve_general(3, raw, [0], [2], {"debug_asserts": True, "trace": True})
    assert out["status"] == "ok" and out["assignment"] == [0, 0, 0]
    inst = make_instance(3, raw, [0], [2], directed=True)
    assert verify_instance_parts(inst, _parts_from(out, 3, 1)).valid
    init = next(json.loads(p) for t, p in out["trace"] if t == "init")
    assert sorted(map(tuple, init["arcs"])) == sorted(inst.arcs)
    outw = _core.solve_weighted(3, raw, [0], [2], [0, 1, 1], {"debug_asserts": True})
    assert outw["status"] == "ok" and outw["assignment"] == [0, 0, 0] and outw["witness"] == [-1, -1, -1]
    # malformed data -> status "error" with a message, never an exception
    bad = [
        (0, [], [], []), (-1, [], [], []), (3, [(0, 5)], [1], [2]), (3, [(-1, 0)], [1], [2]), (3, [(0, 1)], [7], [2]),
        (3, [(0, 1)], [1, 1], [1, 1]), (3, [(0, 1), (2, 1)], [1], [-2]), (3, [(0, 1)], [1], [1, 1]),
    ]
    for n, arcs, terms, caps in bad:
        out = _core.solve_general(n, arcs, terms, caps, {})
        assert out["status"] == "error" and out["message"], (n, arcs, terms, caps, out["status"])
        outw = _core.solve_weighted(n, arcs, terms, caps, [1] * max(n, 0), {})
        assert outw["status"] == "error" and outw["message"], (n, arcs, terms, caps, outw["status"])
    assert _core.solve_weighted(3, [(2, 0)], [0, 1], [1, 1], [0, 0, 0], {})["status"] == "error"  # weight 0
    assert _core.solve_weighted(3, [(2, 0)], [0, 1], [1, 1], [0, 0], {})["status"] == "error"  # wrong length
    # random junk (vertex ids out of range, duplicate terminals, capacity vectors of the wrong length, negative
    # weights, all option combinations, up to 16 threads): every call returns a dict; every "ok" is verified
    rng = random.Random(1)
    seen = {"ok": 0, "error": 0, "precondition_failed": 0}
    for i in range(3000):
        n = rng.randint(0, 7)
        arcs = [(rng.randint(-1, n), rng.randint(-1, n)) for _ in range(rng.randint(0, 20))]
        k = rng.randint(0, n)
        terms = [rng.randint(-1, n) for _ in range(k)] if rng.random() < 0.3 else rng.sample(range(n), k)
        caps = [rng.randint(-1, n) for _ in range(len(terms) + rng.choice([0, 0, 0, 1, -1]))]
        opts = {"threads": rng.choice([0, 1, 2, 16]), "debug_asserts": True, "trace": rng.random() < 0.5,
                "greedy_contraction": rng.random() < 0.5, "lazy_shift": rng.random() < 0.5,
                "batch_unused_arcs": rng.random() < 0.5, "record_cuts": rng.random() < 0.5, "seed": rng.randint(-5, 5)}
        out = _core.solve_general(n, arcs, terms, caps, opts)
        w = [rng.randint(-1, 4) for _ in range(n + rng.choice([0, 0, 1]))]
        outw = _core.solve_weighted(n, arcs, terms, caps, w, opts)
        for o, weights in ((out, None), (outw, w)):
            assert o["status"] in seen, (i, o["status"])
            seen[o["status"]] += 1
            if o["status"] == "ok":
                inst = make_instance(n, arcs, terms, caps, weights=weights, directed=True)
                assert verify_instance_parts(inst, _parts_from(o, n, len(terms))).valid, (i, arcs, terms, caps, weights)
    assert seen["ok"] >= 3 and seen["precondition_failed"] >= 20 and seen["error"] >= 1000, seen


# =====================================================================================================
# 5. determinism across thread counts (results and counters), including threads=0 (auto)
# =====================================================================================================
def test_threads_16_vs_1_determinism_on_50_instances() -> None:
    rng = random.Random(3)
    pool: list[Instance] = []
    while len(pool) < 50:
        kind = rng.choice([k for k in KINDS if k != "feac_only"])  # (the NetworkX confirmation is too slow at n = 40)
        inst = make_case(kind, 1000 + len(pool), n_min=8, n_max=40, k_max=6)
        if inst is not None:
            pool.append(inst)
    keys = ("steps", "contractions", "deletions", "cycle_shifts", "greedy_attempts", "greedy_successes", "batched_deletions",
            "shift_calls", "terminal_removals", "roundings", "min_cost_flow_calls", "max_flow_calls", "augment_calls", "cut_calls")
    n_ok = 0
    for i, inst in enumerate(pool):
        for algo in algorithms_for(inst):
            r1 = solve(inst, algo, threads=1)
            r16 = solve(inst, algo, threads=16)
            r0 = solve(inst, algo, threads=0, seed=12345)
            assert r1.status == r16.status == r0.status, (i, algo, r1.status, r16.status, r0.status)
            assert r1.status != "error", (i, algo, r1.message)
            if r1.status == "ok":
                assert_valid(r16, inst, (i, algo))
                n_ok += 1
            assert r1.parts == r16.parts == r0.parts, (i, algo)
            assert r1.certificate == r16.certificate == r0.certificate, (i, algo)
            assert {k: r1.stats[k] for k in keys} == {k: r16.stats[k] for k in keys} == {k: r0.stats[k] for k in keys}, (i, algo)
            t1 = solve(inst, algo, threads=1, trace=True).trace
            t16 = solve(inst, algo, threads=16, trace=True).trace
            assert t1 == t16, (i, algo, "traces differ between thread counts")
    assert n_ok >= 60


def test_concurrent_calls_from_python_threads_do_not_interfere() -> None:
    """run() releases the GIL and the oracle spawns worker threads: several solves in flight at once (each with
    its own worker pool) must produce exactly the sequential results."""
    from concurrent.futures import ThreadPoolExecutor

    pool = [inst for kind in KINDS for _s, inst in cases(kind, range(6))]
    jobs = [(inst, algo) for inst in pool for algo in algorithms_for(inst)]
    sequential = [(r.status, r.parts, r.certificate, r.stats["steps"]) for r in (solve(i, a, threads=3) for i, a in jobs)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        concurrent = list(ex.map(lambda ia: solve(ia[0], ia[1], threads=3), jobs * 2))
    for j, res in enumerate(concurrent):
        inst, algo = jobs[j % len(jobs)]
        assert (res.status, res.parts, res.certificate, res.stats["steps"]) == sequential[j % len(jobs)], (j, algo)
        if res.status == "ok":
            assert verify_instance_parts(inst, res.parts).valid
    assert sum(1 for s in sequential if s[0] == "ok") >= 40


def test_k_above_64_uses_multiword_bitsets() -> None:
    """O9: terminal sets are bitsets; k > 64 needs more than one word."""
    rng = random.Random(64)
    for inst in (generators.complete_graph(80, 70, seed=1), generators.harary_graph(100, 66, seed=2)):
        for mode in ("balanced", "extreme", "zeros"):
            caps = list(random_capacities(inst.n - inst.k, inst.k, rng, mode))
            inst2 = make_instance(inst.n, inst.undirected_edges, inst.terminals, caps, directed=False)
            for algo in ("general", "weighted"):
                res = solve(inst2, algo, debug=True, threads=4)
                assert_valid(res, inst2, (inst.name, mode, algo))
                assert res.stats["k_T_connected"] is True
        winst = generators.weighted_variant(inst, 7, 5, 3)
        res = solve(winst, "weighted", threads=4)
        assert_valid(res, winst, (inst.name, "weighted"))


# =====================================================================================================
# 6. the official counterexample (17 copies) with the greedy contraction on and off
# =====================================================================================================
@pytest.mark.slow
def test_counterexample_copies17_greedy_on_and_off() -> None:
    from glref.counterexample import build_counterexample_instance

    inst = build_counterexample_instance(17)
    assert inst.n == 3789 and inst.k == 9
    for algo in ("general", "weighted"):
        for greedy in (True, False):
            t0 = time.perf_counter()
            res = solve(inst, algo, threads=8, options={"greedy_contraction": greedy, "debug_asserts": False})
            dt = time.perf_counter() - t0
            assert_valid(res, inst, (algo, greedy))
            st = res.stats
            print(f"\ncounterexample copies=17 {algo} greedy={greedy}: {dt:.2f}s steps={st['steps']} deletions={st['deletions']} "
                  f"cycle_shifts={st['cycle_shifts']} greedy_successes={st['greedy_successes']} batched={st['batched_deletions']}")
            assert dt < 120.0, (algo, greedy, dt)
            if not greedy:
                assert st["greedy_successes"] == 0
