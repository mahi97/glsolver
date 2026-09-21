"""Exhaustive small-instance tests (docs/verification.md, Stage D).

Undirected: every connected graph of the NetworkX atlas with ``n <= 6``
vertices, every ``k`` in ``1..min(κ(G), 3)``, every ``k``-subset of terminals
and every composition of ``n-k`` into ``k`` capacities for ``n <= 5``; for
``n == 6`` a seeded sample (3 terminal subsets, up to 6 compositions per
subset) unless ``GL_EXHAUSTIVE_FULL=1``.  Every applicable registry solver
(:func:`glsolver.testing.registry.solvers_for`) must return ``"ok"`` -- the
Győri–Lovász theorem guarantees a partition -- and the independent verifier
must accept the output; a seeded 10 % sample is also handed to the
``bruteforce`` oracle, which must find a partition as well.

Directed: every digraph on ``n <= 4`` vertices with ``k in {1, 2}`` (3278
instances) plus a seeded sample of the ``n = 6``, ``k = 3``, ``c = (1, 1, 1)``
layer (:data:`K3_LAYER`: 64 arc subsets per terminal subset, 1280
instances).  The extra layer exists because ``k <= 2`` never exercises a
reassignment cycle of length 3 in ShiftAssignment [Alg 2] -- and neither
does ``k = 3`` with ``n = 5``, where ``Σ c = 2`` forces a zero-capacity
terminal that step (i) of [Alg 1] removes before any shift.  The solvers run
on the instances that satisfy FEAC [Def 5.1] according to the NetworkX
by-definition check (:func:`glsolver.preconditions.check_preconditions`) and
must produce valid partitions; on the others they must answer
``"precondition_failed"`` -- never ``"infeasible"``, never a crash.  The
brute-force oracle additionally records how many non-FEAC instances still
admit a partition (the theorem is sufficient, not necessary).

Every theorem solver runs with ``debug=True``: the reference solvers check
the paper's invariants A1–A8 (§7.2) after every step and the C++ core
enables its ``debug_asserts``.  This matters -- on instances this small
several algorithm bugs still produce a *valid* partition, so output
verification alone would not see them (docs/verification.md §4.3).  Every
solver call is bounded by ``$GL_SOLVER_TIME_LIMIT`` seconds (default 20,
:mod:`glsolver.testing.timebound`): a hang is recorded as a ``timeout``
failure, stored under ``regression/`` and ends the enumeration, since every
further hanging instance would cost the full limit again.

A tally (instances, solver runs and cycle shifts per backend, failures) is
printed at the end and written to ``$GL_EXHAUSTIVE_TALLY`` (default: a file
in pytest's temporary directory, so the source tree stays clean).  Failures
are stored under ``regression/`` via
:func:`glsolver.testing.regression.save_regression`.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pytest

from glsolver.api import GLResult
from glsolver.preconditions import check_preconditions, is_dag
from glsolver.testing.registry import available_solvers, solvers_for
from glsolver.testing.regression import save_regression
from glsolver.testing.timebound import SolverTimeout, default_time_limit, time_bound
from glsolver.verify import verify_instance_parts
from strategies import (
    count_directed_exhaustive,
    instance_summary,
    iter_directed_exhaustive,
    iter_undirected_exhaustive,
)

MAX_N_UNDIRECTED = 6
FULL_UP_TO = 5
DIRECTED_MAX_N = 4
DIRECTED_KS = (1, 2)
#: sampled digraph layer with three live terminals of positive capacity
#: (``n = 6``, ``k = 3``, ``c = (1, 1, 1)``, 64 seeded arc subsets per terminal subset)
K3_LAYER: dict[str, Any] = {"min_n": 6, "max_n": 6, "ks": (3,), "mask_sample": 64, "min_capacity": 1}
BRUTEFORCE_FRACTION = 0.10
SEED = 20260921
TALLY_ENV = "GL_EXHAUSTIVE_TALLY"
FULL = os.environ.get("GL_EXHAUSTIVE_FULL", "") not in ("", "0", "false", "no")
ORACLES = ("bruteforce", "ilp")
#: the oracles' own limit (they answer ``status == "timeout"`` themselves)
ORACLE_TIME_LIMIT = 60.0
#: wall-clock bound of one solver call in seconds (``0`` disables it), from ``$GL_SOLVER_TIME_LIMIT``
TIME_LIMIT = default_time_limit()


def solver_kwargs(name: str) -> dict[str, Any]:
    """Per-backend options: ``debug=True`` for every theorem solver (reference
    solvers: invariants A1–A8; C++ core: ``debug_asserts``); for the oracles
    their own time limit and no (slow, redundant) NetworkX precondition
    cross-check."""
    if name in ORACLES:
        return {"verify_preconditions": False, "time_limit": ORACLE_TIME_LIMIT}
    return {"debug": True}


def time_limit_for(name: str) -> float | None:
    """Bound of one call; the self-limiting oracles get their own limit on top."""
    if TIME_LIMIT <= 0:
        return None
    return TIME_LIMIT + (ORACLE_TIME_LIMIT if name in ORACLES else 0.0)


def tally_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``$GL_EXHAUSTIVE_TALLY`` when set, else a file in pytest's temporary directory."""
    custom = os.environ.get(TALLY_ENV, "")
    if custom:
        return Path(custom)
    return tmp_path_factory.mktemp("exhaustive") / "tally.json"


# ---------------------------------------------------------------------------
# tally fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tally(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Module-wide counters, printed and written to :func:`tally_path` at teardown."""
    path = tally_path(tmp_path_factory)
    data: dict[str, Any] = {
        "undirected": {}, "directed": {}, "directed_k3_sample": {},
        "profile": "full" if FULL else "sampled-n6", "time_limit": TIME_LIMIT,
        "solvers_available": sorted(available_solvers(include_oracles=True)),
        "tally_path": str(path),
    }
    yield data
    text = json.dumps(data, indent=1, sort_keys=True)
    print("\nexhaustive tally:\n" + text)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"tally written to {path}")
    except OSError as exc:  # pragma: no cover - read-only locations
        print(f"could not write tally to {path}: {exc}")


# ---------------------------------------------------------------------------
# running one backend on one instance
# ---------------------------------------------------------------------------


def _run(fn, inst, name: str, **extra: Any) -> tuple[GLResult | None, BaseException | None]:
    """Run backend ``name`` on ``inst`` with :func:`solver_kwargs` (plus
    ``extra``) under the time bound.  Returns ``(result, None)``,
    ``(None, exception)`` for a crash or ``(None, SolverTimeout)`` for a hang
    -- nothing is hidden."""
    kw = {**solver_kwargs(name), **extra}
    try:
        with time_bound(time_limit_for(name), f"{name} on {inst.name}"):
            return fn(inst, **kw), None
    except SolverTimeout as exc:
        return None, exc
    except Exception as exc:  # a crash is a failure, never hide it
        return None, exc


def _is_ok(res: GLResult | None, inst) -> bool:
    """``ok`` status, verifier accepted, and an *independent* re-verification agrees."""
    if res is None or res.status != "ok" or res.valid is not True:
        return False
    return verify_instance_parts(inst, res.parts).valid


def _solver_failure(inst, name: str, res: GLResult | None, exc: BaseException | None) -> dict[str, Any]:
    if isinstance(exc, SolverTimeout):
        kind, msg, errors = "timeout", str(exc), []
    elif exc is not None:
        kind, msg, errors = "crash", f"{type(exc).__name__}: {exc}", []
    else:
        assert res is not None
        kind = "invalid"
        msg = f"status={res.status} valid={res.valid}: {res.message}"
        errors = list(res.verification.errors) if res.verification is not None else []
    return {"solver": name, "kind": kind, "message": msg, "errors": errors, "instance": instance_summary(inst)}


def _record_failure(counts: dict[str, Any], inst, name: str, res: GLResult | None,
                    exc: BaseException | None, *, reason: str, expect: str) -> None:
    """Book a failure, store the instance and -- for a hang -- stop the enumeration."""
    failure = _solver_failure(inst, name, res, exc)
    counts["failures"].append(failure)
    save_regression(inst, reason=f"{reason}: {name} failed: {failure['message']}",
                    extra={"solver": name, "message": failure["message"], "errors": failure["errors"],
                           "expect": expect, "kind": failure["kind"]})
    if failure["kind"] == "timeout":
        counts["timeouts"] += 1
        raise AssertionError(
            f"{name} hung on {inst.name}: {failure['message']}; enumeration stopped after "
            f"{counts['instances']} instances (every further hang would cost the full limit); "
            f"failures: {json.dumps(counts['failures'][:3], indent=1)}"
        )


def _book_run(counts: dict[str, Any], name: str, res: GLResult | None) -> None:
    counts["solver_runs"][name] = counts["solver_runs"].get(name, 0) + 1
    if res is not None and name not in ORACLES:
        shifts = int(res.stats.get("cycle_shifts", 0) or 0)
        counts["cycle_shifts"][name] = counts["cycle_shifts"].get(name, 0) + shifts


def _new_counts(**extra: Any) -> dict[str, Any]:
    return {"instances": 0, "solver_runs": {}, "cycle_shifts": {}, "failures": [], "timeouts": 0, **extra}


# ---------------------------------------------------------------------------
# undirected
# ---------------------------------------------------------------------------


def test_exhaustive_undirected_small(tally) -> None:
    rng = random.Random(SEED)
    counts = _new_counts(by_n={}, bruteforce_runs=0, max_n=MAX_N_UNDIRECTED,
                         full_up_to=MAX_N_UNDIRECTED if FULL else FULL_UP_TO)
    tally["undirected"] = counts  # shared reference: partial counts survive an abort
    t0 = time.perf_counter()
    try:
        for inst in iter_undirected_exhaustive(MAX_N_UNDIRECTED, full_up_to=FULL_UP_TO, full=FULL, seed=SEED):
            counts["instances"] += 1
            counts["by_n"][inst.n] = counts["by_n"].get(inst.n, 0) + 1
            applicable = solvers_for(inst)
            assert "reference" in applicable and "reference-weighted" in applicable
            for name, fn in applicable.items():
                res, exc = _run(fn, inst, name)
                _book_run(counts, name, res)
                if exc is not None or not _is_ok(res, inst):
                    _record_failure(counts, inst, name, res, exc, reason="exhaustive undirected", expect="ok")
            if rng.random() < BRUTEFORCE_FRACTION:
                res, exc = _run(available_solvers()["bruteforce"], inst, "bruteforce")
                counts["bruteforce_runs"] += 1
                if exc is not None or not _is_ok(res, inst):
                    _record_failure(counts, inst, "bruteforce", res, exc,
                                    reason="exhaustive undirected: bruteforce found no partition", expect="ok")
    finally:
        counts["seconds"] = round(time.perf_counter() - t0, 2)
    assert counts["instances"] > 0
    # C(n,k) * C(n-k+k-1, k-1) summed over the atlas graphs (all subsets/compositions for n <= 5)
    for n_, expected in {2: 2, 3: 12, 4: 90, 5: 685}.items():
        assert counts["by_n"][n_] == expected, counts["by_n"]
    assert counts["by_n"][6] == (8272 if FULL else 1482), counts["by_n"]
    assert counts["bruteforce_runs"] > 0
    assert counts["cycle_shifts"]["reference"] > 0, "the ShiftAssignment path was never exercised"
    assert not counts["failures"], (
        f"{len(counts['failures'])} failures, first: {json.dumps(counts['failures'][:3], indent=1)}"
    )


# ---------------------------------------------------------------------------
# directed
# ---------------------------------------------------------------------------


def _check_directed(inst, counts: dict[str, Any], bruteforce, reason: str, *, trace_cycles: bool = False) -> None:
    """One digraph instance: FEAC ⇒ every applicable backend valid; otherwise
    ``precondition_failed`` from the theorem solvers and the brute force
    decides whether a partition exists anyway.  ``trace_cycles`` records the
    lengths of the reassignment cycles of the ``reference`` solver."""
    pre = check_preconditions(inst, True)
    if pre["k_T_connected"]:
        counts["k_T_connected"] += 1
        assert pre["feac"], f"k-T-connected instance without FEAC witness: {inst.name}"
    if pre["feac"]:
        counts["feac"] += 1
        for name, fn in solvers_for(inst).items():
            extra = {"trace": True} if trace_cycles and name == "reference" else {}
            res, exc = _run(fn, inst, name, **extra)
            _book_run(counts, name, res)
            if exc is not None or not _is_ok(res, inst):
                _record_failure(counts, inst, name, res, exc, reason=f"{reason} (FEAC holds)", expect="ok")
            elif extra and res is not None and res.trace:
                for ev in res.trace:
                    if ev.get("type") == "reassignment_graph" and len(ev.get("cycle", ())) >= 3:
                        counts["cycles_len3"] += 1
        return
    counts["non_feac"] += 1
    names = ["reference", "reference-weighted"] + (["reference-dag"] if is_dag(inst) else [])
    solvers = available_solvers()
    for name in names:
        res, exc = _run(solvers[name], inst, name)
        _book_run(counts, name, res)
        if exc is not None or res is None or res.status != "precondition_failed":
            _record_failure(counts, inst, name, res, exc,
                            reason=f"{reason} (FEAC fails): expected precondition_failed",
                            expect="precondition_failed")
    res, exc = _run(bruteforce, inst, "bruteforce")
    assert exc is None and res is not None and res.status in ("ok", "infeasible"), (inst.name, exc, res)
    if res.status == "ok":
        assert res.valid
        counts["non_feac_with_partition"] += 1


def test_exhaustive_directed_tiny(tally) -> None:
    counts = _new_counts(feac=0, k_T_connected=0, non_feac=0, non_feac_with_partition=0,
                         max_n=DIRECTED_MAX_N, ks=list(DIRECTED_KS))
    k3 = _new_counts(feac=0, k_T_connected=0, non_feac=0, non_feac_with_partition=0, cycles_len3=0,
                     layer={k: (list(v) if isinstance(v, tuple) else v) for k, v in K3_LAYER.items()})
    tally["directed"] = counts
    tally["directed_k3_sample"] = k3
    bruteforce = available_solvers()["bruteforce"]
    t0 = time.perf_counter()
    t1: float | None = None
    try:
        for inst in iter_directed_exhaustive(DIRECTED_MAX_N, ks=DIRECTED_KS):
            counts["instances"] += 1
            _check_directed(inst, counts, bruteforce, "exhaustive directed")
        t1 = time.perf_counter()
        for inst in iter_directed_exhaustive(seed=SEED, **K3_LAYER):
            k3["instances"] += 1
            _check_directed(inst, k3, bruteforce, "exhaustive directed k=3 sample", trace_cycles=True)
    finally:
        counts["seconds"] = round((t1 if t1 is not None else time.perf_counter()) - t0, 2)
        k3["seconds"] = None if t1 is None else round(time.perf_counter() - t1, 2)
    assert counts["instances"] == count_directed_exhaustive(DIRECTED_MAX_N, ks=DIRECTED_KS) == 3278
    assert counts["feac"] > 0 and counts["non_feac"] > 0
    assert counts["k_T_connected"] <= counts["feac"]
    # the theorem is sufficient but not necessary: some non-FEAC instances still have partitions
    assert 0 < counts["non_feac_with_partition"] < counts["non_feac"]
    assert not counts["failures"], (
        f"{len(counts['failures'])} failures, first: {json.dumps(counts['failures'][:3], indent=1)}"
    )
    # the sampled k = 3 layer: deterministic size, and it really reaches length-3 reassignment cycles
    assert k3["instances"] == count_directed_exhaustive(**K3_LAYER) == 20 * 64
    assert k3["feac"] > 0 and k3["k_T_connected"] <= k3["feac"]
    assert k3["cycle_shifts"]["reference"] > 0 and k3["cycles_len3"] > 0, k3
    assert not k3["failures"], (
        f"{len(k3['failures'])} failures, first: {json.dumps(k3['failures'][:3], indent=1)}"
    )
