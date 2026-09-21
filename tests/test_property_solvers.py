"""Hypothesis property tests across every registered solver (docs/verification.md).

For each strategy of :mod:`strategies` and each backend of
:func:`glsolver.testing.registry.available_solvers` that applies to it, a drawn
instance must be solved with ``status == "ok"`` and accepted by the
independent verifier; unweighted parts have exact sizes ``c_i + 1`` and
weighted parts respect ``Σ w ≤ c_t + w_max - 1`` [Thm weighted-k-t-conn]
(asserted here explicitly, on top of the verifier's verdict).  Every theorem
solver runs with ``debug=True`` -- the paper's invariants A1–A8 (§7.2) for
the reference solvers, ``debug_asserts`` for the C++ core -- and every call
is bounded by ``$GL_SOLVER_TIME_LIMIT`` seconds
(:mod:`glsolver.testing.timebound`).

On a failure the offending instance is stored with
:func:`glsolver.testing.regression.save_regression` (``regression/hyp_<strategy>_<solver>.json``),
shrunk with :func:`glsolver.testing.minimize.minimize_instance` under the
predicate "the precondition still holds and the solver still fails" (stored as
``..._min.json``), and the assertion is re-raised.  Two outcomes are handled
differently: a *hang* (the time bound fired) is stored as ``..._hang.json``
and raised as :class:`HarnessHang` -- a ``BaseException``, so Hypothesis
does not shrink it (every attempt would cost the full limit again) -- without
minimization; and an oracle answering ``"timeout"`` within its own limit is
inconclusive rather than wrong, so the example is rejected
(``hypothesis.reject``) and no regression file is written.

Further properties: the reference solvers' certificates are in-arborescences
of original arcs; on arbitrary digraphs a solver answers only ``ok`` (valid)
or ``precondition_failed`` and never misses a FEAC-satisfying instance;
weighted variants of FEAC-only digraphs are solved iff FESAC [Def 5.3] holds;
and, when the C++ core is built, core and reference agree (both valid).
"""
from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from hypothesis import currently_in_test_context, event, given, reject
from hypothesis import strategies as st

from glsolver.api import GLResult, core_available
from glsolver.instance import Instance
from glsolver.preconditions import check_preconditions
from glsolver.testing.minimize import minimize_instance
from glsolver.testing.registry import available_solvers, solvers_for
from glsolver.testing.regression import save_regression
from glsolver.testing.timebound import SolverTimeout, default_time_limit, time_bound
from glsolver.verify import verify_instance_parts
from strategies import (
    st_dag,
    st_directed_kT,
    st_feac_only,
    st_random_digraph,
    st_undirected_k_connected,
    st_weighted,
)

SOLVERS = available_solvers(include_oracles=True)
SOLVER_NAMES = sorted(SOLVERS)
ORACLES = ("bruteforce", "ilp")
DAG_SOLVERS = ("reference-dag", "dag")
UNWEIGHTED_ONLY = ("reference", "general")
#: the oracles' own limit (they answer ``status == "timeout"`` themselves)
ORACLE_TIME_LIMIT = 60.0
#: wall-clock bound of one solver call in seconds (``0`` disables it), from ``$GL_SOLVER_TIME_LIMIT``
TIME_LIMIT = default_time_limit()

#: strategy name -> (strategy, weighted?, dag?)
STRATEGIES: dict[str, tuple[st.SearchStrategy[Instance], bool, bool]] = {
    "undirected": (st_undirected_k_connected(), False, False),
    "directed_kT": (st_directed_kT(), False, False),
    "feac_only": (st_feac_only(), False, False),
    "dag": (st_dag(), True, True),
    "weighted_undirected": (st_weighted(st_undirected_k_connected(max_n=9, max_k=3)), True, False),
    "weighted_directed_kT": (st_weighted(st_directed_kT(max_n=8)), True, False),
    "weighted_dag": (st_weighted(st_dag(max_n=12)), True, True),
}
# ``dag`` is unweighted (weights None) but every solver applies; mark it "unweighted".
STRATEGIES["dag"] = (STRATEGIES["dag"][0], False, True)


def applicable(solver: str, weighted: bool, dag: bool) -> bool:
    if solver in UNWEIGHTED_ONLY and weighted:
        return False
    if solver in DAG_SOLVERS and not dag:
        return False
    return True


def _params() -> list[Any]:
    out = []
    for strat_name, (_s, weighted, dag) in STRATEGIES.items():
        for solver in SOLVER_NAMES:
            marks = []
            if not applicable(solver, weighted, dag):
                marks.append(pytest.mark.skip(reason=f"{solver} does not apply to {strat_name} instances"))
            out.append(pytest.param(strat_name, solver, id=f"{strat_name}-{solver}", marks=marks))
    return out


def solver_kwargs(solver: str) -> dict[str, Any]:
    """Per-backend options: ``debug=True`` for every theorem solver (reference
    solvers: invariants A1–A8; C++ core: ``debug_asserts`` via
    ``api._core_options``); for the oracles their own time limit and no
    (slow, redundant) NetworkX precondition cross-check."""
    if solver in ORACLES:
        return {"verify_preconditions": False, "time_limit": ORACLE_TIME_LIMIT}
    return {"debug": True}


def time_limit_for(solver: str) -> float | None:
    """Bound of one call; the self-limiting oracles get their own limit on top."""
    if TIME_LIMIT <= 0:
        return None
    return TIME_LIMIT + (ORACLE_TIME_LIMIT if solver in ORACLES else 0.0)


def solve(inst: Instance, solver: str, **extra: Any) -> GLResult:
    """``SOLVERS[solver]`` with :func:`solver_kwargs` (plus ``extra``) under
    the time bound; a hang raises :class:`SolverTimeout` (a ``BaseException``)."""
    kw = {**solver_kwargs(solver), **extra}
    with time_bound(time_limit_for(solver), f"{solver} on {inst.name}"):
        return SOLVERS[solver](inst, **kw)


class HarnessHang(BaseException):
    """A solver call hit the time bound.  ``BaseException`` on purpose:
    Hypothesis re-raises it without shrinking (each shrink attempt of a
    hanging instance would cost the full limit again), ``glpartition`` does
    not swallow it, and pytest reports it as a failed test."""


def oracle_timed_out(solver: str, res: GLResult | None) -> bool:
    """An oracle that stopped at its own time limit is inconclusive, not wrong."""
    return solver in ORACLES and res is not None and res.status == "timeout"


# ---------------------------------------------------------------------------
# checking + failure bookkeeping
# ---------------------------------------------------------------------------


def size_errors(inst: Instance, parts: list[list[int]]) -> list[str]:
    """Explicit size/weight assertions (independent of the verifier's wording)."""
    errors: list[str] = []
    if len(parts) != inst.k:
        return [f"{len(parts)} parts for k={inst.k}"]
    tset = set(inst.terminals)
    if inst.weights is None:
        for i, part in enumerate(parts):
            if len(part) != inst.capacities[i] + 1:
                errors.append(f"part {i}: size {len(part)} != c_i+1 = {inst.capacities[i] + 1}")
    else:
        bound_slack = inst.w_max - 1
        for i, part in enumerate(parts):
            wt = sum(inst.weights[v] for v in part if v not in tset)
            if wt > inst.capacities[i] + bound_slack:
                errors.append(f"part {i}: weight {wt} > c_i + w_max - 1 = {inst.capacities[i] + bound_slack}")
    return errors


def run_solver(inst: Instance, solver: str) -> tuple[bool, str, list[str], GLResult | None]:
    """``(ok, message, errors, result)``; ``ok`` iff status ok, verifier valid,
    independent re-verification valid and explicit size/weight checks pass.
    A crash is a failure; a hang propagates as :class:`SolverTimeout`."""
    try:
        res = solve(inst, solver)
    except Exception as exc:  # crash
        return False, f"{type(exc).__name__}: {exc}", [], None
    if res.status != "ok":
        return False, f"status={res.status}: {res.message}", [], res
    errors = list(res.verification.errors) if res.verification is not None else []
    if res.valid is not True:
        return False, "verifier rejected the partition", errors, res
    rep = verify_instance_parts(inst, res.parts)
    if not rep.valid:
        return False, "independent re-verification rejected the partition", rep.errors, res
    errs = size_errors(inst, res.parts)
    if errs:
        return False, "size/weight bound violated", errs, res
    return True, "", [], res


def run_or_hang(inst: Instance, solver: str, tag: str) -> tuple[bool, str, list[str], GLResult | None]:
    """:func:`run_solver`; a hang stores the instance (``hyp_<tag>_<solver>_hang``,
    not minimized) and raises :class:`HarnessHang`."""
    try:
        return run_solver(inst, solver)
    except SolverTimeout as exc:
        path = save_regression(
            inst, reason=f"property test {tag}: {solver} hung: {exc}", name=f"hyp_{tag}_{solver}_hang",
            extra={"solver": solver, "message": str(exc), "errors": [], "expect": "ok", "kind": "timeout"},
        )
        raise HarnessHang(
            f"{solver} hung on {inst.name} (n={inst.n}, m={inst.m}, k={inst.k}): {exc}; saved {path} "
            f"(not minimized: each shrink attempt would cost the full limit); instance={json.dumps(_brief(inst))}"
        ) from None


def precondition_holds(inst: Instance, solver: str) -> bool:
    """Whether the theorem's guarantee applies to ``inst`` for ``solver``
    (FEAC/FESAC by definition; DAG solvers need the out-degree criterion)."""
    if solver not in solvers_for(inst, include_oracles=True):
        return False
    return bool(check_preconditions(inst, True)["feac"])


def still_fails_for(solver: str) -> Callable[[Instance], bool]:
    """Minimization predicate: the precondition still holds and ``solver``
    still fails (an oracle stopping at its own time limit does not count)."""
    def still_fails(candidate: Instance) -> bool:
        if not precondition_holds(candidate, solver):
            return False
        ok, _message, _errors, res = run_solver(candidate, solver)
        return not ok and not oracle_timed_out(solver, res)
    return still_fails


def record_failure(inst: Instance, solver: str, tag: str, message: str, errors: list[str],
                   still_fails: Callable[[Instance], bool], expect: str = "ok") -> list[str]:
    """Save the failing instance, try to minimize it, save the minimized one; return the paths."""
    paths = [str(save_regression(
        inst, reason=f"property test {tag} failed for {solver}: {message}",
        name=f"hyp_{tag}_{solver}",
        extra={"solver": solver, "message": message, "errors": errors[:20], "expect": expect},
    ))]
    try:
        small = minimize_instance(inst, still_fails, max_rounds=8)
    except SolverTimeout as exc:  # a shrink candidate hung: keep the unminimized instance
        paths.append(f"(minimization stopped: {exc})")
        return paths
    except Exception as exc:  # never let the minimizer mask the real failure
        paths.append(f"(minimization failed: {type(exc).__name__}: {exc})")
        return paths
    if small.n < inst.n or small.m < inst.m:
        paths.append(str(save_regression(
            small, reason=f"minimized: property test {tag} failed for {solver}: {message}",
            name=f"hyp_{tag}_{solver}_min",
            extra={"solver": solver, "message": message, "errors": errors[:20], "expect": expect,
                   "minimized_from": f"n={inst.n} m={inst.m}"},
        )))
    return paths


def assert_solved(inst: Instance, solver: str, tag: str) -> GLResult:
    ok, message, errors, res = run_or_hang(inst, solver, tag)
    if ok:
        assert res is not None
        return res
    if oracle_timed_out(solver, res):
        if currently_in_test_context():  # visible in --hypothesis-show-statistics
            event(f"{solver}: own time limit of {ORACLE_TIME_LIMIT:g} s reached; inconclusive example rejected")
        reject()
    paths = record_failure(inst, solver, tag, message, errors, still_fails_for(solver))
    raise AssertionError(
        f"{solver} failed on {inst.name} (n={inst.n}, m={inst.m}, k={inst.k}): {message}; "
        f"errors={errors[:5]}; saved {paths}; instance={json.dumps(_brief(inst))}"
    )


def _brief(inst: Instance) -> dict[str, Any]:
    return {
        "n": inst.n, "directed": inst.directed, "terminals": list(inst.terminals),
        "capacities": list(inst.capacities), "weights": inst.weights,
        "edges": [list(e) for e in (inst.arcs if inst.directed else (inst.undirected_edges or ()))],
    }


# ---------------------------------------------------------------------------
# the main property: every applicable solver solves every drawn instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_name,solver", _params())
@given(data=st.data())
def test_solver_solves_drawn_instances(strategy_name: str, solver: str, data: st.DataObject) -> None:
    strategy, _weighted, _dag = STRATEGIES[strategy_name]
    inst = data.draw(strategy, label=strategy_name)
    if solver in DAG_SOLVERS:
        assert solver in solvers_for(inst, include_oracles=True), "st_dag must yield k-T-connected DAGs"
    res = assert_solved(inst, solver, strategy_name)
    assert res.algorithm == solver
    assert sorted(v for part in res.parts for v in part) == list(range(inst.n))
    for i, part in enumerate(res.parts):
        assert inst.terminals[i] in part


# ---------------------------------------------------------------------------
# certificates: in-arborescences of original arcs
# ---------------------------------------------------------------------------


def assert_in_arborescence(inst: Instance, parts: list[list[int]], parents: dict[Any, Any]) -> None:
    """``parents`` maps every non-terminal to an original out-neighbour in the
    same part and every parent chain ends at the part's terminal (no cycles)."""
    original = set(inst.arcs)
    tset = set(inst.terminals)
    part_of: dict[int, int] = {v: i for i, part in enumerate(parts) for v in part}
    par = {int(v): int(p) for v, p in parents.items()}
    assert set(par) == set(range(inst.n)) - tset, "certificate must cover exactly the non-terminals"
    for v in range(inst.n):
        if v in tset:
            continue
        x, seen = v, {v}
        while x not in tset:
            p = par[x]
            assert (x, p) in original, f"certificate arc ({x},{p}) is not an original arc"
            assert part_of[p] == part_of[x], f"certificate arc ({x},{p}) leaves the part"
            assert p not in seen, f"parent chain of {v} cycles at {p}"
            seen.add(p)
            x = p
        assert x == inst.terminals[part_of[v]], f"chain of {v} ends at a foreign terminal"


@given(inst=st.one_of(st_undirected_k_connected(), st_directed_kT(), st_feac_only()))
def test_reference_certificate_is_in_arborescence(inst: Instance) -> None:
    res = solve(inst, "reference")
    assert res.status == "ok" and res.valid is True, res.message
    assert "parents" in res.certificate
    assert_in_arborescence(inst, res.parts, res.certificate["parents"])


@given(inst=st.one_of(st_dag(), st_weighted(st_dag(max_n=12))))
def test_reference_dag_certificate_is_in_arborescence(inst: Instance) -> None:
    res = solve(inst, "reference-dag")
    assert res.status == "ok" and res.valid is True, res.message
    assert_in_arborescence(inst, res.parts, res.certificate["parents"])


@given(inst=st.one_of(st_undirected_k_connected(max_n=9, max_k=3), st_feac_only()))
def test_reference_weighted_on_unweighted_certificate(inst: Instance) -> None:
    """Unit weights with tight capacities: the weighted algorithm is exact and
    never rounds (paper_notes §13.6); its certificate is an in-arborescence."""
    res = solve(inst, "reference-weighted")
    assert res.status == "ok" and res.valid is True, res.message
    assert res.stats.get("roundings", 0) == 0
    assert size_errors(inst, res.parts) == []
    assert_in_arborescence(inst, res.parts, res.certificate["parents"])


# ---------------------------------------------------------------------------
# honesty on arbitrary digraphs (no precondition)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", [s for s in SOLVER_NAMES if s not in ORACLES and s not in DAG_SOLVERS])
@given(inst=st_random_digraph())
def test_solver_is_honest_on_random_digraphs(solver: str, inst: Instance) -> None:
    """Only ``ok`` (then valid) or ``precondition_failed``; never ``infeasible``,
    never a crash; and FEAC-satisfying instances are always solved."""
    res = solve(inst, solver)
    assert res.status in ("ok", "precondition_failed"), (res.status, res.message)
    feac = check_preconditions(inst, True)["feac"]
    if res.status == "ok":
        assert res.valid is True and verify_instance_parts(inst, res.parts).valid
        assert size_errors(inst, res.parts) == []
    else:
        assert not feac, f"{solver} reported precondition_failed although FEAC holds: {res.message}"
    if feac:
        assert res.status == "ok", f"{solver} missed a FEAC-satisfying instance: {res.message}"


@pytest.mark.parametrize("solver", [s for s in SOLVER_NAMES if s in ("reference-weighted", "weighted")])
@given(inst=st_weighted(st_feac_only(max_n=8)))
def test_weighted_feac_only_solved_iff_fesac(solver: str, inst: Instance) -> None:
    """Weighted variants of FEAC-only digraphs: solved (valid, within the
    bound) exactly when FESAC [Def 5.3] holds by definition."""
    fesac = check_preconditions(inst, True)["feac"]
    ok, message, errors, res = run_or_hang(inst, solver, "weighted_feac_only")
    if fesac:
        if not ok:
            paths = record_failure(inst, solver, "weighted_feac_only", message, errors, still_fails_for(solver))
            raise AssertionError(f"{solver} failed although FESAC holds: {message}; {errors[:5]}; saved {paths}")
    else:
        assert res is not None and res.status == "precondition_failed", (
            f"{solver} must report precondition_failed when FESAC fails, got {message}"
        )


# ---------------------------------------------------------------------------
# core vs reference agreement (needs the C++ core)
# ---------------------------------------------------------------------------

_PAIRS = (("general", "reference"), ("weighted", "reference-weighted"), ("dag", "reference-dag"))


@pytest.mark.skipif(not core_available(), reason="C++ core not built")
@pytest.mark.parametrize("core_name,ref_name", _PAIRS, ids=[p[0] for p in _PAIRS])
@given(data=st.data())
def test_core_and_reference_agree(core_name: str, ref_name: str, data: st.DataObject) -> None:
    if core_name == "dag":
        inst = data.draw(st.one_of(st_dag(), st_weighted(st_dag(max_n=12))), label="dag")
    elif core_name == "weighted":
        inst = data.draw(st.one_of(
            st_weighted(st_undirected_k_connected(max_n=9, max_k=3)),
            st_weighted(st_directed_kT(max_n=8)),
            st_undirected_k_connected(),
        ), label="weighted")
    else:
        inst = data.draw(st.one_of(st_undirected_k_connected(), st_directed_kT(), st_feac_only()), label="general")
    core_res = assert_solved(inst, core_name, f"core_{core_name}")
    ref_res = assert_solved(inst, ref_name, f"ref_{ref_name}")
    assert sorted(len(p) for p in core_res.parts) == sorted(len(p) for p in ref_res.parts) or inst.is_weighted
    if "parents" in core_res.certificate:
        assert_in_arborescence(inst, core_res.parts, core_res.certificate["parents"])
