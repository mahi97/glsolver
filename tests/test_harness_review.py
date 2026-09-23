"""Review of the correctness harness itself (docs/verification.md):

1. **Mutation testing** -- bugs are injected into the *reference* solver via
   ``monkeypatch`` (the real modules are never edited) and the harness's own
   test functions (``tests/test_exhaustive_small.py``,
   ``tests/test_property_solvers.py``) are executed against the mutant.  Each
   parametrized pair ``(mutant, harness test)`` asserts that the harness test
   FAILS, i.e. the safety net really catches the bug -- including the mutants
   that are *output-valid* on every enumerated instance (only the paper's
   debug invariants expose them, :data:`INVARIANT_ONLY`) and the one that
   makes the reference solver loop forever (only the per-call time bound
   turns it into a verdict, :data:`HANGS`).  The regression store is
   redirected to a temporary directory (``REGRESSION_DIR`` monkeypatched) so the
   permanent ``regression/`` folder is never touched.
2. **Strategy audits** -- the Hypothesis strategies of ``tests/strategies.py``
   are re-checked with *independent* precondition tests (Even's algorithm from
   :mod:`glsolver.generators`, the by-definition NetworkX flows of
   :mod:`glsolver.preconditions`, the reference solver's own flow machinery)
   and their coverage (capacity modes, extreme vectors, ``k == 1``, ``k == max``)
   is asserted on a deterministic sample.
3. **Enumeration counts** -- the exhaustive enumerations are recounted from
   scratch (NetworkX atlas + binomials) and compared with the harness's counts.
4. **Regression side effects** -- a simulated failure must write the instance
   to the (patched) regression directory, both through the property-test path
   and through ``scripts/run_exhaustive.py``.
5. **Time bound and oracle time limits** -- ``glsolver.testing.timebound``
   raises a ``BaseException`` that ``glpartition`` cannot swallow and that
   Hypothesis does not shrink; the exhaustive tests, the property tests and
   ``scripts/run_exhaustive.py`` report a hang as a stored ``timeout``
   failure, and an oracle stopping at its own time limit is inconclusive
   rather than wrong.
6. **Shared enumeration and tally location** -- the script and the harness
   use one digraph block generator, and the tally never lands in the source tree.

All tests are deterministic (``derandomize=True`` for Hypothesis, seeded
``random.Random`` for the mutants).
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import itertools
import json
import math
import random
import signal
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import networkx as nx
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from hypothesis.errors import UnsatisfiedAssumption
from networkx.generators.atlas import graph_atlas_g

import glref.assignment
import glref.graph
import glref.unweighted
import glsolver.testing.regression as regression_mod
import test_edge_cases as tec
import test_exhaustive_small as tes
import test_property_solvers as tps
from glref.essential import all_essential
from glref.graph import DiGraphState
from glsolver.api import GLResult, glpartition
from glsolver.generators import is_k_t_connected, is_k_vertex_connected
from glsolver.instance import Instance, make_instance
from glsolver.preconditions import check_preconditions, is_k_T_connected, terminal_connectivity_nx
from glsolver.testing.registry import solvers_for
from glsolver.testing.regression import instance_hash, load_all, save_regression
from glsolver.testing.timebound import SolverTimeout, time_bound
from glsolver.verify import verify_instance_parts
from strategies import (
    CAPACITY_MODES,
    atlas_connected_graphs,
    count_directed_exhaustive,
    count_undirected_exhaustive,
    directed_blocks,
    directed_cases_for_block,
    iter_directed_exhaustive,
    iter_undirected_exhaustive,
    st_capacities,
    st_dag,
    st_directed_kT,
    st_feac_only,
    st_random_digraph,
    st_undirected_k_connected,
    st_weighted,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_REGRESSION_DIR = regression_mod.REGRESSION_DIR
_REAL_REGRESSION_SNAPSHOT = sorted(p.name for p in REAL_REGRESSION_DIR.glob("*.json"))


def _assert_real_store_untouched() -> None:
    assert sorted(p.name for p in REAL_REGRESSION_DIR.glob("*.json")) == _REAL_REGRESSION_SNAPSHOT, (
        "the review must never write into the permanent regression store"
    )


# ---------------------------------------------------------------------------
# source-level mutants of the reference solver
# ---------------------------------------------------------------------------


def _mutate(func: Callable[..., Any], replacements: list[tuple[str, str]],
            extra_globals: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Re-compile ``func`` from its source with exact, unique textual
    substitutions, in a *copy* of its module globals (the module is untouched)."""
    src = textwrap.dedent(inspect.getsource(func))
    for old, new in replacements:
        assert src.count(old) == 1, f"{func.__qualname__}: snippet not unique/found:\n{old}"
        src = src.replace(old, new)
    ns = dict(func.__globals__)
    ns.update(extra_globals or {})
    exec(compile(src, f"<mutant:{func.__qualname__}>", "exec"), ns)
    return ns[func.__name__]


_SHIFT_BUDGET_HITS: list[str] = []


def _mutant_rng(g: DiGraphState) -> random.Random:
    """Deterministic per graph state (so Hypothesis' final replay reproduces)."""
    return random.Random(f"{g.n}|{g.terminals}|{sorted(g.arcs())}")


def _mutant_shift_budget(g: DiGraphState, stats: Any) -> None:
    if stats.cycle_shifts > 100 * (g.num_nonterminals() + 1) * max(1, g.k):
        _SHIFT_BUDGET_HITS.append(repr(g))
        raise RuntimeError("mutant: cycle-shift budget exhausted (the loop would not terminate)")


def apply_skip_capacity_check(mp: pytest.MonkeyPatch) -> None:
    """(1) ``find_witness``/``is_witness`` ignore the terminal capacities."""
    fw = _mutate(glref.assignment.find_witness, [
        ("    if sum(capacities[t] for t in terms) != len(nonterms):\n        return None\n",
         "    pass  # mutant: capacity sum not checked\n"),
        ("        net.add_arc(tid[t], z, capacities[t])\n",
         "        net.add_arc(tid[t], z, len(nonterms))  # mutant: capacities ignored\n"),
    ])
    iw = _mutate(glref.assignment.is_witness, [
        ("        if counts[t] != capacities[t]:\n", "        if False:  # mutant: capacity check skipped\n"),
    ])
    for mod in (glref.assignment, glref.unweighted):
        mp.setattr(mod, "find_witness", fw)
        mp.setattr(mod, "is_witness", iw)


def apply_random_cycle_shift(mp: pytest.MonkeyPatch) -> None:
    """(2) the cycle shift assigns every shifted vertex to a random terminal."""
    sa = _mutate(glref.unweighted.shift_assignment, [
        ("    stats.shift_calls += 1\n", "    stats.shift_calls += 1\n    _rng = _mutant_rng(g)\n"),
        ("        for v, old, new in changes:\n            assert phi[v] == old\n            phi[v] = new\n"
         "        stats.cycle_shifts += 1\n",
         "        for v, old, new in changes:\n            assert phi[v] == old\n"
         "            phi[v] = _rng.choice(list(g.terminals))  # mutant\n"
         "        stats.cycle_shifts += 1\n        _mutant_shift_budget(g, stats)\n"),
    ], {"_mutant_rng": _mutant_rng, "_mutant_shift_budget": _mutant_shift_budget})
    mp.setattr(glref.unweighted, "shift_assignment", sa)


def apply_contract_out_degree_two(mp: pytest.MonkeyPatch) -> None:
    """(3) step (ii) also contracts pre-terminals of out-degree 2 into their
    terminal neighbour, without first deleting the other arc ([Lem 7.4] needs
    ``d^+(p) == 1``; the paper's contraction counterexample)."""
    gp = _mutate(glref.unweighted.gl_partition, [
        ("        p_deg1 = next((p for p in g.pre_terminals() if g.out_degree(p) == 1), None)\n",
         "        p_deg1 = next((p for p in g.pre_terminals() if g.out_degree(p) <= 2), None)  # mutant\n"),
        ("            t = g.out_neighbors(p)[0]\n", "            t = g.terminal_out_neighbors(p)[0]  # mutant\n"),
    ])
    mp.setattr(glref.unweighted, "gl_partition", gp)


def apply_swap_two_vertices(mp: pytest.MonkeyPatch) -> None:
    """(4) the returned parts have one non-terminal of two different parts swapped
    (sizes stay right; the certificate is left stale)."""
    orig = glref.unweighted.gl_partition

    def swapped(inst: Instance, **kw: Any) -> Any:
        r = orig(inst, **kw)
        if r.status == "ok":
            tset = set(inst.terminals)
            first: dict[int, int] = {}
            for i, part in enumerate(r.parts):
                for v in part:
                    if v not in tset:
                        first.setdefault(i, v)
                        break
            if len(first) >= 2:
                (i, a), (j, b) = list(first.items())[:2]
                r.parts[i] = sorted(b if x == a else x for x in r.parts[i])
                r.parts[j] = sorted(a if x == b else x for x in r.parts[j])
        return r

    mp.setattr(glref.unweighted, "gl_partition", swapped)


def apply_stale_ess_after_terminal_removal(mp: pytest.MonkeyPatch) -> None:
    """(5) step (i) does not recompute ``Ess`` after removing a zero-capacity
    terminal (paper_notes §13.3 says it must)."""
    gp = _mutate(glref.unweighted.gl_partition, [
        ("            kappa, ess = _all_essential(g, stats)  # §13.3: new essential terminals may appear\n",
         "            pass  # mutant: Ess not recomputed after the terminal removal\n"),
    ])
    mp.setattr(glref.unweighted, "gl_partition", gp)


def apply_contract_drops_in_arcs(mp: pytest.MonkeyPatch) -> None:
    """(6) ``DiGraphState.contract`` deletes the incoming arcs of ``p`` instead
    of redirecting them to ``t`` [Def 2.1] (shared by every reference solver)."""
    ct = _mutate(glref.graph.DiGraphState.contract, [
        ("        if t not in self.out[u]:\n            self.out[u][t] = None\n"
         "            self.inn[t].add(u)\n            self.orig_head[(u, t)] = p\n",
         "        pass  # mutant: incoming arcs are dropped instead of redirected\n"),
    ])
    mp.setattr(glref.graph.DiGraphState, "contract", ct)


MUTANTS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "skip_capacity_check": apply_skip_capacity_check,
    "random_cycle_shift": apply_random_cycle_shift,
    "contract_out_degree_two": apply_contract_out_degree_two,
    "swap_two_vertices": apply_swap_two_vertices,
    "stale_ess_after_terminal_removal": apply_stale_ess_after_terminal_removal,
    "contract_drops_in_arcs": apply_contract_drops_in_arcs,
}


@pytest.fixture
def patched_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``save_regression`` (used by every harness test) to ``tmp_path``."""
    store = tmp_path / "regression"
    monkeypatch.setattr(regression_mod, "REGRESSION_DIR", store)
    yield store
    _assert_real_store_untouched()


def _small_corpus() -> list[Instance]:
    """90 undirected instances (atlas, n <= 4) + the FEAC digraphs on n <= 3."""
    corpus = list(iter_undirected_exhaustive(4, full=True))
    corpus += [i for i in iter_directed_exhaustive(3) if check_preconditions(i, True)["feac"]]
    return corpus


def _outcome(inst: Instance) -> str:
    """Raw outcome of the (possibly mutated) reference solver on ``inst``."""
    try:
        r = glref.unweighted.gl_partition(inst)
    except Exception as exc:  # noqa: BLE001 - a crash is one possible outcome
        return f"crash:{type(exc).__name__}"
    if r.status != "ok":
        return f"status:{r.status}"
    return "valid" if verify_instance_parts(inst, r.parts).valid else "invalid"


def _named_examples() -> dict[str, Instance]:
    from glsolver.generators import (
        paper_contract_counterexample,
        paper_essential_example,
        paper_running_example,
    )

    named = {"contract_cex": paper_contract_counterexample(), "running": paper_running_example(),
             "essential": paper_essential_example()}
    for path, inst, _meta in load_all():
        named[path.stem] = inst
    return named


def test_mutants_are_real_bugs_independently_of_the_harness() -> None:
    """Sanity check of the mutation machinery: the real solver is valid on a
    small corpus (n <= 4 undirected, FEAC digraphs n <= 3) and on the paper
    examples / regression seeds, while every mutant shows up somewhere -- as a
    rejected partition, a crash or a wrong status (``debug=False``), or as a
    violated paper invariant (``debug=True``).  The tallies are printed:
    measured, four of the six mutants are *output-valid* on the whole small
    corpus and only the debug invariants expose them (see the review report)."""
    corpus = _small_corpus()
    named = _named_examples()
    assert len(corpus) > 100 and len(named) >= 6
    assert all(_outcome(inst) == "valid" for inst in corpus)
    for inst in named.values():
        res = glpartition(inst, algorithm="reference", debug=True)
        assert res.status == "ok" and res.valid is True
    for name, apply in MUTANTS.items():
        with pytest.MonkeyPatch.context() as mp:
            apply(mp)
            tally = Counter(_outcome(inst) for inst in corpus)
            invariants: Counter[str] = Counter()
            for inst in named.values():
                try:
                    res = glpartition(inst, algorithm="reference", debug=True)
                except Exception as exc:  # noqa: BLE001
                    invariants[f"{type(exc).__name__}: {str(exc)[:40]}"] += 1
                    continue
                invariants["valid" if res.status == "ok" and res.valid else f"status:{res.status}"] += 1
        print(f"[mutant] {name:34s} corpus(debug=False)={dict(tally)} named(debug=True)={dict(invariants)}")
        visible = sum(v for k, v in tally.items() if k != "valid") + sum(
            v for k, v in invariants.items() if k != "valid")
        assert visible > 0, (name, tally, invariants)


# ---------------------------------------------------------------------------
# running the harness's own test functions against a mutant
# ---------------------------------------------------------------------------


def _fresh_property(test_fn: Callable[..., Any], max_examples: int = 100) -> Callable[..., Any]:
    """A new Hypothesis wrapper around the harness test's *inner* function with
    review settings (deterministic, no database, no shrinking).  The original
    test object is not modified (``@settings`` would mutate it in place)."""
    handle = test_fn.hypothesis
    given_kwargs = getattr(handle, "_given_kwargs", None) or {"data": st.data()}
    review = settings(
        max_examples=max_examples, database=None, derandomize=True, deadline=None,
        suppress_health_check=list(HealthCheck), phases=(Phase.generate,), print_blob=False,
    )
    return review(given(**given_kwargs)(handle.inner_test))


class _Hang(BaseException):
    """Raised by the review's alarm.  Derives from ``BaseException`` on purpose:
    ``glpartition`` turns every ``Exception`` into ``status == "error"``, so an
    ``Exception``-based timeout would be swallowed and reported as a solver
    error instead of a hang."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _Hang()


@contextlib.contextmanager
def _time_bound(seconds: float) -> Iterator[None]:
    """Abort the block with :class:`_Hang` after ``seconds`` (main thread only)."""
    previous = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


HARNESS_TIME_BOUND = 150.0  # outer guard for the review itself; the harness has its own per-call bound

#: ``(caught, detail, failures)`` -- ``failures`` are the exhaustive tally's failure records
Verdict = tuple[bool, str, list[dict[str, Any]]]


def _tally_failures(tally: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for section in tally.values() if isinstance(section, dict) for f in section.get("failures", [])]


def _exhaustive(test_fn: Callable[[dict[str, Any]], None]) -> Verdict:
    tally: dict[str, Any] = {}
    try:
        with _time_bound(HARNESS_TIME_BOUND):
            test_fn(tally)
    except AssertionError as exc:
        fails = _tally_failures(tally)
        per_solver = Counter(f["solver"] for f in fails)
        per_kind = Counter(f["kind"] for f in fails)
        return True, f"{len(fails)} failures {dict(per_solver)} {dict(per_kind)}; {str(exc)[:120]}", fails
    except _Hang:
        return False, f"HANG: no verdict within {HARNESS_TIME_BOUND:.0f} s", []
    return False, "passed", []


def harness_exhaustive_undirected() -> Verdict:
    return _exhaustive(tes.test_exhaustive_undirected_small)


def harness_exhaustive_directed() -> Verdict:
    return _exhaustive(tes.test_exhaustive_directed_tiny)


def _property(test_fn: Callable[..., Any], **kw: Any) -> Verdict:
    try:
        with _time_bound(HARNESS_TIME_BOUND):
            _fresh_property(test_fn)(**kw)
    except tps.HarnessHang as exc:  # the harness reported a hang: a verdict, not a hang of the suite
        return True, f"HarnessHang: {str(exc)[:160]}", []
    except Exception as exc:  # noqa: BLE001 - Hypothesis re-raises the test's own error
        return True, f"{type(exc).__name__}: {str(exc)[:160]}", []
    except _Hang:
        return False, f"HANG: no verdict within {HARNESS_TIME_BOUND:.0f} s", []
    return False, "passed", []


def harness_property_undirected_reference() -> Verdict:
    return _property(tps.test_solver_solves_drawn_instances, strategy_name="undirected", solver="reference")


def harness_property_feac_only_reference() -> Verdict:
    return _property(tps.test_solver_solves_drawn_instances, strategy_name="feac_only", solver="reference")


def harness_property_certificate() -> Verdict:
    return _property(tps.test_reference_certificate_is_in_arborescence)


def harness_honesty_reference() -> Verdict:
    return _property(tps.test_solver_is_honest_on_random_digraphs, solver="reference")


HARNESS: dict[str, Callable[[], Verdict]] = {
    "exhaustive_undirected": harness_exhaustive_undirected,
    "exhaustive_directed": harness_exhaustive_directed,
    "property_undirected_reference": harness_property_undirected_reference,
    "property_feac_only_reference": harness_property_feac_only_reference,
    "property_certificate": harness_property_certificate,
    "honesty_reference": harness_honesty_reference,
}

#: Every (mutant, harness test) pair must be caught -- there are no known misses
#: any more.  The first review measured six misses of the exhaustive tests, which
#: then ran the reference solvers with ``debug=False`` and had no time bound
#: (docs/verification.md §4.3): ``stale_ess_after_terminal_removal`` (both
#: enumerations), ``contract_out_degree_two`` (undirected) and
#: ``random_cycle_shift`` (directed: with ``k <= 2`` the only wrong shift is a
#: no-op) are *output-valid* on every enumerated instance and only a debug
#: invariant exposes them; ``contract_drops_in_arcs`` makes ``shift_assignment``
#: loop forever, so neither exhaustive test ever produced a verdict.  With
#: ``debug=True`` every pair is caught -- the A7 invariant even fires on the
#: hanging instances *before* they hang -- and :data:`INVARIANT_ONLY` asserts
#: that the silent mutants are exposed by an invariant.  :data:`HANGS` lists the
#: pairs that only the per-call time bound turns into a verdict once the
#: invariants are off (``test_exhaustive_harness_bounds_a_real_hang``).
KNOWN_MISSES: set[tuple[str, str]] = set()
INVARIANT_ONLY: set[tuple[str, str]] = {
    ("stale_ess_after_terminal_removal", "exhaustive_undirected"),
    ("stale_ess_after_terminal_removal", "exhaustive_directed"),
    ("contract_out_degree_two", "exhaustive_undirected"),
    ("random_cycle_shift", "exhaustive_directed"),
}
HANGS: set[tuple[str, str]] = {
    ("contract_drops_in_arcs", "exhaustive_undirected"),
    ("contract_drops_in_arcs", "exhaustive_directed"),
}
MUST_CATCH: list[tuple[str, str]] = [(m, h) for m in MUTANTS for h in HARNESS if (m, h) not in KNOWN_MISSES]

REVIEW_TIME_LIMIT = 3.0  # per solver call while a hanging mutant is injected (the harness default is 20 s)


@pytest.fixture
def review_time_limit(monkeypatch: pytest.MonkeyPatch) -> float:
    """Shorten the harness's per-call time bound so a hanging mutant costs seconds."""
    monkeypatch.setattr(tes, "TIME_LIMIT", REVIEW_TIME_LIMIT)
    monkeypatch.setattr(tps, "TIME_LIMIT", REVIEW_TIME_LIMIT)
    return REVIEW_TIME_LIMIT


def _kwargs_without_debug(original: Callable[[str], dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    """``solver_kwargs`` with the ``debug`` option removed (the harness as it
    was before the first review)."""
    def solver_kwargs(name: str) -> dict[str, Any]:
        return {k: v for k, v in original(name).items() if k != "debug"}
    return solver_kwargs


@pytest.fixture
def debug_invariants_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the theorem solvers with ``debug=False``: the paper invariants then
    cannot pre-empt a hang, so ``contract_drops_in_arcs`` really loops forever."""
    monkeypatch.setattr(tes, "solver_kwargs", _kwargs_without_debug(tes.solver_kwargs))
    monkeypatch.setattr(tps, "solver_kwargs", _kwargs_without_debug(tps.solver_kwargs))


@pytest.mark.parametrize("mutant,harness", MUST_CATCH, ids=[f"{m}--{h}" for m, h in MUST_CATCH])
def test_harness_catches_mutant(mutant: str, harness: str, patched_store: Path, review_time_limit: float,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness test ``harness`` must FAIL when ``mutant`` is injected."""
    _SHIFT_BUDGET_HITS.clear()
    MUTANTS[mutant](monkeypatch)
    caught, detail, failures = HARNESS[harness]()
    print(f"[mutation] {mutant:34s} {harness:30s} caught={caught} {detail}")
    assert caught, f"{harness} did NOT catch mutant {mutant}: {detail}"
    if harness.startswith("exhaustive"):
        # failures were stored (in the patched directory), with the expected status recorded
        files = list(patched_store.glob("*.json"))
        assert files, "exhaustive failures must be written to the regression store"
        meta = json.loads(files[0].read_text())["regression"]
        assert meta["expect"] in ("ok", "precondition_failed") and meta["reason"]
        assert meta["kind"] in ("invalid", "crash", "timeout")
    if (mutant, harness) in INVARIANT_ONLY:
        kinds = Counter(f["kind"] for f in failures)
        assert any(f["kind"] == "crash" and "InvariantError" in f["message"] for f in failures), (
            f"{mutant} is output-valid on these instances; only a debug invariant exposes it: {dict(kinds)}")
    if (mutant, harness) in HANGS:
        # with the invariants on, A7 fires before any instance hangs; the time-bound
        # path is exercised with the invariants off in test_exhaustive_harness_bounds_a_real_hang
        assert failures and all(f["kind"] in ("crash", "invalid", "timeout") for f in failures), detail
    if mutant == "random_cycle_shift":
        print(f"[mutation] cycle-shift budget hit {len(_SHIFT_BUDGET_HITS)} times (would loop forever)")


def test_harness_passes_on_real_code_smoke(patched_store: Path) -> None:
    """Control: the same harness entry points pass without a mutant (fast ones only)."""
    for name in ("property_undirected_reference", "property_feac_only_reference", "property_certificate",
                 "honesty_reference"):
        caught, detail, _failures = HARNESS[name]()
        assert not caught, (name, detail)
    assert not list(patched_store.glob("*.json"))


def test_exhaustive_directed_passes_on_real_code(patched_store: Path) -> None:
    """Control for the ``INVARIANT_ONLY`` pairs: the directed enumeration with the
    debug invariants on has no false positive on the real code."""
    caught, detail, failures = harness_exhaustive_directed()
    assert not caught and not failures, detail
    assert not list(patched_store.glob("*.json"))


# ---------------------------------------------------------------------------
# time bound: a hang fails the harness instead of hanging the suite
# ---------------------------------------------------------------------------


def _hang_instance() -> Instance:
    """Atlas graph 31 (a tree on 5 vertices), ``k = 1``: under
    ``contract_drops_in_arcs`` the reference solver loops forever in
    ``shift_assignment``."""
    return make_instance(5, [(0, 1), (0, 4), (1, 2), (2, 3)], [1], [4], directed=False, name="atlas31_k1")


def _spin(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def test_time_bound_raises_a_base_exception_and_nests() -> None:
    assert issubclass(SolverTimeout, BaseException) and not issubclass(SolverTimeout, Exception)
    with pytest.raises(SolverTimeout, match="did not return within 0.2 s"):
        with time_bound(0.2, "review spin"):
            _spin(5.0)
    with time_bound(0, "off") as armed:
        assert armed is False
    with time_bound(None, "off") as armed:
        assert armed is False
    # nested: the inner bound fires first and the outer one is re-armed with its remaining time
    t0 = time.monotonic()
    with pytest.raises(_Hang):
        with _time_bound(0.6):
            with pytest.raises(SolverTimeout):
                with time_bound(0.2, "inner"):
                    _spin(5.0)
            _spin(5.0)
    assert time.monotonic() - t0 < 3.0
    # an inner bound that completes leaves the outer one armed
    with pytest.raises(_Hang):
        with _time_bound(0.3):
            with time_bound(5.0, "inner"):
                pass
            _spin(5.0)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)  # nothing left armed


def test_glpartition_does_not_swallow_the_time_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """``glpartition`` turns every ``Exception`` into ``status == "error"`` -- the
    reason :class:`SolverTimeout` is a ``BaseException``."""
    inst = _hang_instance()
    assert glpartition(inst, algorithm="reference").valid is True
    apply_contract_drops_in_arcs(monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(SolverTimeout):
        with time_bound(1.0, "review"):
            glpartition(inst, algorithm="reference")
    assert time.monotonic() - t0 < 5.0

    class Soft(Exception):
        pass

    def soft_alarm(_signum: int, _frame: Any) -> None:
        raise Soft()

    previous = signal.signal(signal.SIGALRM, soft_alarm)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        res = glpartition(inst, algorithm="reference")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert res.status == "error" and "Soft" in res.message  # an Exception-based alarm is swallowed


def test_debug_invariant_pre_empts_the_hang_but_the_bound_catches_it_anyway(
        review_time_limit: float, debug_invariants_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the hanging instance the fixed harness (``debug=True``) gets an A7
    ``InvariantError`` crash; with the invariants off the same call loops
    forever and only the time bound returns a verdict."""
    apply_contract_drops_in_arcs(monkeypatch)
    inst = _hang_instance()
    fn = tes.solvers_for(inst)["reference"]
    res, exc = tes._run(fn, inst, "reference", debug=True)
    assert res is None and type(exc).__name__ == "InvariantError" and "A7" in str(exc)
    t0 = time.monotonic()
    res, exc = tes._run(fn, inst, "reference")  # debug off through the fixture
    assert res is None and isinstance(exc, SolverTimeout)
    assert review_time_limit <= time.monotonic() - t0 < review_time_limit + 5


@pytest.mark.perf
@pytest.mark.parametrize("harness", sorted({h for _m, h in HANGS}))
def test_exhaustive_harness_bounds_a_real_hang(harness: str, patched_store: Path, review_time_limit: float,
                                               debug_invariants_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the invariants off ``contract_drops_in_arcs`` really hangs the
    reference solver inside the enumeration (measured before the fix: no
    verdict after 13 minutes).  The time bound turns that into a stored
    ``timeout`` failure and a verdict within seconds, and the enumeration
    stops there instead of paying the limit for every further hang."""
    apply_contract_drops_in_arcs(monkeypatch)
    t0 = time.monotonic()
    caught, detail, failures = HARNESS[harness]()
    elapsed = time.monotonic() - t0
    assert caught and failures, detail
    last = failures[-1]
    assert last["kind"] == "timeout" and "did not return within" in last["message"], last
    assert sum(1 for f in failures if f["kind"] == "timeout") == 1, "the enumeration must stop at the first hang"
    assert elapsed < 60, elapsed
    stored = [meta for _p, _i, meta in load_all(patched_store)]
    assert any(m["kind"] == "timeout" and m["expect"] in ("ok", "precondition_failed") for m in stored)


def test_property_harness_reports_a_hang_without_shrinking(
        patched_store: Path, review_time_limit: float, debug_invariants_off: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert issubclass(tps.HarnessHang, BaseException) and not issubclass(tps.HarnessHang, Exception)
    apply_contract_drops_in_arcs(monkeypatch)
    inst = _hang_instance()
    with pytest.raises(tps.HarnessHang, match="not minimized"):
        tps.assert_solved(inst, "reference", "review")
    files = sorted(p.name for p in patched_store.glob("*.json"))
    assert files == ["hyp_review_reference_hang.json"], files  # stored once, never minimized
    meta = json.loads((patched_store / files[0]).read_text())["regression"]
    assert meta["kind"] == "timeout" and meta["solver"] == "reference" and meta["expect"] == "ok"


def test_hypothesis_does_not_shrink_a_harness_hang() -> None:
    """Hypothesis re-raises a ``BaseException`` from the first failing example
    instead of shrinking it (each shrink attempt would cost the full limit)."""
    calls: list[int] = []

    @settings(max_examples=50, database=None, derandomize=True, deadline=None,
              suppress_health_check=list(HealthCheck))
    @given(x=st.integers(10, 1000))
    def prop(x: int) -> None:
        calls.append(x)
        raise tps.HarnessHang(f"hang at {x}")

    with pytest.raises(tps.HarnessHang):
        prop()
    assert len(calls) == 1


def test_oracle_time_limit_is_inconclusive_not_a_failure(patched_store: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """An oracle answering ``timeout`` within its own limit rejects the example
    (no regression file); a wrong oracle answer is still a failure."""
    inst = _c4_instance()

    def timed_out(inst: Instance, **_kw: Any) -> GLResult:
        return GLResult(inst, "bruteforce", "timeout", [], [], message="time limit reached")

    monkeypatch.setitem(tps.SOLVERS, "bruteforce", timed_out)
    with pytest.raises(UnsatisfiedAssumption):
        tps.assert_solved(inst, "bruteforce", "review")
    assert not list(patched_store.glob("*.json"))
    assert tps.still_fails_for("bruteforce")(inst) is False

    def infeasible(inst: Instance, **_kw: Any) -> GLResult:
        return GLResult(inst, "bruteforce", "infeasible", [], [], message="no partition")

    monkeypatch.setitem(tps.SOLVERS, "bruteforce", infeasible)
    with pytest.raises(AssertionError, match="status=infeasible"):
        tps.assert_solved(inst, "bruteforce", "review")
    assert (patched_store / "hyp_review_bruteforce.json").exists()


# ---------------------------------------------------------------------------
# strategy audits (independent precondition checks + coverage)
# ---------------------------------------------------------------------------


def _collect(strategy: st.SearchStrategy[Any], check: Callable[[Any], None], n: int = 200,
             min_seen: int | None = None) -> list[Any]:
    """Draw ``n`` deterministic examples of ``strategy``, run ``check`` on each
    and return them; at least ``min_seen`` (default ``n // 2``) must be drawn
    (Hypothesis stops early when a strategy's search space is exhausted)."""
    seen: list[Any] = []

    @settings(max_examples=n, database=None, derandomize=True, deadline=None,
              suppress_health_check=list(HealthCheck), phases=(Phase.generate,))
    @given(inst=strategy)
    def run(inst: Any) -> None:
        check(inst)
        seen.append(inst)

    run()
    need = n // 2 if min_seen is None else min_seen
    assert len(seen) >= need, f"only {len(seen)} examples generated (rejection too high?)"
    return seen


def _undirected_graph(inst: Instance) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(inst.n))
    graph.add_edges_from(inst.undirected_edges or ())
    return graph


def _is_extreme(inst: Instance) -> bool:
    return inst.k >= 2 and sum(1 for c in inst.capacities if c == 0) == inst.k - 1


def test_st_undirected_k_connected_is_k_connected_and_covers_shapes() -> None:
    def check(inst: Instance) -> None:
        assert not inst.directed and inst.weights is None
        assert 2 <= inst.n <= 10 and 1 <= inst.k <= min(4, inst.n - 1)
        assert sum(inst.capacities) == inst.n - inst.k
        graph = _undirected_graph(inst)
        assert is_k_vertex_connected(graph, inst.k), "Even's test (generators) disagrees"
        assert is_k_T_connected(inst) == (True, None), "flow test by definition disagrees"
        assert inst.meta["connectivity"] == nx.node_connectivity(graph)
        assert inst.meta["mode"] in CAPACITY_MODES

    seen = _collect(st_undirected_k_connected(), check, n=250)
    modes = Counter(i.meta["mode"] for i in seen)
    kinds = Counter(i.meta["kind"] for i in seen)
    print("undirected strategy:", len(seen), dict(modes), dict(kinds), Counter(i.k for i in seen))
    assert set(modes) == set(CAPACITY_MODES)
    assert set(kinds) == {"gnp", "regular", "harary", "complete"}
    assert any(i.k == 1 for i in seen) and any(i.k == 4 for i in seen)
    assert any(i.n == 2 for i in seen) and any(i.n == 10 for i in seen)
    assert any(_is_extreme(i) for i in seen), "no extreme capacity vector drawn"
    assert any(i.meta["mode"] == "zeros-allowed" and i.capacities.count(0) >= 2 for i in seen)
    assert any(i.meta["connectivity"] == i.k for i in seen), "no tight (κ == k) instance drawn"
    assert all(i.k < i.n for i in seen)  # k == n is never drawn (documented: k <= n-1)


def test_st_directed_kT_is_k_T_connected_and_covers_shapes() -> None:
    def check(inst: Instance) -> None:
        assert inst.directed and inst.weights is None
        assert is_k_T_connected(inst) == (True, None)
        assert inst.meta["claims_kT_connected"] is True

    seen = _collect(st_directed_kT(), check, n=150)
    modes = Counter(i.meta["mode"] for i in seen)
    print("directed_kT strategy:", len(seen), dict(modes), Counter(i.k for i in seen))
    assert set(modes) == set(CAPACITY_MODES)
    assert any(i.k == 1 for i in seen) and any(i.k == 3 for i in seen)
    assert any(_is_extreme(i) for i in seen)


def test_st_feac_only_is_feac_but_not_kT_connected() -> None:
    def check(inst: Instance) -> None:
        assert inst.directed and inst.k >= 2
        # independent (1): the generators' own flow checker says "not k-T-connected"
        assert is_k_t_connected(inst) is False
        bad = inst.meta["violating_vertex"]
        assert bad is not None and terminal_connectivity_nx(inst, bad) < inst.k
        # independent (2): the reference solver's tightest-cut machinery finds a FEAC witness
        g = DiGraphState.from_instance(inst)
        _kappa, ess = all_essential(g)
        cap = dict(zip(inst.terminals, inst.capacities))
        assert glref.assignment.find_witness(g, ess, cap) is not None

    seen = _collect(st_feac_only(), check, n=120)
    schemes = Counter(i.meta["scheme"] for i in seen)
    print("feac_only strategy:", len(seen), dict(schemes), Counter(i.k for i in seen))
    assert set(schemes) == {"random", "damaged"}
    assert {i.k for i in seen} == {2, 3}
    assert any(_is_extreme(i) for i in seen)


def test_st_dag_is_acyclic_k_T_connected_and_covers_shapes() -> None:
    def check(inst: Instance) -> None:
        assert inst.directed
        digraph = nx.DiGraph()
        digraph.add_nodes_from(range(inst.n))
        digraph.add_edges_from(inst.arcs)
        assert nx.is_directed_acyclic_graph(digraph)
        tset = set(inst.terminals)
        out_deg = Counter(u for u, _v in inst.arcs)
        assert all(out_deg[v] >= inst.k for v in range(inst.n) if v not in tset)
        assert is_k_T_connected(inst) == (True, None)
        assert "reference-dag" in solvers_for(inst)

    seen = _collect(st_dag(), check, n=120)
    fams = Counter(i.meta["family"] for i in seen)
    print("dag strategy:", len(seen), dict(fams), Counter(i.k for i in seen), Counter(i.meta["mode"] for i in seen))
    assert set(fams) == {"random_kT_dag", "layered_dag"}
    assert any(i.k == 1 for i in seen) and any(i.k == 4 for i in seen)
    assert any(_is_extreme(i) for i in seen)
    assert all(i.weights is None for i in seen)


def test_st_weighted_variants_are_well_formed() -> None:
    def check_und(inst: Instance) -> None:
        assert inst.weights is not None and not inst.directed
        tset = set(inst.terminals)
        assert all(inst.weights[v] == 0 for v in tset)
        assert all(1 <= inst.weights[v] <= 5 for v in range(inst.n) if v not in tset)
        assert 0 <= sum(inst.capacities) - inst.total_weight <= 3
        assert is_k_vertex_connected(_undirected_graph(inst), inst.k)
        assert "reference" not in solvers_for(inst) and "reference-weighted" in solvers_for(inst)

    def check_dag(inst: Instance) -> None:
        assert inst.weights is not None and inst.directed
        assert "reference-dag" in solvers_for(inst) and "reference" not in solvers_for(inst)

    und = _collect(st_weighted(st_undirected_k_connected(max_n=9, max_k=3)), check_und, n=100)
    dag = _collect(st_weighted(st_dag(max_n=12)), check_dag, n=80)
    print("weighted strategies:", len(und), len(dag), Counter(i.w_max for i in und))
    assert any(i.w_max == 1 for i in und) and any(i.w_max == 5 for i in und)
    assert any(sum(i.capacities) - i.total_weight == 0 for i in und)
    assert any(sum(i.capacities) - i.total_weight == 3 for i in und)


def test_st_random_digraph_exercises_both_honesty_branches() -> None:
    seen = _collect(st_random_digraph(), lambda inst: None, n=120)
    feac = [check_preconditions(i, True)["feac"] for i in seen]
    print("random_digraph strategy:", len(seen), Counter(feac))
    assert any(feac) and not all(feac)
    assert any(i.k == 1 for i in seen)


def test_st_capacities_modes_and_extreme_vectors() -> None:
    seen = _collect(st_capacities(7, 3), lambda caps: None, n=200)
    assert all(sum(c) == 7 and len(c) == 3 and min(c) >= 0 for c in seen)
    extreme = {tuple(c) for c in seen if sum(1 for x in c if x == 0) == 2}
    assert extreme == {(7, 0, 0), (0, 7, 0), (0, 0, 7)}, extreme
    assert any(sorted(c) == [1, 1, 5] for c in seen)  # unbalanced
    assert any(sorted(c) == [2, 2, 3] for c in seen)  # balanced
    # explicit modes are honoured (fixed modes have tiny search spaces: 2 vectors for "balanced")
    for mode in CAPACITY_MODES:
        for caps in _collect(st_capacities(5, 2, mode), lambda caps: None, n=20, min_seen=2):
            assert sum(caps) == 5 and len(caps) == 2
    assert all(0 in caps for caps in _collect(st_capacities(5, 3, "zeros-allowed"), lambda c: None, n=20, min_seen=2))
    assert {tuple(c) for c in _collect(st_capacities(5, 2, "extreme"), lambda c: None, n=10, min_seen=2)} == {(5, 0), (0, 5)}


# ---------------------------------------------------------------------------
# enumeration counts, recomputed independently
# ---------------------------------------------------------------------------


def _independent_atlas() -> dict[int, list[nx.Graph]]:
    out: dict[int, list[nx.Graph]] = {}
    for graph in graph_atlas_g():
        n = graph.number_of_nodes()
        if 1 <= n <= 5 and nx.is_connected(graph):
            out.setdefault(n, []).append(graph)
    return out


def test_atlas_connected_graph_counts() -> None:
    atlas = _independent_atlas()
    assert {n: len(gs) for n, gs in atlas.items()} == {1: 1, 2: 1, 3: 2, 4: 6, 5: 21}  # 31 in total
    harness = atlas_connected_graphs(5)
    assert len(harness) == 30, "n >= 2 by default: 31 minus K_1"
    assert Counter(g.n for g in harness) == {2: 1, 3: 2, 4: 6, 5: 21}
    # K_1 is dropped even with min_n=1 (κ(K_1) = 0, it yields no instance for any k >= 1)
    assert len(atlas_connected_graphs(5, min_n=1)) == 30
    # connectivity recorded per graph agrees with Even's algorithm (independent of nx.node_connectivity)
    for entry in harness:
        graph = nx.Graph()
        graph.add_nodes_from(range(entry.n))
        graph.add_edges_from(entry.edges)
        assert is_k_vertex_connected(graph, entry.connectivity)
        assert not is_k_vertex_connected(graph, entry.connectivity + 1)


def _independent_instance_count(max_k: int = 3) -> dict[int, int]:
    counts: dict[int, int] = {}
    for n, graphs in _independent_atlas().items():
        for graph in graphs:
            kappa = nx.node_connectivity(graph) if n >= 2 else 0
            for k in range(1, min(kappa, max_k) + 1):
                # k-subsets of terminals × weak compositions of n-k into k parts = C(n-1, k-1)
                counts[n] = counts.get(n, 0) + math.comb(n, k) * math.comb(n - 1, k - 1)
    return counts


def test_undirected_enumeration_counts_match_independent_recount() -> None:
    expected = _independent_instance_count()
    assert expected == {2: 2, 3: 12, 4: 90, 5: 685}
    assert count_undirected_exhaustive(5) == expected
    insts = list(iter_undirected_exhaustive(5, full=True))
    assert Counter(i.n for i in insts) == expected
    assert len({i.name for i in insts}) == len(insts), "duplicate instances in the enumeration"
    ks = Counter(i.k for i in insts)
    assert ks[1] > 0 and ks[3] > 0 and all(i.k < i.n for i in insts)
    for inst in insts:
        assert sum(inst.capacities) == inst.n - inst.k
        assert is_k_vertex_connected(_undirected_graph(inst), inst.k)
    assert any(_is_extreme(i) for i in insts)
    # the sampled n = 6 layer is deterministic and a subset of the full layer
    sampled = [i.name for i in iter_undirected_exhaustive(6, full_up_to=5, seed=tes.SEED, min_n=6)]
    sampled2 = [i.name for i in iter_undirected_exhaustive(6, full_up_to=5, seed=tes.SEED, min_n=6)]
    assert sampled == sampled2 and len(sampled) == 1482
    full6 = {i.name for i in iter_undirected_exhaustive(6, full=True, min_n=6)}
    assert len(full6) == 8272 and set(sampled) <= full6


def test_directed_enumeration_counts_match_independent_recount() -> None:
    expected = 0
    for n in range(1, 5):
        for k in (1, 2):
            if k <= n:
                expected += math.comb(n, k) * 2 ** ((n - k) * (n - 1)) * math.comb(n - 1, k - 1)
    assert expected == 3278 == count_directed_exhaustive(4)
    insts = list(iter_directed_exhaustive(4))
    assert len(insts) == 3278 and len({i.name for i in insts}) == 3278
    assert any(i.k == i.n == 1 for i in insts) and any(i.k == i.n == 2 for i in insts)
    for inst in insts:
        assert inst.directed and sum(inst.capacities) == inst.n - inst.k
        assert not any(u in inst.terminals for u, _v in inst.arcs)


def test_edge_case_module_covers_k_equals_n_and_k_equals_one() -> None:
    marks = {m.args[0]: m.args[1] for m in tec.test_k_equals_n_every_part_is_a_singleton.pytestmark
             if m.name == "parametrize"}
    assert set(marks["n"]) == {1, 2, 3, 5} and set(marks["directed"]) == {False, True}
    assert tec.test_k_equals_one_whole_graph.pytestmark[0].name == "parametrize"
    assert len(tec.test_k_equals_one_whole_graph.pytestmark[0].args[1]) == 4
    # and test_complete_graph_every_k builds K_n with k == n for n = 2..6
    src = inspect.getsource(tec.test_complete_graph_every_k)
    assert "range(1, n + 1)" in src and "[2, 3, 4, 5, 6]" in inspect.getsource(tec)


# ---------------------------------------------------------------------------
# regression side effects of a failure
# ---------------------------------------------------------------------------


def _c4_instance() -> Instance:
    """C_4 with adjacent terminals 0, 1 and capacities (1, 1): the only valid
    partition is {0,3},{1,2}, so swapping the non-terminals is always invalid."""
    return make_instance(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [0, 1], [1, 1], directed=False,
                         name="review_c4", meta={"family": "review"})


def test_property_failure_saves_instance_to_patched_store(patched_store: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _c4_instance()
    assert tps.run_solver(inst, "reference")[0]
    apply_swap_two_vertices(monkeypatch)
    with pytest.raises(AssertionError, match="reference failed on review_c4"):
        tps.assert_solved(inst, "reference", "review")
    path = patched_store / "hyp_review_reference.json"
    assert path.exists(), sorted(p.name for p in patched_store.glob("*"))
    entries = {p.stem: (loaded, meta) for p, loaded, meta in load_all(patched_store)}
    loaded, meta = entries["hyp_review_reference"]
    assert loaded.n == 4 and loaded.terminals == (0, 1) and loaded.undirected_edges == inst.undirected_edges
    assert meta["solver"] == "reference" and meta["expect"] == "ok"
    assert "review" in meta["reason"] and meta["hash"] == instance_hash(inst)
    assert any("not connected" in e for e in meta["errors"]), meta["errors"]


def test_exhaustive_style_failure_is_saved_idempotently(patched_store: Path) -> None:
    inst = next(iter_undirected_exhaustive(4, full=True))
    p1 = save_regression(inst, reason="review: simulated exhaustive failure", extra={"expect": "ok"})
    p2 = save_regression(inst, reason="review: simulated exhaustive failure", extra={"expect": "ok"})
    assert p1 == p2 and p1.parent == patched_store
    assert p1.name.startswith(f"atlas_exhaustive_n{inst.n}_k{inst.k}_")
    (_p, loaded, meta) = next(iter(load_all(patched_store)))
    assert loaded == inst and meta["expect"] == "ok"


def _load_run_exhaustive():
    spec = importlib.util.spec_from_file_location("run_exhaustive_review", ROOT / "scripts" / "run_exhaustive.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_exhaustive_script_counts_and_failure_reporting(patched_store: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    rx = _load_run_exhaustive()
    out = tmp_path / "tally.json"
    rc = rx.main(["--max-n", "4", "--directed-max-n", "3", "--solver", "reference", "--out", str(out)])
    tally = json.loads(out.read_text())
    assert rc == 0 and tally["failures"] == [] and tally["failures_saved"] == []
    assert tally["undirected"]["by_n"] == {"2": 2, "3": 12, "4": 90}
    assert tally["directed"]["instances"] == 1 + 4 + 1 + 48 + 24 == count_directed_exhaustive(3)
    assert tally["undirected"]["solver_runs"] == {"reference": 104}
    assert not list(patched_store.glob("*.json"))
    # with a mutant: non-zero exit, failures listed and stored (in the patched directory)
    apply_swap_two_vertices(monkeypatch)
    out2 = tmp_path / "tally_mutant.json"
    rc = rx.main(["--max-n", "4", "--directed-max-n", "3", "--solver", "reference", "--out", str(out2)])
    tally = json.loads(out2.read_text())
    assert rc == 1 and tally["failures"]
    assert {f["expect"] for f in tally["failures"]} == {"ok"}
    saved = sorted(patched_store.glob("*.json"))
    assert saved and len(saved) == len(set(tally["failures_saved"]))
    for _p, loaded, meta in load_all(patched_store):
        assert meta["expect"] == "ok" and meta["solver"] == "reference"
        assert "run_exhaustive" in meta["reason"]
        loaded.validate()


def test_verifier_rejects_the_swapped_partition_directly() -> None:
    """The verifier -- not only the harness bookkeeping -- rejects the mutant output."""
    inst = _c4_instance()
    rep = verify_instance_parts(inst, [[0, 2], [1, 3]])
    assert not rep.valid and any("not connected" in e for e in rep.errors)
    assert verify_instance_parts(inst, [[0, 3], [1, 2]]).valid
    # a size violation is reported even when connectivity holds
    rep = verify_instance_parts(inst, [[0, 2, 3], [1]])
    assert not rep.valid and any("expected c_" in e for e in rep.errors)


def test_directed_exhaustive_k_equals_n_instances_are_solved() -> None:
    """k == n (n = 1, 2) lives only in the directed enumeration and the edge cases."""
    for inst in iter_directed_exhaustive(2):
        if inst.k == inst.n:
            for name, fn in solvers_for(inst).items():
                res = fn(inst)
                assert res.status == "ok" and res.parts == [[t] for t in inst.terminals], name


def test_atlas_iteration_sizes_are_small() -> None:
    """Guard for the harness's runtime claim: n <= 5 needs < 800 instances."""
    assert sum(count_undirected_exhaustive(5).values()) == 789
    assert list(itertools.islice(iter_undirected_exhaustive(2), 3))  # generator works lazily


# ---------------------------------------------------------------------------
# the offline runner under a hang, the shared digraph enumeration, the tally location
# ---------------------------------------------------------------------------


def test_run_exhaustive_script_bounds_hangs(patched_store: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """With the hanging mutant the script reports ``timeout`` failures, stops
    after ``--max-timeouts`` of them and still writes its tally."""
    rx = _load_run_exhaustive()
    monkeypatch.setattr(rx, "_kw", _kwargs_without_debug(rx._kw))  # invariants off: the mutant really hangs
    apply_contract_drops_in_arcs(monkeypatch)
    out = tmp_path / "tally_hang.json"
    t0 = time.monotonic()
    rc = rx.main(["--max-n", "5", "--min-n", "5", "--no-directed", "--solver", "reference",
                  "--time-limit", "2", "--max-timeouts", "1", "--out", str(out)])
    assert rc == 1 and time.monotonic() - t0 < 90
    tally = json.loads(out.read_text())
    assert tally["aborted"] is True and tally["timeouts"] == 1 and tally["time_limit"] == 2.0
    assert tally["undirected"]["timeouts"] == 1 and 0 < tally["undirected"]["instances"] < 685
    hung = [f for f in tally["failures"] if f["kind"] == "timeout"]
    assert len(hung) == 1 and hung[0]["solver"] == "reference" and "did not return within 2 s" in hung[0]["message"]
    stored = [meta for _p, _i, meta in load_all(patched_store)]
    assert any(m["kind"] == "timeout" and m["expect"] == "ok" for m in stored)


def test_directed_block_generator_is_shared_by_script_and_harness() -> None:
    """``scripts/run_exhaustive.directed_job`` iterates ``strategies.directed_cases_for_block``
    instead of re-implementing the enumeration, and the block generator
    reproduces the flat enumeration exactly."""
    rx = _load_run_exhaustive()
    assert rx.directed_cases_for_block is directed_cases_for_block and rx.directed_blocks is directed_blocks
    src = inspect.getsource(rx.directed_job)
    assert "directed_cases_for_block(" in src and "make_instance" not in src
    flat = [i.name for i in iter_directed_exhaustive(3)]
    blocks = [i.name for blk in directed_blocks(3) for i in directed_cases_for_block(*blk)]
    assert flat == blocks and len(flat) == count_directed_exhaustive(3) == 78
    # the script's job on one block yields exactly that block
    counts, failures = rx.directed_job({"block": (3, 2, (0, 2)), "seed": 0, "mask_sample": None,
                                        "min_capacity": 0, "solvers": ["reference"], "oracles": False,
                                        "time_limit": 5.0})
    assert failures == [] and counts["instances"] == 4 * 2 == len(list(directed_cases_for_block(3, 2, (0, 2))))
    # a seeded arc-subset sample is deterministic, a subset of the full block and counted correctly
    full = {i.name for i in directed_cases_for_block(4, 2, (0, 1))}
    s1 = [i.name for i in directed_cases_for_block(4, 2, (0, 1), mask_sample=5, seed=7)]
    s2 = [i.name for i in directed_cases_for_block(4, 2, (0, 1), mask_sample=5, seed=7)]
    s3 = [i.name for i in directed_cases_for_block(4, 2, (0, 1), mask_sample=5, seed=8)]
    assert s1 == s2 != s3 and set(s1) <= full and len(s1) == 5 * 3  # 5 masks x 3 compositions of 2 into 2
    assert len(full) == 64 * 3
    assert count_directed_exhaustive(4, (2,), min_n=4, mask_sample=5) == 6 * 15
    # min_capacity keeps only compositions with every part >= 1: the harness's k = 3 layer
    layer = list(iter_directed_exhaustive(seed=tes.SEED, **tes.K3_LAYER))
    assert len(layer) == count_directed_exhaustive(**tes.K3_LAYER) == 20 * 64 == 1280
    assert len({i.name for i in layer}) == 1280
    assert all(i.n == 6 and i.k == 3 and tuple(i.capacities) == (1, 1, 1) for i in layer)
    assert all(not any(u in i.terminals for u, _v in i.arcs) for i in layer)


def test_tally_is_not_written_into_the_source_tree(tmp_path_factory: pytest.TempPathFactory,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tes.TALLY_ENV, raising=False)
    path = tes.tally_path(tmp_path_factory)
    assert ROOT not in path.parents and path.name == "tally.json"
    assert tmp_path_factory.getbasetemp() in path.parents
    custom = tmp_path_factory.getbasetemp() / "custom_tally.json"
    monkeypatch.setenv(tes.TALLY_ENV, str(custom))
    assert tes.tally_path(tmp_path_factory) == custom
    assert not (ROOT / "tests" / ".exhaustive_tally.json").exists(), (
        "stale artifact of the old harness: delete tests/.exhaustive_tally.json")
