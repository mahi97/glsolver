"""Registry of solver backends for parametrized tests and benchmarks."""
from __future__ import annotations

from typing import Any, Callable

from glsolver.api import GLResult, core_available, glpartition
from glsolver.instance import Instance

SolverFn = Callable[..., GLResult]


def _make(algo: str) -> SolverFn:
    def run(inst: Instance, **kw: Any) -> GLResult:
        kw.setdefault("verify", True)
        return glpartition(inst, algorithm=algo, **kw)

    run.__name__ = f"solve_{algo.replace('-', '_')}"
    return run


def available_solvers(include_oracles: bool = True) -> dict[str, SolverFn]:
    """Name → callable for every backend usable in this environment."""
    names = ["reference", "reference-weighted", "reference-dag"]
    names += [algo for algo in ("general", "weighted", "dag") if core_available(algo)]
    if include_oracles:
        names += ["bruteforce"]
        try:
            import scipy.optimize  # noqa: F401

            names.append("ilp")
        except Exception:  # pragma: no cover
            pass
    return {n: _make(n) for n in names}


def solvers_for(inst: Instance, include_oracles: bool = False) -> dict[str, SolverFn]:
    """Backends applicable to ``inst`` (weighted/unweighted, DAG or not)."""
    from glsolver.preconditions import is_dag, is_k_T_connected_dag

    all_ = available_solvers(include_oracles)
    dag_ok = is_dag(inst) and is_k_T_connected_dag(inst)[0]
    out: dict[str, SolverFn] = {}
    for name, fn in all_.items():
        if name in ("reference", "general") and inst.is_weighted:
            continue
        if name in ("reference-dag", "dag") and not dag_ok:
            continue
        out[name] = fn
    return out
