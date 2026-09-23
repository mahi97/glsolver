"""Public solver API: :func:`glpartition`, :func:`partition`, :class:`GLResult`.

Dispatch (docs/implementation.md):

* ``algorithm="auto"``: DAG with out-degree ≥ k → DAG solver; weighted → weighted
  core; otherwise the general core. If the C++ core is not built, the pure
  Python reference solvers are used instead (``stats["backend"]`` says which).
* explicit names: ``general``, ``weighted``, ``dag`` (C++ core), ``reference``,
  ``reference-weighted``, ``reference-dag`` (pure Python), ``bruteforce``, ``ilp``
  (oracles).

Every result carries the normalized :class:`Instance`, the partition, an
in-arborescence certificate when the algorithm produces one, statistics, and —
unless ``verify=False`` — the verdict of the independent verifier
(:mod:`glsolver.verify`), which shares no code with any solver.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from glsolver.instance import Instance, from_networkx, make_instance
from glsolver.verify import VerificationReport, verify_instance_parts

CORE_ALGORITHMS = ("general", "weighted", "dag")
REFERENCE_ALGORITHMS = ("reference", "reference-weighted", "reference-dag")
ORACLE_ALGORITHMS = ("bruteforce", "ilp")
ALL_ALGORITHMS = ("auto",) + CORE_ALGORITHMS + REFERENCE_ALGORITHMS + ORACLE_ALGORITHMS


@dataclass
class GLResult:
    instance: Instance
    algorithm: str
    status: str  # "ok" | "precondition_failed" | "infeasible" | "timeout" | "error"
    parts: list[list[int]]
    assignment: list[int]
    message: str = ""
    valid: bool | None = None
    verification: VerificationReport | None = None
    runtime: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)
    certificate: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None
    preconditions: dict[str, Any] | None = None
    labels: list[Any] | None = None  # original node labels (networkx input)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def parts_labels(self) -> list[list[Any]]:
        """Parts expressed in the original node labels (identity for integer input)."""
        if self.labels is None:
            return [list(p) for p in self.parts]
        return [[self.labels[v] for v in p] for p in self.parts]

    def to_dict(self) -> dict[str, Any]:
        from glsolver.io import result_to_dict

        d = result_to_dict(self)
        d["preconditions"] = _jsonable(self.preconditions)
        d["n"], d["k"], d["m"] = self.instance.n, self.instance.k, self.instance.m
        return d

    def summary(self) -> str:
        v = "unverified" if self.valid is None else ("VALID" if self.valid else "INVALID")
        return (
            f"{self.algorithm}: status={self.status} n={self.instance.n} m={self.instance.m} "
            f"k={self.instance.k} runtime={self.runtime:.4f}s verifier={v}"
        )


def _lazy_certificate_property() -> property:
    """``GLResult.certificate`` backed by ``_certificate``: a dict, or a zero-argument callable that builds
    it on first access (the C++ backends hand over ``parent`` / ``witness`` arrays of length n; the
    ``{v: parent_v}`` dicts are only built when somebody reads the certificate, RESEARCH_NOTES.md E4).
    Installed after the dataclass is built, so the field keeps its documented type and default."""

    def fget(self: GLResult) -> dict[str, Any]:
        c = self.__dict__.get("_certificate")
        if callable(c):
            c = c()
            self.__dict__["_certificate"] = c
        return c

    def fset(self: GLResult, value: Any) -> None:
        self.__dict__["_certificate"] = value

    return property(fget, fset, doc="in-arborescence certificate (docs/api.md); built lazily by the C++ backends")


GLResult.certificate = _lazy_certificate_property()  # type: ignore[assignment]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


#: set to ``1`` to ignore the C++ core even when it is installed (e.g. a build
#: instrumented with a sanitizer that aborts the interpreter at import time);
#: every ``algorithm="auto"`` call then uses the pure-Python reference solvers
NO_CORE_ENV = "GLSOLVER_NO_CORE"


def _core():
    if os.environ.get(NO_CORE_ENV, "") not in ("", "0", "false", "no"):
        return None
    try:
        from glsolver import _core  # type: ignore

        return _core
    except Exception:  # pragma: no cover - core not built
        return None


_CORE_ENTRY = {"general": "solve_general", "weighted": "solve_weighted", "dag": "dag_partition"}


def core_available(algorithm: str | None = None) -> bool:
    """Is the C++ backend usable? With ``algorithm`` given, checks that specific binding
    (a partially built extension must never make a backend look available)."""
    c = _core()
    if c is None:
        return False
    if algorithm is None:
        return all(hasattr(c, name) for name in _CORE_ENTRY.values())
    return hasattr(c, _CORE_ENTRY[algorithm])


def coerce_instance(
    graph: Any,
    terminals: Sequence[Any] | None = None,
    capacities: Sequence[int] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    weights: Any = None,
    directed: bool | None = None,
    name: str = "",
) -> tuple[Instance, list[Any] | None]:
    """Accept an Instance, a networkx graph, or ``(n, edges)``; return ``(instance, labels)``."""
    if isinstance(graph, Instance):
        if any(x is not None for x in (terminals, capacities, sizes, weights)):
            raise ValueError("when passing an Instance, do not pass terminals/capacities/sizes/weights")
        return graph, None
    if terminals is None:
        raise ValueError("terminals are required")
    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        inst, labels = from_networkx(
            graph, terminals, capacities, sizes=sizes, weights=weights, directed=directed, name=name
        )
        return inst, labels
    if isinstance(graph, tuple) and len(graph) == 2:
        n, edges = graph
        inst = make_instance(
            int(n), edges, terminals, capacities, sizes=sizes, weights=weights,
            directed=True if directed is None else directed, name=name,
        )
        return inst, None
    raise TypeError("graph must be an Instance, a networkx graph, or a tuple (n, edges)")


def choose_algorithm(inst: Instance, prefer_core: bool = True) -> str:
    """The ``auto`` policy (docs/implementation.md)."""
    from glsolver.preconditions import is_dag, is_k_T_connected_dag

    dag_ok = is_dag(inst) and is_k_T_connected_dag(inst)[0]
    if dag_ok:
        return "dag" if prefer_core and core_available("dag") else "reference-dag"
    if inst.is_weighted:
        return "weighted" if prefer_core and core_available("weighted") else "reference-weighted"
    return "general" if prefer_core and core_available("general") else "reference"


def _peak_rss_mb() -> float:
    """Peak resident set of THIS process, in MiB.

    ``ru_maxrss`` is inherited across ``fork``: a process spawned from a large
    parent reports the parent's footprint (measured: a child of a 3 GB parent
    reports 2 879 MB while its true peak is 18 MB).  Linux exposes the honest
    figure as ``VmHWM`` in ``/proc/self/status``; fall back to ``ru_maxrss``
    elsewhere.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def glpartition(
    graph: Any,
    terminals: Sequence[Any] | None = None,
    capacities: Sequence[int] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    weights: Any = None,
    directed: bool | None = None,
    algorithm: str = "auto",
    verify: bool = True,
    verify_preconditions: Any = "auto",
    trace: bool = False,
    seed: int = 0,
    threads: int = 0,
    time_limit: float | None = None,
    debug: bool = False,
    options: dict[str, Any] | None = None,
    name: str = "",
) -> GLResult:
    """Solve a Győri–Lovász instance (docs/api.md).

    ``verify_preconditions``: ``False`` — rely on the solver's intrinsic checks
    only; ``True`` — additionally run the NetworkX-based cross-check
    (:func:`glsolver.preconditions.check_preconditions`) before solving;
    ``"auto"`` — run it only for the oracles and only when ``n <= 400``.
    The general/weighted solvers *always* detect a failing FEAC/FESAC
    themselves (status ``precondition_failed``); the DAG solver always checks
    acyclicity and out-degrees.
    """
    inst, labels = coerce_instance(
        graph, terminals, capacities, sizes=sizes, weights=weights, directed=directed, name=name
    )
    if algorithm not in ALL_ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {ALL_ALGORITHMS}")
    algo = choose_algorithm(inst) if algorithm == "auto" else algorithm
    opts = dict(options or {})
    opts.setdefault("seed", seed)
    opts.setdefault("threads", threads)
    opts.setdefault("debug", debug)
    opts.setdefault("trace", trace)
    if time_limit is not None:
        opts.setdefault("time_limit", time_limit)

    pre: dict[str, Any] | None = None
    if verify_preconditions is True or (
        verify_preconditions == "auto" and algo in ORACLE_ALGORITHMS and inst.n <= 400
    ):
        from glsolver.preconditions import check_preconditions

        pre = check_preconditions(inst, True)

    t0 = time.perf_counter()
    try:
        res = _run(inst, algo, opts)
    except Exception as exc:  # solver crashed: report, never hide
        res = GLResult(inst, algo, "error", [], [], message=f"{type(exc).__name__}: {exc}")
        if debug:
            raise
    res.runtime = time.perf_counter() - t0
    res.labels = labels
    res.preconditions = pre
    res.stats.setdefault("peak_rss_mb", _peak_rss_mb())
    res.stats.setdefault("backend", "core" if algo in CORE_ALGORITHMS else ("reference" if algo in REFERENCE_ALGORITHMS else "oracle"))
    res.stats.setdefault("requested_algorithm", algorithm)
    if res.status == "ok":
        res.assignment = _assignment_from_parts(inst, res.parts)
        if verify:
            rep = verify_instance_parts(inst, res.parts)
            res.verification = rep
            res.valid = rep.valid
            if not rep.valid:
                res.message = (res.message + "; " if res.message else "") + "VERIFIER REJECTED: " + "; ".join(rep.errors[:5])
    return res


def partition(graph: Any, terminals: Sequence[Any], sizes: Sequence[int] | None = None, **kw: Any) -> GLResult:
    """Classical undirected Győri–Lovász: ``sizes`` are the part sizes ``|V_i|``."""
    if sizes is None and "capacities" not in kw:
        raise ValueError("partition() needs sizes= (or capacities=)")
    kw.setdefault("directed", False)
    return glpartition(graph, terminals, sizes=sizes, **kw)


# --------------------------------------------------------------------------- backends

def _assignment_from_parts(inst: Instance, parts: Sequence[Sequence[int]]) -> list[int]:
    """Part index per vertex (-1 if in no part); a later part wins; ids outside ``0..n-1`` are ignored."""
    n = inst.n
    a = np.full(n, -1, dtype=np.int64)
    for i, p in enumerate(parts):
        pv = np.asarray(list(p), dtype=np.int64).reshape(-1)
        a[pv[(pv >= 0) & (pv < n)]] = i
    return a.tolist()


def _parts_from_assignment(inst: Instance, assignment: Sequence[int]) -> list[list[int]]:
    """``parts[i]`` = ascending vertices with ``assignment[v] == i`` (negative entries are unassigned)."""
    k = len(inst.terminals)
    a = np.asarray(assignment, dtype=np.int64).reshape(-1)
    idx = np.flatnonzero(a >= 0)
    if idx.size and int(a[idx].max()) >= k:
        raise IndexError("part index out of range in the assignment")
    order = idx[np.argsort(a[idx], kind="stable")]  # grouped by part, ascending within a part
    counts = np.bincount(a[idx], minlength=k)
    ends = np.cumsum(counts)
    return [order[s:e].tolist() for s, e in zip((ends - counts).tolist(), ends.tolist())]


def _run(inst: Instance, algo: str, opts: dict[str, Any]) -> GLResult:
    if algo in REFERENCE_ALGORITHMS:
        return _run_reference(inst, algo, opts)
    if algo in ORACLE_ALGORITHMS:
        return _run_oracle(inst, algo, opts)
    if algo in CORE_ALGORITHMS:
        return _run_core(inst, algo, opts)
    raise ValueError(algo)


def _run_reference(inst: Instance, algo: str, opts: dict[str, Any]) -> GLResult:
    from glref.trace import Tracer

    tracer = Tracer(enabled=bool(opts.get("trace")), record_cuts=bool(opts.get("record_cuts", False)))
    debug = bool(opts.get("debug"))
    if algo == "reference":
        if inst.is_weighted:
            raise ValueError("algorithm='reference' is the unweighted algorithm; use 'reference-weighted'")
        from glref.unweighted import gl_partition

        r = gl_partition(inst, tracer=tracer, debug=debug)
    elif algo == "reference-weighted":
        from glref.weighted import gl_weighted_partition

        r = gl_weighted_partition(inst, tracer=tracer, debug=debug)
    else:
        from glref.dag import gl_dag_partition

        policy = opts.get("dag_policy", "max_residual")
        r = gl_dag_partition(inst, policy=policy, tracer=tracer, debug=debug)
    stats = r.stats.as_dict() if hasattr(r.stats, "as_dict") else dict(r.stats)
    cert: dict[str, Any] = {}
    if getattr(r, "parents", None):
        cert["parents"] = dict(r.parents)
    if getattr(r, "witness", None):
        cert["witness"] = dict(r.witness)
    return GLResult(
        inst, algo, r.status, [list(p) for p in r.parts] if r.status == "ok" else [], [],
        message=r.message, stats=stats, certificate=cert, trace=r.trace,
    )


def _run_oracle(inst: Instance, algo: str, opts: dict[str, Any]) -> GLResult:
    tl = opts.get("time_limit")
    if algo == "bruteforce":
        from glsolver.oracle.bruteforce import bruteforce_partition

        status, parts = bruteforce_partition(inst, time_limit=tl, seed=int(opts.get("seed", 0)))
    else:
        from glsolver.oracle.ilp import ilp_partition

        status, parts = ilp_partition(inst, time_limit=tl)
    msg = {"ok": "ok", "infeasible": "no partition exists (exhaustive/exact oracle)", "timeout": "time limit reached"}[status]
    return GLResult(inst, algo, status, [list(p) for p in parts] if parts else [], [], message=msg)


def _core_options(inst: Instance, opts: dict[str, Any]) -> dict[str, Any]:
    o = {
        "threads": int(opts.get("threads", 0)),
        "seed": int(opts.get("seed", 0)),
        "greedy_contraction": bool(opts.get("greedy_contraction", True)),
        "lazy_shift": bool(opts.get("lazy_shift", True)),
        "batch_unused_arcs": bool(opts.get("batch_unused_arcs", True)),
        # C1 deletion-aware routing (RESEARCH_NOTES E5): "bfs" (default) | "avoid"
        "routing": str(opts.get("routing", "bfs")),
        "debug_asserts": bool(opts.get("debug", False)),
        "trace": bool(opts.get("trace", False)),
        "record_cuts": bool(opts.get("record_cuts", False)),
        "check_precondition": bool(opts.get("check_precondition", True)),
    }
    return o


def _core_certificate_builder(
    inst: Instance, parent: Sequence[int] | None, witness: Sequence[int] | None
) -> Callable[[], dict[str, Any]]:
    """Deferred ``{"parents": {v: parent_v}, "witness": {v: terminal}}`` from the core's per-vertex arrays
    (entries ``< 0`` / ``None`` mean "none"); ``witness`` is present iff the core returned a non-empty list."""
    terminals = inst.terminals

    def build() -> dict[str, Any]:
        cert: dict[str, Any] = {}
        par = np.asarray([-1 if p is None else p for p in (parent or [])], dtype=np.int64)
        keep = np.flatnonzero(par >= 0)
        cert["parents"] = dict(zip(keep.tolist(), par[keep].tolist()))
        if witness:
            wit = np.asarray([-1 if t is None else t for t in witness], dtype=np.int64)
            keep = np.flatnonzero(wit >= 0)
            cert["witness"] = {v: terminals[t] for v, t in zip(keep.tolist(), wit[keep].tolist())}
        return cert

    return build


def _run_core(inst: Instance, algo: str, opts: dict[str, Any]) -> GLResult:
    core = _core()
    if core is None or not core_available(algo):
        raise RuntimeError(f"the C++ backend for {algo!r} is not built; use a reference/oracle algorithm or rebuild")
    arcs = inst.arc_array  # int32 (m, 2): read in place by the binding, no per-arc Python objects (E4)
    caps = [int(c) for c in inst.capacities]
    w = [int(x) for x in inst.weights] if inst.weights is not None else [1] * inst.n
    if algo == "dag":
        policy = {"max_residual": 0, "round_robin": 1, "first": 2}.get(opts.get("dag_policy", "max_residual"), 0)
        variant = int(opts.get("dag_variant", 1))
        out = core.dag_partition(inst.n, arcs, list(inst.terminals), caps, w, policy, variant,
                                 bool(opts.get("check_precondition", True)), bool(opts.get("trace", False)))
    elif algo == "weighted":
        out = core.solve_weighted(inst.n, arcs, list(inst.terminals), caps, w, _core_options(inst, opts))
    else:
        if inst.is_weighted:
            raise ValueError("algorithm='general' is unweighted; use 'weighted'")
        out = core.solve_general(inst.n, arcs, list(inst.terminals), caps, _core_options(inst, opts))
    status = out["status"]
    parts = _parts_from_assignment(inst, out["assignment"]) if status == "ok" else []
    cert: dict[str, Any] | Callable[[], dict[str, Any]] = {}
    if status == "ok":
        cert = _core_certificate_builder(inst, out.get("parent"), out.get("witness"))
    trace = None
    if opts.get("trace") and out.get("trace") is not None:
        trace = []
        for ev in out["trace"]:
            if isinstance(ev, dict):
                trace.append(ev)
            else:
                typ, payload = ev
                d = json.loads(payload) if payload else {}
                d["type"] = typ
                trace.append(d)
    stats = dict(out.get("stats", {}))
    if "k_T_connected" in out:
        stats["k_T_connected"] = out["k_T_connected"]
    return GLResult(inst, algo, status, parts, [], message=out.get("message", ""), stats=stats,
                    certificate=cert, trace=trace)
