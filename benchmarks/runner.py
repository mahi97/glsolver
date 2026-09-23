"""Benchmark runner: config grid -> deterministic cached instances -> isolated runs -> JSONL.

A configuration (``benchmarks/configs/*.json``) describes a grid of instances
and algorithms::

    {
      "name": "smoke",
      "families": ["harary", "erdos_renyi", {"name": "counterexample", "copies": 1},
                   {"name": "regular_margin", "generator": "random_regular_graph",
                    "params": {"d": "4*k"}}],
      "sizes": [10, 20],                 # n (ignored by paper_* and counterexample)
      "k": [2, 0.25],                    # absolute (int) or fraction of n (0 < x < 1)
      "modes": ["balanced"],             # capacity modes of glsolver.generators.capacity_vector
      "seeds": [1],
      "variants": ["plain", {"type": "directed", "drop_fraction": 0.5},
                   {"type": "weighted", "w_max": 4, "slack": 0}],
      "algorithms": ["reference", "bruteforce"],          # names of the solver registry
      "algorithm_options": {"dag": [{"label": "heap", "dag_variant": 0}]},
      "algorithm_limits": {"bruteforce": {"max_n": 12}, "reference": {"max_n": 100}},
      "repeats": 1, "warmup": 0, "timeout": 30, "threads": 1
    }

Every ``(instance, algorithm[:option set])`` pair runs in its own **subprocess**
(``python benchmarks/runner.py --worker ...``) so that crashes and timeouts are
contained.  The worker streams one JSON line per trial on stdout; the parent
enforces the per-trial timeout (no output for ``timeout`` seconds -> the
process group is killed), reaps the child with ``os.wait4`` to obtain its CPU
time and peak RSS, and appends one row per pair to
``benchmarks/results/<config>.jsonl``.  The independent verifier is always run
by the worker, separately timed, and its verdict is stored per trial.

Instances are generated deterministically from ``(family, n, k, mode, seed,
variant)`` and cached as canonical JSON under ``benchmarks/instances/`` (plus a
small ``*.info.json`` sidecar with n, m, k, density, generator metadata, the
SHA-256 of the instance file and a fingerprint of the generator sources), so
that every algorithm sees identical inputs.  A cached entry is re-used only
when its digest still matches the file and the generator fingerprint still
matches the installed code; otherwise it is regenerated.

The driver never imports the compiled ``glsolver._core`` itself: its
availability is probed in a subprocess (:func:`benchmarks.machine.core_info`),
so a broken or sanitizer-linked build cannot kill a benchmark run.

Timeouts: ``timeout`` is a per-phase inactivity limit; the process group is
killed once no progress has been reported for ``timeout + grace`` seconds,
where ``grace`` defaults to ``max(2, 0.1 * timeout)`` and can be set in the
config.  The row stores both values.

Exit codes of :func:`run_from_cli`: ``0`` when every executed pair is ``ok``
(or ``precondition_failed``/``infeasible``), ``2`` when at least one pair
crashed (``error``) or timed out, ``3`` when at least one partition was
INVALID (takes precedence).

Entry points: :func:`run_config`, :func:`run_from_cli` (used by ``glsolve
benchmark``) and ``python -m benchmarks.runner CONFIG [options]``.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import functools
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow ``python benchmarks/runner.py`` from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.machine import collect_machine_info, core_info, summarize_stderr  # noqa: E402

BENCH_DIR = REPO_ROOT / "benchmarks"
CONFIG_DIR = BENCH_DIR / "configs"
RESULTS_DIR = BENCH_DIR / "results"
INSTANCES_DIR = BENCH_DIR / "instances"

SCHEMA_VERSION = 1
MAX_SIZE_SAMPLES = 200  # graph_size_over_time is thinned to at most this many samples
DEFAULT_STORE_PARTS_MAX_N = 30  # rows keep the partition (for the dashboard SVG) up to this n
EXACT_CONNECTIVITY_MAX_N = 100  # undirected instances up to this n get exact vertex connectivity
FIXED_FAMILIES = ("counterexample", "paper_running_example", "paper_contract_counterexample",
                  "paper_essential_example")
VARIANT_TYPES = ("plain", "directed", "weighted")
KNOWN_ALGORITHMS = ("auto", "general", "weighted", "dag", "reference", "reference-weighted",
                    "reference-dag", "bruteforce", "ilp")
CORE_ALGORITHMS = ("general", "weighted", "dag")
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GLSOLVER_THREADS")
# environment variables whose *effective* values in the worker are recorded per row
PROVENANCE_ENV = _THREAD_ENV + ("PYTHONHASHSEED",)
# source files whose content determines the generated instances (cache fingerprint)
_GENERATOR_SOURCES = ("glsolver.generators", "glref.counterexample")
STDERR_HEAD_CHARS = 1000  # characters of the worker's stderr kept at the head / tail of a row
STDERR_TAIL_CHARS = 1000

# exit codes of run_from_cli
EXIT_OK, EXIT_FAILED_RUNS, EXIT_INVALID = 0, 2, 3

# keys every result row carries (checked by tests/test_benchmarks.py)
REQUIRED_ROW_KEYS = (
    "schema", "config", "instance", "n", "m", "k", "density", "algorithm", "algorithm_label",
    "options", "threads", "repeats", "warmup", "timeout", "grace", "trials", "median", "status",
    "valid", "timed_out", "error", "result", "child", "git_commit", "glsolver_version", "machine",
    "load_average", "timestamp",
)

Log = Callable[[str], None]


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchConfig:
    """Normalized benchmark configuration (see the module docstring for the JSON form)."""

    name: str
    families: list[dict[str, Any]]
    sizes: list[int]
    ks: list[float]
    modes: list[str]
    seeds: list[int]
    variants: list[dict[str, Any]]
    algorithms: list[str]
    algorithm_options: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    algorithm_limits: dict[str, dict[str, Any]] = field(default_factory=dict)
    repeats: int = 1
    warmup: int = 0
    timeout: float = 60.0
    threads: int = 1
    description: str = ""
    store_parts_max_n: int = DEFAULT_STORE_PARTS_MAX_N
    seed_offset: int = 0
    grace: float | None = None  # None -> max(2, 0.1 * timeout), see effective_grace()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_grace(cfg: BenchConfig) -> float:
    """Seconds added to ``cfg.timeout`` before the process group is killed.

    The parent polls the worker's progress lines; a worker that has been silent
    for ``timeout + grace`` seconds is killed.  The default ``max(2, 0.1 *
    timeout)`` absorbs scheduling jitter and the cost of the final flush; a
    config may pin ``grace`` explicitly (``0`` for a hard limit).
    """
    if cfg.grace is not None:
        return max(0.0, float(cfg.grace))
    return max(2.0, 0.1 * float(cfg.timeout))


def _normalize_family(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry}
    if isinstance(entry, dict) and "name" in entry:
        return {str(k): v for k, v in entry.items()}
    raise ValueError(f"family entries must be names or objects with a 'name': {entry!r}")


def _normalize_variant(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        entry = {"type": entry}
    if not isinstance(entry, dict) or "type" not in entry:
        raise ValueError(f"variant entries need a 'type': {entry!r}")
    v = {str(k): val for k, val in entry.items()}
    if v["type"] not in VARIANT_TYPES:
        raise ValueError(f"unknown variant type {v['type']!r}; choose from {VARIANT_TYPES}")
    if v["type"] == "directed":
        v.setdefault("drop_fraction", 0.5)
    if v["type"] == "weighted":
        v.setdefault("w_max", 4)
        v.setdefault("slack", 0)
    return v


def _normalize_ks(ks_raw: Iterable[Any]) -> list[float]:
    ks: list[float] = []
    for x in ks_raw:
        x = float(x)
        if x <= 0:
            raise ValueError(f"k values must be positive, got {x}")
        ks.append(int(x) if x >= 1 and float(x).is_integer() else x)
    return ks


# family-level keys that override the grid instead of being generator parameters
_FAMILY_OVERRIDES = ("sizes", "k", "modes", "seeds", "variants")


def config_from_dict(d: dict[str, Any], name: str | None = None) -> BenchConfig:
    """Validate and normalize a config dict (missing lists get sensible defaults)."""
    if not isinstance(d, dict):
        raise TypeError("config must be a JSON object")
    cfg_name = str(d.get("name") or name or "benchmark")
    ks = _normalize_ks(d.get("k", d.get("ks", [2])))
    algorithms = [str(a) for a in d.get("algorithms", ["reference"])]
    for a in algorithms:
        if a not in KNOWN_ALGORITHMS:
            raise ValueError(f"unknown algorithm {a!r}; choose from {KNOWN_ALGORITHMS}")
    options = {}
    for algo, sets in (d.get("algorithm_options") or {}).items():
        if not isinstance(sets, list):
            raise ValueError(f"algorithm_options[{algo!r}] must be a list of option objects")
        options[str(algo)] = [dict(s) for s in sets]
    return BenchConfig(
        name=cfg_name,
        families=[_normalize_family(f) for f in d.get("families", ["harary"])],
        sizes=[int(n) for n in d.get("sizes", d.get("n", [20]))],
        ks=ks,
        modes=[str(m) for m in d.get("modes", ["balanced"])],
        seeds=[int(s) for s in d.get("seeds", [1])],
        variants=[_normalize_variant(v) for v in d.get("variants", ["plain"])],
        algorithms=algorithms,
        algorithm_options=options,
        algorithm_limits={str(a): dict(v) for a, v in (d.get("algorithm_limits") or {}).items()},
        repeats=max(1, int(d.get("repeats", 1))),
        warmup=max(0, int(d.get("warmup", 0))),
        timeout=float(d.get("timeout", 60.0)),
        threads=int(d.get("threads", 1)),
        description=str(d.get("description", "")),
        store_parts_max_n=int(d.get("store_parts_max_n", DEFAULT_STORE_PARTS_MAX_N)),
        seed_offset=int(d.get("seed_offset", 0)),
        grace=None if d.get("grace") is None else float(d["grace"]),
    )


def resolve_config_path(spec: str | Path) -> Path:
    """``smoke`` -> ``benchmarks/configs/smoke.json``; existing paths are returned as is."""
    p = Path(spec)
    if p.exists():
        return p
    cand = CONFIG_DIR / (p.name if p.suffix == ".json" else f"{p.name}.json")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"config {spec!r} not found (looked in {CONFIG_DIR})")


def load_config(spec: str | Path) -> BenchConfig:
    path = resolve_config_path(spec)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return config_from_dict(data, name=path.stem)


# ---------------------------------------------------------------------------
# instance specifications and deterministic generation
# ---------------------------------------------------------------------------
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "min": min, "max": max, "int": int, "round": round, "abs": abs,
    "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
}


def eval_param(expr: Any, n: int, k: int) -> Any:
    """Evaluate a generator parameter that may be an arithmetic expression in ``n`` and ``k``.

    Only numbers, ``n``/``k``, the operators ``+ - * / // % **``, unary minus
    and the functions ``min max int round abs ceil floor sqrt log log2`` are
    accepted (a tiny ``ast`` walker, no ``eval``).  Non-string values are
    returned unchanged.
    """
    if not isinstance(expr, str):
        return expr
    tree = ast.parse(expr, mode="eval")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "n":
                return n
            if node.id == "k":
                return k
            raise ValueError(f"unknown name {node.id!r} in parameter expression {expr!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = walk(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            ops: dict[type, Callable[[Any, Any], Any]] = {
                ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y,
                ast.Mult: lambda x, y: x * y, ast.Div: lambda x, y: x / y,
                ast.FloorDiv: lambda x, y: x // y, ast.Mod: lambda x, y: x % y,
                ast.Pow: lambda x, y: x ** y,
            }
            if type(node.op) not in ops:
                raise ValueError(f"operator not allowed in {expr!r}")
            return ops[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.func.id](*[walk(a) for a in node.args])
        raise ValueError(f"unsupported syntax in parameter expression {expr!r}")

    return walk(tree)


def resolve_k(kspec: float, n: int) -> int:
    """Absolute ``k`` (int >= 1) or a fraction of ``n`` (``0 < k < 1``), clamped to ``[1, n-1]``."""
    if 0 < kspec < 1:
        k = int(round(kspec * n))
    else:
        k = int(kspec)
    return max(1, min(k, max(1, n - 1)))


@dataclass(frozen=True)
class InstanceSpec:
    """Everything that determines one benchmark instance (hashable, JSON-able)."""

    family: str
    n: int
    k: int
    mode: str
    seed: int
    variant: tuple[tuple[str, Any], ...] = (("type", "plain"),)
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def variant_dict(self) -> dict[str, Any]:
        return _thaw(self.variant)

    @property
    def params_dict(self) -> dict[str, Any]:
        return _thaw(self.params)

    @property
    def is_fixed(self) -> bool:
        return self.family in FIXED_FAMILIES

    @property
    def name(self) -> str:
        """Deterministic file/instance name."""
        v = self.variant_dict
        if self.family == "counterexample":
            base = f"counterexample_c{int(self.params_dict.get('copies', 17))}"
        elif self.family.startswith("paper_"):
            base = self.family
        else:
            base = f"{self.family}_n{self.n}_k{self.k}_{self.mode}_s{self.seed}"
            extra = {key: val for key, val in self.params_dict.items() if key not in ("generator", "params")}
            gp = self.params_dict.get("params") or {}
            for key in sorted(gp):
                extra[key] = gp[key]
            if extra:
                tag = "_".join(f"{key}{_slug(val)}" for key, val in sorted(extra.items()))
                base += f"_{tag}"
        if v.get("type") == "directed":
            base += f"_dir{_slug(v.get('drop_fraction', 0.5))}"
        elif v.get("type") == "weighted":
            base += f"_w{int(v.get('w_max', 4))}s{int(v.get('slack', 0))}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "n": self.n, "k": self.k, "mode": self.mode, "seed": self.seed,
                "variant": self.variant_dict, "params": self.params_dict, "name": self.name}


def _slug(val: Any) -> str:
    s = str(val).replace(".", "p").replace("-", "m").replace("*", "x").replace(" ", "")
    return "".join(ch for ch in s if ch.isalnum() or ch in "p_")


def _freeze(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Hashable, order-independent form of a (nested) JSON dict: sorted ``(key, value)`` pairs."""
    out = []
    for key in sorted(d):
        val = d[key]
        if isinstance(val, dict):
            val = _freeze(val)
        elif isinstance(val, list):
            val = tuple(val)
        out.append((key, val))
    return tuple(out)


def _is_frozen_dict(val: Any) -> bool:
    return isinstance(val, tuple) and all(
        isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in val
    ) and len(val) > 0


def _thaw(pairs: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Inverse of :func:`_freeze` (nested frozen dicts become dicts again, tuples stay tuples)."""
    out: dict[str, Any] = {}
    for key, val in pairs:
        out[key] = _thaw(val) if _is_frozen_dict(val) else ({} if val == () else val)
    return out


def expand_grid(cfg: BenchConfig) -> list[InstanceSpec]:
    """The Cartesian grid of the config, de-duplicated by instance name, in a stable order."""
    specs: list[InstanceSpec] = []
    seen: set[str] = set()
    for fam in cfg.families:
        name = str(fam["name"])
        params = {key: val for key, val in fam.items() if key != "name" and key not in _FAMILY_OVERRIDES}
        fparams = _freeze(params)
        if name in FIXED_FAMILIES:
            spec = InstanceSpec(name, 0, 0, "fixed", 0, (("type", "plain"),), fparams)
            if spec.name not in seen:
                seen.add(spec.name)
                specs.append(spec)
            continue
        # a family entry may override the grid lists (e.g. huge sizes only for the DAG family)
        sizes = [int(x) for x in fam.get("sizes", cfg.sizes)]
        ks = _normalize_ks(fam["k"]) if "k" in fam else cfg.ks
        modes = [str(m) for m in fam.get("modes", cfg.modes)]
        seeds = [int(s) for s in fam.get("seeds", cfg.seeds)]
        variants = [_normalize_variant(v) for v in fam["variants"]] if "variants" in fam else cfg.variants
        for n in sizes:
            for kspec in ks:
                k = resolve_k(kspec, n)
                for mode in modes:
                    for seed in seeds:
                        for var in variants:
                            spec = InstanceSpec(name, n, k, mode, seed + cfg.seed_offset,
                                                _freeze(var), fparams)
                            if spec.name in seen:
                                continue
                            seen.add(spec.name)
                            specs.append(spec)
    return specs


def build_instance(spec: InstanceSpec) -> Any:
    """Generate the :class:`glsolver.instance.Instance` of ``spec`` deterministically.

    ``ValueError`` from a generator (e.g. ``cycle`` with ``k > 2``) propagates
    to the caller, which records the pair as skipped.
    """
    import random

    from glsolver import generators as G
    from glsolver.instance import make_instance

    params = spec.params_dict
    if spec.family == "counterexample":
        from glref.counterexample import build_counterexample_instance

        inst = build_counterexample_instance(int(params.get("copies", 17)))
        meta = {**inst.meta, "family": "counterexample", "copies": int(params.get("copies", 17)),
                "seed": 0, "mode": "fixed"}
        inst = make_instance(inst.n, inst.arcs, inst.terminals, inst.capacities, weights=inst.weights,
                             directed=True, name=spec.name, meta=meta)
    elif spec.family.startswith("paper_"):
        inst = G.family_catalog()[spec.family](0, 0, 0)
        inst = make_instance(inst.n, inst.arcs if inst.directed else (inst.undirected_edges or ()),
                             inst.terminals, inst.capacities, weights=inst.weights,
                             directed=inst.directed, name=spec.name,
                             meta={**inst.meta, "family": spec.family, "seed": 0, "mode": "fixed"})
    elif "generator" in params:
        import inspect

        fn = getattr(G, str(params["generator"]), None)
        if fn is None or not callable(fn):
            raise ValueError(f"unknown generator function {params['generator']!r}")
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        for key, val in (("n", spec.n), ("k", spec.k), ("seed", spec.seed), ("mode", spec.mode)):
            if key in sig.parameters:
                kwargs[key] = val
        for key, val in (params.get("params") or {}).items():
            kwargs[key] = eval_param(val, spec.n, spec.k)
        inst = fn(**kwargs)
        inst = make_instance(inst.n, inst.arcs if inst.directed else (inst.undirected_edges or ()),
                             inst.terminals, inst.capacities, weights=inst.weights,
                             directed=inst.directed, name=spec.name,
                             meta={**inst.meta, "bench_family": spec.family,
                                   "generator": params["generator"], "seed": spec.seed})
    else:
        catalog = G.family_catalog()
        if spec.family not in catalog:
            raise ValueError(f"unknown family {spec.family!r}; known: {sorted(catalog)}")
        inst = catalog[spec.family](spec.n, spec.k, spec.seed)
        if spec.mode != "balanced" and inst.weights is None:
            caps = G.capacity_vector(inst.num_nonterminals, inst.k, random.Random(spec.seed), spec.mode)
            edges = inst.arcs if inst.directed else (inst.undirected_edges or ())
            inst = make_instance(inst.n, edges, inst.terminals, caps, weights=inst.weights,
                                 directed=inst.directed, name=inst.name, meta={**inst.meta, "mode": spec.mode})
        elif spec.mode != "balanced":
            raise ValueError(f"capacity mode {spec.mode!r} is only applied to unweighted base instances")
    var = spec.variant_dict
    if var.get("type") == "directed":
        if inst.directed:
            raise ValueError("directed variant needs an undirected base family")
        inst = G.directed_variant(inst, spec.seed, float(var.get("drop_fraction", 0.5)))
    elif var.get("type") == "weighted":
        inst = G.weighted_variant(inst, spec.seed, int(var.get("w_max", 4)), slack=int(var.get("slack", 0)))
    meta = {**inst.meta, "bench_spec": spec.to_dict()}
    meta.setdefault("family", spec.family)
    if not inst.directed and inst.n <= EXACT_CONNECTIVITY_MAX_N:
        try:
            from glsolver.preconditions import undirected_vertex_connectivity

            meta["vertex_connectivity"] = int(undirected_vertex_connectivity(inst))
        except Exception:  # pragma: no cover - never fail a benchmark on a diagnostic
            pass
    edges = inst.arcs if inst.directed else (inst.undirected_edges or ())
    return make_instance(inst.n, edges, inst.terminals, inst.capacities, weights=inst.weights,
                         directed=inst.directed, name=spec.name, meta=meta)


@functools.lru_cache(maxsize=1)
def generator_fingerprint() -> str:
    """SHA-256 over the ``glsolver`` version and the sources that determine generated instances.

    Stored in every instance sidecar and row; a cached instance whose
    fingerprint differs from the installed code is regenerated (so editing a
    generator can never silently feed stale instances to a new run).
    """
    import importlib

    h = hashlib.sha256()
    try:
        from glsolver import __version__ as version
    except Exception:  # pragma: no cover
        version = "?"
    h.update(f"glsolver {version}\n".encode())
    for mod_name in _GENERATOR_SOURCES:
        try:
            mod = importlib.import_module(mod_name)
            src = Path(mod.__file__).read_bytes()  # type: ignore[arg-type]
        except Exception:
            src = b"<unavailable>"
        h.update(f"{mod_name}\n".encode())
        h.update(hashlib.sha256(src).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes (streamed; instance files can be tens of MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def instance_info(inst: Any, spec: InstanceSpec, path: Path, gen_time: float) -> dict[str, Any]:
    """The sidecar summary of a cached instance (what the parent process needs).

    Includes ``sha256`` of the instance file at ``path`` (which must already be
    written), the ``generator_fingerprint`` and the ``glsolver_version`` that
    produced it.
    """
    from glsolver import __version__ as glsolver_version
    from glsolver.preconditions import is_dag, is_k_T_connected_dag

    n, m, k = inst.n, inst.m, inst.k
    m_edges = len(inst.undirected_edges) if inst.undirected_edges is not None else None
    if inst.directed:
        density = m / (n * (n - 1)) if n > 1 else 0.0
    else:
        density = (2.0 * (m_edges or 0)) / (n * (n - 1)) if n > 1 else 0.0
    dag = is_dag(inst)
    dag_kt = bool(dag and is_k_T_connected_dag(inst)[0])
    meta = dict(inst.meta)
    connectivity: dict[str, Any] = {}
    for key in ("kappa", "vertex_connectivity", "claims_k_connected", "claims_kT_connected",
                "connectivity_verified", "d", "drop_fraction", "dropped"):
        if key in meta:
            connectivity[key] = meta[key]
    return {
        "name": spec.name,
        "family": spec.family,
        "base_family": meta.get("base_family") or meta.get("family"),
        "variant": spec.variant_dict,
        "mode": spec.mode,
        "seed": spec.seed,
        "spec": spec.to_dict(),
        "n": n, "m": m, "m_edges": m_edges, "k": k, "density": density,
        "directed": bool(inst.directed), "weighted": bool(inst.is_weighted),
        "is_dag": bool(dag), "dag_kT_connected": dag_kt,
        "terminals": list(inst.terminals) if n <= 10_000 else None,
        "capacities": list(inst.capacities),
        "total_weight": int(inst.total_weight), "w_max": int(inst.w_max),
        "connectivity": connectivity,
        "meta": {key: val for key, val in meta.items() if key != "bench_spec"},
        "path": str(path),
        "generation_time": gen_time,
        "sha256": file_sha256(path) if path.exists() else None,
        "generator_fingerprint": generator_fingerprint(),
        "glsolver_version": str(glsolver_version),
    }


def cache_entry_status(path: Path, info_path: Path, verify_digest: bool = True) -> tuple[dict[str, Any] | None, str]:
    """``(info, "")`` when the cached instance at ``path`` is usable, else ``(None, reason)``.

    Usable means: both files exist, the sidecar parses, its
    ``generator_fingerprint`` equals :func:`generator_fingerprint` of the
    installed code and (when ``verify_digest``) the file's SHA-256 equals the
    sidecar's ``sha256``.
    """
    if not path.exists() or not info_path.exists():
        return None, "not cached"
    try:
        with open(info_path, encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable sidecar ({type(exc).__name__})"
    if not isinstance(info, dict) or not info.get("sha256") or not info.get("generator_fingerprint"):
        return None, "sidecar has no digest (written by an older runner)"
    if info["generator_fingerprint"] != generator_fingerprint():
        return None, "generator sources or glsolver version changed"
    if verify_digest and file_sha256(path) != info["sha256"]:
        return None, "instance file does not match its recorded sha256"
    return info, ""


def get_instance(spec: InstanceSpec, instances_dir: Path = INSTANCES_DIR,
                 log: Log = _log_stderr, verify_digest: bool = True) -> dict[str, Any]:
    """Return the sidecar info of ``spec``, generating and caching the instance if needed.

    A cache hit requires the sidecar's generator fingerprint to match the
    installed code and (``verify_digest``) the file to match its recorded
    SHA-256; a stale or corrupt entry is regenerated in place (deterministic,
    so the bytes are identical unless the generator really changed) and the
    reason is logged.  An untouched hit never rewrites the files.
    """
    instances_dir.mkdir(parents=True, exist_ok=True)
    path = instances_dir / f"{spec.name}.json"
    info_path = instances_dir / f"{spec.name}.info.json"
    info, why = cache_entry_status(path, info_path, verify_digest)
    if info is not None:
        info["path"] = str(path)
        return info
    if why != "not cached":
        log(f"  regenerating {spec.name}: {why}")
    from glsolver.io import save_instance

    t0 = time.perf_counter()
    inst = build_instance(spec)
    gen_time = time.perf_counter() - t0
    save_instance(inst, path)
    info = instance_info(inst, spec, path, gen_time)
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    log(f"  generated {spec.name}: n={inst.n} m={inst.m} k={inst.k} ({gen_time:.2f}s)")
    return info


# ---------------------------------------------------------------------------
# algorithm applicability
# ---------------------------------------------------------------------------
def algorithm_applies(algo: str, info: dict[str, Any], limits: dict[str, Any] | None = None,
                      core_ok: bool | None = None) -> tuple[bool, str]:
    """Whether ``algo`` can run on the instance described by ``info`` (and why not)."""
    lim = dict(limits or {})
    n, k = int(info["n"]), int(info["k"])
    if "max_n" in lim and n > int(lim["max_n"]):
        return False, f"n={n} > max_n={lim['max_n']}"
    if "min_n" in lim and n < int(lim["min_n"]):
        return False, f"n={n} < min_n={lim['min_n']}"
    if "max_k" in lim and k > int(lim["max_k"]):
        return False, f"k={k} > max_k={lim['max_k']}"
    if "families" in lim and info.get("family") not in lim["families"]:
        return False, f"family {info.get('family')!r} not in {lim['families']}"
    if algo in ("reference", "general") and info.get("weighted"):
        return False, "unweighted algorithm on a weighted instance"
    if algo in ("reference-dag", "dag") and not info.get("dag_kT_connected"):
        return False, "instance is not a k-T-connected DAG"
    if algo in CORE_ALGORITHMS:
        if core_ok is None:  # probed out of process: never import glsolver._core here
            core_ok = bool(core_info().get("available"))
        if not core_ok:
            return False, "C++ core not built"
    if algo == "ilp":
        try:
            import scipy.optimize  # noqa: F401
        except Exception:
            return False, "scipy not installed"
    return True, ""


def algorithm_runs(cfg: BenchConfig, algo: str) -> list[tuple[str, dict[str, Any]]]:
    """``[(label, options), ...]`` for ``algo``: one per configured option set (or a single default)."""
    sets = cfg.algorithm_options.get(algo)
    if not sets:
        return [(algo, {})]
    out = []
    for s in sets:
        opts = {key: val for key, val in s.items() if key != "label"}
        label = s.get("label") or ",".join(f"{key}={val}" for key, val in sorted(opts.items()))
        out.append((f"{algo}:{label}" if label else algo, opts))
    return out


# ---------------------------------------------------------------------------
# worker (runs in the subprocess)
# ---------------------------------------------------------------------------
def _rss_mb(ru: resource.struct_rusage) -> float:
    """Peak resident set of the worker, in MiB.

    ``ru_maxrss`` is inherited across ``fork``, so a worker spawned from a
    driver that has just generated a 10^7-arc instance reports the *driver's*
    footprint as its own (measured: a child of a 3 GB parent reports 2 879 MB
    when its true peak is 18 MB; this silently clamped the memory column of
    earlier sweeps to the driver's high-water mark).  Linux reports the honest
    figure as ``VmHWM`` in ``/proc/self/status``; ``ru_maxrss`` is the fallback.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0


def thin_series(seq: Sequence[Any], limit: int = MAX_SIZE_SAMPLES) -> list[Any]:
    """Uniformly subsample ``seq`` to at most ``limit`` entries keeping first and last."""
    seq = list(seq)
    if len(seq) <= limit:
        return seq
    if limit <= 1:
        return seq[:1]
    step = (len(seq) - 1) / (limit - 1)
    return [seq[int(round(i * step))] for i in range(limit)]


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly copy of ``result.stats`` with long series thinned."""
    out: dict[str, Any] = {}
    for key, val in (stats or {}).items():
        if key == "graph_size_over_time":
            val = [list(x) if isinstance(x, (list, tuple)) else x for x in val]
            out[key] = thin_series(val)
            out["graph_size_samples_total"] = len(val)
        elif isinstance(val, (list, tuple)):
            out[key] = thin_series(list(val))
        elif isinstance(val, (int, float, str, bool)) or val is None:
            out[key] = val
        elif isinstance(val, dict):
            out[key] = {str(a): b for a, b in val.items() if isinstance(b, (int, float, str, bool))}
        else:
            out[key] = str(val)
    return out


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.flush()


_PRELOAD = {
    "reference": ("glref.unweighted",), "reference-weighted": ("glref.weighted",),
    "reference-dag": ("glref.dag",), "bruteforce": ("glsolver.oracle.bruteforce",),
    "ilp": ("glsolver.oracle.ilp", "scipy.optimize"),
    "general": ("glsolver._core",), "weighted": ("glsolver._core",), "dag": ("glsolver._core",),
    "auto": ("glsolver.preconditions", "glref.unweighted", "glref.weighted", "glref.dag"),
}


def preload_backend(algorithm: str) -> None:
    """Import the modules a backend needs so lazy imports never land inside a timed trial."""
    import importlib

    for mod in _PRELOAD.get(algorithm, ()):
        try:
            importlib.import_module(mod)
        except Exception:
            pass


def worker_main(args: argparse.Namespace) -> int:
    """Body of the benchmark subprocess: load, warm up, time ``repeats`` trials, verify each."""
    from glsolver.api import glpartition
    from glsolver.io import load_instance
    from glsolver.verify import verify_instance_parts

    _emit({"event": "start", "pid": os.getpid(), "algorithm": args.algorithm,
           "env": {key: os.environ.get(key) for key in PROVENANCE_ENV}})
    preload_backend(args.algorithm)
    t0 = time.perf_counter()
    inst = load_instance(args.instance)
    load_time = time.perf_counter() - t0
    _emit({"event": "loaded", "load_time": load_time, "n": inst.n, "m": inst.m, "k": inst.k})
    opts = json.loads(args.options) if args.options else {}
    store_parts = inst.n <= args.store_parts_max_n
    kw: dict[str, Any] = dict(
        algorithm=args.algorithm, verify=False, verify_preconditions=False, seed=args.seed,
        threads=args.threads, options=dict(opts),
    )
    if args.time_limit and args.time_limit > 0:
        kw["time_limit"] = float(args.time_limit)
    for i in range(args.warmup):
        res = glpartition(inst, **kw)
        _emit({"event": "warmup", "trial": i, "status": res.status, "wall_time": res.runtime})
    for i in range(args.repeats):
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        res = glpartition(inst, **kw)
        wall = time.perf_counter() - t0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
        trial: dict[str, Any] = {
            "event": "trial", "trial": i, "wall_time": wall, "solver_runtime": float(res.runtime),
            "cpu_time": cpu, "status": res.status, "algorithm_run": res.algorithm,
            "message": (res.message or "")[:500], "valid": None, "verify_time": None,
            "verify_errors": [], "part_sizes": None, "part_weights": None,
            "stats": compact_stats(res.stats), "peak_rss_mb": _rss_mb(ru1),
            "certificate_parents": len((res.certificate or {}).get("parents", {}) or {}),
        }
        if res.status == "ok":
            t1 = time.perf_counter()
            rep = verify_instance_parts(inst, res.parts)
            trial["verify_time"] = time.perf_counter() - t1
            trial["valid"] = bool(rep.valid)
            trial["verify_errors"] = list(rep.errors[:5])
            trial["part_sizes"] = [len(p) for p in res.parts]
            if inst.is_weighted:
                trial["part_weights"] = rep.details.get("part_weights")
            if store_parts and i == 0:
                trial["parts"] = [[int(v) for v in p] for p in res.parts]
        _emit(trial)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    _emit({"event": "done", "load_time": load_time, "peak_rss_mb": _rss_mb(ru),
           "cpu_time_total": ru.ru_utime + ru.ru_stime})
    return 0


# ---------------------------------------------------------------------------
# parent-side subprocess control
# ---------------------------------------------------------------------------
@dataclass
class WorkerOutcome:
    """What the parent learned from one worker subprocess."""

    events: list[dict[str, Any]]
    returncode: int | None
    timed_out: bool
    wall_total: float
    cpu_user: float
    cpu_sys: float
    peak_rss_mb: float
    stderr_tail: str
    error: str | None = None
    stderr_head: str = ""

    @property
    def stderr_text(self) -> str:
        """Head and tail of the captured stderr (joined; identical when the output was short)."""
        if not self.stderr_head or self.stderr_tail.endswith(self.stderr_head):
            return self.stderr_tail
        return self.stderr_head + "\n...\n" + self.stderr_tail


def run_subprocess(cmd: Sequence[str], timeout: float, env: dict[str, str] | None = None,
                   grace: float | None = None, cwd: Path | None = None) -> WorkerOutcome:
    """Run ``cmd`` in its own session, stream its stdout JSON lines, enforce an inactivity timeout.

    The timeout is *per phase*: the deadline restarts whenever the child prints
    a line, so ``timeout`` bounds each trial (and the load phase) rather than
    the whole process.  The whole process group is killed once the child has
    been silent for ``timeout + grace`` seconds (``grace`` defaults to
    ``max(2, 0.1 * timeout)``; see :func:`effective_grace`).  The child is
    reaped with :func:`os.wait4`, which yields its CPU times and peak RSS even
    after a crash or kill.  The first and last few KB of stderr are kept.
    """
    grace = max(2.0, 0.1 * timeout) if grace is None else max(0.0, float(grace))
    stderr_path = None
    try:
        import tempfile

        stderr_file = tempfile.NamedTemporaryFile("w+", prefix="glbench_", suffix=".err", delete=False)
        stderr_path = Path(stderr_file.name)
    except OSError:  # pragma: no cover
        stderr_file = subprocess.DEVNULL  # type: ignore[assignment]
    t_start = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1,
            start_new_session=True, env=env, cwd=str(cwd) if cwd else None,
        )
    except OSError as exc:
        return WorkerOutcome([], None, False, 0.0, 0.0, 0.0, 0.0, "", error=f"spawn failed: {exc}")
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    last_activity = [time.monotonic()]

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                ev = {"event": "text", "text": line[:2000]}
            with lock:
                events.append(ev)
                last_activity[0] = time.monotonic()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    timed_out = False
    rusage = None
    status = None
    while True:
        try:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:  # pragma: no cover - already reaped
            pid, status, rusage = proc.pid, 0, None
        if pid != 0:
            break
        with lock:
            idle = time.monotonic() - last_activity[0]
        if idle > timeout + grace:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _, status, rusage = os.wait4(proc.pid, 0)
            except ChildProcessError:  # pragma: no cover
                status, rusage = 0, None
            break
        time.sleep(0.01 if time.monotonic() - t_start < 2 else 0.05)
    wall_total = time.monotonic() - t_start
    th.join(timeout=5.0)
    try:
        proc.returncode = os.waitstatus_to_exitcode(status) if status is not None else None
    except ValueError:  # pragma: no cover
        proc.returncode = -1
    if proc.stdout is not None:
        proc.stdout.close()
    head = tail = ""
    if stderr_path is not None:
        try:
            stderr_file.close()
            text = stderr_path.read_text(encoding="utf-8", errors="replace")
            head, tail = text[:4000], text[-4000:]
        except OSError:  # pragma: no cover
            head = tail = ""
        finally:
            try:
                stderr_path.unlink()
            except OSError:
                pass
    cpu_user = float(rusage.ru_utime) if rusage is not None else 0.0
    cpu_sys = float(rusage.ru_stime) if rusage is not None else 0.0
    rss = _rss_mb(rusage) if rusage is not None else 0.0
    return WorkerOutcome(events, proc.returncode, timed_out, wall_total, cpu_user, cpu_sys, rss, tail,
                         stderr_head=head)


def worker_command(instance_path: Path, algo: str, options: dict[str, Any], cfg: BenchConfig,
                   repeats: int, threads: int, seed: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--instance", str(instance_path), "--algorithm", algo, "--options", json.dumps(options),
        "--repeats", str(repeats), "--warmup", str(cfg.warmup), "--threads", str(threads),
        "--time-limit", str(cfg.timeout), "--store-parts-max-n", str(cfg.store_parts_max_n),
        "--seed", str(seed),
    ]


def worker_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    if threads > 0:
        for key in _THREAD_ENV:
            env[key] = str(threads)
    return env


# ---------------------------------------------------------------------------
# row assembly
# ---------------------------------------------------------------------------
def _median(vals: Sequence[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _quartiles(vals: Sequence[float]) -> tuple[float | None, float | None]:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], vals[0]
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return q[0], q[2]


def _load_average() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):  # pragma: no cover - not POSIX
        return None


def make_row(cfg: BenchConfig, info: dict[str, Any], algo: str, label: str, options: dict[str, Any],
             outcome: WorkerOutcome, repeats: int, threads: int, machine: dict[str, Any],
             git_commit: str | None, version: str | None, *,
             load_average: list[float] | None = None, grace: float | None = None) -> dict[str, Any]:
    """Combine the worker's events with the parent's measurements into one JSONL row.

    ``load_average`` is the 1/5/15-minute load sampled by the driver just
    before the worker started (collected here when omitted); ``grace`` is the
    kill grace period that applied (:func:`effective_grace` when omitted).
    The worker's effective thread environment is taken from its ``start``
    event (``child.env``).
    """
    trials = [dict(ev) for ev in outcome.events if ev.get("event") == "trial"]
    error_events = [ev for ev in outcome.events if ev.get("event") == "error"]
    start_event = next((ev for ev in outcome.events if ev.get("event") == "start"), {})
    load_time = next((ev.get("load_time") for ev in outcome.events if ev.get("event") in ("loaded", "done")), None)
    grace = effective_grace(cfg) if grace is None else float(grace)
    for t in trials:
        t.pop("event", None)
    parts = None
    for t in trials:
        if "parts" in t:
            parts = t.pop("parts")
    error: str | None = outcome.error
    if outcome.timed_out:
        trials.append({"trial": len(trials), "status": "timeout", "wall_time": None, "solver_runtime": None,
                       "cpu_time": None, "valid": None, "verify_time": None, "timed_out": True,
                       "message": f"no progress for {cfg.timeout:g}s (+{grace:g}s grace); process group killed",
                       "stats": {}, "peak_rss_mb": outcome.peak_rss_mb, "part_sizes": None})
    if error_events:
        error = str(error_events[-1].get("error"))
    elif outcome.returncode not in (0, None) and not outcome.timed_out:
        error = f"worker exited with code {outcome.returncode}"
        headline = summarize_stderr(outcome.stderr_text)
        if headline:
            error += ": " + headline
    env = start_event.get("env")
    if not isinstance(env, dict):  # worker never reported: what the driver exported
        exported = worker_env(threads)
        env = {key: exported.get(key) for key in PROVENANCE_ENV}
    if error and not trials:
        trials.append({"trial": 0, "status": "error", "wall_time": None, "solver_runtime": None,
                       "cpu_time": None, "valid": None, "verify_time": None, "message": error,
                       "stats": {}, "peak_rss_mb": outcome.peak_rss_mb, "part_sizes": None})
    statuses = [t.get("status") for t in trials]
    if trials and all(s == "ok" for s in statuses):
        status = "ok"
    else:
        status = next((s for s in statuses if s != "ok"), "error")
    valids = [t.get("valid") for t in trials]
    if status == "ok" and valids and all(v is True for v in valids):
        valid: bool | None = True
    elif any(v is False for v in valids):
        valid = False
    else:
        valid = None
    ok_trials = [t for t in trials if t.get("status") == "ok" and t.get("wall_time") is not None]
    walls = [t["wall_time"] for t in ok_trials]
    q1, q3 = _quartiles(walls)
    median = {
        "wall_time": _median(walls),
        "solver_runtime": _median([t.get("solver_runtime") for t in ok_trials]),
        "cpu_time": _median([t.get("cpu_time") for t in ok_trials]),
        "verify_time": _median([t.get("verify_time") for t in ok_trials]),
        "wall_q1": q1, "wall_q3": q3, "wall_min": min(walls) if walls else None,
        "wall_max": max(walls) if walls else None, "n_ok": len(ok_trials),
    }
    rep_trial: dict[str, Any] | None = None
    if ok_trials and median["wall_time"] is not None:
        rep_trial = min(ok_trials, key=lambda t: abs(t["wall_time"] - median["wall_time"]))
    elif trials:
        rep_trial = trials[-1]
    result = {
        "status": status,
        "valid": valid,
        "message": (rep_trial or {}).get("message", ""),
        "verify_errors": (rep_trial or {}).get("verify_errors", []),
        "part_sizes": (rep_trial or {}).get("part_sizes"),
        "part_weights": (rep_trial or {}).get("part_weights"),
        "stats": (rep_trial or {}).get("stats", {}),
        "algorithm_run": (rep_trial or {}).get("algorithm_run"),
        "parts": parts,
    }
    return {
        "schema": SCHEMA_VERSION,
        "config": cfg.name,
        "instance": {key: val for key, val in info.items() if key not in ("terminals",)},
        "n": int(info["n"]), "m": int(info["m"]), "k": int(info["k"]), "density": float(info["density"]),
        "algorithm": algo,
        "algorithm_label": label,
        "options": options,
        "threads": threads,
        "repeats": repeats,
        "warmup": cfg.warmup,
        "timeout": cfg.timeout,
        "grace": grace,
        "trials": trials,
        "median": median,
        "status": status,
        "valid": valid,
        "timed_out": bool(outcome.timed_out or status == "timeout"),
        "error": error,
        "result": result,
        "child": {
            "returncode": outcome.returncode, "wall_time_total": outcome.wall_total,
            "cpu_time_user": outcome.cpu_user, "cpu_time_sys": outcome.cpu_sys,
            "cpu_time": outcome.cpu_user + outcome.cpu_sys, "peak_rss_mb": outcome.peak_rss_mb,
            "load_time": load_time,
            "stderr_head": outcome.stderr_head[:STDERR_HEAD_CHARS] or outcome.stderr_tail[:STDERR_HEAD_CHARS],
            "stderr_tail": outcome.stderr_tail[-STDERR_TAIL_CHARS:],
            "env": env,
        },
        "git_commit": git_commit,
        "glsolver_version": version,
        "machine": machine,
        "load_average": _load_average() if load_average is None else list(load_average),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def _existing_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((str(row.get("instance", {}).get("name")), str(row.get("algorithm_label"))))
    return keys


def run_config(
    cfg: BenchConfig,
    output: Path | None = None,
    *,
    repeats: int | None = None,
    threads: int | None = None,
    limit: int | None = None,
    algorithms: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    instances_dir: Path | None = None,
    resume: bool = False,
    fresh: bool = False,
    dry_run: bool = False,
    log: Log = _log_stderr,
) -> dict[str, Any]:
    """Run every ``(instance, algorithm)`` pair of ``cfg`` and append rows to ``output``.

    Returns a summary dict (``rows``, ``ok``, ``invalid``, ``timeouts``,
    ``errors``, ``skipped``, ``output``).  ``limit`` caps the number of
    instances, ``algorithms``/``families`` restrict the grid, ``resume`` skips
    pairs already present in ``output`` and ``fresh`` truncates it first.
    """
    from glsolver import __version__ as glsolver_version

    repeats = cfg.repeats if repeats is None else max(1, int(repeats))
    threads = cfg.threads if threads is None else int(threads)
    instances_dir = INSTANCES_DIR if instances_dir is None else Path(instances_dir)
    output = RESULTS_DIR / f"{cfg.name}.jsonl" if output is None else Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fresh and output.exists():
        output.unlink()
    done = _existing_keys(output) if resume else set()
    algos = list(cfg.algorithms) if algorithms is None else [a for a in cfg.algorithms if a in set(algorithms)]
    if algorithms is not None:
        for a in algorithms:
            if a not in cfg.algorithms and a in KNOWN_ALGORITHMS:
                algos.append(a)
    specs = expand_grid(cfg)
    if families is not None:
        fams = set(families)
        specs = [s for s in specs if s.family in fams]
    if limit is not None:
        specs = specs[: max(0, int(limit))]
    machine = collect_machine_info()  # probes the C++ core out of process
    git_commit = machine.get("git", {}).get("commit")
    core = machine.get("core") or {}
    core_ok = bool(core.get("available"))
    core_error = core.get("error")
    core_skip_reason = f"C++ core not usable: {core_error}" if core_error else "C++ core not built"
    grace = effective_grace(cfg)
    summary = {"config": cfg.name, "output": str(output), "instances": len(specs), "rows": 0, "ok": 0,
               "invalid": 0, "timeouts": 0, "errors": 0, "skipped": 0, "precondition_failed": 0,
               "skipped_reasons": {}, "core_available": core_ok, "core_error": core_error}
    if not core_ok and any(a in CORE_ALGORITHMS for a in algos):
        log(f"note: {core_skip_reason}; 'general'/'weighted'/'dag' pairs are skipped")
    pairs: list[tuple[InstanceSpec, str, str, dict[str, Any]]] = []
    for spec in specs:
        for algo in algos:
            for label, opts in algorithm_runs(cfg, algo):
                pairs.append((spec, algo, label, opts))
    log(f"config {cfg.name}: {len(specs)} instances x {len(algos)} algorithms -> up to {len(pairs)} runs "
        f"(repeats={repeats}, warmup={cfg.warmup}, timeout={cfg.timeout}s, grace={grace:g}s, threads={threads})")
    info_cache: dict[str, dict[str, Any] | None] = {}
    for idx, (spec, algo, label, opts) in enumerate(pairs, 1):
        if dry_run:  # list the grid without generating (possibly huge) instances
            log(f"[{idx}/{len(pairs)}] would run {spec.name} with {label} (n={spec.n}, k={spec.k})")
            summary["rows"] += 1
            continue
        if spec.name not in info_cache:
            try:
                info_cache[spec.name] = get_instance(spec, instances_dir, log=log)
            except Exception as exc:
                info_cache[spec.name] = None
                log(f"  cannot generate {spec.name}: {type(exc).__name__}: {exc}")
                summary["skipped_reasons"][spec.name] = f"generation failed: {exc}"
        info = info_cache[spec.name]
        if info is None:
            summary["skipped"] += 1
            continue
        ok, why = algorithm_applies(algo, info, cfg.algorithm_limits.get(algo), core_ok)
        if not ok:
            if why == "C++ core not built":
                why = core_skip_reason
            summary["skipped"] += 1
            summary["skipped_reasons"].setdefault(why, 0)
            summary["skipped_reasons"][why] += 1
            continue
        if (spec.name, label) in done:
            summary["skipped"] += 1
            continue
        cmd = worker_command(Path(info["path"]), algo, opts, cfg, repeats, threads, spec.seed)
        load_before = _load_average()
        outcome = run_subprocess(cmd, cfg.timeout, env=worker_env(threads), grace=grace, cwd=REPO_ROOT)
        row = make_row(cfg, info, algo, label, opts, outcome, repeats, threads, machine, git_commit,
                       glsolver_version, load_average=load_before, grace=grace)
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        summary["rows"] += 1
        if row["status"] == "ok" and row["valid"] is True:
            summary["ok"] += 1
        if row["valid"] is False:
            summary["invalid"] += 1
        if row["timed_out"]:
            summary["timeouts"] += 1
        if row["status"] == "error":
            summary["errors"] += 1
        if row["status"] == "precondition_failed":
            summary["precondition_failed"] += 1
        med = row["median"]["wall_time"]
        verdict = {True: "VALID", False: "INVALID", None: "-"}[row["valid"]]
        med_text = f"{med:.4f}s" if med is not None else "n/a"
        log(f"[{idx}/{len(pairs)}] {spec.name} {label}: {row['status']} {verdict} "
            f"median={med_text} rss={row['child']['peak_rss_mb']:.0f}MB")
    log(f"done: {summary['rows']} rows -> {output} (ok={summary['ok']} invalid={summary['invalid']} "
        f"timeouts={summary['timeouts']} errors={summary['errors']} skipped={summary['skipped']})")
    return summary


def load_results(paths: Iterable[str | Path] | str | Path | None = None) -> list[dict[str, Any]]:
    """Read result rows from JSONL files (default: every ``benchmarks/results/*.jsonl``)."""
    if paths is None:
        files = sorted(RESULTS_DIR.glob("*.jsonl"))
    elif isinstance(paths, (str, Path)):
        p = Path(paths)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    else:
        files = [Path(p) for p in paths]
    rows: list[dict[str, Any]] = []
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "algorithm" in row:
                    row.setdefault("_source", str(f))
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def exit_code(summary: dict[str, Any]) -> int:
    """Exit status for a :func:`run_config` summary.

    ``EXIT_INVALID`` (3) when any partition was INVALID -- a correctness bug,
    which outranks everything else; ``EXIT_FAILED_RUNS`` (2) when any pair
    crashed (``error``) or timed out; ``EXIT_OK`` (0) otherwise (``ok``,
    ``precondition_failed`` and ``infeasible`` rows, and skipped pairs, are not
    failures).  ``scripts/run_benchmarks.sh`` refuses to continue past the
    smoke grid on any non-zero code.
    """
    if summary.get("invalid"):
        return EXIT_INVALID
    if summary.get("errors") or summary.get("timeouts"):
        return EXIT_FAILED_RUNS
    return EXIT_OK


def run_from_cli(args: Any) -> int:
    """Entry point for ``glsolve benchmark CONFIG [-o OUT] [--repeat R] [--threads T] [--limit L]``.

    Extra attributes (``algorithms``, ``families``, ``instances_dir``,
    ``resume``, ``fresh``, ``dry_run``) are honoured when present.  Returns
    :func:`exit_code` of the run: ``3`` for INVALID partitions, ``2`` for
    crashed or timed-out pairs, ``0`` otherwise.
    """
    cfg = load_config(args.config)
    algos = getattr(args, "algorithms", None)
    if isinstance(algos, str):
        algos = [a for a in algos.split(",") if a]
    fams = getattr(args, "families", None)
    if isinstance(fams, str):
        fams = [f for f in fams.split(",") if f]
    inst_dir = getattr(args, "instances_dir", None)
    summary = run_config(
        cfg,
        Path(args.output) if getattr(args, "output", None) else None,
        repeats=getattr(args, "repeat", None),
        threads=getattr(args, "threads", None),
        limit=getattr(args, "limit", None),
        algorithms=algos,
        families=fams,
        instances_dir=Path(inst_dir) if inst_dir else None,
        resume=bool(getattr(args, "resume", False)),
        fresh=bool(getattr(args, "fresh", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    return exit_code(summary)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmarks.runner", description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", help="config name (benchmarks/configs/<name>.json) or path")
    ap.add_argument("-o", "--output", default=None, help="JSONL output (default benchmarks/results/<config>.jsonl)")
    ap.add_argument("--repeat", type=int, default=None, help="override repeats")
    ap.add_argument("--threads", type=int, default=None, help="override thread count (0 = solver default)")
    ap.add_argument("--limit", type=int, default=None, help="max number of instances")
    ap.add_argument("--algorithms", default=None, help="comma-separated subset of algorithms")
    ap.add_argument("--families", default=None, help="comma-separated subset of families")
    ap.add_argument("--instances-dir", default=None, help="instance cache directory")
    ap.add_argument("--resume", action="store_true", help="skip pairs already in the output file")
    ap.add_argument("--fresh", action="store_true", help="truncate the output file first")
    ap.add_argument("--dry-run", action="store_true", help="list the runs without executing them")
    ap.add_argument("--info", action="store_true", help="print machine info and exit")
    # worker mode (internal)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--instance", help=argparse.SUPPRESS)
    ap.add_argument("--algorithm", help=argparse.SUPPRESS)
    ap.add_argument("--options", default="", help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--warmup", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--time-limit", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--store-parts-max-n", type=int, default=DEFAULT_STORE_PARTS_MAX_N, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        try:
            return worker_main(args)
        except Exception as exc:  # report, never hide
            _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-4000:]})
            return 1
    if args.info:
        print(json.dumps(collect_machine_info(), indent=1, sort_keys=True))
        return 0
    if not args.config:
        build_parser().print_help()
        return 2
    return run_from_cli(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
