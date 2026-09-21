#!/usr/bin/env python
"""Offline exhaustive runner (docs/verification.md).

Enumerates every classical Győri–Lovász instance on the connected graphs of
the NetworkX atlas up to ``--max-n`` vertices (``<= 7``; every ``k`` in
``1..min(κ(G), --max-k)``, every terminal subset, every capacity
composition) and, unless ``--no-directed``, every digraph on
``--directed-min-n <= n <= --directed-max-n`` vertices with ``k`` in
``--directed-ks`` (default ``1,2``).  ``--directed-mask-sample M`` keeps a
seeded sample of ``M`` arc subsets per ``(n, k, terminals)`` block and
``--directed-min-capacity`` restricts the capacity compositions; the blocks
come from ``tests/strategies.directed_cases_for_block`` -- the very
generator the pytest harness uses, so the two cannot drift.

Every applicable backend (or only ``--solver`` ones) runs with
``debug=True`` (the paper's invariants for the reference solvers,
``debug_asserts`` for the C++ core) and must return a valid partition
whenever the theorem's precondition holds, and ``precondition_failed``
otherwise.  Every solver call is bounded by ``--time-limit`` seconds
(default ``$GL_SOLVER_TIME_LIMIT`` or 20): a hang is a ``timeout`` failure,
the job (atlas graph / digraph block) that hit it stops, and the run stops
dispatching work after ``--max-timeouts`` hangs.

Failures are stored under ``regression/`` (``save_regression``) and the JSON
tally goes to ``benchmarks/results/exhaustive_<timestamp>.json``.

    python scripts/run_exhaustive.py --max-n 6 --jobs 8
    python scripts/run_exhaustive.py --max-n 7 --solver reference --jobs 16
    python scripts/run_exhaustive.py --max-n 5 --oracles      # also bruteforce / ilp
    # the full n = 5, k = 3 digraph layer (15 360 instances; a zero-capacity
    # terminal is always removed first, so shifts happen among 2 terminals)
    python scripts/run_exhaustive.py --no-undirected --directed-min-n 5 --directed-max-n 5 --directed-ks 3
    # sampled n = 6, k = 3, c = (1,1,1) layer: length-3 reassignment cycles
    python scripts/run_exhaustive.py --no-undirected --directed-min-n 6 --directed-max-n 6 \\
        --directed-ks 3 --directed-min-capacity 1 --directed-mask-sample 512 --jobs 8

Work is split per atlas graph (undirected) and per ``(n, k, terminals)``
block (directed) over a ``multiprocessing`` pool.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))  # the enumerations live next to the tests

from glsolver.api import GLResult  # noqa: E402
from glsolver.instance import Instance  # noqa: E402
from glsolver.io import instance_from_dict, instance_to_dict  # noqa: E402
from glsolver.preconditions import check_preconditions, is_dag  # noqa: E402
from glsolver.testing.registry import available_solvers, solvers_for  # noqa: E402
from glsolver.testing.regression import save_regression  # noqa: E402
from glsolver.testing.timebound import (  # noqa: E402
    DEFAULT_TIME_LIMIT,
    TIME_LIMIT_ENV,
    SolverTimeout,
    default_time_limit,
    time_bound,
)
from glsolver.verify import verify_instance_parts  # noqa: E402
from strategies import (  # noqa: E402
    AtlasGraph,
    atlas_connected_graphs,
    count_directed_exhaustive,
    count_undirected_exhaustive,
    directed_blocks,
    directed_cases_for_block,
    instance_summary,
    undirected_cases_for_graph,
)

ORACLES = ("bruteforce", "ilp")
#: the oracles' own limit (they answer ``status == "timeout"`` themselves)
ORACLE_TIME_LIMIT = 60.0
RESULTS_DIR = ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------------
# per-instance checks
# ---------------------------------------------------------------------------


def _kw(name: str) -> dict[str, Any]:
    """Backend options: the oracles' own limit, ``debug=True`` for every theorem
    solver (reference: invariants A1–A8; core: ``debug_asserts``)."""
    if name in ORACLES:
        return {"verify_preconditions": False, "time_limit": ORACLE_TIME_LIMIT}
    return {"debug": True}


def _bound(name: str, time_limit: float) -> float | None:
    """Wall-clock bound of one call; the self-limiting oracles get their own limit on top."""
    if time_limit <= 0:
        return None
    return time_limit + (ORACLE_TIME_LIMIT if name in ORACLES else 0.0)


def _run(fn, inst: Instance, name: str, time_limit: float) -> tuple[GLResult | None, str | None, bool]:
    """``(result, crash message, hung)``: a crash or a hang is reported, never hidden."""
    try:
        with time_bound(_bound(name, time_limit), f"{name} on {inst.name}"):
            return fn(inst, **_kw(name)), None, False
    except SolverTimeout as exc:
        return None, f"timeout: {exc}", True
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", False


def _ok(res: GLResult | None, inst: Instance) -> bool:
    return (
        res is not None and res.status == "ok" and res.valid is True
        and verify_instance_parts(inst, res.parts).valid
        and (inst.weights is not None or [len(p) for p in res.parts] == [c + 1 for c in inst.capacities])
    )


def _failure(inst: Instance, name: str, res: GLResult | None, crash: str | None, expect: str,
             kind: str) -> dict[str, Any]:
    if crash is not None:
        message, errors = crash, []
    else:
        assert res is not None
        message = f"status={res.status} valid={res.valid}: {res.message}"
        errors = list(res.verification.errors) if res.verification is not None else []
    return {"solver": name, "expect": expect, "kind": kind, "message": message, "errors": errors[:20],
            "instance": instance_to_dict(inst), "summary": instance_summary(inst)}


def check_instance(inst: Instance, solver_names: list[str] | None, oracles: bool,
                   counts: dict[str, Any], failures: list[dict[str, Any]], time_limit: float) -> bool:
    """Run the applicable backends on one instance and update ``counts`` /
    ``failures``; returns ``True`` when a backend hung (the caller stops)."""
    counts["instances"] += 1
    if inst.directed:
        pre = check_preconditions(inst, True)
        holds = bool(pre["feac"])
        counts["feac" if holds else "non_feac"] += 1
        if pre["k_T_connected"]:
            counts["k_T_connected"] += 1
    else:
        holds = True  # k-connected undirected graph: the theorem applies
    if holds:
        backends = solvers_for(inst, include_oracles=oracles)
    else:
        all_ = available_solvers(include_oracles=oracles)
        names = ["reference", "reference-weighted", "general", "weighted"]
        if is_dag(inst):
            names += ["reference-dag", "dag"]
        backends = {n: all_[n] for n in names if n in all_}
        if oracles:
            backends.update({n: all_[n] for n in ORACLES if n in all_})
    for name, fn in backends.items():
        if solver_names is not None and name not in solver_names:
            continue
        res, crash, hung = _run(fn, inst, name, time_limit)
        counts["solver_runs"][name] = counts["solver_runs"].get(name, 0) + 1
        kind = "timeout" if hung else ("crash" if crash is not None else "invalid")
        if holds:
            if not _ok(res, inst):
                failures.append(_failure(inst, name, res, crash, "ok", kind))
        elif name in ORACLES:
            if crash is not None or res is None or res.status not in ("ok", "infeasible", "timeout"):
                failures.append(_failure(inst, name, res, crash, "ok|infeasible", kind))
            elif res.status == "ok":
                counts["non_feac_with_partition"] += 1
        elif crash is not None or res is None or res.status != "precondition_failed":
            failures.append(_failure(inst, name, res, crash, "precondition_failed", kind))
        if hung:
            counts["timeouts"] += 1
            return True
    return False


def _new_counts() -> dict[str, Any]:
    return {"instances": 0, "feac": 0, "non_feac": 0, "k_T_connected": 0,
            "non_feac_with_partition": 0, "timeouts": 0, "solver_runs": {}, "by_n": {}}


def _merge(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, val in part.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                total[key][k2] = total[key].get(k2, 0) + v2
        else:
            total[key] += val


# ---------------------------------------------------------------------------
# work items
# ---------------------------------------------------------------------------


def undirected_job(job: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """All instances of one atlas graph (``job["graph"]``); stops at the first hang."""
    graph: AtlasGraph = job["graph"]
    counts, failures = _new_counts(), []
    for inst in undirected_cases_for_graph(graph, max_k=job["max_k"], full=True):
        if check_instance(inst, job["solvers"], job["oracles"], counts, failures, job["time_limit"]):
            break  # every further instance of this graph could cost the full limit again
    counts["by_n"][str(graph.n)] = counts["instances"]
    return counts, failures


def directed_job(job: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """All instances of one ``(n, k, terminals)`` digraph block; stops at the first hang."""
    n, k, terminals = job["block"]
    counts, failures = _new_counts(), []
    for inst in directed_cases_for_block(
        n, k, terminals, mask_sample=job["mask_sample"], seed=job["seed"], min_capacity=job["min_capacity"]
    ):
        if check_instance(inst, job["solvers"], job["oracles"], counts, failures, job["time_limit"]):
            break
    counts["by_n"][str(n)] = counts["instances"]
    return counts, failures


def _imap(pool: Pool | None, fn, items: list[Any]):
    if pool is None:
        return map(fn, items)
    return pool.imap_unordered(fn, items, chunksize=1)


def _parse_ks(text: str) -> tuple[int, ...]:
    ks = tuple(int(x) for x in text.split(",") if x.strip())
    if not ks or any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("--directed-ks needs positive integers, e.g. 1,2")
    return ks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-n", type=int, default=7, help="largest undirected n (atlas: <= 7)")
    ap.add_argument("--max-k", type=int, default=3, help="largest k per graph (default 3)")
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--no-undirected", action="store_true", help="skip the undirected enumeration")
    ap.add_argument("--directed-max-n", type=int, default=4, help="largest n of the digraph enumeration")
    ap.add_argument("--directed-min-n", type=int, default=1, help="smallest n of the digraph enumeration")
    ap.add_argument("--directed-ks", type=_parse_ks, default=(1, 2), help="k values of the digraph enumeration (default 1,2)")
    ap.add_argument("--directed-mask-sample", type=int, default=None,
                    help="seeded sample of this many arc subsets per (n, k, terminals) block (default: all)")
    ap.add_argument("--directed-min-capacity", type=int, default=0,
                    help="only capacity compositions with every part >= this (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the digraph arc-subset sample")
    ap.add_argument("--no-directed", action="store_true", help="skip the digraph enumeration")
    ap.add_argument("--solver", action="append", default=None, help="restrict to these backends (repeatable)")
    ap.add_argument("--oracles", action="store_true", help="also run bruteforce / ilp on every instance")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--time-limit", type=float, default=default_time_limit(),
                    help=f"wall-clock bound per solver call in seconds, 0 disables (default ${TIME_LIMIT_ENV} or {DEFAULT_TIME_LIMIT:g})")
    ap.add_argument("--max-timeouts", type=int, default=3, help="stop dispatching work after this many hangs (default 3)")
    ap.add_argument("--out", type=Path, default=None, help="tally path (default benchmarks/results/exhaustive_<ts>.json)")
    ap.add_argument("--no-save-failures", action="store_true", help="do not store failures under regression/")
    args = ap.parse_args(argv)
    if args.max_n > 7:
        ap.error("--max-n must be <= 7 (NetworkX graph atlas)")

    available = available_solvers(include_oracles=True)
    if args.solver:
        unknown = [s for s in args.solver if s not in available]
        if unknown:
            ap.error(f"unknown/unavailable solvers {unknown}; available: {sorted(available)}")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or RESULTS_DIR / f"exhaustive_{stamp}.json"
    common = {"solvers": args.solver, "oracles": args.oracles, "time_limit": args.time_limit}

    graphs = [] if args.no_undirected else atlas_connected_graphs(args.max_n, min_n=args.min_n)
    und_items = [{"graph": g, "max_k": args.max_k, **common} for g in sorted(graphs, key=lambda g: -g.n)]
    directed_opts = {"mask_sample": args.directed_mask_sample, "min_capacity": args.directed_min_capacity}
    dir_items: list[dict[str, Any]] = []
    if not args.no_directed:
        dir_items = [
            {"block": block, "seed": args.seed, **directed_opts, **common}
            for block in directed_blocks(args.directed_max_n, args.directed_ks, min_n=args.directed_min_n)
        ]
    expected_und = {} if args.no_undirected else count_undirected_exhaustive(args.max_n, max_k=args.max_k, min_n=args.min_n)
    expected_dir = 0 if args.no_directed else count_directed_exhaustive(
        args.directed_max_n, args.directed_ks, min_n=args.directed_min_n, **directed_opts
    )
    print(f"undirected: {len(graphs)} atlas graphs, {sum(expected_und.values())} instances "
          f"{expected_und}; directed: {expected_dir} instances in {len(dir_items)} blocks; "
          f"jobs={args.jobs}; time limit {args.time_limit:g} s per call", flush=True)

    t0 = time.perf_counter()
    und_counts, dir_counts = _new_counts(), _new_counts()
    failures: list[dict[str, Any]] = []
    pool = Pool(args.jobs) if args.jobs > 1 else None
    aborted = False
    try:
        for label, job_fn, items, total in (("undirected", undirected_job, und_items, und_counts),
                                            ("directed", directed_job, dir_items, dir_counts)):
            if aborted or not items:
                continue
            done = 0
            for counts, fails in _imap(pool, job_fn, items):
                _merge(total, counts)
                failures.extend(fails)
                done += 1
                if label == "undirected" and (done % 50 == 0 or done == len(items)):
                    print(f"  undirected graphs {done}/{len(items)}: {total['instances']} instances, "
                          f"{len(failures)} failures, {time.perf_counter() - t0:.0f}s", flush=True)
                hangs = und_counts["timeouts"] + dir_counts["timeouts"]
                if args.max_timeouts > 0 and hangs >= args.max_timeouts:
                    print(f"  {hangs} hangs: stopping after {done}/{len(items)} {label} jobs", flush=True)
                    aborted = True
                    break
            if label == "directed":
                print(f"  directed: {total['instances']} instances, {len(failures)} failures total", flush=True)
    finally:
        if pool is not None:
            if aborted:
                pool.terminate()
            else:
                pool.close()
            pool.join()
    seconds = time.perf_counter() - t0

    saved: list[str] = []
    if failures and not args.no_save_failures:
        for i, f in enumerate(failures):
            inst = instance_from_dict(f["instance"])
            path = save_regression(
                inst, reason=f"run_exhaustive {stamp}: {f['solver']} expected {f['expect']}: {f['message']}",
                extra={"solver": f["solver"], "message": f["message"], "errors": f["errors"],
                       "kind": f["kind"], "expect": "ok" if f["expect"] == "ok" else "precondition_failed"},
            )
            saved.append(str(path))
            if i >= 199:  # keep the store manageable
                break

    hangs = und_counts["timeouts"] + dir_counts["timeouts"]
    if hangs == 0:  # a hang stops its job early, so the counts are complete only without one
        assert und_counts["instances"] == sum(expected_und.values()), (und_counts["instances"], expected_und)
        assert dir_counts["instances"] == expected_dir, (dir_counts["instances"], expected_dir)
    tally = {
        "timestamp": stamp, "seconds": round(seconds, 2), "jobs": args.jobs,
        "max_n": None if args.no_undirected else args.max_n, "max_k": args.max_k,
        "directed_max_n": None if args.no_directed else args.directed_max_n,
        "directed_min_n": None if args.no_directed else args.directed_min_n,
        "directed_ks": None if args.no_directed else list(args.directed_ks),
        "directed_mask_sample": args.directed_mask_sample, "directed_min_capacity": args.directed_min_capacity,
        "seed": args.seed, "time_limit": args.time_limit, "timeouts": hangs, "aborted": aborted,
        "solvers": args.solver or sorted(available), "oracles": args.oracles,
        "undirected": und_counts, "directed": dir_counts,
        "failures": [{k: v for k, v in f.items() if k != "instance"} for f in failures],
        "failures_saved": saved,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tally, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in tally.items() if k not in ("failures", "failures_saved")}, indent=1, sort_keys=True))
    print(f"failures: {len(failures)} ({hangs} hangs); tally written to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
