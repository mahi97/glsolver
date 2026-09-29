"""One (instance, method) run, in its own process.

Run as ``python worker.py '<json>'`` where the JSON is
``{"spec": {...}, "method": "...", "budget": 300.0}``.  A single JSON object is
printed on the line after the marker ``@@RESULT@@``; everything a library
writes to stdout before that is ignored by the driver.

A subprocess is used for three reasons: a hard, external wall-clock kill;
memory blow-ups cannot take the driver down; and ``ortools`` and ``highspy``
cannot be imported into the same interpreter here (both ship a
``libhighs.so.1`` and OR-Tools' copy is the older one -- importing highspy
first makes ``import ortools`` fail with an undefined ``HighsLogOptions``
symbol).
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instances import build_instance, instance_id, target_sizes, undirected_view  # noqa: E402


def failure_modes(inst, parts) -> dict:
    """The three failure modes, measured independently of the verifier.

    * ``parts_disconnected`` -- parts whose vertices do not all reach the
      part's terminal inside the part (directed reachability on the instance's
      own arcs; for an undirected instance that is ordinary connectivity).
    * ``size_errors`` / ``size_l1`` -- parts of the wrong size, and the total
      absolute size deviation.
    * ``terminal_misplaced`` -- terminals that are not in the part they were
      matched to.
    """
    import networkx as nx

    n, k = inst.n, inst.k
    sizes = target_sizes(inst)
    terms = [int(t) for t in inst.terminals]

    cover_ok = sorted(v for p in parts for v in p) == list(range(n))

    misplaced = sum(1 for i in range(k) if terms[i] not in set(parts[i]))
    size_err = sum(1 for i in range(k) if len(parts[i]) != sizes[i])
    size_l1 = sum(abs(len(parts[i]) - sizes[i]) for i in range(k))

    # directed reachability to the terminal inside the part
    out_adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in inst.arcs:
        out_adj[int(u)].append(int(v))
    rev: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in out_adj[u]:
            rev[v].append(u)
    disconnected = 0
    for i in range(k):
        S = set(parts[i])
        t = terms[i]
        if t not in S:
            disconnected += 1
            continue
        seen = {t}
        stack = [t]
        while stack:
            x = stack.pop()
            for y in rev[x]:
                if y in S and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != len(S):
            disconnected += 1

    # undirected connectivity of the induced subgraph, for reporting on
    # undirected instances (what a partitioner would call "connected")
    G = undirected_view(inst)
    undirected_disconnected = 0
    for i in range(k):
        if not parts[i]:
            undirected_disconnected += 1
            continue
        sub = G.subgraph(parts[i])
        if not nx.is_connected(sub):
            undirected_disconnected += 1

    return {
        "exact_cover": cover_ok,
        # the number the comparison reports: undirected connectivity for an
        # undirected instance (what a partitioner is actually being judged on;
        # the directed count would double-charge a misplaced terminal, because
        # the undirected->directed reduction turns terminals into sinks) and
        # directed reachability for a genuinely directed instance.
        "parts_disconnected": undirected_disconnected if not inst.directed else disconnected,
        "parts_disconnected_directed": disconnected,
        "parts_disconnected_undirected": undirected_disconnected,
        "size_errors": size_err,
        "size_l1": size_l1,
        "terminal_misplaced": misplaced,
    }


def main() -> int:
    payload = json.loads(sys.argv[1])
    spec, method, budget = payload["spec"], payload["method"], float(payload["budget"])

    t_start = time.perf_counter()
    inst = build_instance(spec)
    gen_s = time.perf_counter() - t_start

    from methods import ALL_METHODS  # noqa: PLC0415

    fn = ALL_METHODS[method]
    t0 = time.perf_counter()
    try:
        out = fn(inst, budget)
    except Exception as e:  # noqa: BLE001
        import traceback

        out = {"status": "error", "parts": None, "build_s": 0.0,
               "solve_s": time.perf_counter() - t0,
               "extra": {"message": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-1500:]}}
    wall = time.perf_counter() - t0

    row = {
        "instance": instance_id(spec),
        "spec": spec,
        "method": method,
        "n": inst.n, "m": inst.m, "k": inst.k,
        "directed": bool(inst.directed),
        "weighted": bool(inst.is_weighted),
        "status": out["status"],
        "wall_s": wall,
        "build_s": out.get("build_s", 0.0),
        "solve_s": out.get("solve_s", 0.0),
        "gen_s": gen_s,
        "budget_s": budget,
        "over_budget": wall > budget,
        "answered": out["parts"] is not None,
        "extra": out.get("extra", {}),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }

    if out["parts"] is not None:
        from glsolver.verify import verify_instance_parts  # noqa: PLC0415

        rep = verify_instance_parts(inst, out["parts"])
        row["valid"] = bool(rep.valid)
        row["verifier_errors"] = rep.errors[:8]
        row["failure_modes"] = failure_modes(inst, out["parts"])
    else:
        row["valid"] = None
        row["verifier_errors"] = []
        row["failure_modes"] = None

    print("@@RESULT@@")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    raise SystemExit(main())
