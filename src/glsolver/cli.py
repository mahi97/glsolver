"""``glsolve`` command-line interface.

    glsolve solve INPUT --terminals 0,10,27,81 --sizes 25,25,25,25 [--algorithm auto] [--verify] [--stats] [-o solution.json]
    glsolve verify INSTANCE.json SOLUTION.json
    glsolve generate FAMILY --n 100 --k 4 --seed 1 [--mode balanced] [-o inst.json]
    glsolve inspect INSTANCE.json
    glsolve visualize INSTANCE.json [--algorithm reference] [--format svg|png|html|gif] [-o out]
    glsolve benchmark CONFIG.json [-o results.jsonl]

``INPUT`` is a ``.json`` instance (docs/instance_format.md) or an edge list;
for edge lists ``--terminals`` and ``--sizes``/``--capacities`` are required.
The first form of the brief, ``glsolve graph.edgelist --terminals ...``, is
accepted as a shorthand for ``solve``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from glsolver.api import ALL_ALGORITHMS, glpartition
from glsolver.instance import Instance
from glsolver.io import (
    load_edgelist,
    load_instance,
    load_solution,
    save_instance,
    save_solution,
)
from glsolver.verify import verify_instance_parts

SUBCOMMANDS = ("solve", "verify", "generate", "inspect", "visualize", "benchmark")


def _ints(s: str | None) -> list[int] | None:
    if s is None or s == "":
        return None
    return [int(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]


def _load_weights(spec: str | None, n: int) -> list[int] | None:
    if spec is None:
        return None
    p = Path(spec)
    if p.exists():
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            w = [int(data.get(str(v), data.get(v, 1))) for v in range(n)]
        else:
            w = [int(x) for x in data]
        return w
    return _ints(spec)


def load_input(path: str, args: argparse.Namespace) -> Instance:
    p = Path(path)
    if p.suffix == ".json":
        return load_instance(p)
    terminals = _ints(getattr(args, "terminals", None))
    if terminals is None:
        raise SystemExit("edge-list input needs --terminals")
    sizes = _ints(getattr(args, "sizes", None))
    caps = _ints(getattr(args, "capacities", None))
    n = getattr(args, "n", None)
    # peek n for weights
    weights = None
    wspec = getattr(args, "weights", None)
    if wspec is not None:
        if n is None:
            max_id = -1
            for line in p.read_text().splitlines():
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                max_id = max(max_id, int(parts[0]), int(parts[1]))
            n = max_id + 1
        weights = _load_weights(wspec, n)
    return load_edgelist(p, terminals, caps, sizes=sizes, directed=bool(getattr(args, "directed", False)),
                         n=n, weights=weights, name=p.stem)


def cmd_solve(args: argparse.Namespace) -> int:
    inst = load_input(args.input, args)
    pre = {"auto": "auto", "true": True, "false": False}[args.preconditions]
    res = glpartition(
        inst, algorithm=args.algorithm, verify=not args.no_verify, verify_preconditions=pre,
        trace=args.trace is not None, seed=args.seed, threads=args.threads, time_limit=args.time_limit,
        debug=args.debug,
    )
    print(res.summary())
    if res.message and res.status != "ok":
        print("message:", res.message)
    if res.status == "ok":
        for i, part in enumerate(res.parts):
            print(f"part {i} (terminal {inst.terminals[i]}, size {len(part)}): {' '.join(map(str, part))}")
    if args.stats:
        for key in sorted(res.stats):
            val = res.stats[key]
            if isinstance(val, list) and len(val) > 12:
                val = f"[{len(val)} entries]"
            print(f"  {key}: {val}")
    if res.preconditions:
        print("preconditions:", res.preconditions.get("message"))
    if args.output:
        save_solution(args.output, res.to_dict())
        print("solution written to", args.output)
    if args.trace is not None and res.trace is not None:
        Path(args.trace).write_text(json.dumps(res.trace, indent=1))
        print(f"trace with {len(res.trace)} events written to", args.trace)
    if res.status != "ok":
        return 2
    return 0 if res.valid in (True, None) else 3


def cmd_verify(args: argparse.Namespace) -> int:
    inst = load_input(args.instance, args)
    if args.parts:
        parts = [[int(x) for x in grp.split(",") if x.strip()] for grp in args.parts.split(";")]
    else:
        sol = load_solution(args.solution)
        parts = sol["parts"]
    rep = verify_instance_parts(inst, parts)
    print("VALID" if rep.valid else "INVALID")
    for e in rep.errors:
        print("  -", e)
    if args.details:
        print(json.dumps(rep.details, indent=1, default=str))
    return 0 if rep.valid else 1


def cmd_generate(args: argparse.Namespace) -> int:
    from glsolver import generators

    fam = args.family
    catalog = generators.family_catalog() if hasattr(generators, "family_catalog") else {}
    if fam in catalog:
        inst = catalog[fam](args.n, args.k, args.seed)
    elif hasattr(generators, fam):
        fn = getattr(generators, fam)
        kwargs: dict[str, Any] = {}
        import inspect as _inspect

        sig = _inspect.signature(fn)
        for key, val in (("n", args.n), ("k", args.k), ("seed", args.seed), ("mode", args.mode)):
            if key in sig.parameters:
                kwargs[key] = val
        for extra in args.param or []:
            key, _, val = extra.partition("=")
            kwargs[key] = json.loads(val)
        inst = fn(**kwargs)
    else:
        raise SystemExit(f"unknown family {fam!r}; known: {sorted(catalog)}")
    if args.weighted:
        inst = generators.weighted_variant(inst, args.seed, args.weighted, slack=args.slack)
    if args.mode != "balanced" and fam in catalog and hasattr(generators, "capacity_vector"):
        import random

        caps = generators.capacity_vector(inst.num_nonterminals, inst.k, random.Random(args.seed), args.mode)
        from glsolver.instance import make_instance

        edges = inst.arcs if inst.directed else inst.undirected_edges
        inst = make_instance(inst.n, edges, inst.terminals, caps, weights=inst.weights, directed=inst.directed,
                             name=inst.name, meta={**inst.meta, "mode": args.mode})
    out = args.output or f"{fam}_n{inst.n}_k{inst.k}_s{args.seed}.json"
    save_instance(inst, out)
    print(f"wrote {out}: n={inst.n} m={inst.m} k={inst.k} directed={inst.directed} weighted={inst.is_weighted}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    inst = load_input(args.instance, args)
    out_deg = [0] * inst.n
    for u, _ in inst.arcs:
        out_deg[u] += 1
    tset = set(inst.terminals)
    nt = [v for v in range(inst.n) if v not in tset]
    info: dict[str, Any] = {
        "name": inst.name, "n": inst.n, "m_arcs": inst.m, "k": inst.k, "directed": inst.directed,
        "weighted": inst.is_weighted, "terminals": list(inst.terminals), "capacities": list(inst.capacities),
        "sizes": list(inst.sizes) if not inst.is_weighted else None,
        "density_arcs": inst.m / max(1, inst.n * (inst.n - 1)),
        "min_out_degree_nonterminal": min((out_deg[v] for v in nt), default=0),
        "max_out_degree_nonterminal": max((out_deg[v] for v in nt), default=0),
        "meta": inst.meta,
    }
    from glsolver.preconditions import is_dag, is_k_T_connected_dag

    info["is_dag"] = is_dag(inst)
    if info["is_dag"]:
        ok, bad = is_k_T_connected_dag(inst)
        info["dag_k_T_connected"] = ok
        if not ok:
            info["dag_violating_vertex"] = bad
    if args.preconditions and inst.n <= 400:
        from glsolver.preconditions import check_preconditions

        pre = check_preconditions(inst, True)
        info["k_T_connected"] = pre["k_T_connected"]
        info["feac"] = pre["feac"]
        info["precondition_message"] = pre["message"]
        if not inst.directed:
            from glsolver.preconditions import undirected_vertex_connectivity

            info["vertex_connectivity"] = undirected_vertex_connectivity(inst)
    from glsolver.api import choose_algorithm

    info["auto_algorithm"] = choose_algorithm(inst)
    print(json.dumps(info, indent=1, default=str))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    try:
        from glsolver.viz import visualize_cli
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"visualization module not available: {exc}") from exc
    inst = load_input(args.instance, args)
    return visualize_cli(inst, args)


def cmd_benchmark(args: argparse.Namespace) -> int:
    try:
        from benchmarks.runner import run_from_cli
    except ImportError:
        try:
            from glsolver.benchmark import run_from_cli  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"benchmark runner not available: {exc}") from exc
    return run_from_cli(args)


def _add_input_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--terminals", help="comma-separated terminal ids (edge-list input)")
    p.add_argument("--sizes", help="comma-separated part sizes |V_i| (undirected/unweighted)")
    p.add_argument("--capacities", help="comma-separated capacities c_i (paper convention)")
    p.add_argument("--directed", action="store_true", help="treat edge-list input as directed arcs")
    p.add_argument("--weights", help="weights: JSON file (list or {vertex: w}) or comma-separated list")
    p.add_argument("--n", type=int, help="number of vertices for edge-list input (default: max id + 1)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="glsolve", description="Exact Győri–Lovász partitioning (arXiv 2608.30945)")
    sub = ap.add_subparsers(dest="command")

    s = sub.add_parser("solve", help="solve an instance")
    s.add_argument("input")
    _add_input_opts(s)
    s.add_argument("--algorithm", default="auto", choices=ALL_ALGORITHMS)
    s.add_argument("--verify", action="store_true", default=True, help="(default) run the independent verifier")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--preconditions", default="auto", choices=["auto", "true", "false"])
    s.add_argument("--stats", action="store_true")
    s.add_argument("--trace", metavar="TRACE.json", help="record the step-by-step trace")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--time-limit", type=float, default=None)
    s.add_argument("--debug", action="store_true", help="enable invariant assertions")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_solve)

    v = sub.add_parser("verify", help="verify a solution independently")
    v.add_argument("instance")
    v.add_argument("solution", nargs="?")
    v.add_argument("--parts", help='parts as "0,1,2;3,4;5,6" (alternative to a solution file)')
    v.add_argument("--details", action="store_true")
    _add_input_opts(v)
    v.set_defaults(func=cmd_verify)

    g = sub.add_parser("generate", help="generate an instance from a family")
    g.add_argument("family")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--mode", default="balanced")
    g.add_argument("--weighted", type=int, default=0, metavar="WMAX", help="make a weighted variant with weights in [1, WMAX]")
    g.add_argument("--slack", type=int, default=0)
    g.add_argument("--param", action="append", help="extra generator parameter key=json")
    g.add_argument("-o", "--output")
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("inspect", help="print instance statistics and precondition checks")
    i.add_argument("instance")
    i.add_argument("--preconditions", action="store_true", help="run NetworkX precondition checks (n <= 400)")
    _add_input_opts(i)
    i.set_defaults(func=cmd_inspect)

    z = sub.add_parser("visualize", help="render the algorithm's execution")
    z.add_argument("instance")
    z.add_argument("--algorithm", default=None,
                   help="trace-emitting backend (default: the instance's meta['viz_algorithm'], else 'reference')")
    z.add_argument("--format", default="svg", choices=["svg", "png", "pdf", "html", "gif", "mp4"])
    z.add_argument("--layout", default="auto")
    z.add_argument("--step", type=int, default=None, help="render only this event index")
    z.add_argument("-o", "--output", default=None)
    _add_input_opts(z)
    z.set_defaults(func=cmd_visualize)

    b = sub.add_parser("benchmark", help="run a benchmark configuration")
    b.add_argument("config")
    b.add_argument("-o", "--output", default=None)
    b.add_argument("--repeat", type=int, default=None)
    b.add_argument("--threads", type=int, default=None)
    b.add_argument("--limit", type=int, default=None, help="max instances (for quick runs)")
    b.set_defaults(func=cmd_benchmark)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["solve"] + argv  # shorthand form of the brief
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
