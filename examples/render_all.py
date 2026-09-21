#!/usr/bin/env python
"""Render every curated example (``examples/curated/*.json``) to
``examples/output/<name>/``: one SVG per trace event, the self-contained
HTML viewer, an animated GIF and the final partition as a standalone SVG.

    python examples/render_all.py                 # everything
    python examples/render_all.py --only small_dag --formats svg,html
    python examples/render_all.py --out /tmp/gl_examples --fps 2

Each instance carries ``meta["viz_algorithm"]`` (the trace-emitting backend
to run) and ``meta["expected"]`` (properties asserted here, e.g. the number
of cycle shifts); a summary table is printed and written to
``<out>/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from glsolver.io import load_instance
from glsolver.viz import (
    TraceRenderer,
    animate_solution,
    partition_svg,
    render_html,
    solve_with_trace,
)

HERE = Path(__file__).resolve().parent
CURATED = HERE / "curated"
OUTPUT = HERE / "output"
FORMATS = ("svg", "html", "gif", "png")


def check_expected(inst_meta: dict[str, Any], res: Any) -> list[str]:
    """Compare the solver's statistics with ``meta["expected"]``; return the violations."""
    exp = inst_meta.get("expected") or {}
    bad: list[str] = []
    if "events" in exp and len(res.trace) != exp["events"]:
        bad.append(f"events {len(res.trace)} != {exp['events']}")
    for key in ("cycle_shifts", "contractions", "deletions", "terminal_removals", "roundings"):
        if key in exp and res.stats.get(key) != exp[key]:
            bad.append(f"{key} {res.stats.get(key)} != {exp[key]}")
    if "min_cycle_shifts" in exp and res.stats.get("cycle_shifts", 0) < exp["min_cycle_shifts"]:
        bad.append("too few cycle shifts")
    if "min_roundings" in exp and res.stats.get("roundings", 0) < exp["min_roundings"]:
        bad.append("no rounding happened")
    if exp.get("dag") and not any(e["type"] == "dag_contract" for e in res.trace):
        bad.append("no dag_contract event")
    return bad


def render_example(path: Path, out_root: Path, formats: tuple[str, ...] = ("svg", "html", "gif"), fps: float = 1.0) -> dict[str, Any]:
    """Solve ``path`` with its trace backend and write the requested outputs."""
    inst = load_instance(path)
    name = path.stem
    algo = inst.meta.get("viz_algorithm", "reference")
    res = solve_with_trace(inst, algo)
    if res.status != "ok" or not res.trace:
        raise RuntimeError(f"{name}: solver status {res.status}: {res.message}")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    renderer = TraceRenderer(inst, res.trace)
    written: dict[str, Any] = {}
    for fmt in formats:
        if fmt in ("svg", "png"):
            frames = renderer.render_all(out / f"frames_{fmt}", fmt)
            written[fmt] = len(frames)
        elif fmt == "html":
            written["html"] = str(render_html(inst, res.trace, out / f"{name}.html", renderer=renderer))
        elif fmt == "gif":
            written["gif"] = str(animate_solution(inst, res.trace, out / f"{name}.gif", fps=fps, renderer=renderer))
    (out / "final_partition.svg").write_text(partition_svg(inst, res.parts, parents=res.certificate.get("parents")))
    (out / "trace.json").write_text(json.dumps(res.trace, indent=1))
    violations = check_expected(inst.meta, res)
    return {
        "name": name, "algorithm": res.algorithm, "n": inst.n, "k": inst.k, "m": inst.m,
        "events": len(res.trace), "valid": res.valid,
        "cycle_shifts": res.stats.get("cycle_shifts", 0), "contractions": res.stats.get("contractions", 0),
        "deletions": res.stats.get("deletions", 0), "terminal_removals": res.stats.get("terminal_removals", 0),
        "roundings": res.stats.get("roundings", 0), "seconds": round(time.perf_counter() - t0, 2),
        "written": written, "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="render only this example (stem of the JSON file)")
    ap.add_argument("--formats", default="svg,html,gif", help=f"comma-separated subset of {FORMATS}")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args(argv)
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    for f in formats:
        if f not in FORMATS:
            raise SystemExit(f"unknown format {f!r}")
    paths = sorted(CURATED.glob("*.json"))
    if args.only:
        paths = [p for p in paths if p.stem == args.only]
        if not paths:
            raise SystemExit(f"no curated example named {args.only!r}")
    rows = []
    for p in paths:
        row = render_example(p, Path(args.out), formats, args.fps)
        rows.append(row)
        flag = "" if not row["violations"] else "  !! " + "; ".join(row["violations"])
        print(f"{row['name']:32s} {row['algorithm']:19s} n={row['n']:2d} k={row['k']} events={row['events']:3d} "
              f"shifts={row['cycle_shifts']} contr={row['contractions']:2d} del={row['deletions']:2d} "
              f"rem={row['terminal_removals']} round={row['roundings']} valid={row['valid']} {row['seconds']:5.1f}s{flag}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(rows, indent=1))
    bad = [r["name"] for r in rows if r["violations"] or not r["valid"]]
    if bad:
        print("expected properties violated for:", ", ".join(bad), file=sys.stderr)
        return 1
    print(f"rendered {len(rows)} examples to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
