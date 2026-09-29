"""Driver for the competitive comparison.

    python benchmarks/compare/run_compare.py [--budget 300] [--out ...] \
        [--methods a,b,c] [--families ...] [--sizes ...] [--resume]

Every (instance, method) pair runs in its own subprocess with an external hard
kill at ``budget + grace`` seconds; a kill is recorded as a timeout, never as a
missing row.  Rows are appended to ``benchmarks/results/compare/compare.jsonl``
as they finish, so the run is resumable and a crash loses at most one row.

For the exact solvers a *ladder* rule keeps the total run finite: once a method
times out (or errors, or hits a licence limit) on a family, larger ``n`` of that
same family are recorded as ``skipped_after_timeout`` instead of burning another
300 s each.  This only ever *helps* the competitor's reported numbers -- the
largest ``n`` it solved is unaffected.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from methods import ALL_METHODS, LADDER_METHODS  # noqa: E402

from instances import SCALED_FAMILIES, SIZES, all_specs, instance_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "compare" / "compare.jsonl"
WORKER = Path(__file__).resolve().parent / "worker.py"
GRACE = 60.0  # seconds of slack over the budget before the external kill


def run_one(spec: dict, method: str, budget: float) -> dict:
    payload = json.dumps({"spec": spec, "method": method, "budget": budget})
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), payload],
            capture_output=True, text=True, timeout=budget + GRACE, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {
            "instance": instance_id(spec), "spec": spec, "method": method,
            "status": "timeout", "wall_s": budget + GRACE, "build_s": 0.0, "solve_s": 0.0,
            "budget_s": budget, "over_budget": True, "answered": False, "valid": None,
            "failure_modes": None, "verifier_errors": [],
            "extra": {"message": "hard-killed by the driver"},
        }
    out = proc.stdout
    if "@@RESULT@@" in out:
        tail = out.split("@@RESULT@@", 1)[1].strip().splitlines()
        for line in tail:
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    return {
        "instance": instance_id(spec), "spec": spec, "method": method,
        "status": "error", "wall_s": time.perf_counter() - t0, "build_s": 0.0, "solve_s": 0.0,
        "budget_s": budget, "over_budget": False, "answered": False, "valid": None,
        "failure_modes": None, "verifier_errors": [],
        "extra": {"message": f"worker exit={proc.returncode}",
                  "stderr": proc.stderr[-1200:], "stdout": out[-600:]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--families", default=",".join(SCALED_FAMILIES) + ",special")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            ap.error(f"unknown method {m!r}; choose from {sorted(ALL_METHODS)}")
    fams = {f.strip() for f in args.families.split(",") if f.strip()}
    sizes = {int(s) for s in args.sizes.split(",") if s.strip()}

    specs = []
    for sp in all_specs():
        fam = sp["family"]
        special = fam.startswith("paper_") or fam == "counterexample"
        if special:
            if "special" in fams:
                specs.append(sp)
        elif fam in fams and sp["n"] in sizes:
            specs.append(sp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["instance"], r["method"]))
    elif args.out.exists():
        args.out.unlink()

    # ladder bookkeeping: (method, family) -> n at which it first failed
    stopped: dict[tuple[str, str], int] = {}

    fh = args.out.open("a")
    total = len(specs) * len(methods)
    i = 0
    for method in methods:
        for spec in specs:
            i += 1
            fam = spec["family"]
            iid = instance_id(spec)
            if (iid, method) in done:
                print(f"[{i}/{total}] skip (done)  {iid:32s} {method}", flush=True)
                continue
            n = spec.get("n")
            key = (method, fam)
            if method in LADDER_METHODS and key in stopped and n is not None and n > stopped[key]:
                row = {
                    "instance": iid, "spec": spec, "method": method,
                    "status": "skipped_after_timeout", "wall_s": None,
                    "build_s": None, "solve_s": None, "budget_s": args.budget,
                    "over_budget": None, "answered": False, "valid": None,
                    "failure_modes": None, "verifier_errors": [],
                    "extra": {"message": f"{method} already failed on {fam} at n={stopped[key]}"},
                    "n": n, "k": spec.get("k"),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"[{i}/{total}] LADDER-SKIP  {iid:32s} {method}", flush=True)
                continue

            row = run_one(spec, method, args.budget)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            st, w, ok = row["status"], row.get("wall_s"), row.get("valid")
            ws = f"{w:8.3f}s" if isinstance(w, (int, float)) else "   n/a  "
            print(f"[{i}/{total}] {iid:32s} {method:20s} {st:22s} {ws} valid={ok}", flush=True)
            if method in LADDER_METHODS and n is not None:
                if st != "ok" or row.get("over_budget"):
                    stopped.setdefault(key, n)
    fh.close()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
