# benchmarks/

Benchmark framework and verification dashboard for `glsolver`
(methodology and results: `docs/benchmarks.md`).

```
runner.py      config grid -> deterministic cached instances -> one subprocess per
               (instance, algorithm) -> benchmarks/results/<config>.jsonl
plot.py        publication figures from the JSONL rows -> benchmarks/results/plots/
report.py      markdown summary tables + self-contained HTML verification dashboard
machine.py     hardware / software provenance stored in every row
configs/*.json the grids: smoke, scaling, families, imbalance, directed, weighted,
               dag, connectivity_margin
instances/     instance cache (gitignored; regenerated deterministically; every
               entry carries the file's sha256 and a generator fingerprint and is
               regenerated when either no longer matches)
results/       raw rows, plots, summary.md, dashboard.html (everything under it is
               gitignored except sample_smoke.jsonl)
```

Quick start (from the repository root, inside the virtualenv):

```bash
python -m benchmarks.runner smoke                 # ~30 s, reference solvers + oracles
python -m benchmarks.runner scaling --repeat 3    # the core solvers up to n = 10^4
python -m benchmarks.plot                         # figures
python -m benchmarks.report                       # summary.md + dashboard.html
scripts/run_benchmarks.sh                         # everything, in order
python -m benchmarks.runner --info                # machine metadata
python -m benchmarks.runner dag --dry-run         # list the grid without running
```

`glsolve benchmark CONFIG [-o OUT] [--repeat R] [--threads T] [--limit L]` calls
`benchmarks.runner.run_from_cli` and needs the repository root on `sys.path`
(run it from the root or set `PYTHONPATH=.`).

Useful runner flags: `--algorithms a,b` / `--families f,g` (subset of the
config), `--resume` (skip pairs already in the output), `--fresh` (truncate the
output), `--instances-dir DIR`, `--limit N` (first N instances).

Exit codes of the runner (and of `glsolve benchmark`): `0` when every executed
pair finished (`ok`, `precondition_failed` or `infeasible`), `2` when at least
one pair crashed (`error`) or timed out, `3` when at least one partition was
INVALID.  `scripts/run_benchmarks.sh` stops after the smoke grid on any
non-zero code.

The driver never imports the compiled `glsolver._core` itself; it is probed in
a subprocess (`benchmarks.machine.core_info`), so a broken or sanitizer-linked
build only makes the core algorithms "skipped: C++ core not usable: ..."
(the probe's error line is stored in `machine.core.error`).

Each row of the JSONL output is one `(instance, algorithm)` pair with every
trial, the median, the independent verifier's verdict, primitive call counts,
the child process' CPU time, peak RSS, effective thread environment
(`child.env`) and stderr head/tail, the load average sampled before the run,
the instance metadata (family, seed, capacity mode, connectivity claims,
sha256 of the instance file) and machine/git provenance; see the docstring of
`runner.py` and `docs/benchmarks.md`.
