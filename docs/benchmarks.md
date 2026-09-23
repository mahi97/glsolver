# Benchmarks: methodology, reproduction and results

This document describes how the `glsolver` benchmarks are run
(`benchmarks/runner.py`), how the figures (`benchmarks/plot.py`) and the
verification dashboard (`benchmarks/report.py`) are produced, and — once the
suite has been executed on the reference machine — the result tables.
The result sections (§5) were filled from the v2 sweep of 2026-09-23; the
machine-generated tables they quote are reproduced by
`python -m benchmarks.report` into `benchmarks/results/summary_v2.md`.

## 1. What is measured

For every pair *(instance, algorithm)* the runner records

| quantity | how |
|---|---|
| wall time of one solve | `time.perf_counter()` around `glpartition(..., verify=False, verify_preconditions=False)` in the worker; the solver's own `result.runtime` is stored alongside (`solver_runtime`) |
| CPU time of one solve | `resource.getrusage(RUSAGE_SELF)` delta (user + system) around the call |
| verifier time | the independent verifier `glsolver.verify.verify_instance_parts` is **always** run on the returned partition, timed separately, and its verdict (`valid`) stored per trial |
| peak RSS | the worker's `ru_maxrss` after each trial, and the parent's `os.wait4` rusage of the whole child (`child.peak_rss_mb`, also valid after a crash or kill) |
| primitive counts | `result.stats`: `max_flow_calls`, `matching_calls`, `contractions`, `deletions`, `cycle_shifts`, `terminal_removals`, `min_cost_flow_calls`, `roundings`, heap pushes/pops, the `time_*` breakdown and `graph_size_over_time` (thinned to ≤ 200 samples) |
| instance metadata | `n`, `m` (arcs), `k`, density, capacities, `family`, `seed`, capacity `mode`, variant, generator connectivity claims (`claims_k_connected`, `connectivity_verified`, `kappa` when recorded), for undirected instances with `n ≤ 100` the exact vertex connectivity, and the `sha256` of the instance file plus the `generator_fingerprint` / `glsolver_version` that produced it |
| provenance | `machine`: git commit (and dirty flag), `glsolver` version, CPU model(s) (x86: the `model name` string; ARM: every core type with its CPU count, e.g. `10x Cortex-X925 + 10x Cortex-A725`, plus MIDR implementer/part codes and per-cluster max MHz), core counts, RAM, Python and package versions, the C++ core's availability and build information (probed **out of process**, `machine.core.error` holds the probe's failure line when the core cannot be imported), the driver's thread environment variables, load average and timestamp at the start of the config; per row: `load_average` sampled just before the worker started, `timestamp` when the row was written, `child.env` (the thread variables and `PYTHONHASHSEED` as the worker actually saw them), `child.stderr_head`/`child.stderr_tail` |

Statuses: `ok`, `precondition_failed` (the theorem's guarantee does not apply;
never "no partition exists"), `infeasible` (oracles only, a proof of
non-existence), `timeout`, `error`.  `valid` is `True`/`False` for `ok` runs
and `None` otherwise.  A crashed worker's `error` text is the most informative
stderr line: the headline of a sanitizer/abort report (`==PID==ERROR:
AddressSanitizer: ...`) or the last exception line of a Python traceback.

## 2. Methodology

* **Identical inputs.** Instances are generated deterministically from
  `(family, n, k, capacity mode, seed, variant)` with the seeded generators of
  `glsolver.generators` (plus the official counterexample of
  `glref.counterexample` and the `paper_*` examples) and cached as canonical
  JSON under `benchmarks/instances/`.  Every algorithm reads the same file;
  the file name is the instance name stored in each row.  The sidecar of a
  cached instance records the file's SHA-256 and a fingerprint of the
  generator sources (`glsolver.generators`, `glref.counterexample`) and the
  `glsolver` version; a cache hit requires both to match, otherwise the
  instance is regenerated and the reason logged, so a stale cache can never
  silently feed old instances to a new run.  Both values are copied into every
  row (`instance.sha256`, `instance.generator_fingerprint`), which makes rows
  produced from different instance bytes distinguishable.
* **Subprocess isolation.** Each pair runs in a fresh Python process started
  in its own session.  A crash (exception, abort, segfault) or a hang affects
  only that pair; the row records the exit code, the head and tail of stderr
  and the status.  The worker preloads the backend's modules so lazy imports
  never land inside a timed trial.  The driver itself never imports the
  compiled core: `glsolver._core` is probed in a subprocess
  (`benchmarks.machine.core_info`), so a broken or sanitizer-linked build
  cannot abort the run — the core algorithms are skipped with the reason
  `C++ core not usable: <probe error>` and `machine.core.error` keeps the line.
* **Warmup and repeats.** The worker performs `warmup` untimed solves, then
  `repeats` timed trials.  Every trial is stored; the row's `median` block
  holds the median (and quartiles, min, max) of the successful trials.  Plots
  aggregate the per-row medians across seeds with an IQR band.
* **Timeouts.** The configured `timeout` is a per-phase inactivity limit: the
  parent restarts the deadline whenever the worker reports progress (instance
  loaded, warmup done, trial done).  The whole process group is killed once
  the worker has been silent for `timeout + grace` seconds, where the grace
  period defaults to `max(2 s, 0.1 · timeout)` and can be pinned with the
  config key `grace` (`0` for a hard limit); the effective value is stored in
  every row (`grace`) and in the timeout trial's message.  A timed-out trial
  may therefore have run up to `grace` seconds longer than `timeout`.  Oracles
  also receive `timeout` as their `time_limit` and return `timeout`
  gracefully.  In the plots a timeout is a hollow marker **at the timeout
  value** (the configured limit, not the kill time), never a missing point; a
  crashed pair (status `error`) is an `x` marker at the same level with an
  "(errors)" legend entry, so a crash is as visible as a timeout.
* **Thread control.** `threads` is passed to the solver and exported as
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
  `GLSOLVER_THREADS` to the worker (`0` = solver default).  All shipped configs
  use one thread; `PYTHONHASHSEED=0` is set for reproducibility.  The worker
  echoes the values it actually saw in its start event and the row stores
  them as `child.env` (`machine.env` describes the driver's own environment).
* **Hardware reporting.** `benchmarks/machine.py` collects the machine
  metadata stored in every row (`python -m benchmarks.machine --info` prints
  it).  On ARM hosts, where `/proc/cpuinfo` has no `model name` line, the core
  types come from `lscpu` (or from the MIDR implementer/part table) and the
  `Features` line supplies the flag sample.  Rows from different machines are
  distinguishable by `machine.hostname`, `machine.cpu.model` and `git_commit`.
* **Honest axes.** Runtime and count figures are log-log; the slope of
  `log(runtime)` vs `log(n)` fitted on the largest sizes is printed in the
  legend and next to the last point so a polynomial degree is visible at a
  glance; slow points are never truncated.
* **Independent verification.** The verifier shares no code with the solvers
  (pure BFS on the original input, docs/paper_notes.md §13.11).  The dashboard
  shows its verdict per run and, where an oracle ran on the same instance,
  whether solver and oracle agree (both valid; or `precondition_failed` vs
  `infeasible`).  A verified solver partition on an instance the oracle proved
  infeasible is flagged as a DISAGREEMENT — it would mean a bug in one of them.

## 3. Configurations

| config | purpose | grid |
|---|---|---|
| `smoke` | sanity grid, < 1 min | harary, erdos_renyi, random_kT_dag at n ∈ {10, 20}, k ∈ {2, 3}, paper examples; reference solvers, brute force (n ≤ 12), ILP (n ≤ 20) |
| `scaling` | runtime vs n | harary, random_regular, erdos_renyi, sparse_k_connected, random_kT_dag; n = 10 … 10 000; k = 4; general/weighted/dag, reference up to n = 100, brute force ≤ 12, ILP ≤ 40 |
| `families` | every family | all generator families at n = 200, k ∈ {2, 4, 8, 16} (unrealisable combinations are skipped), paper examples, counterexample with 1 and 17 copies |
| `imbalance` | capacity modes | balanced / unbalanced / random / random_positive / extreme on harary, erdos_renyi, random_regular; n ∈ {50, 200, 1000}, k ∈ {4, 8} |
| `directed` | orientation | erdos_renyi and random_geometric with 25/50/75 % of the edges oriented (`directed_variant`, flow-verified up to n = 200), directed_random, random_kT_digraph |
| `weighted` | weights | `weighted_variant` with w_max ∈ {2, 4, 16, 64} and a slack variant; weighted core vs reference-weighted (n ≤ 100) vs ILP (n ≤ 30) |
| `dag` | DAG solver | random_kT_dag, layered_dag, weighted_kT_dag up to n = 10^6; heap (variant 0) vs linear (variant 1), max_residual vs round_robin; reference-dag up to 10^5 |
| `connectivity_margin` | connectivity vs k | harary (exactly k-connected) vs random 2k- and 4k-regular graphs, G(n,p), dense G(n,0.8) and K_n |

Config JSON keys: `families` (names of `glsolver.generators.family_catalog()`
plus `counterexample` and `paper_*`; an entry may be an object with `name`,
generator parameters such as `copies`, a `generator`/`params` override where
parameter values may be expressions in `n` and `k`, and grid overrides
`sizes`/`k`/`modes`/`seeds`/`variants`), `sizes`, `k` (integers or fractions
of n), `modes`, `seeds`, `variants` (`plain`, `{"type": "directed",
"drop_fraction": f}`, `{"type": "weighted", "w_max": w, "slack": s}`),
`algorithms` (registry names), `algorithm_options` (labelled option sets, e.g.
DAG variants), `algorithm_limits` (`max_n`, `min_n`, `max_k`, `families` per
algorithm), `repeats`, `warmup`, `timeout` (seconds), `grace` (seconds added
to `timeout` before the kill; default `max(2, 0.1 · timeout)`), `threads`,
`store_parts_max_n`.

## 4. How to reproduce

```bash
cd /path/to/gs && source .venv/bin/activate
pip install -e ".[bench]"                      # once: matplotlib, psutil, scipy, tqdm
python -m benchmarks.runner smoke              # sanity: must exit 0 (see below)
scripts/run_benchmarks.sh                      # all configs, then plots + report
# or one at a time
python -m benchmarks.runner scaling --repeat 3 --threads 1
python -m benchmarks.plot                      # -> benchmarks/results/plots/*.png
python -m benchmarks.report                    # -> summary.md, dashboard.html
```

Exit codes of `benchmarks.runner` (and `glsolve benchmark`): `0` when every
executed pair finished with status `ok` (VALID), `precondition_failed` or
`infeasible`; `2` when at least one pair has status `error` (the worker
crashed) or `timeout`; `3` when at least one `ok` partition was INVALID
according to the independent verifier (a correctness bug; outranks `2`).
Skipped pairs (inapplicable algorithm, core not built, `--resume`) never
affect the code.  The smoke gate of `scripts/run_benchmarks.sh` requires exit
`0`: a smoke run with any `error`, `timeout` or INVALID row stops the script
with that code before anything expensive starts; for the other configs a
non-zero code is printed as a warning and the run continues.

`--resume` continues an interrupted config; `--limit N`, `--algorithms` and
`--families` cut the grid down for quick checks; `--dry-run` lists the runs.
The instance cache is gitignored and regenerated on demand; an entry whose
generator fingerprint or file digest no longer matches is regenerated
automatically (deleting `benchmarks/instances/` still works).  Everything
under `benchmarks/results/` is gitignored except the small
`sample_smoke.jsonl`.

The benchmark test (`tests/test_benchmarks.py`) runs the smoke config with
`--limit 6` on the reference backend, checks the row schema and that every run
is VALID, exercises the plotting and reporting code paths, and checks the
driver's isolation from the compiled core, the exit codes, the crash
summaries, the per-row provenance, the cache digests and the failure markers
of the figures; `tests/test_benchmarks_review.py` adds fault injection through
the real worker path.

## 5. Results

All numbers below come from the JSONL rows of the **v2 sweep** (2026-09-23,
`git_commit 480173e755b8cd093d8994c52f0e3686bb5b849e`, `glsolver 0.1.0`,
release build of the C++ core) and are quoted from three keys only:
`median.wall_time` (the median of the timed trials of one row),
`child.peak_rss_mb` (the worker's `ru_maxrss` reaped by `os.wait4`) and
`result.stats.*` (primitive counts).  The tables aggregate the per-row medians
across seeds with the median again.

Machine (`machine` block of every row): `spark-96cb`, aarch64,
**10x Cortex-X925 (3.90 GHz) + 10x Cortex-A725 (2.81 GHz)**, 20 logical cores,
121.63 GiB RAM, CPython 3.12.13, `glsolver._core` available
(`machine.core.available = true`, `debug_asserts = false`).  The general and
weighted solvers were run with `threads = 8`, the DAG grid single-threaded
(`threads = 1`), matching the baseline sweep of RESEARCH_NOTES.md §E2.

Raw data of this sweep:

| file | grid | rows |
|---|---|---:|
| `benchmarks/results/smoke_v2.jsonl` | `smoke` | 46 |
| `benchmarks/results/scaling_v2.jsonl` | `tmp_scaling_v2_{a,b,ref_small,ref_100}` | 457 |
| `benchmarks/results/dag_v2.jsonl` | `tmp_dag_v2_{a,b}` | 252 |
| `benchmarks/results/harary_rss_v2.jsonl` | `tmp_harary_rss` (clean peak-RSS re-measurement) | 2 |
| `benchmarks/results/families_v2.jsonl` | `tmp_families_v2` | 394 |
| `benchmarks/results/plots_v2/` | figures | 16 PNG |
| `benchmarks/results/summary_v2.md`, `dashboard_v2.html` | report | 1 151 rows |

The committed baseline files (`scaling.jsonl`, `dag.jsonl`, `families.jsonl`,
`imbalance.jsonl`, `weighted.jsonl`, `directed.jsonl`,
`connectivity_margin.jsonl`, `smoke.jsonl`) are the *before* evidence of §5.9
and were not overwritten.

### 5.1 Overview (all configs)

| config | rows | ok | INVALID | timeouts | errors | exit | wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke` | 46 | 46 | 0 | 0 | 0 | 0 | 8 s |
| `scaling_v2` stage A (n ≤ 2000, 3 seeds, repeat 3) | 333 | 333 | 0 | 0 | 0 | 0 | 388 s |
| `scaling_v2` stage B (n ∈ {5000, 10000}, seed 1, repeat 1) | 24 | 24 | 0 | 0 | 0 | 0 | 1410 s |
| `scaling_v2` reference n ≤ 50 | 90 | 90 | 0 | 0 | 0 | 0 | 94 s |
| `scaling_v2` reference n = 100 | 10 | 10 | 0 | 0 | 0 | 0 | 243 s |
| `dag_v2` part A (n ≤ 10^5, repeat 3) | 234 | 234 | 0 | 0 | 0 | 0 | 187 s |
| `dag_v2` part B (n = 10^6, repeat 1) | 18 | 18 | 0 | 0 | 0 | 0 | 302 s |
| `harary_rss` (peak RSS re-measurement) | 2 | 2 | 0 | 0 | 0 | 0 | 446 s |
| `families_v2` (n = 200, core solvers only) | 394 | 394 | 0 | 0 | 0 | 0 | 139 s |
| **total** | **1 151** | **1 151** | **0** | **0** | **0** | **0** | **3 217 s = 53.6 min** |

**Every one of the 1 151 rows has `status = "ok"` and `valid = true`.** There
is no `precondition_failed`, no `infeasible`, no `timeout` and no `error` row
in the sweep, and the dashboard (`dashboard_v2.html`, all 1 151 rows) reports
**0 INVALID** and **0 solver/oracle disagreements**.

The `imbalance`, `directed`, `weighted` and `connectivity_margin` grids were
**not** re-run in this sweep (budget); §5.4, §5.5 and §5.7 therefore still
describe the committed baseline files.

### 5.2 Scaling (runtime vs n)

`benchmarks/results/plots_v2/runtime_vs_n_all.png` and the per-family figures.
Median of `median.wall_time` over the seeds (seconds), k = 4, capacity mode
`balanced`; 3 seeds and 3 timed trials per row up to n = 2000, 1 seed and 1
timed trial at n ∈ {5000, 10000}; `reference`/`reference-weighted` 3 seeds up
to n = 50 and seed 1 at n = 100 (they are 3–4 orders of magnitude slower).

**Harary H_{4,n}**

| algorithm | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general | 0.000271 | 0.000991 | 0.001470 | 0.003527 | 0.009082 | 0.08821 | 0.5340 | 3.4352 | 30.524 | **417.63** |
| weighted | 0.000408 | 0.000820 | 0.001618 | 0.003000 | 0.008455 | 0.03246 | 0.1273 | 0.8023 | 13.335 | **266.48** |
| reference | 0.008737 | 0.07176 | 1.0485 | 9.0108 | — | — | — | — | — | — |
| reference-weighted | 0.006696 | 0.06245 | 0.9019 | 8.0706 | — | — | — | — | — | — |
| bruteforce | 7.75e-05 | — | — | — | — | — | — | — | — | — |
| ilp | 0.005335 | 0.09519 | — | — | — | — | — | — | — | — |

**random 4-regular**

| algorithm | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general | 0.000432 | 0.000663 | 0.001923 | 0.004853 | 0.01354 | 0.05463 | 0.1873 | 0.4714 | 2.2485 | 12.369 |
| weighted | 0.000460 | 0.000896 | 0.001871 | 0.004939 | 0.01476 | 0.07208 | 0.2280 | 0.6861 | 3.2983 | 18.067 |
| reference | 0.01014 | 0.1191 | 1.9580 | 15.711 | — | — | — | — | — | — |
| reference-weighted | 0.01014 | 0.09642 | 2.1066 | 16.755 | — | — | — | — | — | — |
| bruteforce | 8.23e-05 | — | — | — | — | — | — | — | — | — |
| ilp | 0.006413 | 0.1435 | — | — | — | — | — | — | — | — |

**Erdős–Rényi (m ≈ n²/10; m = 10 000 610 arcs at n = 10 000)**

| algorithm | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general | 0.000376 | 0.000648 | 0.001351 | 0.002980 | 0.006434 | 0.02533 | 0.1288 | 1.1373 | 28.136 | 210.18 |
| weighted | 0.000239 | 0.000957 | 0.001614 | 0.002611 | 0.005791 | 0.02380 | 0.1365 | 1.0988 | 26.400 | 211.27 |
| reference | 0.03361 | 0.3804 | 8.8614 | 84.309 | — | — | — | — | — | — |
| reference-weighted | 0.02597 | 0.2755 | 7.6954 | 72.708 | — | — | — | — | — | — |
| bruteforce | 8.84e-05 | — | — | — | — | — | — | — | — | — |
| ilp | 0.006878 | 0.01352 | — | — | — | — | — | — | — | — |

**sparse k-connected (Harary + chords)**

| algorithm | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general | 0.000292 | 0.001113 | 0.001558 | 0.004433 | 0.008868 | 0.05068 | 0.1779 | 0.3990 | 2.0771 | 12.070 |
| weighted | 0.000321 | 0.001323 | 0.002311 | 0.004368 | 0.01147 | 0.06170 | 0.1588 | 0.4326 | 2.0262 | 6.7131 |
| reference | 0.01345 | 0.1155 | 1.2285 | 9.3234 | — | — | — | — | — | — |
| reference-weighted | 0.008698 | 0.08889 | 1.2194 | 8.5665 | — | — | — | — | — | — |
| bruteforce | 9.41e-05 | — | — | — | — | — | — | — | — | — |
| ilp | 0.005538 | 0.4894 | — | — | — | — | — | — | — | — |

**random k-T DAG** (the DAG solver and the DAG reference run here too)

| algorithm | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dag | 7.83e-05 | 6.79e-05 | 7.56e-05 | 8.88e-05 | 0.000120 | 0.000204 | 0.000326 | 0.000683 | 0.001611 | **0.002554** |
| general | 0.000425 | 0.000850 | 0.003139 | 0.005333 | 0.01365 | 0.04785 | 0.1329 | 0.3788 | 1.8133 | 8.0212 |
| weighted | 0.000344 | 0.000874 | 0.002138 | 0.006696 | 0.01844 | 0.07738 | 0.2457 | 0.7508 | 4.1950 | 17.008 |
| reference-dag | 0.000106 | 0.000186 | 0.000426 | 0.000660 | 0.001276 | 0.004414 | 0.008087 | 0.01271 | 0.05048 | 0.08795 |
| reference | 0.01068 | 0.07626 | 1.0984 | 8.0375 | — | — | — | — | — | — |
| reference-weighted | 0.007578 | 0.06543 | 1.0925 | 8.5658 | — | — | — | — | — | — |
| bruteforce | 7.24e-05 | — | — | — | — | — | — | — | — | — |
| ilp | 0.004931 | 0.008457 | — | — | — | — | — | — | — | — |

**Fitted scaling exponents.** Least-squares slope of `log(median.wall_time)`
against `log n` over the sizes n ≥ 500 (five points: 500, 1000, 2000, 5000,
10 000; four points for the DAG families of §5.6):

| family | general | weighted | dag | reference-dag |
|---|---:|---:|---:|---:|
| Harary H_{4,n} | **2.76** | **2.99** | — | — |
| random 4-regular | 1.75 | 1.81 | — | — |
| Erdős–Rényi (m ≈ n²/10) | 3.09 | 3.09 | — | — |
| sparse k-connected | 1.77 | 1.57 | — | — |
| random k-T DAG | 1.69 | 1.79 | **0.88** | 1.04 |

The Erdős–Rényi exponent of 3.09 in n is ≈ 1.5 in m (m grows as n²).  The DAG
solver's 0.88 is sub-linear in n over this range because the constant
CSR/topological-order work dominates the small solves; over the DAG grid up to
n = 10^6 (§5.6) it is 1.00–1.13, i.e. linear, as [Alg 5] / P1 predicts.

**Largest instance solved by each algorithm in this sweep** (status `ok`,
verdict VALID, `median.wall_time`):

| algorithm | largest instance | n | m | wall time | 
|---|---|---:|---:|---:|
| `dag` | `layered_dag_n1000000_k4_balanced_s1` | 1 000 000 | 4 999 364 | 0.2771 s (linear) / 0.4122 s (heap) |
| `general` | `erdos_renyi_n10000_k4_balanced_s1` | 10 000 | 10 000 610 | 210.18 s |
| `weighted` | `erdos_renyi_n10000_k4_balanced_s1` | 10 000 | 10 000 610 | 211.27 s |
| `general` (sparse, largest n at m = O(n)) | `harary_n10000_k4_balanced_s1` | 10 000 | 39 984 | 417.63 s |
| `reference-dag` | `random_kT_dag_n10000_k16_balanced_s1` | 10 000 | 159 744 | 0.2800 s |
| `reference` | `erdos_renyi_n100_k4_balanced_s1` | 100 | 1 250 | 84.31 s |
| `reference-weighted` | `erdos_renyi_n100_k4_balanced_s1` | 100 | 1 250 | 72.71 s |
| `ilp` | `erdos_renyi_n20_k4_balanced_s2` | 20 | 162 | 0.01663 s |
| `bruteforce` | `erdos_renyi_n10_k4_balanced_s1` | 10 | 53 | 0.000115 s |

**Peak memory at the largest sizes** (`child.peak_rss_mb`).  Caveat: this key
is the worker's `ru_maxrss` reaped by `os.wait4` and it inherits a floor from
the driver process at spawn time; in stage B the driver's own resident set
grew to 4 106.50 MB while it generated the Erdős–Rényi n = 10 000 instance
(10^7 arcs) and never gave it back, so every row *after* that one reports at
least that value.  The affected rows are marked "≤".  The Harary rows were
therefore re-measured in a separate driver process that only ever touches
Harary instances (`harary_rss_v2.jsonl`).

| family | n | algorithm | peak RSS |
|---|---:|---|---:|
| random 4-regular | 10 000 | general / weighted | 88.996 MB / 95.199 MB |
| sparse k-connected | 10 000 | general / weighted | 133.457 MB / 113.250 MB |
| random k-T DAG | 10 000 | general / weighted / dag / reference-dag | ≤ 96.594 MB (all four at the driver floor) |
| Erdős–Rényi (m = 2.5·10^6) | 5 000 | general / weighted | 1 134.465 MB |
| Erdős–Rényi (m = 10^7) | 10 000 | general / weighted | 4 106.496 MB |
| Harary H_{4,n} | 5 000 | general | **735.438 MB** (clean re-measurement, wall 31.875 s) |
| Harary H_{4,n} | 10 000 | general | **2 735.469 MB** (clean re-measurement, wall 413.043 s) |
| random k-T DAG / layered / weighted DAG | 1 000 000 | dag (all six option sets) | ≤ 2 584.984 MB (driver floor: the 10^6 instance is 119–149 MB of JSON) |

### 5.3 Families at n = 200

`families_v2.jsonl`, 394 rows, **all `ok` and VALID**, core solvers only
(`general`, `weighted`, `dag`); the slow pure-Python reference sub-grids of the
committed `families` config were skipped for budget (they are unchanged — see
the control in §5.9).  n = 200 (n = 60 for `random_kT_digraph`, n = 333 and
n = 3 789 for the official counterexample with 1 and 17 copies), 3 seeds, 3
timed trials, `threads = 8`; median of `median.wall_time` in seconds.  Empty
cells are (family, k) combinations the generator cannot realise, which the
runner skips (269 skipped pairs).

| family | algorithm | k = 2 | k = 4 | k = 8 | k = 16 |
|---|---|---:|---:|---:|---:|
| complete K_200 | general / weighted | 0.010576 / 0.010755 | 0.011652 / 0.010946 | 0.014379 / 0.012555 | 0.015588 / 0.014397 |
| cycle | general / weighted | 0.003090 / 0.003144 | — | — | — |
| wheel | general / weighted | 0.003658 / 0.002812 | — | — | — |
| grid | general / weighted | 0.002989 / 0.003020 | — | — | — |
| grid3d | general / weighted | 0.004190 / 0.003573 | — | — | — |
| harary | general / weighted | 0.003304 / 0.002955 | 0.011401 / 0.008474 | 0.044347 / 0.054053 | 0.045387 / 0.051806 |
| random_regular | general / weighted | 0.007069 / 0.007902 | 0.011884 / 0.012381 | 0.065056 / 0.061687 | 0.18972 / 0.15399 |
| erdos_renyi | general / weighted | 0.004964 / 0.004752 | 0.007079 / 0.006537 | 0.014981 / 0.013656 | 0.074261 / 0.052673 |
| random_geometric | general / weighted | 0.005862 / 0.005247 | 0.012336 / 0.010557 | 0.047398 / 0.040905 | 0.16136 / 0.092183 |
| expander | general / weighted | 0.005504 / 0.006665 | 0.013627 / 0.014512 | 0.048111 / 0.045446 | 0.19242 / 0.15502 |
| dense G(n, 0.8) | general / weighted | 0.013859 / 0.011537 | 0.012386 / 0.012239 | 0.013455 / 0.012677 | 0.015789 / 0.013877 |
| sparse_k_connected | general / weighted | 0.003977 / 0.003775 | 0.010063 / 0.010702 | 0.020157 / 0.031265 | 0.089226 / 0.086935 |
| adversarial_ladder | general / weighted | 0.004133 / 0.003291 | 0.017696 / 0.010259 | 0.016496 / 0.015084 | 0.020379 / 0.023631 |
| random_kT_dag | general / weighted / **dag** | 0.005137 / 0.006340 / **0.000104** | 0.012301 / 0.014468 / **0.000124** | 0.035366 / 0.035110 / **0.000142** | 0.15120 / 0.096044 / **0.000187** |
| layered_dag | general / weighted / **dag** | 0.002805 / 0.002641 / **0.000109** | 0.004156 / 0.004138 / **0.000119** | 0.020982 / 0.014074 / **0.000150** | 0.13332 / 0.097133 / **0.000166** |
| weighted_kT_dag | weighted / **dag** | 0.007707 / **0.000113** | 0.015744 / **0.000111** | 0.043529 / **0.000154** | 0.10910 / **0.000214** |
| directed_random | general / weighted | 0.004962 / 0.004536 | 0.006519 / 0.005801 | 0.011930 / 0.012627 | 0.040593 / 0.023969 |
| random_kT_digraph (n = 60) | general / weighted | 0.001456 / 0.001421 | 0.001816 / 0.001488 | 0.003491 / 0.002811 | 0.003641 / 0.003290 |

Fixed instances (their own k, no k sweep):

| instance | n | m | k | general | weighted |
|---|---:|---:|---:|---:|---:|
| `paper_running_example` | 9 | 12 | 3 | 0.000409 s | 0.000236 s |
| `paper_contract_counterexample` | — | — | 3 | 0.000196 s | 0.000208 s |
| `paper_essential_example` | 13 | 20 | 4 | 0.001198 s | 0.000577 s |
| `counterexample_c1` (official counterexample) | 333 | 2 160 | 9 | 0.009184 s | 0.012153 s |
| `counterexample_c17` (17 copies) | 3 789 | 33 264 | 9 | **0.47277 s** | 0.35222 s |

`counterexample_c17` with `general` is the slowest row of the whole grid
(0.47277 s); every other family at n = 200 is below 0.20 s, and the DAG solver
is between 1.04e-04 s and 2.14e-04 s on all three DAG families at every k.

### 5.4 Capacity imbalance

Not re-run in this sweep; see the `## imbalance` table of the committed
`benchmarks/results/summary.md` (`benchmarks/results/imbalance.jsonl`).

### 5.5 Directed and weighted variants

Not re-run in this sweep; see the `## directed` and `## weighted` tables of
the committed `benchmarks/results/summary.md`
(`benchmarks/results/directed.jsonl`, `benchmarks/results/weighted.jsonl`).

### 5.6 DAG solver up to n = 10^6

`benchmarks/results/plots_v2/dag_variants.png`, `dag_v2.jsonl`, 252 rows, all
`ok` and VALID, single-threaded.  Median of `median.wall_time` over the seeds
(3 timed trials and 2 seeds up to n = 10^5, 1 trial and 1 seed at n = 10^6):

| family (k) | variant / policy | 100 | 1 000 | 10 000 | 100 000 | 1 000 000 |
|---|---|---:|---:|---:|---:|---:|
| random k-T DAG (4) | heap, max_residual | 5.94e-05 | 0.000336 | 0.003031 | 0.04178 | 0.5439 |
| | linear (P1), max_residual | 6.33e-05 | 0.000329 | 0.002843 | 0.04057 | 0.6808 |
| | heap, round_robin | 6.63e-05 | 0.000341 | 0.002897 | 0.03856 | 0.6592 |
| | linear, round_robin | 6.12e-05 | 0.000352 | 0.002757 | 0.04171 | 0.5755 |
| | heap, first | 5.87e-05 | 0.000328 | 0.003008 | 0.03955 | 0.6239 |
| | linear, first | 6.92e-05 | 0.000315 | 0.002665 | 0.04102 | 0.5331 |
| layered DAG (4) | heap, max_residual | 6.48e-05 | 0.000313 | 0.002919 | 0.03564 | 0.4122 |
| | linear (P1), max_residual | 6.07e-05 | 0.000303 | 0.002341 | 0.02752 | 0.2770 |
| | heap, round_robin | 6.24e-05 | 0.000304 | 0.003243 | 0.03431 | 0.4182 |
| | linear, round_robin | 6.75e-05 | 0.000256 | 0.002657 | 0.02561 | 0.2812 |
| | heap, first | 7.17e-05 | 0.000271 | 0.002505 | 0.03086 | 0.3463 |
| | linear, first | 5.88e-05 | 0.000229 | 0.002035 | 0.02439 | 0.2381 |
| weighted k-T DAG (4) | heap, max_residual | 7.94e-05 | 0.000314 | 0.002819 | 0.03362 | 0.5750 |
| | linear (P1), max_residual | 8.65e-05 | 0.000271 | 0.003079 | 0.03454 | 0.6962 |
| | heap, round_robin | 7.70e-05 | 0.000314 | 0.002683 | 0.03208 | 0.5657 |
| | linear, round_robin | 7.16e-05 | 0.000268 | 0.002334 | 0.03102 | 0.6125 |
| | heap, first | 8.27e-05 | 0.000276 | 0.002702 | 0.03040 | 0.5431 |
| | linear, first | 6.35e-05 | 0.000273 | 0.002285 | 0.03078 | 0.5726 |

The table lists the k = 4 sub-grids; `dag_v2.jsonl` also holds a random k-T
DAG k = 16 sub-grid at n ∈ {1 000, 10 000, 100 000} (heap, max_residual
0.000441 s / 0.003727 s / 0.055534 s; linear, max_residual 0.000407 s /
0.003633 s / 0.052423 s).
`reference-dag` (pure Python, up to n = 10^4) and `general` (up to n = 10^3)
ran on the same instances as a control: `reference-dag` needs 0.0701–0.280 s
at n = 10^4 (0.0701 s weighted k-T, 0.0874 s layered, 0.167 s random k-T k = 4,
0.280 s random k-T k = 16) and `general` 0.0225 s (layered) / 0.502 s
(random k-T, k = 4) / 0.987 s (random k-T, k = 16) at n = 10^3.

**Heap (variant 0, exactly [Alg 5], O(m log n)) vs linear (variant 1, P1,
O(n + m)) — `linear / heap` on the same instance and policy:**

| family | n = 100 000 (median of 3 trials × 2 seeds) | n = 10^6 (1 trial) |
|---|---:|---:|
| random k-T DAG, max_residual | 0.04057 / 0.04178 = **0.971** | 0.6808 / 0.5439 = **1.252** |
| random k-T DAG, round_robin | 0.04171 / 0.03856 = **1.082** | 0.5755 / 0.6592 = **0.873** |
| random k-T DAG, first | 0.04102 / 0.03955 = **1.037** | 0.5331 / 0.6239 = **0.854** |
| layered DAG, max_residual | 0.02752 / 0.03564 = **0.772** | 0.2770 / 0.4122 = **0.672** |
| layered DAG, round_robin | 0.02561 / 0.03431 = **0.746** | 0.2812 / 0.4182 = **0.672** |
| layered DAG, first | 0.02439 / 0.03086 = **0.790** | 0.2381 / 0.3463 = **0.688** |
| weighted k-T DAG, max_residual | 0.03454 / 0.03362 = **1.028** | 0.6962 / 0.5750 = **1.211** |
| weighted k-T DAG, round_robin | 0.03102 / 0.03208 = **0.967** | 0.6125 / 0.5657 = **1.083** |
| weighted k-T DAG, first | 0.03078 / 0.03040 = **1.012** | 0.5726 / 0.5431 = **1.054** |

The E2 conclusion stands with one refinement.  On the two k-T DAG families the
ratio scatters on both sides of 1.0 (0.85–1.25) with no size trend, i.e. the
log factor is invisible — at n = 10^6 each of these is a single timed trial,
so a ±25 % spread is what one trial resolves.  On **layered DAG** the linear
variant is consistently and reproducibly faster — 0.75–0.79 at n = 10^5
(median of 3 trials × 2 seeds, all three policies) and 0.67–0.69 at n = 10^6 —
because that family's long chains make the heap deep; there P1 is worth
20–33 %.  The three active-terminal policies are within a few percent of each
other everywhere; `first` is marginally the cheapest.

All six option sets solve n = 10^6 (m ≈ 5·10^6) in **0.238–0.696 s**
single-threaded, and the fitted exponents over n ∈ [10^3, 10^6] are
**1.00–1.13**, i.e. linear.

### 5.7 Connectivity margin

Not re-run in this sweep; see the `## connectivity_margin` table of the
committed `benchmarks/results/summary.md`
(`benchmarks/results/connectivity_margin.jsonl`).

### 5.8 Reference vs core, oracle agreement

`benchmarks/results/plots_v2/speedup_reference_vs_core.png` and
`oracle_vs_solver.png`.  Speed-up of the compiled core over the pure-Python
reference on the *same instance* at n = 100 (seed 1, `median.wall_time`):

| family | reference | general | speed-up | reference-weighted | weighted | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| Harary H_{4,100} | 9.0108 s | 0.002729 s | **3 302×** | 8.0706 s | 0.002615 s | **3 086×** |
| random 4-regular | 15.711 s | 0.004621 s | **3 400×** | 16.755 s | 0.004939 s | **3 392×** |
| Erdős–Rényi | 84.309 s | 0.002980 s | **28 295×** | 72.708 s | 0.003414 s | **21 299×** |
| sparse k-connected | 9.3234 s | 0.004433 s | **2 103×** | 8.5665 s | 0.004368 s | **1 961×** |
| random k-T DAG | 8.0375 s | 0.005255 s | **1 529×** | 8.5658 s | 0.006696 s | **1 279×** |

The DAG reference is a different matter — it is the same linear algorithm in
Python, so the gap is a constant factor, not an asymptotic one:
`reference-dag / dag` is 7.4× at n = 100 (0.000660 s vs 8.88e-05 s) and
**34.4×** at n = 10 000 (0.087950 s vs 0.002554 s).

**Oracle agreement.**  The smoke grid ran `bruteforce` (n ≤ 12) and the ILP
(n ≤ 20) next to the reference solvers on 46 pairs; the scaling grid adds
`bruteforce` at n = 10 and the ILP at n ∈ {10, 20} on all five families.  Every
one of those rows is `ok` and VALID, no oracle ever reported `infeasible` on an
instance a solver partitioned, and the dashboard's disagreement counter is
**0**.

### 5.9 Before and after the oracle optimization (E3)

The committed `benchmarks/results/scaling.jsonl` (the E2 baseline sweep,
2026-09-21, 453 rows) and `benchmarks/results/dag.jsonl` (252 rows) are the
*before* evidence.  The comparison below is **like for like**: rows are paired
on (family, n, k, seed, algorithm, option label) **and** on
`instance.sha256`, and all 453 + 252 pairs matched with identical instance
bytes — the generator fingerprint changed since the baseline, so every
instance was regenerated, but the regenerated files are byte-identical.

**Harary H_{4,n} — the headline.**  This is the family that forced the baseline
sweep to stop at n = 2000 (56 GB of RSS).  `general`, per seed:

| n | seed | before: wall | after: wall | speed-up | before: peak RSS | after: peak RSS | memory factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1 | 0.31279 s | 0.06900 s | 4.53× | 442.3 MB | 100.0 MB | 4.4× |
| 500 | 2 | 1.26076 s | 0.40972 s | 3.08× | 1 196.3 MB | 100.0 MB | 12.0× |
| 500 | 3 | 0.25916 s | 0.08821 s | 2.94× | 283.8 MB | 100.0 MB | 2.8× |
| 1 000 | 1 | 2.9699 s | 0.53405 s | 5.56× | 3 384.6 MB | 102.0 MB | 33.2× |
| 1 000 | 2 | 9.79917 s | 2.85159 s | 3.44× | 9 463.2 MB | 102.0 MB | 92.8× |
| 1 000 | 3 | 1.8813 s | 0.39534 s | 4.76× | 1 998.4 MB | 104.0 MB | 19.2× |
| 2 000 | 1 | 29.562 s | 3.43517 s | 8.61× | 22 227.6 MB | 179.7 MB | 123.7× |
| 2 000 | 2 | 129.519 s | 20.816 s | 6.22× | 56 551.1 MB | 247.4 MB | **228.6×** |
| 2 000 | 3 | 18.0863 s | 2.07213 s | 8.73× | 14 142.5 MB | 164.9 MB | 85.8× |
| 5 000 | 1 | *not run (out of memory budget)* | 30.524 s | — | *—* | 735.438 MB | — |
| 10 000 | 1 | *not run (out of memory budget)* | **417.63 s** | — | *—* | **2 735.469 MB** | — |

`weighted` on the same instances is the same picture, with larger spread
(n = 2000: 25.074 s → 0.80226 s = 31.25× on seed 1, 94.334 s → 21.425 s =
4.40× on seed 2, 17.423 s → 0.44063 s = 39.54× on seed 3; peak RSS
20 583.9 → 176.0 MB, 56 753.3 → 247.9 MB, 13 956.7 → 157.6 MB).

The 228.6× memory factor at H_{4,2000} seed 2 reproduces RESEARCH_NOTES.md
§E3-verify exactly (56 551 MB → 248 MB).  **The consequence is the point of
this sweep: Harary is no longer memory-bound and now reaches n = 10 000 like
every other family** — 417.63 s and 2.74 GB of RSS — less than 5 % of the
56.5 GB the baseline needed already at n = 2 000.  Extrapolating the baseline's
memory exponent (RSS ≈ n^2.9, §E2) to n = 10 000 would have required several
terabytes.

**The other families** (matched rows, n ≥ 500, geometric mean of the per-row
`before / after` ratio of `median.wall_time`):

| family | general | weighted | memory at the largest matched size |
|---|---:|---:|---|
| Harary | **4.94×** (9 rows, 2.94–8.73×), 7.76× at n = 2000 | **10.85×** (9 rows, 4.40–39.54×), 17.59× at n = 2000 | 22 228 → 180 MB … 56 551 → 247 MB |
| random 4-regular | 1.03× (0.59–1.80×); 1.60× at n = 10 000 (19.7945 → 12.3686 s) | 0.80× (0.63–1.29×); 1.14× at n = 10 000 (20.5864 → 18.0673 s) | 231.8 → 89.0 MB (2.6×) |
| sparse k-connected | 1.09× (0.65–1.55×); 1.32× at n = 5 000, 0.83× at n = 10 000 | 1.17× (0.94–1.52×); 1.34× / 1.52× | 4 093.7 → 89.0 MB at n = 5 000 (46×; the *before* value is itself a driver-floor artefact) |
| Erdős–Rényi | 1.11× (0.82–1.32×); 1.08× at n = 10 000 (226.194 → 210.177 s) | 1.15× (1.02–1.32×); 1.02× (214.569 → 211.274 s) | 4 093.7 → 4 106.5 MB (unchanged) |
| random k-T DAG | 1.62× (1.13–2.64×) | 0.84× (0.61–1.31×) | unchanged (≤ driver floor) |

This is exactly the shape §E3 predicted: the families that never suffered from
the per-arc user index gain little or nothing (Erdős–Rényi 11–15 %, random
regular 0–60 %, and a few sub-1.0 ratios that are load and single-trial
noise), while Harary — where `after_contraction` and the index were 69 % of
the run — gains 3–40× in time and 3–229× in memory.

**The DAG solver also got substantially faster**, which E2 did not predict
(252/252 rows matched on identical instance bytes; geometric means of
`before / after` over all matched rows):

| family | heap | linear | heap-rr | linear-rr | heap-first | linear-first |
|---|---:|---:|---:|---:|---:|---:|
| random k-T DAG | 3.68× | 3.64× | 3.61× | 3.71× | 3.91× | 3.91× |
| layered DAG | 2.86× | 3.13× | 2.92× | 3.16× | 3.03× | 3.64× |
| weighted k-T DAG | 2.30× | 2.08× | 2.35× | 2.41× | 2.30× | 2.47× |

At n = 10^6 the baseline needed 1.96–2.37 s for every option set; this sweep
needs **0.238–0.696 s** (random k-T DAG heap 1.9410 → 0.5439 s, layered DAG
linear 2.0174 → 0.2770 s, weighted k-T DAG heap 2.1350 → 0.5750 s).  The
control is the pure-Python `reference-dag` on the very same instances, whose
`before / after` ratios are 0.74–0.99× — unchanged, as it must be, since no
Python code changed.  The gain is therefore in the C++ core: the flow-engine
micro-optimizations of §E3.5 (the shared `flow`/`forbidden` byte array and the
maintained `next_arc[x]`) act on the CSR/topological pass that E2 identified
as the DAG solver's dominant cost.
