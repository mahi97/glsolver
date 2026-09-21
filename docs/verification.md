# Verification: how we know the solvers are correct

This document describes the correctness harness of the project (Stage D):
what the independent verifier checks, what the exact oracles guarantee, how
the exhaustive and property-based tests are organised, what they *cannot*
see and how that gap is closed, how failing instances are stored and
minimized, how to run everything, and the counts observed on the reference
machine.  Section references (`§`, `[Def …]`, `[Lem …]`) point to
`docs/paper_notes.md`.

## 1. The independent verifier (`glsolver.verify`)

`verify_partition(graph, terminals, capacities_or_sizes, parts, ...)` /
`verify_instance_parts(inst, parts)` is the ground truth of the project.  It
is pure Python, uses only the standard library, the shared `Instance`
container and breadth-first search on the **original** input, and imports no
solver, oracle or `glref` module.  All problems are collected (it never stops
at the first one) and returned in a `VerificationReport(valid, errors,
details)`.  Checks:

| check | what it asserts |
|---|---|
| (a) part count | exactly `k` parts, one per terminal |
| (b) exact cover | every vertex `0..n-1` appears in exactly one part (duplicates and missing vertices reported individually) |
| (c) vertex ids | every id is an in-range integer |
| (d) terminals | `terminals[i] ∈ parts[i]` and no part contains a foreign terminal |
| (e) sizes / weights | unweighted: `|parts[i]| == c_i + 1` exactly; weighted: non-terminal weight of `parts[i]` `≤ c_i + w_max − 1` [Thm weighted-k-t-conn]; parts exceeding `c_i` are listed in `details["exceeds_capacity"]` |
| (f) connectivity | directed: every vertex of `parts[i]` reaches `terminals[i]` by a directed path inside the part [Def connected-to] (BFS over reversed original arcs); undirected: `G[parts[i]]` is connected in the original undirected edge set, and the directed check is run as well — the two must agree (§1.1, §13.11) |

Malformed *arguments* (bad keyword combinations, an instance failing
`Instance.validate`) raise; every problem with the *partition* is reported.
Every solver result produced through `glpartition(..., verify=True)` carries
the verifier's verdict in `result.valid` / `result.verification`; the harness
additionally re-runs `verify_instance_parts` on every output and asserts the
size/weight bounds explicitly, so a regression in the API's own bookkeeping
cannot hide a wrong partition.

## 2. What the oracles guarantee (`glsolver.oracle`)

* **`bruteforce`** — exact backtracking over connected vertex sets grown
  from each terminal.  It relies on **no** connectivity precondition and is
  complete: `"infeasible"` is returned only when *no* partition satisfying
  the verifier's conditions exists.  It is the only component that can prove
  non-existence, and it is used to (i) confirm that the theorem solvers'
  partitions exist independently on a seeded sample, (ii) tell apart
  "precondition fails, partition exists" from "no partition at all" on tiny
  digraphs, and (iii) cross-check the ILP.
* **`ilp`** — exact single-commodity-flow-per-part MILP (HiGHS via
  `scipy.optimize.milp`); the formulation is exact for [Def connected-to]
  and agrees with the brute force on all atlas instances (`tests/test_oracles.py`).
* The theorem solvers never answer `"infeasible"`: `"precondition_failed"`
  means the guarantee of the theorem does not apply (§13.9) and a partition
  may still exist.  `tests/test_exhaustive_small.py` records how often that
  happens (see §9).
* Both oracles take a `time_limit` and answer `"timeout"` when they hit it.
  The harness gives them 60 s per call; a `"timeout"` is **inconclusive**,
  not a failure (§5).

`glsolver.preconditions` (NetworkX max-flows, by definition, no shared code)
decides the preconditions used by the harness: `k`-`T`-connectivity
[Def 3.2], the Flow-Essential Assignment Condition [Def 5.1] (FESAC
[Def 5.3] for weighted instances) and the DAG out-degree criterion [Lem 9.1].

## 3. Solver registry

`glsolver.testing.registry.available_solvers()` lists every backend usable
in the current environment: the pure-Python reference solvers `reference`,
`reference-weighted`, `reference-dag`, the C++ core `general`, `weighted`,
`dag` when it is built (`glsolver.core_available()`), and the oracles
`bruteforce` and `ilp` (the latter when SciPy/HiGHS are installed).
`solvers_for(inst)` narrows this to the backends applicable to an instance
(unweighted-only solvers are skipped for weighted input; DAG solvers need an
acyclic, `k`-`T`-connected instance).  Every test below is parametrized over
the registry, so a newly built backend is tested automatically.

`GLSOLVER_NO_CORE=1` makes `core_available()` answer `False` even when a
`_core` extension is installed.  It exists because a core built with
`GLCORE_SANITIZE=ON` aborts the interpreter at import time unless
`libasan` is preloaded (`LD_PRELOAD=$(gcc -print-file-name=libasan.so)
ASAN_OPTIONS=detect_leaks=0`), which would otherwise kill every pytest run
silently; with the switch the pure-Python harness always runs.

Every theorem solver is run by the harness with **`debug=True`**: the
reference solvers then check the paper's invariants A1–A8 (§7.2) after every
step and raise `InvariantError` on a violation, and for the C++ backends
`glpartition` maps `debug` to the core's `debug_asserts` option
(`api._core_options`).  §4.3 explains why this is not optional.

## 4. Exhaustive small instances (`tests/test_exhaustive_small.py`)

**Undirected.**  Every connected graph of the NetworkX graph atlas
(`networkx.graph_atlas_g()`, one representative per isomorphism class) with
`2 ≤ n ≤ 6` and vertex connectivity `κ(G) ≥ 1`; for every `k ∈ 1..min(κ(G), 3)`
every `k`-subset of terminals and every composition of `n − k` into `k`
nonnegative capacities.  For `n ≤ 5` all subsets and compositions are used;
for `n = 6` a seeded sample of 3 terminal subsets and up to 6 compositions
per subset keeps the default run short (`GL_EXHAUSTIVE_FULL=1` runs all
8272 instances of `n = 6`).  Every applicable backend must answer `"ok"`
(the Győri–Lovász theorem guarantees a partition on a `k`-connected graph)
and the verifier must accept; a seeded 10 % sample is also solved by the
brute force.  Expected counts are asserted: `n = 2, 3, 4, 5` give `2, 12, 90,
685` instances.  The tally also records the number of cycle shifts per
backend and asserts that the `reference` solver performed at least one, so
the ShiftAssignment path is known to be exercised.

**Directed.**  Every digraph on `n ≤ 4` vertices with `k ∈ {1, 2}`: every
terminal subset, every subset of the `(n−k)(n−1)` arcs leaving non-terminals
(arcs out of terminals are dropped anyway, §2), every capacity composition —
3278 instances.  `check_preconditions(inst, True)` decides FEAC by
definition.  On FEAC instances every applicable backend must return a valid
partition; on the others the theorem solvers must return
`"precondition_failed"` — never `"infeasible"`, never a crash — and the brute
force records how many of them still admit a partition (the theorem is
sufficient, not necessary).  The test also asserts the sanity relation
`k`-`T`-connected ⇒ FEAC.

### 4.1 The sampled `k = 3` layer

With `k ≤ 2` the reassignment graph of ShiftAssignment [Alg 2] has at most
two terminals, so the only cycle is a swap: a wrong cycle shift is either
caught or a no-op, and the length-`r ≥ 3` case of [Lem 7.11] is never
reached.  `k = 3` on `n = 5` does not help: `Σ c = 2` forces a zero-capacity
terminal, which step (i) of [Alg 1] removes *before* any shift, leaving two
terminals again.  Three live terminals with positive capacity need `n ≥ 6`,
`c = (1, 1, 1)`.  The directed test therefore adds a seeded sample of that
layer (`K3_LAYER`: `n = 6`, `k = 3`, every terminal subset, 64 of the
`2^15` arc subsets per subset, `c = (1, 1, 1)` only — 1280 instances,
`strategies.iter_directed_exhaustive(min_n=6, max_n=6, ks=(3,),
mask_sample=64, min_capacity=1)`), runs the `reference` solver with a trace
on it and asserts that reassignment cycles of length 3 really occur
(79 on the reference machine, §9).  The full layer (`20 · 32768` arc
subsets) and the full `n = 5`, `k = 3` layer are available offline through
`scripts/run_exhaustive.py` (§8).

### 4.2 Time bound: a hang is a failure, not a hung suite

Nothing in an exhaustive loop bounds a single solver call, so a
non-terminating solver bug (the review's `contract_drops_in_arcs` mutant
makes `shift_assignment` loop forever on 8 of the 789 `n ≤ 5` undirected
instances) used to hang the whole test-suite instead of failing it — no
verdict after 13 minutes.  `glsolver.testing.timebound` closes this:

* `time_bound(seconds)` arms a `SIGALRM` timer around one solver call and
  raises `SolverTimeout` when it expires.  `SolverTimeout` derives from
  **`BaseException`** on purpose: `glpartition` converts every `Exception`
  raised by a backend into `status == "error"`, so an `Exception`-based
  alarm would be misreported as a solver error (the review demonstrates
  this), and Hypothesis shrinks every `Exception` but re-raises other
  `BaseException`s immediately — each shrink attempt of a hanging instance
  would otherwise cost the full limit again.  pytest reports such an
  exception as an ordinary failed test.  Nested bounds re-arm the outer timer
  with its remaining time.
* The per-call limit is `$GL_SOLVER_TIME_LIMIT` seconds (default 20; `0`
  disables the bound); the self-limiting oracles get their own 60 s on top.
  The exhaustive instances take milliseconds, so the default is a generous
  margin even on a loaded CI machine.
* In the exhaustive tests a hang is booked as a failure of kind `timeout`,
  stored under `regression/` with `kind: "timeout"`, and **ends the
  enumeration** with an assertion naming the instance — every further
  hanging instance would cost the full limit again.  The property tests store
  the instance as `hyp_<strategy>_<solver>_hang.json` (not minimized, for the
  same reason) and raise `HarnessHang`, also a `BaseException`.
  `scripts/run_exhaustive.py` has `--time-limit` and `--max-timeouts` (§8).
* Limitations: `SIGALRM` handlers can only be installed in the main thread
  and on POSIX (elsewhere the block runs unbounded), and a Python signal
  handler runs only when the interpreter regains control — a hang *inside* a
  C extension (the C++ core, HiGHS) is reported when that call returns.

### 4.3 Debug invariants: what output verification alone does not see

Output verification is the ground truth, but on instances this small it is
**not** a complete safety net: several algorithm bugs still produce a valid
partition on every enumerated instance and are visible only as a violated
paper invariant.  The review's mutation matrix (`tests/test_harness_review.py`)
measured, with the reference solvers run at `debug=False`:

| mutant (bug injected into `glref`) | undirected `n ≤ 6` (2271) | directed `n ≤ 4` (3278) |
|---|---|---|
| `stale_ess_after_terminal_removal` — step (i) does not recompute `Ess` (§13.3) | valid on all | valid on all |
| `contract_out_degree_two` — step (ii) contracts pre-terminals of out-degree 2 (the paper's contraction counterexample) | valid on all | 12 crashes |
| `random_cycle_shift` — every shifted vertex goes to a random terminal | 6 failures | valid on all (`k ≤ 2`) |
| `contract_drops_in_arcs` — contraction drops incoming arcs [Def 2.1] | no verdict (hang) | no verdict (hang) |

With `debug=True` the same enumerations catch them massively (`stale_ess`
255 + 144, `contract_out_degree_two` 824 + 48, `random_cycle_shift`
694 + 41 `InvariantError`s: A7, [Lem 7.4], [Lem 7.11]) with no false
positive on the real code, at a cost of about +1 s per enumeration.  The A7
check even fires on the hanging instances of `contract_drops_in_arcs`
*before* they hang.  Hence every theorem solver runs with `debug=True` in
the exhaustive tests, the property tests and the offline runner, the review
asserts for each of these pairs that the harness fails *through an
`InvariantError`* (`INVARIANT_ONLY`), and the time bound of §4.2 remains the
safety net for hangs that no invariant pre-empts (asserted with the
invariants switched off, `HANGS`).  The C++ backends get the same treatment
through `debug_asserts`; note that the reference invariants are the paper's
A1–A8 while the core's assertions are its own, so both should stay on.

### 4.4 Tally

A tally (instances per `n`, solver runs and cycle shifts per backend, oracle
runs, failures with their kind, seconds, the per-call time limit) is printed
and written to `$GL_EXHAUSTIVE_TALLY` when set, otherwise to a file in
pytest's temporary directory (`tally.json` under `tmp_path_factory`; the
path is printed).  The old default `tests/.exhaustive_tally.json` left an
untracked artifact in the source tree and is gone.  Failures are saved into
the regression store.

## 5. Property-based tests (`tests/strategies.py`, `tests/test_property_solvers.py`)

Hypothesis strategies produce normalized `Instance` objects:

| strategy | instances |
|---|---|
| `st_undirected_k_connected(max_n=10, max_k=4)` | `G(n,p)` with drawn density (random regular / Harary / complete with small probability), rejected unless `κ(G) ≥ 1`; `k ∈ [1, min(κ(G), max_k, n−1)]`; terminals in random order; capacities from `st_capacities` |
| `st_directed_kT(max_n=9, max_k=3)` | random digraphs with density biased so out-degrees reach `k`, rejected unless `generators.is_k_t_connected` (flow test [Def 3.2]) |
| `st_feac_only(max_n=9)` | digraphs with FEAC **true** and `k`-`T`-connectivity **false** by `check_preconditions` — Theorem 2 beyond Theorem 1 (a plain random digraph, or a `k`-`T`-connected one in which one vertex keeps only `1..k−1` out-arcs; ~45 % acceptance) |
| `st_dag(max_n=14, max_k=4)` | `generators.random_kT_connected_dag` / `layered_dag` with drawn `n, k, seed, extra_out`, layer shape and capacity mode |
| `st_weighted(base, w_max≤5, slack≤3)` | `generators.weighted_variant` of a drawn base instance with drawn seed, `w_max` and slack |
| `st_random_digraph(max_n=8)` | arbitrary digraphs with **no** precondition (honesty tests) |
| `st_capacities(total, k, mode)` | `balanced` (all equal up to one), `unbalanced` (`(1,…,1,total−k+1)`), `extreme` (all zero but one), `random` (weak composition), `zeros-allowed` (one zero plus a weak composition); every vector under a random permutation |

**`k == n` is never drawn** by any strategy (`k ≤ n − 1` by construction),
nor produced by the undirected enumeration (`k ≤ min(κ(G), 3) ≤ n − 1`).
That degenerate shape (every part a singleton, all capacities `0`, all arcs
dropped) is covered by `tests/test_edge_cases.py` (`n ∈ {1, 2, 3, 5}`, `K_n`
for `n = 2..6`) and by the directed enumeration for `n = 1, 2`; the review
asserts this so nobody assumes the strategies cover it.

Properties (each parametrized over the registry; inapplicable backends are
skipped explicitly):

* **every applicable solver solves every drawn instance** for the strategies
  `undirected`, `directed_kT`, `feac_only`, `dag`, `weighted_undirected`,
  `weighted_directed_kT`, `weighted_dag`: `status == "ok"`, `valid is True`,
  independent re-verification, exact sizes (unweighted) or the
  `c_t + w_max − 1` bound (weighted), the parts cover `0..n−1` and
  `terminals[i] ∈ parts[i]`.  Every theorem solver runs with `debug=True`
  (reference: invariants A1–A8; core: `debug_asserts`), every call under
  the time bound of §4.2.
* **certificates**: the `parents` map returned by `reference`,
  `reference-dag` (and the core, when built) is an in-arborescence of
  *original* arcs — every non-terminal's parent chain stays inside its part
  and ends at the part's terminal, without cycles.  On unit weights with tight
  capacities the weighted reference never rounds (§13.6) and its certificate
  is checked the same way.
* **honesty on arbitrary digraphs**: a theorem solver answers only `"ok"`
  (then valid) or `"precondition_failed"`, never `"infeasible"` or a crash,
  and never reports a precondition failure on a FEAC instance.
* **weighted FEAC-only digraphs are solved iff FESAC holds** by definition.
* **core vs reference agreement** (`general`/`reference`,
  `weighted`/`reference-weighted`, `dag`/`reference-dag`): both valid on
  the same instance; skipped while the C++ core is not built.

Two outcomes are deliberately not failures: an **oracle answering
`"timeout"`** within its own 60 s limit (`bruteforce`/`ilp` on DAG instances
up to `n = 14`) is inconclusive, so the example is rejected
(`hypothesis.reject()`, counted as a Hypothesis event) and no regression file
is written — a slow machine cannot turn a benign timeout into a red test and a
spurious regression entry; and a **hang** is reported as described in §4.2.

Profiles (`tests/conftest.py`): `ci` (60 examples), `dev` (200, default),
`thorough` (2000); pick one with `--hypothesis-profile NAME` or
`HYPOTHESIS_PROFILE=NAME`.  All profiles disable the per-example deadline.

## 6. Regression store and minimization

`glsolver.testing.regression.save_regression(inst, reason, name=None,
extra=None)` writes `regression/<name>.json` — the canonical instance JSON
(docs/instance_format.md) plus a `regression` block with the reason, the
UTC creation time, a 12-hex-digit content hash and any extra fields.
Without `name` the file is named `<family>_n<n>_k<k>_<hash>.json`, so a
given instance is stored once.  `load_all()` yields
`(path, instance, meta)` triples.

Conventions used by the harness:

* `meta["expect"]` is `"ok"` (default) or `"precondition_failed"`;
  `tests/test_regression_store.py` re-solves every stored instance with
  every applicable backend and asserts the expectation (oracles only for
  `n ≤ 14`, reference solvers on `n > 300` are marked slow).
  `meta["kind"]` (`"invalid"`, `"crash"`, `"timeout"`) says how the solver
  failed when the entry was created.
* A failing Hypothesis example is stored as `hyp_<strategy>_<solver>.json`
  (fixed name, so the shrinking phase overwrites the same file and the final
  minimal example wins), then shrunk further with
  `glsolver.testing.minimize.minimize_instance(inst, fails)` — ddmin over
  arcs, then non-terminal vertex removal — under the predicate *"the
  precondition still holds (`solvers_for` + `check_preconditions`) and the
  solver still fails (an oracle time limit does not count)"*, and the result
  is stored as `hyp_<strategy>_<solver>_min.json` when it is smaller.  The
  predicate matters: without the precondition clause the minimizer would
  happily reduce to an instance where `"precondition_failed"` is the
  *correct* answer.  A hang is stored as `hyp_<strategy>_<solver>_hang.json`
  and never minimized (§4.2).
* Exhaustive failures (pytest and `scripts/run_exhaustive.py`) are stored
  with hash-based names and `expect` set to the expected status.

The store is seeded with three instances (all FEAC-only):

| file | reason |
|---|---|
| `paper_running_example.json` | the paper's running example (§1.2, §4): FEAC holds, not 3-`T`-connected |
| `terminal_removal_13_3.json` | §13.3: terminal removal creates new essential terminals; capacities `(1, 0, 1)` make FEAC hold and force step (i) of [Alg 1] |
| `feac_only_random_seed1337.json` | random FEAC-only digraph (`n = 9`, `k = 3`, `m = 24`), found deterministically from seed 1337 |

## 7. Edge cases (`tests/test_edge_cases.py`)

`k == n` (every part is a singleton, all arcs dropped, capacities all 0,
also through `partition(G, T, sizes=[1]*n)`), `k == n − 1`, `k == 1`,
`n == 1`, several zero capacities (with the reference solver's
terminal-removal count checked against its trace), a terminal without
in-arcs (fine with capacity 0; `precondition_failed` for the theorem solvers
and `infeasible` for the oracles with capacity > 0), Harary graphs
`H_{k,n}` (connectivity exactly `k`), complete graphs for every `k`,
self-loops / duplicate edges / multigraphs / arcs out of terminals
normalized, NetworkX graphs with string labels round-tripped through
`parts_labels` and `verify_partition`, the three worked examples of the
paper through every applicable backend, and the authors' official
counterexample with `copies = 1` (`n = 333`, `k = 9`, `m = 2160`) through
`reference` (`@pytest.mark.slow`) and through the C++ `general` backend when
it is built.

## 8. Running everything

```bash
cd /home/admin/app/gs && source .venv/bin/activate

# the harness (default "dev" profile, 200 Hypothesis examples per property)
pytest tests/test_exhaustive_small.py tests/test_property_solvers.py \
       tests/test_edge_cases.py tests/test_regression_store.py -q -p no:cacheprovider

# the review of the harness itself (mutation matrix, strategy audits, time bound, ...)
pytest tests/test_harness_review.py -q -p no:cacheprovider

# quicker / more thorough Hypothesis runs
pytest tests/test_property_solvers.py --hypothesis-profile=ci
HYPOTHESIS_PROFILE=thorough pytest tests/test_property_solvers.py

# every n = 6 instance instead of the seeded sample; custom tally path
GL_EXHAUSTIVE_FULL=1 GL_EXHAUSTIVE_TALLY=/tmp/tally.json pytest tests/test_exhaustive_small.py -s

# per-call time bound (seconds; 0 disables it) and ignoring an installed C++ core
GL_SOLVER_TIME_LIMIT=5 pytest tests/test_exhaustive_small.py
GLSOLVER_NO_CORE=1 pytest tests/test_property_solvers.py

# slow tests (official counterexample through the pure-Python reference solver)
pytest tests/test_edge_cases.py --runslow -k counterexample

# offline exhaustive runner: n <= 7 undirected (136 339 instances for n = 7 alone),
# all digraphs on n <= 4, every backend, 16 processes; tally under benchmarks/results/
python scripts/run_exhaustive.py --max-n 7 --jobs 16
python scripts/run_exhaustive.py --max-n 6 --solver reference --oracles --jobs 8
python scripts/run_exhaustive.py --max-n 5 --no-directed --out /tmp/small.json
# hangs: --time-limit per call (default $GL_SOLVER_TIME_LIMIT or 20), stop after --max-timeouts
python scripts/run_exhaustive.py --max-n 6 --time-limit 10 --max-timeouts 1
# extra digraph layers: the full n = 5, k = 3 layer (15 360 instances) and a
# sampled n = 6, k = 3, c = (1,1,1) layer (length-3 reassignment cycles)
python scripts/run_exhaustive.py --no-undirected --directed-min-n 5 --directed-max-n 5 --directed-ks 3
python scripts/run_exhaustive.py --no-undirected --directed-min-n 6 --directed-max-n 6 --directed-ks 3 \
    --directed-min-capacity 1 --directed-mask-sample 512 --jobs 8

# lint
ruff check tests scripts
```

The full test-suite of the repository (`pytest -q -m "not slow"`) includes
the harness; CI runs it with `--hypothesis-profile=ci`.

## 9. Observed counts (reference machine, C++ core not built)

Backends registered: `reference`, `reference-weighted`, `reference-dag`,
`bruteforce`, `ilp`.  All theorem solvers at `debug=True`, time limit 20 s
per call (never reached).

`pytest tests/test_exhaustive_small.py` (default sampled run, 13 s total):

| enumeration | instances | solver runs | cycle shifts (`reference`) | oracle runs | failures |
|---|---|---|---|---|---|
| undirected, `n ≤ 6` (`n = 6` sampled), 4.5 s | 2271 (`n=2..6`: 2, 12, 90, 685, 1482) | `reference` 2271, `reference-weighted` 2271, `reference-dag` 65 | 937 | bruteforce 226 | 0 |
| undirected, `n ≤ 6`, `GL_EXHAUSTIVE_FULL=1` | 9061 (`n = 6`: 8272) | 9061 / 9061 / 119 | | bruteforce 905 | 0 |
| directed, `n ≤ 4`, `k ∈ {1,2}`, 3.1 s | 3278: FEAC 1760 (of which `k`-`T`-connected 1502), non-FEAC 1518 | `reference` 3278, `reference-weighted` 3278, `reference-dag` 1508 | 54 | bruteforce on all 1518 non-FEAC instances: 48 still have a partition | 0 |
| directed `k = 3` sample (`n = 6`, `c = (1,1,1)`, 64 arc subsets per terminal subset), 4.1 s | 1280: FEAC 635 (of which `k`-`T`-connected 145), non-FEAC 645 | `reference` 1280, `reference-weighted` 1280, `reference-dag` 268 | 386, of which 79 reassignment cycles of length 3 | bruteforce on all 645 non-FEAC instances: none has a partition | 0 |

Before the debug invariants were switched on the two default enumerations
took 3.4 s and 2.4 s; the invariants cost about +1 s each.

`scripts/run_exhaustive.py` (measured before `debug=True` became the
default; expect roughly +30 %):

| run | instances | solver runs | wall time | failures |
|---|---|---|---|---|
| `--max-n 6 --jobs 8` | 9061 undirected + 3278 directed | 12 339 × `reference`, 12 339 × `reference-weighted`, 1627 × `reference-dag` | 6.8 s | 0 |
| `--max-n 7 --min-n 7 --no-directed --jobs 16` | 136 339 (all 853 connected 7-vertex graphs) | 136 339 × `reference`, 136 339 × `reference-weighted`, 73 × `reference-dag` | 106 s | 0 |

Official counterexample, `copies = 1` (`n = 333`, `k = 9`, `m = 2160`),
`algorithm="reference"`: valid partition after 900 s of pure Python
(965 560 max-flow calls, 324 contractions, 546 arc deletions, 48 cycle
shifts, 8 terminal removals) — hence `@pytest.mark.slow`.

Mutation matrix of the review (`tests/test_harness_review.py`): all 36
(mutant, harness test) pairs are caught; the four `INVARIANT_ONLY` pairs
through an `InvariantError`, the two `HANGS` pairs through a `timeout`
failure within the 3 s review limit once the invariants are switched off.

Test counts (`GLSOLVER_NO_CORE=1`, C++ core not registered):
`tests/test_harness_review.py` 67 passed in 110 s;
`tests/test_property_solvers.py` + `tests/test_exhaustive_small.py` at the
`ci` profile 35 passed, 11 skipped (statically inapplicable strategy/backend
pairs and the three core-vs-reference tests) in 20 s;
`tests/test_edge_cases.py` + `tests/test_regression_store.py` +
`tests/test_api_cli.py` 71 passed, 2 skipped (slow / core counterexample).

## 10. Results (whole project, C++ core built)

The counts of §9 were taken with the C++ core absent.  This section records
the state of the project once the core is built and registered, i.e. with the
backends `reference`, `reference-weighted`, `reference-dag`, `general`,
`weighted`, `dag`, `bruteforce` and `ilp` all available.

**Test suite.**  2 469 tests across 31 suites pass (pytest; Hypothesis at the
`ci`, `dev` and `thorough` profiles).  Every theorem solver runs with
`debug=True` — the reference checks the paper's invariants A1–A8 (§7.2), the
C++ backends their own `debug_asserts` — and every call runs under the time
bound of §4.2.

### 10.1 Exhaustive enumerations

Both C++ solvers were run, with debug assertions, over the full enumerations of
§4 in addition to the reference solvers.  **No failure of any kind** (invalid
partition, wrong status, crash, invariant violation, timeout) was observed.

| enumeration | instances | solver runs | failures |
|---|---|---|---|
| undirected, every connected graph on `n ≤ 7` | **145 400** (`n ≤ 6`: 9 061, of which `n = 6`: 8 272; `n = 7`: 136 339) | `general` 145 400, `weighted` 145 400, `dag` 119 (`n ≤ 6`) | 0 |
| directed, every digraph on `n ≤ 4`, `k ∈ {1, 2}` | **3 278** (FEAC 1 760, of which `k`-`T`-connected 1 502; non-FEAC 1 518) | `general` 3 278, `weighted` 3 278, `dag` 1 508 | 0 |

The enumerations are the ones of §4, driven offline by `scripts/run_exhaustive.py`
with 16 processes: the `n ≤ 6` run (9 061 undirected + 3 278 directed) takes
1.1 s, the `n = 7` layer (136 339 undirected) 17.5 s.  Both tallies are stored,
under `benchmarks/results/exhaustive_core_n6.json` and `exhaustive_core_n7.json`.
On the FEAC instances every applicable backend returns a valid partition; on the
1 518 non-FEAC digraphs every theorem solver returns `precondition_failed`
(never `infeasible`, never a crash), and the brute force run of §9 confirms that
48 of them nevertheless admit a partition — the theorem is sufficient, not
necessary.

### 10.2 The official counterexample

The appendix's compact-connectivity counterexample (§10) is reproduced exactly
for `copies = 1`, `2` and `17`; the generated instances match the authors'
construction and the claims of their script (compact connectivity holds, every
arc deletion breaks it, every pre-terminal contraction breaks it) are
re-checked by `glref.counterexample.check_counterexample_claims`.

| instance | `n` / `m` / `k` | result |
|---|---|---|
| 1 copy, `algorithm="reference"` | 333 / 2 160 / 9 | valid partition after 900 s of pure Python (§9) |
| 17 copies, C++ `general` | 3 789 / 33 264 / 9 | valid partition in 1.8 s |
| 17 copies, C++ `weighted` | 3 789 / 33 264 / 9 | valid partition in 1.4 s |

This is the instance the FEAC machinery was invented to survive: compact
connectivity is maintained by *no* single operation, while FEAC is, so a solver
that contracts or deletes greedily without re-establishing the condition fails
here and nowhere else in the suite.

### 10.3 Differential fuzzing of the C++ core

`tests/test_core_review.py` (primitives) and `tests/test_core_solver_review.py`
(solvers) push seeded random instances through the core and compare with the
validated reference `glref` and the independent verifier.  All randomness is
seeded and every assertion names the seed of the failing case, so any failure
is reproducible from the test id.

* **Primitives — about 4 400 instances** (`n` 3..15, `k` 1..5, directed and
  undirected, including the degenerate shapes: `k = 1`, `k = n − 1`, isolated
  vertices, vertices with no path to `T`, terminals without in-arcs, raw arc
  lists with self-loops / parallel arcs / arcs leaving terminals).  Checked:
  `tightest_cut` [Prop 4.2] — `κ`, the `L`/`S`/`R` sides and `Ess`
  **identical** to `glref.flow.tightest_min_cut`, returned paths a valid family
  [Def 3.2]; `Graph` mutations [Def 2.1] arc-for-arc identical to
  `glref.graph.DiGraphState` after random operation sequences, `orig_head`
  parents included (§13.4); the `EssentialOracle` after random operation
  sequences (O1–O4, P2) — `κ` always exact, the stored `Ess` a certified subset
  and exactly the reference's whenever the oracle claims an exact cut;
  `evaluate_deletion` (O2/O5) against `glref.critical.is_critical` for every
  `(v, t)` and against the reference on `G \ D` for arc sets; matching
  [Lem 7.8] and the minimal Hall-deficient set [Lem 7.6]; min-cost flow
  [Prop 5.4] on networks with small and with `2^40`-sized capacities and costs
  (int64 overflow reported, never wrapped); `dag_partition` [Alg 5] in both
  variants and all three policies; and bit-identical results at `threads=8`
  versus `threads=1`.
* **Solvers — about 6 000 instances** (`n` 3..14, `k` 1..5: undirected
  `k`-connected, directed `k`-`T`-connected, FEAC-only confirmed with
  `check_preconditions`, DAGs, arbitrary digraphs, weighted variants;
  capacities balanced / unbalanced / random / with zeros / all-in-one) through
  `algorithm="general"` and `"weighted"` with **every** combination of
  `greedy_contraction`, `lazy_shift`, `batch_unused_arcs` and `debug_asserts`.
  Every partition is accepted by the independent verifier and its
  in-arborescence certificate checked against the **original** arcs; statuses
  agree with the literal reference solvers and, for `n ≤ 8`, with the brute
  force (the core never answers `precondition_failed` where the reference
  answers `ok` or vice versa; the core never returns `error`).
* **Trace replay.**  Instances on which the greedy (O5) or batched (O1) paths
  fired are re-solved with them off, and every trace is replayed with the exact
  reference primitives: after each delete / contract / remove / shift event the
  witness `φ` carried by the trace satisfies `φ(v) ∈ Ess_G(v)` with **exact**
  essential sets and exact capacity counts, independently of the core's
  certified subsets (`RESEARCH_NOTES.md` P2); weighted traces are checked for
  FESAC feasibility with `glref.weighted.min_cost_split_assignment`.
* **Degenerate and adversarial inputs**: `k == n`, `k == 1`, `n == 1`,
  single-capacity vectors, isolated vertices, terminals without in-arcs, raw
  arc lists fed straight into `_core`, huge weights, `k > 64`, random junk,
  vertex-relabelling invariance, and thread determinism at `threads=16`.
* **Watch-list checks** for the failure modes a review of this code flagged:
  terminal index versus vertex id confusion, a stale `φ` after terminal
  removal, capacity off-by-one, a secondary arc equal to its matching arc,
  cycle detection at `k = 1` (no shift can happen, [Lem 7.10]) and `k = 2`
  (every cycle has length 2), and rounding parents being original arcs.

Two bugs found this way are kept as named regression tests: an
`EssentialOracle::after_contraction` that kept serving stale "exact" cuts after
contracting a pre-terminal of out-degree ≥ 2 (fixed by advancing the exact-cut
epoch), and a greedy-contraction case found by fuzzing a random digraph
(`n = 28`, `k = 6`, unbalanced capacities).

### 10.4 Sanitizers

The core builds cleanly with `GLCORE_SANITIZE=ON` **and**
`GLCORE_DEBUG_ASSERTS=ON`, and the core test files run under ASan/UBSan with
no report (`ASAN_OPTIONS=detect_leaks=0`, both `libasan` and `libstdc++`
preloaded — see `docs/implementation.md` for why both are needed).  This is the
`sanitizers` job of `.github/workflows/ci.yml`; it is roughly 20x slower than
the release build, which is why it runs only the `tests/test_core_*.py` files.
