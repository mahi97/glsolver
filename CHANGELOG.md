# Changelog

## 0.1.0 (2026-09-21)

First release: a complete, independently written implementation of the
polynomial-time Győri–Lovász algorithm of arXiv:2608.30945, with two solver
stacks (a literal pure-Python reference and an optimized C++20 core), an
independent verifier, exact oracles, a trace visualizer and a benchmark suite.
Paper labels below are those of `docs/paper_notes.md`.

### Algorithms

* **Unweighted `GLPartition` [Alg 1] + `ShiftAssignment` [Alg 2]** — the three
  operations (zero-capacity terminal removal, contraction of an out-degree-one
  pre-terminal, deletion of a non-critical arc), saturating pre-terminal
  matching [Lem 7.8], secondary arcs, criticality table [Def 6.1], reassignment
  graph and potential-decreasing cycle shifts [Lem 7.11].
* **Weighted `GLWeightedPartition` [Alg 3] + `RoundAndRemove` [Alg 4]** —
  min-cost split assignments [Prop 5.4] under FESAC [Def 5.3], the minimal
  Hall-deficient set [Lem 7.6], and the `c_t + w_max − 1` output bound.
* **DAG `GLDAGPartition` [Alg 5]** — canonical topological order [Def 9.2],
  `O(m log n)` heap version [Lem 9.6] and an `O(n+m)` stack variant
  (`RESEARCH_NOTES.md` P1) producing the identical contraction sequence, with
  the `max_residual`, `round_robin` and `first` terminal policies.
* **Supported formulations**: classical undirected Győri–Lovász (sizes `n_i`),
  Lovász's directed version (capacities `c_i`, in-arborescence output),
  weighted instances, and DAGs; preconditions are `k`-`T`-connectivity or the
  weaker FEAC [Def 5.1] / FESAC, with the DAG out-degree criterion [Lem 9.1].
  A failing precondition yields `status == "precondition_failed"`, never a
  claim that no partition exists.
* **Certificates**: every solve returns per-part in-arborescence parents built
  from *original* arcs through `orig_head` bookkeeping (§13.4), plus the final
  witness where available.

### Reference implementation (`reference/glref/`)

* Literal transcription of the paper — first-found choices, full recomputation
  of essential sets after every terminal removal and arc deletion, an explicit
  criticality table — used as ground truth for the core.
* Primitives: `DiGraphState` (deletion, contraction [Def 2.1], terminal
  removal), vertex-split max-flow with tightest-cut extraction [Prop 4.2],
  essential terminals plus the literal `k+1`-flow cross-check [Def 4.1],
  witness search [Def 5.1], criticality tests, Hall-deficient sets and
  saturating matchings, min-cost flow, cut union/intersection helpers
  [Def 3.5].
* Compact-connectivity recognizer [Lem A.9] and a generator for the paper's
  appendix counterexample with its claim checks.
* `debug=True` enables the paper's invariants A1–A8 (§7.2) after every step,
  raising `InvariantError` on a violation.

### C++20 core (`src/glcore/`, built to `glsolver._core` via pybind11)

* `Graph` (lazy vertex deletion, arc membership hashing, `orig_head`),
  CSR vertex-split `FlowEngine`, `EssentialOracle` (per-vertex flow
  certificates, bitset essential sets, tightest cuts, warm starts, parallel
  recomputation), `CriticalityOracle`, Hopcroft–Karp `Matching` with minimal
  Hall-deficient sets, `MinCostFlow` (successive shortest paths with
  potentials), `GLSolver`, `GLWeightedSolver`, `dag_partition`, plus `Stats`
  and `Trace`.
* Exact optimizations O1–O10 of `docs/optimizations.md`: unused-arc batching,
  warm-started criticality, no recomputation after contraction, incremental
  terminal removal, greedy contraction with a credit throttle, lazy
  `ShiftAssignment`, batched secondary-arc deletion, OpenMP parallel oracle,
  bitset essential sets and the DAG heap/stack variants.
* Certified-subset discipline (`RESEARCH_NOTES.md` P2): stored essential sets
  are subsets of the truth containing `φ(v)`, while every decision uses an
  exact cut.
* Build switches `GLCORE_DEBUG_ASSERTS=ON` (invariants A1–A8 and the
  FESAC/DAG invariants compiled in) and `GLCORE_SANITIZE=ON` (ASan/UBSan).

### Python package (`src/glsolver/`)

* `glpartition()` dispatcher (`auto`, `general`, `weighted`, `dag`,
  `reference`, `reference-weighted`, `reference-dag`, `bruteforce`, `ilp`),
  the `partition()` alias for classical undirected input, and `GLResult`
  with parts, assignment, status, message, verifier verdict, runtime, stats,
  certificate and optional trace.
* `Instance` / `make_instance` / `from_networkx` normalization (self-loops,
  parallel arcs and arcs out of terminals dropped; `sizes` ↔ `capacities`),
  canonical JSON and edge-list I/O.
* **Independent verifier** `glsolver.verify` — pure Python, standard library
  and BFS on the original input only, importing no solver module; checks part
  count, exact cover, ids, terminals, exact sizes or the weighted bound, and
  connectivity (directed reachability, plus undirected connectivity for
  undirected input, which must agree).
* `glsolver.preconditions` — `k`-`T`-connectivity, FEAC/FESAC and the DAG
  criterion decided by definition with NetworkX max-flows, sharing no code
  with the solvers.
* Exact oracles: `bruteforce` (backtracking over connected sets, the only
  component that can prove non-existence) and `ilp` (single-commodity-flow
  MILP via HiGHS), both with time limits.
* Seeded generators: 21 families (Harary, random regular, Erdős–Rényi,
  sparse/dense `k`-connected, expander, grid, grid3d, wheel, cycle, complete,
  random geometric, adversarial ladder, random `k`-`T`-connected digraphs and
  DAGs, layered and weighted DAGs, the three paper examples, …) plus
  `weighted_variant`, `directed_variant` and capacity modes.
* Testing helpers: solver registry, regression store with content-addressed
  names, ddmin instance minimizer, and a `SIGALRM` time bound that turns a
  hanging solver into a test failure.

### CLI (`glsolve`)

* `solve`, `verify`, `generate`, `inspect`, `visualize`, `benchmark`, plus the
  `glsolve GRAPH --terminals …` shorthand for `solve`.
* JSON instances and edge lists, `--capacities`/`--sizes`/`--weights`,
  `--algorithm`, `--preconditions`, `--stats`, `--trace`, `--seed`,
  `--threads`, `--time-limit`, `--debug`, `-o`; exit codes `0` (verified),
  `2` (non-`ok` status) and `3` (verifier rejected the partition).

### Visualization

* `glsolver.viz`: `TraceRenderer` (per-event SVG/PNG/PDF frames), a
  self-contained interactive `render_html` viewer, `animate_solution` (GIF, MP4
  with `imageio-ffmpeg`), `partition_svg`/`final_figure`, an abstracted figure
  of the official counterexample, deterministic layouts (`grid`, `kamada`,
  `spring`, `auto`) and a trace-only state replay that imports no solver
  internals.
* Seven curated examples under `examples/curated/` covering every operation
  (running example, essential-set example, contraction counterexample, a
  three-cycle-shift instance, zero-capacity terminal removal creating a new
  essential terminal, weighted rounding, a DAG) with asserted trace properties
  and `examples/render_all.py`.

### Benchmarks

* `benchmarks/runner.py`: deterministic cached instances (SHA-256 plus
  generator fingerprint), one subprocess per `(instance, algorithm)` pair,
  warmup and repeats, per-phase inactivity timeouts with a grace period, thread
  pinning, always-on independent verification, and machine/git provenance in
  every row.
* Eight configurations (`smoke`, `scaling`, `families`, `imbalance`,
  `directed`, `weighted`, `dag`, `connectivity_margin`), `benchmarks/plot.py`
  figures, `benchmarks/report.py` summary tables and a self-contained HTML
  verification dashboard, and `scripts/run_benchmarks.sh` with a smoke gate.
* Exit codes `0` / `2` (crash or timeout) / `3` (INVALID partition).

### Correctness and verification

* 2 469 tests across 31 suites (pytest, Hypothesis with `ci`/`dev`/`thorough`
  profiles), including per-module adversarial review suites.
* Exhaustive: every connected graph on `n ≤ 7` (145 400 instances) and every
  digraph on `n ≤ 4` with `k ∈ {1,2}` (3 278 instances) solved by the reference
  *and* the C++ solvers with debug assertions — 0 failures
  (`scripts/run_exhaustive.py`, `docs/verification.md` §10).
* Differential fuzzing of the C++ core against the reference: about 4 400
  instances on the primitives (identical tightest cuts) and about 6 000 on the
  solvers with all option combinations, including trace replay checked against
  exact essential sets.
* The official compact-connectivity counterexample is reproduced exactly
  (1, 2 and 17 copies); the 17-copy instance (`n = 3 789`, `m = 33 264`,
  `k = 9`) is solved in 1.8 s by the general core and 1.4 s by the weighted
  core, against 900 s for the 1-copy instance in pure Python.
* Mutation matrix showing that several injected algorithm bugs are caught only
  by the paper invariants, which is why every theorem solver runs with
  `debug=True` in the harness.
* Sanitizer builds (ASan/UBSan with debug assertions) are clean.

### Documentation

* `docs/paper_notes.md` (master specification, lemma → function map,
  underspecified points, trace model), `theorem.md`, `algorithm.md`,
  `flow_essential_assignment.md`, `weighted_algorithm.md`, `dag_algorithm.md`,
  `optimizations.md`, `implementation.md`, `verification.md`, `benchmarks.md`,
  `api.md`, `instance_format.md`; READMEs for `examples/`, `visualization/` and
  `benchmarks/`.
* `RESEARCH_NOTES.md`: proved results P1 (`O(n+m)` DAG algorithm), P2
  (certified-subset essential sets), P3 (sparse preprocessing for dense
  undirected inputs); conjectures C1–C3; first measurements (§E1).

### Packaging and CI

* `scikit-build-core` + `pybind11` build, optional extras `ilp`, `viz`,
  `bench`, `test`, `dev`; `glsolve` console script; ruff and mypy configuration.
* Reproducible `Dockerfile` (builds the core, installs `[dev]`, smoke-tests the
  import).
* GitHub Actions: test job on Python 3.10 and 3.12 (ruff, `ci` Hypothesis
  profile), a sanitizer job building with `GLCORE_SANITIZE=ON` and
  `GLCORE_DEBUG_ASSERTS=ON` and running the core tests under ASan/UBSan, and a
  Docker build with a CLI smoke test.
