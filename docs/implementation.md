# Implementation architecture

```
src/glsolver/           Python package (API, CLI, verifier, oracles, generators, viz, benchmarks glue)
src/glcore/             C++20 core, built with pybind11 → glsolver._core
reference/glref/        transparent pure-Python reference implementation of the paper (ground truth for the core)
tests/                  pytest + Hypothesis (exhaustive small instances, property tests, reference-vs-core)
benchmarks/             benchmark runner, configs, raw results (JSONL), plotting
visualization/          step-by-step trace visualizer (SVG/PNG/HTML/GIF)
examples/               curated instances and scripts
scripts/                reproducibility scripts (exhaustive runs, CI helpers)
regression/             permanently stored regression instances (JSON)
```

## Layers

1. **Instance layer** (`glsolver.instance`): normalized directed instance
   (arcs, terminals, capacities, optional weights). All other layers consume it.
2. **Reference layer** (`glref`): `DiGraphState` + explicit flow network +
   literal paper algorithms. Optimized for readability; every function cites the
   paper. Used by tests and as an `algorithm="reference"` option.
3. **Core layer** (`glcore`, C++):
   * `Graph`: mutable simple digraph, lazy vertex deletion, arc membership hash,
     `orig_head` bookkeeping for contraction (paper_notes §13.4).
   * `SplitFlowNetwork`: the vertex-split unit-capacity network of [Prop 4.2],
     kept in CSR form for the current graph version; per-vertex flows are stored
     compactly (arc-id lists) so they can be warm-started (docs/optimizations.md O2).
   * `EssentialOracle`: per-vertex `κ`, `Ess` bitsets, stored flows, tightest cuts.
   * `CriticalityOracle`: O1/O2 tests for one arc or an arc set against all vertices.
   * `Matching`: Hopcroft–Karp terminals→pre-terminals; minimal Hall-deficient set.
   * `MinCostFlow`: successive shortest paths with potentials for split assignments.
   * `GLSolver` (unweighted), `GLWeightedSolver`, `dag_partition`: the state machines.
   * `Stats`/`Trace`: counters, timers, event log.
4. **API layer** (`glsolver.api`): `glpartition()` dispatcher (`algorithm="auto"`
   chooses DAG → general → weighted…), `GLResult`, precondition handling,
   verifier invocation, NetworkX interop.
5. **Tools**: CLI (`glsolve`), visualization, dashboard, benchmarks.

## Dispatcher (`algorithm="auto"`)

```
DAG and every non-terminal has out-degree ≥ k   → dag        (C++; reference-dag if the binding is missing)
weighted instance                               → weighted   (C++; reference-weighted if missing)
otherwise                                       → general    (C++; reference if missing)
```

Backend availability is checked per binding (`core_available("general" | "weighted" | "dag")`),
so a partially built extension never makes a backend look available. The oracles
(`bruteforce`, `ilp`) and the pure-Python solvers are only used when named explicitly.

## Determinism

All heuristic choices are seeded (`seed=` argument); parallel loops write into
per-index slots and reduce deterministically.

## Debug builds

`GLCORE_DEBUG_ASSERTS=ON` compiles the invariant assertions A1–A8 of
paper_notes §7.2 plus the FESAC/DAG invariants into the core; on failure the
core raises an exception carrying the full state, which the Python layer
serializes into `regression/` via `glsolver.testing.regression.save_failure`.
`GLCORE_SANITIZE=ON` builds with ASan/UBSan (used in CI). Running the tests then
needs both runtimes preloaded, `LD_PRELOAD="$(gcc -print-file-name=libasan.so)
$(gcc -print-file-name=libstdc++.so)" pytest ...` (plus `ASAN_OPTIONS=detect_leaks=0`):
CPython does not link libstdc++, and libasan's `__cxa_throw` interceptor resolves
the real symbol at ASan init, before the extension module loads libstdc++, so
preloading libasan alone aborts the interpreter on the first C++ exception.
