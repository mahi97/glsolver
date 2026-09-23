# Final report

An independent implementation of the polynomial-time Győri–Lovász algorithm of
**arXiv 2608.30945**, *Breaking the Exponential Barrier: The First
Polynomial-Time Algorithm for the Győri–Lovász Theorem*, by Mohammad T.
Hajiaghayi, Mahdi JafariRaviz, Alireza Kaviani and Soheil Mohammadkhani.

This repository is not affiliated with, endorsed by, or reviewed by the
authors. The mathematics is theirs; the code, the errors and the engineering
claims are ours. The authors' own repository (`mahdi-jfri/Gyori-Lovasz-Codes`,
MIT) contains the appendix counterexample only; it is reproduced here
bit-for-bit as a regression test (§5.4).

---

## 1. What was implemented

| Component | Where | Status |
|---|---|---|
| Reference solver, unweighted (`GLPartition` + `ShiftAssignment`) | `reference/glref/unweighted.py` | complete, literal transcription of the paper |
| Reference solver, weighted (`GLWeightedPartition`, `RoundAndRemove`) | `reference/glref/weighted.py` | complete |
| Reference solver, DAG (`GLDAGPartition`) | `reference/glref/dag.py` | complete, three terminal policies |
| Compact connectivity + official counterexample | `reference/glref/compact.py`, `counterexample.py` | complete, matches the authors' code exactly |
| Optimized core: graph, flows, essential-terminal oracle | `src/glcore/{graph,flow,essential}.cpp` | complete, C++20 |
| Optimized core: general and weighted solvers | `src/glcore/solver_{general,weighted}.cpp` | complete |
| Optimized core: DAG solver (heap + O(n+m) variant) | `src/glcore/dag.cpp` | complete |
| Independent verifier | `src/glsolver/verify.py` | complete, shares no code with any solver |
| Oracles: exhaustive brute force, MILP (HiGHS) | `src/glsolver/oracle/` | complete |
| Seeded instance generators (19 families) | `src/glsolver/generators.py` | complete |
| Python API, dispatcher, CLI | `src/glsolver/{api,cli}.py` | complete |
| Step-by-step visualization (SVG/PNG/PDF/HTML/GIF) | `src/glsolver/viz/` | complete, 7 curated examples |
| Benchmark framework + verification dashboard | `benchmarks/` | complete, 8 grids |
| Documentation | `docs/` (12 files), `README.md`, `RESEARCH_NOTES.md` | complete |

Size: 6 676 lines of C++, 6 646 of Python package code, 2 514 of reference
implementation, 18 726 of tests across 33 files, 4 111 of documentation,
3 649 of benchmark/visualization/scripting code. 23 commits.

## 2. Exact mathematical coverage

Everything the paper states constructively is implemented:

* **Classical undirected Győri–Lovász** — via the reduction to the directed
  formulation (each edge becomes two arcs, arcs out of terminals dropped,
  `c_i = n_i − 1`), with the equivalence argued in `docs/paper_notes.md` §1.1.
* **Lovász's directed formulation** — the paper's own setting: `k`-`T`-connected
  digraphs, parts that are connected *to* their terminal (spanning
  in-arborescences).
* **Weighted / confluent-flow generalization** — integer weights, capacities as
  upper bounds, the unavoidable additive slack `w_max − 1`, `RoundAndRemove`,
  minimum-cost split assignments.
* **DAG special case** — the near-linear algorithm, both as the paper's
  `O(m log n)` heap version and as an `O(n + m)` variant we proved (§9, P1).
* **The relaxed conditions of the appendix** — flow-essential assignment
  (FEAC), its split-assignment form (FESAC), compact connectivity and local
  connectivity, including the hierarchy proofs and the counterexample.

The one thing deliberately *not* implemented as a solver is the appendix's
Győri-style cascade proof (`GLIncremental`): it is exponential and exists in
the paper only to prove the theorem under local connectivity. Its statement is
documented; its algorithm is not shipped.

## 3. Algorithms and how they are dispatched

`algorithm="auto"` picks: a DAG whose non-terminals all have out-degree ≥ k →
the DAG solver; a weighted instance → the weighted solver; otherwise the
general solver. Each falls back to its pure-Python reference when the
corresponding C++ binding is unavailable, and the oracles are only used when
named explicitly. Backend availability is probed per binding, so a partially
built extension never masquerades as available.

## 4. Deviations from the paper

Each is documented at the point of use and validated experimentally.

1. **Any FEAC-preserving deletion is legal** (`docs/paper_notes.md` §13.1). The
   paper's `ShiftAssignment` is how it *proves* a deletable arc exists; the
   correctness induction only needs the condition to hold after the step. The
   core therefore tries cheaper deletions first (O1, O5) and falls back to the
   paper's procedure, which still guarantees progress.
2. **Certified-subset essential sets** (P2). The core stores a subset of each
   vertex's essential terminals containing its assigned terminal, refreshing to
   exactness wherever a decision depends on it. This is what makes incremental
   re-validation sound.
3. **Essential sets are unchanged by a degree-one contraction** (§13.2) and
   **can grow when a terminal is removed** (§13.3). Both follow from the
   paper's proofs but are not stated there; both are asserted in debug builds.
4. **Original-arc bookkeeping** (§13.4). Contraction rewrites arcs, so each arc
   into a terminal remembers the original head, which is what makes the
   returned in-arborescence a certificate in the *input* graph.
5. **Choice points** (§13.7, §13.8) — which terminal to serve in the DAG
   algorithm, which matching, which secondary arc, which `v_i`, which cycle —
   are free in the paper. The defaults are documented and each is exercised by
   tests; none affects validity.

## 5. Correctness evidence

### 5.1 Independent verification

Every result returned by any solver, in every test and every benchmark row, is
checked by `glsolver.verify`, which imports no solver module and re-derives
everything from the original input by breadth-first search: partition
coverage, terminal membership, exact sizes (or the weighted bound), and
connectivity to the terminal inside each part. Adversarial reviewers tried to
construct partitions it wrongly accepts and failed.

### 5.2 Exhaustive enumeration

| Enumeration | Instances | Failures |
|---|---:|---:|
| All connected graphs on n ≤ 6, every terminal subset, every capacity composition | 9 061 | 0 |
| All connected graphs on n = 7 (853 graphs) | 136 339 | 0 |
| All digraphs on n ≤ 4, k ∈ {1,2} | 3 278 | 0 |

Every instance was solved by the reference solvers *and* by the C++ general and
weighted solvers with internal invariant assertions enabled. On the 1 518
directed instances where the flow-essential assignment condition fails, the
solvers report `precondition_failed` — never a wrong answer and never a claim
that no partition exists.

### 5.3 Differential and property testing

2 599 tests pass (33 files; Hypothesis property tests with `ci`, `dev` and
`thorough` profiles). The C++ primitives were fuzzed against the reference on
roughly 4 400 instances (identical tightest cuts, identical essential sets
after random operation sequences, identical criticality for every vertex and
terminal pair); the solvers on roughly 6 000 more, including a trace replay
that re-checks the flow-essential assignment condition with *exact* essential
sets after every recorded step. The seven core suites are clean under
AddressSanitizer and UndefinedBehaviorSanitizer.

### 5.4 The authors' counterexample

The compact-connectivity construction of the appendix is rebuilt vertex for
vertex (identical order, 2 160 arcs, identical capacity vectors for 1, 2 and 17
copies). We confirm independently that compact connectivity holds, that every
arc deletion breaks it, and that every pre-terminal contraction breaks it —
i.e. the naive contract/delete strategy really is stuck — while the paper's
condition still admits a witness, so the new algorithm solves it. The 17-copy
instance (n = 3 789, m = 33 264, k = 9) is solved in about a second.

### 5.5 Mutation testing of the harness

Six deliberate bugs were injected into the reference solver (skipped capacity
check, random cycle shift, illegal contraction, swapped vertices, stale
essential sets, dropped in-arcs). Each is caught by at least four of the six
test entry points, which is evidence that the suite tests behaviour rather than
agreement with itself.

## 6. Benchmarks

All figures below come from `benchmarks/results/*_v2.jsonl` (1 151 rows, every one checked by the
independent verifier: **0 invalid, 0 timeouts, 0 errors**). Medians over three seeds up to n = 2 000
and single runs above, 8 threads, on a 20-core aarch64 machine; each run is an isolated subprocess
carrying its own machine, commit and instance hash. Full tables: `docs/benchmarks.md` §5.

### 6.1 Runtime, general solver, k = 4

| family | n = 100 | n = 1 000 | n = 2 000 | n = 5 000 | n = 10 000 | exponent (n ≥ 500) |
|---|---:|---:|---:|---:|---:|---:|
| random 4-regular | 0.005 s | 0.19 s | 0.47 s | 2.25 s | 12.4 s | 1.75 |
| sparse k-connected | 0.004 s | 0.18 s | 0.40 s | 2.08 s | 12.1 s | 1.77 |
| random k-T DAG (general solver) | 0.005 s | 0.13 s | 0.38 s | 1.81 s | 8.0 s | 1.69 |
| Harary H_{4,n} (exactly k-connected) | 0.004 s | 0.53 s | 3.44 s | 30.5 s | 418 s | 2.76 |
| Erdős–Rényi (m ≈ n²/10) | 0.003 s | 0.13 s | 1.14 s | 28.1 s | 210 s | 3.09 in n ≈ 1.5 in m |

The DAG solver on the same DAG instances: 0.0003 s at n = 1 000 and 0.0026 s at n = 10 000, with a
fitted exponent of **1.00–1.13** over the grid to n = 10⁶ — the near-linear behaviour the paper
predicts, confirmed empirically.

### 6.2 Largest instances solved

| algorithm | instance | n | arcs | time | peak memory |
|---|---|---:|---:|---:|---:|
| DAG solver | layered DAG | 1 000 000 | 4 999 364 | 0.28 s | — |
| general | Erdős–Rényi | 10 000 | 10 000 610 | 210 s | 2 864 MB (2 574 of it the instance) |
| general | Harary H_{4,10000} | 10 000 | 39 984 | 418 s | 2 794 MB |
| weighted | Erdős–Rényi | 10 000 | 10 000 610 | 211 s | — |
| reference (pure Python) | Erdős–Rényi | 100 | 1 250 | 84 s | — |
| MILP oracle | Erdős–Rényi | 20 | 162 | 0.017 s | — |
| brute force | Erdős–Rényi | 10 | 53 | 0.0001 s | — |

### 6.3 Against the baselines

At n = 100, the largest size the pure-Python reference reaches in reasonable time, the C++ core is
**1 500–28 000× faster** (Erdős–Rényi 84.3 s → 0.003 s; Harary 9.0 s → 0.0035 s; random regular
15.7 s → 0.0049 s). Brute force is limited to about n = 12 and the MILP to about n = 40; past that the
evidence is the independent verifier rather than a second solver.

### 6.4 Effect of the optimization stage

Paired on identical instance bytes against the committed baseline:

| family | speed-up (geometric mean, n ≥ 500) | memory |
|---|---:|---|
| Harary, general | **4.94×** (2.94–8.73×) | 86–229× lower; n = 2 000 went 56.5 GB → 247 MB |
| Harary, weighted | **10.85×** (4.40–39.5×) | same |
| DAG solver | **2.9–3.9×** (unintended; see §7) | unchanged |
| random regular / sparse / Erdős–Rényi | 1.0–1.6× | 2.6× lower where it was not already small |

Harary at n = 5 000 and 10 000 was impossible before (it needed 56 GB at n = 2 000) and is now routine.

### 6.5 Peak memory, measured honestly

While writing this report we found that `ru_maxrss` is inherited across `fork`, so a benchmark worker
spawned from a driver holding a 10⁷-arc instance reported the driver's footprint as its own (a child
of a 3 GB parent reports 2 879 MB when its true peak is 18 MB). Both the runner and the Python API now
read `VmHWM` from `/proc/self/status`. Corrected direct measurements at n = 10 000, separating the
instance's own footprint from the solver's:

| instance | arcs | instance | peak total | solver's share |
|---|---:|---:|---:|---:|
| Erdős–Rényi | 10 000 610 | 2 574 MB | 2 864 MB | ~290 MB |
| Harary | 39 984 | 64 MB | 2 794 MB | ~2 730 MB |
| sparse k-connected | 41 984 | 65 MB | 144 MB | ~79 MB |
| random 4-regular | 39 984 | 65 MB | 102 MB | ~37 MB |

On dense inputs the memory is the *input*, not the algorithm. On sparse high-diameter inputs it is the
flow certificates, which is the same structural fact that makes Harary the slowest family.

### 6.6 Where the time goes

Operation counts are exactly linear in n — contractions and full max-flow computations are both
n − k, deletions 2.5–3.5 n, cycle shifts about n — so every super-linear trend is per-operation cost.
After the optimization stage that cost is concentrated in one place: re-validating the flow
certificates that a deletion touches, which is 67–93 % of a sparse run. On dense inputs the picture
inverts: the certificate-guided fast paths do all the work and the paper's cycle-shift machinery is
never invoked at all.


## 7. Research results

Recorded in `RESEARCH_NOTES.md`, separated into proved, conjectured, empirical
and refuted.

**Proved.**

* **P1 — the DAG algorithm runs in O(n + m).** The paper gives `O(m log n)`
  using a heap per terminal. Because every vertex pushed after popping `p`
  precedes `p` in the topological order, each heap behaves exactly as a
  depth-first stack over position-sorted in-lists, so the heap can be replaced
  by stacks with pointers. The contraction sequence is identical, so the
  paper's correctness proofs carry over unchanged. Implemented and verified to
  produce byte-identical output to the heap version.
* **P2 — certified-subset essential sets suffice**, which is what licenses
  re-validating only the certificates that a deletion actually touches.
* **P3 — sparse-certificate preprocessing** is sound for the initial
  certificates on dense undirected inputs.
* **P4 — deletion-aware routing is exact.** Choosing a different maximum flow
  cannot change terminal connectivity, the tightest minimum cut (it is the
  intersection of all minimum cuts, hence unique) or the essential sets, so
  every decision the algorithm makes is unchanged.

**Refuted, with the counter-measurement.** C1 conjectured that routing
certificates through the matched pre-terminals would make most deletions free
and leave the running time dominated by the initial certificate computation.
Implemented exactly as stated and measured: on Harary graphs the number of
certificates repaired per deletion moves from 268.4 to 268.1, and the run is
*slower*. The flaw is structural: the fan lemma controls how paths *arrive* at
the chosen set, while the cost is dominated by paths that *transit* a
terminal's neighbourhood en route elsewhere; in a graph whose degree equals `k`
there is no alternative route, so no routing can avoid the deletable arcs. What
survives is weaker and is kept as an option: where the graph has routing slack,
the same mechanism removes 41 % of the augmentations and 45 % of the repairs
and runs 13–19 % faster.

**Open.** C2 (an amortized bound below the paper's) is untouched. C3 (a GPU
kernel for the certificate phase) remains unprototyped and is now less
attractive: after the optimization stage the certificate phase is 0.6–14 % of a
sparse run, so accelerating it cannot help much.

## 7.1 Optimizations that worked

| Optimization | Effect |
|---|---|
| O1 — skip arcs no certificate uses | most deletions cost nothing; on dense inputs the cycle-shift machinery never runs at all |
| O2 — warm-started re-validation | one bounded search instead of a full recomputation per affected vertex |
| O3 — no recomputation after contraction | essential sets provably unchanged |
| O5 — greedy contraction | 32–100 % of contractions avoid `ShiftAssignment` entirely |
| Exact user index | memory 228× lower (56 GB → 248 MB at n = 2 000) |
| O(1) path splicing | contraction translation 8.3 s → 0.04 s at n = 1 000 |
| Persistent worker pool | ~2× on the many small parallel regions |
| numpy arc path | instance handling for 10⁷ arcs reduced to 2.6 s of a 189 s run |

## 7.2 Optimizations that failed

* **Gomory–Hu / cactus reuse across the run.** Not applicable: these model edge
  cuts of undirected graphs, while the algorithm needs vertex cuts between a
  vertex and a terminal set in a digraph that stops being symmetric after the
  first deletion.
* **Skipping the cut refresh after a terminal removal when connectivity is
  unchanged.** Wrong: new essential terminals can appear (§13.3).
* **Deletion-aware routing as a default** (C1 above).
* **A periodic re-routing pass.** It reaches a better routing but costs far
  more than it saves (on sparse k-connected n = 10 000, +3.6 s of re-routing
  against no reduction in evaluation time).
* **Templating the augmenting search on the routing mode.** Correct but it
  slowed the default path through worse code generation; fixed by isolating the
  two instantiations behind non-inlined entry points. Caught by the reviewer,
  not by the author.

## 8. Limitations and open problems

1. The worst-case bound is the paper's; nothing here improves it. The measured
   growth on sparse families is between n^1.7 and n^3.3 depending on structure.
2. The dominant cost is now re-validating certificates after deletions
   (67–93 % of a sparse run). C1 shows routing alone does not fix it. The open
   question is whether the *number* of affected certificates can be bounded
   better, which is what C2 asks.
3. Exactly-`k`-connected, high-diameter graphs (Harary) remain the hardest
   family: every certificate is forced through the same bottlenecks.
4. Precondition checking is expensive (one max-flow per vertex), so `"auto"`
   only cross-checks below n = 400; the solvers always detect a failing
   condition themselves.
5. The MILP oracle is practical to about n = 40 and brute force to about n = 12;
   beyond that the evidence is the independent verifier, not a second solver.
6. No GPU path (C3).

## 9. Highest-priority next improvements

1. **Bound the repair count.** Measure which certificates are repaired
   repeatedly; if a small set of "hot" vertices dominates, recomputing them
   lazily in bulk may beat repairing them one deletion at a time.
2. **Batch the deletions.** The algorithm deletes one arc at a time by
   construction, but O1 already shows many deletions are independent; a
   provably safe batch step would cut the number of re-validation rounds.
3. **Finish the family sweep** for the pure-Python reference (it stalled on the
   slowest directed grids) so the reference-versus-core comparison covers every
   family.
4. **Make the wall-clock assertions portable** so continuous integration on
   slow shared runners is not measuring the runner.
5. **Prototype C3** only if a use case appears with n ≥ 10⁵ where the
   certificate phase dominates.

## 10. Reproducing everything

```bash
pip install -e ".[dev]"                       # CMake ≥ 3.20, a C++20 compiler
pytest -m "not slow" -n auto --hypothesis-profile=ci     # the whole suite
python scripts/run_exhaustive.py --max-n 6 --jobs 16     # exhaustive enumeration
python -m benchmarks.runner scaling --repeat 3           # benchmarks
python -m benchmarks.plot && python -m benchmarks.report # figures + dashboard
python examples/render_all.py                            # visual traces
docker build -t glsolver . && docker run --rm glsolver solve --help
```

Every random choice is seeded; every benchmark row carries the machine, the
git commit, the thread environment and a hash of the instance file.
