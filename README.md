# glsolver — exact polynomial-time Győri–Lovász partitioning

`glsolver` is an independent, from-scratch implementation of the first polynomial-time algorithm
for the Győri–Lovász theorem (Hajiaghayi, JafariRaviz, Kaviani, Mohammadkhani, *Breaking the
Exponential Barrier*, arXiv:2608.30945, 2026). Given a `k`-connected graph, `k` terminals and `k`
prescribed part sizes, it returns a partition of the vertices into `k` connected parts of exactly
those sizes, plus a spanning in-arborescence per part that certifies the connectivity. It also
covers Lovász's directed formulation, the weighted generalization with its unavoidable `w_max − 1`
slack, and the near-linear DAG special case. It ships a pure-Python reference implementation
following the paper line by line, an optimized C++20 core, an independent verifier sharing no code
with any solver, exact brute-force and MILP oracles, a trace visualizer and a benchmark suite.

Status: version 0.1.0, MIT, Python ≥ 3.10; the C++ core is optional (every algorithm also exists
in pure Python). 2 469 tests across 31 suites pass (pytest + Hypothesis). Every connected graph on
`n ≤ 7` (145 400 instances) and every digraph on `n ≤ 4` with `k ∈ {1,2}` (3 278 instances) is
solved by the reference *and* the C++ solvers with debug assertions, 0 failures. Final benchmark
tables: see `docs/benchmarks.md`.

## 1. The Győri–Lovász theorem in plain language

**Győri 1976, Lovász 1977.** Take a graph in which you cannot disconnect the remaining vertices by
deleting fewer than `k` vertices (a *`k`-connected* graph). Pick any `k` distinct vertices `t_1,
…, t_k` and any positive integers `n_1, …, n_k` summing to `|V|`. Then `V` splits into groups
`V_1, …, V_k` where `V_i` contains `t_i`, has exactly `n_i` vertices and induces a connected
subgraph: `k` connected sub-networks of *any* prescribed sizes around prescribed gateway nodes,
whatever the topology. The hypothesis is tight twice over — `k`-connectivity is necessary for the
statement to hold for all choices (Győri 1981), and without it the problem is NP-complete even for
equal sizes (Dyer–Frieze 1985).

**Lovász's directed version.** For a digraph `G` and terminals `T = {t_1, …, t_k}` such that every
other vertex has `k` paths to `k` distinct terminals sharing nothing but their start (`G` is
*`k`-`T`-connected*), any `n_i` summing to `|V|` are realized by disjoint in-arborescences with
root `t_i` and `n_i` vertices covering `V`. The undirected theorem is the case where each edge
becomes two opposite arcs.

**Why it was hard.** Győri's proof is constructive but explores "cascades" whose number can be
exponential; Lovász's uses algebraic topology and yields no algorithm. Polynomial algorithms were
known only for `k ≤ 4` and special graph classes; the problem sits in PLS and was widely suspected
PLS- or PPAD-hard. References: `docs/theorem.md`.

## 2. What changed in the 2026 paper

**The flow-essential assignment.** A terminal `t` is *essential* for a non-terminal `v` if
deleting `t` lowers the number of vertex-disjoint `v → T` paths ([Def 4.1]) — equivalently, if `t`
lies in the separator of the *tightest* minimum cut separating `v` from `T` ([Lem 4.1]), so one
max-flow per vertex yields `κ(v)` and `Ess(v)` at once ([Prop 4.2]). The **Flow-Essential
Assignment Condition** (FEAC, [Def 5.1]) asks for a map `φ: V\T → T` with `φ(v) ∈ Ess(v)` and
`|φ⁻¹(t)| = c_t` exactly; a witness comes from one bipartite max-flow. FEAC is strictly weaker than
`k`-`T`-connectivity and, crucially, *maintainable* — the algorithm's operations preserve it, while
`k`-`T`-connectivity and compact connectivity do not.

**Three operations.** `GLPartition` [Alg 1] repeatedly fires exactly one of: (i) remove a terminal
of capacity `0`, closing its part ([Lem 7.2]); (ii) contract a pre-terminal of out-degree one into
its terminal, growing that part and leaving every `Ess` literally unchanged ([Lem 7.3]); (iii)
delete an arc *non-critical* for the current witness ([Lem 7.5]). Each round removes a vertex or
an arc, so there are at most `n + m` rounds.

**Cycle shifts.** When no arc is deletable outright, `ShiftAssignment` [Alg 2] takes a matching
from pre-terminals onto `T` ([Lem 7.8]), picks one secondary arc per matched pre-terminal, and —
while all are critical — builds a reassignment graph on `T` in which every terminal has in-degree
one, hence a directed cycle. Shifting the witness along it keeps capacities and essentiality
intact and strictly decreases the potential `Φ(φ) = Σ_v ξ_v(φ(v)) ≤ k|V\T|` ([Lem 7.11], via the
cut-lattice transfer lemma [Lem 7.12]), so a deletable arc appears after at most `k|V\T|` shifts.

**Weighted and DAG results.** With integer weights the potential can be exponential in the encoding
length, so [Alg 3] replaces the shift loop by one **min-cost split assignment** ([Prop 5.4]) under
FESAC [Def 5.3], plus a fourth operation, `RoundAndRemove` [Alg 4], for when the saturating
matching is missing; the output obeys `Σ_{v ∈ V_t\T} w_v ≤ c_t + w_max − 1`, a provably tight bound.
On DAGs no flow machinery is needed: `k`-`T`-connectivity means "every non-terminal has out-degree
≥ `k`" ([Lem 9.1]), and [Alg 5] contracts the topologically earliest pre-terminal, `O(m log n)`.

## 3. Supported formulations and preconditions

| formulation | input | output | precondition |
|---|---|---|---|
| classical undirected GL | undirected `G`, terminals `t_i`, sizes `n_i`, `Σ n_i = n` | `V_i ∋ t_i`, `|V_i| = n_i`, `G[V_i]` connected | `G` is `k`-vertex-connected (sufficient) |
| Lovász directed | digraph `G`, terminals, capacities `c_i ≥ 0`, `Σ c_i = |V\T|` | `|V_i| = c_i + 1`, `G[V_i]` connected **to** `t_i` (spanning in-arborescence) | `k`-`T`-connected, or the weaker FEAC [Def 5.1] |
| weighted | plus integer weights `w_v ≥ 1`, `Σ w_v ≤ Σ c_t` | `Σ_{v ∈ V_t\T} w_v ≤ c_t + w_max − 1` | `k`-`T`-connected, or the weaker FESAC [Def 5.3] |
| DAG | acyclic `G`, optional weights | as above; exact for unit weights with `Σ c = |V\T|` | acyclic, every non-terminal has out-degree ≥ `k` |

Undirected input is reduced to the directed formulation by replacing each edge with two opposite
arcs, dropping arcs leaving terminals (a terminal is the root of its part) and setting
`c_i = n_i − 1`; self-loops and parallel arcs are normalized away on load.

**`precondition_failed` never means "no partition exists".** The theorem gives a *sufficient*
condition. When no FEAC/FESAC witness exists — or a DAG has a non-terminal of out-degree `< k`, or
a vertex cannot reach `T` — the solvers return `status == "precondition_failed"` with a message
naming the offending vertex or Hall-deficient set, and stop. Only the exact oracles (`bruteforce`,
`ilp`) can return `"infeasible"` and prove non-existence: in the exhaustive digraph enumeration,
48 of the 1 518 non-FEAC instances still admit a partition.

## 4. Installation

The repository's virtualenv is a **uv-managed CPython 3.12** (`uv 0.12.3`), so the usual entry
point is

```bash
cd /path/to/gs && source .venv/bin/activate
pip install -e ".[dev]"
```

Building the C++ core needs **CMake ≥ 3.20**, **Ninja** and a **C++20** compiler (`g++` ≥ 10 or
clang); `scikit-build-core` and `pybind11` come from the build backend. Without one the install
still succeeds and `algorithm="auto"` uses the reference solvers (`glsolver.core_available()` says
which). Extras: `ilp` (SciPy/HiGHS oracle), `viz` (matplotlib, pillow, imageio), `bench`, `test`,
`dev`.

```bash
GLCORE_DEBUG_ASSERTS=ON pip install -e ".[dev]"   # compile invariants A1–A8 into the core
GLCORE_SANITIZE=ON      pip install -e ".[dev]"   # ASan/UBSan; LD_PRELOAD note in docs/implementation.md
GLSOLVER_NO_CORE=1 pytest ...                     # ignore an installed core at runtime

docker build -t glsolver .                        # CPU-only image: builds the core, installs [dev]
docker run --rm glsolver solve --help
docker run --rm -v "$PWD:/data" glsolver solve /data/instance.json --stats
```

## 5. Thirty-second example

```python
import networkx as nx
from glsolver import partition

G = nx.circulant_graph(12, [1, 2])              # 4-connected
res = partition(G, terminals=[0, 3, 6, 9], sizes=[3, 3, 3, 3])

res.summary()      # 'general: status=ok n=12 m=32 k=4 runtime=0.0004s verifier=VALID'
res.parts          # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
res.valid          # True — verdict of the independent verifier
res.certificate["parents"]   # {1: 0, 2: 0, 4: 3, 5: 3, 7: 6, 8: 6, 10: 9, 11: 9}
```

`partition` is the classical undirected front end; `glpartition` is the full API (directed input,
`capacities=`, `weights=`, `algorithm=`, `trace=`, `debug=`, `seed=`, `verify_preconditions=`),
documented in `docs/api.md`. The `glsolve` CLI covers the same ground plus generation, inspection,
visualization and benchmarking:

```bash
glsolve generate harary --n 200 --k 4 --seed 1 -o harary200.json
#   wrote harary200.json: n=200 m=784 k=4 directed=False weighted=False
glsolve inspect harary200.json --preconditions     # degrees, is_dag, FEAC, auto algorithm
glsolve solve harary200.json --stats -o solution.json
#   general: status=ok n=200 m=784 k=4 runtime=0.0331s verifier=VALID
#   part 0 (terminal 16, size 50): 0 1 2 3 ...
glsolve verify harary200.json solution.json        # VALID  (independent re-check)
glsolve visualize examples/curated/paper_running_example.json --format html -o paper.html
glsolve benchmark smoke --limit 4                  # or: python -m benchmarks.runner smoke
```

`glsolve solve` also takes plain edge lists (`glsolve graph.edgelist --terminals 0,10,27,81
--sizes 25,25,25,25 [--directed]`; formats in `docs/instance_format.md`), writes a replayable trace
with `--trace trace.json`, and exits `0` on a verified partition, `2` on a non-`ok` status, `3` if
the verifier rejects the output.

## 6. Visualization

`glsolver.viz` renders the trace events of `docs/paper_notes.md §14`. Solvers emit the *full*
state needed to replay a run and the viewer imports no solver internals, so a C++ trace is drawn
exactly as the C++ solver saw it. Each frame draws every non-terminal as a circle whose sectors
are its essential terminals `Ess(v)` and whose ring is its assigned terminal `φ(v)` (split
proportionally to `ψ` when weighted), contracted vertices in their part's colour with the
arborescence arc thick, deleted arcs greyed out, and — inside a `ShiftAssignment` call — the
matching arcs, secondary arcs `e_i`, criticality table and reassignment inset with the chosen cycle
green and `Φ` before → after each shift. Vertices whose `Ess`/`κ` just changed glow, cuts appear as
translucent `L`/`S`/`R` hulls, and a side panel tracks capacities, part sizes and the operation
counters. Layouts are deterministic per instance; frames are byte-stable across runs.

Seven curated instances in `examples/curated/` exercise every operation in readable frames: the
paper's running example (2 cycle shifts), its Fig. 6 essential-set example, its Fig. 1 contraction
counterexample (every contraction breaks `k`-`T`-connectivity, so 6 deletions must come first), a
Harary instance with 3 cycle shifts, the `§13.3` gadget where removing a zero-capacity terminal
creates a *new* essential terminal, a weighted instance that triggers `RoundAndRemove`, and a small
DAG. Each records the trace properties `examples/render_all.py` and `tests/test_viz.py` assert.

```bash
python examples/render_all.py                     # all curated examples → examples/output/
glsolve visualize INSTANCE.json --format gif      # svg | png | pdf | html | gif | mp4
glsolve visualize INSTANCE.json --format svg --step 9 -o step9.svg
```

Conventions, Python API and the state-replay contract: `visualization/README.md`.

## 7. Performance

**Architecture.** `reference/glref/` is a literal transcription of the paper: first-found choices
everywhere, full recomputation of all essential sets after every terminal removal and arc deletion,
an explicit criticality table. It is pure Python, deliberately slow, and is the ground truth for
everything else. `src/glcore/` is a C++20 core (pybind11 → `glsolver._core`): a mutable digraph
with lazy vertex deletion and `orig_head` bookkeeping, a CSR vertex-split flow network, an
`EssentialOracle` holding a per-vertex max-flow certificate and tightest cut, a `CriticalityOracle`,
Hopcroft–Karp matching and a min-cost flow for split assignments. `glsolver.api.glpartition`
dispatches: `auto` takes the DAG solver for an acyclic instance with out-degree ≥ `k`, else the
weighted core for weighted input, else the general core.

The core computes the same objects faster *and* may reorder the paper's operations, because the
induction behind [Thm gl-partition-correctness] only needs FEAC in the graph handed to the next
step (`paper_notes §13.1`): any arc set `D` with `φ(v) ∈ Ess_{G\D}(v)` for all `v` may be deleted
at once. It also follows a *certified-subset* discipline (`RESEARCH_NOTES.md` P2): the stored
`Ess(v)` is always a subset of the truth containing `φ(v)`, while every *decision* — criticality of
`(v, φ(v))`, the terminal adopted by a cycle shift — uses an exact cut, and `ShiftAssignment`
remains the exact fallback guaranteeing progress. The optimizations, each with a correctness
argument in `docs/optimizations.md` (which also records the shortcuts that are *not* valid):

* **O1 — unused-arc shortcut.** If a vertex's stored flow uses no arc of a set `D`, its `κ` is
  unchanged and `Ess(v)` can only grow ([Lem 4.3] generalized), so `D` is deleted for free.
* **O2 — warm-started criticality.** Testing arc `e` against `v` drops the one flow path through
  `e` and runs a single augmenting-path search in `G\e`, `O(n+m)` instead of `O(k(n+m))`; the
  resulting flow becomes the new certificate if `e` is deleted.
* **O3 — no recomputation after contraction.** Contracting an out-degree-one pre-terminal leaves
  `κ` and `Ess` literally unchanged ([Lem 7.3]); only stored paths through it are translated.
* **O4 — terminal removal.** Drop the path ending at the removed terminal, re-augment once,
  rebuild the tightest cut with one reverse reachability search per vertex — needed for *every*
  vertex, since new essential terminals can appear (`§13.3`), at most `k` times.
* **O5 — greedy contraction.** Make a pre-terminal `p` contractible by deleting `out(p) \ {(p,
  φ(p))}` in one step, checked by O1/O2 against `G\D`; a credit-based throttle and per-vertex
  backoff keep failed attempts cheap where this does not pay.
* **O6 — lazy ShiftAssignment.** The criticality table is built only for pairs that can be true
  (`v`, `e_i` with `e_i ∈ F_v`) and reused across shifts, which change only pairs already in it.

O7–O10 (batched secondary-arc deletion, OpenMP-parallel per-vertex oracle, bitset essential sets,
DAG heap/stack variants) are in the same document; the `O(n+m)` DAG variant is proved in
`RESEARCH_NOTES.md` P1 and yields the heap version's contraction sequence.

**Measured** (final sweep, `RESEARCH_NOTES.md` §E6 and `docs/benchmarks.md` §5; 20-core aarch64,
8 threads, release build; medians over three seeds to `n = 2000`, single runs above; every partition
accepted by the independent verifier):

| family (`k = 4`) | `n = 1000` | `n = 2000` | `n = 5000` | `n = 10000` | growth |
|---|---:|---:|---:|---:|---:|
| random 4-regular | 0.19 s | 0.47 s | 2.25 s | 12.4 s | `n^1.75` |
| sparse `k`-connected | 0.18 s | 0.40 s | 2.08 s | 12.1 s | `n^1.77` |
| Harary `H_{4,n}` (exactly `k`-connected) | 0.53 s | 3.44 s | 30.5 s | 418 s | `n^2.76` |
| Erdős–Rényi (`m ≈ n²/10`) | 0.13 s | 1.14 s | 28.1 s | 210 s (`m = 10^7`) | `m^1.5` |
| DAG solver on `k`-`T`-connected DAGs | 0.3 ms | 0.7 ms | 1.6 ms | 2.6 ms | `n^1.0` |

The DAG solver takes **0.24–0.70 s** on `n = 10^6`, `m ≈ 5·10^6` single-threaded. The official
counterexample is solved in 0.04 s (1 copy, `n = 333`) and 1.8 s (17 copies, `n = 3789`); the
pure-Python reference needs **900 s** for the 1-copy instance. At `n = 100`, the largest size the
reference reaches in reasonable time, the core is **1 500–28 000×** faster on identical instances.

Peak memory, measured per process with `VmHWM` (`ru_maxrss` is inherited across `fork` and silently
reported a parent's footprint — see §E6): `n = 10000` costs 102 MB on random regular, 144 MB on
sparse `k`-connected and 2.79 GB on Harary, while the 10-million-arc Erdős–Rényi instance costs
2.86 GB of which 2.57 GB is the instance itself and only ≈290 MB the solver.

Two structural observations fall out: on dense inputs the certificate-guided path (O1 + O5) alone
finishes the run and the cycle-shift machinery is never invoked (zero `ShiftAssignment` calls on the
Erdős–Rényi instance), whereas on sparse regular graphs the greedy attempt usually fails and the
exact fallback dominates. The full sweep is run by
`scripts/run_benchmarks.sh`; final numbers **see `docs/benchmarks.md`**, dashboard at
`benchmarks/results/dashboard.html`.

## 8. Mathematical guarantees

**Exactness.** Nothing here is an approximation or a heuristic: unweighted output satisfies
`|V_i| = c_i + 1` exactly, weighted output the provably tight `c_t + w_max − 1` bound. On
unit-weight instances with `Σ c = |V\T|` rounding can never trigger (`§13.6`), so the weighted
algorithm is an exact alternative unweighted solver — tests assert `roundings == 0`.

**Independent verifier.** `glsolver.verify` is pure Python, uses only the standard library and BFS
on the *original* input, and imports no solver, oracle or reference module. It checks part count,
exact cover of `0..n−1`, id ranges, `t_i ∈ V_i` with no foreign terminal, exact sizes (or the
weighted bound, listing parts that exceed `c_i`), and connectivity — reachability to `t_i` inside
the part for directed input, and for undirected input *both* that check and undirected
connectivity in the original edge set, which must agree. All problems are collected, not just the
first.

**Debug assertions A1–A8** (`paper_notes §7.2`; `GLCORE_DEBUG_ASSERTS=ON` in the core,
`debug=True` in the reference): A1 `φ` is a witness (essentiality *and* exact capacity counts)
after every operation; A2 the matching is valid and saturates `T`; A3 each secondary arc differs
from its matching arc and is present; A4 the chosen `v_i` satisfies [Lem 7.9] and [Lem 7.10]; A5
`Φ` strictly decreases at every shift; A6 `φ` survives each arc deletion; A7 `Ess` is unchanged by
contraction; A8 `φ` survives each terminal removal. Not decoration: the mutation matrix in
`tests/test_harness_review.py` shows injected bugs (a stale `Ess` after terminal removal,
contraction of out-degree-2 pre-terminals) that still produce a *valid partition on every
enumerated instance* and are caught only by a violated invariant.

**What the tests check.** Exhaustive small instances; Hypothesis property tests over seven
strategies (`ci`/`dev`/`thorough` profiles) asserting that every applicable backend solves every
drawn instance and that the certificate uses original arcs; honesty tests on arbitrary digraphs (a
theorem solver answers only `"ok"` or `"precondition_failed"`, never `"infeasible"`, never a
crash); core-vs-reference agreement; the two exact oracles against each other; a regression store
re-solved on every run; a per-call `SIGALRM` bound so a non-terminating solver bug fails the suite
instead of hanging it; and reviews auditing the harness itself.

**Exhaustive and adversarial coverage.** Beyond the exhaustive enumerations quoted at the top, the
C++ primitives were differentially fuzzed on about 4 400 seeded instances against the reference
(identical tightest cuts), and the solvers on about 6 000 instances with **all** option
combinations, including trace replay against exact essential sets. The paper's compact-connectivity
counterexample is reproduced exactly (1, 2 and 17 copies) and solved; ASan/UBSan runs with debug
assertions are clean and run in CI. Details: `docs/verification.md`.

## 9. Limitations

* **Worst-case complexity is polynomial, not small.** [Alg 1] runs `≤ n + m` rounds, each with up
  to `k|V\T|` cycle shifts over a criticality table costing `k·|V\T|` max-flows; the weighted bound
  is `O(nkm^{2+o(1)} + m(nk)^{1+o(1)} log(n w_max) log n)`. The optimizations cut constants and
  usually the flow count, but prove no better bound (open conjectures C1/C2, `RESEARCH_NOTES.md`).
* **Memory of the stored certificates.** The essential oracle keeps, per live non-terminal, a
  max-flow certificate (≤ `k` paths) and — when its cut is exact — the `L`/`S`/`R` side of *every*
  vertex: `Θ(n²)` bytes for the sides alone. The general solver goes memory-bound before it goes
  time-bound on large sparse inputs; the DAG solver has no such structure.
* **Trace mode is for small instances.** Every event carries the state needed for replay, an
  `essential` event follows each deletion, removal and rounding, and `record_cuts` adds per-vertex
  cut sides. Traces target curated examples and debugging, not benchmark sizes.
* **The oracles do not scale.** `bruteforce` (backtracking over connected sets) and `ilp` (a
  single-commodity-flow MILP via HiGHS) alone can prove non-existence, and both are exponential in
  practice; the harness gives them 60 s, treats `"timeout"` as inconclusive, and the benchmark
  grids cap them around `n ≤ 12` and `n ≤ 40`.
* **Precondition cross-checks are expensive.** `glsolver.preconditions` decides
  `k`-`T`-connectivity and FEAC/FESAC by definition with NetworkX max-flows (one per non-terminal
  plus a bipartite flow), so `verify_preconditions="auto"` and `glsolve inspect --preconditions`
  run them only for the oracles and only for `n ≤ 400`. The solvers' own checks are cheap.
* **No GPU path yet.** The one embarrassingly parallel phase (a unit-capacity flow per vertex) is
  OpenMP over CPU cores; a batched GPU kernel is conjecture C3 in `RESEARCH_NOTES.md` and has not
  been prototyped. The sequential main loop is not a GPU workload.

## 10. Citation

Cite the paper this implements — and, if the implementation itself matters for your results, the
software entry too:

```bibtex
@misc{hajiaghayi2026gyorilovasz,
  title  = {Breaking the Exponential Barrier: The First Polynomial-Time Algorithm for the
            {G}y{\H{o}}ri--{L}ov{\'a}sz Theorem},
  author = {Hajiaghayi and JafariRaviz and Kaviani and Mohammadkhani},
  year   = {2026}, eprint = {2608.30945}, archivePrefix = {arXiv}, primaryClass = {cs.DS},
  url    = {https://arxiv.org/abs/2608.30945}
}

@software{glsolver2026,
  title  = {glsolver: an independent implementation of the polynomial-time
            {G}y{\H{o}}ri--{L}ov{\'a}sz algorithm},
  author = {{glsolver contributors}}, year = {2026}, version = {0.1.0},
  note   = {Independent implementation of arXiv:2608.30945; not affiliated with its authors}
}
```

## 11. Relationship to the original paper

`docs/paper_notes.md` is the single source of truth for this repository: every non-trivial
function in `reference/` and `src/` cites a label from it (`[Def X]`, `[Lem Y]`, `[Alg Z]`) in its
docstring or header comment, and its §12 is a lemma → function map kept in sync with the code. Its
§13 records every point where the paper is underspecified, with the derivation used here and the
tests validating it — cycle detection in the reassignment graph, recovery of original arcs after
contraction, terminal removal creating new essential terminals, the choice of
`v_i`/matching/secondary arcs, the DAG terminal-selection policy, and the `κ(v) = 0`, `k = 1` and
`k = n` edge cases.

The appendix's relaxed conditions (local and compact connectivity) are implemented as
*recognizers* only; Győri's exponential cascade argument (`Thm A.3`) is not implemented as a
solver.

## 12. Independence and disclaimer

This is an **independent implementation**. It is not affiliated with, endorsed by, or reviewed by
the authors of arXiv:2608.30945, and it is not their reference implementation. Their supplementary
repository (`mahdi-jfri/Gyori-Lovasz-Codes`, MIT) contains **only** the appendix's
compact-connectivity counterexample, not the algorithm; that counterexample is reproduced
independently here (`reference/glref/counterexample.py`) and used as a regression test, and its
claims — compact connectivity holds, every arc deletion and every pre-terminal contraction breaks
it — are re-checked by this code rather than assumed.

Everything in `docs/` is our reading of the paper, not the paper itself. Any errors in the
interpretation, the implementation, the proofs in `RESEARCH_NOTES.md` or the measurements are
ours, not the authors'.

## Repository map

```
src/glsolver/      API, CLI, instance/IO, independent verifier, preconditions, generators,
                   oracles (bruteforce, ILP), testing helpers, visualization (viz/)
src/glcore/        C++20 core → glsolver._core: graph, flow, essential/criticality oracles,
                   matching, min-cost flow, the three solvers
reference/glref/   pure-Python reference implementation of the paper (ground truth)
tests/             pytest suites: exhaustive, property-based, per-module adversarial reviews
benchmarks/        runner, configs, plotting, report/dashboard, machine provenance
examples/          curated instances (curated/) and render_all.py
visualization/     drawing conventions and API reference for glsolver.viz
scripts/           run_exhaustive.py, run_benchmarks.sh
regression/        permanently stored regression instances (JSON)
```

| document | contents |
|---|---|
| `docs/paper_notes.md` | master specification: paper → implementation map, §13 underspecified points, §14 trace model |
| `docs/theorem.md` | the Győri–Lovász theorem and its history, in plain language |
| `docs/algorithm.md` | [Alg 1] + [Alg 2]: the three operations, cycle shifts, the running example, certificates |
| `docs/flow_essential_assignment.md` | essential terminals, the cut lattice, FEAC/FESAC, where they sit, proof of [Lem 7.12] |
| `docs/weighted_algorithm.md` | [Alg 3] + [Alg 4]: min-cost split assignment, `RoundAndRemove`, the `w_max − 1` slack |
| `docs/dag_algorithm.md` | [Alg 5], the out-degree characterization, the `O(n+m)` stack variant, policies |
| `docs/optimizations.md` | O1–O10 with correctness arguments, and the shortcuts that are *not* valid |
| `docs/implementation.md` | layers, dispatcher, determinism, debug and sanitizer builds |
| `docs/verification.md` | verifier, oracles, exhaustive and property tests, regression store, results |
| `docs/benchmarks.md` | what is measured, methodology, configurations, reproduction, result tables |
| `docs/api.md` | `glpartition`/`partition` contract, `GLResult` fields, precondition semantics |
| `docs/instance_format.md` | canonical instance/solution JSON and the edge-list format |

`RESEARCH_NOTES.md` holds results obtained here rather than taken from the paper: proved (P1
`O(n+m)` DAG algorithm, P2 certified-subset essential sets, P3 sparse preprocessing), conjectures
C1–C3, and the first measurements.

## Reproducibility

```bash
cd /path/to/gs && source .venv/bin/activate

pytest -q -m "not slow"                      # the whole suite (dev profile)
pytest -q --hypothesis-profile=ci            # fast; HYPOTHESIS_PROFILE=thorough for 2000 examples
pytest -q tests/test_edge_cases.py --runslow -k counterexample

python scripts/run_exhaustive.py --max-n 7 --jobs 16    # 145 400 undirected + 3 278 directed
python scripts/run_exhaustive.py --max-n 6 --solver reference --oracles --jobs 8

python -m benchmarks.runner smoke            # must exit 0 before anything expensive
scripts/run_benchmarks.sh                    # all configs, then plots + dashboard
python -m benchmarks.machine --info          # hardware/software provenance

docker build -t glsolver . && docker run --rm glsolver solve --help
ruff check src reference tests benchmarks scripts
```

Everything is seeded: instances come from `(family, n, k, capacity mode, seed, variant)` and are
cached with the file's SHA-256 plus a fingerprint of the generator sources, so a stale cache cannot
feed old instances to a new run; heuristic choices take a `seed=` and parallel loops reduce into
per-index slots; benchmark runs set `PYTHONHASHSEED=0`, pin and record the thread environment per
row, and isolate each `(instance, algorithm)` pair in its own subprocess. Environment variables:
`GL_SOLVER_TIME_LIMIT` (per-call bound, default 20 s), `GL_EXHAUSTIVE_FULL`,
`GL_EXHAUSTIVE_TALLY`, `HYPOTHESIS_PROFILE`, `GLSOLVER_NO_CORE`.

CI (`.github/workflows/ci.yml`) runs three jobs per push and pull request: the suite on Python 3.10
and 3.12 with ruff and the `ci` Hypothesis profile; a sanitizer job rebuilding the core with
`GLCORE_SANITIZE=ON` and `GLCORE_DEBUG_ASSERTS=ON` under ASan/UBSan; and a Docker build with a CLI
smoke test. Benchmark exit codes are part of the contract: `0` when every executed pair finished,
`2` when a pair crashed or timed out, `3` when any partition was rejected by the verifier —
`scripts/run_benchmarks.sh` aborts if the smoke grid is not green.
