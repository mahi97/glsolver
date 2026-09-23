# Research notes

Living document. Sections: **Proved**, **Conjectures**, **Empirical observations**,
**Failed ideas**. Each entry is dated and states its status explicitly. Paper
labels refer to `docs/paper_notes.md`.

---

## Proved

### P1. The DAG algorithm runs in O(n + m) with a stack instead of heaps (2026-09-21)

**Statement.** GLDAGPartition [Alg 5] with the *round-robin* or *first-active*
terminal policy can be implemented in `O(n + m)` time (the paper proves
`O(m log n)` with binary heaps, [Lem 9.6]); with the *max-residual* policy in
`O(n + m + n log k)`. The sequence of contractions is identical to the heap
version, so all correctness lemmas of §9 apply unchanged.

**Proof.** Fix a terminal `t_i` and its heap `H_i`. When the algorithm pops the
minimum unused entry `p` of `H_i`, it pushes the in-neighbours of `p`, all of
which precede `p` in the canonical topological order (`(u,p) ∈ E ⇒ u ≺ p`).
Every other unused entry of `H_i` is `⪰ p` (since `p` was the minimum). Hence the
next pop from `H_i` is the minimum *unused* in-neighbour of `p` if one exists.
Inductively, `H_i`'s pops follow a depth-first traversal of the reversed graph
from `V_i`, where at each vertex the unused in-neighbours are visited in
increasing `≺`-order, and where backtracking resumes at the deepest level that
still has unused pending entries: pending entries at a deeper level are all
smaller than pending entries at any shallower level (deeper entries are
`≺ p_j`, while the shallower pending entries were `⪰ p_j` when `p_j` was popped
and lists are scanned in sorted order). Therefore the minimum unused entry of
`H_i` is always the first unused element of the deepest non-exhausted level.
Implementation: sort every in-list once by position (a global counting sort,
`O(n + m)`); per terminal keep a stack of `(vertex, pointer into its sorted
in-list)`; "pop min" = advance the pointer of the top level past used vertices
(each arc's pointer position is passed at most once overall) or pop the level
when exhausted. The "used" test is `O(1)`. Total `O(n + m)`. Interleaving with
other terminals only marks vertices used, which the pointer scans skip.
The max-residual policy additionally needs a priority queue over `≤ k`
terminals, updated once per contraction: `O(n log k)`. ∎

Status: proved; implemented as `dag_partition(..., linear=True)` in the core;
tests assert identical output to the heap version; benchmarks compare both.

### P2. Certified-subset essential sets suffice for the algorithm (2026-09-21)

**Statement.** The unweighted algorithm remains correct if, instead of the exact
`Ess_G(v)`, the solver stores any *certified subset* `E(v) ⊆ Ess_G(v)` with
`φ(v) ∈ E(v)`, provided (a) criticality of an arc `e` for a pair `(v, t)` with
`t ∈ E(v)` is evaluated with the exact `Ess_{G\e}(v)` (one warm-started flow +
one reverse reachability), and (b) after a cycle shift `φ(v_i) := t_i`, `t_i` is
added to `E(v_i)`.

**Proof.** Witness validity only needs `φ(v) ∈ Ess_G(v)`, implied by
`φ(v) ∈ E(v) ⊆ Ess_G(v)`; under arc deletion `Ess` can only grow [Lem 4.3]
so subsets stay subsets. The non-criticality test for `(v, φ(v))` is exact by
(a). `t_i ∈ Ess_G(v_i)` after a shift holds by [Lem 7.9] regardless of what is
stored, so (b) keeps `E` a certified subset. Termination is governed by the
true potential `Φ`, which the paper proves strictly decreasing; the stored sets
never influence which cycle is shifted beyond the choice of `v_i`, and any `v_i`
with `e_i` critical for `(v_i, φ(v_i))` is admissible. ∎

Consequence: after deleting an arc `e`, only vertices whose stored flow used `e`
need any work (docs/optimizations.md O1–O2); the others keep their (now possibly
stale-but-subset) sets. Exact sets are refreshed lazily when a witness must be
(re)computed. For the weighted algorithm a spurious infeasibility of the split
assignment on stale subsets is resolved by refreshing all cuts and retrying.

### P3. Sound sparse preprocessing for dense undirected inputs (2026-09-21)

For an undirected `k`-connected input, a Nagamochi–Ibaraki sparse certificate
`H ⊆ G` with `≤ k n` edges preserves all local vertex connectivities up to
`k`. Hence the *initial* flow certificates `F_v` (which only need value `k`)
can be computed in `H` in `O(n k (n + kn))` instead of `O(n k (n + m))`; they
remain valid families of paths in `G`, and `Ess(v) = T` is known a priori.
(Only the initial state is symmetric; after the first deletion the digraph is
asymmetric and `H` is not reused.) Status: proved; to be benchmarked on dense
families.

### P4. Deletion-aware (penalized) routing preserves kappa, the tightest cut and Ess (2026-09-23)

**Statement.** Let `w: E -> {0,1}` be any per-arc penalty and let every augmenting-path search of the flow
engine return, instead of a BFS-shortest residual path, a residual path of minimum total penalty of its
forward arcs (ties broken by the BFS order). Then for every vertex `v` the stored `F_v` is still a *maximum*
flow, `kappa_G(v)` is unchanged, and the tightest minimum cut and `Ess_G(v)` derived from it are unchanged.
Consequently every decision of [Alg 1] / [Alg 2] — criticality, FEAC preservation, the witness search, the
greedy-contraction test, the split assignment — is unchanged; only *which arcs the certificates use* changes.

**Proof.** The penalized search visits exactly the residual moves the BFS visited (the same arcs, the same
split arcs, the same `forbidden` set, the same "never route into the source" rule); it only picks a different
one of them when several reach the sink. Hence it returns a path **iff** the residual network contains an
augmenting path, so the augmentation loop of `compute_max_flow` stops exactly when the flow is maximum, with
the same value `kappa_G(v)` [Prop 4.2], and the warm-started single augmentation of O2 succeeds in `G \ e`
exactly when it succeeded before. Given a *maximum* flow, `compute_cut` computes the tightest cut from
residual reachability of the sink, and the tightest minimum cut is **unique** [Def 3.8] — in particular
independent of which maximum flow realizes it — so `S`, `|S| = kappa` and `Ess = T ∩ S` [Lem 4.1] are the same
for every maximum flow of `v`. O1 and O2 are statements about `kappa` and `Ess` only, so they carry over
verbatim; P2's certified-subset discipline is unaffected (it only ever compares stored subsets of `Ess`). ∎

Status: proved; implemented as `options["routing"] = "avoid"` (`FlowEngine::set_penalties` + a 0-1 search in
`flow.cpp`), default off. Empirically confirmed: on 60 + 30 small instances the exact per-vertex `kappa` /
`Ess` of the first `essential` trace event are identical to the `routing="bfs"` run and to the
definition-based reference `glsolver.preconditions.essential_sets_by_definition`, the statuses agree, and
every partition is accepted by the independent verifier (tests/test_core_solver.py, section 10). The
performance question C1 asked about this routing is answered in E5 (and in "Failed ideas").

## Conjectures

### C2. Amortized bound below O(n k m²) (2026-09-21)

With O1–O2 the work per deletion step is `O(Σ_i |U_i| (n+m))` where
`U_i = {v : e_i ∈ F_v}`; the trivial bound `Σ_i |U_i| ≤ nk` reproduces the
paper's per-step cost. Conjecture: with flows chosen as in C1 and the greedy
contraction O5, the *total* number of flow repairs over the run is `O(n k)`
on `k`-connected undirected inputs, giving `O(n k (n+m))` overall. Open.
(C1's routing is now implemented, proved exact in P4 and measured in E5; its
premise — that the routing keeps `Σ_i |U_i|` small — holds only where the graph
has routing slack, so this conjecture is open only for such inputs.)

### C3. GPU acceleration of the certificate phase (2026-09-21)

The machine has an NVIDIA GB10 (Grace–Blackwell, unified memory, CUDA 13.0
toolkit with nvcc; no CuPy/numba installed). The only embarrassingly parallel
phase of the general algorithm is "one unit-capacity flow per vertex" (the
initial certificates and the ≤ k full refreshes after terminal removals). A
batched multi-source level-synchronous BFS kernel with per-source residual
flags could plausibly beat the 20-core CPU version by 5–20× on that phase for
`n ≥ 10^5`. The sequential main loop (incremental re-validation of a few
vertices per step) is not a GPU workload. Status: idea; to be prototyped only
if profiling (Stage G) shows the certificate phase dominating at scale.

## Empirical observations

### E1. First measurements of the C++ core (2026-09-21, 20-core Cortex-X925, 8 threads, release build)

| instance | n / m / k | general solver | weighted solver (unit weights) | notes |
|---|---|---|---|---|
| official counterexample, 1 copy | 333 / 2 160 / 9 | 0.04 s | 0.05 s | pure-Python reference: 900 s |
| official counterexample, 17 copies | 3 789 / 33 264 / 9 | 1.8 s | 1.4 s | 3 780 contractions, 24 105 deletions, 147 cycle shifts; 3 768 of 3 780 contractions via greedy O5 |
| Harary H_{8,2000} | 2 000 / 8 000 / 8 | 10–15 s | 9.7 s | ~60 % of the time in `after_contraction` path translation (long circulant flow paths) |
| random 8-regular | 5 000 / 20 000 / 8 | 30–40 s | — | dominated by warm-started evaluations inside ShiftAssignment; O5 succeeds in only ~10 % of attempts |
| Erdős–Rényi | 2 000 / 399 355 / 6 | 2–3 s | — | 368 073 deletions, almost all in O1 batches; every contraction greedy; **zero** ShiftAssignment calls |

Observations: (a) on dense inputs the certificate-guided path (O1 + O5) alone finishes the run — the paper's
cycle-shift machinery is never needed; (b) on sparse regular graphs the greedy attempt often fails and the exact
fallback dominates; a credit-based throttle (attempt cost 4, success credit 16) and per-vertex backoff halved the
running time; (c) contraction cost is not free: translating every stored path through the contracted vertex is
`O(Σ_v |path through p|)`, which on high-diameter graphs (Harary) is the bottleneck — a cheaper representation
(store paths as predecessor pointers so translation is O(1) per path) is the obvious next optimization;
(d) exhaustive correctness: 145 400 undirected instances (n ≤ 7) and 3 278 digraphs (n ≤ 4) solved by both C++
solvers with debug assertions, 0 failures.

### E2. Baseline benchmark sweep (2026-09-21 21:02–22:39, 2 894 rows, 0 invalid, 0 timeouts, 0 errors)

Full data: `benchmarks/results/*.jsonl`, `summary.md`, `plots/`; methodology in docs/benchmarks.md.
Medians over 3 seeds (1 seed at n ≥ 5 000), 8 threads, release build, k = 4:

| family | n = 1 000 | n = 2 000 | n = 5 000 | n = 10 000 | slope (n ≥ 500) |
|---|---:|---:|---:|---:|---:|
| Harary H_{4,n} | 3.0 s | 29.6 s | — (memory) | — | **3.3** |
| random 4-regular | 0.15 s | 0.56 s | 4.0 s | 19.8 s | 2.0 |
| Erdős–Rényi (m ≈ n²/10) | 0.15 s | 1.4 s | 27 s | 226 s (m = 10⁷) | 3.1 in n ≈ 1.5 in m |
| sparse k-connected (Harary + chords) | 0.20 s | 0.49 s | 2.8 s | 10.1 s | 1.7 |
| random k-T DAG via general solver | 0.19 s | 0.62 s | 4.8 s | 19.6 s | 1.9 |
| random k-T DAG via DAG solver | 0.6 ms | 1.3 ms | 3.7 ms | 6.6 ms | **1.02** |

The pure-Python reference has slope 3.0–3.6 and needs 9–87 s at n = 100; the C++ core is 1 500–40 000× faster at n = 100.
DAG solver: n = 10⁶, m = 5·10⁶ in 2.0–2.4 s single-threaded; **heap and linear (P1) variants are within ±10 %** at every size — the
log factor is invisible because both are dominated by the O(n + m) CSR/topological-order construction. P1 stands as a
theoretical improvement without practical effect at these sizes.

Operation counts are linear: contractions = max-flow calls = n − k exactly; deletions ≈ 2.5 n (Harary) / 3.5 n (regular);
cycle shifts ≈ n^1.0–1.1. The super-linear runtime is therefore *per-operation* cost. Two causes were isolated on
Harary n = 1 000 (12 s, 7.9 GB): (1) 780 000 warm-started augmentations for 996 contractions — every deletion re-validates
≈ 270 flows because BFS-shortest paths on a circulant graph all funnel through the arcs near the terminals; (2) 8.3 of
12 s in `after_contraction` path translation (paths are O(n/k) long and stored as arc vectors, so locating the arc to
splice is O(path length) per affected flow). Memory (RSS ≈ n^2.9, 56 GB at n = 2 000 seed 2) comes from the per-arc
user index, which appends an entry for every arc of a changed flow and only compacts lazily: (#flow changes) × (path
length) entries. Greedy contraction (O5) succeeds in 67 % of attempts at n = 100 but only 32 % at n = 10⁴ on random regular
graphs, so the exact ShiftAssignment fallback increasingly dominates there. These three items define the optimization stage.


### E3. Oracle engineering: exact user index, O(1) contraction translation, worker pool (2026-09-22)

Target: the two per-operation hotspots of E2 and the memory blow-up, without changing a single
operation of the algorithm (every counter — `augment_calls`, `cut_calls`, `cycle_shifts`,
`greedy_attempts` — is identical before and after on every instance below; the nine differential
suites are green on the release build (1 909 passed, 16 skipped, four new oracle tests included) and
under ASan+UBSan (the final committed code was re-checked under ASan+UBSan: the seven core suites pass with 0 sanitizer reports, see E3-verify).

What changed (src/glcore/flow.*, essential.*, threadpool.hpp; call sites touched by two lines each):

1. **Exact user index** (memory). `arc_users_[a]` / `vertex_users_[x]` used to be append-only lists of
   `(v, stamp)` entries compacted lazily, so every change of a flow appended its whole path length
   again: RSS ≈ (#flow changes) × (path length) = 7.9 GB at H_{4,1000}, 62 GB at H_{4,2000}. Now every
   path arc `paths[i][j]` of a registered flow owns one registry entry `reg_[v][i][j] = {arc, slot in
   arc_users_[arc], slot in vertex_users_[head]}` and the two lists hold back-references `(v, i, j)`.
   Removing an entry is a swap-remove that patches the position stored by the entry moved into the
   hole (O(1)); a flow is unregistered before any change and re-registered after, so the index holds
   exactly the arcs of the valid flows: memory O(total live path length) at all times (3.29 M arcs ×
   36 B ≈ 118 MB on H_{4,2000}). `num_users_of_arc(a)` is O(1) (the solvers' O1/O5/O6 scans used
   `users_of_arc(a).size()`, an allocation + sort per arc).
2. **O(1) contraction translation** (time). `after_contraction(p, t)` located the arc into `p` by
   scanning every path of every flow through `p` (`O(Σ path lengths)`, 8–9 s of 12 s on H_{4,1000},
   97 s of 129 s on H_{4,2000}). The vertex index now names the registry entry, i.e. the position
   `(i, j)` of that arc; since `(p, t)` is the last arc of the path, the translation is
   `paths[i][j] = (x, t); paths[i].pop_back()` plus two O(1) index removals and two appends. No path is
   scanned; the flows through `p` are read off `vertex_users_[p]` alone.
3. **Sides never stored.** `VertexFlow::side` (n bytes per flow, n² in total) is empty by default:
   `compute_cut` derives |S| = #reached out-nodes − #reached in-nodes and Ess = {t : t_in unreached}
   from the reverse BFS alone; `EssentialOracle::sides(v)` materializes the L/S/R labels on demand by
   one reverse BFS (trace / record_cuts / the Python `sides()` diagnostic — same values: the tightest
   cut is unique, [Def 3.8]).
4. **Persistent worker pool** (`threadpool.hpp`). `run_parallel` used to spawn `std::thread`s per
   region (one region per deletion evaluation: tens of thousands per run). A pool created lazily per
   oracle and joined in its destructor is woken through a generation counter; the atomic work counter
   and per-index result slots are unchanged, so results stay deterministic (asserted for 1 vs 4 and 1
   vs 8 threads). 400 small evaluate_deletion regions on a random 8-regular graph: 0.38 s → 0.19 s.
5. **Exact micro-optimizations in the flow engine.** (a) `next_arc[x]` (the flow arc leaving x) is
   maintained by every flag change, so the path lists are rebuilt after an augmentation in
   O(total path length) instead of scanning every out-list along the paths, and the flags are only
   re-derived when the augmentation left flow on a cycle (detected by counting flow arcs);
   (b) `remove_paths_using_arcs(D)` drops all paths through D in one pass (was |D| passes);
   (c) the per-arc `flow` and `forbidden` flags share one byte array (one random access per arc in the
   BFS loops instead of two). Same augmenting paths, same cuts: the counters are identical.

Not done: an "early exit" of the reverse BFS is not possible when Ess is needed exactly — a terminal
is essential iff its in-node is *not* reached, which only the complete search decides; and the
remaining Harary cost is the 780 k warm-started augmentations (algorithmic, see E2/E4), not the
oracle's bookkeeping.

Measurements (general solver, 8 threads, `verify=False`, each run in its own subprocess reading
`ru_maxrss`; medians of 3). The machine was shared with another agent's jobs throughout (load average
4–10), so absolute times drift between columns; the decisive columns are the two *interleaved* A/B
runs (HEAD core vs HEAD + this oracle, alternating run by run under the same load, both built from the
same tree into separate directories), whose ratios are load-fair. "before, first run" is the HEAD core
measured first (lightest load seen); "after, single-build run" is a separate 3-repeat run of the new
oracle made before the last O(path-length) trim of flow.cpp (reset/remove passes), which the A/B
columns include.

| instance | metric | before, first run | after, single-build run | before (A/B, interleaved) | after (A/B, interleaved) | after / before (A/B) |
|---|---|---:|---:|---:|---:|---:|
| **Harary H_{4,1000} seed 2 (n=1000, m=3984, k=4)** | wall s | 12.95 | 3.15 | 16.52 | 3.16 | **0.19** |
| | peak RSS MB | 7873 | 103 | 7896 | 102 | **0.0129** |
| | after_contraction s | 9.211 | 0.084 | 11.917 | 0.089 | **0.0075** |
| | evaluate_deletion s | 2.28 | 2.91 | 2.72 | 2.86 | 1.05 |
| | augment / cut calls, cycle shifts | 780561 / 782129, 862 | same | same | same | identical |
| **Harary H_{4,2000} seed 2 (n=2000, m=7984, k=4)** | wall s | 128.81 | 25.02 | 125.15 | 22.59 | **0.18** |
| | peak RSS MB | 62344 | 255 | 62372 | 250 | **0.0040** |
| | after_contraction s | 97.417 | 0.492 | 94.073 | 0.492 | **0.0052** |
| | evaluate_deletion s | 19.59 | 23.88 | 20.10 | 21.42 | 1.07 |
| | augment / cut calls, cycle shifts | 3105050 / 3108151, 1732 | same | same | same | identical |
| **random 8-regular n=5000, k=8 (m=39936)** | wall s | 22.66 | 26.78 | 36.21 | 31.83 | **0.88** |
| | peak RSS MB | 186 | 88 | 196 | 88 | **0.4497** |
| | after_contraction s | 0.066 | 0.010 | 0.093 | 0.013 | **0.1430** |
| | evaluate_deletion s | 21.18 | 25.11 | 33.87 | 30.27 | 0.89 |
| | augment / cut calls, cycle shifts | 235214 / 125897, 1419 | same | same | same | identical |
| **Erdős–Rényi n=2000, p=0.2, k=6 (m=797756)** | wall s | 3.78 | 2.94 | 4.09 | 2.88 | **0.70** |
| | peak RSS MB | 358 | 300 | 309 | 300 | **0.9696** |
| | after_contraction s | 0.017 | 0.015 | 0.022 | 0.015 | **0.6980** |
| | evaluate_deletion s | 0.91 | 0.99 | 1.06 | 0.96 | 0.91 |
| | augment / cut calls, cycle shifts | 6310 / 4858, 0 | same | same | same | identical |

Reading the table: the two Harary hotspots are gone — `after_contraction` drops from 94 s to 0.49 s on
H_{4,2000} (0.5 %) and peak memory from 62 GB to 250 MB (0.4 %; H_{4,1000}: 7.9 GB → 102 MB), both
targets met (< 1 GB, < 0.5 s); what remains on Harary is `evaluate_deletion` (the 780 k / 3.1 M
warm-started augmentations, ≈ 2.9 s / 21 s), i.e. the algorithmic item of E2/E4. The random-regular
and Erdős–Rényi instances, which never suffered from the index, gain 12 % and 30 % from the worker
pool, the O(1) user counts and the flow-engine micro-optimizations, at half (regular) or the same
(ER) memory. The per-operation costs are now O(1) per affected flow for contraction and
O(|D| + path length + n + m) per evaluated flow for deletion, with the index memory bounded by the
live path length.

Commands: `python /tmp/glbench/bench.py before 3 h1000 er2000 rr5000 h2000` (HEAD, first run),
`BENCH_CORE_DIR=/tmp/glbench/mine/build python /tmp/glbench/bench.py after 3 …` (single-build run) and
`python /tmp/glbench/ab.py ab2 3 h1000 rr5000 er2000 h2000` (interleaved; HEAD in /tmp/glbench/old/build
vs HEAD + src/glcore/{flow,essential,threadpool} in /tmp/glbench/mine/build, both `cmake -DCMAKE_BUILD_TYPE=Release`,
selected through `BENCH_CORE_DIR` which substitutes `glsolver._core`). Instances: `harary_graph(n, 4, seed=2)`,
`random_regular_graph(5000, 8, 8, seed=2)`, `erdos_renyi_graph(2000, 0.2, 6, seed=2)`; run through
`glpartition(inst, algorithm="general", threads=8, verify=False)`; `result.stats["time_seconds"]`
keys `essential.after_contraction`, `essential.evaluate_deletion`, `essential.compute_all`.

### E3-verify. Independent re-measurement of E3/E4 (2026-09-23, quiet machine)

The optimization agents were interrupted by a usage limit before reporting; their code was kept only
after it was re-verified from scratch. Whole suite on the release build: **2 585 passed, 16 skipped,
0 failed**; the seven core differential suites also pass under ASan+UBSan with 0 sanitizer reports (`pytest -m "not slow" -n 12 --hypothesis-profile=ci tests`). Timings below are my own
runs, each in a fresh subprocess reading `ru_maxrss`, 8 threads, `verify=False`, `algorithm="general"`
(median of 3 for the first row, single runs otherwise), on an otherwise idle machine. The "before"
column is E2 (the baseline sweep).

| instance | before: time | after: time | before: peak RSS | after: peak RSS |
|---|---:|---:|---:|---:|
| Harary H_{4,1000} seed 2 | 12.0 s | **2.05 s** | 7 868 MB | **102 MB** |
| Harary H_{4,2000} seed 2 | 129.5 s | **17.7 s** | 56 551 MB | **248 MB** |
| random 4-regular n = 10 000 | 19.8 s | **15.4 s** | 232 MB | **102 MB** |
| random 8-regular n = 5 000, k = 8 | 30–40 s | **17.0 s** | — | 88 MB |
| Erdős–Rényi n = 10 000, m = 10⁷ | 226 s | **189 s** | 4 094 MB | **2 855 MB** |

Speed-ups: 5.9× and 7.3× on the two Harary sizes, memory 77× and **228×**. The contraction
translation is gone as a cost centre (8.3 s → 0.04 s at n = 1 000, 0.32 s at n = 2 000, i.e. from 69 %
of the run to 2 %). On the 10⁷-arc instance the numpy path (E4) leaves only 2.6 s of the 189 s wall
clock outside the core, so that instance is now genuinely core-bound.

**The remaining bottleneck is now purely algorithmic**, exactly as E2 predicted: `evaluate_deletion`
(the warm-started augmentations of O2) is 1.7 s of 2.05 s on H_{4,1000}, 15.3 s of 17.7 s on
H_{4,2000}, 13.6 s of 15.4 s on random 4-regular n = 10 000, and 10.0 s of 12.1 s on sparse
k-connected n = 10 000. The augmentation count itself is unchanged (778 k on H_{4,1000}), so the
engineering stage removed every constant-factor obstacle and the open question is the one C1 states:
can the stored flows be routed so that deletions touch fewer of them?

### E5. C1 deletion-aware routing, measured (2026-09-23, 8 threads, `verify=False`; every number below re-measured after the review fix of E5-review)

**What was implemented** (`options["routing"] = "bfs" | "avoid"`, default `"bfs"`, in both core solvers and in
`glsolver.api`; exactness is P4 above):

1. *A per-arc penalty byte owned by the solver.* `penalty_[a] = 1` iff the tail `p` of `a` is a live
   pre-terminal and `a` is an out-arc of `p` that the algorithm may delete — with a witness, every out-arc
   other than `(p, phi(p))` (the `D_p` of steps (iii-a)/(iii-b) and the secondary-arc candidates of [Alg 2]);
   before the witness exists (during the initial `compute_all`), every out-arc that does not enter a terminal.
   The weighted solver uses `psi_target_arc(p)` [Def 5.2] in the role of `(p, phi(p))`.
   Maintained incrementally, because a penalty depends only on the pre-terminal status of the TAIL and on
   phi/psi of the tail: the tails of deleted arcs (and only when a *deleted arc entered a terminal*, which is
   the only way the pre-terminal status or the kept arc can change), the in-neighbours of a contracted vertex
   (their arc into `p` became an arc into `t`, with a new arc id), the vertices moved by a cycle shift, and a
   full `O(n + m)` recomputation on a terminal removal (at most `k` times). With `debug_asserts` the whole
   array is compared against a full recomputation after *every* operation.
2. *A 0-1 augmenting search.* Instead of a deque (whose `push_front` would make the penalty-0 part of the
   search depth-first) the frontier is two FIFOs — current penalty level and next — so ties between
   equal-penalty paths are broken in the plain BFS order and the all-zero-penalty case is bit-identical to the
   old BFS. A node is relabelled when a cheaper level reaches it; a target found by a penalty-0 move is
   returned immediately, a target first reached by a penalty-1 move is returned when the current level is
   exhausted. `compute_max_flow` and the warm-started `augment_once` both honour it.
3. *No cost when off — verified, after a codegen trap that cost 6-19 % (see E5-review).* The two searches
   are separate instantiations of one template (`search_target<bool Penalized>`) reached through two
   never-inlined entry points (`search_plain` / `search_penalized`); the penalty array is **never maintained**
   with `routing="bfs"` — it is recomputed only at the two diagnostic points inside `run()` and is stale in
   between, is never handed to the oracle, and its incremental maintenance is compiled out at every call site
   (`penalty_update_vertex` is `if (opt_.routing_avoid) ...`); the 0-1 workspaces (`dist`, `queue_next`) are
   allocated only by the penalized search. *Semantics*: with `routing="bfs"` this core reproduces the pre-C1
   core (HEAD 15be540, built separately from `git archive` with the same compiler, flags and pybind11)
   **counter for counter and partition for partition** (`max_flow_calls`, `augment_calls`, `cut_calls`,
   `matching_calls`, `contractions`, `deletions`, `batched_deletions`, `cycle_shifts`, `terminal_removals`,
   `greedy_attempts/successes`, `shift_calls`, `steps`, parts hash) on H_{4,1000}, random 4-regular n = 2 000
   and n = 10 000, random 8-regular n = 5 000, sparse k-connected n = 10 000, Erdos-Renyi n = 2 000 and the
   official counterexample with 17 copies. *Cost* (5-7 interleaved paired reps, fresh subprocess each,
   medians with the per-rep range): pure augmenting-search benchmark (`EssentialOracle.compute_all`, one
   thread) **0.96x (0.93-1.04) on rr10000 and 0.95x (0.90-1.02) on sk10000** of the pre-C1 core; the same
   benchmark through `_core.tightest_cut` over every vertex 0.99x / 0.96x; end to end at `routing="bfs"`
   0.93x-1.03x on the seven instances (h1000 0.933, rr2000 0.894, er2000 1.026, ce17 0.969, sk10000 0.965,
   rr10000 0.989, rr5000k8 1.018) — i.e. free within the noise, and *faster* than the pre-C1 core where the
   difference resolves. The first version of this code was **6-17 % slower** on those micro-benchmarks
   (medians; up to 19-22 % in the single-threaded variant runs of E5-review) with exactly the same
   instructions; E5-review documents why and what fixed it.
4. *Two new diagnostics in `Stats`* (both measured in both modes, so the A/B is meaningful):
   `flow_repairs` = stored flows re-validated by `evaluate_deletion`, summed over the run (speculative O5
   attempts included) — "repairs per deletion" is this over `deletions`. It is **not** an independent signal
   next to `augment_calls`: a repaired flow drops the paths through the deleted arcs and re-augments once per
   dropped path, so where almost every affected flow loses exactly one path the two counters nearly coincide
   (H_{4,1000}: 777 783 repairs vs 778 191 augmentations; H_{4,2000}: 3 079 261 vs 3 079 852). They separate
   only where the number of *deletions* also moves (sk10000 below: 25 378 -> 25 555 deletions, so
   repairs/deletion 8.11 -> 4.43 while the augmentations fall by 41 %). `penalized_users_initial` /
   `penalized_users_witness` = the number of stored flows using an arc a pre-terminal may lose, measured right
   after `compute_all` and right after the initial witness. (`reroutes` counts the optional re-routing pass.)

**Measurements.** Medians of **5 paired reps**, each run in a fresh subprocess reading `ru_maxrss`, the two
options **interleaved run by run** (order alternating) so the comparison is load-fair; the per-rep ratio range
is quoted next to every median, and a median whose range straddles 1.0 is not a difference this many runs can
resolve. Load during the session: **1.2-4.8**, of which the benchmark (one solver
process, 8 threads, at a time) is the larger part: with nothing of mine running the 1-minute load average
decays to **0.30-0.37** (sampled every 20 s for 11 minutes after the last run), so the permanent background of
long-lived interactive processes on this host (two `htop`, two `grok` sessions, `gpustat`, `code-tunnel`, the
agent itself, all up for ~12 days, ~0.4 CPU in aggregate per `ps -eo pcpu,etimes`) is small but **bursty** —
an interactive agent session on the same machine can add a core for a while, which is exactly why every
comparison here is interleaved rather than run in blocks. Absolute times carry that load; the ratios do not.

| instance (n / m / k) | routing | wall s | evaluate_deletion s | augment | cut | cycle shifts | greedy att/succ | peak RSS MB | repairs/deletion | users of penalized arcs (after compute_all -> after witness) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Harary H_{4,1000} seed 2** (1000 / 3984 / 4) | bfs | 2.80 | 2.36 | 778191 | 779795 | 862 | 1685/537 | 103 | 268.4 | 5008 -> 7996 |
|  | avoid | 2.69 | 2.34 | 777422 | 778940 | 862 | 1682/536 | 104 | 268.1 | 4032 -> 7020 |
| | *ratio avoid/bfs (5 paired reps)* | **0.974** (0.894-1.061) | 0.992 (0.922-1.063) | 0.999 | 0.999 | | | 1.005 | | same partition: True; verified: (True, True) |
| **Harary H_{4,2000} seed 2** (2000 / 7984 / 4) | bfs | 19.83 | 16.70 | 3079852 | 3082938 | 1732 | 3212/1024 | 251 | 542.2 | 8233 -> 14221 |
|  | avoid | 19.01 | 16.38 | 2940071 | 2943241 | 1732 | 2993/935 | 253 | 517.6 | 8032 -> 14020 |
| | *ratio avoid/bfs (5 paired reps)* | **0.989** (0.935-1.147) | 0.978 (0.933-1.121) | 0.955 | 0.955 | | | 1.009 | | same partition: True; verified: (True, True) |
| **random 4-regular n=2000 seed 1** (2000 / 7984 / 4) | bfs | 0.57 | 0.46 | 23399 | 11943 | 479 | 2291/1589 | 60 | 4.14 | 51 -> 7047 |
|  | avoid | 0.75 | 0.63 | 24684 | 12575 | 481 | 2187/1464 | 61 | 4.41 | 48 -> 7044 |
| | *ratio avoid/bfs (5 paired reps)* | **1.369** (0.986-1.678) | 1.435 (0.988-1.791) | 1.055 | 1.053 | | | 1.020 | | same partition: False; verified: (True, True) |
| **random 4-regular n=10000 seed 1** (10000 / 39984 / 4) | bfs | 15.51 | 13.73 | 129724 | 68314 | 2608 | 8633/6622 | 103 | 4.62 | 50 -> 30232 |
|  | avoid | 17.92 | 15.88 | 136534 | 69328 | 2505 | 8437/6315 | 106 | 4.89 | 48 -> 30228 |
| | *ratio avoid/bfs (5 paired reps)* | **1.150** (1.126-1.209) | 1.157 (1.128-1.212) | 1.052 | 1.015 | | | 1.026 | | same partition: False; verified: (True, True) |
| **random 8-regular n=5000, k=8, seed 1** (5000 / 39936 / 8) | bfs | 17.76 | 16.41 | 194280 | 108379 | 1443 | 6633/3504 | 89 | 6.18 | 516 -> 35291 |
|  | avoid | 18.13 | 16.79 | 189932 | 105810 | 1305 | 6407/3402 | 90 | 6.04 | 452 -> 35221 |
| | *ratio avoid/bfs (5 paired reps)* | **1.021** (0.997-1.041) | 1.023 (0.998-1.045) | 0.978 | 0.976 | | | 1.013 | | same partition: False; verified: (True, True) |
| **Erdos-Renyi n=2000, p=0.1, k=6, seed 1** (2000 / 398524 / 6) | bfs | 0.93 | 0.32 | 5641 | 4016 | 0 | 1993/1993 | 178 | 0.005 | 4332 -> 14300 |
|  | avoid | 1.16 | 0.38 | 5641 | 4016 | 0 | 1993/1993 | 178 | 0.005 | 4332 -> 14300 |
| | *ratio avoid/bfs (5 paired reps)* | **1.210** (0.982-1.705) | 1.221 (1.140-1.339) | 1.000 | 1.000 | | | 1.003 | | same partition: True; verified: (True, True) |
| **sparse k-connected n=10000, k=4, seed 1** (10000 / 41984 / 4) | bfs | 11.67 | 9.64 | 205885 | 122521 | 19 | 10060/9959 | 144 | 8.11 | 219 -> 30207 |
|  | avoid | 9.74 | 7.61 | 120701 | 69899 | 10 | 8689/8632 | 139 | 4.43 | 122 -> 30110 |
| | *ratio avoid/bfs (5 paired reps)* | **0.815** (0.790-0.876) | 0.780 (0.745-0.827) | 0.586 | 0.570 | | | 0.964 | | same partition: False; verified: (True, True) |
| **official counterexample, 17 copies** (3789 / 33264 / 9) | bfs | 0.45 | 0.30 | 19347 | 18393 | 148 | 3795/3756 | 69 | 0.760 | 0 -> 12960 |
|  | avoid | 0.49 | 0.33 | 19347 | 18393 | 148 | 3795/3756 | 69 | 0.760 | 0 -> 12960 |
| | *ratio avoid/bfs (5 paired reps)* | **1.071** (0.926-1.172) | 1.086 (0.921-1.221) | 1.000 | 1.000 | | | 1.008 | | same partition: True; verified: (True, True) |

Every partition was accepted by `glsolver.verify.verify_instance_parts` in both modes (8/8 instances, column
"verified"); the partitions differ on 4 of the 8 instances (both valid — §13.8: every choice is free).

**Reading the table.**

* **Where the graph is minimally connected (degree = k), there is nothing to route around.** On Harary
  H_{4,n} (4-regular, exactly 4-connected), random 4-regular with k = 4 and random 8-regular with k = 8 the
  vertex-disjoint path families are essentially forced: the users of the deletable arcs drop by at most 20 %
  after `compute_all` (H_{4,1000}: 5 008 -> 4 032) and the augmentation count by at most 4.5 % (H_{4,2000}) —
  on random 4-regular it even *rises* by 5 % (the penalized path is longer, so it crosses more arcs that are
  deleted later). Meanwhile every search costs more: it must prove that no penalty-free path exists before it
  may use a penalized one, i.e. it explores the whole penalty-0 region instead of stopping at the first
  terminal it meets. Net on the wall clock: the two Harary sizes come out **indistinguishable from 1.0**
  (0.974 and 0.989, both ranges straddling 1.0 — the small saving in augmentations pays for the dearer
  search), random 8-regular is a **2 % slowdown** (1.021, range 0.997-1.041) and random 4-regular is a clear
  **15-37 % slowdown** (rr10000 1.150 with every rep > 1.12; rr2000 1.369 on a 0.6 s run whose range
  0.99-1.68 is wide, so read it as "clearly slower, size uncertain").
* **On dense graphs and on the official counterexample the penalized search returns exactly the BFS path**
  (the first path the BFS finds is already penalty-free), so every counter is identical — but the search
  itself is *not* free there: with identical counters Erdos-Renyi n = 2 000 pays **+22 % on
  `evaluate_deletion`** (1.221, range 1.140-1.339) and the counterexample +9 % (1.086, range 0.921-1.221),
  because a 0-1 search on a dense graph reads one penalty byte per scanned arc and may not stop at the first
  terminal it reaches. On the wall clock that is 1.21 (range 0.98-1.71, a ~1 s run: noisy) and 1.07
  (0.93-1.17). The earlier claim that this shows "0-3 % on the wall clock" understated the per-search cost.
* **On the one family with sparse slack it works exactly as C1 hoped.** `sparse_k_connected(10000, 4)` is
  H_{4,10000} plus 1 000 random chords, so a path can leave the terminal's neighbourhood by a chord: users of
  penalized arcs after `compute_all` 219 -> 122, **augmentations 205 885 -> 120 701 (-41 %)**, flows repaired
  per deletion **8.11 -> 4.43 (-45 %; the deletions themselves move too, 25 378 -> 25 555)**, cycle shifts
  19 -> 10, greedy attempts 10 060 -> 8 689 at a nearly unchanged success rate, `evaluate_deletion`
  9.64 s -> 7.61 s (0.780, range 0.745-0.827), wall **11.67 s -> 9.74 s (0.815, range 0.790-0.876)**, peak RSS
  -3.6 %. This is the only instance where the cost of the search is repaid.

**Verdict: keep as an option, default `"bfs"`.** The adoption rule of the task ("never more than 10 % slower
anywhere *and* meaningfully faster on the Harary/regular families") still fails on both counts, though less
dramatically than the pre-review measurement suggested: `avoid` is 15 % slower on random 4-regular n = 10 000
(every paired rep > 1.12), ~37 % on random 4-regular n = 2 000, 21 % on the dense Erdos-Renyi instance and
2-7 % on random 8-regular / the counterexample, while on the two Harary sizes it is a wash (0.97 / 0.99, both
ranges straddling 1.0) — it is *not* faster on any Harary or regular instance. It is kept because on
sparse-with-slack inputs it is a 1.23x end-to-end speed-up and a 1.7x reduction of the augmentation count,
which is the one quantity E3-verify identified as the remaining algorithmic bottleneck.

**The optional re-routing pass, measured and left off.** After the initial witness the arcs the algorithm will
really delete are known, so the flows of `compute_all` (routed under the weaker pre-witness rule) can be
recomputed once: `EssentialOracle::reroute` re-computes exactly the flows that use a penalized arc (bounded by
one `compute_all`, once per run; `GLCORE_C1_REROUTE=1`, read per solver at construction so both settings can
run in one process). Note it is more than a re-routing: those flows come back with *exact* cuts, which can
change later decisions that read `ess(v)` — sound (an exact cut is a certified subset of itself, P2) but a
broader change than "same flows, different arcs". It does reach a better routing — on random 4-regular
n = 2 000 it halves the users of penalized arcs (7 044 -> 4 064) and on sparse k-connected n = 10 000 it
removes a further 12 % of the augmentations (120 701 -> 106 276) and 13 % of the repairs (113 145 -> 98 723) —
but it costs one penalized from-scratch flow per offending vertex (`max_flow_calls` 9 996 -> 19 992), which is
far more than it saves. Medians of 3 interleaved paired reps, `routing="avoid"` in both arms: sk10000 wall
9.70 s -> 13.29 s (**1.371**, range 1.370-1.375) and — against the intuition, and against what an earlier
version of this note claimed — `evaluate_deletion` goes **up**, 7.55 s -> 7.78 s (1.065, range 1.029-1.082),
despite 12 % fewer augmentations: the re-routed flows carry exact cuts, so the work that follows is not the
same work. rr2000 0.77 s -> 0.88 s (1.169), H_{4,1000} unchanged (0.977, range 0.968-1.077; 996 flows
re-routed, not one augmentation saved — the flows are forced). It is therefore disabled by default; the code
and the environment switch are kept so the counter-measurement is reproducible, and
`tests/test_core_routing_review.py` covers the pass both in-process and in a subprocess.

**Tests.** Whole suite on the release build (`pytest -m "not slow" -n 12 --hypothesis-profile=ci tests`):
**2 583 passed, 16 skipped, 0 failed**; under ASan+UBSan (`-n 4`, `detect_leaks=0`, LD_PRELOAD): **2 582
passed, 17 skipped, 0 failed, 0 sanitizer reports** in 39 min (the extra skip is the wall-clock scaling bound,
which is skipped under sanitizers), and after the last (comment-only) edits the five core suites were re-run
under the sanitizers on the shipped code: **1 028 passed, 8 skipped, 0 failed, 0 sanitizer reports**. The release build was restored afterwards (`GLCORE_SANITIZE=OFF`), and
`-Wall -Wextra -Wpedantic` stayed clean throughout. The option's own suite is
`tests/test_core_routing_review.py`: a differential fuzz over > 2 000 seeded instances of every kind
(`avoid` vs `bfs`: same status, both partitions accepted by the independent verifier), the oracle's per-vertex
`kappa`/`Ess` against `glref.essential.all_essential` at every trace event, the FEAC/FESAC trace replay under
`avoid`, the incremental penalty maintenance under `debug_asserts`, a C++ probe of the 0-1 search on 20 000
seeded residual states, the re-routing pass in-process and in a subprocess, and — added by the review — the
path-level invariants of the penalized search through the new bindings (see E5-review §5). An earlier run
with the option forced **on** for every core call showed one artifact of the forcing, not of the option:
`test_core_solver_review.py::test_initial_witness_is_an_exact_feac_witness` compares a *raw*
`_core.solve_general(..., {})` call (which keeps the default `"bfs"`) with the api call (forced to `"avoid"`)
and finds different in-arborescence parents, i.e. two different valid partitions; with `routing="avoid"` on
both sides all of its assertions hold on 106 instances (the witness itself is identical in both modes — it
only depends on `Ess`).

**The weighted solver** carries the same option (penalty 1 on every out-arc of a live pre-terminal other than
its `psi_target_arc`), but it was only *tested*, not benchmarked: on 44 weighted instances (`weighted_variant`
of the small pool, `debug_asserts` on, so the penalty array is compared against a full recomputation after
every operation) `routing="avoid"` agrees with `routing="bfs"` on the status and every partition is accepted
by the verifier. Whether it pays there is open — the weighted runs of E1/E2 are dominated by the same
`evaluate_deletion`, so the same family dependence is expected.

**Consequences for the conjectures.** C1's *exactness* half is P4. Its *performance* half — "choosing
per-vertex min-cost flows keeps the number of vertices affected by each deletion small throughout the run, so
the practical running time is dominated by the initial n flows" — is refuted on the minimally-connected
families and confirmed only where the graph has routing slack (see "Failed ideas"). C2 assumed flows "chosen
as in C1"; its premise therefore only holds on the sparse-with-slack family, where `Σ_i |U_i|` did fall by
45 % (205 754 -> 113 145 repairs for n = 10 000, k = 4), and the total is still ~2.8 n k, so C2's
`O(n k)` total remains open and is not contradicted by these runs.

Commands (all re-run for the numbers above; the driver is `/tmp/glrev/ab2.py`, one arm = build directory +
routing + environment, arms alternating run by run, each run a fresh subprocess reading `ru_maxrss`):
`AB_ARMS='{"bfs":{"build":"<new>","routing":"bfs"},"avoid":{"build":"<new>","routing":"avoid"}}' python
/tmp/glrev/ab2.py 5 solve 8 h1000 h2000 rr2000 rr10000 rr5000k8 er2000 sk10000 ce17` (the table; instances
`harary_graph(n, 4, seed=2)`, `random_regular_graph(n, d, k, seed=1)`, `erdos_renyi_graph(2000, 0.1, 6,
seed=1)`, `sparse_k_connected(10000, 4, seed=1)`, `glref.counterexample.build_counterexample_instance(17)`,
all through `glpartition(inst, algorithm="general", threads=8, verify=False, options={"routing": ...})`);
the same driver with `{"off":{...,"routing":"avoid"},"on":{...,"routing":"avoid","env":{"GLCORE_C1_REROUTE":
"1"}}}` and 3 reps (the re-routing pass); `python /tmp/glrev/verify_ab.py` (both options through the
independent verifier + partition comparison); and, for §3 above, the same driver with arms
`{"head":{"build":"<HEAD from git archive>"},"new":{"build":"<this tree>"}}` in modes `solve` (end to end),
`computeall` (`EssentialOracle.compute_all`, one thread) and `tightest` (`_core.tightest_cut` over every
vertex).

### E5-review. Adversarial review of E5: one wrong cost claim, one codegen trap, seven corrections (2026-09-23)

E5 was reviewed against separately built HEAD binaries. Every *semantic* claim survived; one *cost* claim did
not, and seven smaller statements were imprecise or untested (§4, §5). All of E5 above is the corrected version; this entry records
what was wrong, and why the wrong thing was invisible in the source.

**1. "Zero cost when off" was false (the major finding, reproduced).** With `routing="bfs"`, with every
counter, every stored path and every partition identical to the pre-C1 core, the first version of the C1 core
ran the *plain BFS* measurably slower. Reproduced here, 7 interleaved paired reps for the `compute_all`
rows and 5 for the others, fresh subprocess each, medians with per-rep ranges:

| benchmark | pre-review C1 core / HEAD | after the fix / HEAD |
|---|---:|---:|
| `EssentialOracle.compute_all`, 1 thread, rr10000 (pure augmenting search) | **1.135** (1.067-1.169) | **0.962** (0.934-1.035) |
| the same, sk10000 | **1.090** (1.031-1.119) | **0.950** (0.896-1.016) |
| `_core.tightest_cut` over every vertex, rr10000 | **1.166** (1.100-1.260) | 0.993 (0.970-1.050) |
| the same, sk10000 | **1.056** (1.006-1.065) | 0.957 (0.907-1.167) |
| end to end `solve_general`, 8 threads, 7 instances (5 paired reps) | 1.04-1.18 (reviewer's figures) | 0.89-1.03, counters identical |

**2. Root cause: two codegen traps, neither visible in the source.** Isolated by building one-line variants
of HEAD's `flow.cpp` and running the same paired benchmark against HEAD (rr10000, `compute_all`, min-of-7 to
suppress load noise):

| variant of HEAD's search | ratio vs HEAD |
|---|---:|
| HEAD + the two unused `Scratch` fields (`dist`, `queue_next`) | 1.02 |
| HEAD + the search moved into its own function | 1.03 |
| HEAD + the `relax` lambda (same loop) | 0.99 |
| HEAD + `for (;;) { for (; head < queue.size() && target < 0; ++head) {...} break; }`, **`head` declared outside** | **1.19** |
| the same but `head` declared *inside* the level loop | 0.98 |
| the C1 template with both instantiations inlined into `augment_once` | **1.16-1.22** |
| the C1 template with both instantiations behind never-inlined entry points | **0.90-0.93** |

So: (a) hoisting the queue index out of the scan loop — which the level loop of the 0-1 search seemed to
require — costs ~19 % on gcc 13.3 -O3/aarch64, although the emitted work is identical; and (b) letting
`augment_once` inline *both* template instantiations costs the plain one another ~10 %, while outlining both
makes it **faster** than the pre-C1 core, which inlined its single search (0.90-0.93 in the variant runs above,
0.95-0.96 in the final 7-rep measurement of the shipped code).

**3. The fix** (`src/glcore/flow.{hpp,cpp}`): `head` is declared inside the level loop — it restarts at 0 on
every level anyway, because a level swaps in a fresh queue — so the scan loop is the pre-C1 loop verbatim for
both instantiations; and `search_target<false>` / `search_target<true>` are reached only through
`search_plain` / `search_penalized`, both `[[gnu::noinline]]`. The `dist` array is also no longer allocated in
`new_scratch` (the plain BFS never touches it). The comments at both places state that this is a *measured*
requirement, so the next reader does not "simplify" it back. Consequence for E5's own table: the penalized
search was paying the same tax, so `avoid` improved too — the Harary slowdowns E5 reported (1.173 / 1.040)
are, on the fixed core, 0.974 / 0.989, i.e. no difference at all; the E5 table above was re-measured in full.

**Lesson (recorded because it will recur): `if constexpr` removes the code, not the cost.** A template that
shares its body with a second instantiation, and a loop that gains an enclosing loop, must be re-measured
against the code they replace — paired, interleaved, and against a separately built baseline binary.

**4. The other corrections.** (a) The `GLSolver::penalty_` comment claimed the array is "kept up to date even
when the option is off"; it is not — every maintenance site is guarded, so with `routing="bfs"` the array is
correct exactly at the two diagnostic points and stale in between (the diagnostics recompute it first, so
they remain valid and comparable). Comment fixed in `solver.hpp` for both solvers. (b) "Repairs per deletion"
and "augmentation count" were quoted as two corroborating signals; they are one measurement reported twice
(E5 item 4 now says so, and so does the `Stats` comment). (c) The re-routing pass's per-term breakdown had the
wrong sign on sk10000: `evaluate_deletion` goes **up** by ~0.2 s (1.065, range 1.029-1.082 over 3 paired
reps), not down by 0.2 s; only the wall-clock deltas reproduced. (d) Three table ratios were quoted more
precisely than the runs could resolve (er2000 1.025, ce17 1.022, rr5000k8 0.999); the table now carries the
per-rep range next to every median, and the re-measurement shows er2000/ce17 *are* slower on the search
(+22 % / +9 % on `evaluate_deletion` with identical counters) while rr5000k8 is a 2 % slowdown, not "no
change". (e) The claim that the load "consisted of this benchmark alone" was wrong — there is a
permanent background of long-lived interactive processes on this host — but the reviewer's figure for it
(~1.4 constant) did not reproduce: sampling `/proc/loadavg` every 20 s for 11 minutes after the last run, the
1-minute average decays to **0.30-0.37**, and `ps` accounts for ~0.4 CPU across those processes. The
background is bursty rather than constant (another agent session on the same host), which the E5 text now
says. The A/B design is interleaved, so no ratio changes; the absolute times carry whatever load there was.
(f) The re-routing pass read its environment switch into a function-local `static const bool`, so the two
settings could not be exercised in one process and nothing covered the pass at all; it is now a per-solver
flag read at construction (`c1_reroute_enabled()` in `solver_common.hpp`), and it is covered in-process as
well as in a subprocess. Its comment now also says what the pass really does: it recomputes the offending
flows from scratch, so their cuts come back *exact*, which can change later decisions that read `ess(v)` —
sound by P2, but more than a re-routing.

**5. The penalized search had no unit-test surface** — neither `_core.tightest_cut` nor `_core.EssentialOracle`
could install a penalty array, so every test of it went through a solver, where the only observables are
kappa/Ess/the partition. Added: `tightest_cut(graph, v, penalties)` and `EssentialOracle.set_penalties`
(the oracle owns the copy, so the raw pointer the engine keeps stays valid), plus four Python tests in
`tests/test_core_routing_review.py` — 6 740 penalized searches whose kappa / L-S-R sides / Ess must equal the
BFS ones (766 of them with a *different* path family, 1 685 with all-zero penalties reproducing the BFS paths
exactly), 3 296 single-terminal instances where the stored path's total penalty must equal an independent
0-1-BFS minimum (1 471 of them with a non-zero minimum, i.e. every path crosses a penalized arc), 680
oracle vertices with warm-started re-validation after a deletion compared against a BFS oracle, and the
re-routing pass toggled twice inside one process (`reroutes` 0 / 108 / 0). The C++ probe
(`tests/cxx/routing_probe.cpp`, 20 000 seeded residual states) is kept: it reaches residual states no Python
API can build.


### E6. Final sweep on the optimized core (2026-09-23, 1 151 rows, 0 invalid, 0 timeouts, 0 errors)

Re-run of the scaling, DAG and families grids after E3/E5, paired against the committed baseline on
`instance.sha256` (453/453 scaling and 252/252 DAG pairs are byte-identical instances, so the
comparison is like for like). Raw rows in `benchmarks/results/*_v2.jsonl`, figures in
`benchmarks/results/plots_v2/`, tables in `docs/benchmarks.md` §5.

**Harary is no longer memory-bound.** The baseline could not run it past n = 2 000 (56.5 GB); it now
reaches n = 10 000 in 418 s using 2.7 GB. Matched speed-ups at n ≥ 500: geometric mean 4.94× for the
general solver (range 2.94–8.73×) and 10.85× for the weighted solver, with memory 86–229× lower. The
fitted exponent falls from 3.3 to **2.76**.

**Unexpected: the DAG solver became 2–4× faster.** Nothing in E3 targeted it, but the flow-engine
micro-optimizations also shortened the CSR/topological pass that dominates it. At n = 10⁶
(m ≈ 5·10⁶) every variant/policy pair went from 1.96–2.37 s to **0.238–0.696 s** single-threaded.
Control: the pure-Python `reference-dag` on the very same instances is unchanged (0.74–0.99×), as it
must be since no Python changed.

**P1 refined.** E2 concluded heap ≈ stack "within noise". With the faster core the difference becomes
visible on *layered* DAGs, where the O(n+m) stack variant is reproducibly faster: 0.746–0.790× at
n = 10⁵ (median of 3 trials × 2 seeds, all three policies) and 0.672–0.688× at n = 10⁶. On the two
random k-T DAG families it remains a wash (0.85–1.25×). So P1's asymptotic gain is real but only
surfaces once the constant factors around it are small enough, and only where the in-lists are long.

**A measurement bug in our own harness, found and fixed.** `ru_maxrss` is inherited across `fork`, so
a worker spawned from a driver that had just built a 10⁷-arc instance reported the *driver's*
footprint as its own. Demonstration: a child of a 3 GB parent reports 2 879 MB peak when its true
peak is 18 MB. Every peak-memory figure in E2 and in the first pass of this sweep is therefore an
upper bound clamped to the driver's high-water mark, and the affected rows are exactly the ones that
report suspiciously identical values (4 093.7 / 4 106.5 MB). Both `benchmarks/runner.py` and
`glsolver.api` now read `VmHWM` from `/proc/self/status`, which is per-process and exec-accurate.
Corrected direct measurements (solver run in its own process, instance footprint separated):

| instance | arcs | instance in memory | peak total | solver's own share |
|---|---:|---:|---:|---:|
| Harary H_{4,10000} | 39 984 | 64 MB | 2 794 MB | ~2 730 MB |
| Harary H_{4,5000} | 19 984 | 56 MB | 755 MB | ~699 MB |
| Erdős–Rényi n = 10 000 | 10 000 610 | 2 574 MB | 2 864 MB | **~290 MB** |
| sparse k-connected n = 10 000 | 41 984 | 65 MB | 144 MB | ~79 MB |
| random 4-regular n = 10 000 | 39 984 | 65 MB | 102 MB | ~37 MB |

The dense instance is dominated by the *input representation*, not by the algorithm: the solver adds
290 MB on top of a 2.6 GB instance. The sparse high-diameter instance is the opposite, and is where
the certificates themselves cost memory.

## Failed ideas

* **Reusing an undirected global cut structure (Gomory–Hu / cactus) across the
  run.** Not applicable: the objects are *vertex* cuts between a vertex and a
  terminal *set* in a digraph that becomes asymmetric after the first deletion.
  Gomory–Hu trees model edge cuts of undirected graphs; the vertex-cut analogue
  does not exist in general. Only the initial symmetric state can exploit
  undirected structure (P3).
* **Skipping cut refresh after terminal removal when κ is unchanged.**
  Counterexample (paper_notes §13.3): new essential terminals appear.
* **C1 as a performance conjecture: "fan-routed certificates make most deletions free" (refuted 2026-09-23,
  measured in E5).** The conjecture, in its original wording:

  > In a `k`-connected undirected graph the fan lemma gives, for every vertex `v` and every set `P` of `k`
  > vertices, `k` vertex-disjoint `v→P` paths. Take `P` to be the matched pre-terminals `p_1..p_k` [Lem 7.8].
  > Then every `F_v` (`v ∉ P`) can be routed so that its paths reach `T` only through the matching arcs
  > `(p_i, t_i)`, making every secondary arc `e_i` unused by every flow but `F_{p_i}` (docs/optimizations.md
  > O1). Conjecture: choosing per-vertex *min-cost* flows (cost 1 on non-gateway out-arcs of pre-terminals, 0
  > elsewhere; same complexity as max-flow with 0–1 BFS) keeps the number of vertices affected by each
  > deletion small throughout the run, so the practical running time is dominated by the initial `n` flows.

  Implemented exactly so (`options["routing"] = "avoid"`, P4 for its exactness) and measured interleaved
  against the plain BFS on eight instances (E5). Counter-measurement: on Harary H_{4,1000} the number of flows
  repaired per deletion is **268.4 with the plain BFS and 268.1 with the penalized routing** (-0.1 %), the
  augmentation count 778 191 vs 777 422, and the wall clock 2.55 s vs 2.99 s (**17 % slower**); on random
  4-regular n = 2 000 the repairs *rise* by 5.5 % and the run is 22 % slower. The reason is structural, not a
  matter of tuning: the flaw in the fan-lemma argument is that the fan only controls how the paths *arrive* at
  `P`, while the cost that dominates the run comes from paths that *transit* the neighbourhood of a terminal
  on their way to a different terminal. In a graph whose degree equals `k` (Harary, `d`-regular with `k = d`)
  those transits have no alternative route, so no routing can avoid the deletable arcs, and the deletions keep
  touching hundreds of flows. The conjecture's conclusion ("the running time is dominated by the initial `n`
  flows") is false on every instance measured: `compute_all` is 0.6-14 % of the run on the seven sparse
  instances (Harary H_{4,2000}: 0.7 %, sparse k-connected n = 10 000: 10-13 %) and 44 % only on the dense
  Erdos-Renyi instance, where the routing changes no counter at all (but still costs 22 % on
  `evaluate_deletion`); `evaluate_deletion` still takes 67-93 % of every sparse run in both modes.
  What survives is the weaker statement recorded in E5: where the graph has routing *slack* (Harary plus
  random chords), the same routing removes 41 % of the augmentations and 45 % of the repairs and is 20 %
  faster — so the option is kept, but not as a default.
