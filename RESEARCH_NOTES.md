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

## Conjectures

### C1. Fan-routed certificates make most deletions free (2026-09-21)

In a `k`-connected undirected graph the fan lemma gives, for every vertex `v`
and every set `P` of `k` vertices, `k` vertex-disjoint `v→P` paths. Take `P` to
be the matched pre-terminals `p_1..p_k` [Lem 7.8]. Then every `F_v` (`v ∉ P`)
can be routed so that its paths reach `T` only through the matching arcs
`(p_i, t_i)`, making every secondary arc `e_i` unused by every flow but
`F_{p_i}` (docs/optimizations.md O1). Conjecture: choosing per-vertex
*min-cost* flows (cost 1 on non-gateway out-arcs of pre-terminals, 0 elsewhere;
same complexity as max-flow with 0–1 BFS) keeps the number of vertices affected
by each deletion small throughout the run, so the practical running time is
dominated by the initial `n` flows. Status: to be tested empirically (Stage I).

### C2. Amortized bound below O(n k m²) (2026-09-21)

With O1–O2 the work per deletion step is `O(Σ_i |U_i| (n+m))` where
`U_i = {v : e_i ∈ F_v}`; the trivial bound `Σ_i |U_i| ≤ nk` reproduces the
paper's per-step cost. Conjecture: with flows chosen as in C1 and the greedy
contraction O5, the *total* number of flow repairs over the run is `O(n k)`
on `k`-connected undirected inputs, giving `O(n k (n+m))` overall. Open.

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

## Failed ideas

* **Reusing an undirected global cut structure (Gomory–Hu / cactus) across the
  run.** Not applicable: the objects are *vertex* cuts between a vertex and a
  terminal *set* in a digraph that becomes asymmetric after the first deletion.
  Gomory–Hu trees model edge cuts of undirected graphs; the vertex-cut analogue
  does not exist in general. Only the initial symmetric state can exploit
  undirected structure (P3).
* **Skipping cut refresh after terminal removal when κ is unchanged.**
  Counterexample (paper_notes §13.3): new essential terminals appear.

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

