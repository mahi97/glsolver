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

(filled during Stage G–I; see benchmarks/ and FINAL_REPORT.md)

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
