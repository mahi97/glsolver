# Optimizations and their correctness arguments

Every optimization below either (a) leaves the sequence of *operations* of the
paper's algorithm unchanged and only computes the same quantities faster, or
(b) changes which FEAC-preserving operation is applied next, which the paper's
correctness proof allows (docs/paper_notes.md §13.1). Nothing here relaxes
exactness: every returned partition is still checked by the independent
verifier, and the paper's ShiftAssignment remains the exact fallback that
guarantees progress.

Notation as in `docs/paper_notes.md`. `F_v` denotes a stored maximum flow
(family of `κ_G(v)` vertex-disjoint paths) for vertex `v` in the current graph.

## O1. Unused-arc shortcut (exact, [Lem 4.3] generalized)

**Claim.** Let `D ⊆ E` and suppose `F_v` uses no arc of `D`. Then
`κ_{G\D}(v) = κ_G(v)` and `Ess_G(v) ⊆ Ess_{G\D}(v)`. Consequently no arc of
`D` is critical for any pair `(v, ·)`, and deleting all of `D` at once keeps
`φ(v)` essential for `v`.

*Proof.* `F_v` survives in `G\D`, so `κ_{G\D}(v) ≥ κ_G(v)`; deletion cannot
increase `κ`, so equality holds. If `t ∈ Ess_G(v)`, every maximum family in
`G` ends at `t`; a maximum family of `G\D` is a maximum family of `G` (same
size), hence ends at `t`; so `t ∈ Ess_{G\D}(v)`. ∎

Use: (1) in ShiftAssignment, `e_i` needs a criticality test only for the
vertices `v` with `e_i ∈ F_v`; (2) any set of arcs used by *no* stored flow can
be deleted simultaneously while keeping the current witness.

## O2. Warm-started criticality test (exact)

For `v` with `e ∈ F_v`: remove from `F_v` the single path using `e` (it is
unique: paths are vertex-disjoint and `e`'s tail has split capacity 1, unless
the tail is `v` itself, in which case `e` is the first arc of one path). The
remainder is a flow of value `κ−1` in `G\e`. Run **one** augmenting-path search
in the residual network of `G\e`: success ⇒ `κ_{G\e}(v) = κ` and (O1-style
argument with the new flow) `Ess_G(v) ⊆ Ess_{G\e}(v)`, so `e` is non-critical
for every `(v, ·)`; failure ⇒ `κ_{G\e}(v) = κ−1` and the residual reachability
of the current (maximum) flow yields the tightest cut of `G\e` [Prop 4.2], hence
`Ess_{G\e}(v)` and `crit(e, v, t) = [t ∈ Ess_G(v) ∧ t ∉ Ess_{G\e}(v)]`.
Cost `O(n+m)` instead of `O(k(n+m))`. The resulting flow is the new `F_v` for
`G\e` if `e` is deleted, so deletion needs no further recomputation.

## O3. No recomputation after contraction (exact, paper_notes §13.2)

After contracting an out-degree-1 pre-terminal `p` into `t`, `κ` and `Ess` of
every remaining vertex are unchanged (proof of [Lem 7.3] shows the realizable
endpoint sets coincide). Stored flows translate: a path `… → x → p → t` becomes
`… → x → t` (still vertex-disjoint from the others, since only that path used
`t`). Only flows passing through `p` are touched: `O(Σ_v [p ∈ F_v])`.

## O4. Terminal removal: drop, re-augment once, recompute the cut (exact)

When `t` with `c_t = 0` is removed, for every vertex `v`: if a path of `F_v`
ends at `t`, drop it (value `κ−1`) and run **one** augmentation in `G\t`
(it succeeds iff `t` was *not* essential for `v`; a path ending at `t` does
not by itself mean `t` is essential). Then recompute the tightest cut by one
reverse reachability search — this is needed for *every* `v`, even those whose
flow avoided `t`, because the tightest cut and hence `Ess` can change (§13.3).
Cost `O(n(n+m))`, at most `k` times overall.

## O5. Greedy contraction attempt (exact, changes the operation order; §13.1)

Choose a pre-terminal `p` with `(p, φ(p)) ∈ E` and `d^+(p) ≥ 2`. Let
`D = out(p) \ {(p, φ(p))}`. Deleting `D` is FEAC-preserving iff
`φ(v) ∈ Ess_{G\D}(v)` for all `v`. By O1 only `v` with `F_v ∩ D ≠ ∅` need a
test; each such `F_v` (for `v ≠ p`) uses exactly one arc of `D`, so O2 applies
with `G\D` in place of `G\e`. For `v = p`: `G\D` leaves `p` with the single arc
`(p, φ(p))`, so `κ = 1` and `Ess = {φ(p)}` — always fine. If every test passes,
delete `D` and contract `p` (this performs `|D|` deletions and one contraction
of the paper's algorithm in one step). If some test fails, fall back to O6/the
paper's ShiftAssignment. Correctness: identical to `|D|` consecutive
single-arc deletions that are each non-critical for `φ`? Not necessarily
*each* — but FEAC in `G\D` is exactly what the induction of
[Thm gl-partition-correctness] needs for the recursive call, so soundness holds
regardless of intermediate states.

Candidate order (heuristic, no effect on correctness): fewest vertices whose
flows use an arc of `D` (known from stored flows), then smallest `|D|`.

## O6. Lazy ShiftAssignment (exact)

The while-loop of [Alg 2] needs, per secondary arc `e_i`, (a) whether it is
critical for some `(v, φ(v))` and (b) if so one such `v_i`. Using O1/O2 this
needs `Ess_{G\e_i}(v)` only for `v ∈ U_i = {v : e_i ∈ F_v}`. After a cycle shift
the only changed pairs are `(v, φ(v))` for the shifted `v`, which are in some
`U_j`; their `Ess_{G\e_j}(v)` is already stored. So the table is computed once,
restricted to `∪_i {i} × U_i`, exactly as the paper's table restricted to the
entries that can be true.

Secondary-arc choice heuristic: pick, for each matched `p_i`, the arc
`(p_i, q)` used by the fewest stored flows (ties by index): fewer tests and a
higher chance of non-criticality. Any choice is valid [Alg 2 line 2].

## O7. Deleting several non-critical secondary arcs (exact)

If several `e_i` are non-critical for `φ` *and* the sets `U_i` are pairwise
disjoint, they can all be deleted at once: for `v ∈ U_i`, the O2 flow for
`G\e_i` avoids every other `e_j` (as `F_v` did and augmentation in `G\e_i`
avoided … ) — **not automatically true**: the re-augmented path could use
`e_j`. We therefore only batch arcs whose O2-recomputed flows avoid all other
batched arcs (checked explicitly). Otherwise delete one and continue.

## O8. Parallel evaluation

The per-vertex flow computations in O1/O2/O4/O5/O6 are independent and run in
parallel (OpenMP over `v`); results are written to per-vertex slots so the
outcome is deterministic. Sequential dependencies of the proof (which operation
is applied next, cycle shifts) are kept sequential.

## O9. Bitset essential sets

`Ess(v)` and criticality sets are bitsets over terminal indices (`k ≤ 64` in a
`uint64_t`, otherwise a small dynamic bitset), making witness checks `O(1)`.

## O10. DAG solver

Binary heaps give `O(m log n)` [Lem 9.6]. Keys pushed after popping `p` are
all smaller than `pos(p)` (they are in-neighbours), so a *stack of sorted
batches* would also work; see RESEARCH_NOTES.md for the `O(n+m)` question.

## Things that are NOT valid (recorded to prevent regressions)

* Skipping the FEAC re-validation after a *speculative* multi-arc deletion:
  intermediate single-arc non-criticality does **not** imply joint
  non-criticality (a second arc's deletion can make the first critical). Always
  test the final `G\D`.
* Assuming `Ess` is monotone under terminal removal: new essential terminals
  can appear (§13.3), so the *cut* must be recomputed even if `κ` is unchanged.
* Reusing `F_v` across an arc deletion without checking whether `F_v` used the
  arc.
