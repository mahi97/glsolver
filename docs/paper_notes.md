# Paper notes: arXiv 2608.30945 → implementation map

**Paper.** Hajiaghayi, JafariRaviz, Kaviani, Mohammadkhani. *Breaking the Exponential
Barrier: The First Polynomial-Time Algorithm for the Győri–Lovász Theorem.*
arXiv:2608.30945v1 (31 Aug 2026), 67 pages. Official supplementary code:
`mahdi-jfri/Gyori-Lovasz-Codes` (MIT), which contains only the appendix
counterexample (compact connectivity), not the algorithm.

This document is the **single source of truth** for the implementation. Every
non-trivial function in `reference/` and `src/` cites a label from this file
(`[Def X]`, `[Lem Y]`, `[Alg Z]`) in its docstring. Section 13 lists every
place where the paper is underspecified and records the derivation we use.

Notation: `n = |V|`, `m = |E|`, `k = |T|`. Vertices are `0..n-1` internally.
Terminal `t_i` is the vertex `terminals[i]`. `V\T` = non-terminals.

---

## 1. Problem formulations

### 1.1 Classical undirected Győri–Lovász (GL)

Input: undirected graph `G=(V,E)`, distinct terminals `t_1..t_k`, positive
integers `n_1..n_k` with `Σ n_i = |V|`. Output: partition `V_1..V_k` with
`t_i ∈ V_i`, `|V_i| = n_i`, each `G[V_i]` connected.
Guarantee (Győri 1976, Lovász 1977): exists whenever `G` is `k`-vertex-connected.

**Reduction to the directed formulation (paper §"Our results", first paragraph):**
replace each undirected edge `{u,v}` by the two arcs `(u,v)` and `(v,u)`, then
drop every arc leaving a terminal (terminals have no out-arcs, see §2). Set
`c_i = n_i − 1`. A `k`-connected graph is `k`-`T`-connected for every `T` of
size `k`. A part `V_i` whose induced *directed* subgraph is "connected to
`t_i`" (every vertex reaches `t_i` inside `V_i`) induces a connected undirected
subgraph, because the reverse arcs make every directed path an undirected path.
Conversely `G[V_i]` connected ⇒ every vertex reaches `t_i` in the symmetric
digraph restricted to `V_i` **except** that arcs out of terminals were dropped:
since `t_i` is the *root* (paths end there), dropping its out-arcs never
matters inside `V_i`. So the two notions coincide on symmetric digraphs.

### 1.2 Lovász's directed formulation (the setting of the paper)

`G=(V,E)` directed, terminals `T={t_1..t_k}`, capacities `c_i ≥ 0` integers with
`Σ c_i = |V\T|`. Output: partition `V_1..V_k` with `t_i ∈ V_i`, `|V_i| = c_i+1`,
and `G[V_i]` **connected to `t_i`**:

> **[Def connected-to]** For `v ∈ V'`, `G[V']` is connected to `v` if every
> `u ∈ V'` has a directed path to `v` inside `G[V']`. Equivalently `G[V']`
> contains a spanning in-arborescence rooted at `v`.

Precondition of Lovász/the paper's Theorem 1 [Thm k-t-conn]: `G` is
`k`-`T`-connected [Def 3.2 below]. The paper's Theorem 2 [Thm ess-assign-cond]
weakens this to the Flow-Essential Assignment Condition [Def 5.1].

Note: the paper allows `c_i = 0` (part is just `{t_i}`); undirected GL requires
`n_i ≥ 1` which is the same thing.

### 1.3 Weighted generalization (paper §5, Chen–Kleinberg–Lovász–Rajaraman–Sundaram–Vetta)

Each non-terminal `v` has integer weight `w_v ≥ 1`; each terminal `t` has
integer capacity `c_t ≥ 0`; `Σ_v w_v ≤ Σ_t c_t` (capacities may exceed total
weight). `w_max = max_v w_v`. Output: partition `⟨V_t⟩` with `t ∈ V_t`,
`G[V_t]` connected to `t`, and

    Σ_{v ∈ V_t \ T} w_v ≤ c_t + w_max − 1      for every t.   [Thm weighted-k-t-conn]

The additive slack `w_max − 1` is unavoidable (paper's example: `k` terminals
of capacity 1 and one vertex of weight `k` adjacent to all). Unit weights with
`Σ c = |V\T|` recover the exact unweighted theorem (the bound `|V_t\T| ≤ c_t`
plus the sum equality forces equality).

Precondition: `k`-`T`-connected [Thm weighted-k-t-conn], weakened to the
Flow-Essential Split-Assignment Condition [Def 5.3] in [Thm gyori-weighted-poly].

### 1.4 DAG special case (paper §6)

Same as 1.3 but `G` acyclic. Precondition `k`-`T`-connected ⇔ every non-terminal
has out-degree `≥ k` [Lem k-conn-dag]. Runtime `O(m log n)` [Thm dag-algo].
Unweighted DAG corollary [Cor gl-dag-unweighted]: exact sizes.

---

## 2. Graph conventions (paper §2 Preliminaries)

* Directed, **no self-loops, no parallel arcs** (simple).
* **Terminals have no outgoing arcs.** Any input arc leaving a terminal is
  dropped at load time (WLOG per paper).
* `N^+(v)`, `d^+(v)`: out-neighbours / out-degree.
* `G[V']`: induced subgraph.

> **[Def 2.1 pre-terminal, contraction]** A non-terminal `p` is a *pre-terminal*
> if it has an arc `(p,t)` to some terminal. `PT(G,S)` = set of pre-terminals
> having an arc to a terminal in `S ⊆ T`. *Contracting `p` into `t`* (requires
> `(p,t) ∈ E`): remove `p`; every incoming arc `(u,p)` becomes `(u,t)`; all
> outgoing arcs of `p` are deleted; duplicate arcs created by redirection are
> merged (keep one). Since `u ≠ t` always holds for an in-neighbour of `p`
> (terminals have no out-arcs), redirection never creates a self-loop.

Implementation object: `DiGraphState` (mutable) with operations
`delete_arc(u,v)`, `remove_terminal(t)`, `contract(p,t)`, and iteration over
`out(v)`, `in(v)`, `pre_terminals()`, `out_degree(v)`. Deleted vertices are
tombstoned (lazy deletion); a `live` bitmap is maintained. Out-neighbour sets
must support O(1) membership (for the duplicate-merge in contraction).

---

## 3. Terminal connectivity, cuts, tightest min cut

> **[Def 3.2 terminal connectivity]** For non-terminal `v`, `κ_G(v)` = maximum
> size of a family of paths that start at `v`, end at **distinct** terminals,
> and are pairwise vertex-disjoint except for the common start `v`.
> `G` is *k-T-connected* if `κ_G(v) = k` for every non-terminal `v`.

Note `κ_G(v) ≤ min(k, d^+(v))` and `κ_G(v) = 0` iff `v` reaches no terminal.

> **[Def 3.3 cut]** A cut separating `v` from `T` is a partition
> `C = (L_C, S_C, R_C)` of `V` with `v ∈ R_C`, `T ⊆ L_C ∪ S_C`, and **no arc
> from `R_C` to `L_C`**. `|C| = |S_C|`. Arcs from `S` or `L` anywhere, and from
> `R` into `S`, are allowed.

> **[Lem 3.4 Menger]** `κ_G(v) = min |C|` over cuts separating `v` from `T`.

> **[Def 3.5 union/intersection of cuts]** Membership tables (row = membership
> in `C_1`, column = in `C_2`):
>
> Union `C_1 ∪ C_2`:  L_1∪L_2 → L;  R_1∩R_2 → R;  rest → S.
> Intersection `C_1 ∩ C_2`:  L_1∩L_2 → L;  R_1∪R_2 → R;  rest → S.

> **[Lem 3.6]** Union and intersection of two cuts separating `v` are cuts
> separating `v`, and `|C_1|+|C_2| = |C_1∪C_2| + |C_1∩C_2|`.
> **[Cor 3.7]** Minimum cuts are closed under union and intersection.

> **[Def 3.8 tightest minimum cut]** The intersection of all minimum cuts
> separating `v` from `T` (smallest `L`, largest `R`). It is a minimum cut.
> **[Lem 3.9]** For the tightest min cut `C` and any min cut `C'`:
> `L_C ⊆ L_{C'}` and `R_{C'} ⊆ R_C`. Hence a terminal in the separator of *any*
> min cut is in `S_C`.

### 3.1 Computing `κ_G(v)` and the tightest min cut with one max-flow [Prop 4.2]

Network `H_v` (vertex splitting), `K = k + 1` (any value `> κ` works; we use
`k+1`):

* for each vertex `x`: nodes `x_in`, `x_out`, arc `(x_in, x_out)` capacity
  **1**, except capacity `K` when `x = v`;
* source `s`, arc `(s, v_in)` capacity `K`;
* for each arc `(x,y) ∈ E`: arc `(x_out, y_in)` capacity `K`;
* sink `z`; for each terminal `t`: arc `(t_out, z)` capacity `K`.

Max-flow value `= κ_G(v)`. Since the value is `≤ k`, **simple BFS augmenting
paths** (Ford–Fulkerson / Edmonds–Karp with at most `k` augmentations) run in
`O(k (n+m))`, which is the intended regime; Dinic is an alternative to benchmark.

Tightest min cut from an integral max flow `f`:
* Residual graph: forward arc `(x,y)` if `f < cap`, backward arc `(y,x)` if `f > 0`.
* `Reach` = set of network nodes that can **reach `z`** in the residual graph
  (reverse BFS from `z` over residual arcs). `s ∉ Reach`.
* `x_in ∈ Reach ⇒ x_out ∈ Reach` (paper proves this), so exactly three cases:
  `L = {x : x_in ∈ Reach}`, `S = {x : x_in ∉ Reach, x_out ∈ Reach}`,
  `R = {x : x_out ∉ Reach}`.
* `|S| = κ_G(v)`, `v ∈ R`, `T ⊆ L ∪ S`, and this is the tightest min cut.

Implementation object: `TightestCut {kappa, side[x] ∈ {L,S,R}}` and function
`tightest_min_cut(G, T, v) -> TightestCut` (plus the flow for warm starts).

---

## 4. Essential terminals

> **[Def 4.1 flow-essential terminal]** Terminal `t` is *essential* for
> non-terminal `v` if `κ_{G\{t}}(v) = κ_G(v) − 1`. Equivalently every
> maximum family of vertex-disjoint `v→T` paths contains a path ending at `t`.

> **[Lem 4.1 essential ⇔ tightest cut]** `t` is essential for `v` **iff**
> `t ∈ S_C` where `C` is the tightest min cut separating `v` from `T`.
> (Terminals are never in `R_C`; non-essential terminals are in `L_C`.)

So `Ess_G(v) = {t ∈ T : side[t] = S}` from one max-flow [Prop 4.2].
Cross-check used in tests: `Ess_G(v) = {t : κ_{G\{t}}(v) = κ_G(v) − 1}` with
`k+1` flows (removing `t` means deleting the vertex `t`, i.e. all its arcs).

Edge cases: `κ_G(v) = 0` ⇒ `S = ∅` ⇒ `Ess(v) = ∅`. A pre-terminal `p` with a
single out-arc `(p,t)` has `Ess(p) = {t}`.

> **[Lem 4.3 essential-connectivity]** For any arc `e`, `κ_{G\e}(v) ≥ κ_G(v) − 1`.
> If `κ_{G\e}(v) = κ_G(v)` then every terminal essential for `v` in `G` is
> still essential in `G\e`. (Generalises to any arc set `E'`: if
> `κ_{G\E'}(v) = κ_G(v)` then `Ess_G(v) ⊆ Ess_{G\E'}(v)`; proof identical.)

Consequence used for speed: an arc `e` not used by *some* maximum flow of `v`
cannot be critical for any pair `(v, ·)` (§6).

---

## 5. Flow-essential assignment conditions

> **[Def 5.0 assignment]** `φ : V\T → T`; `φ^{-1}(t)` = vertices assigned to `t`.

> **[Def 5.1 Flow-Essential Assignment Condition, FEAC]** `G` (with `T`, `c`)
> satisfies FEAC if there is `φ` with (1) `φ(v)` essential for `v` for every
> non-terminal `v`; (2) `|φ^{-1}(t)| = c_t` for every `t`. Such `φ` is a *witness*.
> Requires `Σ c_t = |V\T|`.

Witness computation (proof of [Thm ess-assign-cond]): bipartite network
source→`v` (cap 1), `v→t` (cap 1) iff `t ∈ Ess(v)`, `t→sink` (cap `c_t`); FEAC
holds iff max-flow `= |V\T|`. Implementation: Hopcroft–Karp style matching with
terminal multiplicities (`t` has `c_t` slots), or a plain unit-capacity flow.

> **[Def 5.2 split-assignment]** `ψ : (V\T)×T → ℕ` with `Σ_t ψ(v,t) = w_v`.
> `ψ(v) = {t : ψ(v,t) > 0}`, `ψ^{-1}(t) = {v : ψ(v,t) > 0}`.

> **[Def 5.3 Flow-Essential Split-Assignment Condition, FESAC]** exists `ψ`
> with (1) `ψ(v,t) > 0 ⇒ t` essential for `v`; (2) `Σ_v ψ(v,t) ≤ c_t` (upper
> bound, not equality).

> **[Prop 5.4 min-cost split assignment / Alg MinCostSplitAssignment]** Given
> costs `ξ_v(t)` on essential pairs, a witness minimising `Σ ψ(v,t) ξ_v(t)` is a
> min-cost flow of value `W = Σ w_v` in: `s→v` (cap `w_v`, cost 0); `v→t` (cap
> `w_v`, cost `ξ_v(t)`) for essential pairs only; `t→z` (cap `c_t`, cost 0).
> If no flow of value `W` exists ⇒ FESAC fails. Network has `O(nk)` arcs; costs
> `≤ k`. Implementation: successive shortest paths with potentials (Dijkstra) is
> adequate (`≤ n` augmentations if we augment by bottleneck; costs small).
> Must be **integral** (SSP on integral data is).

---

## 6. Criticality

> **[Def 6.1 critical arc]** Arc `e` is *critical for assigning `v` to `t`*
> (critical for the pair `(v,t)`) if `t` is essential for `v` in `G` but not
> in `G\e`. Otherwise `e` is non-critical for `(v,t)`. `e` is critical for an
> assignment `φ` if it is critical for some pair `(v, φ(v))`.

Computation: `crit(e, v, t) = [t ∈ Ess_G(v)] ∧ [t ∉ Ess_{G\e}(v)]`.
By [Lem 4.3], `crit(e,v,·)` can be true only if `κ_{G\e}(v) = κ_G(v) − 1`,
which requires `e` to be used by **every** maximum flow of `v` in `G`. Hence:

* if the stored max flow `F_v` in `G` does not use `e` ⇒ `e` non-critical for all `(v,·)`;
* otherwise compute the tightest cut in `G\e` (one flow; warm start: remove the
  one flow path through `e` from `F_v`, re-augment at most once — value is
  `κ` or `κ−1` — then reverse-BFS).

> **[Def 6.2 criticality cost]** Given matching `M = {(p_i,t_i)}` and secondary
> arcs `e_i = (p_i,q_i) ≠ (p_i,t_i)`, `ξ_v(t) = |{i : e_i critical for (v,t)}|`.
> **[Def 6.3 potential]** `Φ(φ) = Σ_v ξ_v(φ(v)) ≤ k·|V\T|`.
> **[Def weighted potential]** `Φ̄(ψ) = Σ_v Σ_t ψ(v,t) ξ_v(t)`.

---

## 7. The unweighted algorithm: GLPartition [Alg 1] + ShiftAssignment [Alg 2]

### 7.1 GLPartition(G, T, c, φ)   [Thm gl-partition-correctness]

Precondition: `(G,T,c,φ)` satisfies FEAC with witness `φ`.
Written iteratively (the paper's recursion is tail-recursive plus a stack of
"undo" records that build the parts):

```
parts[t] = {t} for all t            # final parts
loop:
  if no non-terminal remains: break
  (i)  if some terminal t with c_t == 0:            # [Lem 7.2 remove-zero-capacity-terminal]
         remove t from graph and from T;            # φ unchanged (no vertex assigned to t)
         parts[t] stays {t}; recompute Ess(v) for all v (see §13.3)
  (ii) elif some pre-terminal p with d^+(p) == 1, (p,t) its arc:   # [Lem 7.4 contract-degree-one]
         assert φ(p) == t                              # forced: Ess(p) = {t}
         contract p into t; c_t -= 1; delete p from φ; parts[t] += {p};
         record parent[p] = t  (arborescence edge)
         # Ess(v) of all other v is unchanged (§13.2)
  (iii) else:                                        # [Lem 7.5 non-critical-edge-exists]
         (φ, e_nc) = ShiftAssignment(G, T, c, φ)
         delete arc e_nc
return parts
```

Every iteration deletes a vertex or an arc ⇒ `≤ n + m` iterations.
Note (ii) records `p`'s part via the arc `(p,t)` that existed *at contraction
time*; in the original graph this corresponds to an arc from `p` to a vertex
already in `parts[t]` (either `t` itself or a previously contracted vertex that
was merged into `t`). Store `parent[p] = the original head of the arc` — see
§13.4 for how to recover the original head. The arborescence
`{(p, parent[p])}` certifies connectivity of each part.

Correctness lemmas:
* **[Lem 7.1 essential-survives-terminal-removal]** `t' ≠ t` essential for `v`
  in `G` ⇒ essential in `G\{t}`.
* **[Lem 7.2]** `c_t = 0` ⇒ removing `t` keeps FEAC with the same `φ`.
* **[Lem 7.3 essential-survives-degree-one-contraction]** `d^+(p)=1`,
  `(p,t)∈E`, contract ⇒ every essential terminal of every remaining `v` stays
  essential (proof shows the realizable endpoint sets are identical ⇒ `Ess`
  unchanged exactly).
* **[Lem 7.4]** contraction with `c_t -= 1`, `φ' = φ \ {p}` keeps FEAC.
* **[Lem 7.5]** if all `c_t > 0` and all pre-terminals have `d^+ ≥ 2` then
  ShiftAssignment returns `(φ', e_nc)` with FEAC holding in `G \ e_nc`.

### 7.2 ShiftAssignment(G, T, c, φ)   [Alg 2, Lem 7.5]

Requires `c_t > 0 ∀t` and `d^+(p) ≥ 2 ∀ p ∈ PT(G,T)`.

```
M = matching from PT(G,T) to T saturating T          # exists: [Lem 7.8 terminal-leaf-matching]
    i.e. distinct pre-terminals p_1..p_k with (p_i, t_i) ∈ E
for each i: choose secondary arc e_i = (p_i, q_i) ≠ (p_i, t_i)   # exists since d^+(p_i) ≥ 2
compute crit[i][v][t] for all i, v, t  (Def 6.1; only needed for t ∈ Ess_G(v))
loop:
  if ∃ i such that crit[i][v][φ(v)] is false for every v:
       return (φ, e_i)
  for each i: pick any v_i with crit[i][v_i][φ(v_i)]       # exists, else returned above
  # Reassignment graph R on T: arc φ(v_i) → t_i labelled v_i, for each i.
  # Every terminal has in-degree exactly 1 (heads t_1..t_k distinct) ⇒ R has a
  # directed cycle; no self-loops [Lem 7.10 own-edge-not-critical ⇒ φ(v_i) ≠ t_i]
  find a directed cycle C = (t_{i_1} → t_{i_2} → ... → t_{i_r} → t_{i_1}), r ≥ 2,
       by walking the unique in-arc backwards from any terminal until repeat
  # Shift: for each arc t_{i_{j-1}} → t_{i_j} on C, labelled v_{i_j}: φ(v_{i_j}) := t_{i_j}
  apply the shift                                          # [Lem 7.11 cycle-shift-decreases-potential]
```

Facts guaranteeing progress (all in paper §4):
* **[Lem 7.8 terminal-leaf-matching]** Under FEAC with all `c_t>0`, the
  bipartite graph terminals–pre-terminals has a matching saturating `T`.
  (Via [Lem 7.6 minimal-hall-set] and [Lem 7.7 hall-deficient-not-essential-outside].)
* **[Lem 7.9 critical-implies-assignable]** `e_i` critical for `(v,t)` ⇒ `t_i`
  essential for `v`. (So the shift keeps essentiality.)
* **[Lem 7.10 own-edge-not-critical]** `e_i` is never critical for `(v, t_i)`.
* **[Lem 7.12 critical-transfer]** `e_i` critical for `(v,t)` ⇒ every `e_j`
  critical for `(v,t_i)` is critical for `(v,t)`.
* **[Lem 7.11]** the shifted `φ'` is a witness and `Φ(φ') < Φ(φ)`; hence
  `≤ k|V\T|` shifts.

The graph does not change inside the loop, so `crit` is computed **once per
ShiftAssignment call**. Each loop iteration is `O(nk)` with the table.

**Debug assertions (enable in development builds):**
`A1` φ is a witness (essentiality + exact capacities) after every operation;
`A2` matching M is a valid saturating matching; `A3` e_i ≠ (p_i,t_i) and
`e_i ∈ E`; `A4` for the chosen v_i, `t_i ∈ Ess(v_i)` [Lem 7.9] and
`φ(v_i) ≠ t_i` [Lem 7.10]; `A5` `Φ` strictly decreases after each shift and
`ξ_{v_i}(t_i) < ξ_{v_i}(φ_old(v_i))`; `A6` after deleting `e_nc`,
`φ(v) ∈ Ess_{G\e_nc}(v)` for all v; `A7` after contraction, `Ess` unchanged for
all remaining v; `A8` after terminal removal, `φ(v) ∈ Ess(v)` still.

### 7.3 Initial witness

Compute `Ess_G(v)` for all `v` [Prop 4.2] and a saturating bipartite flow
(§5). If none exists ⇒ **FEAC fails**: the theorem's guarantee does not apply
(a partition may still exist — the algorithm cannot decide that; report
`precondition_failed`, never "no solution exists").
If `G` is `k`-`T`-connected, every terminal is essential for every vertex, so
any capacity-respecting assignment is a witness [proof of Thm k-t-conn].

---

## 8. The weighted algorithm: GLWeightedPartition [Alg 3]

Instance `(G,T,c,w)` satisfying FESAC [Def 5.3]. Output bound
`Σ_{v∈V_t\T} w_v ≤ c_t + w_max − 1`.

```
loop:
  if no non-terminal remains: break
  (i)  if ∃ t with c_t == 0: remove t (parts[t] = {t})            # [Lem 8.1 frac-remove-saturated-terminal]
  (ii) elif ∃ pre-terminal p with d^+(p)==1, arc (p,t):
         contract p into t; c_t -= w_p  (stays ≥ 0: ψ(p,t) = w_p ≤ c_t)   # [Lem 8.2]
         parts[t] += {p}
  (iii) elif ∃ matching M = {(p_i,t_i)} from PT(G,T) to T saturating T:      # [Lem 8.5 weighted-matching-removable-edge]
         choose secondary arcs e_i = (p_i,q_i) ≠ (p_i,t_i)
         ξ_v(t) = |{i : e_i critical for (v,t)}|  (for essential pairs)
         ψ = MinCostSplitAssignment(G,T,c,w,ξ)          # [Prop 5.4]; must be feasible under FESAC
         choose e_nc ∈ {e_i} that is critical for NO pair (v,t) with ψ(v,t) > 0   # exists: [Lem 8.4]
         delete e_nc
  (iv) else:  RoundAndRemove                                                   # [Alg 4, Lem 8.6, 8.7, 8.8]
         S = inclusion-minimal nonempty S ⊆ T with no matching saturating S
             (equivalently inclusion-minimal with |S| > |PT(G,S)|; then |PT(G,S)| = |S|−1 [Lem 7.6])
         pick t_S ∈ S (any)
         M' = matching from S\{t_S} to PT(G,S) saturating both sides  # exists [Lem 7.6]
         parts[t_S] = {t_S};  parts[t] = {t, M'(t)} for t ∈ S\{t_S}
         delete S ∪ PT(G,S) from the graph; T := T \ S
return parts
```

Finding the minimal Hall-deficient `S` (paper, proof of Thm gyori-weighted-poly):
start `S = T`; repeatedly scan `t ∈ S`, and if `S\{t}` still has no saturating
matching set `S := S\{t}` and restart the scan; stop when every `S\{t}` is
saturable. `≤ |T|^2` matching tests. Shortcut: a terminal with no pre-terminal
neighbour is a singleton deficient set.

Notes:
* In the weighted algorithm the witness `ψ` is **recomputed from scratch** at
  each step (iii) by min-cost flow; it is not carried along. FESAC is an
  existence property preserved by all four operations [Lem 8.1, 8.2, 8.5, 8.8],
  so step (iii)'s min-cost flow is always feasible.
* `c_t − w_p ≥ 0` in (ii) is guaranteed by FESAC [Lem 8.2].
* Only step (iv) can exceed `c_t` (by `≤ w_max − 1`) [Lem 8.6].
* Unit weights with `Σ c = |V\T|`: rounding never triggers (§13.6), so the
  weighted algorithm is an alternative exact unweighted algorithm; the paper says
  it is faster because the min-cost flow replaces `≤ kn` cycle shifts.
* Runtime [Prop 8.9]: `O(nkm^{2+o(1)} + m(nk)^{1+o(1)} log(n w_max) log n)`.

---

## 9. The DAG algorithm: GLDAGPartition [Alg 5]

Precondition: DAG, `k`-`T`-connected ⇔ `d^+(v) ≥ k` for all non-terminals
[Lem 9.1 k-conn-dag]; `Σ w_v ≤ Σ c_t`.

> **[Def 9.2 canonical topological order]** a topological order in which all
> non-terminals precede `t_1 ≺ t_2 ≺ … ≺ t_k` (terminals have no out-arcs, so
> this exists; any topological order of the non-terminals works).

```
compute canonical topological order ≺ (position pos[v])
for each i: ĉ_i = c_i; V_i = {t_i}; H_i = min-heap (by pos) of in-neighbours of t_i
active = {i : ĉ_i > 0}; r = |V\T|; used[v] = false
while r > 0:
    pick any i ∈ active                     # any policy; see §13.7 for our choice
    p = extract-min(H_i); if used[p]: continue   # stale entry
    V_i += {p}; ĉ_i -= w_p; used[p] = true; r -= 1; parent[p] = (the vertex of V_i that p's arc enters)
    push every in-neighbour of p into H_i
    if ĉ_i ≤ 0: active -= {i}
return V_1..V_k
```

Correctness: [Lem 9.3 dag-contract-theorem] (no common predecessor ⇒
contraction preserves out-degrees), [Lem 9.4 contract-exists-dag] (the
`≺`-earliest pre-terminal of `t` has no common predecessor with `t`),
[Lem 9.5 dag-algo-correctness]. An active terminal always exists while `r>0`
because `Σ w ≤ Σ c`. Each heap `H_i` never runs dry while `r>0` and `i` active
(paper proves the contracted graph stays `k`-`T`-connected).
Runtime: `O(m)` pushes/pops, `O(m log n)` total [Lem 9.6 algo-complexity].
`parent[p]`: `p` is in `H_i` because some arc `(p, x)` with `x ∈ V_i` was seen;
record `x` when pushing (push pairs `(pos[p], p, x)`; the first non-stale pop
wins). The arcs `(p, parent[p])` are original arcs and form the in-arborescence.

The DAG solver does **not** need FEAC machinery. A DAG that is *not*
`k`-`T`-connected (some out-degree `< k`) but satisfies FEAC is routed to the
general solver.

---

## 10. Appendix: relaxed conditions and the official counterexample

Definitions (unweighted, `Σ c_t = n − k`):
* internally vertex-disjoint paths to `T`: may share only `v` and terminal endpoints.
* **[Def A.2 local connectivity]** `v` locally connected to `T' ⊆ T` if `|T'|`
  internally vertex-disjoint `v→T'` paths exist (pre-terminals of `t ∈ T'`
  count as having arbitrarily many paths to `t`). `L(T')` = such `v`.
  Condition: `|L(T')| ≥ Σ_{t∈T'} c_t` for all `T'`.
* **[Def A.5 compact connectivity]** `v` compact-connected to `t` if there are
  `k` internally vertex-disjoint `v→T` paths with every terminal `t' ≠ t`
  receiving at most one path (pre-terminals of `t` are in `𝒞(t)` by
  convention). Condition: `|∪_{t∈T'} 𝒞(t)| ≥ Σ_{t∈T'} c_t` for all `T'` ⇔
  (Hall) a capacity-exact assignment `σ` with `v ∈ 𝒞(σ(v))` exists [Lem A.6].
* Hierarchy: `k`-`T`-connected ⇒ compact ⇒ local [Lem A.8]; compact ⇒ FEAC
  [Lem A.12] (compact-connected ⇒ essential [Lem A.11]); local ⇒ GL partition
  exists [Thm A.3] but only via an exponential Győri-style cascade argument
  (appendix B, Alg GLIncremental — **not** implemented as a solver; optional).
* Recognizing compact connectivity [Lem A.9]: for each pair `(v,t)` a
  vertex-split flow where the split arcs of `v` and `t` have capacity `k`, all
  others 1, arcs cap `k`, `t'_out → sink` cap `k`; `v ∈ 𝒞(t)` iff value `= k`.
  Then one bipartite flow. This is exactly the authors' `compact_connectivity.py`.

**Official counterexample** (README of the authors' repo + [Lem A.13]):
`k = 9` terminals. For each pair `i<j`: three pre-terminals `p^1_{ij}, p^2_{ij},
p^3_{ij}`, each with out-arcs to `t_i` and `t_j` only. For each arc
`e = (p^r_{x,y}, t_y)`: let `a<b<c<d` be the four smallest indices in
`[9]\{x,y}`; add **17 copies** (in the repo: `mult = 17` on a single vertex with
in-degree 0) of a forcing vertex `v_e` with `N^+(v_e) = P_{a,b} ∪ P_{a,c} ∪
{p^1_{d,x}, p^2_{d,x}, p^r_{x,y}}` (9 out-arcs). Capacities: `p^1,p^3` assigned
to `t_j`, `p^2` to `t_i`, all 17 copies of `v_e` to `t_a`; `c_t` = number of
vertices assigned to `t`. Claims verified by the script: the instance satisfies
compact connectivity; deleting **any** arc breaks it; contracting **any**
pre-terminal into either of its terminals breaks it. Size: 36 pairs × 3 = 108
pre-terminals, 216 pre-terminal arcs, 216 × 17 = 3672 forcing vertices (as 216
weighted vertices), `k=9`. The paper's Lemma A.13 uses one copy (`mult=1`) and
proves only "every arc is critical". The full 17-copy instance satisfies FEAC
(compact ⇒ FEAC) and our general solver must solve it — a strong regression test
of exactly the "why naive contraction/deletion fails" phenomenon.

---

## 11. Runtime bounds stated in the paper

| Result | Bound |
|---|---|
| essential terminals of one vertex [Prop 4.2] | one max-flow, `O((n+m)^{1+o(1)})` with vdBrand et al.; `O(k(n+m))` with augmenting paths |
| min-cost split assignment [Prop 5.4] | `O(n(n+m)^{1+o(1)} + (nk)^{1+o(1)} log(n w_max) log n)` |
| unweighted GLPartition | polynomial: `≤ n+m` steps, each `≤ k|V\T|` shifts + `O(nk)` flows |
| weighted [Prop 8.9] | `O(nkm^{2+o(1)} + m(nk)^{1+o(1)} log(n w_max) log n)` |
| DAG [Thm dag-algo] | `O(m log n)` |

---

## 12. Lemma → function map (to be kept in sync with the code)

| Paper item | Reference impl (`reference/glref`) | Optimized impl (`src/`) |
|---|---|---|
| Def 2.1 contraction | `graph.DiGraphState.contract` | `glcore::Graph::contract` |
| Prop 4.2 tightest cut / Ess | `essential.tightest_min_cut`, `essential.essential_terminals` | `glcore::EssentialOracle` |
| Def 4.1 (k+1 flows cross-check) | `essential.essential_terminals_by_definition` | tests only |
| Def 5.1 witness | `assignment.find_witness` | `glcore::find_witness` |
| Prop 5.4 min-cost split assignment | `weighted.min_cost_split_assignment` | `glcore::min_cost_split_assignment` |
| Def 6.1 criticality | `critical.is_critical`, `critical.criticality_table` | `glcore::CriticalityOracle` |
| Alg 1 GLPartition | `unweighted.gl_partition` | `glcore::GLSolver::run` |
| Alg 2 ShiftAssignment | `unweighted.shift_assignment` | `glcore::GLSolver::shift_assignment` |
| Lem 7.8 matching | `matching.saturating_matching` | `glcore::hopcroft_karp` |
| Alg 3 GLWeightedPartition | `weighted.gl_weighted_partition` | `glcore::GLWeightedSolver` |
| Alg 4 RoundAndRemove | `weighted.round_and_remove` | same |
| Alg 5 GLDAGPartition | `dag.gl_dag_partition` | `glcore::dag_partition` |
| Lem 9.1 out-degree test | `dag.is_k_t_connected_dag` | same |
| Lem A.9 compact connectivity | `compact.compact_sets`, `compact.satisfies_condition` | — |
| Verifier (independent) | `glsolver.verify.verify_partition` | pure Python, no shared code |

---

## 13. Underspecified points and our derivations (validated by tests)

**13.1 Any FEAC-preserving arc deletion is allowed.** The correctness proof
[Thm gl-partition-correctness] is an induction that only needs FEAC at every
step plus the three extension lemmas. ShiftAssignment is how the paper
*guarantees* a deletable arc exists; but deleting *any* arc `e` that is
non-critical for the current `φ` (i.e. `φ(v) ∈ Ess_{G\e}(v)` for all `v`) keeps
FEAC and is therefore valid. Moreover a *set* `D` of arcs may be deleted at once
if `φ(v) ∈ Ess_{G\D}(v)` for all `v` (same lemma applied to `G\D` directly;
checkable with one flow per `v` in `G\D`, and free for every `v` whose stored max
flow avoids `D` by the generalised [Lem 4.3]). The optimized solver uses this
for heuristics (e.g. "try to make `p` out-degree 1 by deleting
`out(p) \ {(p, φ(p))}` when `(p, φ(p)) ∈ E`"); the paper's ShiftAssignment is
the exact fallback that guarantees progress. The **reference** implementation
follows the paper literally.

**13.2 Contraction leaves `Ess` unchanged for every remaining vertex.** The
proof of [Lem 7.3] shows the families of realizable terminal-endpoint sets are
the same in `G` and `G'`; therefore `κ` and `Ess` are identical (both
directions). So no recomputation is needed after (ii). Tests assert this.

**13.3 Terminal removal can create new essential terminals.** Example: `v→t_1`,
`v→x`, `x→t_2`, `x→t_3`: `Ess(v) = {t_1}`; after removing `t_2` (capacity 0),
`Ess(v) = {t_1,t_3}`. Old essentials survive [Lem 7.1], so `φ` stays valid, but
`Ess` must be **recomputed** after step (i) (at most `k` times overall). Also
`κ(v)` may drop by one.

**13.4 Recovering original arcs after contraction.** When `p` is contracted
into `t`, an arc `(u,p)` becomes `(u,t)`. To rebuild the in-arborescence in
the *original* graph, each live arc `(u,t)` into a terminal must remember the
original head. We keep, per live arc into a terminal, `orig_head` (initially the
terminal itself; on redirect, `orig_head[(u,t)] = p` **only if** `(u,t)` did not
already exist — if it did, the existing arc and its `orig_head` are kept, which
is still a valid original arc from `u` into the part). Contracting `p` with its
unique arc `(p,t)` sets `parent[p] = orig_head[(p,t)]`. Correctness: `parent[p]`
is an original out-neighbour of `p` that is already in `parts[t]`.

**13.5 Cycle detection in the reassignment graph.** `R` has exactly `k` arcs,
one entering each terminal; in-degree 1 everywhere; walk `pred` pointers from any
terminal; the first repeated terminal closes a cycle of length `≥ 2` (self-loops
impossible by [Lem 7.10]). Any cycle works.

**13.6 Rounding never triggers on unit-weight, tight instances.** With `w ≡ 1`
and `Σ c = |V\T|`, a FESAC witness saturates every terminal; a Hall-deficient
`S` with `|PT(G,S)| = |S|−1` would need `≥ |S|` distinct vertices in `PT(G,S)`
sending unit weight to `S` [Lem 7.7] — impossible. So the weighted algorithm on
unweighted input performs only (i)–(iii). Tested.

**13.7 Choice of active terminal in the DAG algorithm.** Any policy is correct.
We use "terminal with the largest residual capacity" (max-heap) by default and
round-robin as an option; benchmark both. With unit weights the choice cannot
change validity.

**13.8 Choice of `v_i`, of the matching, and of secondary arcs.** All choices
are valid. Reference: first found. Optimized: heuristics (§docs/optimizations.md).

**13.9 Inputs with `κ(v)=0`** (vertex cannot reach `T`): `Ess(v)=∅`, FEAC fails,
solver reports `precondition_failed` with the offending vertex.

**13.10 `k` may be `1`**: `c_1 = |V\T|`, any vertex reaching `t_1` has
`Ess = {t_1}`; algorithm degenerates to contractions along a BFS tree.

**13.11 Undirected verification.** The verifier for undirected input checks
undirected connectivity of `G[V_i]` in the *original undirected graph*; the
directed verifier checks reachability to `t_i` inside `G[V_i]` following
original arcs. Both are implemented independently of the solver.

**13.12 Weighted output is validated** against `c_t + w_max − 1`; in addition
the verifier reports which parts exceed `c_t` (only rounding can cause it).

---

## 14. Trace model (for visualization and invariant debugging)

The reference and optimized solvers can emit an ordered list of JSON events
with the *full* state needed to replay the algorithm on small graphs:

```
{"type":"init", "n":..,"k":..,"terminals":[..],"capacities":[..],"arcs":[[u,v],..]}
{"type":"essential", "ess": {v: [t,..]}, "kappa": {v: κ}}
{"type":"witness", "phi": {v: t}}
{"type":"remove_terminal", "t": t}
{"type":"contract", "p": p, "t": t, "parent": x, "capacities":[..]}
{"type":"matching", "pairs": [[p_i,t_i],..], "secondary": [[p_i,q_i],..]}
{"type":"criticality", "crit": [[i,v,t],..]}        # only true entries
{"type":"reassignment_graph", "arcs": [[phi(v_i), t_i, v_i],..], "cycle": [i..]}
{"type":"cycle_shift", "changes": [[v, old_t, new_t],..], "potential_before":.., "potential_after":..}
{"type":"delete_arc", "u":u,"v":v}
{"type":"round_and_remove", "S":[..], "pairs":[[t,p],..], "t_S": t}
{"type":"min_cost_split", "psi": [[v,t,units],..], "cost":..}
{"type":"dag_contract", "p":p, "t":t, "parent":x}
{"type":"done", "parts": {t: [..]}}
```

Every event may carry `"cuts": {v: {"L":[..],"S":[..],"R":[..]}}` when the
tracer is asked to record tightest cuts.
