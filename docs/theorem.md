# The Győri–Lovász theorem in plain language

**Statement (Győri 1976, Lovász 1977).** Take a graph in which you cannot
disconnect the remaining vertices by deleting fewer than `k` vertices (a
*`k`-connected* graph). Pick any `k` distinct vertices `t_1, …, t_k` and any
positive integers `n_1, …, n_k` adding up to the number of vertices. Then the
vertex set can be split into `k` groups `V_1, …, V_k` such that group `i`
contains `t_i`, has exactly `n_i` vertices, and is connected on its own (the
subgraph induced by `V_i` is connected).

Think of a data centre whose network is `k`-connected: the theorem says you
can always carve out `k` connected sub-networks of *any* prescribed sizes,
each around a prescribed gateway node — regardless of the topology. The
condition is tight in two senses: `k`-connectivity is necessary for the
statement to hold for all choices (Győri 1981), and without it the problem is
NP-complete even for equal sizes (Dyer–Frieze 1985).

**Lovász's directed version.** Let `G` be a digraph and `T = {t_1, …, t_k}` a
set of terminals such that every other vertex has `k` paths to `k` distinct
terminals that share no vertex except their start (`G` is *`k`-connected to
`T`*). Then for any `n_i` summing to `|V|` there are disjoint in-arborescences
`A_1, …, A_k` (directed trees in which every vertex has a path to the root),
with root `t_i` and `n_i` vertices, covering all vertices. The undirected
theorem is the special case where every edge is replaced by two opposite
arcs.

**Weighted version** (Chen, Kleinberg, Lovász, Rajaraman, Sundaram, Vetta,
JACM 2007, existence only). Vertices carry integer weights `w_v`, terminals
carry capacities `c_t`, `Σ w ≤ Σ c`; parts may exceed their capacity by at
most `w_max − 1`, which is unavoidable.

**Why it was hard to compute.** Győri's proof is constructive but explores
"cascades" whose number can be exponential; Lovász's proof uses algebraic
topology and gives no algorithm. Polynomial algorithms were known only for
`k ≤ 4` and for special graph classes. Chandran, Cheung, Chudnovsky et al.
placed the problem in PLS; it was widely suspected to be PLS- or PPAD-hard.

**What arXiv 2608.30945 (Hajiaghayi, JafariRaviz, Kaviani, Mohammadkhani,
2026) changed.** It gives the first polynomial-time algorithm for all `k`, for
Lovász's directed version and for the weighted version, plus a near-linear
algorithm for DAGs. The key idea is the *flow-essential assignment*: a terminal
`t` is *essential* for a vertex `v` if deleting `t` reduces the number of
vertex-disjoint paths from `v` to the terminals. The algorithm maintains an
assignment of every vertex to an essential terminal respecting the exact
capacities and grows the parts by contracting vertices into terminals, deleting
arcs that are "non-critical" for the assignment whenever it is stuck, and
repairing the assignment by *cycle shifts* whose potential strictly decreases.
See `docs/algorithm.md` for the mechanics and `docs/paper_notes.md` for the
complete mapping of the paper onto this implementation.
