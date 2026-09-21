"""Reference implementation of GLDAGPartition [Alg 5] of arXiv 2608.30945
(docs/paper_notes.md §9): the near-linear-time algorithm for ``k``-``T``-connected
DAGs, weighted or unweighted.

The DAG solver needs no FEAC machinery. Its precondition is [Lem 9.1
k-conn-dag]: a DAG is ``k``-``T``-connected iff every non-terminal has out-degree
``≥ k``. Each step contracts, into an active terminal ``t_i``, the
``≺``-earliest unused pre-terminal of the current part ``V_i`` (canonical
topological order [Def 9.2]); by [Lem 9.4 contract-exists-dag] it has no common
predecessor with ``t_i``, so the contraction keeps the contracted DAG
``k``-``T``-connected [Lem 9.3 dag-contract-theorem] and the parts satisfy
``Σ_{v∈V_i\\T} w_v ≤ c_i + w_max − 1`` [Lem 9.5 dag-algo-correctness].
"""
from __future__ import annotations

import heapq

from glsolver.instance import Instance

from .trace import Tracer
from .unweighted import InvariantError, RefResult, RefStats

POLICIES = ("max_residual", "round_robin", "first")


# ---------------------------------------------------------------------------
# structure tests
# ---------------------------------------------------------------------------


def _distinct_out_adjacency(inst: Instance) -> list[list[int]]:
    """Out-neighbour lists with duplicate arcs merged (first occurrence kept).

    The paper's digraphs are simple (paper_notes §2), so ``d^+(v) = |N^+(v)|``
    counts *distinct* out-neighbours. A normalized :class:`Instance` has no
    duplicate arcs; the structure tests below must nevertheless not mistake
    the arc multiplicity of a hand-built instance for its out-degree [Lem 9.1].
    """
    return [list(dict.fromkeys(nbrs)) for nbrs in inst.out_adjacency()]


def _kahn_nonterminals(inst: Instance) -> tuple[list[int], list[int]]:
    """Kahn's algorithm restricted to the non-terminals (arcs into terminals
    are irrelevant for acyclicity since terminals have no out-arcs).

    Returns ``(order, leftover)``: ``order`` is a topological order of the
    non-terminals that were processed (smallest available id first, for
    determinism) and ``leftover`` lists the non-terminals on or behind a cycle
    (empty iff acyclic). Duplicate arcs are merged first, so in-degrees and
    out-lists agree on distinct neighbours.
    """
    tset = set(inst.terminals)
    indeg = [0] * inst.n
    out = _distinct_out_adjacency(inst)
    for u in range(inst.n):
        for v in out[u]:
            if v not in tset:
                indeg[v] += 1
    ready = [v for v in range(inst.n) if v not in tset and indeg[v] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        v = heapq.heappop(ready)
        order.append(v)
        for x in out[v]:
            if x in tset:
                continue
            indeg[x] -= 1
            if indeg[x] == 0:
                heapq.heappush(ready, x)
    leftover = [v for v in range(inst.n) if v not in tset and indeg[v] > 0]
    return order, leftover


def is_dag(inst: Instance) -> bool:
    """``True`` iff the arc set of ``inst`` is acyclic."""
    return not _kahn_nonterminals(inst)[1]


def canonical_topological_order(inst: Instance) -> list[int]:
    """[Def 9.2] canonical topological order: the non-terminals in a topological
    order (Kahn's algorithm, smallest available id first) followed by the
    terminals in ``inst.terminals`` order. Raises ``ValueError`` on a cycle."""
    order, leftover = _kahn_nonterminals(inst)
    if leftover:
        raise ValueError(f"graph has a directed cycle through {leftover[:5]}; no topological order")
    return order + list(inst.terminals)


def _precondition_detail(inst: Instance) -> tuple[bool, int | None, str]:
    """[Lem 9.1] on the *distinct* out-neighbour sets, then acyclicity.

    Well-formedness (no duplicate arcs, no self-loops, ...) is not checked
    here; :func:`gl_dag_partition` calls ``inst.validate()`` for that.
    """
    k = inst.k
    tset = set(inst.terminals)
    out = _distinct_out_adjacency(inst)  # d^+(v) = |N^+(v)|, not the arc multiplicity
    for v in range(inst.n):
        if v in tset:
            continue
        if len(out[v]) < k:
            return False, v, (
                f"non-terminal {v} has out-degree {len(out[v])} < k = {k}; "
                f"the DAG is not k-T-connected [Lem 9.1]"
            )
    _order, leftover = _kahn_nonterminals(inst)
    if leftover:
        v = leftover[0]
        return False, v, f"graph is not acyclic: vertex {v} lies on or behind a directed cycle"
    return True, None, "ok"


def dag_precondition(inst: Instance) -> tuple[bool, int | None]:
    """[Lem 9.1] precondition of [Alg 5]: ``inst`` is a DAG **and** every
    non-terminal has out-degree ``≥ k``. Returns ``(ok, offending_vertex)``
    where the vertex is a non-terminal of out-degree ``< k`` or a vertex on a
    cycle (``None`` when ``ok``).

    Out-degree means the number of *distinct* out-neighbours (paper_notes §2),
    so a hand-built instance with duplicate arcs is judged like its normalized
    form. This is a pure query and never validates the instance; malformed
    input is rejected by :func:`gl_dag_partition` (``inst.validate()``)."""
    ok, v, _why = _precondition_detail(inst)
    return ok, v


def is_k_t_connected_dag(inst: Instance) -> bool:
    """[Lem 9.1 k-conn-dag] (name used in the paper_notes §12 lemma map)."""
    return dag_precondition(inst)[0]


# ---------------------------------------------------------------------------
# [Alg 5] GLDAGPartition
# ---------------------------------------------------------------------------


def gl_dag_partition(
    inst: Instance,
    *,
    policy: str = "max_residual",
    tracer: Tracer | None = None,
    debug: bool = False,
) -> RefResult:
    """[Alg 5] GLDAGPartition (paper_notes §9).

    Per terminal ``t_i`` a min-heap ``H_i`` keyed by the position in the
    canonical topological order ``≺`` holds entries ``(pos[p], p, x)`` where
    ``x ∈ V_i`` is the head of the arc ``(p, x)`` that put ``p`` into the heap
    (so ``parents[p] = x`` is an original arc into the part, §9). Stale entries
    (``used[p]``) are skipped on extraction. ``policy`` chooses the active
    terminal (§13.7): ``"max_residual"`` (largest residual capacity, ties by
    terminal index), ``"round_robin"`` or ``"first"`` (smallest index); every
    policy is valid. Weights are supported: the residual capacity drops by
    ``w_p`` and the terminal becomes inactive once it is ``≤ 0``.

    Raises ``ValueError`` for an unknown ``policy`` or a structurally
    malformed instance (``inst.validate()``: duplicate arcs, self-loops, arcs
    leaving a terminal, wrong capacity/weight sums), so such input never
    reaches the heap loop. Returns ``status == "precondition_failed"`` when
    [Lem 9.1] fails (some out-degree ``< k`` or a cycle) with the offending
    vertex in ``message``.
    Debug mode checks, before each contraction, the hypothesis of [Lem 9.3]:
    ``p`` and ``t_i`` have no common predecessor among the unused vertices
    ([Lem 9.4]; ``O(in-degree × |out|)`` per step) and, after it, that no
    unused non-terminal lost out-degree in the contracted graph (simulated).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    inst.validate()  # well-formedness (duplicate arcs, self-loops, sums, ...): ValueError
    tracer = tracer or Tracer(enabled=False)
    stats = RefStats()
    ok, _bad, why = _precondition_detail(inst)
    if not ok:
        return RefResult("precondition_failed", [], {}, {}, stats, why, tracer.events if tracer.enabled else None)

    n, k = inst.n, inst.k
    terminals = list(inst.terminals)
    tset = set(terminals)
    order = canonical_topological_order(inst)
    pos = {v: i for i, v in enumerate(order)}
    in_adj = inst.in_adjacency()
    out_adj = inst.out_adjacency()
    if inst.weights is None:
        w = [1] * n
    else:
        w = [int(x) for x in inst.weights]
    residual = list(inst.capacities)
    parts: list[list[int]] = [[t] for t in terminals]
    part_of: list[int] = [-1] * n  # index of the part a used vertex (or terminal) belongs to
    for i, t in enumerate(terminals):
        part_of[t] = i
    used = [False] * n
    heaps: list[list[tuple[int, int, int]]] = []
    for t in terminals:
        h = [(pos[u], u, t) for u in in_adj[t]]
        heapq.heapify(h)
        stats.heap_pushes += len(h)
        heaps.append(h)
    active = [i for i in range(k) if residual[i] > 0]
    rr = 0
    r = n - k
    parents: dict[int, int] = {}

    tracer.record(
        "init", n=n, k=k, terminals=terminals, capacities=list(inst.capacities),
        arcs=[list(a) for a in inst.arcs], weights=list(w), order=order, policy=policy,
    )

    def _check_out_degrees() -> None:
        """Debug: every unused non-terminal keeps out-degree >= k in the contracted graph."""
        for u in range(n):
            if u in tset or used[u]:
                continue
            heads: set[int] = set()
            for y in out_adj[u]:
                if used[y] or y in tset:
                    heads.add(terminals[part_of[y]])
                else:
                    heads.add(y)
            if len(heads) < k:
                raise InvariantError(
                    f"[Lem 9.3] vertex {u} has contracted out-degree {len(heads)} < k = {k}"
                )

    while r > 0:
        if not active:
            raise InvariantError("no active terminal although non-terminals remain (Σw ≤ Σc violated)")
        if policy == "first":
            i = active[0]
        elif policy == "round_robin":
            i = active[rr % len(active)]
            rr += 1
        else:  # max_residual: largest residual capacity, ties by terminal index
            i = min(active, key=lambda j: (-residual[j], j))
        H = heaps[i]
        p = -1
        x = -1
        while H:
            _, cand, head = heapq.heappop(H)
            stats.heap_pops += 1
            if used[cand]:
                stats.stale_pops += 1
                continue
            p, x = cand, head
            break
        if p < 0:
            raise InvariantError(f"[Lem 9.4] heap of active terminal {terminals[i]} ran dry")
        if debug:  # [Lem 9.4]: no unused common predecessor of p and the part V_i
            for u in in_adj[p]:
                if used[u]:
                    continue
                if any((used[y] or y in tset) and part_of[y] == i for y in out_adj[u]):
                    raise InvariantError(
                        f"[Lem 9.4] {u} is a common predecessor of {p} and the part of {terminals[i]}"
                    )
        used[p] = True
        part_of[p] = i
        parts[i].append(p)
        parents[p] = x
        residual[i] -= w[p]
        r -= 1
        stats.contractions += 1
        for u in in_adj[p]:  # new pre-terminals of the enlarged part (stale if already used)
            heapq.heappush(H, (pos[u], u, p))
            stats.heap_pushes += 1
        tracer.record("dag_contract", p=p, t=terminals[i], parent=x, residual=residual[i])
        if residual[i] <= 0:
            active.remove(i)
        if debug:
            _check_out_degrees()

    stats.steps = stats.contractions
    tracer.record("done", parts={terminals[i]: sorted(parts[i]) for i in range(k)}, parents=dict(parents))
    return RefResult("ok", parts, parents, {}, stats, "ok", tracer.events if tracer.enabled else None)
