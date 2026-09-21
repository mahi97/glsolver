"""Compact connectivity [Def A.5] and local connectivity [Def A.2] of the paper's
appendix (docs/paper_notes.md §10), ported from the authors' official
``compact_connectivity.py`` (repo ``mahdi-jfri/Gyori-Lovasz-Codes``) with the
**same semantics**, but on :class:`glref.flow.FlowNetwork` instead of networkx.

Conventions of the authors' script, kept exactly:

* vertices are ``0..n-1``, the terminals are ``0..k-1`` and have no out-arcs;
* ``mult[v]`` is the number of identical copies of ``v``; a vertex with
  ``mult[v] > 1`` must have in-degree 0 (so it is never an internal vertex of
  a path and the copies are interchangeable);
* ``cap[t]`` is the number of (copies of) non-terminals to be assigned to ``t``.

``v ∈ 𝒞(t)`` is decided by the vertex-split flow of [Lem A.9] (split arcs of
``v`` and ``t`` have capacity ``k``, others ``mult[x]``, all remaining arcs
``k``; ``v ∈ 𝒞(t)`` iff the max-flow value is ``k``). The condition itself
[Def A.5 / Lem A.6] is one bipartite flow with multiplicities.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Sequence

from glsolver.instance import Instance

from .flow import FlowNetwork


class CompactInstance:
    """Mirror of the authors' ``Instance``: ``k`` terminals ``0..k-1``,
    out-adjacency lists, multiplicities and capacities."""

    def __init__(
        self, k: int, out_adj: Sequence[Iterable[int]], mult: Sequence[int], cap: Sequence[int]
    ) -> None:
        self.k = k
        self.n = len(out_adj)
        self.out_adj: list[list[int]] = [list(vs) for vs in out_adj]
        self.mult: list[int] = list(mult)
        self.cap: list[int] = list(cap)
        if len(self.mult) != self.n or len(self.cap) != k:
            raise ValueError("mult must have one entry per vertex and cap one per terminal")
        self.in_adj: list[list[int]] = [[] for _ in range(self.n)]
        for u in range(self.n):
            if u < k and self.out_adj[u]:
                raise ValueError(f"terminal {u} has out-arcs")
            for v in self.out_adj[u]:
                self.in_adj[v].append(u)
        if not all(self.mult[v] == 1 or not self.in_adj[v] for v in range(self.n)):
            raise ValueError("a vertex with mult > 1 must have in-degree 0")

    # ----------------------------------------------------------- accessors
    def terminals(self) -> range:
        return range(self.k)

    def non_terminals(self) -> range:
        return range(self.k, self.n)

    def total_mult(self) -> int:
        return sum(self.mult[v] for v in self.non_terminals())

    def total_cap(self) -> int:
        return sum(self.cap)

    def num_arcs(self) -> int:
        return sum(len(vs) for vs in self.out_adj)

    def pre_terminals(self) -> list[int]:
        """Non-terminals with an arc to some terminal [Def 2.1]."""
        return [v for v in self.non_terminals() if any(t < self.k for t in self.out_adj[v])]

    # --------------------------------------------------------- conversions
    @classmethod
    def from_instance(cls, inst: Instance) -> tuple[CompactInstance, dict[int, int]]:
        """Convert an unweighted :class:`Instance` (all ``mult = 1``). Terminals
        are remapped to ``0..k-1`` in ``inst.terminals`` order and the
        non-terminals to ``k..n-1`` in increasing id order. Returns
        ``(ci, new_id)`` with ``new_id[original_vertex] = compact_vertex``."""
        if inst.weights is not None:
            raise ValueError("compact connectivity is defined for unweighted instances")
        new_id: dict[int, int] = {t: i for i, t in enumerate(inst.terminals)}
        for v in range(inst.n):
            if v not in new_id:
                new_id[v] = len(new_id)
        out_adj: list[list[int]] = [[] for _ in range(inst.n)]
        for u, v in inst.arcs:
            out_adj[new_id[u]].append(new_id[v])
        ci = cls(inst.k, out_adj, [1] * inst.n, list(inst.capacities))
        return ci, new_id

    def to_instance(self, name: str = "") -> Instance:
        """Expand the multiplicities into distinct vertices and return a
        directed unweighted :class:`Instance` with terminals ``0..k-1``.
        ``inst.meta["compact_vertices"][v]`` lists the instance vertices that
        copy compact vertex ``v``."""
        copies: dict[int, list[int]] = {}
        nxt = self.k
        for t in self.terminals():
            copies[t] = [t]
        for v in self.non_terminals():
            copies[v] = list(range(nxt, nxt + self.mult[v]))
            nxt += self.mult[v]
        arcs: list[tuple[int, int]] = []
        for u in self.non_terminals():
            for cu in copies[u]:
                for y in self.out_adj[u]:
                    if len(copies[y]) != 1:
                        raise ValueError("arc into a vertex with mult > 1 (must have in-degree 0)")
                    arcs.append((cu, copies[y][0]))
        inst = Instance(
            n=nxt,
            arcs=tuple(sorted(set(arcs))),
            terminals=tuple(range(self.k)),
            capacities=tuple(self.cap),
            weights=None,
            directed=True,
            name=name,
            meta={"compact_vertices": {v: list(c) for v, c in copies.items()}},
        )
        inst.validate()
        return inst

    def copy(self) -> CompactInstance:
        return CompactInstance(self.k, self.out_adj, self.mult, self.cap)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CompactInstance(k={self.k}, n={self.n}, arcs={self.num_arcs()}, cap={self.cap})"


# ---------------------------------------------------------------------------
# compact connectivity [Def A.5], recognition [Lem A.9]
# ---------------------------------------------------------------------------


def compact_connected(ci: CompactInstance, v: int, t: int) -> bool:
    """Whether ``v ∈ 𝒞(t)`` [Def A.5] via the vertex-split flow of [Lem A.9].

    Only vertices reachable from ``v`` are put into the network (as in the
    authors' script). Split arc ``x_in→x_out`` has capacity ``k`` for
    ``x ∈ {v, t}`` and ``mult[x]`` otherwise; ``source→v_in``, every graph arc
    and every ``t'_out→sink`` have capacity ``k``. ``v ∈ 𝒞(t)`` iff the
    max-flow value is ``k`` (pre-terminals of ``t`` are included automatically,
    matching the convention of [Def A.5]).
    """
    k = ci.k
    reach: list[int] = [v]
    seen = {v}
    stack = [v]
    while stack:
        x = stack.pop()
        if x < k:
            continue
        for y in ci.out_adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
                reach.append(y)
    ids = {x: i for i, x in enumerate(reach)}
    net = FlowNetwork(2 * len(reach) + 2)
    s, z = 2 * len(reach), 2 * len(reach) + 1
    for x in reach:
        i = ids[x]
        net.add_arc(2 * i, 2 * i + 1, k if x in (v, t) else ci.mult[x])
        if x < k:
            net.add_arc(2 * i + 1, z, k)
            continue
        for y in ci.out_adj[x]:
            net.add_arc(2 * i + 1, 2 * ids[y], k)
    net.add_arc(s, 2 * ids[v], k)
    return net.max_flow(s, z, limit=k) >= k


def compact_sets(ci: CompactInstance, vertices: Iterable[int] | None = None) -> dict[int, frozenset[int]]:
    """``{v: 𝒞^{-1}(v)}`` = ``{v: {t : v ∈ 𝒞(t)}}`` for the given non-terminals
    (default: all of them)."""
    if vertices is None:
        vertices = ci.non_terminals()
    return {
        v: frozenset(t for t in ci.terminals() if compact_connected(ci, v, t)) for v in vertices
    }


def _dinic_max_flow(net: FlowNetwork, s: int, z: int) -> int:
    """Dinic's blocking flows on a :class:`FlowNetwork` (used for the bipartite
    condition network, where Edmonds–Karp needs thousands of augmentations)."""
    from collections import deque

    n = net.num_nodes

    def dfs(u: int, pushed: int, level: list[int], it: list[int]) -> int:
        if u == z:
            return pushed
        while it[u] < len(net.adj[u]):
            a = net.adj[u][it[u]]
            v = net.head[a]
            r = net.cap[a] - net.flow[a]
            if r > 0 and level[v] == level[u] + 1:
                got = dfs(v, min(pushed, r), level, it)
                if got > 0:
                    net.flow[a] += got
                    net.flow[a ^ 1] -= got
                    return got
            it[u] += 1
        return 0

    total = 0
    while True:
        level = [-1] * n
        level[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for a in net.adj[u]:
                v = net.head[a]
                if level[v] < 0 and net.cap[a] - net.flow[a] > 0:
                    level[v] = level[u] + 1
                    dq.append(v)
        if level[z] < 0:
            return total
        it = [0] * n
        while True:
            f = dfs(s, 1 << 62, level, it)
            if f == 0:
                break
            total += f


def _condition_network(
    ci: CompactInstance, sets: dict[int, frozenset[int]]
) -> tuple[FlowNetwork, int, int, dict[tuple[int, int], int]] | None:
    """Bipartite network of [Lem A.6]/[Lem A.9]: ``s→v`` (``mult[v]``),
    ``v→t`` (``mult[v]``) iff ``t ∈ sets[v]``, ``t→z`` (``cap[t]``). ``None`` if
    some non-terminal is compact-connected to no terminal."""
    nt = list(ci.non_terminals())
    s, z = 0, 1
    vid = {v: 2 + i for i, v in enumerate(nt)}
    tid = {t: 2 + len(nt) + t for t in ci.terminals()}
    net = FlowNetwork(2 + len(nt) + ci.k)
    arc_of: dict[tuple[int, int], int] = {}
    for t in ci.terminals():
        net.add_arc(tid[t], z, ci.cap[t])
    for v in nt:
        if not sets[v]:
            return None
        net.add_arc(s, vid[v], ci.mult[v])
        for t in sorted(sets[v]):
            arc_of[(v, t)] = net.add_arc(vid[v], tid[t], ci.mult[v])
    return net, s, z, arc_of


def satisfies_compact_condition(ci: CompactInstance, sets: dict[int, frozenset[int]] | None = None) -> bool:
    """The compact connectivity condition [Def A.5], tested as in the authors'
    ``satisfies_condition``: requires ``Σ cap = Σ mult``; then the bipartite
    flow (``s→v`` cap ``mult[v]``, ``v→t`` cap ``mult[v]`` iff ``v ∈ 𝒞(t)``,
    ``t→z`` cap ``cap[t]``) has value ``Σ mult`` iff the condition holds
    [Lem A.6]. A non-terminal with empty ``𝒞^{-1}(v)`` fails at once."""
    if ci.total_cap() != ci.total_mult():
        return False
    if sets is None:
        sets = compact_sets(ci)
    built = _condition_network(ci, sets)
    if built is None:
        return False
    net, s, z, _ = built
    return _dinic_max_flow(net, s, z) == ci.total_mult()


satisfies_condition = satisfies_compact_condition  # name used in paper_notes §12 and by the authors


def compact_witness(
    ci: CompactInstance, sets: dict[int, frozenset[int]] | None = None
) -> dict[tuple[int, int], int] | None:
    """A witness ``σ`` of [Lem A.6] as ``{(v, t): copies of v assigned to t}``
    (all copies of ``v`` go to one terminal when ``mult[v] = 1``), or ``None``."""
    if ci.total_cap() != ci.total_mult():
        return None
    if sets is None:
        sets = compact_sets(ci)
    built = _condition_network(ci, sets)
    if built is None:
        return None
    net, s, z, arc_of = built
    if _dinic_max_flow(net, s, z) != ci.total_mult():
        return None
    return {(v, t): net.flow[a] for (v, t), a in arc_of.items() if net.flow[a] > 0}


# ---------------------------------------------------------------------------
# mutations (exactly as in the authors' script)
# ---------------------------------------------------------------------------


def delete_edge(ci: CompactInstance, u: int, v: int) -> CompactInstance:
    """``G − (u, v)`` with ``T``, ``mult`` and ``cap`` unchanged."""
    out_adj = [list(vs) for vs in ci.out_adj]
    out_adj[u].remove(v)
    return CompactInstance(ci.k, out_adj, ci.mult, ci.cap)


def contract(ci: CompactInstance, p: int, t: int) -> tuple[CompactInstance, dict[int, int]]:
    """Contract the pre-terminal ``p`` into the terminal ``t`` [Def 2.1]: the
    in-arcs of ``p`` are redirected to ``t`` (duplicates merged, no
    self-loops), ``p`` disappears, ``cap[t] -= mult[p]``. Returns the new
    instance and the map from old to new vertex numbers (``p`` is dropped,
    later vertices shift down by one)."""
    if not (t in ci.out_adj[p] and t < ci.k <= p):
        raise ValueError(f"({p},{t}) must be an arc from a non-terminal into a terminal")
    new_id = {x: (x if x < p else x - 1) for x in range(ci.n) if x != p}
    out_adj: list[list[int]] = [[] for _ in range(ci.n - 1)]
    for x in new_id:
        targets: list[int] = []
        for y in ci.out_adj[x]:
            zz = new_id[t if y == p else y]
            if zz != new_id[x] and zz not in targets:
                targets.append(zz)
        out_adj[new_id[x]] = targets
    mult = [ci.mult[x] for x in new_id]
    cap = list(ci.cap)
    cap[t] -= ci.mult[p]
    return CompactInstance(ci.k, out_adj, mult, cap), new_id


# ---------------------------------------------------------------------------
# local connectivity [Def A.2]
# ---------------------------------------------------------------------------


def is_locally_connected(ci: CompactInstance, v: int, Tprime: Iterable[int]) -> bool:
    """[Def A.2] ``v`` is locally connected to ``T'`` iff ``|T'|`` internally
    vertex-disjoint ``v→T'`` paths exist (several may end at the same
    terminal; a pre-terminal of some ``t ∈ T'`` counts as having arbitrarily
    many paths to ``t``). Flow: split arcs of ``v`` and of the terminals of
    ``T'`` have capacity ``|T'|``, other vertices ``mult[x]``; only terminals
    of ``T'`` are joined to the sink."""
    Tp = sorted(set(Tprime))
    if not Tp:
        return True
    if any(t >= ci.k for t in Tp):
        raise ValueError("T' must consist of terminals")
    q = len(Tp)
    Tset = set(Tp)
    reach: list[int] = [v]
    seen = {v}
    stack = [v]
    while stack:
        x = stack.pop()
        if x < ci.k:
            continue
        for y in ci.out_adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
                reach.append(y)
    ids = {x: i for i, x in enumerate(reach)}
    net = FlowNetwork(2 * len(reach) + 2)
    s, z = 2 * len(reach), 2 * len(reach) + 1
    for x in reach:
        i = ids[x]
        net.add_arc(2 * i, 2 * i + 1, q if (x == v or x in Tset) else ci.mult[x])
        if x < ci.k:
            if x in Tset:
                net.add_arc(2 * i + 1, z, q)
            continue
        for y in ci.out_adj[x]:
            net.add_arc(2 * i + 1, 2 * ids[y], q)
    net.add_arc(s, 2 * ids[v], q)
    return net.max_flow(s, z, limit=q) >= q


def local_set(ci: CompactInstance, Tprime: Iterable[int]) -> list[int]:
    """``L(T')`` [Def A.2]: the non-terminals locally connected to ``T'``."""
    Tp = list(Tprime)
    return [v for v in ci.non_terminals() if is_locally_connected(ci, v, Tp)]


def satisfies_local_condition(ci: CompactInstance, max_k: int = 8) -> bool:
    """The local connectivity condition [Def A.2]: ``Σ cap = Σ mult`` and
    ``Σ_{v ∈ L(T')} mult[v] ≥ Σ_{t ∈ T'} cap[t]`` for every ``T' ⊆ T``
    (all ``2^k − 1`` nonempty subsets; refuses ``k > max_k``)."""
    if ci.k > max_k:
        raise ValueError(f"exhaustive subset check limited to k <= {max_k} (k = {ci.k})")
    if ci.total_cap() != ci.total_mult():
        return False
    terms = list(ci.terminals())
    for size in range(1, ci.k + 1):
        for Tp in combinations(terms, size):
            need = sum(ci.cap[t] for t in Tp)
            have = sum(ci.mult[v] for v in ci.non_terminals() if is_locally_connected(ci, v, Tp))
            if have < need:
                return False
    return True


def describe(ci: CompactInstance) -> dict[str, Any]:
    """Small summary used by tests and scripts."""
    pre = ci.pre_terminals()
    return {
        "k": ci.k,
        "n": ci.n,
        "arcs": ci.num_arcs(),
        "pre_terminals": len(pre),
        "total_mult": ci.total_mult(),
        "total_cap": ci.total_cap(),
    }
