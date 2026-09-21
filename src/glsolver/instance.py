"""Shared instance container and input normalization.

This module is deliberately tiny and dependency-free so that the independent
verifier (``glsolver.verify``), the reference solvers (``glref``), the oracles
and the optimized solver all agree on *what an instance is* without sharing any
algorithmic code.

Conventions (see docs/paper_notes.md §1–2):

* Vertices are ``0..n-1``.
* ``arcs`` is the *directed* arc list actually solved on. For undirected input
  every edge ``{u,v}`` becomes ``(u,v)`` and ``(v,u)``. Arcs leaving a terminal,
  self-loops and duplicate arcs are dropped at normalization time (WLOG per the
  paper).
* ``capacities[i]`` is the paper's ``c_i``: the number (unweighted) or total
  weight (weighted) of *non-terminal* vertices that part ``i`` must receive.
  For classical undirected GL with part sizes ``n_i`` (which count the
  terminal), ``c_i = n_i - 1``.
* ``weights`` is ``None`` for unweighted instances, else a length-``n`` tuple
  with ``weights[t] == 0`` for terminals and ``>= 1`` for non-terminals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Instance:
    n: int
    arcs: tuple[tuple[int, int], ...]
    terminals: tuple[int, ...]
    capacities: tuple[int, ...]
    weights: tuple[int, ...] | None = None
    directed: bool = True
    undirected_edges: tuple[tuple[int, int], ...] | None = None
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # ---- derived quantities -------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.terminals)

    @property
    def m(self) -> int:
        return len(self.arcs)

    @property
    def is_weighted(self) -> bool:
        return self.weights is not None

    @property
    def num_nonterminals(self) -> int:
        return self.n - self.k

    @property
    def sizes(self) -> tuple[int, ...]:
        """Classical part sizes ``n_i = c_i + 1`` (only meaningful when unweighted)."""
        return tuple(c + 1 for c in self.capacities)

    @property
    def total_weight(self) -> int:
        if self.weights is None:
            return self.num_nonterminals
        tset = set(self.terminals)
        return sum(w for v, w in enumerate(self.weights) if v not in tset)

    @property
    def w_max(self) -> int:
        if self.weights is None:
            return 1
        tset = set(self.terminals)
        return max((w for v, w in enumerate(self.weights) if v not in tset), default=1)

    def is_terminal(self, v: int) -> bool:
        return v in self.terminals

    def terminal_index(self) -> dict[int, int]:
        return {t: i for i, t in enumerate(self.terminals)}

    def out_adjacency(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self.arcs:
            adj[u].append(v)
        return adj

    def in_adjacency(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self.arcs:
            adj[v].append(u)
        return adj

    def validate(self) -> None:
        """Raise ``ValueError`` if the instance is structurally malformed.

        This checks *well-formedness only* (indices in range, distinct
        terminals, capacity sum), never the theorem's connectivity
        preconditions.
        """
        n, k = self.n, self.k
        if n <= 0:
            raise ValueError("instance must have at least one vertex")
        if k == 0:
            raise ValueError("at least one terminal is required")
        if len(set(self.terminals)) != k:
            raise ValueError("terminals must be distinct")
        for t in self.terminals:
            if not 0 <= t < n:
                raise ValueError(f"terminal {t} out of range")
        if len(self.capacities) != k:
            raise ValueError("capacities must have one entry per terminal")
        if any(c < 0 for c in self.capacities):
            raise ValueError("capacities must be nonnegative")
        tset = set(self.terminals)
        for u, v in self.arcs:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"arc ({u},{v}) out of range")
            if u == v:
                raise ValueError(f"self-loop at {u}")
            if u in tset:
                raise ValueError(f"arc ({u},{v}) leaves a terminal; normalize() drops these")
        if len(set(self.arcs)) != len(self.arcs):
            raise ValueError("duplicate arcs; normalize() merges these")
        if self.weights is None:
            if sum(self.capacities) != n - k:
                raise ValueError(
                    f"unweighted instance needs sum(capacities) == n - k "
                    f"({sum(self.capacities)} != {n - k})"
                )
        else:
            if len(self.weights) != n:
                raise ValueError("weights must have one entry per vertex")
            for v, w in enumerate(self.weights):
                if v in tset:
                    if w != 0:
                        raise ValueError(f"terminal {v} must have weight 0")
                elif w < 1:
                    raise ValueError(f"non-terminal {v} must have positive integer weight")
            if self.total_weight > sum(self.capacities):
                raise ValueError("weighted instance needs sum(weights) <= sum(capacities)")


def normalize_arcs(
    n: int,
    edges: Iterable[Sequence[int]],
    terminals: Sequence[int],
    directed: bool,
) -> tuple[tuple[int, int], ...]:
    """Build the canonical arc tuple: symmetrize if undirected, drop self-loops,
    arcs out of terminals, and duplicates; sort for determinism."""
    tset = set(terminals)
    seen: set[tuple[int, int]] = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of range for n={n}")
        if u == v:
            continue
        cand = [(u, v), (v, u)] if not directed else [(u, v)]
        for a, b in cand:
            if a in tset:
                continue
            seen.add((a, b))
    return tuple(sorted(seen))


def make_instance(
    n: int,
    edges: Iterable[Sequence[int]],
    terminals: Sequence[int],
    capacities: Sequence[int] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    weights: Sequence[int] | None = None,
    directed: bool = True,
    name: str = "",
    meta: dict[str, Any] | None = None,
) -> Instance:
    """Construct a normalized :class:`Instance`.

    Exactly one of ``capacities`` (paper convention, non-terminal count/weight
    per part) or ``sizes`` (classical convention, ``|V_i|`` including the
    terminal; unweighted only) must be given.
    """
    if (capacities is None) == (sizes is None):
        raise ValueError("give exactly one of capacities= or sizes=")
    if sizes is not None:
        if weights is not None:
            raise ValueError("sizes= is only meaningful for unweighted instances")
        if any(s < 1 for s in sizes):
            raise ValueError("sizes must be positive (each part contains its terminal)")
        capacities = [int(s) - 1 for s in sizes]
    assert capacities is not None
    edges = [tuple(e) for e in edges]
    arcs = normalize_arcs(n, edges, terminals, directed)
    undirected_edges = None
    if not directed:
        ue: set[tuple[int, int]] = set()
        for u, v in edges:
            u, v = int(u), int(v)
            if u != v:
                ue.add((min(u, v), max(u, v)))
        undirected_edges = tuple(sorted(ue))
    w = None
    if weights is not None:
        w = list(int(x) for x in weights)
        if len(w) != n:
            raise ValueError("weights must have length n")
        for t in terminals:
            w[t] = 0
        w = tuple(w)
    inst = Instance(
        n=int(n),
        arcs=arcs,
        terminals=tuple(int(t) for t in terminals),
        capacities=tuple(int(c) for c in capacities),
        weights=w,
        directed=bool(directed),
        undirected_edges=undirected_edges,
        name=name,
        meta=dict(meta or {}),
    )
    inst.validate()
    return inst


def from_networkx(
    G: Any,
    terminals: Sequence[int],
    capacities: Sequence[int] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    weights: Sequence[int] | dict[Any, int] | None = None,
    directed: bool | None = None,
    name: str = "",
) -> tuple[Instance, list[Any]]:
    """Convert a NetworkX graph. Returns ``(instance, node_list)`` where
    ``node_list[i]`` is the original node label of internal vertex ``i``.
    Terminals may be given as original labels. Weights may be a dict keyed by
    node label or a sequence aligned with ``node_list``."""
    import networkx as nx  # local import: keep this module importable without networkx

    if directed is None:
        directed = bool(G.is_directed())
    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in G.edges()]
    term_idx = [index[t] if t in index else int(t) for t in terminals]
    w = None
    if weights is not None:
        if isinstance(weights, dict):
            w = [int(weights.get(u, 1)) for u in nodes]
        else:
            w = [int(x) for x in weights]
    inst = make_instance(
        len(nodes), edges, term_idx, capacities, sizes=sizes, weights=w,
        directed=directed, name=name or str(getattr(G, "name", "")),
    )
    return inst, nodes
