"""Shared instance container and input normalization.

This module is deliberately tiny and dependency-free (numpy only) so that the
independent verifier (``glsolver.verify``), the reference solvers (``glref``),
the oracles and the optimized solver all agree on *what an instance is* without
sharing any algorithmic code.

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

Large instances (RESEARCH_NOTES.md E4). ``arcs`` and ``undirected_edges`` are
*canonically* tuples of ``(u, v)`` pairs, but an :class:`Instance` may be
constructed with an ``int32`` numpy array of shape ``(m, 2)`` instead (this is
what :func:`make_instance` does). The array is the internal storage; the tuple
form is materialized lazily on first access of ``.arcs`` / ``.undirected_edges``
and cached, so code that only needs the array (``inst.arc_array``, the C++
bindings) never pays for ``m`` Python tuples. Equality, hashing, ``repr`` and
``dataclasses.replace`` see the tuple form, exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

_PAIR_DTYPE = np.int32


def _array_to_pairs(arr: np.ndarray) -> tuple[tuple[int, int], ...]:
    """``(m, 2)`` array -> tuple of ``(u, v)`` tuples of Python ints."""
    return tuple(map(tuple, arr.tolist()))


def _as_pair_array(a: Any) -> np.ndarray:
    """Canonical array storage: ``int32``, shape ``(m, 2)``, C-contiguous, read-only.

    An int32 C-contiguous read-only input is used as is (zero copy); a writeable
    one is copied so that freezing it never touches the caller's array.
    """
    arr = np.asarray(a)
    if arr.ndim == 1 and arr.size == 0:
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"arc array must have shape (m, 2), got {arr.shape}")
    if arr.dtype != _PAIR_DTYPE:
        if arr.dtype.kind not in "iu":
            raise TypeError(f"arc array must have an integer dtype, got {arr.dtype}")
        if arr.size and (int(arr.min()) < -(2**31) or int(arr.max()) >= 2**31):
            raise ValueError("arc array holds a vertex id that does not fit in int32")
        arr = arr.astype(_PAIR_DTYPE)  # copies
    elif arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.array(arr, dtype=_PAIR_DTYPE, order="C")  # private copy
    arr.flags.writeable = False
    return arr


def _pairs_from_python(edges: Iterable[Sequence[int]], n: int) -> np.ndarray:
    """Historical element-wise conversion (``int(e[0]), int(e[1])`` with the range check in input order)."""
    pairs = [(int(e[0]), int(e[1])) for e in edges]
    for u, v in pairs:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of range for n={n}")
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def _edges_to_int_array(n: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Any edge input -> ``int64`` array of shape ``(m, 2)`` with every id in ``0..n-1``.

    Fast path for numpy arrays and lists of pairs (``np.asarray``); inputs numpy
    cannot vectorize (ragged, strings, huge ints, ...) take the element-wise
    path with the exact historical semantics. A third field per edge is
    ignored, as ``int(e[0]), int(e[1])`` always did.
    """
    if isinstance(edges, np.ndarray):
        arr: np.ndarray | None = edges
    else:
        if not isinstance(edges, (list, tuple)):
            edges = list(edges)
        try:
            arr = np.asarray(edges)
        except (ValueError, TypeError):
            arr = None
        if arr is not None and (arr.dtype.kind not in "iufb" or (arr.ndim == 1 and arr.size) or arr.ndim > 2):
            arr = None
    if arr is None:
        return _pairs_from_python(edges, n)
    if arr.ndim == 1 and arr.size == 0:
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return _pairs_from_python(edges, n)
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    if arr.dtype.kind not in "iu":
        arr = arr.astype(np.int64)  # floats truncate like int(), bools become 0/1
    arr = arr.astype(np.int64, copy=False)
    if arr.size:
        bad = (arr < 0) | (arr >= n)
        if bad.any():
            i = int(np.flatnonzero(bad.any(axis=1))[0])
            raise ValueError(f"edge ({int(arr[i, 0])},{int(arr[i, 1])}) out of range for n={n}")
    return arr


def _decode_keys(keys: np.ndarray, n: int) -> np.ndarray:
    """Sorted unique ``u * n + v`` keys -> read-only int32 ``(m, 2)`` array."""
    out = np.empty((keys.size, 2), dtype=_PAIR_DTYPE)
    if keys.size:
        u = keys // n
        out[:, 0] = u
        out[:, 1] = keys - u * n
    out.flags.writeable = False
    return out


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

    # ---- array views (module docstring) -------------------------------------
    @property
    def arc_array(self) -> np.ndarray:
        """The arcs as a read-only ``int32`` array of shape ``(m, 2)`` (cached)."""
        arr = self.__dict__.get("_arcs_array")
        if arr is None:
            arr = _as_pair_array(np.asarray(self.__dict__["_arcs_tuple"], dtype=_PAIR_DTYPE).reshape(-1, 2))
            object.__setattr__(self, "_arcs_array", arr)
        return arr

    @property
    def undirected_edge_array(self) -> np.ndarray | None:
        """``undirected_edges`` as a read-only ``int32`` array of shape ``(m_e, 2)``, or ``None``."""
        arr = self.__dict__.get("_undirected_edges_array")
        if arr is None:
            ue = self.__dict__["_undirected_edges_tuple"]
            if ue is None:
                return None
            arr = _as_pair_array(np.asarray(ue, dtype=_PAIR_DTYPE).reshape(-1, 2))
            object.__setattr__(self, "_undirected_edges_array", arr)
        return arr

    def _arc_pairs(self) -> Sequence[Sequence[int]]:
        """Iterable of ``(u, v)`` pairs of Python ints without forcing the tuple cache."""
        t = self.__dict__["_arcs_tuple"]
        return t if t is not None else self.__dict__["_arcs_array"].tolist()

    # ---- derived quantities -------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.terminals)

    @property
    def m(self) -> int:
        t = self.__dict__["_arcs_tuple"]
        return len(t) if t is not None else len(self.__dict__["_arcs_array"])

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
        for u, v in self._arc_pairs():
            adj[u].append(v)
        return adj

    def in_adjacency(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self._arc_pairs():
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
        if self.__dict__["_arcs_tuple"] is None:
            self._validate_arc_array(tset)
        else:
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

    def _validate_arc_array(self, tset: set[int]) -> None:
        """The arc checks of :meth:`validate`, vectorized (same errors, first offending arc in stored order)."""
        n = self.n
        a = self.__dict__["_arcs_array"]
        if not a.size:
            return
        u, v = a[:, 0], a[:, 1]
        bad_range = (u < 0) | (u >= n) | (v < 0) | (v >= n)
        bad_loop = u == v
        is_term = np.zeros(n, dtype=bool)
        is_term[[t for t in tset if 0 <= t < n]] = True
        bad_term = is_term[np.clip(u, 0, n - 1)] & ~bad_range
        first = np.flatnonzero(bad_range | bad_loop | bad_term)
        if first.size:
            i = int(first[0])
            uu, vv = int(u[i]), int(v[i])
            if bad_range[i]:
                raise ValueError(f"arc ({uu},{vv}) out of range")
            if bad_loop[i]:
                raise ValueError(f"self-loop at {uu}")
            raise ValueError(f"arc ({uu},{vv}) leaves a terminal; normalize() drops these")
        keys = u.astype(np.int64) * n + v
        if not bool(np.all(np.diff(keys) > 0)) and np.unique(keys).size != keys.size:
            raise ValueError("duplicate arcs; normalize() merges these")


def _lazy_pairs_property(name: str) -> property:
    """Field ``name`` of :class:`Instance` backed by ``_<name>_tuple`` / ``_<name>_array``.

    Installed after the dataclass is built so that the dataclass machinery sees
    a plain field (``__init__``, ``__eq__``, ``__hash__``, ``__repr__``,
    ``dataclasses.replace`` all read the tuple form through the getter). The
    frozen ``__init__`` assigns through ``object.__setattr__``, which reaches the
    setter: a numpy array is stored as the canonical read-only int32 array and
    the tuple is materialized on first read; anything else is stored verbatim.
    """
    tup_slot, arr_slot = f"_{name}_tuple", f"_{name}_array"

    def fget(self: Instance) -> Any:
        d = self.__dict__
        t = d[tup_slot]
        if t is None and d[arr_slot] is not None:
            t = _array_to_pairs(d[arr_slot])
            object.__setattr__(self, tup_slot, t)
        return t

    def fset(self: Instance, value: Any) -> None:
        if isinstance(value, np.ndarray):
            object.__setattr__(self, arr_slot, _as_pair_array(value))
            object.__setattr__(self, tup_slot, None)
        else:
            object.__setattr__(self, tup_slot, value)
            object.__setattr__(self, arr_slot, None)

    return property(fget, fset, doc=f"canonical tuple of (u, v) pairs; see ``{name.rstrip('s')}_array``")


Instance.arcs = _lazy_pairs_property("arcs")  # type: ignore[assignment]
Instance.undirected_edges = _lazy_pairs_property("undirected_edges")  # type: ignore[assignment]


def normalize_arcs_array(
    n: int,
    edges: Iterable[Sequence[int]],
    terminals: Sequence[int],
    directed: bool,
) -> np.ndarray:
    """Canonical arcs as a read-only ``int32`` array of shape ``(m, 2)``: symmetrized if undirected,
    self-loops, arcs out of terminals and duplicates dropped, sorted lexicographically (vectorized)."""
    arr = _edges_to_int_array(n, edges)
    if n <= 0 or not arr.size:
        return _decode_keys(np.empty(0, dtype=np.int64), max(n, 1))
    u, v = arr[:, 0], arr[:, 1]
    if not directed:
        u, v = np.concatenate([u, v]), np.concatenate([v, u])
    keep = u != v
    t = np.asarray([int(x) for x in terminals], dtype=np.int64)
    t = t[(t >= 0) & (t < n)]
    if t.size:
        is_term = np.zeros(n, dtype=bool)
        is_term[t] = True
        keep &= ~is_term[u]
    keys = np.unique(u[keep] * n + v[keep])
    return _decode_keys(keys, n)


def normalize_arcs(
    n: int,
    edges: Iterable[Sequence[int]],
    terminals: Sequence[int],
    directed: bool,
) -> tuple[tuple[int, int], ...]:
    """Build the canonical arc tuple: symmetrize if undirected, drop self-loops,
    arcs out of terminals, and duplicates; sort for determinism."""
    return _array_to_pairs(normalize_arcs_array(n, edges, terminals, directed))


def _undirected_edge_array(n: int, arr: np.ndarray) -> np.ndarray:
    """Distinct undirected edges ``(min, max)`` of an in-range ``(m, 2)`` array, self-loops dropped, sorted."""
    if n <= 0 or not arr.size:
        return _decode_keys(np.empty(0, dtype=np.int64), max(n, 1))
    lo, hi = np.minimum(arr[:, 0], arr[:, 1]), np.maximum(arr[:, 0], arr[:, 1])
    keep = lo != hi
    return _decode_keys(np.unique(lo[keep] * n + hi[keep]), n)


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
    terminal; unweighted only) must be given. ``edges`` may be any iterable of
    pairs or an integer numpy array of shape ``(m, 2)``; normalization is
    vectorized either way and the instance stores its arcs as an int32 array.
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
    n = int(n)
    edge_arr = _edges_to_int_array(n, edges)
    arcs = normalize_arcs_array(n, edge_arr, terminals, directed)
    undirected_edges = None if directed else _undirected_edge_array(n, edge_arr)
    w = None
    if weights is not None:
        w = list(int(x) for x in weights)
        if len(w) != n:
            raise ValueError("weights must have length n")
        for t in terminals:
            w[t] = 0
        w = tuple(w)
    inst = Instance(
        n=n,
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
