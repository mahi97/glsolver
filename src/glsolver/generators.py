"""Seeded graph-family generators producing normalized :class:`Instance` objects.

Every generator is deterministic for a given ``seed`` (all randomness goes
through ``random.Random(seed)``) and returns an instance built with
:func:`glsolver.instance.make_instance` whose ``meta`` records at least
``{"family": ..., "seed": ..., **params}``.  NetworkX is used only for graph
construction and connectivity checks.

Connectivity bookkeeping in ``meta`` (used by the tests and the benchmarks):

* ``claims_k_connected`` – the generator asserts that the undirected graph is
  ``k``-vertex-connected (hence ``k``-``T``-connected for any ``k``
  terminals, docs/paper_notes.md §1.1);
* ``connectivity_verified`` – whether that claim was checked exactly
  (:func:`is_k_vertex_connected`: degree bounds, else Even's flow test for
  ``n <= 500`` and ``n·|E| <= 5e6``); larger graphs get only a
  degree/connectedness check;
* ``claims_kT_connected`` – for directed families: every non-terminal has
  ``κ_G(v) = k`` [Def 3.2]; for DAG families this is the out-degree test
  [Lem 9.1].
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Callable, Sequence

import networkx as nx

from glsolver.instance import Instance, make_instance

CapacityMode = str
_MODES = ("balanced", "unbalanced", "random", "random_positive", "extreme")
_MAX_TRIES = 200
_VERIFY_LIMIT = 500  # exact k-connectivity test is only run for n <= this ...
_VERIFY_BUDGET = 5_000_000  # ... and n * |E| <= this (flow cost is O(n * m))
_FLOW_LIMIT = 200  # directed_variant verifies k-T-connectivity by default up to this n
_DIGRAPH_LIMIT = 60  # random_kT_connected_digraph is limited to this n


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def random_terminals(n: int, k: int, rng: random.Random) -> list[int]:
    """Choose ``k`` distinct terminals among ``0..n-1`` (returned sorted)."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    return sorted(rng.sample(range(n), k))


def capacity_vector(total: int, k: int, rng: random.Random, mode: CapacityMode) -> list[int]:
    """Split ``total`` into ``k`` nonnegative integer capacities ``c_i`` (§1.2).

    Modes: ``"balanced"`` (as equal as possible, randomly permuted),
    ``"unbalanced"`` (``[1,...,1,total-k+1]`` permuted; if ``total < k-1``
    the surplus parts get ``0``), ``"random"`` (uniformly random weak
    composition, all ``>= 0``), ``"random_positive"`` (uniformly random
    composition with all ``>= 1``; needs ``total >= k``), ``"extreme"`` (one
    random part gets everything, the others ``0``).
    """
    if k < 1:
        raise ValueError("k must be positive")
    if total < 0:
        raise ValueError("total must be nonnegative")
    if mode == "balanced":
        q, r = divmod(total, k)
        caps = [q + 1] * r + [q] * (k - r)
        rng.shuffle(caps)
        return caps
    if mode == "unbalanced":
        ones = min(k - 1, total)
        caps = [1] * ones + [0] * (k - 1 - ones) + [total - ones]
        rng.shuffle(caps)
        return caps
    if mode == "random":
        # stars and bars: k-1 bars among total+k-1 slots -> uniform weak composition
        bars = sorted(rng.sample(range(total + k - 1), k - 1))
        prev = -1
        caps = []
        for b in bars:
            caps.append(b - prev - 1)
            prev = b
        caps.append(total + k - 1 - prev - 1)
        return caps
    if mode == "random_positive":
        if total < k:
            raise ValueError("random_positive needs total >= k")
        cuts = sorted(rng.sample(range(1, total), k - 1))
        bounds = [0] + cuts + [total]
        return [bounds[i + 1] - bounds[i] for i in range(k)]
    if mode == "extreme":
        caps = [0] * k
        caps[rng.randrange(k)] = total
        return caps
    raise ValueError(f"unknown capacity mode {mode!r}; choose one of {_MODES}")


def _local_connectivity(adj: Sequence[set[int]], s: int, t: int, k: int) -> int:
    """Number of internally vertex-disjoint ``s``-``t`` paths, capped at ``k``
    (BFS augmenting paths on the vertex-split network; a direct edge counts once)."""
    n = len(adj)
    res: list[dict[int, int]] = [{} for _ in range(2 * n)]

    def add(a: int, b: int) -> None:
        res[a][b] = 1
        res[b].setdefault(a, 0)

    for x in range(n):
        if x != s and x != t:
            add(2 * x, 2 * x + 1)
        for y in adj[x]:
            add(2 * x + 1, 2 * y)
    src, sink = 2 * s + 1, 2 * t
    value = 0
    while value < k:
        parent = {src: -1}
        dq = deque([src])
        found = False
        while dq and not found:
            a = dq.popleft()
            for b, c in res[a].items():
                if c > 0 and b not in parent:
                    parent[b] = a
                    if b == sink:
                        found = True
                        break
                    dq.append(b)
        if not found:
            break
        b = sink
        while b != src:
            a = parent[b]
            res[a][b] -= 1
            res[b][a] += 1
            b = a
        value += 1
    return value


def is_k_vertex_connected(G: nx.Graph, k: int) -> bool:
    """Exact test whether the undirected graph ``G`` is ``k``-vertex-connected.

    Even's algorithm: with any ``k`` vertices ``v_1..v_k``, ``G`` (on more than
    ``k`` vertices) is ``k``-connected iff every pair ``v_i, v_j`` has ``k``
    internally disjoint paths and every other vertex ``u`` has ``k`` paths to
    distinct ``v_i`` that are vertex-disjoint except for ``u`` (the latter is
    the ``κ`` computation of [Def 3.2] with ``T = {v_1..v_k}``).  Costs
    ``O(k^2 + n)`` max-flows of at most ``k`` augmentations each.  Two exact
    shortcuts precede the flows: minimum degree ``< k`` (not ``k``-connected)
    and the classical bound ``2·δ(G) >= n + k - 2`` (``k``-connected: a
    separator of size ``< k`` would leave a component whose vertices have
    degree at most ``(n + k - 3)/2``).
    """
    n = G.number_of_nodes()
    if k <= 0:
        return True
    if n <= k:
        return False
    if k == 1:
        return nx.is_connected(G)
    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in G.edges():
        if u != v:
            adj[index[u]].add(index[v])
            adj[index[v]].add(index[u])
    min_deg = min(len(a) for a in adj)
    if min_deg < k:
        return False
    if 2 * min_deg >= n + k - 2:
        return True
    # pick the k vertices of smallest degree (any choice is valid)
    base = sorted(range(n), key=lambda x: (len(adj[x]), x))[:k]
    for i in range(k):
        for j in range(i + 1, k):
            if _local_connectivity(adj, base[i], base[j], k) < k:
                return False
    bset = set(base)
    for u in range(n):
        if u not in bset and _kappa(adj, base, k, u)[0] < k:
            return False
    return True


def _check_k_connected(G: nx.Graph, k: int) -> tuple[bool, bool]:
    """Return ``(ok, verified)``: ``ok`` iff ``G`` is ``k``-connected as far as we
    checked; ``verified`` iff the exact test (:func:`is_k_vertex_connected`) ran."""
    n = G.number_of_nodes()
    if n <= 1:
        return k <= max(0, n - 1), True
    if not nx.is_connected(G):
        return False, True
    min_deg = min(d for _, d in G.degree())
    if min_deg < k:
        return False, True
    if 2 * min_deg >= n + k - 2:  # exact sufficient condition, see is_k_vertex_connected
        return True, True
    if n <= _VERIFY_LIMIT and n * G.number_of_edges() <= _VERIFY_BUDGET:
        return is_k_vertex_connected(G, k), True
    return True, False


def _instance_from_undirected(
    G: nx.Graph,
    k: int,
    rng: random.Random,
    mode: CapacityMode,
    *,
    family: str,
    seed: int,
    params: dict[str, Any],
    claims: bool,
    verified: bool,
) -> Instance:
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    n = G.number_of_nodes()
    terminals = random_terminals(n, k, rng)
    caps = capacity_vector(n - k, k, rng, mode)
    meta: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "mode": mode,
        **params,
        "claims_k_connected": bool(claims),
        "connectivity_verified": bool(verified),
    }
    return make_instance(
        n, sorted(G.edges()), terminals, caps, directed=False,
        name=f"{family}_n{n}_k{k}_s{seed}", meta=meta,
    )


def _sample_k_connected(
    build: Callable[[int], nx.Graph], k: int, rng: random.Random, *, require: bool
) -> tuple[nx.Graph, bool, bool, int]:
    """Rejection-sample ``build(seed_int)`` until the graph is ``k``-connected.

    Returns ``(G, ok, verified, tries)``.  Raises ``RuntimeError`` after
    ``_MAX_TRIES`` failures when ``require`` is true; otherwise returns the
    last sample with ``ok=False``.
    """
    tries = 0
    G = None
    ok = verified = False
    while tries < _MAX_TRIES:
        tries += 1
        G = build(rng.randrange(1 << 31))
        ok, verified = _check_k_connected(G, k)
        if ok or not require:
            break
    assert G is not None
    if require and not ok:
        raise RuntimeError(f"no {k}-connected sample found in {_MAX_TRIES} tries")
    return G, ok, verified, tries


# ---------------------------------------------------------------------------
# Structured undirected families
# ---------------------------------------------------------------------------
def complete_graph(n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced") -> Instance:
    """Complete graph ``K_n`` (connectivity ``n-1``); requires ``k <= n - 1``
    (so ``K_1`` is refused for every ``k >= 1``, consistent with
    :func:`is_k_vertex_connected`, which needs ``n > k``)."""
    if k > n - 1:
        raise ValueError(f"K_{n} is only {n - 1}-connected")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.complete_graph(n), k, rng, mode, family="complete", seed=seed,
        params={"n": n, "k": k}, claims=True, verified=True,
    )


def cycle_graph(n: int, k: int = 2, seed: int = 0, mode: CapacityMode = "balanced") -> Instance:
    """Cycle ``C_n`` (connectivity exactly 2 for ``n >= 3``); requires ``k <= 2``."""
    if n < 3:
        raise ValueError("cycle needs n >= 3")
    if k > 2:
        raise ValueError("a cycle is only 2-connected")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.cycle_graph(n), k, rng, mode, family="cycle", seed=seed,
        params={"n": n, "k": k}, claims=True, verified=True,
    )


def wheel_graph(n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced") -> Instance:
    """Wheel ``W_n`` on ``n`` vertices (hub + ``n-1`` rim; connectivity 3 for
    ``n >= 4``); requires ``k <= 3``."""
    if n < 4:
        raise ValueError("wheel needs n >= 4")
    if k > 3:
        raise ValueError("a wheel is only 3-connected")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.wheel_graph(n), k, rng, mode, family="wheel", seed=seed,
        params={"n": n, "k": k}, claims=True, verified=True,
    )


def grid_graph(
    rows: int, cols: int, k: int, seed: int = 0, mode: CapacityMode = "balanced"
) -> Instance:
    """``rows x cols`` grid (connectivity 2 when both dimensions are ``>= 2``,
    1 for a path, 0 for the single vertex); requires ``k`` at most that."""
    conn = 0 if rows * cols <= 1 else (2 if rows >= 2 and cols >= 2 else 1)
    if k > conn:
        raise ValueError(f"a {rows}x{cols} grid is only {conn}-connected")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.grid_2d_graph(rows, cols), k, rng, mode, family="grid", seed=seed,
        params={"rows": rows, "cols": cols, "k": k}, claims=True, verified=True,
    )


def grid3d_graph(
    a: int, b: int, c: int, k: int, seed: int = 0, mode: CapacityMode = "balanced"
) -> Instance:
    """``a x b x c`` grid (connectivity = number of dimensions ``>= 2``, at most
    3; 1 for a path, 0 for the single vertex); requires ``k`` at most that."""
    conn = 0 if a * b * c <= 1 else max(1, sum(1 for d in (a, b, c) if d >= 2))
    if k > conn:
        raise ValueError(f"a {a}x{b}x{c} grid is only {conn}-connected")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.grid_graph(dim=[a, b, c]), k, rng, mode, family="grid3d", seed=seed,
        params={"a": a, "b": b, "c": c, "k": k}, claims=True, verified=True,
    )


def harary_graph(n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced") -> Instance:
    """Harary graph ``H_{k,n}``: exactly ``k``-connected with ``ceil(kn/2)``
    edges for ``k >= 2``, the minimum possible for a ``k``-connected graph on
    ``n`` vertices.  For ``k = 1`` the networkx construction
    ``hkn_harary_graph(1, n)`` is a path with ``n - 1`` edges."""
    if not 1 <= k < n:
        raise ValueError("Harary graph needs 1 <= k < n")
    rng = random.Random(seed)
    return _instance_from_undirected(
        nx.hkn_harary_graph(k, n), k, rng, mode, family="harary", seed=seed,
        params={"n": n, "k": k}, claims=True, verified=True,
    )


def adversarial_ladder(
    n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced", *, closed: bool = False
) -> Instance:
    """Ladder of ``n/k`` cliques ``K_k`` joined by random perfect matchings.

    Consecutive cliques are joined by a random perfect matching (a
    ``K_k``-times-path / ``C_{n/k} x K_k``-like ladder), giving a long diameter
    (about ``n/k``).  With ``closed=False`` (default) the ladder is a path of
    cliques whose end cliques have degree ``k``, so the connectivity is
    *exactly* ``k``; ``closed=True`` also joins the last clique to the first,
    which raises the connectivity to ``k+1`` (with ``>= 3`` cliques).
    Requires ``n`` divisible by ``k`` with at least two cliques.
    """
    if k < 1 or n % k != 0 or n // k < 2:
        raise ValueError("adversarial_ladder needs n divisible by k and n >= 2k")
    r = n // k
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    cliques = [list(range(j * k, (j + 1) * k)) for j in range(r)]
    for cl in cliques:
        G.add_edges_from((cl[x], cl[y]) for x in range(k) for y in range(x + 1, k))
    pairs = list(range(r - 1)) + ([r - 1] if closed and r >= 3 else [])
    for j in pairs:
        perm = cliques[(j + 1) % r][:]
        rng.shuffle(perm)
        G.add_edges_from(zip(cliques[j], perm))
    return _instance_from_undirected(
        G, k, rng, mode, family="adversarial_ladder", seed=seed,
        params={"n": n, "k": k, "closed": bool(closed), "cliques": r,
                "connectivity": k + 1 if (closed and r >= 3) else k},
        claims=True, verified=True,
    )


def sparse_k_connected(
    n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced", *, chords: int | None = None
) -> Instance:
    """Harary ``H_{k,n}`` plus a few random chords (default ``max(1, n // 10)``);
    still ``k``-connected (adding edges never lowers connectivity)."""
    if not 1 <= k < n:
        raise ValueError("sparse_k_connected needs 1 <= k < n")
    rng = random.Random(seed)
    G = nx.hkn_harary_graph(k, n)
    want = max(1, n // 10) if chords is None else int(chords)
    max_chords = n * (n - 1) // 2 - G.number_of_edges()
    added: list[tuple[int, int]] = []
    while len(added) < min(want, max_chords):
        u, v = rng.sample(range(n), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added.append((min(u, v), max(u, v)))
    return _instance_from_undirected(
        G, k, rng, mode, family="sparse_k_connected", seed=seed,
        params={"n": n, "k": k, "chords": len(added)}, claims=True, verified=True,
    )


# ---------------------------------------------------------------------------
# Random undirected families (rejection sampling for k-connectivity)
# ---------------------------------------------------------------------------
def random_regular_graph(
    n: int, d: int, k: int, seed: int = 0, mode: CapacityMode = "balanced",
    *, require_k_connected: bool = True,
) -> Instance:
    """Random ``d``-regular graph (``n*d`` even), rejection-sampled until
    ``k``-connected (exact check up to the module's size budget, degree check otherwise)."""
    if d < k:
        raise ValueError("a d-regular graph cannot be k-connected for d < k")
    if (n * d) % 2 or d >= n:
        raise ValueError("random regular graph needs n*d even and d < n")
    rng = random.Random(seed)
    G, ok, verified, tries = _sample_k_connected(
        lambda s: nx.random_regular_graph(d, n, seed=s), k, rng, require=require_k_connected
    )
    return _instance_from_undirected(
        G, k, rng, mode, family="random_regular", seed=seed,
        params={"n": n, "d": d, "k": k, "tries": tries}, claims=ok, verified=verified,
    )


def erdos_renyi_graph(
    n: int, p: float, k: int, seed: int = 0, mode: CapacityMode = "balanced",
    require_k_connected: bool = True,
) -> Instance:
    """``G(n, p)`` rejection-sampled (up to 200 tries) until ``k``-connected.

    The exact test :func:`is_k_vertex_connected` is used for ``n <= 500`` and
    ``n·|E| <= 5e6``, otherwise only a minimum-degree + connectedness check; ``meta["connectivity_verified"]``
    records which.  With ``require_k_connected=False`` the first sample is
    returned and ``meta["claims_k_connected"]`` reports the check's outcome.
    """
    rng = random.Random(seed)
    G, ok, verified, tries = _sample_k_connected(
        lambda s: nx.gnp_random_graph(n, p, seed=s), k, rng, require=require_k_connected
    )
    return _instance_from_undirected(
        G, k, rng, mode, family="erdos_renyi", seed=seed,
        params={"n": n, "p": p, "k": k, "tries": tries}, claims=ok, verified=verified,
    )


def random_geometric_graph(
    n: int, radius: float, k: int, seed: int = 0, mode: CapacityMode = "balanced",
    require_k_connected: bool = True,
) -> Instance:
    """Random geometric graph in the unit square, rejection-sampled until
    ``k``-connected (same verification policy as :func:`erdos_renyi_graph`)."""
    rng = random.Random(seed)
    G, ok, verified, tries = _sample_k_connected(
        lambda s: nx.random_geometric_graph(n, radius, seed=s), k, rng,
        require=require_k_connected,
    )
    return _instance_from_undirected(
        G, k, rng, mode, family="random_geometric", seed=seed,
        params={"n": n, "radius": radius, "k": k, "tries": tries}, claims=ok, verified=verified,
    )


def expander_graph(
    n: int, d: int, k: int, seed: int = 0, mode: CapacityMode = "balanced"
) -> Instance:
    """Random ``d``-regular graph used as an expander.

    A uniformly random ``d``-regular graph is, with high probability, a
    near-Ramanujan expander (Friedman's theorem: second eigenvalue
    ``2 sqrt(d-1) + o(1)``) and ``d``-connected.  The sample is nevertheless
    rejection-checked for ``k``-connectivity like :func:`random_regular_graph`.
    """
    inst = random_regular_graph(n, d, k, seed, mode)
    meta = dict(inst.meta)
    meta["family"] = "expander"
    meta["expander_whp"] = True
    return make_instance(
        inst.n, inst.undirected_edges or (), inst.terminals, inst.capacities,
        directed=False, name=f"expander_n{n}_k{k}_s{seed}", meta=meta,
    )


def dense_graph(
    n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced", density: float = 0.8
) -> Instance:
    """Dense ``G(n, density)`` rejection-sampled until ``k``-connected."""
    inst = erdos_renyi_graph(n, density, k, seed, mode, require_k_connected=True)
    meta = dict(inst.meta)
    meta["family"] = "dense"
    meta["density"] = density
    return make_instance(
        inst.n, inst.undirected_edges or (), inst.terminals, inst.capacities,
        directed=False, name=f"dense_n{n}_k{k}_s{seed}", meta=meta,
    )


# ---------------------------------------------------------------------------
# Paper examples
# ---------------------------------------------------------------------------
def paper_running_example() -> Instance:
    """The paper's directed running example (docs/paper_notes.md §1.2 setting).

    Terminals ``t1..t3 = 0..2``, non-terminals ``v4..v9 = 3..8``, 12 arcs,
    capacities ``(2,2,2)``.  It is *not* 3-``T``-connected (``v4`` has
    out-degree 2); it illustrates essential terminals and the assignment
    condition [Def 4.1], [Def 5.1].
    """
    arcs = [(7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1),
            (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2)]
    return make_instance(
        9, arcs, [0, 1, 2], [2, 2, 2], directed=True, name="paper_running_example",
        meta={"family": "paper_running_example", "seed": 0, "claims_kT_connected": False},
    )


def paper_contract_counterexample() -> Instance:
    """Undirected 9-vertex example where naive contraction fails (paper §4 figure).

    Terminals ``0,1,2``; 12 edges; sizes ``(3,3,3)`` i.e. ``c = (2,2,2)``.
    The graph is 3-``T``-connected [Def 3.2] but only 2-vertex-connected
    (each terminal has degree 2).
    """
    edges = [(0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 8),
             (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (3, 8)]
    return make_instance(
        9, edges, [0, 1, 2], sizes=[3, 3, 3], directed=False,
        name="paper_contract_counterexample",
        meta={"family": "paper_contract_counterexample", "seed": 0,
              "claims_k_connected": False, "claims_kT_connected": True,
              "connectivity_verified": True},
    )


def paper_essential_example() -> Instance:
    """Directed 13-vertex example from the paper figure
    "k-conn-essential-and-assignment" ([Def 4.1], [Def 5.1]).

    Terminals ``t1..t4 = 0..3``, non-terminals ``v5..v13 = 4..12``, 20 arcs,
    capacities ``(2,3,2,2)``.
    """
    def v(j: int) -> int:
        return j - 1

    def t(i: int) -> int:
        return i - 1

    arcs = [
        (v(13), v(5)), (v(13), v(11)), (v(13), v(12)), (v(13), v(10)),
        (v(11), v(6)), (v(11), v(7)), (v(11), v(8)),
        (v(12), v(7)), (v(12), v(8)), (v(12), v(9)),
        (v(6), v(7)), (v(7), v(8)), (v(8), v(9)), (v(9), v(6)),
        (v(5), t(1)), (v(6), t(1)), (v(7), t(2)), (v(8), t(3)), (v(9), t(4)), (v(10), t(4)),
    ]
    return make_instance(
        13, arcs, [0, 1, 2, 3], [2, 3, 2, 2], directed=True, name="paper_essential_example",
        meta={"family": "paper_essential_example", "seed": 0, "claims_kT_connected": False},
    )


# ---------------------------------------------------------------------------
# k-T-connectivity test (flow based)
# ---------------------------------------------------------------------------
def _kappa(
    out_adj: Sequence[set[int]], terminals: Sequence[int], k: int, v: int
) -> tuple[int, set[tuple[int, int]]]:
    """``κ_G(v)`` [Def 3.2] by BFS augmenting paths on the vertex-split network
    of §3.1 / [Prop 4.2] (unit split arcs, ``K = k+1`` elsewhere; at most ``k``
    augmentations).  Returns ``(value, arcs carrying flow)``; the arc set is a
    family of ``value`` paths that certifies the bound and lets an incremental
    checker skip vertices whose family survives an arc deletion."""
    n = len(out_adj)
    K = k + 1
    sink = 2 * n
    # node 2x = x_in, 2x+1 = x_out; paths start at v_out (v's split arc is free)
    # and never re-enter v (arcs into v are omitted: cycles through v are useless).
    res: list[dict[int, int]] = [{} for _ in range(2 * n + 1)]

    def add(a: int, b: int, c: int) -> None:
        res[a][b] = c
        res[b].setdefault(a, 0)

    for x in range(n):
        if x != v:
            add(2 * x, 2 * x + 1, 1)
        for y in out_adj[x]:
            if y != v:
                add(2 * x + 1, 2 * y, K)
    for t in terminals:
        add(2 * t + 1, sink, K)
    src = 2 * v + 1
    value = 0
    while value < k:
        parent = {src: -1}
        dq = deque([src])
        found = False
        while dq and not found:
            a = dq.popleft()
            for b, c in res[a].items():
                if c > 0 and b not in parent:
                    parent[b] = a
                    if b == sink:
                        found = True
                        break
                    dq.append(b)
        if not found:
            break
        b = sink
        while b != src:
            a = parent[b]
            res[a][b] -= 1
            res[b][a] += 1
            b = a
        value += 1
    used = {
        (x, y) for x in range(n) for y in out_adj[x]
        if y != v and res[2 * y].get(2 * x + 1, 0) > 0
    }
    return value, used


class _KTChecker:
    """Incremental ``k``-``T``-connectivity checker [Def 3.2].

    Keeps, for every non-terminal ``x``, the arc set of a maximum family of
    ``x -> T`` paths.  Deleting arcs can only lower ``κ(x)`` for vertices whose
    stored family uses one of them, so :meth:`try_remove` recomputes just those.
    """

    def __init__(self, n: int, arcs: Sequence[tuple[int, int]], terminals: Sequence[int]):
        self.n = n
        self.k = len(terminals)
        self.terminals = list(terminals)
        tset = set(terminals)
        self.nonterminals = [x for x in range(n) if x not in tset]
        self.out: list[set[int]] = [set() for _ in range(n)]
        for u, v in arcs:
            self.out[u].add(v)
        self.used: dict[int, set[tuple[int, int]]] = {}

    def kappa(self, v: int) -> int:
        return _kappa(self.out, self.terminals, self.k, v)[0]

    def full_check(self) -> bool:
        """``True`` iff every non-terminal has ``κ = k`` (stops at the first failure)."""
        self.used = {}
        for x in self.nonterminals:
            val, used = _kappa(self.out, self.terminals, self.k, x)
            if val < self.k:
                return False
            self.used[x] = used
        return True

    def try_remove(self, batch: Sequence[tuple[int, int]]) -> bool:
        """Delete ``batch`` if ``k``-``T``-connectivity survives; else leave the
        graph unchanged.  Requires a successful :meth:`full_check` before."""
        present = [(u, v) for u, v in batch if v in self.out[u]]
        bset = set(present)
        for u, v in present:
            self.out[u].discard(v)
        fresh: dict[int, set[tuple[int, int]]] = {}
        for x in self.nonterminals:
            if self.used[x] & bset:
                val, used = _kappa(self.out, self.terminals, self.k, x)
                if val < self.k:
                    for u, v in present:
                        self.out[u].add(v)
                    return False
                fresh[x] = used
        self.used.update(fresh)
        return True

    def arcs(self) -> list[tuple[int, int]]:
        return sorted((u, v) for u in range(self.n) for v in self.out[u])


def terminal_connectivity(inst: Instance, v: int) -> int:
    """``κ_G(v)`` [Def 3.2]: maximum number of paths from ``v`` to *distinct*
    terminals that are vertex-disjoint except for ``v`` (one max-flow on the
    vertex-split network of §3.1 / [Prop 4.2])."""
    if v in inst.terminals:
        raise ValueError("κ is defined for non-terminals only")
    return _KTChecker(inst.n, inst.arcs, inst.terminals).kappa(v)


def is_k_t_connected(inst: Instance) -> bool:
    """``True`` iff every non-terminal has ``κ_G(v) = k`` [Def 3.2] (§1.2 precondition).

    Self-contained flow test (no solver code involved), so the generators'
    connectivity claims are independent of ``glsolver.preconditions``.
    """
    return _KTChecker(inst.n, inst.arcs, inst.terminals).full_check()


# ---------------------------------------------------------------------------
# Directed families
# ---------------------------------------------------------------------------
def random_kT_connected_dag(
    n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced",
    extra_out: int = 0, weighted: bool = False, w_max: int = 1,
) -> Instance:
    """Random ``k``-``T``-connected DAG [Lem 9.1].

    The non-terminals are placed in a random order with the terminals last;
    each non-terminal gets ``k + extra_out`` out-arcs (capped by the number
    of candidates) chosen uniformly among the later vertices and the
    terminals, which is always at least ``k`` candidates.  Every out-degree is
    therefore ``>= k``, which is exactly ``k``-``T``-connectivity for DAGs.
    With ``weighted=True`` non-terminals get weights in ``[1, w_max]`` and the
    capacities split the total weight (§1.3).
    """
    if not 1 <= k < n:
        raise ValueError("need 1 <= k < n")
    rng = random.Random(seed)
    terminals = random_terminals(n, k, rng)
    nonterm = [v for v in range(n) if v not in set(terminals)]
    rng.shuffle(nonterm)
    arcs: list[tuple[int, int]] = []
    nt = len(nonterm)
    for pos, u in enumerate(nonterm):
        # candidates: nonterm[pos+1:] followed by the terminals (>= k of them)
        ncand = (nt - pos - 1) + k
        d = min(ncand, k + max(0, extra_out))
        for j in rng.sample(range(ncand), d):
            x = nonterm[pos + 1 + j] if j < nt - pos - 1 else terminals[j - (nt - pos - 1)]
            arcs.append((u, x))
    weights = None
    if weighted:
        weights = [0] * n
        for u in nonterm:
            weights[u] = rng.randint(1, max(1, w_max))
        caps = capacity_vector(sum(weights), k, rng, mode)
    else:
        caps = capacity_vector(n - k, k, rng, mode)
    return make_instance(
        n, arcs, terminals, caps, weights=weights, directed=True,
        name=f"random_kT_dag_n{n}_k{k}_s{seed}",
        meta={"family": "random_kT_dag", "seed": seed, "mode": mode, "n": n, "k": k,
              "extra_out": extra_out, "weighted": weighted, "w_max": w_max,
              "claims_kT_connected": True, "connectivity_verified": True, "dag": True},
    )


def layered_dag(
    layers: int, width: int, k: int, seed: int = 0, mode: CapacityMode = "balanced",
    weighted: bool = False, w_max: int = 3,
) -> Instance:
    """Layered ``k``-``T``-connected DAG [Lem 9.1].

    ``layers * width`` non-terminals ``0..layers*width-1`` (layer of ``v`` is
    ``v // width``) followed by ``k`` terminals.  Arcs go only to the next one
    or two layers and to the terminals; each vertex draws an out-degree in
    ``[k, k+2]`` (capped by its candidate pool) uniformly from that pool.
    """
    if layers < 1 or width < 1 or k < 1:
        raise ValueError("layers, width and k must be positive")
    rng = random.Random(seed)
    nt = layers * width
    n = nt + k
    terminals = list(range(nt, n))
    arcs: list[tuple[int, int]] = []
    for u in range(nt):
        layer = u // width
        cands = list(range((layer + 1) * width, min(nt, (layer + 3) * width))) + terminals
        d = min(len(cands), k + rng.randint(0, 2))
        arcs.extend((u, x) for x in rng.sample(cands, d))
    weights = None
    if weighted:
        weights = [0] * n
        for u in range(nt):
            weights[u] = rng.randint(1, max(1, w_max))
        caps = capacity_vector(sum(weights), k, rng, mode)
    else:
        caps = capacity_vector(nt, k, rng, mode)
    return make_instance(
        n, arcs, terminals, caps, weights=weights, directed=True,
        name=f"layered_dag_L{layers}_W{width}_k{k}_s{seed}",
        meta={"family": "layered_dag", "seed": seed, "mode": mode, "layers": layers,
              "width": width, "k": k, "weighted": weighted, "w_max": w_max,
              "claims_kT_connected": True, "connectivity_verified": True, "dag": True},
    )


def random_kT_connected_digraph(
    n: int, k: int, seed: int = 0, mode: CapacityMode = "balanced", p: float | None = None
) -> Instance:
    """Random digraph (arc probability ``p``) rejection-sampled until the flow
    test :func:`is_k_t_connected` [Def 3.2] passes; ``n <= 60``.

    Default ``p = min(1, max(0.3, 2k/(n-1)))``.  Raises ``RuntimeError`` after
    200 failed samples.
    """
    if not 1 <= k < n:
        raise ValueError("need 1 <= k < n")
    if n > _DIGRAPH_LIMIT:
        raise ValueError(f"random_kT_connected_digraph is limited to n <= {_DIGRAPH_LIMIT}")
    if p is None:
        p = min(1.0, max(0.3, 2.0 * k / max(1, n - 1)))
    rng = random.Random(seed)
    terminals = random_terminals(n, k, rng)
    tset = set(terminals)
    for tries in range(1, _MAX_TRIES + 1):
        arcs = [
            (u, v) for u in range(n) if u not in tset
            for v in range(n) if v != u and rng.random() < p
        ]
        caps = capacity_vector(n - k, k, rng, mode)
        inst = make_instance(
            n, arcs, terminals, caps, directed=True,
            name=f"random_kT_digraph_n{n}_k{k}_s{seed}",
            meta={"family": "random_kT_digraph", "seed": seed, "mode": mode, "n": n,
                  "k": k, "p": p, "tries": tries, "claims_kT_connected": True,
                  "connectivity_verified": True},
        )
        if is_k_t_connected(inst):
            return inst
    raise RuntimeError(f"no {k}-T-connected digraph found in {_MAX_TRIES} tries")


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _edges_of(inst: Instance) -> Sequence[tuple[int, int]]:
    return inst.undirected_edges if not inst.directed else inst.arcs


def weighted_variant(inst: Instance, seed: int, w_max: int, slack: int = 0) -> Instance:
    """Weighted copy of ``inst`` (§1.3): random weights in ``[1, w_max]`` on the
    non-terminals and capacities forming a random composition of
    ``total_weight + slack``; terminals and arcs are kept."""
    if w_max < 1 or slack < 0:
        raise ValueError("w_max must be >= 1 and slack >= 0")
    rng = random.Random(seed)
    tset = set(inst.terminals)
    weights = [0 if v in tset else rng.randint(1, w_max) for v in range(inst.n)]
    caps = capacity_vector(sum(weights) + slack, inst.k, rng, "random")
    meta = {
        **inst.meta,
        "family": "weighted_variant",
        "base_family": inst.meta.get("family"),
        "seed": seed,
        "w_max": w_max,
        "slack": slack,
    }
    return make_instance(
        inst.n, _edges_of(inst), inst.terminals, caps, weights=weights,
        directed=inst.directed, name=f"{inst.name}_weighted_s{seed}", meta=meta,
    )


def directed_variant(
    inst: Instance, seed: int, drop_fraction: float, *, verify: bool | None = None
) -> Instance:
    """Directed copy of an undirected instance that stays ``k``-``T``-connected.

    Every undirected edge is *proposed* for orientation with probability
    ``drop_fraction`` (one of its two arcs, chosen at random, is dropped; an
    edge at a terminal has a single arc, so orienting it towards the terminal
    is a no-op and away from it deletes the edge).  When ``verify`` is on
    (default: ``n <= 200``) the proposals are accepted greedily with the flow
    test [Def 3.2]: the whole batch is tried first, and a rejected batch is
    bisected, so every accepted deletion provably keeps ``k``-``T``-connectivity
    (the input must be ``k``-``T``-connected, else ``ValueError``).  Without
    verification all proposals are applied and ``meta["claims_kT_connected"]``
    is ``False``.
    """
    if inst.directed:
        raise ValueError("directed_variant expects an undirected instance")
    if not 0.0 <= drop_fraction <= 1.0:
        raise ValueError("drop_fraction must be in [0, 1]")
    if verify is None:
        verify = inst.n <= _FLOW_LIMIT
    rng = random.Random(seed)
    edges = list(inst.undirected_edges or ())
    rng.shuffle(edges)
    proposals: list[tuple[int, int]] = []
    for u, v in edges:
        if rng.random() < drop_fraction:
            # drop one arc: (u,v) means "orient as v -> u"
            proposals.append((u, v) if rng.random() < 0.5 else (v, u))
    checker = _KTChecker(inst.n, inst.arcs, inst.terminals)
    if verify:
        if not checker.full_check():
            raise ValueError("directed_variant needs a k-T-connected base instance")

        def accept(batch: list[tuple[int, int]]) -> None:
            if not batch or checker.try_remove(batch):
                return
            if len(batch) > 1:
                h = len(batch) // 2
                accept(batch[:h])
                accept(batch[h:])

        accept(proposals)
        arcs = checker.arcs()
    else:
        drop = set(proposals)
        arcs = [a for a in inst.arcs if a not in drop]
    meta = {
        **inst.meta,
        "family": "directed_variant",
        "base_family": inst.meta.get("family"),
        "seed": seed,
        "drop_fraction": drop_fraction,
        "proposed": len(proposals),
        "dropped": len(inst.arcs) - len(arcs),
        "claims_kT_connected": bool(verify),
        "connectivity_verified": bool(verify),
    }
    return make_instance(
        inst.n, arcs, inst.terminals, inst.capacities, weights=inst.weights,
        directed=True, name=f"{inst.name}_directed_s{seed}", meta=meta,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def _grid_dims(n: int) -> tuple[int, int]:
    rows = max(2, int(math.isqrt(n)))
    cols = max(2, -(-n // rows))
    return rows, cols


def _regular_degree(n: int, k: int) -> int:
    d = max(3, k + 1)
    if (n * d) % 2:
        d += 1
    return min(d, n - 1)


def _er_p(n: int, k: int) -> float:
    return min(1.0, max(0.1, (2.0 * math.log(max(n, 2)) + k) / max(n, 2)))


def _geo_radius(n: int, k: int) -> float:
    return min(1.0, 1.3 * math.sqrt((k + 2.0 * math.log(max(n, 2))) / (math.pi * max(n, 2))))


def family_catalog() -> dict[str, Callable[[int, int, int], Instance]]:
    """``name -> callable(n, k, seed)`` producing balanced-capacity instances.

    Covers every family of this module (used by the benchmarks).  Sizes are
    derived from ``n`` where the family has its own shape parameters (grids,
    layered DAGs, ladders); the three ``paper_*`` entries ignore their
    arguments.  Entries raise ``ValueError`` for combinations the family
    cannot realise (e.g. ``cycle`` with ``k > 2``).
    """
    def grid(n: int, k: int, seed: int) -> Instance:
        r, c = _grid_dims(n)
        return grid_graph(r, c, k, seed)

    def grid3d(n: int, k: int, seed: int) -> Instance:
        a = max(2, round(n ** (1 / 3)))
        b = max(2, round(math.sqrt(max(1, n / a))))
        c = max(2, -(-n // (a * b)))
        return grid3d_graph(a, b, c, k, seed)

    def ladder(n: int, k: int, seed: int) -> Instance:
        return adversarial_ladder(max(2 * k, (n // k) * k), k, seed)

    def layered(n: int, k: int, seed: int) -> Instance:
        width = max(k, 2)
        layers = max(1, (n - k) // width)
        return layered_dag(layers, width, k, seed)

    def weighted_dag(n: int, k: int, seed: int) -> Instance:
        return random_kT_connected_dag(n, k, seed, weighted=True, w_max=4)

    def directed_random(n: int, k: int, seed: int) -> Instance:
        # a Harary base is degree-tight (nothing can be dropped), so use G(n,p)
        base = erdos_renyi_graph(n, _er_p(n, 2 * k), k, seed)
        return directed_variant(base, seed, drop_fraction=0.5)

    return {
        "complete": lambda n, k, s: complete_graph(n, k, s),
        "cycle": lambda n, k, s: cycle_graph(n, k, s),
        "wheel": lambda n, k, s: wheel_graph(n, k, s),
        "grid": grid,
        "grid3d": grid3d,
        "harary": lambda n, k, s: harary_graph(n, k, s),
        "random_regular": lambda n, k, s: random_regular_graph(n, _regular_degree(n, k), k, s),
        "erdos_renyi": lambda n, k, s: erdos_renyi_graph(n, _er_p(n, k), k, s),
        "random_geometric": lambda n, k, s: random_geometric_graph(n, _geo_radius(n, k), k, s),
        "expander": lambda n, k, s: expander_graph(n, _regular_degree(n, k), k, s),
        "dense": lambda n, k, s: dense_graph(n, k, s),
        "sparse_k_connected": lambda n, k, s: sparse_k_connected(n, k, s),
        "adversarial_ladder": ladder,
        "random_kT_dag": lambda n, k, s: random_kT_connected_dag(n, k, s),
        "layered_dag": layered,
        "random_kT_digraph": lambda n, k, s: random_kT_connected_digraph(n, k, s),
        "weighted_kT_dag": weighted_dag,
        "directed_random": directed_random,
        "paper_running_example": lambda n, k, s: paper_running_example(),
        "paper_contract_counterexample": lambda n, k, s: paper_contract_counterexample(),
        "paper_essential_example": lambda n, k, s: paper_essential_example(),
    }


__all__ = [
    "adversarial_ladder",
    "capacity_vector",
    "complete_graph",
    "cycle_graph",
    "dense_graph",
    "directed_variant",
    "erdos_renyi_graph",
    "expander_graph",
    "family_catalog",
    "grid3d_graph",
    "grid_graph",
    "harary_graph",
    "is_k_t_connected",
    "is_k_vertex_connected",
    "layered_dag",
    "paper_contract_counterexample",
    "paper_essential_example",
    "paper_running_example",
    "random_geometric_graph",
    "random_kT_connected_dag",
    "random_kT_connected_digraph",
    "random_regular_graph",
    "random_terminals",
    "sparse_k_connected",
    "terminal_connectivity",
    "weighted_variant",
    "wheel_graph",
]
