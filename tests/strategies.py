"""Instance sources for the correctness harness (docs/verification.md).

Two kinds of sources live here:

* **Hypothesis strategies** (``st_*``) that draw :class:`glsolver.instance.Instance`
  objects satisfying a theorem precondition -- ``k``-vertex-connected
  undirected graphs (classical Győri–Lovász, paper_notes §1.1),
  ``k``-``T``-connected digraphs [Def 3.2], digraphs that satisfy the
  Flow-Essential Assignment Condition [Def 5.1] *without* being
  ``k``-``T``-connected (Theorem 2 beyond Theorem 1), ``k``-``T``-connected
  DAGs [Lem 9.1] and weighted variants (§1.3);
* **exhaustive enumerations** (``iter_*``) of every small instance: all
  connected undirected graphs of the NetworkX graph atlas with every terminal
  subset and every capacity composition, and all digraphs on ``n <= 4``.

Everything is deterministic: Hypothesis draws are reproducible from the
example database / ``print_blob`` output, and the enumerations use
``random.Random(seed)`` for their optional sub-sampling.  No solver code is
imported here; the precondition tests come from :mod:`glsolver.generators`
(self-contained flow test) and :mod:`glsolver.preconditions` (NetworkX by
definition).
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import networkx as nx
from hypothesis import assume
from hypothesis import strategies as st
from networkx.generators.atlas import graph_atlas_g

from glsolver.generators import (
    is_k_t_connected,
    layered_dag,
    random_kT_connected_dag,
    weighted_variant,
)
from glsolver.instance import Instance, make_instance
from glsolver.preconditions import check_preconditions

#: capacity modes of :func:`st_capacities` (paper convention ``c_i``, §1.2).
CAPACITY_MODES: tuple[str, ...] = ("balanced", "unbalanced", "random", "extreme", "zeros-allowed")

#: modes accepted by :func:`glsolver.generators.capacity_vector` that never fail
_GENERATOR_MODES: tuple[str, ...] = ("balanced", "unbalanced", "random", "extreme")


# ---------------------------------------------------------------------------
# capacities
# ---------------------------------------------------------------------------


@st.composite
def st_capacities(draw: st.DrawFn, total: int, k: int, mode: str | None = None) -> list[int]:
    """Nonnegative ``c_1..c_k`` with ``Σ c_i = total`` (unweighted requirement, [Def 5.1]).

    ``mode`` (drawn from :data:`CAPACITY_MODES` when ``None``):

    * ``"balanced"`` -- all equal up to one (``total = q·k + r`` -> ``r`` parts
      ``q+1``, the rest ``q``);
    * ``"unbalanced"`` -- the extreme vector ``(1, 1, ..., 1, total-k+1)`` (with
      zeros instead of ones when ``total < k-1``);
    * ``"extreme"`` -- all zero but one part, which gets everything;
    * ``"random"`` -- a random weak composition;
    * ``"zeros-allowed"`` -- one part is ``0`` and the rest is a random weak
      composition (so several zeros are possible).

    Every vector is returned under a random permutation, so all permutations
    of the extreme vectors are reachable.
    """
    if k < 1 or total < 0:
        raise ValueError("need k >= 1 and total >= 0")
    if mode is None:
        mode = draw(st.sampled_from(CAPACITY_MODES))
    if mode == "balanced":
        q, r = divmod(total, k)
        caps = [q + 1] * r + [q] * (k - r)
    elif mode == "unbalanced":
        ones = min(k - 1, total)
        caps = [1] * ones + [0] * (k - 1 - ones) + [total - ones]
    elif mode == "extreme":
        caps = [0] * k
        caps[0] = total
    elif mode == "random":
        caps = _draw_weak_composition(draw, total, k)
    elif mode == "zeros-allowed":
        caps = [0] + (_draw_weak_composition(draw, total, k - 1) if k > 1 else [])
        if k == 1:
            caps = [total]
    else:
        raise ValueError(f"unknown capacity mode {mode!r}; choose from {CAPACITY_MODES}")
    perm = draw(st.permutations(range(k)))
    caps = [caps[i] for i in perm]
    assert sum(caps) == total and len(caps) == k
    return caps


def _draw_weak_composition(draw: st.DrawFn, total: int, parts: int) -> list[int]:
    """Weak composition of ``total`` into ``parts`` parts from ``parts-1`` drawn cut points."""
    if parts == 1:
        return [total]
    cuts = sorted(draw(st.lists(st.integers(0, total), min_size=parts - 1, max_size=parts - 1)))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


# ---------------------------------------------------------------------------
# undirected k-connected instances (classical GL, §1.1)
# ---------------------------------------------------------------------------

_UNDIRECTED_KINDS: tuple[str, ...] = ("gnp",) * 7 + ("regular", "harary", "complete")


@st.composite
def st_undirected_k_connected(
    draw: st.DrawFn, max_n: int = 10, max_k: int = 4, min_n: int = 2
) -> Instance:
    """Classical Győri–Lovász instances: a ``k``-vertex-connected undirected
    graph (``k`` drawn in ``[1, min(κ(G), max_k, n-1)]``), ``k`` terminals in
    a random order and capacities from :func:`st_capacities`.

    The graph is ``G(n, p)`` with a drawn density most of the time, and with
    small probability a random regular graph, a Harary graph ``H_{d,n}`` or
    the complete graph.  Graphs with vertex connectivity ``0`` are rejected.
    ``meta["connectivity"]`` records ``networkx.node_connectivity``.

    ``k == n`` (every part a singleton, all capacities ``0``) is **never**
    drawn -- ``k <= n - 1`` by construction, and the same holds for every
    other strategy here and for the undirected enumeration.  That degenerate
    shape is covered by ``tests/test_edge_cases.py`` and by
    :func:`iter_directed_exhaustive` for ``n = 1, 2`` (docs/verification.md §5).
    """
    n = draw(st.integers(min_n, max_n))
    kind = draw(st.sampled_from(_UNDIRECTED_KINDS))
    if kind == "gnp":
        density = draw(st.integers(20, 100))
        pairs = list(itertools.combinations(range(n), 2))
        edges = [e for e in pairs if draw(st.integers(0, 99)) >= 100 - density]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)
    elif kind == "regular":
        d = draw(st.integers(1, n - 1))
        assume((n * d) % 2 == 0)
        graph = nx.random_regular_graph(d, n, seed=draw(st.integers(0, 2**20)))
    elif kind == "harary":
        d = draw(st.integers(1, n - 1))
        graph = nx.hkn_harary_graph(d, n)
    else:
        graph = nx.complete_graph(n)
    conn = int(nx.node_connectivity(graph)) if n >= 2 else 0
    assume(conn >= 1)
    k = draw(st.integers(1, min(conn, max_k, n - 1)))
    terminals = list(draw(st.permutations(range(n)))[:k])
    mode = draw(st.sampled_from(CAPACITY_MODES))
    caps = draw(st_capacities(n - k, k, mode))
    return make_instance(
        n, sorted(graph.edges()), terminals, caps, directed=False,
        name=f"hyp_undirected_{kind}_n{n}_k{k}",
        meta={
            "family": f"hyp_undirected_{kind}", "seed": 0, "mode": mode, "kind": kind,
            "n": n, "k": k, "connectivity": conn, "claims_k_connected": True,
            "connectivity_verified": True,
        },
    )


# ---------------------------------------------------------------------------
# directed instances
# ---------------------------------------------------------------------------


def _draw_arcs(
    draw: st.DrawFn, n: int, terminals: Sequence[int], density: int
) -> list[tuple[int, int]]:
    """Every arc ``(u, v)`` with ``u`` a non-terminal is present with
    probability ``density / 100`` (absent arcs are the shrink direction)."""
    tset = set(terminals)
    cands = [(u, v) for u in range(n) if u not in tset for v in range(n) if v != u]
    return [a for a in cands if draw(st.integers(0, 99)) >= 100 - density]


def _min_density(n: int, k: int) -> int:
    """Lower bound on the arc density so that out-degrees ``>= k`` are likely."""
    return min(95, int(math.ceil(100.0 * (k + 1) / max(1, n - 1))))


@st.composite
def st_random_digraph(draw: st.DrawFn, max_n: int = 8, max_k: int = 3) -> Instance:
    """Arbitrary random digraph instances *without* any precondition
    (rejection-free): used for honesty properties -- a solver may only answer
    ``ok`` (then valid) or ``precondition_failed``."""
    n = draw(st.integers(2, max_n))
    k = draw(st.integers(1, min(max_k, n - 1)))
    terminals = list(draw(st.permutations(range(n)))[:k])
    density = draw(st.integers(10, 100))
    arcs = _draw_arcs(draw, n, terminals, density)
    mode = draw(st.sampled_from(CAPACITY_MODES))
    caps = draw(st_capacities(n - k, k, mode))
    return make_instance(
        n, arcs, terminals, caps, directed=True, name=f"hyp_digraph_n{n}_k{k}",
        meta={"family": "hyp_random_digraph", "seed": 0, "mode": mode, "n": n, "k": k,
              "density": density},
    )


@st.composite
def st_directed_kT(draw: st.DrawFn, max_n: int = 9, max_k: int = 3) -> Instance:
    """``k``-``T``-connected digraphs [Def 3.2] by rejection sampling: random
    arcs with a drawn density (biased so that out-degrees reach ``k``), a
    cheap out-degree rejection, then the flow test
    :func:`glsolver.generators.is_k_t_connected`."""
    n = draw(st.integers(2, max_n))
    k = draw(st.integers(1, min(max_k, n - 1)))
    terminals = list(draw(st.permutations(range(n)))[:k])
    density = draw(st.integers(_min_density(n, k), 100))
    arcs = _draw_arcs(draw, n, terminals, density)
    out_deg = [0] * n
    for u, _v in arcs:
        out_deg[u] += 1
    tset = set(terminals)
    assume(all(out_deg[v] >= k for v in range(n) if v not in tset))
    mode = draw(st.sampled_from(CAPACITY_MODES))
    caps = draw(st_capacities(n - k, k, mode))
    inst = make_instance(
        n, arcs, terminals, caps, directed=True, name=f"hyp_kT_digraph_n{n}_k{k}",
        meta={"family": "hyp_kT_digraph", "seed": 0, "mode": mode, "n": n, "k": k,
              "density": density, "claims_kT_connected": True, "connectivity_verified": True},
    )
    assume(is_k_t_connected(inst))
    return inst


@st.composite
def st_feac_only(draw: st.DrawFn, max_n: int = 9) -> Instance:
    """Digraphs that satisfy FEAC [Def 5.1] but are **not** ``k``-``T``-connected
    (Theorem 2 beyond Theorem 1; the paper's running example is one).

    Two schemes, both filtered with
    ``glsolver.preconditions.check_preconditions(inst, True)`` (keep iff
    ``feac`` and not ``k_T_connected``): a plain random digraph, or a random
    digraph with out-degrees ``>= k`` in which one non-terminal keeps only
    ``1..k-1`` of its out-arcs ("damaged", ~45 % acceptance).  ``k >= 2``
    because for ``k = 1`` FEAC is equivalent to ``1``-``T``-connectivity.
    """
    n = draw(st.integers(3, max_n))
    k = draw(st.integers(2, min(3, n - 1)))
    terminals = list(draw(st.permutations(range(n)))[:k])
    scheme = draw(st.sampled_from(("random", "damaged", "damaged")))
    tset = set(terminals)
    if scheme == "random":
        density = draw(st.integers(20, 90))
        arcs = _draw_arcs(draw, n, terminals, density)
    else:
        density = draw(st.integers(_min_density(n, k), 100))
        arcs = _draw_arcs(draw, n, terminals, density)
        nonterms = [v for v in range(n) if v not in tset]
        victim = draw(st.sampled_from(nonterms))
        out_v = [a for a in arcs if a[0] == victim]
        assume(len(out_v) >= 2)
        keep = draw(st.integers(1, min(len(out_v) - 1, k - 1)))
        kept = draw(st.permutations(out_v))[:keep]
        arcs = [a for a in arcs if a[0] != victim] + list(kept)
    mode = draw(st.sampled_from(CAPACITY_MODES))
    caps = draw(st_capacities(n - k, k, mode))
    inst = make_instance(
        n, arcs, terminals, caps, directed=True, name=f"hyp_feac_only_n{n}_k{k}",
        meta={"family": "hyp_feac_only", "seed": 0, "mode": mode, "n": n, "k": k,
              "scheme": scheme, "density": density, "claims_kT_connected": False},
    )
    pre = check_preconditions(inst, True)
    assume(pre["feac"] is True and pre["k_T_connected"] is False)
    inst.meta["violating_vertex"] = pre["violating_vertex"]
    inst.meta["feac"] = True
    return inst


@st.composite
def st_dag(draw: st.DrawFn, max_n: int = 14, max_k: int = 4, weighted: bool = False) -> Instance:
    """``k``-``T``-connected DAGs [Lem 9.1] from
    :func:`glsolver.generators.random_kT_connected_dag` (random order, drawn
    ``extra_out``) or :func:`glsolver.generators.layered_dag` (drawn layer
    shape), with drawn seed and capacity mode.  ``weighted=True`` lets the
    generators draw weights (``w_max`` in ``[1, 5]``)."""
    kind = draw(st.sampled_from(("random", "random", "layered")))
    k = draw(st.integers(1, max_k))
    seed = draw(st.integers(0, 2**31 - 1))
    mode = draw(st.sampled_from(_GENERATOR_MODES))
    w_max = draw(st.integers(1, 5)) if weighted else 1
    if kind == "random":
        n = draw(st.integers(k + 1, max(k + 1, max_n)))
        extra = draw(st.integers(0, 3))
        inst = random_kT_connected_dag(
            n, k, seed=seed, mode=mode, extra_out=extra, weighted=weighted, w_max=w_max
        )
    else:
        width = draw(st.integers(1, 4))
        max_layers = max(1, (max_n - k) // width)
        layers = draw(st.integers(1, max_layers))
        inst = layered_dag(layers, width, k, seed=seed, mode=mode, weighted=weighted, w_max=w_max)
    return inst


@st.composite
def st_weighted(
    draw: st.DrawFn, base: st.SearchStrategy[Instance], w_max: int = 5, slack: int = 3
) -> Instance:
    """Weighted variant (§1.3) of an instance drawn from ``base``:
    :func:`glsolver.generators.weighted_variant` with drawn seed, drawn
    ``w_max`` in ``[1, w_max]`` (``<= 5``) and drawn slack in ``[0, slack]``
    (``<= 3``; ``Σ c = Σ w + slack``)."""
    if w_max > 5 or slack > 3:
        raise ValueError("st_weighted is meant for w_max <= 5 and slack <= 3")
    inst = draw(base)
    seed = draw(st.integers(0, 2**31 - 1))
    wm = draw(st.integers(1, w_max))
    s = draw(st.integers(0, slack))
    return weighted_variant(inst, seed, wm, s)


# ---------------------------------------------------------------------------
# exhaustive enumerations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasGraph:
    """A connected graph of the NetworkX atlas with its vertex connectivity."""

    index: int
    n: int
    connectivity: int
    edges: tuple[tuple[int, int], ...]


def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """All weak compositions of ``total`` into ``parts`` nonnegative parts
    (lexicographic order, ``C(total+parts-1, parts-1)`` of them)."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def count_compositions(total: int, parts: int) -> int:
    return math.comb(total + parts - 1, parts - 1) if parts >= 1 else int(total == 0)


def atlas_connected_graphs(max_n: int, min_n: int = 2) -> list[AtlasGraph]:
    """Every connected graph with ``min_n <= n <= max_n`` vertices of
    ``networkx.graph_atlas_g()`` (all graphs up to 7 vertices, one per
    isomorphism class) whose vertex connectivity is ``>= 1``."""
    if max_n > 7:
        raise ValueError("the NetworkX graph atlas stops at n = 7")
    out: list[AtlasGraph] = []
    for idx, graph in enumerate(graph_atlas_g()):
        n = graph.number_of_nodes()
        if n < min_n or n > max_n:
            continue
        if not nx.is_connected(graph):
            continue
        conn = int(nx.node_connectivity(graph))
        if conn < 1:
            continue
        out.append(AtlasGraph(idx, n, conn, tuple(sorted(tuple(sorted(e)) for e in graph.edges()))))
    return out


def undirected_cases_for_graph(
    graph: AtlasGraph,
    *,
    max_k: int = 3,
    full: bool,
    subset_sample: int = 3,
    composition_sample: int = 6,
    seed: int = 0,
) -> Iterator[Instance]:
    """All (or a seeded sample of the) classical GL instances on one atlas graph.

    For each ``k`` in ``1..min(κ(G), max_k)``: every ``k``-subset of terminals
    and every composition of ``n-k`` into ``k`` parts when ``full`` is true;
    otherwise ``subset_sample`` terminal subsets and up to
    ``composition_sample`` compositions per subset, drawn with
    ``random.Random(seed + index)`` (deterministic per graph).
    ``meta`` records ``atlas_index``, ``connectivity``, ``k``, ``terminals``
    and ``capacities``.
    """
    rng = random.Random(seed * 1_000_003 + graph.index)
    n = graph.n
    for k in range(1, min(graph.connectivity, max_k) + 1):
        subsets = list(itertools.combinations(range(n), k))
        if not full and len(subsets) > subset_sample:
            subsets = rng.sample(subsets, subset_sample)
        for terminals in subsets:
            comps = list(compositions(n - k, k))
            if not full and len(comps) > composition_sample:
                comps = rng.sample(comps, composition_sample)
            for caps in comps:
                yield make_instance(
                    n, graph.edges, list(terminals), list(caps), directed=False,
                    name=f"atlas{graph.index}_k{k}_t{'-'.join(map(str, terminals))}_c{'-'.join(map(str, caps))}",
                    meta={
                        "family": "atlas_exhaustive", "seed": seed, "atlas_index": graph.index,
                        "n": n, "k": k, "connectivity": graph.connectivity,
                        "terminals": list(terminals), "capacities": list(caps),
                        "claims_k_connected": True, "connectivity_verified": True,
                    },
                )


def iter_undirected_exhaustive(
    max_n: int,
    *,
    full_up_to: int = 5,
    full: bool = False,
    max_k: int = 3,
    subset_sample: int = 3,
    composition_sample: int = 6,
    seed: int = 0,
    min_n: int = 2,
) -> Iterator[Instance]:
    """Exhaustive undirected enumeration: every atlas graph with
    ``min_n <= n <= max_n`` is expanded fully when ``n <= full_up_to`` or
    ``full`` is set, and sampled otherwise (see
    :func:`undirected_cases_for_graph`)."""
    for graph in atlas_connected_graphs(max_n, min_n=min_n):
        yield from undirected_cases_for_graph(
            graph, max_k=max_k, full=full or graph.n <= full_up_to,
            subset_sample=subset_sample, composition_sample=composition_sample, seed=seed,
        )


def count_undirected_exhaustive(max_n: int, *, max_k: int = 3, min_n: int = 2) -> dict[int, int]:
    """Number of instances per ``n`` that the *full* enumeration produces."""
    counts: dict[int, int] = {}
    for graph in atlas_connected_graphs(max_n, min_n=min_n):
        total = 0
        for k in range(1, min(graph.connectivity, max_k) + 1):
            total += math.comb(graph.n, k) * count_compositions(graph.n - k, k)
        counts[graph.n] = counts.get(graph.n, 0) + total
    return counts


def directed_candidate_arcs(n: int, terminals: Sequence[int]) -> list[tuple[int, int]]:
    """The ``(n-k)(n-1)`` arcs leaving non-terminals, in lexicographic order
    (the bit positions of ``meta["arc_mask"]``)."""
    tset = set(terminals)
    return [(u, v) for u in range(n) if u not in tset for v in range(n) if v != u]


def _block_masks(count: int, mask_sample: int | None, rng_key: str) -> list[int]:
    """All ``2^count`` arc masks, or a seeded sample of ``mask_sample`` of them."""
    total = 1 << count
    if mask_sample is None or mask_sample >= total:
        return list(range(total))
    return sorted(random.Random(rng_key).sample(range(total), mask_sample))


def _block_compositions(total: int, parts: int, min_capacity: int) -> list[tuple[int, ...]]:
    return [c for c in compositions(total, parts) if min(c, default=0) >= min_capacity]


def directed_cases_for_block(
    n: int,
    k: int,
    terminals: Sequence[int],
    *,
    mask_sample: int | None = None,
    seed: int = 0,
    min_capacity: int = 0,
) -> Iterator[Instance]:
    """Every digraph instance of one ``(n, k, terminals)`` block: all (or a
    seeded sample of ``mask_sample``) subsets of the arcs leaving non-terminals
    (arcs out of terminals are dropped anyway, paper §2) and every composition
    of ``n-k`` into ``k`` capacities with every part ``>= min_capacity``.

    This is the single source of the digraph enumeration: both
    :func:`iter_directed_exhaustive` and ``scripts/run_exhaustive.py`` call
    it, so the two cannot drift.  The sample is drawn with
    ``random.Random(f"{seed}|{n}|{k}|{terminals}")`` (deterministic per
    block, independent of the iteration order).  ``meta["arc_mask"]`` is the
    bitmask over :func:`directed_candidate_arcs`.
    """
    if k < 1 or k > n:
        return
    terminals = tuple(terminals)
    cands = directed_candidate_arcs(n, terminals)
    masks = _block_masks(len(cands), mask_sample, f"{seed}|{n}|{k}|{terminals}")
    comps = _block_compositions(n - k, k, min_capacity)
    for mask in masks:
        arcs = [a for i, a in enumerate(cands) if mask >> i & 1]
        for caps in comps:
            yield make_instance(
                n, arcs, list(terminals), list(caps), directed=True,
                name=f"digraph_n{n}_k{k}_t{'-'.join(map(str, terminals))}_m{mask}_c{'-'.join(map(str, caps))}",
                meta={"family": "digraph_exhaustive", "seed": seed, "n": n, "k": k,
                      "terminals": list(terminals), "arc_mask": mask,
                      "capacities": list(caps)},
            )


def directed_blocks(max_n: int = 4, ks: Sequence[int] = (1, 2), *, min_n: int = 1) -> list[tuple[int, int, tuple[int, ...]]]:
    """The ``(n, k, terminals)`` blocks of the digraph enumeration, in order."""
    return [
        (n, k, terminals)
        for n in range(min_n, max_n + 1)
        for k in ks
        if k <= n
        for terminals in itertools.combinations(range(n), k)
    ]


def iter_directed_exhaustive(
    max_n: int = 4,
    ks: Sequence[int] = (1, 2),
    *,
    min_n: int = 1,
    mask_sample: int | None = None,
    seed: int = 0,
    min_capacity: int = 0,
) -> Iterator[Instance]:
    """Every digraph instance on ``min_n <= n <= max_n`` vertices: all
    terminal subsets of size ``k in ks``, then :func:`directed_cases_for_block`
    (all ``2^((n-k)(n-1))`` arc subsets, or ``mask_sample`` of them per block,
    and every composition of ``n-k`` into ``k`` capacities ``>= min_capacity``).

    ``k == n`` occurs only for ``n in ks`` (``n = 1, 2`` with the default
    ``ks``): all capacities are ``0`` and every part is a singleton."""
    for n, k, terminals in directed_blocks(max_n, ks, min_n=min_n):
        yield from directed_cases_for_block(
            n, k, terminals, mask_sample=mask_sample, seed=seed, min_capacity=min_capacity
        )


def count_directed_exhaustive(
    max_n: int = 4,
    ks: Sequence[int] = (1, 2),
    *,
    min_n: int = 1,
    mask_sample: int | None = None,
    min_capacity: int = 0,
) -> int:
    """Number of instances :func:`iter_directed_exhaustive` yields for the same arguments."""
    total = 0
    for n, k, _terminals in directed_blocks(max_n, ks, min_n=min_n):
        masks = 1 << ((n - k) * (n - 1))
        if mask_sample is not None:
            masks = min(masks, mask_sample)
        total += masks * len(_block_compositions(n - k, k, min_capacity))
    return total


def instance_summary(inst: Instance) -> dict[str, Any]:
    """Compact JSON-able description used in tallies and failure reports."""
    return {
        "name": inst.name, "n": inst.n, "m": inst.m, "k": inst.k, "directed": inst.directed,
        "weighted": inst.is_weighted, "terminals": list(inst.terminals),
        "capacities": list(inst.capacities), "meta": dict(inst.meta),
    }


__all__ = [
    "CAPACITY_MODES",
    "AtlasGraph",
    "atlas_connected_graphs",
    "compositions",
    "count_compositions",
    "count_directed_exhaustive",
    "count_undirected_exhaustive",
    "directed_blocks",
    "directed_candidate_arcs",
    "directed_cases_for_block",
    "instance_summary",
    "iter_directed_exhaustive",
    "iter_undirected_exhaustive",
    "st_capacities",
    "st_dag",
    "st_directed_kT",
    "st_feac_only",
    "st_random_digraph",
    "st_undirected_k_connected",
    "st_weighted",
    "undirected_cases_for_graph",
]
