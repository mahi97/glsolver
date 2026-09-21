"""Deterministic small-instance builders for ``tests/test_reference_*.py``.

Everything here is test-only scaffolding: seeded ``random.Random`` generators
built on networkx plus :func:`glsolver.instance.make_instance`, the two worked
examples of the paper (docs/paper_notes.md), and a small *independent*
partition checker that is used next to ``glsolver.verify`` whenever that
module is importable (both must agree).  This module never imports
``glsolver.generators`` (it may not exist yet).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Sequence

import networkx as nx

from glsolver.instance import Instance, make_instance

try:  # written by another agent; optional at import time
    from glsolver.verify import verify_instance_parts as _verify_instance_parts
except Exception:  # pragma: no cover - depends on sibling work
    _verify_instance_parts = None

HAVE_VERIFIER = _verify_instance_parts is not None

CAPACITY_MODES: tuple[str, ...] = ("balanced", "unbalanced", "extreme", "random", "zeros")


# ---------------------------------------------------------------------------
# capacities
# ---------------------------------------------------------------------------


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    """Uniformly random weak composition of ``total`` into ``parts`` parts (stars and bars)."""
    if parts == 0:
        if total != 0:
            raise ValueError("cannot split a positive total into zero parts")
        return []
    bars = sorted(rng.sample(range(total + parts - 1), parts - 1))
    res: list[int] = []
    prev = -1
    for b in bars:
        res.append(b - prev - 1)
        prev = b
    res.append(total + parts - 1 - prev - 1)
    assert sum(res) == total and len(res) == parts
    return res


def random_capacities(total: int, k: int, rng: random.Random, mode: str = "random") -> tuple[int, ...]:
    """Nonnegative ``c_1..c_k`` with ``Σ c_i = total`` (the unweighted requirement of [Def 5.1]).

    ``mode``: ``balanced`` (as equal as possible), ``unbalanced`` (one terminal
    gets about two thirds), ``extreme`` (one terminal gets everything, the rest
    0), ``random`` (uniform weak composition), ``zeros`` (about half of the
    terminals get 0, the rest a random composition).  The result is shuffled so
    the "big" terminal is at a random index.
    """
    if k <= 0 or total < 0:
        raise ValueError("need k >= 1 and total >= 0")
    if mode == "balanced":
        base, extra = divmod(total, k)
        caps = [base + (1 if i < extra else 0) for i in range(k)]
    elif mode == "extreme":
        caps = [0] * k
        caps[0] = total
    elif mode == "unbalanced":
        big = (2 * total) // 3
        caps = [big] + _random_composition(total - big, k - 1, rng) if k > 1 else [total]
    elif mode == "random":
        caps = _random_composition(total, k, rng)
    elif mode == "zeros":
        nz = max(1, k // 2)
        caps = _random_composition(total, nz, rng) + [0] * (k - nz)
    else:
        raise ValueError(f"unknown capacity mode {mode!r}; choose from {CAPACITY_MODES}")
    rng.shuffle(caps)
    assert sum(caps) == total and len(caps) == k
    return tuple(caps)


# ---------------------------------------------------------------------------
# graphs and instances
# ---------------------------------------------------------------------------


def random_k_connected_undirected(
    n: int, k: int, rng: random.Random, max_tries: int = 5000
) -> nx.Graph:
    """Random simple graph on ``0..n-1`` with ``nx.node_connectivity(G) >= k``.

    Rejection sampling over ``G(n, p)`` with ``p`` growing after repeated
    failures; ``k >= n - 1`` forces the complete graph.  ``k``-connected implies
    ``k``-``T``-connected for every ``T`` of size ``k`` (paper_notes §1.1).
    """
    if n < 1 or k < 1:
        raise ValueError("need n >= 1 and k >= 1")
    if k >= n - 1:
        return nx.complete_graph(n)
    p = k / (n - 1) + 0.05
    for attempt in range(max_tries):
        q = min(1.0, p + rng.random() * 0.3)
        G = nx.gnp_random_graph(n, q, seed=rng.randrange(1 << 30))
        if nx.node_connectivity(G) >= k:
            return G
        if attempt % 20 == 19:
            p = min(1.0, p + 0.1)
    raise RuntimeError(f"could not sample a {k}-connected graph on {n} vertices")


def random_undirected_instance(
    n: int, k: int, rng: random.Random, mode: str = "random", *, name: str = ""
) -> Instance:
    """A classical GL instance: ``k``-connected graph, random terminals, capacities per ``mode``."""
    G = random_k_connected_undirected(n, k, rng)
    terminals = rng.sample(range(n), k)
    caps = random_capacities(n - k, k, rng, mode)
    return make_instance(n, list(G.edges()), terminals, caps, directed=False, name=name)


def random_digraph(
    n: int,
    k: int,
    rng: random.Random,
    p: float = 0.4,
    mode: str = "random",
    *,
    terminals: Sequence[int] | None = None,
    name: str = "",
) -> Instance:
    """Random simple digraph: each arc ``(u, v)`` with ``u`` a non-terminal is
    present with probability ``p``; terminals are a random sample (random
    order) unless given; capacities from :func:`random_capacities`."""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    terms = list(terminals) if terminals is not None else rng.sample(range(n), k)
    tset = set(terms)
    arcs = [
        (u, v)
        for u in range(n)
        if u not in tset
        for v in range(n)
        if v != u and rng.random() < p
    ]
    caps = random_capacities(n - k, k, rng, mode)
    return make_instance(n, arcs, terms, caps, name=name)


def random_kT_connected_digraph(
    n: int, k: int, rng: random.Random, mode: str = "random", max_tries: int = 2000, *, name: str = ""
) -> Instance:
    """Random ``k``-``T``-connected digraph [Def 3.2]: rejection sampling until
    ``glref.essential.terminal_connectivity(g, v) == k`` for every non-terminal."""
    from glref.essential import terminal_connectivity
    from glref.graph import DiGraphState

    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    if k == n:
        return make_instance(n, [], list(range(n)), [0] * n, name=name)
    p = min(1.0, (k + 0.5) / (n - 1))
    for attempt in range(max_tries):
        q = min(1.0, p + rng.random() * 0.3)
        inst = random_digraph(n, k, rng, p=q, mode=mode, name=name)
        out_deg = [0] * n
        for u, _v in inst.arcs:
            out_deg[u] += 1
        tset = set(inst.terminals)
        if any(out_deg[v] < k for v in range(n) if v not in tset):
            continue  # κ(v) <= d^+(v): cheap rejection before any flow
        g = DiGraphState.from_instance(inst)
        if all(terminal_connectivity(g, v) == k for v in g.nonterminals()):
            return inst
        if attempt % 20 == 19:
            p = min(1.0, p + 0.1)
    raise RuntimeError(f"could not sample a {k}-T-connected digraph on {n} vertices")


# ---------------------------------------------------------------------------
# the paper's worked examples (docs/paper_notes.md)
# ---------------------------------------------------------------------------

PAPER_RUNNING_ARCS: tuple[tuple[int, int], ...] = (
    (7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1),
    (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2),
)

PAPER_ESSENTIAL_ARCS: tuple[tuple[int, int], ...] = (
    (12, 4), (12, 10), (12, 11), (12, 9),
    (10, 5), (10, 6), (10, 7),
    (11, 6), (11, 7), (11, 8),
    (5, 6), (6, 7), (7, 8), (8, 5),
    (4, 0), (5, 0), (6, 1), (7, 2), (8, 3), (9, 3),
)

# Expected essential sets of the paper's example (vertex ids: v_i = i - 1, t_i = i - 1).
PAPER_ESSENTIAL_EXPECTED: dict[int, frozenset[int]] = {
    4: frozenset({0}),
    5: frozenset({0}),
    6: frozenset({1}),
    7: frozenset({2}),
    8: frozenset({3}),
    9: frozenset({3}),
    10: frozenset({0, 1}),
    11: frozenset({1, 2}),
    12: frozenset({0, 1, 2, 3}),
}

# The witness displayed in the paper for c = (2, 3, 2, 2).
PAPER_ESSENTIAL_WITNESS: dict[int, int] = {
    12: 2, 10: 1, 11: 1, 4: 0, 5: 0, 6: 1, 7: 2, 8: 3, 9: 3,
}


def paper_running_example() -> Instance:
    """Running example of the paper: terminals 0,1,2, six non-terminals, c = (2,2,2)."""
    return make_instance(9, PAPER_RUNNING_ARCS, [0, 1, 2], [2, 2, 2], name="paper-running")


def paper_essential_example() -> Instance:
    """The paper's essential-terminal example: terminals 0..3, c = (2,3,2,2)."""
    return make_instance(13, PAPER_ESSENTIAL_ARCS, [0, 1, 2, 3], [2, 3, 2, 2], name="paper-essential")


# ---------------------------------------------------------------------------
# independent partition checker
# ---------------------------------------------------------------------------


def check_partition(inst: Instance, parts: Sequence[Sequence[Any]]) -> list[str]:
    """Local, solver-independent check of an unweighted partition (paper_notes §1.2).

    Returns a list of error strings (empty iff valid).  Checks: ``k`` parts,
    exact cover of ``0..n-1``, ``t_i ∈ parts[i]``, no foreign terminal,
    ``|parts[i]| == c_i + 1`` and [Def connected-to] by BFS on *reversed*
    original arcs inside the part.  For undirected input the connectivity of
    ``G[V_i]`` in the original undirected edge set is checked as well.
    """
    errors: list[str] = []
    n, k = inst.n, inst.k
    if len(parts) != k:
        errors.append(f"expected {k} parts, got {len(parts)}")
    owner: dict[int, int] = {}
    for i, part in enumerate(parts):
        for v in part:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n:
                errors.append(f"part {i}: bad vertex id {v!r}")
                continue
            if v in owner:
                errors.append(f"vertex {v} in parts {owner[v]} and {i}")
            owner[v] = i
    missing = [v for v in range(n) if v not in owner]
    if missing:
        errors.append(f"vertices missing from the partition: {missing}")
    in_adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in inst.arcs:
        in_adj[v].append(u)
    tset = set(inst.terminals)
    for i in range(min(k, len(parts))):
        t = inst.terminals[i]
        pset = {v for v in parts[i] if isinstance(v, int) and 0 <= v < n}
        if len(pset) != inst.capacities[i] + 1:
            errors.append(f"part {i} has size {len(pset)}, expected {inst.capacities[i] + 1}")
        if t not in pset:
            errors.append(f"terminal {t} not in part {i}")
            continue
        foreign = sorted((pset & tset) - {t})
        if foreign:
            errors.append(f"part {i} contains foreign terminals {foreign}")
        reached = {t}
        dq: deque[int] = deque([t])
        while dq:
            x = dq.popleft()
            for u in in_adj[x]:
                if u in pset and u not in reached:
                    reached.add(u)
                    dq.append(u)
        if reached != pset:
            errors.append(f"part {i}: vertices {sorted(pset - reached)} cannot reach {t} inside the part")
        if not inst.directed and inst.undirected_edges is not None:
            G = nx.Graph()
            G.add_nodes_from(pset)
            G.add_edges_from((u, v) for u, v in inst.undirected_edges if u in pset and v in pset)
            if not nx.is_connected(G):
                errors.append(f"part {i} is not connected in the undirected graph")
    return errors


def partition_is_valid(inst: Instance, parts: Sequence[Sequence[Any]]) -> bool:
    """Local check, cross-checked against ``glsolver.verify`` when available."""
    local_ok = not check_partition(inst, parts)
    if _verify_instance_parts is not None:
        report = _verify_instance_parts(inst, [list(p) for p in parts])
        assert report.valid == local_ok, (
            f"verifier ({report.valid}: {report.errors}) and local checker "
            f"({local_ok}: {check_partition(inst, parts)}) disagree"
        )
    return local_ok


def assert_valid_partition(inst: Instance, parts: Sequence[Sequence[Any]]) -> None:
    errors = check_partition(inst, parts)
    assert not errors, errors
    if _verify_instance_parts is not None:
        report = _verify_instance_parts(inst, [list(p) for p in parts])
        assert report.valid, report.errors


__all__ = [
    "CAPACITY_MODES",
    "HAVE_VERIFIER",
    "PAPER_ESSENTIAL_ARCS",
    "PAPER_ESSENTIAL_EXPECTED",
    "PAPER_ESSENTIAL_WITNESS",
    "PAPER_RUNNING_ARCS",
    "assert_valid_partition",
    "check_partition",
    "paper_essential_example",
    "paper_running_example",
    "partition_is_valid",
    "random_capacities",
    "random_digraph",
    "random_k_connected_undirected",
    "random_kT_connected_digraph",
    "random_undirected_instance",
]
