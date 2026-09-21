"""Delta-debugging minimizer: shrink an instance while a failure predicate stays true.

Typical use::

    def fails(inst):  # e.g. solver output rejected by the verifier / assertion raised
        ...
    small = minimize_instance(inst, fails)

Reduction moves (each keeps the instance well-formed via :func:`make_instance`):
arc deletion (ddmin over arcs), removal of a non-terminal vertex (decrementing
the capacity of the part that a reference solution or the largest capacity
assigns), removal of a terminal (its capacity is added to another terminal).
"""
from __future__ import annotations

from typing import Callable

from glsolver.instance import Instance, make_instance


def minimize_instance(inst: Instance, fails: Callable[[Instance], bool], max_rounds: int = 50) -> Instance:
    """Greedy ddmin-style shrinking. ``fails`` must be True on ``inst``."""
    if not fails(inst):
        raise ValueError("predicate is not true on the starting instance")
    cur = inst
    for _ in range(max_rounds):
        changed = False
        # 1. try dropping arcs in chunks (ddmin)
        arcs = list(cur.arcs) if cur.directed else [tuple(e) for e in (cur.undirected_edges or [])]
        n_chunks = 2
        while n_chunks <= len(arcs) and arcs:
            size = max(1, len(arcs) // n_chunks)
            progressed = False
            for start in range(0, len(arcs), size):
                trial = arcs[:start] + arcs[start + size:]
                cand = _try_arcs(cur, trial)
                if cand is not None and fails(cand):
                    cur, arcs, progressed, changed = cand, trial, True, True
                    n_chunks = max(2, n_chunks - 1)
                    break
            if not progressed:
                if n_chunks >= len(arcs):
                    break
                n_chunks = min(len(arcs), n_chunks * 2)
        # 2. try dropping non-terminal vertices
        for v in range(cur.n):
            if v in cur.terminals:
                continue
            cand = _try_drop_vertex(cur, v)
            if cand is not None and fails(cand):
                cur, changed = cand, True
                break
        if not changed:
            break
    return cur


def _try_arcs(inst: Instance, arcs: list[tuple[int, int]]) -> Instance | None:
    try:
        return make_instance(inst.n, arcs, inst.terminals, inst.capacities, weights=inst.weights,
                             directed=inst.directed, name=inst.name + "_min", meta=dict(inst.meta))
    except ValueError:
        return None


def _try_drop_vertex(inst: Instance, v: int) -> Instance | None:
    caps = list(inst.capacities)
    w = 1 if inst.weights is None else inst.weights[v]
    # give the reduction to the largest capacity that can absorb it
    i = max(range(inst.k), key=lambda j: caps[j])
    if inst.weights is None:
        if caps[i] < 1:
            return None
        caps[i] -= 1
    else:
        caps[i] = max(0, caps[i] - w)
    edges = list(inst.arcs) if inst.directed else [tuple(e) for e in (inst.undirected_edges or [])]
    keep = [u for u in range(inst.n) if u != v]
    remap = {u: j for j, u in enumerate(keep)}
    edges2 = [(remap[a], remap[b]) for a, b in edges if a != v and b != v]
    terms = [remap[t] for t in inst.terminals]
    weights = None if inst.weights is None else [inst.weights[u] for u in keep]
    try:
        return make_instance(len(keep), edges2, terms, caps, weights=weights, directed=inst.directed,
                             name=inst.name + "_min", meta=dict(inst.meta))
    except ValueError:
        return None
