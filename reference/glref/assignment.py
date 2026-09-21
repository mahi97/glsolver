"""Witnesses of the Flow-Essential Assignment Condition [Def 5.1]."""
from __future__ import annotations

from .flow import FlowNetwork
from .graph import DiGraphState


def find_witness(
    g: DiGraphState, ess: dict[int, set[int]], capacities: dict[int, int]
) -> dict[int, int] | None:
    """Find ``φ`` with ``φ(v) ∈ Ess(v)`` and ``|φ^{-1}(t)| = c_t`` [Def 5.1].

    Bipartite max-flow (proof of [Thm ess-assign-cond]): source→v (1),
    v→t (1) iff ``t ∈ Ess(v)``, t→sink (``c_t``). Returns ``None`` if the flow
    value is less than the number of non-terminals (FEAC fails).
    """
    nonterms = g.nonterminals()
    terms = list(g.terminals)
    if sum(capacities[t] for t in terms) != len(nonterms):
        return None
    vid = {v: i for i, v in enumerate(nonterms)}
    tid = {t: len(nonterms) + j for j, t in enumerate(terms)}
    s = len(nonterms) + len(terms)
    z = s + 1
    net = FlowNetwork(z + 1)
    arc_of: dict[tuple[int, int], int] = {}
    for v in nonterms:
        net.add_arc(s, vid[v], 1)
        for t in sorted(ess[v]):
            if t in tid:
                arc_of[(v, t)] = net.add_arc(vid[v], tid[t], 1)
    for t in terms:
        net.add_arc(tid[t], z, capacities[t])
    value = net.max_flow(s, z)
    if value != len(nonterms):
        return None
    phi: dict[int, int] = {}
    for (v, t), a in arc_of.items():
        if net.flow[a] == 1:
            phi[v] = t
    assert len(phi) == len(nonterms)
    return phi


def is_witness(
    g: DiGraphState, phi: dict[int, int], ess: dict[int, set[int]], capacities: dict[int, int]
) -> tuple[bool, str]:
    """Check [Def 5.1] for a given ``φ`` against precomputed ``Ess``. Returns (ok, reason)."""
    nonterms = g.nonterminals()
    if set(phi) != set(nonterms):
        return False, "domain of phi differs from the set of live non-terminals"
    for v in nonterms:
        if phi[v] not in ess[v]:
            return False, f"phi({v}) = {phi[v]} is not essential for {v} (Ess = {sorted(ess[v])})"
    counts = {t: 0 for t in g.terminals}
    for v in nonterms:
        counts[phi[v]] += 1
    for t in g.terminals:
        if counts[t] != capacities[t]:
            return False, f"terminal {t} receives {counts[t]} vertices, capacity {capacities[t]}"
    return True, "ok"
