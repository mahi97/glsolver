"""Bipartite matchings between terminals and pre-terminals [Lem 7.6, Lem 7.8]
and the inclusion-minimal Hall-deficient set used by RoundAndRemove [Alg 4]."""
from __future__ import annotations

from typing import Iterable, Sequence

from .graph import DiGraphState


def _neighbors(g: DiGraphState, S: Sequence[int]) -> dict[int, list[int]]:
    """For each terminal in ``S``, the pre-terminals with an arc to it."""
    return {t: sorted(g.inn[t]) for t in S}


def max_matching(left: Sequence[int], nbrs: dict[int, Iterable[int]]) -> dict[int, int]:
    """Kuhn's augmenting-path bipartite matching. Returns ``{left_vertex: right_vertex}``."""
    match_right: dict[int, int] = {}
    match_left: dict[int, int] = {}

    def try_augment(u: int, seen: set[int]) -> bool:
        for w in nbrs[u]:
            if w in seen:
                continue
            seen.add(w)
            if w not in match_right or try_augment(match_right[w], seen):
                match_right[w] = u
                match_left[u] = w
                return True
        return False

    for u in left:
        try_augment(u, set())
    return match_left


def saturating_matching(g: DiGraphState, S: Sequence[int] | None = None) -> dict[int, int] | None:
    """A matching from the terminals ``S`` (default all of ``T``) to distinct
    pre-terminals ``p`` with ``(p, t) ∈ E``, saturating ``S``; ``None`` if none exists.

    Under FEAC with all capacities positive, a saturating matching of ``T``
    always exists [Lem 7.8]. Returns ``{t: p}``.
    """
    S = list(g.terminals) if S is None else list(S)
    nbrs = _neighbors(g, S)
    m = max_matching(S, nbrs)
    if len(m) < len(S):
        return None
    return m


def minimal_hall_deficient_set(g: DiGraphState) -> list[int] | None:
    """An inclusion-minimal nonempty ``S ⊆ T`` with no matching saturating ``S``
    (equivalently inclusion-minimal with ``|S| > |PT(G,S)|``, [Lem 7.6]).

    Procedure from the paper (proof of [Thm gyori-weighted-poly]): start with
    ``S = T``; repeatedly drop any terminal ``t`` such that ``S \\ {t}`` still has no
    saturating matching, restarting the scan after each removal; stop when every
    ``S \\ {t}`` is saturable. Returns ``None`` if ``T`` itself is saturable.
    """
    S = list(g.terminals)
    if saturating_matching(g, S) is not None:
        return None
    changed = True
    while changed:
        changed = False
        for t in list(S):
            S2 = [x for x in S if x != t]
            if saturating_matching(g, S2) is None:
                S = S2
                changed = True
                break
    assert S, "the empty set is always saturable"
    pt = g.pre_terminals(S)
    assert len(pt) == len(S) - 1, "[Lem 7.6] |PT(G,S)| = |S| - 1"
    return S
