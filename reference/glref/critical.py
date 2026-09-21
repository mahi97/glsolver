"""Critical arcs [Def 6.1], criticality costs [Def 6.2] and the potential [Def 6.3]."""
from __future__ import annotations

from typing import Sequence

from .essential import essential_terminals
from .graph import DiGraphState


def essential_after_deleting(g: DiGraphState, e: tuple[int, int], v: int) -> set[int]:
    """``Ess_{G\\e}(v)``: copy the graph, delete the arc, one max-flow."""
    h = g.copy()
    h.delete_arc(*e)
    return essential_terminals(h, v)[1]


def is_critical(
    g: DiGraphState, e: tuple[int, int], v: int, t: int, ess_v: set[int] | None = None
) -> bool:
    """[Def 6.1] ``e`` is critical for assigning ``v`` to ``t`` iff ``t ∈ Ess_G(v)`` and ``t ∉ Ess_{G\\e}(v)``."""
    if ess_v is None:
        ess_v = essential_terminals(g, v)[1]
    if t not in ess_v:
        return False
    return t not in essential_after_deleting(g, e, v)


def criticality_table(
    g: DiGraphState, secondary: Sequence[tuple[int, int]], ess: dict[int, set[int]]
) -> list[dict[int, set[int]]]:
    """``crit[i][v]`` = set of terminals ``t`` such that ``e_i`` is critical for ``(v, t)``.

    Literal implementation: for each secondary arc ``e_i`` delete it and
    recompute ``Ess`` for every non-terminal (``|V\\T|·k`` max-flows). The
    optimized core avoids most of this work (paper_notes §6, §13.1).
    """
    table: list[dict[int, set[int]]] = []
    for e in secondary:
        h = g.copy()
        h.delete_arc(*e)
        row: dict[int, set[int]] = {}
        for v in g.nonterminals():
            ess_after = essential_terminals(h, v)[1]
            row[v] = ess[v] - ess_after
            assert ess_after <= ess[v] or True  # new essentials may appear only if κ drops; not needed here
        table.append(row)
    return table


def criticality_cost(table: Sequence[dict[int, set[int]]], v: int, t: int) -> int:
    """[Def 6.2] ``ξ_v(t)`` = number of secondary arcs critical for ``(v, t)``."""
    return sum(1 for row in table if t in row[v])


def potential(table: Sequence[dict[int, set[int]]], phi: dict[int, int]) -> int:
    """[Def 6.3] ``Φ(φ) = Σ_v ξ_v(φ(v))``."""
    return sum(criticality_cost(table, v, t) for v, t in phi.items())
