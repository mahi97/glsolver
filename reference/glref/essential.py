"""Essential terminals [Def 4.1] via the tightest minimum cut [Lem 4.1, Prop 4.2]."""
from __future__ import annotations

from .flow import TightestCut, tightest_min_cut
from .graph import DiGraphState


def essential_terminals(g: DiGraphState, v: int) -> tuple[int, set[int]]:
    """Return ``(κ_G(v), Ess_G(v))`` using one max-flow [Prop 4.2] + [Lem 4.1]."""
    cut = tightest_min_cut(g, v)
    return cut.kappa, cut.essential_terminals(g)


def essential_terminals_with_cut(g: DiGraphState, v: int) -> tuple[TightestCut, set[int]]:
    cut = tightest_min_cut(g, v)
    return cut, cut.essential_terminals(g)


def all_essential(g: DiGraphState) -> tuple[dict[int, int], dict[int, set[int]]]:
    """``κ`` and ``Ess`` for every live non-terminal (``|V\\T|`` max-flows)."""
    kappa: dict[int, int] = {}
    ess: dict[int, set[int]] = {}
    for v in g.nonterminals():
        kappa[v], ess[v] = essential_terminals(g, v)
    return kappa, ess


def terminal_connectivity(g: DiGraphState, v: int) -> int:
    """``κ_G(v)`` [Def 3.2] (one max-flow)."""
    return tightest_min_cut(g, v).kappa


def essential_terminals_by_definition(g: DiGraphState, v: int) -> tuple[int, set[int]]:
    """[Def 4.1] literally: ``t`` is essential iff ``κ_{G\\{t}}(v) = κ_G(v) - 1``.

    Uses ``k + 1`` max-flows; only for cross-checking [Lem 4.1] in tests.
    """
    kappa = terminal_connectivity(g, v)
    ess: set[int] = set()
    for t in list(g.terminals):
        h = g.copy()
        h.remove_terminal(t)
        if terminal_connectivity(h, v) == kappa - 1:
            ess.add(t)
    return kappa, ess
