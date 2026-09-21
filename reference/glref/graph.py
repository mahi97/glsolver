"""Mutable directed graph state for the reference solvers.

Implements the graph conventions of docs/paper_notes.md §2: simple digraph,
terminals without outgoing arcs, lazy vertex deletion, and the three mutations
used by the algorithms — arc deletion, pre-terminal contraction [Def 2.1] and
terminal removal — plus the bookkeeping needed to recover the in-arborescence
in the *original* graph after contractions (§13.4).

Everything here is deliberately simple (dicts and sets); it is the transparent
reference, not the optimized core.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from glsolver.instance import Instance


class DiGraphState:
    """A simple digraph with terminals and support for the paper's mutations.

    Attributes
    ----------
    n : int
        Number of vertex ids (deleted vertices keep their id; see ``live``).
    terminals : list[int]
        Current terminal set, in the original order (removed terminals dropped).
    live : list[bool]
        ``live[v]`` is False once ``v`` was contracted or removed.
    out : list[dict[int, None]]
        ``out[u]`` is an insertion-ordered set of out-neighbours (dict keys).
    inn : list[set[int]]
        ``inn[v]`` is the set of in-neighbours.
    orig_head : dict[tuple[int, int], int]
        For every live arc ``(u, t)`` into a terminal, the head of the
        *original* arc it descends from (``t`` itself, or a pre-terminal that
        was contracted into ``t``). See paper_notes §13.4.
    """

    def __init__(self, n: int, arcs: Iterable[tuple[int, int]], terminals: Sequence[int]) -> None:
        self.n = n
        self.terminals: list[int] = list(terminals)
        self._is_terminal: list[bool] = [False] * n
        for t in self.terminals:
            self._is_terminal[t] = True
        self.live: list[bool] = [True] * n
        self.out: list[dict[int, None]] = [dict() for _ in range(n)]
        self.inn: list[set[int]] = [set() for _ in range(n)]
        self.orig_head: dict[tuple[int, int], int] = {}
        for u, v in arcs:
            if u == v or self._is_terminal[u]:
                continue  # WLOG conventions of paper §2
            if v in self.out[u]:
                continue
            self.out[u][v] = None
            self.inn[v].add(u)
            if self._is_terminal[v]:
                self.orig_head[(u, v)] = v

    # ------------------------------------------------------------------ basics
    @classmethod
    def from_instance(cls, inst: Instance) -> DiGraphState:
        return cls(inst.n, inst.arcs, inst.terminals)

    def copy(self) -> DiGraphState:
        g = DiGraphState.__new__(DiGraphState)
        g.n = self.n
        g.terminals = list(self.terminals)
        g._is_terminal = list(self._is_terminal)
        g.live = list(self.live)
        g.out = [dict(d) for d in self.out]
        g.inn = [set(s) for s in self.inn]
        g.orig_head = dict(self.orig_head)
        return g

    def is_terminal(self, v: int) -> bool:
        return self._is_terminal[v]

    @property
    def k(self) -> int:
        return len(self.terminals)

    def vertices(self) -> Iterator[int]:
        return (v for v in range(self.n) if self.live[v])

    def nonterminals(self) -> list[int]:
        return [v for v in range(self.n) if self.live[v] and not self._is_terminal[v]]

    def num_nonterminals(self) -> int:
        return sum(1 for v in range(self.n) if self.live[v] and not self._is_terminal[v])

    def arcs(self) -> list[tuple[int, int]]:
        return [(u, v) for u in range(self.n) if self.live[u] for v in self.out[u]]

    def num_arcs(self) -> int:
        return sum(len(self.out[u]) for u in range(self.n) if self.live[u])

    def out_degree(self, v: int) -> int:
        return len(self.out[v])

    def has_arc(self, u: int, v: int) -> bool:
        return v in self.out[u]

    def out_neighbors(self, v: int) -> list[int]:
        return list(self.out[v])

    def in_neighbors(self, v: int) -> list[int]:
        return sorted(self.inn[v])

    def terminal_out_neighbors(self, v: int) -> list[int]:
        return [x for x in self.out[v] if self._is_terminal[x]]

    def is_pre_terminal(self, v: int) -> bool:
        """[Def 2.1] non-terminal with an arc to a terminal."""
        return self.live[v] and not self._is_terminal[v] and any(
            self._is_terminal[x] for x in self.out[v]
        )

    def pre_terminals(self, S: Iterable[int] | None = None) -> list[int]:
        """``PT(G, S)`` [Def 2.1]: pre-terminals with an arc into ``S`` (default ``T``)."""
        Sset = set(self.terminals) if S is None else set(S)
        return [
            v
            for v in range(self.n)
            if self.live[v] and not self._is_terminal[v] and any(x in Sset for x in self.out[v])
        ]

    # --------------------------------------------------------------- mutations
    def delete_arc(self, u: int, v: int) -> None:
        if v not in self.out[u]:
            raise KeyError(f"arc ({u},{v}) not present")
        del self.out[u][v]
        self.inn[v].discard(u)
        self.orig_head.pop((u, v), None)

    def remove_vertex(self, v: int) -> None:
        """Delete ``v`` and all incident arcs (used for terminal removal and rounding)."""
        if not self.live[v]:
            raise KeyError(f"vertex {v} already deleted")
        for x in list(self.out[v]):
            self.delete_arc(v, x)
        for u in list(self.inn[v]):
            self.delete_arc(u, v)
        self.live[v] = False
        if self._is_terminal[v]:
            self.terminals.remove(v)
            self._is_terminal[v] = False

    def remove_terminal(self, t: int) -> None:
        """Operation (i) of [Alg 1]/[Alg 3]: delete terminal ``t`` [Lem 7.2]."""
        if not self._is_terminal[t]:
            raise ValueError(f"{t} is not a terminal")
        self.remove_vertex(t)

    def contract(self, p: int, t: int) -> int:
        """Contract pre-terminal ``p`` into terminal ``t`` [Def 2.1].

        Requires the arc ``(p, t)``. Incoming arcs ``(u, p)`` are redirected to
        ``(u, t)`` (merged if ``(u, t)`` already exists), all outgoing arcs of
        ``p`` are deleted, and ``p`` becomes dead.

        Returns the *original* out-neighbour of ``p`` behind the arc ``(p, t)``
        (``orig_head[(p, t)]``), which is the parent of ``p`` in the
        in-arborescence of the part of ``t`` (paper_notes §13.4).
        """
        if not self.live[p] or self._is_terminal[p]:
            raise ValueError(f"{p} is not a live non-terminal")
        if not self._is_terminal[t] or t not in self.out[p]:
            raise ValueError(f"arc ({p},{t}) into a terminal required for contraction")
        parent = self.orig_head[(p, t)]
        # redirect incoming arcs
        for u in list(self.inn[p]):
            del self.out[u][p]
            if t not in self.out[u]:
                self.out[u][t] = None
                self.inn[t].add(u)
                self.orig_head[(u, t)] = p
            # else: duplicate arc merged; keep existing (u,t) and its orig_head
        self.inn[p].clear()
        # delete outgoing arcs
        for x in list(self.out[p]):
            self.inn[x].discard(p)
            self.orig_head.pop((p, x), None)
        self.out[p].clear()
        self.live[p] = False
        return parent

    # ------------------------------------------------------------- utilities
    def check_invariants(self) -> None:
        """Debug: adjacency symmetry, no arcs out of terminals, orig_head coverage."""
        for u in range(self.n):
            if not self.live[u]:
                assert not self.out[u] and not self.inn[u], f"dead vertex {u} has arcs"
                continue
            for v in self.out[u]:
                assert self.live[v], f"arc ({u},{v}) to dead vertex"
                assert u in self.inn[v], f"in/out mismatch for ({u},{v})"
                assert not self._is_terminal[u], f"terminal {u} has an out-arc"
                if self._is_terminal[v]:
                    assert (u, v) in self.orig_head, f"missing orig_head for ({u},{v})"
            for x in self.inn[u]:
                assert u in self.out[x], f"in/out mismatch for ({x},{u})"
        for (u, v) in self.orig_head:
            assert self.live[u] and v in self.out[u], f"stale orig_head for ({u},{v})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DiGraphState(n={self.n}, live={sum(self.live)}, arcs={self.num_arcs()}, T={self.terminals})"
