"""Exact backtracking oracle for Győri–Lovász partitions.

This module is a *correctness baseline*: it never relies on any connectivity
precondition and it is complete, i.e. it returns ``"infeasible"`` only when
no partition satisfying the verifier's conditions exists
(docs/paper_notes.md §1.2 / §1.3):

* ``parts[i]`` contains ``terminals[i]``;
* unweighted: ``|parts[i]| == c_i + 1``;
* weighted: the non-terminal weight of part ``i`` is at most
  ``c_i + w_max - 1`` and every vertex is assigned to some part;
* every part is *connected to its terminal* [Def connected-to]: every vertex
  of the part has a directed path to the terminal inside the part.  For
  undirected instances the arcs are symmetric, so the same test is the usual
  undirected connectivity of ``G[V_i]`` (§1.1, §13.11).

Search strategy
---------------
Parts are built one at a time.  Part ``i`` is grown from ``{t_i}`` by
repeatedly adding an unassigned vertex ``u`` that has an arc ``(u, x)`` to a
vertex ``x`` already in the part; this guarantees that ``u`` reaches ``t_i``
inside the part.  To enumerate every connected set exactly once we use the
standard connected-set enumeration scheme: keep an ordered *frontier* of
candidates and, when branching on "exclude ``u``", forbid ``u`` for the rest
of this part.  The last part is not enumerated: the remaining vertices are
checked directly for size/weight and connectivity.
"""
from __future__ import annotations

import random
import time
from typing import Iterator

from glsolver.instance import Instance

_TIME_CHECK_PERIOD = 256


class _Timeout(Exception):
    """Internal signal: the deadline has been exceeded."""


class _Search:
    """State of the backtracking search (one object per solver call).

    All mutations are undone on backtracking so the object is reusable only
    inside a single :meth:`solutions` traversal.
    """

    def __init__(self, inst: Instance, *, deadline: float | None, rng: random.Random):
        inst.validate()
        self.inst = inst
        self.n = inst.n
        self.k = inst.k
        self.terminals = list(inst.terminals)
        self.deadline = deadline
        self.nodes = 0
        self.weighted = inst.weights is not None
        self.weights: list[int] = (
            list(inst.weights) if inst.weights is not None else [1] * self.n
        )
        for t in self.terminals:
            self.weights[t] = 0
        slack = inst.w_max - 1 if self.weighted else 0
        # bound[i]: maximal non-terminal weight (count) of part i.
        self.bound: list[int] = [c + slack for c in inst.capacities]
        # In the unweighted case the count must be exact (see §1.3: the sum
        # equality forces equality); the search only ever fills up to cap.
        self.exact = not self.weighted

        in_adj = inst.in_adjacency()
        # Deterministic tie-breaking: the seed only changes which solution is
        # found first, never the set of solutions.
        for lst in in_adj:
            lst.sort()
            rng.shuffle(lst)
        self.in_adj = in_adj

        self.assigned: list[int] = [-1] * self.n
        for i, t in enumerate(self.terminals):
            self.assigned[t] = i
        self.parts: list[list[int]] = [[t] for t in self.terminals]
        # forbid[v] == i  <=>  v is excluded for part i (current branch)
        self.forbid: list[int] = [-1] * self.n
        # infront[v] == i <=> v is currently in the frontier of part i
        self.infront: list[int] = [-1] * self.n
        self.unassigned_w = sum(self.weights[v] for v in range(self.n) if self.assigned[v] < 0)
        self.unassigned_cnt = self.n - self.k
        # Process the parts with the smallest bound first (cheap pruning);
        # the output is always in terminal order.
        self.order: list[int] = sorted(range(self.k), key=lambda i: (self.bound[i], i))
        # later_bound[pos] = sum of bounds of parts order[pos+1:]
        self.later_bound: list[int] = [0] * self.k
        acc = 0
        for pos in range(self.k - 1, -1, -1):
            self.later_bound[pos] = acc
            acc += self.bound[self.order[pos]]

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        self.nodes += 1
        if self.deadline is not None and self.nodes % _TIME_CHECK_PERIOD == 0:
            if time.perf_counter() >= self.deadline:
                raise _Timeout

    def solutions(self) -> Iterator[list[list[int]]]:
        """Yield every valid partition exactly once (parts in terminal order)."""
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise _Timeout
        yield from self._part(0)

    # ------------------------------------------------------------------
    def _part(self, pos: int) -> Iterator[list[list[int]]]:
        """Build part ``order[pos]``; the last part is checked directly."""
        if pos == self.k - 1:
            yield from self._last_part()
            return
        i = self.order[pos]
        t = self.terminals[i]
        frontier: list[int] = []
        saved: list[tuple[int, int]] = []
        for v in self.in_adj[t]:
            if self.assigned[v] < 0 and self.infront[v] != i:
                saved.append((v, self.infront[v]))
                self.infront[v] = i
                frontier.append(v)
        # avail: unassigned vertices not forbidden for part i (count / weight)
        yield from self._extend(pos, i, frontier, 0, self.unassigned_cnt, self.unassigned_w)
        for v, old in saved:
            self.infront[v] = old

    def _extend(
        self,
        pos: int,
        i: int,
        frontier: list[int],
        cur_w: int,
        avail_cnt: int,
        avail_w: int,
    ) -> Iterator[list[list[int]]]:
        self._tick()
        bound = self.bound[i]
        if self.exact:
            if cur_w == bound:
                yield from self._part(pos + 1)
                return
            if bound - cur_w > avail_cnt:
                return
        else:
            # Weight that can still enter part i is at most min(bound-cur, avail_w);
            # everything else must fit into the later parts.
            can_add = min(bound - cur_w, avail_w)
            if self.unassigned_w - can_add > self.later_bound[pos]:
                return
            if cur_w == bound:
                # No candidate fits any more (weights are >= 1): the only leaf
                # below this node excludes the whole frontier.
                yield from self._part(pos + 1)
                return
        if not frontier:
            if not self.exact:
                yield from self._part(pos + 1)
            return
        u = frontier[-1]
        rest = frontier[:-1]
        wu = self.weights[u]
        # ---- branch 1: include u -------------------------------------------
        if cur_w + wu <= bound:
            self.assigned[u] = i
            self.parts[i].append(u)
            self.unassigned_cnt -= 1
            self.unassigned_w -= wu
            added: list[tuple[int, int]] = []
            new_frontier = rest[:]
            for v in self.in_adj[u]:
                if self.assigned[v] < 0 and self.forbid[v] != i and self.infront[v] != i:
                    added.append((v, self.infront[v]))
                    self.infront[v] = i
                    new_frontier.append(v)
            yield from self._extend(
                pos, i, new_frontier, cur_w + wu, avail_cnt - 1, avail_w - wu
            )
            for v, old in added:
                self.infront[v] = old
            self.unassigned_w += wu
            self.unassigned_cnt += 1
            self.parts[i].pop()
            self.assigned[u] = -1
        # ---- branch 2: exclude u for the rest of part i --------------------
        old_forbid = self.forbid[u]
        self.forbid[u] = i
        yield from self._extend(pos, i, rest, cur_w, avail_cnt - 1, avail_w - wu)
        self.forbid[u] = old_forbid

    def _last_part(self) -> Iterator[list[list[int]]]:
        self._tick()
        i = self.order[self.k - 1]
        t = self.terminals[i]
        if self.unassigned_w > self.bound[i]:
            return
        if self.exact and self.unassigned_cnt != self.bound[i]:
            return
        # Every unassigned vertex must reach t through unassigned vertices.
        seen = [False] * self.n
        seen[t] = True
        stack = [t]
        reached = 0
        while stack:
            x = stack.pop()
            for v in self.in_adj[x]:
                if not seen[v] and self.assigned[v] < 0:
                    seen[v] = True
                    reached += 1
                    stack.append(v)
        if reached != self.unassigned_cnt:
            return
        rest = [v for v in range(self.n) if self.assigned[v] < 0]
        parts = [list(p) for p in self.parts]
        parts[i] = [t] + rest
        yield parts


def enumerate_partitions(
    inst: Instance,
    limit: int | None = None,
    *,
    seed: int = 0,
) -> Iterator[list[list[int]]]:
    """Enumerate *all* valid partitions of ``inst`` (each exactly once).

    Each solution is a list ``parts`` with ``parts[i]`` containing
    ``inst.terminals[i]`` first and satisfying the verifier's conditions of
    docs/paper_notes.md §1.2 (unweighted) / §1.3 (weighted, bound
    ``c_i + w_max - 1``) and [Def connected-to].  ``limit`` caps the number of
    solutions yielded.  The order of solutions depends on ``seed`` only.
    """
    if limit is not None and limit <= 0:
        return
    search = _Search(inst, deadline=None, rng=random.Random(seed))
    count = 0
    for parts in search.solutions():
        yield parts
        count += 1
        if limit is not None and count >= limit:
            return


def bruteforce_partition(
    inst: Instance,
    *,
    time_limit: float | None = None,
    seed: int = 0,
) -> tuple[str, list[list[int]] | None]:
    """Exact, complete backtracking search for a Győri–Lovász partition.

    Returns ``(status, parts)`` with ``status`` in ``{"ok", "infeasible",
    "timeout"}``.  ``"infeasible"`` is a *proof* that no partition satisfies
    the conditions of docs/paper_notes.md §1.2 / §1.3 with
    [Def connected-to]; no connectivity precondition ([Def 3.2], [Def 5.1]) is
    assumed.  ``time_limit`` is in seconds; ``seed`` only affects which of
    several solutions is returned first.
    """
    deadline = None if time_limit is None else time.perf_counter() + float(time_limit)
    search = _Search(inst, deadline=deadline, rng=random.Random(seed))
    try:
        for parts in search.solutions():
            return "ok", parts
    except _Timeout:
        return "timeout", None
    return "infeasible", None


__all__ = ["bruteforce_partition", "enumerate_partitions"]
