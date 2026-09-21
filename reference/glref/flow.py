"""Unit-capacity max-flow on the vertex-split network and the tightest minimum cut.

Implements [Prop 4.2] of docs/paper_notes.md §3.1 literally: build the network
``H_v`` with vertex splitting, run breadth-first augmenting paths (at most
``k`` augmentations, since the value is ``κ_G(v) ≤ k``), then compute the set
``Reach`` of network nodes that can reach the sink in the residual graph and
read off the tightest minimum cut ``(L, S, R)``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .graph import DiGraphState

L, S, R = "L", "S", "R"


class FlowNetwork:
    """Plain adjacency-list flow network with paired residual arcs."""

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.head: list[int] = []
        self.cap: list[int] = []
        self.flow: list[int] = []
        self.adj: list[list[int]] = [[] for _ in range(num_nodes)]

    def add_arc(self, u: int, v: int, cap: int) -> int:
        """Add arc ``u→v`` with capacity ``cap``; returns its index (partner is index^1)."""
        idx = len(self.head)
        self.head.append(v)
        self.cap.append(cap)
        self.flow.append(0)
        self.adj[u].append(idx)
        self.head.append(u)
        self.cap.append(0)
        self.flow.append(0)
        self.adj[v].append(idx + 1)
        return idx

    def residual(self, a: int) -> int:
        return self.cap[a] - self.flow[a]

    def augment_once(self, s: int, z: int) -> int:
        """One BFS augmenting path (Edmonds–Karp step). Returns the pushed amount (0 if none)."""
        parent_arc = [-1] * self.num_nodes
        parent_arc[s] = -2
        dq = deque([s])
        while dq and parent_arc[z] == -1:
            u = dq.popleft()
            for a in self.adj[u]:
                v = self.head[a]
                if parent_arc[v] == -1 and self.residual(a) > 0:
                    parent_arc[v] = a
                    dq.append(v)
        if parent_arc[z] == -1:
            return 0
        # bottleneck
        amt = None
        v = z
        while v != s:
            a = parent_arc[v]
            r = self.residual(a)
            amt = r if amt is None else min(amt, r)
            v = self.head[a ^ 1]
        assert amt is not None and amt > 0
        v = z
        while v != s:
            a = parent_arc[v]
            self.flow[a] += amt
            self.flow[a ^ 1] -= amt
            v = self.head[a ^ 1]
        return amt

    def max_flow(self, s: int, z: int, limit: int | None = None) -> int:
        total = 0
        while limit is None or total < limit:
            pushed = self.augment_once(s, z)
            if pushed == 0:
                break
            total += pushed
        return total

    def reach_sink(self, z: int) -> list[bool]:
        """Nodes that can reach ``z`` in the residual graph (reverse BFS from ``z``)."""
        reach = [False] * self.num_nodes
        reach[z] = True
        dq = deque([z])
        while dq:
            y = dq.popleft()
            for b in self.adj[y]:
                # b is an arc y→x; its partner b^1 is x→y; x reaches y iff residual(b^1) > 0
                x = self.head[b]
                if not reach[x] and self.residual(b ^ 1) > 0:
                    reach[x] = True
                    dq.append(x)
        return reach


@dataclass
class TightestCut:
    """The tightest minimum cut separating ``v`` from ``T`` [Def 3.8]."""

    v: int
    kappa: int
    side: dict[int, str]  # vertex -> "L" | "S" | "R" (live vertices only)

    def separator(self) -> list[int]:
        return sorted(x for x, s in self.side.items() if s == S)

    def left(self) -> list[int]:
        return sorted(x for x, s in self.side.items() if s == L)

    def right(self) -> list[int]:
        return sorted(x for x, s in self.side.items() if s == R)

    def essential_terminals(self, g: DiGraphState) -> set[int]:
        """[Lem 4.1] essential terminals = terminals in the separator."""
        return {t for t in g.terminals if self.side[t] == S}


def build_split_network(g: DiGraphState, v: int) -> tuple[FlowNetwork, int, int]:
    """Construct ``H_v`` of [Prop 4.2] (paper_notes §3.1).

    Node ids: ``x_in = 2x``, ``x_out = 2x + 1``, source ``2n``, sink ``2n + 1``.
    Split arcs have capacity 1 except for ``v`` (capacity ``K = k + 1``);
    all other arcs have capacity ``K``.
    """
    n = g.n
    K = g.k + 1
    net = FlowNetwork(2 * n + 2)
    s, z = 2 * n, 2 * n + 1
    for x in range(n):
        if not g.live[x]:
            continue
        net.add_arc(2 * x, 2 * x + 1, K if x == v else 1)
        for y in g.out[x]:
            net.add_arc(2 * x + 1, 2 * y, K)
    for t in g.terminals:
        net.add_arc(2 * t + 1, z, K)
    net.add_arc(s, 2 * v, K)
    return net, s, z


def tightest_min_cut(g: DiGraphState, v: int) -> TightestCut:
    """[Prop 4.2] ``κ_G(v)`` and the tightest minimum cut via one max-flow.

    ``v`` must be a live non-terminal. The returned ``side`` map covers every
    live vertex; ``v`` is in ``R``, terminals are in ``L ∪ S``, ``|S| = κ``.
    """
    if not g.live[v] or g.is_terminal(v):
        raise ValueError(f"{v} must be a live non-terminal")
    net, s, z = build_split_network(g, v)
    kappa = net.max_flow(s, z, limit=g.k)
    reach = net.reach_sink(z)
    assert not reach[s], "augmenting path remains after max-flow"
    side: dict[int, str] = {}
    for x in range(g.n):
        if not g.live[x]:
            continue
        r_in, r_out = reach[2 * x], reach[2 * x + 1]
        if r_in:
            assert r_out, "x_in in Reach implies x_out in Reach [Prop 4.2]"
            side[x] = L
        elif r_out:
            side[x] = S
        else:
            side[x] = R
    cut = TightestCut(v=v, kappa=kappa, side=side)
    # sanity checks from the proof of [Prop 4.2]
    assert side[v] == R
    assert all(side[t] != R for t in g.terminals)
    assert len(cut.separator()) == kappa, (len(cut.separator()), kappa)
    return cut


def is_valid_cut(g: DiGraphState, v: int, side: dict[int, str]) -> bool:
    """[Def 3.3]: ``v ∈ R``, ``T ⊆ L ∪ S``, no arc from ``R`` to ``L``."""
    if side.get(v) != R:
        return False
    if any(side.get(t) == R for t in g.terminals):
        return False
    for x in range(g.n):
        if g.live[x] and side[x] == R:
            for y in g.out[x]:
                if side[y] == L:
                    return False
    return True


def cut_union(side1: dict[int, str], side2: dict[int, str]) -> dict[int, str]:
    """[Def 3.5] union of two cuts: ``L = L1 ∪ L2``, ``R = R1 ∩ R2``."""
    out: dict[int, str] = {}
    for x in side1:
        a, b = side1[x], side2[x]
        if a == L or b == L:
            out[x] = L
        elif a == R and b == R:
            out[x] = R
        else:
            out[x] = S
    return out


def cut_intersection(side1: dict[int, str], side2: dict[int, str]) -> dict[int, str]:
    """[Def 3.5] intersection of two cuts: ``L = L1 ∩ L2``, ``R = R1 ∪ R2``."""
    out: dict[int, str] = {}
    for x in side1:
        a, b = side1[x], side2[x]
        if a == L and b == L:
            out[x] = L
        elif a == R or b == R:
            out[x] = R
        else:
            out[x] = S
    return out
