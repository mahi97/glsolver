"""Integral min-cost flow by successive shortest paths (reference implementation).

Used by :func:`glref.weighted.min_cost_split_assignment` [Prop 5.4] of
docs/paper_notes.md §5. All arc costs of that network are small nonnegative
integers (``ξ_v(t) ≤ k``), so the transparent choice is *successive shortest
paths* with a Bellman–Ford/SPFA shortest-path search on the residual graph for
every augmentation (residual arcs carry negative costs, which SPFA handles;
Dijkstra with potentials would be the optimised alternative). Augmentation is by
the bottleneck of the path, so all flows stay integral (paper_notes §5:
"Must be integral (SSP on integral data is)").
"""
from __future__ import annotations

from collections import deque

INF = float("inf")


class MinCostFlow:
    """Adjacency-list min-cost flow network with paired residual arcs.

    ``add_arc(u, v, cap, cost)`` returns the index ``idx`` of the forward arc;
    its residual partner is ``idx ^ 1`` (capacity 0, cost ``-cost``).
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.head: list[int] = []
        self.cap: list[int] = []
        self.cost: list[int] = []
        self._flow: list[int] = []
        self.adj: list[list[int]] = [[] for _ in range(num_nodes)]

    # ------------------------------------------------------------ building
    def add_arc(self, u: int, v: int, cap: int, cost: int) -> int:
        if cap < 0:
            raise ValueError("capacities must be nonnegative")
        idx = len(self.head)
        self.head.append(v)
        self.cap.append(cap)
        self.cost.append(cost)
        self._flow.append(0)
        self.adj[u].append(idx)
        self.head.append(u)
        self.cap.append(0)
        self.cost.append(-cost)
        self._flow.append(0)
        self.adj[v].append(idx + 1)
        return idx

    def flow(self, idx: int) -> int:
        """Flow currently on arc ``idx`` (negative on residual partners)."""
        return self._flow[idx]

    def residual(self, a: int) -> int:
        return self.cap[a] - self._flow[a]

    # ------------------------------------------------------------ solving
    def _shortest_path(self, s: int, z: int) -> list[int] | None:
        """SPFA (queue-based Bellman–Ford) on the residual graph.

        Returns ``parent_arc`` (arc index entering each node) or ``None`` when
        ``z`` is unreachable. The residual graph never contains a negative cycle
        along a successive-shortest-path run started from the zero flow with
        nonnegative costs, so SPFA terminates.
        """
        n = self.num_nodes
        dist: list[float] = [INF] * n
        parent: list[int] = [-1] * n
        in_queue = [False] * n
        dist[s] = 0
        dq: deque[int] = deque([s])
        in_queue[s] = True
        relaxations = 0
        limit = n * len(self.head) + 1  # negative-cycle guard (cannot trigger in SSP)
        while dq:
            u = dq.popleft()
            in_queue[u] = False
            du = dist[u]
            for a in self.adj[u]:
                if self.cap[a] - self._flow[a] <= 0:
                    continue
                v = self.head[a]
                nd = du + self.cost[a]
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = a
                    relaxations += 1
                    if relaxations > limit:
                        raise RuntimeError("negative cycle detected in residual graph")
                    if not in_queue[v]:
                        in_queue[v] = True
                        dq.append(v)
        if dist[z] == INF:
            return None
        return parent

    def min_cost_flow(self, s: int, z: int, required: int) -> tuple[int, int]:
        """Push up to ``required`` units from ``s`` to ``z`` at minimum cost.

        Successive shortest paths, each augmented by its bottleneck (capped at
        the remaining requirement). Returns ``(value, cost)``; ``value <
        required`` means no flow of the requested value exists.
        """
        if s == z:
            raise ValueError("source and sink must differ")
        value = 0
        total_cost = 0
        while value < required:
            parent = self._shortest_path(s, z)
            if parent is None:
                break
            # bottleneck along the path
            amt = required - value
            v = z
            while v != s:
                a = parent[v]
                r = self.cap[a] - self._flow[a]
                if r < amt:
                    amt = r
                v = self.head[a ^ 1]
            assert amt > 0
            v = z
            path_cost = 0
            while v != s:
                a = parent[v]
                self._flow[a] += amt
                self._flow[a ^ 1] -= amt
                path_cost += self.cost[a]
                v = self.head[a ^ 1]
            value += amt
            total_cost += amt * path_cost
        return value, total_cost

    def total_cost(self) -> int:
        """Cost of the current flow, ``Σ flow(a) · cost(a)`` over forward arcs."""
        return sum(self._flow[a] * self.cost[a] for a in range(0, len(self.head), 2))
