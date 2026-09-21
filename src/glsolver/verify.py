"""Independent verifier for Győri–Lovász partitions.

This module is the *ground truth* of the project.  It deliberately shares no
code, data structure or assumption with any solver or oracle: it uses only the
standard library, the shared :class:`glsolver.instance.Instance` container and
plain breadth-first search on the **original** input (docs/paper_notes.md
§13.11 and §13.12).  Never import ``glref`` or a solver module here.

Checks performed by :func:`verify_instance_parts` (all errors are collected,
the verifier never stops at the first one):

(a) the number of parts equals ``k``;
(b) every vertex ``0..n-1`` appears in exactly one part (duplicates and missing
    vertices are reported individually);
(c) every vertex id is an in-range integer; an entry of a NetworkX partition
    that is not a node label of the graph is reported as unknown (it is never
    reinterpreted as an internal id);
(d) ``terminals[i]`` lies in ``parts[i]`` and no part contains a foreign terminal;
(e) unweighted: ``len(parts[i]) == capacities[i] + 1`` exactly;
    weighted: non-terminal weight of ``parts[i]`` ``<= capacities[i] + w_max - 1``
    [Thm weighted-k-t-conn]; parts exceeding ``capacities[i]`` are reported in
    ``details["exceeds_capacity"]`` (allowed, only rounding can cause it);
(f) connectivity: directed instance -> every vertex of ``parts[i]`` reaches
    ``terminals[i]`` by a directed path inside ``parts[i]`` [Def connected-to];
    undirected instance -> ``G[parts[i]]`` is connected in the original
    undirected edge set (and the directed check is run as well; the two must
    agree, paper_notes §1.1).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

from glsolver.instance import Instance, from_networkx, make_instance

__all__ = ["VerificationReport", "verify_partition", "verify_instance_parts"]


@dataclass
class VerificationReport:
    """Outcome of an independent verification.

    ``valid`` is ``True`` iff ``errors`` is empty.  ``details`` always contains
    ``"n"``, ``"k"``, ``"part_sizes"``, ``"bound_used"`` (``"exact"`` or
    ``"c_t + w_max - 1"``), ``"connectivity_ok"`` (one bool per checked part,
    i.e. for ``i < min(len(parts), k)``), ``"checked"`` (names of the checks
    that ran) and, for weighted instances, ``"part_weights"`` and
    ``"exceeds_capacity"``.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def verify_partition(
    graph: Any,
    terminals: Sequence[Any] | None = None,
    capacities: Sequence[int] | None = None,
    parts: Sequence[Sequence[Any]] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    weights: Any = None,
    directed: bool | None = None,
) -> VerificationReport:
    """Verify ``parts`` against an instance given in any supported input form.

    ``graph`` may be

    * an :class:`Instance` -- then ``terminals``, ``capacities``, ``sizes`` and
      ``weights`` must be ``None`` (they are taken from the instance);
    * a NetworkX ``Graph``/``DiGraph`` -- converted with
      :func:`glsolver.instance.from_networkx`; ``terminals`` and ``parts`` are
      then given as the graph's own node labels, never as internal ids (even
      when the labels are integers): a terminal that is not a node of the
      graph raises ``ValueError``, a part entry that is not a node of the
      graph is reported as an error of the partition (docs/api.md: the
      verifier judges the *original* input);
    * a tuple ``(n, edges)`` -- ``directed`` defaults to ``True``.

    Malformed *arguments* (missing ``parts``, contradictory keyword
    combinations, an instance that fails :meth:`Instance.validate`) raise
    ``ValueError``/``TypeError``; every problem with the *partition* itself is
    reported in the returned :class:`VerificationReport` instead.
    """
    if parts is None:
        raise ValueError("verify_partition: parts= is required")

    if isinstance(graph, Instance):
        if any(x is not None for x in (terminals, capacities, sizes, weights)):
            raise ValueError(
                "verify_partition: when graph is an Instance, terminals/capacities/"
                "sizes/weights must be None (they are taken from the instance)"
            )
        return verify_instance_parts(graph, parts)

    if terminals is None:
        raise ValueError("verify_partition: terminals= is required unless graph is an Instance")

    if isinstance(graph, tuple) and len(graph) == 2 and isinstance(graph[0], Integral):
        n, edges = graph
        inst = make_instance(
            int(n),
            edges,
            list(terminals),
            capacities,
            sizes=sizes,
            weights=weights,
            directed=True if directed is None else bool(directed),
        )
        return verify_instance_parts(inst, parts)

    if _looks_like_networkx(graph):
        inst, nodes = from_networkx(
            graph, list(terminals), capacities, sizes=sizes, weights=weights, directed=directed
        )
        index = {u: i for i, u in enumerate(nodes)}
        unknown = [t for t in terminals if isinstance(_map_label(index, t), _UnknownLabel)]
        if unknown:
            raise ValueError(
                f"verify_partition: terminals {unknown!r} are not nodes of the graph "
                "(terminals must be given as node labels, not as internal ids)"
            )
        mapped = [[_map_label(index, v) for v in part] for part in parts]
        return verify_instance_parts(inst, mapped)

    raise TypeError(
        "verify_partition: graph must be an Instance, a networkx Graph/DiGraph or a tuple (n, edges)"
    )


def verify_instance_parts(inst: Instance, parts: Sequence[Sequence[Any]]) -> VerificationReport:
    """Core verifier: check ``parts`` against a normalized :class:`Instance`.

    Implements checks (a)-(f) of the module docstring; see [Def connected-to],
    [Thm weighted-k-t-conn] and paper_notes §13.11/§13.12.  ``parts[i]`` must
    be the part of ``inst.terminals[i]``.  Entries wrapped in
    :class:`_UnknownLabel` (NetworkX labels that are not nodes of the graph,
    produced by :func:`verify_partition`) are reported as unknown labels.
    """
    errors: list[str] = []
    details: dict[str, Any] = {}
    checked: list[str] = []
    details["checked"] = checked

    n, k = inst.n, inst.k
    terminals = list(inst.terminals)
    capacities = list(inst.capacities)
    weighted = inst.weights is not None
    details["n"] = n
    details["k"] = k
    details["terminals"] = terminals
    details["capacities"] = capacities
    details["weighted"] = weighted
    details["directed"] = bool(inst.directed)

    checked.append("instance")
    inst.validate()  # raises ValueError on a malformed instance (API misuse, not a partition error)
    if not inst.directed and inst.undirected_edges is None:
        raise ValueError("undirected Instance without undirected_edges; build it with make_instance")

    parts_list: list[list[Any]] = [list(p) for p in parts]

    # (a) number of parts --------------------------------------------------
    checked.append("part_count")
    if len(parts_list) != k:
        errors.append(f"expected {k} parts (one per terminal), got {len(parts_list)}")

    # (c) vertex ids -------------------------------------------------------
    checked.append("vertex_ids")
    clean: list[list[int]] = []
    for i, part in enumerate(parts_list):
        good: list[int] = []
        for v in part:
            if isinstance(v, _UnknownLabel):
                errors.append(
                    f"part {i} contains unknown node label {v.label!r} (not a node of the graph)"
                )
                continue
            if isinstance(v, bool) or not isinstance(v, Integral):
                errors.append(f"part {i} contains non-integer vertex id {v!r}")
                continue
            v = int(v)
            if not 0 <= v < n:
                errors.append(f"part {i} contains out-of-range vertex id {v} (n = {n})")
                continue
            good.append(v)
        clean.append(good)

    # (b) exact cover --------------------------------------------------------
    checked.append("cover")
    owners: list[list[int]] = [[] for _ in range(n)]
    for i, good in enumerate(clean):
        for v in good:
            owners[v].append(i)
    for v in range(n):
        if not owners[v]:
            errors.append(f"vertex {v} is missing from the partition")
        elif len(owners[v]) > 1:
            errors.append(
                f"vertex {v} appears {len(owners[v])} times in the partition "
                f"(parts {sorted(set(owners[v]))})"
            )

    # (d) terminal membership ---------------------------------------------
    checked.append("terminal_membership")
    tindex = {t: j for j, t in enumerate(terminals)}
    checked_parts = min(len(clean), k)
    part_sets = [set(good) for good in clean]
    for i in range(checked_parts):
        if terminals[i] not in part_sets[i]:
            errors.append(f"terminal t_{i} (vertex {terminals[i]}) is not in part {i}")
        for v in sorted(part_sets[i]):
            j = tindex.get(v)
            if j is not None and j != i:
                errors.append(f"part {i} contains terminal t_{j} (vertex {v})")

    # (e) sizes / weights -------------------------------------------------
    part_sizes = [len(p) for p in parts_list]
    details["part_sizes"] = part_sizes
    if not weighted:
        checked.append("sizes")
        details["bound_used"] = "exact"
        for i in range(checked_parts):
            expected = capacities[i] + 1
            if part_sizes[i] != expected:
                errors.append(
                    f"part {i} has size {part_sizes[i]}, expected c_{i} + 1 = {expected}"
                )
    else:
        checked.append("weights")
        details["bound_used"] = "c_t + w_max - 1"
        w = inst.weights
        assert w is not None
        tset = set(terminals)
        w_max = max((w[v] for v in range(n) if v not in tset), default=1)
        details["w_max"] = w_max
        part_weights: list[int] = []
        exceeds: list[int] = []
        for i, pset in enumerate(part_sets):
            wt = sum(w[v] for v in pset if v not in tset)
            part_weights.append(wt)
            if i < k:
                bound = capacities[i] + w_max - 1
                if wt > capacities[i]:
                    exceeds.append(i)
                if wt > bound:
                    errors.append(
                        f"part {i} has non-terminal weight {wt}, exceeding the bound "
                        f"c_{i} + w_max - 1 = {capacities[i]} + {w_max} - 1 = {bound}"
                    )
        details["part_weights"] = part_weights
        details["exceeds_capacity"] = exceeds

    # (f) connectivity ----------------------------------------------------
    checked.append("connectivity")
    in_adj = inst.in_adjacency()
    und_adj: list[list[int]] | None = None
    if not inst.directed:
        assert inst.undirected_edges is not None
        und_adj = [[] for _ in range(n)]
        for u, v in inst.undirected_edges:
            und_adj[u].append(v)
            und_adj[v].append(u)

    connectivity_ok: list[bool] = []
    unreachable: dict[int, list[int]] = {}
    for i in range(checked_parts):
        pset = part_sets[i]
        t = terminals[i]
        if t not in pset:
            # [Def connected-to] requires t_i in the part; already reported in (d).
            connectivity_ok.append(False)
            unreachable[i] = sorted(pset)
            continue
        dir_unreached = sorted(pset - _reverse_bfs(t, pset, in_adj))
        if inst.directed:
            ok = not dir_unreached
            if not ok:
                errors.append(
                    f"part {i} is not connected to its terminal: vertices {dir_unreached} "
                    f"cannot reach terminal t_{i} (vertex {t}) by a directed path inside the part"
                )
        else:
            assert und_adj is not None
            und_unreached = sorted(pset - _bfs(t, pset, und_adj))
            ok = not und_unreached
            if not ok:
                errors.append(
                    f"part {i} is not connected: vertices {und_unreached} are not connected "
                    f"to terminal t_{i} (vertex {t}) inside the induced undirected subgraph"
                )
            has_foreign_terminal = any(v in tindex and v != t for v in pset)
            if (not dir_unreached) != ok and not has_foreign_terminal:
                # paper_notes §1.1: on symmetric digraphs both notions coincide.
                errors.append(
                    f"internal inconsistency in part {i}: undirected connectivity check says "
                    f"{'connected' if ok else 'disconnected'} but directed reachability check says "
                    f"{'connected' if not dir_unreached else 'disconnected'}"
                )
            dir_unreached = und_unreached
        connectivity_ok.append(ok)
        if dir_unreached:
            unreachable[i] = dir_unreached
    details["connectivity_ok"] = connectivity_ok
    details["unreachable"] = unreachable

    return VerificationReport(valid=not errors, errors=errors, details=details)


# ---------------------------------------------------------------------------
# helpers (plain BFS only)
# ---------------------------------------------------------------------------


def _reverse_bfs(root: int, allowed: set[int], in_adj: list[list[int]]) -> set[int]:
    """Vertices of ``allowed`` that reach ``root`` by a directed path whose
    vertices all lie in ``allowed`` (BFS from ``root`` over reversed arcs)."""
    seen = {root}
    queue: deque[int] = deque([root])
    while queue:
        x = queue.popleft()
        for u in in_adj[x]:
            if u in allowed and u not in seen:
                seen.add(u)
                queue.append(u)
    return seen


def _bfs(root: int, allowed: set[int], adj: list[list[int]]) -> set[int]:
    """Connected component of ``root`` in the undirected subgraph induced by ``allowed``."""
    seen = {root}
    queue: deque[int] = deque([root])
    while queue:
        x = queue.popleft()
        for u in adj[x]:
            if u in allowed and u not in seen:
                seen.add(u)
                queue.append(u)
    return seen


def _looks_like_networkx(obj: Any) -> bool:
    """Duck-type test for a NetworkX graph (keeps networkx an optional import)."""
    return all(hasattr(obj, name) for name in ("nodes", "edges", "is_directed"))


@dataclass(frozen=True)
class _UnknownLabel:
    """A part entry that is not a node label of the NetworkX graph being verified.

    Produced by :func:`_map_label` and reported by :func:`verify_instance_parts`
    as ``"part i contains unknown node label ..."``.
    """

    label: Any


def _map_label(index: dict[Any, int], label: Any) -> int | _UnknownLabel:
    """Map a NetworkX node label to its internal id.

    A label that is not a node of the graph (or is unhashable) is wrapped in
    :class:`_UnknownLabel` so that :func:`verify_instance_parts` reports it.
    It is never reinterpreted as an internal vertex id: an unknown integer
    label that happens to lie in ``0..n-1`` would otherwise let a partition of
    the wrong node set pass, and a partition given in internal ids would be
    judged against the wrong vertices whenever the graph's integer labels
    overlap ``0..n-1``.
    """
    try:
        return index[label]
    except (KeyError, TypeError):
        return _UnknownLabel(label)
