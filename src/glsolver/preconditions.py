"""Independent NetworkX-based cross-checks of the theorem preconditions.

Everything here is computed *by definition* with NetworkX max-flows and is
deliberately slow and simple; it shares no code with the solvers.  Used by
tests as a cross-check of the fast oracles and by ``check_preconditions`` to
explain ``precondition_failed`` results (docs/paper_notes.md §13.9).

* :func:`terminal_connectivity_nx` -- ``κ_G(v)`` [Def 3.2] via the vertex-split
  network of paper_notes §3.1 [Prop 4.2];
* :func:`essential_terminals_by_definition` -- ``Ess(v)`` [Def 4.1] with
  ``k+1`` flows;
* :func:`is_k_T_connected` / :func:`is_k_T_connected_dag` -- [Def 3.2] and the
  DAG out-degree criterion [Lem 9.1];
* :func:`undirected_vertex_connectivity` -- classical ``k``-connectivity;
* :func:`feac_witness_nx` / :func:`fesac_witness_nx` -- witnesses of the
  Flow-Essential (Split-)Assignment Condition [Def 5.1] / [Def 5.3];
* :func:`check_preconditions` -- the entry point used by the solver front-end.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import networkx as nx
from networkx.algorithms.flow import edmonds_karp

from glsolver.instance import Instance

__all__ = [
    "AUTO_MAX_N",
    "terminal_connectivity_nx",
    "essential_terminals_by_definition",
    "essential_sets_by_definition",
    "is_k_T_connected",
    "is_dag",
    "is_k_T_connected_dag",
    "undirected_vertex_connectivity",
    "feac_witness_nx",
    "fesac_witness_nx",
    "check_preconditions",
]

#: ``check_preconditions(mode="auto")`` runs the (slow, flow-based) checks only
#: for instances with at most this many vertices.
AUTO_MAX_N = 400


# ---------------------------------------------------------------------------
# terminal connectivity [Def 3.2]
# ---------------------------------------------------------------------------


def _kappa_raw(n: int, arcs: Iterable[tuple[int, int]], terminals: Sequence[int], v: int) -> int:
    """``κ(v)`` [Def 3.2] on raw data via the vertex-split network ``H_v`` of
    paper_notes §3.1 [Prop 4.2]: every vertex ``x`` is split into ``x_in ->
    x_out`` with capacity 1 (capacity ``K = k+1`` for ``x = v``), arcs get
    capacity ``K``, ``s -> v_in`` and every ``t_out -> z`` get capacity ``K``.
    The max-flow value is the maximum number of ``v -> T`` paths ending at
    distinct terminals that are vertex-disjoint except for ``v``.
    """
    k = len(terminals)
    big = k + 1
    net = nx.DiGraph()
    net.add_node("s")
    net.add_node("z")
    for x in range(n):
        net.add_edge(("in", x), ("out", x), capacity=big if x == v else 1)
    net.add_edge("s", ("in", v), capacity=big)
    for x, y in arcs:
        net.add_edge(("out", x), ("in", y), capacity=big)
    for t in terminals:
        net.add_edge(("out", t), "z", capacity=big)
    value, _ = nx.maximum_flow(net, "s", "z", flow_func=edmonds_karp)
    return int(value)


def _check_nonterminal(inst: Instance, v: int) -> None:
    if not 0 <= v < inst.n:
        raise ValueError(f"vertex {v} out of range (n = {inst.n})")
    if v in inst.terminals:
        raise ValueError(f"vertex {v} is a terminal; κ is defined for non-terminals only")


def terminal_connectivity_nx(inst: Instance, v: int) -> int:
    """Terminal connectivity ``κ_G(v)`` of non-terminal ``v`` [Def 3.2]: the
    maximum number of paths from ``v`` to *distinct* terminals that are
    pairwise vertex-disjoint except for ``v``.  One NetworkX max-flow on the
    vertex-split network of paper_notes §3.1 [Prop 4.2]."""
    _check_nonterminal(inst, v)
    return _kappa_raw(inst.n, inst.arcs, inst.terminals, v)


def essential_terminals_by_definition(inst: Instance, v: int) -> set[int]:
    """``Ess_G(v)`` computed literally from [Def 4.1]:
    ``{t : κ_{G-t}(v) == κ_G(v) - 1}`` where ``G - t`` deletes the vertex ``t``
    (all arcs incident to it) and removes ``t`` from the terminal set.
    ``k + 1`` max-flows; ``κ_G(v) = 0`` gives the empty set."""
    _check_nonterminal(inst, v)
    base = _kappa_raw(inst.n, inst.arcs, inst.terminals, v)
    ess: set[int] = set()
    for t in inst.terminals:
        arcs_without_t = [(x, y) for x, y in inst.arcs if x != t and y != t]
        terminals_without_t = [u for u in inst.terminals if u != t]
        if _kappa_raw(inst.n, arcs_without_t, terminals_without_t, v) == base - 1:
            ess.add(t)
    return ess


def essential_sets_by_definition(inst: Instance) -> dict[int, set[int]]:
    """``Ess_G(v)`` [Def 4.1] for every non-terminal ``v`` (by definition)."""
    tset = set(inst.terminals)
    return {
        v: essential_terminals_by_definition(inst, v) for v in range(inst.n) if v not in tset
    }


def is_k_T_connected(inst: Instance) -> tuple[bool, int | None]:
    """``(True, None)`` if ``κ_G(v) = k`` for every non-terminal ``v`` [Def 3.2],
    otherwise ``(False, v)`` for the smallest violating vertex ``v``."""
    k = inst.k
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        if _kappa_raw(inst.n, inst.arcs, inst.terminals, v) < k:
            return False, v
    return True, None


def is_dag(inst: Instance) -> bool:
    """``True`` iff the arc digraph of ``inst`` is acyclic (NetworkX check)."""
    return bool(nx.is_directed_acyclic_graph(_arc_digraph(inst)))


def is_k_T_connected_dag(inst: Instance) -> tuple[bool, int | None]:
    """DAG criterion [Lem 9.1]: an acyclic instance is ``k``-``T``-connected iff
    every non-terminal has out-degree ``>= k``.  Returns ``(True, None)`` or
    ``(False, v)`` for the smallest non-terminal ``v`` with ``d^+(v) < k``.
    The verdict always agrees with :func:`is_k_T_connected`; the reported
    vertex may differ (a vertex of out-degree ``>= k`` can still have
    ``κ(v) < k`` when its out-neighbours are deficient, but then some
    non-terminal with ``d^+ < k`` exists).  Raises ``ValueError`` when the arc
    digraph is not acyclic."""
    digraph = _arc_digraph(inst)
    if not nx.is_directed_acyclic_graph(digraph):
        raise ValueError("is_k_T_connected_dag: the arc digraph is not acyclic")
    k = inst.k
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        if digraph.out_degree(v) < k:
            return False, v
    return True, None


def undirected_vertex_connectivity(inst: Instance) -> int:
    """Classical vertex connectivity of the original undirected graph
    (``networkx.node_connectivity`` on ``inst.undirected_edges``).  The
    Győri–Lovász guarantee [§1.1] needs this to be ``>= k``."""
    if inst.undirected_edges is None:
        raise ValueError("undirected_vertex_connectivity needs an undirected instance")
    graph = nx.Graph()
    graph.add_nodes_from(range(inst.n))
    graph.add_edges_from(inst.undirected_edges)
    return int(nx.node_connectivity(graph))


# ---------------------------------------------------------------------------
# FEAC / FESAC witnesses [Def 5.1] / [Def 5.3]
# ---------------------------------------------------------------------------


def _assignment_flow(
    inst: Instance,
    ess: dict[int, set[int]],
    supply: dict[int, int],
) -> tuple[int, dict[int, dict[int, int]]]:
    """Bipartite flow ``s -> v`` (cap ``supply[v]``), ``v -> t`` (cap
    ``supply[v]``, essential pairs only), ``t -> z`` (cap ``c_t``).  Returns
    the flow value and ``{v: {t: units}}`` restricted to positive entries."""
    net = nx.DiGraph()
    net.add_node("s")
    net.add_node("z")
    for i, t in enumerate(inst.terminals):
        net.add_edge(("t", t), "z", capacity=int(inst.capacities[i]))
    for v, cap in supply.items():
        net.add_edge("s", ("v", v), capacity=cap)
        for t in sorted(ess[v]):
            net.add_edge(("v", v), ("t", t), capacity=cap)
    value, flow = nx.maximum_flow(net, "s", "z", flow_func=edmonds_karp)
    psi: dict[int, dict[int, int]] = {}
    for v in supply:
        for node, units in flow[("v", v)].items():
            if units > 0:
                psi.setdefault(v, {})[node[1]] = int(units)
    return int(value), psi


def feac_witness_nx(
    inst: Instance, ess: dict[int, set[int]] | None = None
) -> dict[int, int] | None:
    """FEAC witness [Def 5.1] by definition: compute ``Ess(v)`` for every
    non-terminal (unless ``ess`` is supplied), then solve the bipartite flow
    ``s -> v`` (cap 1), ``v -> t`` (cap 1 iff ``t ∈ Ess(v)``), ``t -> z``
    (cap ``c_t``).  Returns the assignment ``{v: t}`` if the max-flow value is
    ``n - k``, else ``None`` (FEAC fails).  Unweighted instances only."""
    if inst.is_weighted:
        raise ValueError("feac_witness_nx is for unweighted instances; use fesac_witness_nx")
    if ess is None:
        ess = essential_sets_by_definition(inst)
    tset = set(inst.terminals)
    supply = {v: 1 for v in range(inst.n) if v not in tset}
    value, psi = _assignment_flow(inst, ess, supply)
    if value != inst.num_nonterminals:
        return None
    return {v: next(iter(psi[v])) for v in supply}


def fesac_witness_nx(
    inst: Instance, ess: dict[int, set[int]] | None = None
) -> dict[int, dict[int, int]] | None:
    """FESAC witness [Def 5.3] by definition: split-assignment ``ψ`` from the
    flow ``s -> v`` (cap ``w_v``), ``v -> t`` (cap ``w_v`` iff ``t ∈ Ess(v)``),
    ``t -> z`` (cap ``c_t``).  Returns ``{v: {t: ψ(v,t)}}`` (positive entries
    only) if the max-flow value equals ``Σ w_v``, else ``None``.  Unweighted
    instances are treated as unit-weight (then FESAC ⇔ FEAC, §13.6)."""
    if ess is None:
        ess = essential_sets_by_definition(inst)
    tset = set(inst.terminals)
    weights = inst.weights
    supply = {
        v: (1 if weights is None else int(weights[v])) for v in range(inst.n) if v not in tset
    }
    value, psi = _assignment_flow(inst, ess, supply)
    if value != sum(supply.values()):
        return None
    return {v: psi.get(v, {}) for v in supply}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def check_preconditions(inst: Instance, mode: Any = "auto") -> dict[str, Any]:
    """Cross-check the theorem preconditions of ``inst``.

    ``mode``: ``False`` -> nothing is checked (both results ``None``);
    ``True`` -> check ``k``-``T``-connectivity [Def 3.2] and FEAC [Def 5.1]
    (FESAC [Def 5.3] for weighted instances); ``"auto"`` -> same as ``True``
    when ``n <= AUTO_MAX_N`` (400), otherwise skipped because the by-definition
    computation costs ``O(n·k)`` NetworkX max-flows.

    Returns a dict with keys ``"k_T_connected"`` (bool | None), ``"feac"``
    (bool | None; means FESAC for weighted instances, see ``"condition"``),
    ``"message"``, plus ``"violating_vertex"`` (vertex with ``κ(v) < k`` or
    ``None``), ``"witness"`` (the FEAC/FESAC witness or ``None``) and
    ``"condition"`` (``"FEAC"``/``"FESAC"``).

    When the instance is ``k``-``T``-connected every terminal is essential for
    every non-terminal (``κ_{G-t}(v) <= k-1 = κ_G(v)-1``), so the witness flow
    is run with ``Ess(v) = T`` instead of ``k+1`` flows per vertex (paper
    §7.3, proof of [Thm k-t-conn]).
    """
    condition = "FESAC" if inst.is_weighted else "FEAC"
    result: dict[str, Any] = {
        "k_T_connected": None,
        "feac": None,
        "message": "",
        "violating_vertex": None,
        "witness": None,
        "condition": condition,
    }
    if mode is False or mode is None:
        result["message"] = "precondition checks disabled"
        return result
    if mode == "auto":
        if inst.n > AUTO_MAX_N:
            result["message"] = (
                f"precondition checks skipped in 'auto' mode (n = {inst.n} > {AUTO_MAX_N})"
            )
            return result
    elif mode is not True:
        raise ValueError(f"check_preconditions: mode must be False, True or 'auto', got {mode!r}")

    ok, bad = is_k_T_connected(inst)
    result["k_T_connected"] = ok
    result["violating_vertex"] = bad
    messages: list[str] = []
    if ok:
        messages.append(f"graph is {inst.k}-T-connected")
        tset = set(inst.terminals)
        ess: dict[int, set[int]] | None = {
            v: set(inst.terminals) for v in range(inst.n) if v not in tset
        }
    else:
        messages.append(
            f"graph is not {inst.k}-T-connected: κ(v) < {inst.k} for vertex {bad}"
        )
        ess = None
    witness: Any = (
        fesac_witness_nx(inst, ess) if inst.is_weighted else feac_witness_nx(inst, ess)
    )
    result["feac"] = witness is not None
    result["witness"] = witness
    if witness is not None:
        messages.append(f"{condition} holds (witness found)")
    else:
        messages.append(
            f"{condition} fails: no flow-essential assignment respects the capacities; "
            "the theorem's guarantee does not apply (a partition may still exist)"
        )
    result["message"] = "; ".join(messages)
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _arc_digraph(inst: Instance) -> nx.DiGraph:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(inst.n))
    digraph.add_edges_from(inst.arcs)
    return digraph
