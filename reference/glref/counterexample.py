"""The authors' official counterexample (README of ``mahdi-jfri/Gyori-Lovasz-Codes``,
paper [Lem A.13]; docs/paper_notes.md §10): an instance that satisfies the
compact connectivity condition [Def A.5] while deleting **any** arc or
contracting **any** pre-terminal into either of its terminals breaks it.

``k = 9``. For each pair ``i < j`` three pre-terminals ``p^1_{ij}, p^2_{ij},
p^3_{ij}`` with out-arcs to ``t_i`` and ``t_j`` only. For each arc
``e = (p^r_{x,y}, t_y)`` let ``a < b < c < d`` be the four smallest indices of
``[9] \\ {x, y}`` and add ``copies`` forcing vertices ``v_e`` (one vertex of
multiplicity ``copies`` in the compact form) with
``N^+(v_e) = P_{a,b} ∪ P_{a,c} ∪ {p^1_{d,x}, p^2_{d,x}, p^r_{x,y}}`` (9 arcs).
Capacities: ``p^1, p^3 → t_j``, ``p^2 → t_i``, all copies of ``v_e → t_a``;
``c_t`` is the number of vertices assigned to ``t``.

:func:`build_counterexample_compact` reproduces the authors' ``build()``
exactly (same vertex order, out-neighbourhoods, multiplicities and
capacities); :func:`build_counterexample_instance` expands the multiplicities
into distinct forcing vertices; :func:`check_counterexample_claims`
replicates the authors' ``main()`` including its "suspects" short-cut.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from glsolver.instance import Instance

from .compact import (
    CompactInstance,
    compact_sets,
    contract,
    delete_edge,
    satisfies_compact_condition,
)

K = 9


def _build(copies: int) -> tuple[CompactInstance, list[str], dict[str, Any]]:
    """Port of the authors' ``build(copies)`` plus the abstract structure."""
    out_adj: list[list[int]] = [[] for _ in range(K)]
    mult = [1] * K
    name = [f"t{i + 1}" for i in range(K)]
    home: dict[int, int] = {}
    pre_info: dict[int, dict[str, Any]] = {}
    forcing: list[dict[str, Any]] = []

    def add(label: str, targets: list[int], weight: int, assigned: int) -> int:
        v = len(out_adj)
        out_adj.append(list(targets))
        mult.append(weight)
        name.append(label)
        home[v] = assigned
        return v

    P: dict[tuple[int, int], list[int]] = {}
    for i in range(K):
        for j in range(i + 1, K):
            # p^1 and p^3 are assigned to t_j, p^2 to t_i.
            P[i, j] = P[j, i] = [
                add(f"p^{r + 1}_{{{i + 1},{j + 1}}}", [i, j], 1, i if r % 2 else j)
                for r in range(3)
            ]
            for r, p in enumerate(P[i, j]):
                pre_info[p] = {"pair": (i, j), "r": r + 1, "assigned": home[p], "name": name[p]}

    for p, y in [(q, t) for q in range(K, len(out_adj)) for t in out_adj[q]]:
        x = out_adj[p][0] if out_adj[p][1] == y else out_adj[p][1]
        a, b, c, d = [i for i in range(K) if i not in (x, y)][:4]
        targets = P[a, b] + P[a, c] + P[d, x][:2] + [p]
        v = add(f"v({name[p]}->t{y + 1})", targets, copies, a)
        forcing.append({
            "vertex": v, "name": name[v], "edge": (p, y), "x": x, "y": y,
            "a": a, "b": b, "c": c, "d": d, "targets": list(targets), "assigned": a,
            "copies": copies,
        })

    cap = [0] * K
    for v, t in home.items():
        cap[t] += mult[v]
    structure = {
        "k": K,
        "copies": copies,
        "pair_to_pre_terminals": {(i, j): list(P[i, j]) for i in range(K) for j in range(i + 1, K)},
        "pre_terminals": pre_info,
        "forcing": forcing,
        "names": list(name),
        "capacities": list(cap),
    }
    return CompactInstance(K, out_adj, mult, cap), name, structure


def build_counterexample_compact(copies: int = 17) -> tuple[CompactInstance, list[str]]:
    """The authors' ``build(copies)``: compact instance (terminals ``0..8``,
    108 pre-terminals ``9..116``, 216 forcing vertices of multiplicity
    ``copies``) and the vertex names."""
    ci, names, _ = _build(copies)
    return ci, names


def counterexample_structure(copies: int = 17) -> dict[str, Any]:
    """Abstract structure for visualization: ``pair_to_pre_terminals``
    (``(i, j) -> [p^1, p^2, p^3]`` compact ids), ``pre_terminals`` (per id:
    pair, ``r``, assigned terminal, name), ``forcing`` (per forcing vertex: the
    targeted arc ``(p, y)``, ``x, y, a, b, c, d``, its 9 targets, assigned
    terminal, copies), ``names`` and ``capacities``. Ids are compact ids; the
    expanded instance's ids are given by ``inst.meta["compact_vertices"]``."""
    return _build(copies)[2]


def build_counterexample_instance(copies: int = 17) -> Instance:
    """The counterexample as a directed unweighted :class:`Instance`, with the
    ``copies`` multiplicity expanded into distinct forcing vertices (each with
    the same 9 out-arcs). For ``copies = 17``: ``k = 9``, 108 pre-terminals,
    216 pre-terminal arcs, 3672 forcing vertices, ``n = 3789``.
    ``inst.meta`` carries ``"structure"`` (see :func:`counterexample_structure`)
    and ``"compact_vertices"`` (compact id → instance ids)."""
    ci, _names, structure = _build(copies)
    inst = ci.to_instance(name=f"gl_counterexample_copies{copies}")
    # JSON-safe copy for instance files (pair keys "i,j"); counterexample_structure() keeps tuple keys.
    meta_structure = dict(structure)
    meta_structure["pair_to_pre_terminals"] = {
        f"{i},{j}": v for (i, j), v in structure["pair_to_pre_terminals"].items()
    }
    inst.meta["structure"] = meta_structure
    inst.meta["copies"] = copies
    return inst


def _fails(ci: CompactInstance, suspects: list[int]) -> bool:
    """Authors' ``fails``: whether ``ci`` violates the condition. ``suspects``
    are the vertices whose compact sets the last modification may have
    changed; if one of them is empty the condition fails at once, otherwise
    the full condition is re-evaluated."""
    sets = compact_sets(ci, suspects)
    return any(not sets[v] for v in suspects) or not satisfies_compact_condition(ci)


def _report(label: str, done: int, total: int, start: float, progress: bool) -> None:
    if not progress or (done % 10 and done != total):
        return
    elapsed = time.time() - start
    end = "\n" if done == total else ""
    sys.stderr.write(f"\r{label} {done}/{total} ({elapsed:.0f}s){end}")


def check_counterexample_claims(
    copies: int = 17, skip_contractions: bool = False, progress: bool = False
) -> dict[str, Any]:
    """Replicate the authors' ``main()``: build the instance, check that the
    compact connectivity condition holds, that every single arc deletion breaks
    it and (unless ``skip_contractions``) that every contraction of a
    pre-terminal into either of its terminals breaks it, using the "suspects"
    short-cut exactly as they do. Returns counts (and the elapsed time);
    ``deletions_break == deletions_total`` etc. are the claims.

    Runtime (pure Python, measured): ``copies = 1`` with
    ``skip_contractions=True`` about 4 s; the full ``copies = 17`` replication
    37–43 s — the multiplicities keep the graph at 333 vertices regardless of
    ``copies``, and 1944 of the 2160 deletions short-circuit on an empty
    compact set of the forcing vertex.
    """
    start = time.time()
    ci, name = build_counterexample_compact(copies)
    pre = [v for v in ci.non_terminals() if all(t < K for t in ci.out_adj[v])]
    others = sorted(set(ci.non_terminals()) - set(pre))
    edges = [(u, v) for u in ci.non_terminals() for v in ci.out_adj[u]]
    result: dict[str, Any] = {
        "k": K,
        "copies": copies,
        "pre_terminals": len(pre),
        "forcing_vertices": len(others),
        "edges": len(edges),
        "cap": list(ci.cap),
    }
    assert all(len(ci.out_adj[v]) == 2 for v in pre)
    assert all(len(ci.out_adj[v]) == K for v in others)
    assert ci.total_cap() == ci.total_mult()

    sets = compact_sets(ci)
    result["compact_holds"] = satisfies_compact_condition(ci, sets)

    breaks = 0
    preserving: list[str] = []
    for i, (u, v) in enumerate(edges):
        if _fails(delete_edge(ci, u, v), [u] + ci.in_adj[u]):
            breaks += 1
        else:
            preserving.append(f"deleting {name[u]} -> {name[v]}")
        _report("edge deletions", i + 1, len(edges), start, progress)
    result["deletions_break"] = breaks
    result["deletions_total"] = len(edges)

    if skip_contractions:
        result["contractions_break"] = 0
        result["contractions_total"] = 0
    else:
        pairs = [(p, t) for p in pre for t in ci.out_adj[p]]
        breaks = 0
        for i, (p, t) in enumerate(pairs):
            smaller, new_id = contract(ci, p, t)
            if _fails(smaller, [new_id[w] for w in ci.in_adj[p]]):
                breaks += 1
            else:
                preserving.append(f"contracting {name[p]} into {name[t]}")
            _report("contractions", i + 1, len(pairs), start, progress)
        result["contractions_break"] = breaks
        result["contractions_total"] = len(pairs)
    result["preserving_operations"] = preserving
    result["seconds"] = time.time() - start
    return result
