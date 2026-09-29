"""Competitor adapters.

Two groups:

**(A) Exact solvers of the same problem.**  ``glsolver`` itself, the MILP
oracle it ships (HiGHS through ``scipy.optimize.milp``), the *same* flow MILP
handed to CBC / SCIP / HiGHS through PuLP and to Gurobi through ``gurobipy``,
and an OR-Tools CP-SAT model of the same problem.  The MILP formulation is the
one in ``src/glsolver/oracle/ilp.py`` (single commodity flow per part), copied
here verbatim in spirit so every solver gets an identical model.

**(B) Balanced graph partitioners.**  METIS, KaHIP, Mt-KaHyPar, NetworKit.
These do **not** solve the Győri–Lovász problem: none of them takes prescribed
roots as a first-class input (Mt-KaHyPar is the exception -- it has fixed
vertices, and we use them), none guarantees connected blocks (METIS is the
exception -- it has a ``contig`` option, and we use it), and all of them treat
the part sizes as a *balance constraint* rather than an equality.  We configure
each one as generously as its API allows and then measure the gap.

Every adapter returns

    {"status": ..., "parts": [[...], ...] | None, "build_s": float,
     "solve_s": float, "extra": {...}}

with ``status`` in ``{"ok", "infeasible", "timeout", "error", "unavailable",
"license_limit"}``.  ``parts[i]`` is the part that is supposed to contain
``inst.terminals[i]``.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import networkx as nx
import numpy as np

from glsolver.instance import Instance
from instances import target_sizes, undirected_view

Result = dict[str, Any]


def _res(status: str, parts=None, build_s=0.0, solve_s=0.0, **extra) -> Result:
    return {
        "status": status,
        "parts": [[int(v) for v in p] for p in parts] if parts is not None else None,
        "build_s": float(build_s),
        "solve_s": float(solve_s),
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# shared MILP description (identical to src/glsolver/oracle/ilp.py)
# ---------------------------------------------------------------------------
class FlowModel:
    """Index / coefficient description of the flow MILP, solver agnostic."""

    def __init__(self, inst: Instance):
        inst.validate()
        self.inst = inst
        self.n, self.k, self.m = inst.n, inst.k, inst.m
        self.tset = set(inst.terminals)
        self.nonterminals = [v for v in range(self.n) if v not in self.tset]
        self.B = max(1, len(self.nonterminals))
        self.weights = list(inst.weights) if inst.weights is not None else [1] * self.n
        self.arcs = [(int(u), int(v)) for u, v in inst.arcs]
        self.out_arcs: list[list[int]] = [[] for _ in range(self.n)]
        self.in_arcs: list[list[int]] = [[] for _ in range(self.n)]
        for a, (u, v) in enumerate(self.arcs):
            self.out_arcs[u].append(a)
            self.in_arcs[v].append(a)

    @property
    def num_vars(self) -> int:
        return self.n * self.k + self.k * self.m

    @property
    def num_cons(self) -> int:
        return len(self.nonterminals) + self.k + 2 * self.k * self.m + self.k * len(self.nonterminals)


def _decode_parts(inst: Instance, assign: Callable[[int, int], float]) -> list[list[int]]:
    parts: list[list[int]] = [[int(t)] for t in inst.terminals]
    tset = set(inst.terminals)
    for v in range(inst.n):
        if v in tset:
            continue
        i = max(range(inst.k), key=lambda j: assign(v, j))
        parts[i].append(v)
    return parts


# ---------------------------------------------------------------------------
# (A) exact solvers
# ---------------------------------------------------------------------------
def m_glsolver(inst: Instance, tl: float) -> Result:
    from glsolver.api import glpartition

    t0 = time.perf_counter()
    r = glpartition(inst, algorithm="auto", verify_preconditions=False, verify=False)
    dt = time.perf_counter() - t0
    if r.status != "ok":
        return _res(r.status if r.status in ("infeasible", "timeout") else "error",
                    None, 0.0, dt, message=r.message, algorithm=r.algorithm)
    return _res("ok", r.parts, 0.0, dt, algorithm=r.algorithm, backend=r.stats.get("backend"))


def m_glsolver_reference(inst: Instance, tl: float) -> Result:
    from glsolver.api import glpartition

    t0 = time.perf_counter()
    r = glpartition(inst, algorithm="reference", verify_preconditions=False, verify=False)
    dt = time.perf_counter() - t0
    if r.status != "ok":
        return _res(r.status if r.status in ("infeasible", "timeout") else "error", None, 0.0, dt,
                    message=r.message)
    return _res("ok", r.parts, 0.0, dt, algorithm=r.algorithm)


def m_bruteforce(inst: Instance, tl: float) -> Result:
    from glsolver.api import glpartition

    t0 = time.perf_counter()
    r = glpartition(inst, algorithm="bruteforce", verify_preconditions=False, verify=False)
    dt = time.perf_counter() - t0
    if r.status != "ok":
        return _res(r.status if r.status in ("infeasible", "timeout") else "error", None, 0.0, dt,
                    message=r.message)
    return _res("ok", r.parts, 0.0, dt)


def m_ilp_highs_scipy(inst: Instance, tl: float) -> Result:
    """The MILP oracle glsolver already ships: HiGHS via ``scipy.optimize.milp``."""
    from glsolver.oracle.ilp import ilp_partition

    fm = FlowModel(inst)
    t0 = time.perf_counter()
    status, parts = ilp_partition(inst, time_limit=tl)
    dt = time.perf_counter() - t0
    return _res(status, parts, 0.0, dt, num_vars=fm.num_vars, num_cons=fm.num_cons)


def _pulp_flow(inst: Instance, make_solver, tl: float) -> Result:
    """The flow MILP built with PuLP 4 and handed to ``make_solver(remaining)``.

    PuLP 4 creates variables through the problem (``add_variable_dict``) and
    reports the outcome in the ``LpSolveStats`` the solve returns, not on the
    problem object.
    """
    import pulp

    fm = FlowModel(inst)
    t0 = time.perf_counter()
    prob = pulp.LpProblem("gl", pulp.LpMinimize)
    x = prob.add_variable_dict("x", (range(fm.n), range(fm.k)), cat="Binary")
    f = prob.add_variable_dict("f", (range(fm.k), range(fm.m)), lowBound=0, upBound=fm.B)
    prob += 0
    for i, t in enumerate(inst.terminals):
        for j in range(fm.k):
            prob += x[(int(t), j)] == (1 if i == j else 0)
    for v in fm.nonterminals:
        prob += pulp.lpSum(x[(v, j)] for j in range(fm.k)) == 1
    slack = inst.w_max - 1 if inst.is_weighted else 0
    for i, c in enumerate(inst.capacities):
        expr = pulp.lpSum(fm.weights[v] * x[(v, i)] for v in fm.nonterminals)
        if inst.is_weighted:
            prob += expr <= int(c) + slack
        else:
            prob += expr == int(c)
    for i in range(fm.k):
        for a, (u, v) in enumerate(fm.arcs):
            prob += f[(i, a)] - fm.B * x[(u, i)] <= 0
            prob += f[(i, a)] - fm.B * x[(v, i)] <= 0
    for i in range(fm.k):
        for u in fm.nonterminals:
            prob += (pulp.lpSum(f[(i, a)] for a in fm.out_arcs[u])
                     - pulp.lpSum(f[(i, a)] for a in fm.in_arcs[u])
                     - x[(u, i)]) == 0
    build_s = time.perf_counter() - t0
    remaining = tl - build_s
    if remaining <= 1.0:
        return _res("timeout", None, build_s, 0.0, num_vars=fm.num_vars,
                    message="budget exhausted while building the model")

    t1 = time.perf_counter()
    try:
        stats = prob.solve(make_solver(remaining))
    except Exception as e:  # noqa: BLE001
        return _res("error", None, build_s, time.perf_counter() - t1,
                    num_vars=fm.num_vars, message=f"{type(e).__name__}: {e}")
    solve_s = time.perf_counter() - t1
    st = getattr(stats.status, "name", str(stats.status))
    if stats.has_solution:
        parts = _decode_parts(inst, lambda v, i: x[(v, i)].value() or 0.0)
        return _res("ok", parts, build_s, solve_s, num_vars=fm.num_vars, pulp_status=st)
    if st == "Infeasible":
        return _res("infeasible", None, build_s, solve_s, num_vars=fm.num_vars, pulp_status=st)
    return _res("timeout", None, build_s, solve_s, num_vars=fm.num_vars, pulp_status=st)


def m_ilp_cbc(inst: Instance, tl: float) -> Result:
    import pulp

    return _pulp_flow(inst, lambda r: pulp.COIN_CMD(msg=False, timeLimit=r, threads=1), tl)


def m_ilp_scip(inst: Instance, tl: float) -> Result:
    import pulp

    return _pulp_flow(inst, lambda r: pulp.SCIP_PY(msg=False, timeLimit=r), tl)


def m_ilp_highs_pulp(inst: Instance, tl: float) -> Result:
    import pulp

    return _pulp_flow(inst, lambda r: pulp.HiGHS(msg=False, timeLimit=r), tl)


def m_ilp_gurobi(inst: Instance, tl: float) -> Result:
    import gurobipy as gp
    from gurobipy import GRB

    fm = FlowModel(inst)
    t0 = time.perf_counter()
    try:
        env = gp.Env(params={"OutputFlag": 0})
        mdl = gp.Model(env=env)
        mdl.Params.OutputFlag = 0
        mdl.Params.Threads = 1
        x = mdl.addVars(fm.n, fm.k, vtype=GRB.BINARY, name="x")
        f = mdl.addVars(fm.k, fm.m, lb=0.0, ub=fm.B, name="f")
        for i, t in enumerate(inst.terminals):
            for j in range(fm.k):
                val = 1.0 if i == j else 0.0
                x[int(t), j].LB = val
                x[int(t), j].UB = val
        for v in fm.nonterminals:
            mdl.addConstr(gp.quicksum(x[v, j] for j in range(fm.k)) == 1)
        slack = inst.w_max - 1 if inst.is_weighted else 0
        for i, c in enumerate(inst.capacities):
            expr = gp.quicksum(fm.weights[v] * x[v, i] for v in fm.nonterminals)
            if inst.is_weighted:
                mdl.addConstr(expr <= int(c) + slack)
            else:
                mdl.addConstr(expr == int(c))
        for i in range(fm.k):
            for a, (u, v) in enumerate(fm.arcs):
                mdl.addConstr(f[i, a] <= fm.B * x[u, i])
                mdl.addConstr(f[i, a] <= fm.B * x[v, i])
        for i in range(fm.k):
            for u in fm.nonterminals:
                mdl.addConstr(gp.quicksum(f[i, a] for a in fm.out_arcs[u])
                              - gp.quicksum(f[i, a] for a in fm.in_arcs[u]) == x[u, i])
        mdl.setObjective(0.0)
        build_s = time.perf_counter() - t0
        mdl.Params.TimeLimit = max(1.0, tl - build_s)
        t1 = time.perf_counter()
        mdl.optimize()
        solve_s = time.perf_counter() - t1
    except gp.GurobiError as e:  # noqa: PERF203
        msg = str(e)
        st = "license_limit" if "size-limited" in msg or "too large" in msg else "error"
        return _res(st, None, time.perf_counter() - t0, 0.0, message=msg,
                    num_vars=fm.num_vars, num_cons=fm.num_cons)
    if mdl.Status == GRB.OPTIMAL or (mdl.SolCount > 0 and mdl.Status == GRB.TIME_LIMIT):
        parts = _decode_parts(inst, lambda v, i: x[v, i].X)
        return _res("ok", parts, build_s, solve_s, num_vars=fm.num_vars, gurobi_status=int(mdl.Status))
    if mdl.Status == GRB.INFEASIBLE:
        return _res("infeasible", None, build_s, solve_s, num_vars=fm.num_vars)
    return _res("timeout", None, build_s, solve_s, num_vars=fm.num_vars, gurobi_status=int(mdl.Status))


def m_cpsat(inst: Instance, tl: float) -> Result:
    """OR-Tools CP-SAT, native parent/level encoding of "reaches its terminal".

    Not the flow MILP: CP-SAT is handed the formulation it is good at --
    ``b[v,i]`` assignment booleans with exact cardinalities, one chosen parent
    arc per non-terminal, and a strictly decreasing integer level towards the
    terminal (an acyclic in-arborescence per part).  That is equivalent to
    "every vertex of part i reaches t_i inside G[V_i]" and is O(m) constraints
    instead of O(k·m) flow variables.
    """
    from ortools.sat.python import cp_model

    fm = FlowModel(inst)
    t0 = time.perf_counter()
    mdl = cp_model.CpModel()
    n, k = fm.n, fm.k
    b = [[mdl.NewBoolVar(f"b{v}_{i}") for i in range(k)] for v in range(n)]
    part = [mdl.NewIntVar(0, k - 1, f"p{v}") for v in range(n)]
    for v in range(n):
        mdl.AddExactlyOne(b[v])
        mdl.Add(part[v] == sum(i * b[v][i] for i in range(k)))
    for i, t in enumerate(inst.terminals):
        mdl.Add(part[int(t)] == i)
    slack = inst.w_max - 1 if inst.is_weighted else 0
    for i, c in enumerate(inst.capacities):
        expr = sum(fm.weights[v] * b[v][i] for v in fm.nonterminals)
        if inst.is_weighted:
            mdl.Add(expr <= int(c) + slack)
        else:
            mdl.Add(expr == int(c))
    level = [mdl.NewIntVar(0, n, f"l{v}") for v in range(n)]
    for t in inst.terminals:
        mdl.Add(level[int(t)] == 0)
    for u in fm.nonterminals:
        outs = fm.out_arcs[u]
        if not outs:
            return _res("infeasible", None, time.perf_counter() - t0, 0.0,
                        message=f"non-terminal {u} has no out-arc")
        ys = []
        for a in outs:
            y = mdl.NewBoolVar(f"y{a}")
            ys.append(y)
            v = fm.arcs[a][1]
            mdl.Add(part[u] == part[v]).OnlyEnforceIf(y)
            mdl.Add(level[u] >= level[v] + 1).OnlyEnforceIf(y)
        mdl.AddExactlyOne(ys)
    build_s = time.perf_counter() - t0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, float(tl) - build_s)
    solver.parameters.num_search_workers = 8
    t1 = time.perf_counter()
    st = solver.Solve(mdl)
    solve_s = time.perf_counter() - t1
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        parts = _decode_parts(inst, lambda v, i: solver.Value(b[v][i]))
        return _res("ok", parts, build_s, solve_s, cpsat_status=solver.StatusName(st))
    if st == cp_model.INFEASIBLE:
        return _res("infeasible", None, build_s, solve_s, cpsat_status=solver.StatusName(st))
    return _res("timeout", None, build_s, solve_s, cpsat_status=solver.StatusName(st))


# ---------------------------------------------------------------------------
# (B) balanced graph partitioners
# ---------------------------------------------------------------------------
def _match_blocks_to_terminals(inst: Instance, block_of: list[int], k: int) -> list[list[int]]:
    """Relabel a partitioner's blocks onto our terminals as generously as possible.

    A maximum-weight bipartite matching between the tool's blocks and our
    terminals; the weight of (block p, terminal i) is ``(n+1)`` if ``t_i`` lies
    in block p (so matching as many terminals as possible dominates) plus a
    size-agreement bonus ``n - ||block p| - n_i|``.  Ties therefore break
    towards the block whose size is closest to the required one.
    """
    from scipy.optimize import linear_sum_assignment

    n = inst.n
    sizes = target_sizes(inst)
    blocks: list[list[int]] = [[] for _ in range(k)]
    for v, p in enumerate(block_of):
        blocks[int(p) % k].append(v)
    big = n + 1
    W = np.zeros((k, k), dtype=float)
    for p in range(k):
        bs = set(blocks[p])
        for i in range(k):
            score = big if int(inst.terminals[i]) in bs else 0.0
            score += n - abs(len(blocks[p]) - sizes[i])
            W[p, i] = score
    rows, cols = linear_sum_assignment(-W)
    parts: list[list[int]] = [[] for _ in range(k)]
    for p, i in zip(rows, cols):
        parts[i] = blocks[p]
    return parts


def _csr(G: nx.Graph, n: int) -> tuple[list[int], list[int]]:
    xadj = [0]
    adjncy: list[int] = []
    for v in range(n):
        nbrs = sorted(G[v]) if v in G else []
        adjncy.extend(int(u) for u in nbrs)
        xadj.append(len(adjncy))
    return xadj, adjncy


def m_metis(inst: Instance, tl: float) -> Result:
    import pymetis

    G = undirected_view(inst)
    k = inst.k
    sizes = target_sizes(inst)
    tp = [s / inst.n for s in sizes]
    tp[-1] = max(0.0, 1.0 - sum(tp[:-1]))
    xadj, adjncy = _csr(G, inst.n)
    t0 = time.perf_counter()
    used_contig = True
    try:
        gp = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy, tpwgts=tp, contiguous=True)
    except Exception:  # noqa: BLE001
        used_contig = False
        try:
            gp = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy, tpwgts=tp)
        except Exception as e:  # noqa: BLE001
            return _res("error", None, 0.0, time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    parts = _match_blocks_to_terminals(inst, list(gp.vertex_part), k)
    return _res("ok", parts, 0.0, dt, contiguous=used_contig, edge_cuts=int(gp.edge_cuts))


def m_kahip(inst: Instance, tl: float) -> Result:
    import kahip

    G = undirected_view(inst)
    k = inst.k
    xadj, adjncy = _csr(G, inst.n)
    vwgt = [1] * inst.n
    adjcwgt = [1] * len(adjncy)
    t0 = time.perf_counter()
    try:
        edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, k, 0.03, True, 1, kahip.STRONG)
    except Exception as e:  # noqa: BLE001
        return _res("error", None, 0.0, time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    parts = _match_blocks_to_terminals(inst, list(blocks), k)
    return _res("ok", parts, 0.0, dt, edge_cut=int(edgecut), imbalance=0.03, mode="STRONG")


def m_mtkahypar(inst: Instance, tl: float) -> Result:
    """Mt-KaHyPar, the most generous configuration available anywhere here.

    Terminals are pinned with ``add_fixed_vertices`` (so the tool is *told*
    which root belongs to which block) and the exact required part sizes are
    passed as ``set_individual_target_block_weights``.  Only connectivity is
    left for it to get right by luck.
    """
    import mtkahypar

    G = undirected_view(inst)
    k = inst.k
    sizes = target_sizes(inst)
    edges = [(int(u), int(v)) for u, v in G.edges()]
    t0 = time.perf_counter()
    try:
        ini = mtkahypar.initialize(8, print_warnings=False)
        mtkahypar.set_seed(1)
        ctx = ini.context_from_preset(mtkahypar.PresetType.QUALITY)
        ctx.set_partitioning_parameters(k, 0.03, mtkahypar.Objective.CUT)
        ctx.set_individual_target_block_weights(list(sizes))
        ctx.logging = False
        g = ini.create_graph(ctx, inst.n, len(edges), edges)
        fixed = [-1] * inst.n
        for i, t in enumerate(inst.terminals):
            fixed[int(t)] = i
        g.add_fixed_vertices(fixed, k)
        build_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        ph = g.partition(ctx)
        solve_s = time.perf_counter() - t1
        block_of = [int(ph.block_id(v)) for v in range(inst.n)]
    except Exception as e:  # noqa: BLE001
        return _res("error", None, 0.0, time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
    parts = _match_blocks_to_terminals(inst, block_of, k)
    return _res("ok", parts, build_s, solve_s, fixed_vertices=True, individual_block_weights=True)


def m_scotch(inst: Instance, tl: float) -> Result:
    """Scotch through pyscotch, with the terminals pinned via ``partition_fixed``.

    Scotch has no prescribed-size input -- it balances the parts itself -- but it
    does take fixed vertices, so it is told which root belongs to which block.
    """
    import numpy as _np
    import pyscotch

    G = undirected_view(inst)
    k = inst.k
    t0 = time.perf_counter()
    try:
        pyscotch.random_reset()
        g = pyscotch.Graph()
        xadj, adjncy = _csr(G, inst.n)
        g.build(_np.asarray(xadj, dtype=_np.int64), _np.asarray(adjncy, dtype=_np.int64))
        fixed = _np.full(inst.n, -1, dtype=_np.int64)
        for i, t in enumerate(inst.terminals):
            fixed[int(t)] = i
        block_of = [int(b) % k for b in g.partition_fixed(k, fixed)]
    except Exception as e:  # noqa: BLE001
        return _res("error", None, 0.0, time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    parts = _match_blocks_to_terminals(inst, block_of, k)
    return _res("ok", parts, 0.0, dt, fixed_vertices=True)


def m_networkit(inst: Instance, tl: float) -> Result:
    import networkit as nk

    G = undirected_view(inst)
    k = inst.k
    t0 = time.perf_counter()
    try:
        g = nk.nxadapter.nx2nk(G)
        sp = nk.community.SpectralPartitioner(g, k, balanced=True)
        sp.run()
        p = sp.getPartition()
        block_of = [int(p.subsetOf(v)) for v in range(inst.n)]
    except Exception as e:  # noqa: BLE001
        return _res("error", None, 0.0, time.perf_counter() - t0, message=f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    ids = sorted({b for b in block_of})
    remap = {b: i for i, b in enumerate(ids)}
    block_of = [remap[b] for b in block_of]
    parts = _match_blocks_to_terminals(inst, block_of, k)
    return _res("ok", parts, 0.0, dt, algorithm="SpectralPartitioner(balanced)")


# ---------------------------------------------------------------------------
EXACT_METHODS: dict[str, Callable[[Instance, float], Result]] = {
    "glsolver": m_glsolver,
    "glsolver-reference": m_glsolver_reference,
    "bruteforce": m_bruteforce,
    "ilp-highs-scipy": m_ilp_highs_scipy,
    "ilp-cbc-pulp": m_ilp_cbc,
    "ilp-scip-pulp": m_ilp_scip,
    "ilp-highs-pulp": m_ilp_highs_pulp,
    "ilp-gurobi": m_ilp_gurobi,
    "cpsat-ortools": m_cpsat,
}
PARTITIONER_METHODS: dict[str, Callable[[Instance, float], Result]] = {
    "metis": m_metis,
    "kahip": m_kahip,
    "mtkahypar": m_mtkahypar,
    "scotch": m_scotch,
    "networkit-spectral": m_networkit,
}
ALL_METHODS = {**EXACT_METHODS, **PARTITIONER_METHODS}

# methods that get the "stop climbing after the first timeout" ladder
LADDER_METHODS = frozenset({
    "bruteforce", "ilp-highs-scipy", "ilp-cbc-pulp", "ilp-scip-pulp",
    "ilp-highs-pulp", "ilp-gurobi", "cpsat-ortools", "glsolver-reference",
})

__all__ = ["ALL_METHODS", "EXACT_METHODS", "PARTITIONER_METHODS", "LADDER_METHODS", "FlowModel"]
