"""Exact mixed-integer programming oracle (HiGHS via :func:`scipy.optimize.milp`).

Single-commodity-flow-per-part formulation of the Győri–Lovász partition
problem of docs/paper_notes.md §1.2 (unweighted) / §1.3 (weighted):

* ``x[v,i] ∈ {0,1}``: vertex ``v`` belongs to part ``i``;
  ``Σ_i x[v,i] = 1`` for non-terminals, ``x[t_i,i] = 1`` and ``x[t_i,j] = 0``
  for ``j ≠ i`` (fixed through variable bounds);
* unweighted: ``Σ_v x[v,i] = c_i + 1``;
  weighted: ``Σ_{v ∉ T} w_v x[v,i] ≤ c_i + w_max − 1``;
* for each part ``i`` and arc ``a = (u,v)`` a continuous flow
  ``f[i,a] ≥ 0`` with ``f[i,a] ≤ B·x[u,i]`` and ``f[i,a] ≤ B·x[v,i]``
  (``B`` = number of non-terminals);
* for every non-terminal ``u`` and part ``i``:
  ``Σ_out f[i,u,*] − Σ_in f[i,*,u] = x[u,i]``.

Every vertex of part ``i`` ships one unit towards ``t_i`` along arcs whose
both endpoints are in part ``i``; terminals absorb (they have no out-arcs,
§2).  Hence each vertex of part ``i`` reaches ``t_i`` inside ``G[V_i]``
[Def connected-to].  Conversely a valid partition admits such a flow along
the spanning in-arborescence of each part (``f`` = subtree size ≤ ``B``), so
the formulation is exact.  The objective is ``0`` (pure feasibility).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from glsolver.instance import Instance
from glsolver.verify import verify_instance_parts


@dataclass(frozen=True)
class VarIndex:
    """Column indexing of the MILP.

    Columns ``0 .. n*k-1`` are the binary ``x[v,i]`` (row-major in ``v``),
    columns ``n*k .. n*k + k*m - 1`` are the flows ``f[i,a]`` for arc index
    ``a`` (row-major in ``i``).
    """

    n: int
    k: int
    m: int

    @property
    def num_x(self) -> int:
        return self.n * self.k

    @property
    def num_f(self) -> int:
        return self.k * self.m

    @property
    def num_vars(self) -> int:
        return self.num_x + self.num_f

    def x(self, v: int, i: int) -> int:
        """Column of ``x[v,i]``."""
        if not (0 <= v < self.n and 0 <= i < self.k):
            raise IndexError(f"x[{v},{i}] out of range")
        return v * self.k + i

    def f(self, i: int, a: int) -> int:
        """Column of the flow of part ``i`` on arc number ``a``."""
        if not (0 <= i < self.k and 0 <= a < self.m):
            raise IndexError(f"f[{i},{a}] out of range")
        return self.num_x + i * self.m + a

    def decode(self, sol: np.ndarray, terminals: tuple[int, ...]) -> list[list[int]]:
        """Read the parts (terminal first, then non-terminals ascending) off a solution."""
        parts: list[list[int]] = [[t] for t in terminals]
        tset = set(terminals)
        for v in range(self.n):
            if v in tset:
                continue
            i = int(np.argmax([sol[self.x(v, j)] for j in range(self.k)]))
            parts[i].append(v)
        return parts


class _Model:
    """Sparse (COO) assembly of the constraint matrix."""

    def __init__(self, ncols: int):
        self.ncols = ncols
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    @property
    def nrows(self) -> int:
        return len(self.lb)

    def add_row(self, entries: list[tuple[int, float]], lb: float, ub: float) -> None:
        r = self.nrows
        for c, val in entries:
            self.rows.append(r)
            self.cols.append(c)
            self.vals.append(val)
        self.lb.append(lb)
        self.ub.append(ub)

    def constraint(self) -> LinearConstraint | None:
        if self.nrows == 0:
            return None
        A = sp.coo_matrix(
            (np.asarray(self.vals, dtype=float), (np.asarray(self.rows), np.asarray(self.cols))),
            shape=(self.nrows, self.ncols),
        ).tocsr()
        return LinearConstraint(A, np.asarray(self.lb, dtype=float), np.asarray(self.ub, dtype=float))


def build_model(inst: Instance) -> tuple[VarIndex, np.ndarray, Bounds, np.ndarray, LinearConstraint | None]:
    """Assemble ``(index, c, bounds, integrality, constraints)`` for :func:`milp`.

    Exposed for tests; see the module docstring for the formulation
    (docs/paper_notes.md §1.2, §1.3, [Def connected-to]).
    """
    inst.validate()
    n, k, m = inst.n, inst.k, inst.m
    idx = VarIndex(n, k, m)
    tset = set(inst.terminals)
    nonterminals = [v for v in range(n) if v not in tset]
    B = float(max(1, len(nonterminals)))
    weights = list(inst.weights) if inst.weights is not None else [1] * n

    lb = np.zeros(idx.num_vars)
    ub = np.full(idx.num_vars, B)
    ub[: idx.num_x] = 1.0
    for i, t in enumerate(inst.terminals):
        for j in range(k):
            col = idx.x(t, j)
            lb[col] = ub[col] = 1.0 if i == j else 0.0
    integrality = np.zeros(idx.num_vars)
    integrality[: idx.num_x] = 1.0

    model = _Model(idx.num_vars)
    # (1) each non-terminal in exactly one part
    for v in nonterminals:
        model.add_row([(idx.x(v, j), 1.0) for j in range(k)], 1.0, 1.0)
    # (2) size / weight of each part
    slack = inst.w_max - 1 if inst.is_weighted else 0
    for i, c in enumerate(inst.capacities):
        entries = [(idx.x(v, i), float(weights[v])) for v in nonterminals]
        if inst.is_weighted:
            model.add_row(entries, -np.inf, float(c + slack))
        else:
            model.add_row(entries, float(c), float(c))
    # (3) flow only on arcs inside the part
    for i in range(k):
        for a, (u, v) in enumerate(inst.arcs):
            model.add_row([(idx.f(i, a), 1.0), (idx.x(u, i), -B)], -np.inf, 0.0)
            model.add_row([(idx.f(i, a), 1.0), (idx.x(v, i), -B)], -np.inf, 0.0)
    # (4) conservation: every non-terminal of part i ships one unit towards t_i
    out_arcs: list[list[int]] = [[] for _ in range(n)]
    in_arcs: list[list[int]] = [[] for _ in range(n)]
    for a, (u, v) in enumerate(inst.arcs):
        out_arcs[u].append(a)
        in_arcs[v].append(a)
    for i in range(k):
        for u in nonterminals:
            entries = [(idx.f(i, a), 1.0) for a in out_arcs[u]]
            entries += [(idx.f(i, a), -1.0) for a in in_arcs[u]]
            entries.append((idx.x(u, i), -1.0))
            model.add_row(entries, 0.0, 0.0)

    c_obj = np.zeros(idx.num_vars)
    return idx, c_obj, Bounds(lb, ub), integrality, model.constraint()


def _is_valid(inst: Instance, parts: list[list[int]]) -> bool:
    """Gate for salvaging a time-limited HiGHS incumbent.

    ``True`` iff the independent verifier
    (:func:`glsolver.verify.verify_instance_parts`) accepts ``parts``: exact
    cover without repeated vertices, terminal membership, sizes (unweighted)
    or the ``c_i + w_max - 1`` bound (weighted, [Thm weighted-k-t-conn]) and
    connectivity to the terminal inside each part [Def connected-to].
    Delegating keeps this gate exactly as strict as the verifier; a private
    re-implementation once worked on ``set(part)`` and accepted a part that
    listed the same vertex twice.
    """
    return verify_instance_parts(inst, parts).valid


def ilp_partition(
    inst: Instance,
    *,
    time_limit: float | None = None,
) -> tuple[str, list[list[int]] | None]:
    """Exact MILP oracle for Győri–Lovász partitions (docs/paper_notes.md §1.2/§1.3).

    Solves the flow formulation of the module docstring with HiGHS through
    :func:`scipy.optimize.milp`.  Returns ``(status, parts)`` with ``status``
    in ``{"ok", "infeasible", "timeout"}``; any other solver outcome raises
    :class:`RuntimeError`.  No connectivity precondition is assumed, so
    ``"infeasible"`` is a proof of non-existence.
    """
    idx, c_obj, bounds, integrality, cons = build_model(inst)
    options: dict[str, object] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    res = milp(
        c_obj,
        constraints=[cons] if cons is not None else None,
        integrality=integrality,
        bounds=bounds,
        options=options,
    )
    if res.status == 0:
        return "ok", idx.decode(res.x, inst.terminals)
    if res.status == 2:
        # An "infeasible" verdict from this oracle is used as *proof* that no
        # partition exists, so never propagate the solver's word for it without
        # a second opinion: the HiGHS shipped with scipy 1.15.3 reports
        # Infeasible on models that are demonstrably feasible (reproduced on an
        # 8-vertex directed instance whose partition brute force finds, and
        # which the same HiGHS solves to optimality once presolve is off; newer
        # scipy solves it either way).  Re-solve once without presolve and only
        # believe the verdict if it survives.
        confirm = dict(options)
        confirm["presolve"] = False
        res2 = milp(
            c_obj,
            constraints=[cons] if cons is not None else None,
            integrality=integrality,
            bounds=bounds,
            options=confirm,
        )
        if res2.status == 0 and res2.x is not None:
            parts = idx.decode(res2.x, inst.terminals)
            if _is_valid(inst, parts):
                return "ok", parts
        if res2.status == 1:
            if res2.x is not None:
                parts = idx.decode(res2.x, inst.terminals)
                if _is_valid(inst, parts):
                    return "ok", parts
            return "timeout", None
        return "infeasible", None
    if res.status == 1:
        # A feasible incumbent found before the limit is still a valid answer
        # (the objective is constant); otherwise report the timeout.
        if res.x is not None:
            parts = idx.decode(res.x, inst.terminals)
            if _is_valid(inst, parts):
                return "ok", parts
        return "timeout", None
    raise RuntimeError(f"milp failed with status {res.status}: {res.message}")


__all__ = ["VarIndex", "build_model", "ilp_partition"]
