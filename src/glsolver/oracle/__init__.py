"""Exact oracles: brute-force backtracking and MILP (HiGHS).

Both oracles are correctness baselines for the polynomial-time solvers.
They assume no connectivity precondition and may return ``"infeasible"``
(a proof of non-existence), which the theorem-based solvers never do
(docs/api.md, docs/paper_notes.md §13.9).
"""
from glsolver.oracle.bruteforce import bruteforce_partition, enumerate_partitions
from glsolver.oracle.ilp import ilp_partition

__all__ = ["bruteforce_partition", "enumerate_partitions", "ilp_partition"]
