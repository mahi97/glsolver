# Changelog

## 0.1.0 (2026-09-21)

* Reference (pure Python) implementation of the polynomial-time Győri–Lovász
  algorithm of arXiv 2608.30945: unweighted GLPartition/ShiftAssignment,
  weighted GLWeightedPartition with RoundAndRemove and min-cost split
  assignments, and the near-linear DAG algorithm.
* Independent verifier, brute-force and MILP oracles, seeded graph-family
  generators, official compact-connectivity counterexample reproduction.
* C++20 core (pybind11) with incremental flow certificates, greedy contraction
  and lazy ShiftAssignment; O(n+m) DAG variant.
* Python API (`glpartition`, `partition`), `glsolve` CLI, visualization,
  benchmark suite and dashboard, documentation set.
