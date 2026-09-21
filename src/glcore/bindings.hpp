// Per-module pybind11 binders; each is defined in its own bindings_*.cpp so parallel work does not collide.
#pragma once
#include <pybind11/pybind11.h>

namespace glcore {
void bind_graph_flow(pybind11::module_& m);   // bindings_flow.cpp: Graph, FlowEngine, EssentialOracle
void bind_matching_dag(pybind11::module_& m); // bindings_dag.cpp: matching, min-cost flow, dag_partition
void bind_solver(pybind11::module_& m);       // bindings_solver.cpp: GLSolver (solve_general)
void bind_solver_weighted(pybind11::module_& m); // bindings_solver_weighted.cpp: GLWeightedSolver (solve_weighted)
}  // namespace glcore
