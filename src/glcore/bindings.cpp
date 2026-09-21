// pybind11 module entry point for the glcore C++ engine.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "glcore: high-performance C++ core for Győri–Lovász partitioning";
    m.attr("__version__") = "0.1.0";
    m.attr("debug_asserts") =
#ifdef GLCORE_DEBUG_ASSERTS
        true;
#else
        false;
#endif
    glcore::bind_graph_flow(m);
    glcore::bind_matching_dag(m);
    glcore::bind_solver(m);
    glcore::bind_solver_weighted(m);
}
