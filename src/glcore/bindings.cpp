// pybind11 bindings for the glcore C++ engine. Extended in later stages.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
}
