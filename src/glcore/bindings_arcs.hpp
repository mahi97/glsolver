// The `arcs` argument of the solver / DAG / matching bindings (RESEARCH_NOTES.md E4, numpy fast path).
//
// Accepted forms:
//   * a numpy integer array of shape (m, 2): int32 C-contiguous arrays are read in place (no copy, no
//     per-element Python object); other integer dtypes / layouts are converted by numpy once and every id is
//     range-checked before narrowing to int. This is what glsolver.api passes (Instance.arc_array).
//   * any sequence of (u, v) integer pairs (the pybind11 STL caster, semantics unchanged: TypeError for a
//     wrong element type, Python ints outside the C int range rejected).
// Type / shape errors are Python TypeError / ValueError exactly like a wrong argument type used to be;
// *value* errors (ids out of 0..n-1) are left to Graph's constructor, which reports them as status "error".
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace glcore {

using ArcList = std::vector<std::pair<int, int>>;

inline ArcList arcs_from_object(const pybind11::handle& obj, const char* where) {
    namespace py = pybind11;
    ArcList arcs;
    if (py::isinstance<py::array>(obj)) {
        const py::array arr = py::reinterpret_borrow<py::array>(obj);
        const char kind = arr.dtype().kind();
        if (kind != 'i' && kind != 'u')
            throw py::type_error(std::string(where) + "(): arcs array must have an integer dtype, got " +
                                 py::str(arr.dtype()).cast<std::string>());
        if (arr.size() == 0) return arcs;  // any empty integer array (shape (0,) or (0, 2)) is the empty arc list
        if (arr.ndim() != 2 || arr.shape(1) != 2)
            throw py::value_error(std::string(where) + "(): arcs array must have shape (m, 2)");
        const py::ssize_t m = arr.shape(0);
        arcs.resize((size_t)m);
        if (arr.dtype().is(py::dtype::of<int32_t>())) {
            // zero-copy when already C-contiguous int32 (the canonical Instance storage)
            const py::array_t<int32_t, py::array::c_style | py::array::forcecast> a(arr);
            const auto r = a.unchecked<2>();
            for (py::ssize_t i = 0; i < m; ++i) arcs[(size_t)i] = {r(i, 0), r(i, 1)};
            return arcs;
        }
        const py::array_t<int64_t, py::array::c_style | py::array::forcecast> a(arr);
        const auto r = a.unchecked<2>();
        constexpr int64_t lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
        for (py::ssize_t i = 0; i < m; ++i) {
            const int64_t u = r(i, 0), v = r(i, 1);
            if (u < lo || u > hi || v < lo || v > hi)
                throw std::invalid_argument(std::string(where) + ": arc (" + std::to_string(u) + "," + std::to_string(v) +
                                            ") does not fit in a C int");
            arcs[(size_t)i] = {(int)u, (int)v};
        }
        return arcs;
    }
    try {
        return obj.cast<ArcList>();
    } catch (const py::cast_error&) {
        throw py::type_error(std::string(where) + "(): incompatible 'arcs' argument: expected a sequence of (u, v) "
                             "integer pairs or a numpy integer array of shape (m, 2), got " +
                             py::str(py::type::handle_of(obj)).cast<std::string>());
    }
}

}  // namespace glcore
