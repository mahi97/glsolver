// pybind11 binding of the unweighted general solver: _core.solve_general(n, arcs, terminals, capacities,
// options) -> dict, the contract of src/glsolver/api.py (_run_core). run() executes with the GIL released;
// every std::exception raised by the core (invalid input, malformed options, an invariant violation reported
// as std::logic_error) is returned as status "error" with its message instead of propagating, so the Python
// layer can report it and tests can assert on it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"
#include "bindings_arcs.hpp"
#include "solver.hpp"
#include "stats.hpp"
#include "trace.hpp"

namespace py = pybind11;

namespace glcore {
namespace {

// Stats -> dict (docs/api.md `stats` field; same layout as the DAG binding).
py::dict solver_stats_to_dict(const Stats& s) {
    py::dict d;
    d["max_flow_calls"] = s.max_flow_calls;
    d["augment_calls"] = s.augment_calls;
    d["cut_calls"] = s.cut_calls;
    d["matching_calls"] = s.matching_calls;
    d["min_cost_flow_calls"] = s.min_cost_flow_calls;
    d["contractions"] = s.contractions;
    d["deletions"] = s.deletions;
    d["batched_deletions"] = s.batched_deletions;
    d["cycle_shifts"] = s.cycle_shifts;
    d["terminal_removals"] = s.terminal_removals;
    d["roundings"] = s.roundings;
    d["greedy_attempts"] = s.greedy_attempts;
    d["greedy_successes"] = s.greedy_successes;
    d["shift_calls"] = s.shift_calls;
    d["assignment_repairs"] = s.assignment_repairs;
    d["steps"] = s.steps;
    py::dict times;
    for (const auto& kv : s.time_seconds) times[py::str(kv.first)] = kv.second;
    d["time_seconds"] = times;
    py::list sizes;
    for (const auto& pr : s.graph_size_over_time) sizes.append(py::make_tuple(pr.first, pr.second));
    d["graph_size_over_time"] = sizes;
    return d;
}

py::list trace_to_list(const Trace& tr) {
    py::list events;
    for (const auto& e : tr.events) events.append(py::make_tuple(e.first, e.second));
    return events;
}

// Python type name of an object, for option error messages ("str", "NoneType", ...).
std::string option_type_name(const py::handle& obj) {
    try {
        return py::str(py::type::handle_of(obj).attr("__name__")).cast<std::string>();
    } catch (const std::exception&) {
        return "?";
    }
}

// Malformed option values (wrong type, out of range) are reported as std::invalid_argument naming the key, so
// that the caller can return them as status "error" like every other invalid input (never a Python exception).
SolverOptions parse_options(const py::dict& options) {
    SolverOptions o;
    auto get_int = [&](const char* key, int def) -> int {
        if (!options.contains(key)) return def;
        const py::object v = options[py::str(key)];
        try {
            return v.cast<int>();
        } catch (const py::cast_error&) {
            throw std::invalid_argument(std::string("solve_general: option '") + key +
                                        (py::isinstance<py::int_>(v) ? "' is out of the range of a C int" : "' must be an int (got " + option_type_name(v) + ")"));
        }
    };
    auto get_bool = [&](const char* key, bool def) -> bool {
        if (!options.contains(key)) return def;
        const py::object v = options[py::str(key)];
        try {
            return v.cast<bool>();
        } catch (const py::cast_error&) {
            throw std::invalid_argument(std::string("solve_general: option '") + key + "' must be a bool (got " + option_type_name(v) + ")");
        }
    };
    o.threads = get_int("threads", 0);
    o.seed = get_int("seed", 0);
    o.greedy_contraction = get_bool("greedy_contraction", true);
    o.lazy_shift = get_bool("lazy_shift", true);
    o.batch_unused_arcs = get_bool("batch_unused_arcs", true);
    o.debug_asserts = get_bool("debug_asserts", false);
    o.trace = get_bool("trace", false);
    o.record_cuts = get_bool("record_cuts", false);
    o.check_precondition = get_bool("check_precondition", true);
    return o;
}

}  // namespace

void bind_solver(py::module_& m) {
    m.def(
        "solve_general",
        [](int n, const py::object& arcs_obj, const std::vector<int>& terminals, const std::vector<int64_t>& capacities,
           const py::dict& options) {
            py::dict d;
            d["status"] = "error";
            d["message"] = "";
            d["assignment"] = py::list();
            d["parent"] = py::list();
            d["witness"] = py::list();
            d["k_T_connected"] = false;
            d["stats"] = solver_stats_to_dict(Stats{});
            d["trace"] = py::list();
            std::unique_ptr<GLSolver> solver;
            ArcList arcs;
            try {
                arcs = arcs_from_object(arcs_obj, "solve_general");  // numpy (m, 2) int array or (u, v) pairs
                const SolverOptions opt = parse_options(options);  // malformed options -> status "error" too
                solver = std::make_unique<GLSolver>(n, arcs, terminals, capacities, opt);
            } catch (const py::builtin_exception&) {
                throw;  // TypeError / ValueError for a wrong arcs type or shape, as for any wrong argument type
            } catch (const std::exception& e) {
                d["message"] = std::string(e.what());
                return d;
            }
            SolveResult r;
            std::string error;
            {
                py::gil_scoped_release release;
                try {
                    r = solver->run();
                } catch (const std::exception& e) {
                    error = e.what();
                    if (error.empty()) error = "unknown error";
                }
            }
            d["stats"] = solver_stats_to_dict(solver->stats());
            d["trace"] = trace_to_list(solver->trace());
            if (!error.empty()) {
                d["status"] = "error";
                d["message"] = error;
                return d;
            }
            d["status"] = r.status;
            d["message"] = r.message;
            d["assignment"] = r.assignment;
            d["parent"] = r.parent;
            d["witness"] = r.witness;
            d["k_T_connected"] = r.k_T_connected;
            return d;
        },
        py::arg("n"), py::arg("arcs"), py::arg("terminals"), py::arg("capacities"), py::arg("options") = py::dict(),
        "solve_general(n, arcs, terminals, capacities, options={}) -> dict(status, message, assignment, parent, witness, "
        "k_T_connected, stats, trace). arcs: a numpy integer array of shape (m, 2) (int32 read in place) or a sequence "
        "of (u, v) pairs. GLPartition [Alg 1] + ShiftAssignment [Alg 2] with the exact optimizations O1–O9 "
        "(docs/optimizations.md). options keys: threads, seed, greedy_contraction, lazy_shift, batch_unused_arcs, "
        "debug_asserts, trace, record_cuts, check_precondition. status is 'ok', 'precondition_failed' (FEAC fails; a "
        "partition may still exist) or 'error' (the C++ exception message: invalid input, a malformed option value or an "
        "invariant violation; never raised). On a non-ok status assignment, parent and witness are empty lists.");
}

}  // namespace glcore
