// pybind11 bindings for matching [Lem 7.8], the minimal Hall-deficient set [Lem 7.6], min-cost flow [Prop 5.4]
// and the DAG solver [Alg 5]. All functions take raw (n, arcs, terminals) triples and build a Graph internally,
// so they do not depend on the Graph binding; heavy work runs with the GIL released.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "bindings.hpp"
#include "dag.hpp"
#include "graph.hpp"
#include "matching.hpp"
#include "mincostflow.hpp"
#include "stats.hpp"
#include "trace.hpp"

namespace py = pybind11;

namespace glcore {
namespace {

using ArcList = std::vector<std::pair<int, int>>;

// Stats -> dict (docs/api.md `stats` field).
py::dict stats_to_dict(const Stats& s) {
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

}  // namespace

void bind_matching_dag(py::module_& m) {
    // [Lem 7.8] saturating matching of S (default: all terminals) into distinct pre-terminals.
    m.def(
        "saturating_matching",
        [](int n, const ArcList& arcs, const std::vector<int>& terminals, std::optional<std::vector<int>> S) -> py::object {
            Graph g(n, arcs, terminals);
            const std::vector<int> Sv = S ? *S : g.terminals();
            std::vector<int> res;
            {
                py::gil_scoped_release release;
                res = saturating_matching(g, Sv);
            }
            if (Sv.empty()) return py::list();  // the empty set is trivially saturated
            if (res.empty()) return py::none();
            return py::cast(res);
        },
        py::arg("n"), py::arg("arcs"), py::arg("terminals"), py::arg("S") = py::none(),
        "saturating_matching(n, arcs, terminals, S=None) -> list[int] | None. Matched pre-terminal per position of S "
        "(default: all terminals), or None if no matching saturating S exists [Lem 7.8].");

    // [Lem 7.6] inclusion-minimal Hall-deficient terminal set, or None if T is saturable.
    m.def(
        "minimal_hall_deficient_set",
        [](int n, const ArcList& arcs, const std::vector<int>& terminals) -> py::object {
            Graph g(n, arcs, terminals);
            std::vector<int> res;
            {
                py::gil_scoped_release release;
                res = minimal_hall_deficient_set(g);
            }
            if (res.empty()) return py::none();
            return py::cast(res);
        },
        py::arg("n"), py::arg("arcs"), py::arg("terminals"),
        "minimal_hall_deficient_set(n, arcs, terminals) -> list[int] | None. Inclusion-minimal nonempty S ⊆ T with no "
        "saturating matching (|PT(G,S)| = |S| - 1, [Lem 7.6]); None if T is saturable.");

    // [Prop 5.4] min-cost flow on an explicit network.
    m.def(
        "min_cost_flow",
        [](int num_nodes, const std::vector<std::tuple<int, int, int64_t, int64_t>>& arcs, int s, int z, int64_t required) {
            MinCostFlow net(num_nodes);
            std::vector<int> ids;
            ids.reserve(arcs.size());
            for (const auto& a : arcs) ids.push_back(net.add_arc(std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a)));
            std::pair<int64_t, int64_t> vc;
            std::vector<int64_t> flows(ids.size());
            {
                py::gil_scoped_release release;
                vc = net.solve(s, z, required);
                for (size_t i = 0; i < ids.size(); ++i) flows[i] = net.flow(ids[i]);
            }
            return py::make_tuple(vc.first, vc.second, flows);
        },
        py::arg("num_nodes"), py::arg("arcs"), py::arg("s"), py::arg("z"), py::arg("required"),
        "min_cost_flow(num_nodes, arcs=[(u, v, cap, cost)], s, z, required) -> (value, cost, flows). Successive shortest "
        "paths with Johnson potentials [Prop 5.4]; flows[i] is the flow on arcs[i]. value < required means no flow of "
        "the requested value exists.");

    // [Alg 5] GLDAGPartition.
    m.def(
        "dag_partition",
        [](int n, const ArcList& arcs, const std::vector<int>& terminals, const std::vector<int64_t>& capacities,
           std::optional<std::vector<int64_t>> weights, int policy, int variant, bool check_precondition, bool trace) {
            Stats stats;
            Trace tr;
            tr.enabled = trace;
            const std::vector<int64_t> w = weights ? *weights : std::vector<int64_t>{};
            DagResult r;
            {
                py::gil_scoped_release release;
                r = dag_partition(n, arcs, terminals, capacities, w, policy, variant, stats, tr, check_precondition);
            }
            py::dict d;
            d["status"] = r.status;
            d["message"] = r.message;
            d["assignment"] = r.assignment;
            d["parent"] = r.parent;
            d["stats"] = stats_to_dict(stats);
            py::list events;
            for (const auto& e : tr.events) events.append(py::make_tuple(e.first, e.second));
            d["trace"] = events;
            return d;
        },
        py::arg("n"), py::arg("arcs"), py::arg("terminals"), py::arg("capacities"), py::arg("weights") = py::none(),
        py::arg("policy") = 0, py::arg("variant") = 0, py::arg("check_precondition") = true, py::arg("trace") = false,
        "dag_partition(n, arcs, terminals, capacities, weights=None, policy=0, variant=0, check_precondition=True, "
        "trace=False) -> dict(status, message, assignment, parent, stats, trace). [Alg 5] on a k-T-connected DAG; "
        "policy 0 = max residual, 1 = round robin, 2 = first active; variant 0 = heaps, 1 = linear stack (P1). "
        "trace is a list of (type, json) pairs.");
}

}  // namespace glcore
