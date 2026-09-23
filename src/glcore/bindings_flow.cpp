// pybind11 bindings for Graph, the tightest-cut primitive and EssentialOracle (test/diagnostic surface).
// Everything is copied out to plain Python containers; heavy calls release the GIL and never touch Python
// objects inside the C++ parallel regions.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"
#include "essential.hpp"
#include "flow.hpp"
#include "graph.hpp"
#include "stats.hpp"

namespace py = pybind11;

namespace glcore {

namespace {

// The inline Graph accessors index vectors without a range check (the C++ callers are trusted); ids coming
// from Python are validated here so that a bad id raises IndexError instead of reading out of bounds.
int vertex_or_throw(const Graph& g, int v) {
    if (v < 0 || v >= g.n())
        throw std::out_of_range("vertex " + std::to_string(v) + " out of range [0, " + std::to_string(g.n()) + ")");
    return v;
}

int arc_index_or_throw(const Graph& g, int a) {
    if (a < 0 || a >= g.num_arc_ids())
        throw std::out_of_range("arc id " + std::to_string(a) + " out of range [0, " + std::to_string(g.num_arc_ids()) + ")");
    return a;
}

int arc_id_or_throw(const Graph& g, int u, int v) {
    int a = g.find_arc(u, v);
    if (a < 0)
        throw std::invalid_argument("arc (" + std::to_string(u) + "," + std::to_string(v) + ") is not present");
    return a;
}

std::vector<int> arc_ids_or_throw(const Graph& g, const std::vector<std::pair<int, int>>& arcs) {
    std::vector<int> ids;
    ids.reserve(arcs.size());
    for (const auto& uv : arcs) ids.push_back(arc_id_or_throw(g, uv.first, uv.second));
    return ids;
}

// paths as vertex lists (source first, terminal last)
py::list paths_as_vertices(const Graph& g, const VertexFlow& f) {
    py::list out;
    for (const auto& path : f.paths) {
        py::list p;
        p.append(f.v);
        for (int a : path) p.append(g.arc(a).head);
        out.append(std::move(p));
    }
    return out;
}

// side[] is materialized on demand (FlowEngine::compute_cut with_sides / EssentialOracle::sides); empty = no cut
py::dict sides_dict(const Graph& g, const std::vector<Side>& side) {
    py::dict d;
    if (side.size() < (size_t)g.n()) return d;
    for (int x = 0; x < g.n(); ++x) {
        if (!g.live(x)) continue;
        const char* s = side[x] == Side::L ? "L" : (side[x] == Side::S ? "S" : "R");
        d[py::int_(x)] = py::str(s);
    }
    return d;
}

py::dict stats_dict(const Stats& s) {
    py::dict d;
    d["max_flow_calls"] = s.max_flow_calls;
    d["augment_calls"] = s.augment_calls;
    d["cut_calls"] = s.cut_calls;
    d["contractions"] = s.contractions;
    d["deletions"] = s.deletions;
    d["batched_deletions"] = s.batched_deletions;
    d["terminal_removals"] = s.terminal_removals;
    py::dict t;
    for (const auto& kv : s.time_seconds) t[py::str(kv.first)] = kv.second;
    d["time_seconds"] = t;
    return d;
}

// Python-facing oracle: owns its Stats and remembers the last evaluate_deletion so that commit_deletion can
// adopt the computed flows after performing the graph deletions itself.
struct PyEssentialOracle {
    Graph& g;
    Stats stats;
    EssentialOracle oracle;
    std::vector<int> pending_D;
    std::vector<int> pending_affected;
    std::vector<VertexFlow> pending_out;
    bool has_pending = false;
    // C1 penalties, owned here: FlowEngine keeps a raw pointer to the caller's array, so the array must
    // outlive every search. Python hands in a copy, which lives exactly as long as this oracle.
    std::vector<unsigned char> penalties;

    PyEssentialOracle(Graph& g_, int threads, int seed) : g(g_), oracle(g_, stats, threads, seed) {}

    void set_penalties(const std::vector<unsigned char>* pen) {
        if (pen == nullptr) {
            oracle.set_penalties(nullptr);
            penalties.clear();
            return;
        }
        penalties = *pen;
        oracle.set_penalties(&penalties);
    }

    std::vector<int> evaluate(const std::vector<int>& D, bool exact_cuts) {
        pending_D = D;
        pending_affected = oracle.evaluate_deletion(D, pending_out, exact_cuts);
        has_pending = true;
        return pending_affected;
    }

    void commit(const std::vector<int>& D) {
        std::vector<int> sd(D), sp(pending_D);
        std::sort(sd.begin(), sd.end());
        std::sort(sp.begin(), sp.end());
        if (!has_pending || sd != sp) evaluate(D, false);
        for (int a : D)
            if (g.arc(a).alive) g.delete_arc(a);
        oracle.commit_deletion(pending_D, pending_affected, pending_out);
        has_pending = false;
        pending_D.clear();
        pending_affected.clear();
    }
};

}  // namespace

void bind_graph_flow(py::module_& m) {
    py::class_<Graph>(m, "Graph", "Mutable simple digraph with terminals (paper_notes §2, [Def 2.1], §13.4)")
        .def(py::init<int, const std::vector<std::pair<int, int>>&, const std::vector<int>&>(),
             py::arg("n"), py::arg("arcs"), py::arg("terminals"))
        .def_property_readonly("n", &Graph::n)
        .def_property_readonly("k", &Graph::k)
        .def_property_readonly("k0", &Graph::k0)
        .def_property_readonly("terminals", [](const Graph& g) { return g.terminals(); })
        .def_property_readonly("version", &Graph::version)
        .def("terminal_index", [](const Graph& g, int v) { return g.terminal_index(vertex_or_throw(g, v)); })
        .def("is_terminal", [](const Graph& g, int v) { return g.is_terminal(vertex_or_throw(g, v)); })
        .def("live", [](const Graph& g, int v) { return g.live(vertex_or_throw(g, v)); })
        .def("out_degree", [](const Graph& g, int v) { return g.out_degree(vertex_or_throw(g, v)); })
        .def("in_degree", [](const Graph& g, int v) { return g.in_degree(vertex_or_throw(g, v)); })
        .def("out_neighbors", [](const Graph& g, int v) {
            std::vector<int> r;
            for (int a : g.out_arcs(vertex_or_throw(g, v))) r.push_back(g.arc(a).head);
            return r;
        })
        .def("in_neighbors", [](const Graph& g, int v) {
            std::vector<int> r;
            for (int a : g.in_arcs(vertex_or_throw(g, v))) r.push_back(g.arc(a).tail);
            return r;
        })
        .def("has_arc", &Graph::has_arc)
        .def("find_arc", &Graph::find_arc)
        .def("orig_head", [](const Graph& g, int u, int v) { return g.arc(arc_id_or_throw(g, u, v)).orig_head; })
        .def("delete_arc", [](Graph& g, int u, int v) { g.delete_arc(arc_id_or_throw(g, u, v)); })
        .def("delete_arc_id", [](Graph& g, int a) { g.delete_arc(a); })
        .def("contract", [](Graph& g, int p, int t) {
            std::vector<int> created, deleted;
            int parent = g.contract(p, t, &created, &deleted);
            return parent;
        })
        .def("contract_detailed", [](Graph& g, int p, int t) {
            std::vector<int> created, deleted;
            int parent = g.contract(p, t, &created, &deleted);
            return py::make_tuple(parent, created, deleted);
        })
        .def("remove_terminal", &Graph::remove_terminal)
        .def("remove_vertex", &Graph::remove_vertex)
        .def("is_pre_terminal", &Graph::is_pre_terminal)
        .def("pre_terminals", &Graph::pre_terminals)
        .def("pre_terminals_of", &Graph::pre_terminals_of)
        .def("live_nonterminals", &Graph::live_nonterminals)
        .def("num_live_nonterminals", &Graph::num_live_nonterminals)
        .def("num_live_arcs", &Graph::num_live_arcs)
        .def("num_arc_ids", &Graph::num_arc_ids)
        .def("arcs", [](const Graph& g) {
            std::vector<std::pair<int, int>> r;
            for (int a = 0; a < g.num_arc_ids(); ++a)
                if (g.arc(a).alive) r.emplace_back(g.arc(a).tail, g.arc(a).head);
            return r;
        })
        .def("arc", [](const Graph& g, int a) {
            const Arc& e = g.arc(arc_index_or_throw(g, a));
            return py::make_tuple(e.tail, e.head, e.alive, e.orig_head);
        })
        .def("check_invariants", &Graph::check_invariants);

    // [Prop 4.2]: (kappa, sides {vertex: 'L'|'S'|'R'}, ess [original terminal indices], paths [vertex lists])
    // `penalties` (C1 deletion-aware routing, RESEARCH_NOTES P4/E5) is an optional per-arc byte array
    // (index = arc id, values 0/1, a short array reads as 0 beyond its end): with it the augmenting searches
    // return paths of minimum total penalty instead of BFS-shortest ones. kappa, the sides and Ess must come
    // out identical — that is P4, and this argument is what lets a test check it path by path.
    m.def("tightest_cut", [](const Graph& g, int v, std::optional<std::vector<unsigned char>> penalties) {
        VertexFlow f;
        {
            py::gil_scoped_release release;
            FlowEngine engine(g);
            if (penalties) engine.set_penalties(&*penalties);
            auto s = engine.make_scratch();
            engine.compute_max_flow(v, f, *s, true, true);
        }
        return py::make_tuple(f.kappa, sides_dict(g, f.side), f.ess.to_list(), paths_as_vertices(g, f));
    }, py::arg("graph"), py::arg("v"), py::arg("penalties") = py::none());

    py::class_<PyEssentialOracle>(m, "EssentialOracle",
                                  "Per-vertex flows / essential sets with the warm-start hooks of docs/optimizations.md")
        .def(py::init<Graph&, int, int>(), py::arg("graph"), py::arg("threads") = 1, py::arg("seed") = 0,
             py::keep_alive<1, 2>())
        .def("compute_all", [](PyEssentialOracle& o) {
            py::gil_scoped_release release;
            o.oracle.compute_all();
        })
        .def("set_penalties", [](PyEssentialOracle& o, std::optional<std::vector<unsigned char>> pen) {
            // C1: None restores the plain BFS. The array is copied into the oracle (the engine holds a raw
            // pointer to it), so it stays alive for every later search; it is read by the searches only, so
            // changing it between calls is what the solvers do too.
            if (pen && pen->size() > (size_t)o.g.num_arc_ids())
                throw std::invalid_argument("penalties: " + std::to_string(pen->size()) + " entries for " +
                                            std::to_string(o.g.num_arc_ids()) + " arc ids");
            o.set_penalties(pen ? &*pen : nullptr);
        }, py::arg("penalties"))
        .def("kappa", [](PyEssentialOracle& o, int v) { return o.oracle.kappa(v); })
        .def("ess", [](PyEssentialOracle& o, int v) { return o.oracle.ess(v).to_list(); })
        .def("cut_exact", [](PyEssentialOracle& o, int v) { return o.oracle.flow(v).cut_exact; })
        .def("sides", [](PyEssentialOracle& o, int v) {
            // {} unless the stored cut is exact; the sides are then computed by one reverse BFS (never stored)
            if (!o.oracle.flow(v).cut_exact) return py::dict();
            std::vector<Side> side;
            o.oracle.sides(v, side);
            return sides_dict(o.g, side);
        })
        .def("paths", [](PyEssentialOracle& o, int v) { return paths_as_vertices(o.g, o.oracle.flow(v)); })
        .def("path_arcs", [](PyEssentialOracle& o, int v) { return o.oracle.flow(v).paths; })
        .def("flow_version", [](PyEssentialOracle& o, int v) { return o.oracle.flow(v).version; })
        .def("users_of_arc", [](PyEssentialOracle& o, int u, int v) {
            return o.oracle.users_of_arc(arc_id_or_throw(o.g, u, v));
        })
        .def("evaluate_deletion", [](PyEssentialOracle& o, const std::vector<std::pair<int, int>>& arcs, bool exact_cuts) {
            std::vector<int> D = arc_ids_or_throw(o.g, arcs);
            std::vector<int> affected;
            {
                py::gil_scoped_release release;
                affected = o.evaluate(D, exact_cuts);
            }
            py::dict d;
            for (int v : affected) {
                const VertexFlow& f = o.pending_out[v];
                d[py::int_(v)] = py::make_tuple(f.kappa, f.ess.to_list());
            }
            return d;
        }, py::arg("arcs"), py::arg("exact_cuts") = false)
        .def("evaluated_paths", [](PyEssentialOracle& o, int v) {
            if (!o.has_pending || (size_t)v >= o.pending_out.size() || o.pending_out[v].v != v)
                throw std::invalid_argument("no pending evaluation for vertex " + std::to_string(v));
            return paths_as_vertices(o.g, o.pending_out[v]);
        })
        .def("commit_deletion", [](PyEssentialOracle& o, const std::vector<std::pair<int, int>>& arcs) {
            std::vector<int> D = arc_ids_or_throw(o.g, arcs);
            py::gil_scoped_release release;
            o.commit(D);
        })
        .def("after_contraction", [](PyEssentialOracle& o, int p, int t) { o.oracle.after_contraction(p, t); })
        .def("after_terminal_removal", [](PyEssentialOracle& o, int t) {
            py::gil_scoped_release release;
            o.oracle.after_terminal_removal(t);
        })
        .def("after_vertex_removal", [](PyEssentialOracle& o, int v) { o.oracle.after_vertex_removal(v); })
        .def("refresh_cut", [](PyEssentialOracle& o, int v) { o.oracle.refresh_cut(v); })
        .def("refresh_all_cuts", [](PyEssentialOracle& o) {
            py::gil_scoped_release release;
            o.oracle.refresh_all_cuts();
        })
        .def("stats", [](const PyEssentialOracle& o) { return stats_dict(o.stats); });
}

}  // namespace glcore
