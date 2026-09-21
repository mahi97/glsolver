// Header-only helpers shared by GLSolver (solver_general.cpp) and GLWeightedSolver (solver_weighted.cpp):
//   * tiny JSON builders for the trace payloads of docs/paper_notes.md §14 (objects, integer-keyed maps,
//     lists of ints / pairs / triples, string escaping);
//   * TermSet <-> terminal-vertex conversions (bitsets are over ORIGINAL terminal indices, O9);
//   * a small explicit Dinic max-flow and, on top of it, the saturating assignment search of paper_notes §5
//     (proof of [Thm ess-assign-cond]): non-terminal -> terminal slots, exactly the FEAC witness flow.
// Everything here is generic (no solver state) and deterministic (adjacency order decides ties).
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "bitset.hpp"

namespace glcore {

// ----------------------------------------------------------------------------------------------- JSON

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

inline std::string json_string(const std::string& s) { return "\"" + json_escape(s) + "\""; }

inline std::string json_bool(bool b) { return b ? "true" : "false"; }

// [1,2,3] for any integral element type.
template <class T>
std::string json_list(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(v[i]);
    }
    s += ']';
    return s;
}

// [[u,v],...]
inline std::string json_pairs(const std::vector<std::pair<int, int>>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += "[" + std::to_string(v[i].first) + "," + std::to_string(v[i].second) + "]";
    }
    s += ']';
    return s;
}

// [[a,b,c],...]
inline std::string json_triples(const std::vector<std::array<int, 3>>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += "[" + std::to_string(v[i][0]) + "," + std::to_string(v[i][1]) + "," + std::to_string(v[i][2]) + "]";
    }
    s += ']';
    return s;
}

// Incremental JSON object: {"key":value,...}. Values passed to raw() must already be JSON text.
class JsonObject {
public:
    JsonObject& raw(const std::string& key, const std::string& value_json) {
        if (!body_.empty()) body_ += ',';
        body_ += json_string(key) + ":" + value_json;
        return *this;
    }
    JsonObject& num(const std::string& key, int64_t v) { return raw(key, std::to_string(v)); }
    JsonObject& real(const std::string& key, double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.17g", v);
        return raw(key, buf);
    }
    JsonObject& str(const std::string& key, const std::string& v) { return raw(key, json_string(v)); }
    JsonObject& boolean(const std::string& key, bool v) { return raw(key, json_bool(v)); }
    JsonObject& null(const std::string& key) { return raw(key, "null"); }
    std::string build() const { return "{" + body_ + "}"; }
    bool empty() const { return body_.empty(); }
private:
    std::string body_;
};

// JSON object whose keys are integers rendered as strings ({"3":...}): the per-vertex / per-terminal maps
// of §14 ("ess", "kappa", "phi", "capacities", "parts", "cuts", "parents").
class JsonIntMap {
public:
    JsonIntMap& add(int64_t key, const std::string& value_json) {
        if (!body_.empty()) body_ += ',';
        body_ += "\"" + std::to_string(key) + "\":" + value_json;
        return *this;
    }
    std::string build() const { return "{" + body_ + "}"; }
private:
    std::string body_;
};

// ----------------------------------------------------------------------------------------------- TermSet

// Original terminal indices set in s -> the terminal vertices, in index order (term_vertex[i] = t_i).
inline std::vector<int> termset_to_vertices(const TermSet& s, const std::vector<int>& term_vertex) {
    std::vector<int> r;
    const int k = std::min<int>(s.k, (int)term_vertex.size());
    for (int i = 0; i < k; ++i)
        if (s.test(i)) r.push_back(term_vertex[i]);
    return r;
}

// JSON list of the terminal vertices of s (sorted by original index).
inline std::string termset_json(const TermSet& s, const std::vector<int>& term_vertex) {
    return json_list(termset_to_vertices(s, term_vertex));
}

inline TermSet termset_from_indices(int k, const std::vector<int>& indices) {
    TermSet s(k);
    for (int i : indices)
        if (i >= 0 && i < k) s.set(i);
    return s;
}

// ----------------------------------------------------------------------------------------------- Dinic

// Explicit-network Dinic max-flow for the small auxiliary networks of the solvers (witness search, O(nk)
// arcs). Iterative DFS (no recursion), deterministic (adjacency order), int64 capacities.
class Dinic {
public:
    explicit Dinic(int n) : n_(n), adj_(n), level_(n, -1), it_(n, 0) {}

    int num_nodes() const { return n_; }

    // Returns the id of the forward arc; flow_on(id) after max_flow.
    int add_arc(int u, int v, int64_t cap) {
        if (u < 0 || u >= n_ || v < 0 || v >= n_) throw std::out_of_range("Dinic::add_arc: node out of range");
        const int id = (int)edges_.size();
        adj_[u].push_back(id);
        edges_.push_back(E{v, cap});
        adj_[v].push_back(id + 1);
        edges_.push_back(E{u, 0});
        return id;
    }

    int64_t flow_on(int arc) const { return edges_[(size_t)arc ^ 1].cap; }

    int64_t max_flow(int s, int t) {
        int64_t total = 0;
        while (bfs(s, t)) {
            std::fill(it_.begin(), it_.end(), 0);
            for (;;) {
                const int64_t f = augment(s, t);
                if (f == 0) break;
                total += f;
            }
        }
        return total;
    }

private:
    struct E {
        int to;
        int64_t cap;
    };
    int n_;
    std::vector<E> edges_;
    std::vector<std::vector<int>> adj_;
    std::vector<int> level_, it_;
    std::vector<int> queue_, stack_nodes_, stack_arcs_;

    bool bfs(int s, int t) {
        std::fill(level_.begin(), level_.end(), -1);
        queue_.clear();
        queue_.push_back(s);
        level_[s] = 0;
        for (size_t h = 0; h < queue_.size(); ++h) {
            const int u = queue_[h];
            for (int e : adj_[u]) {
                const int v = edges_[e].to;
                if (edges_[e].cap > 0 && level_[v] < 0) {
                    level_[v] = level_[u] + 1;
                    queue_.push_back(v);
                }
            }
        }
        return level_[t] >= 0;
    }

    // One augmenting path in the level graph (iterative DFS with the usual current-arc pointers).
    int64_t augment(int s, int t) {
        stack_nodes_.clear();
        stack_arcs_.clear();
        stack_nodes_.push_back(s);
        while (!stack_nodes_.empty()) {
            const int u = stack_nodes_.back();
            if (u == t) {
                int64_t f = std::numeric_limits<int64_t>::max();
                for (int e : stack_arcs_) f = std::min(f, edges_[e].cap);
                for (int e : stack_arcs_) {
                    edges_[e].cap -= f;
                    edges_[(size_t)e ^ 1].cap += f;
                }
                return f;
            }
            bool advanced = false;
            for (; it_[u] < (int)adj_[u].size(); ++it_[u]) {
                const int e = adj_[u][it_[u]];
                const int v = edges_[e].to;
                if (edges_[e].cap > 0 && level_[v] == level_[u] + 1) {
                    stack_nodes_.push_back(v);
                    stack_arcs_.push_back(e);
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                level_[u] = -1;  // dead end for this phase
                stack_nodes_.pop_back();
                if (!stack_arcs_.empty()) stack_arcs_.pop_back();
                if (!stack_nodes_.empty()) ++it_[stack_nodes_.back()];
            }
        }
        return 0;
    }
};

// ----------------------------------------------------------------------------------------------- witness search

struct AssignmentResult {
    bool saturated = false;          // every left vertex received a slot
    int64_t value = 0;               // number of left vertices assigned
    std::vector<int> assignment;     // per left vertex: terminal index (only meaningful when saturated)
};

// Saturating assignment of left vertices to terminal slots (paper_notes §5, proof of [Thm ess-assign-cond]):
// source -> i (cap 1), i -> t (cap 1) for t in allowed[i], t -> sink (cap capacity[t]). `saturated` iff the
// max-flow value equals the number of left vertices; then every terminal t receives exactly capacity[t]
// vertices whenever Σ capacity == |left| (all sink arcs saturated). Deterministic.
inline AssignmentResult saturating_assignment(const std::vector<std::vector<int>>& allowed,
                                              const std::vector<int64_t>& capacity) {
    const int L = (int)allowed.size();
    const int K = (int)capacity.size();
    const int s = 0, z = 1 + L + K;
    Dinic net(z + 1);
    std::vector<std::vector<std::pair<int, int>>> arcs(L);  // (terminal index, arc id)
    for (int i = 0; i < L; ++i) {
        net.add_arc(s, 1 + i, 1);
        for (int t : allowed[i]) {
            if (t < 0 || t >= K) throw std::out_of_range("saturating_assignment: terminal index out of range");
            arcs[i].emplace_back(t, net.add_arc(1 + i, 1 + L + t, 1));
        }
    }
    for (int t = 0; t < K; ++t)
        if (capacity[t] > 0) net.add_arc(1 + L + t, z, capacity[t]);
    AssignmentResult r;
    r.value = net.max_flow(s, z);
    r.saturated = (r.value == L);
    r.assignment.assign(L, -1);
    for (int i = 0; i < L; ++i)
        for (const auto& ta : arcs[i])
            if (net.flow_on(ta.second) > 0) { r.assignment[i] = ta.first; break; }
    return r;
}

}  // namespace glcore
