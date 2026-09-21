// GLDAGPartition [Alg 5] (docs/paper_notes.md §9) in two exactly equivalent implementations:
//   variant 0: one binary min-heap per terminal keyed by the canonical position [Def 9.2], O(m log n) [Lem 9.6];
//   variant 1: the stack-of-sorted-in-lists variant of RESEARCH_NOTES.md P1, O(n + m) (+ O(n log k) for the
//              max-residual policy).
// Both variants perform the same sequence of contractions and record the same parents (proof in P1), which
// tests/test_core_dag.py asserts against each other and against the reference glref.dag.gl_dag_partition.
#include "dag.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace glcore {
namespace {

template <class T>
std::string vec_json(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(v[i]);
    }
    s += ']';
    return s;
}

// Heap entry of [Alg 5]: (pos[p], p, x) with x ∈ V_i the head of the arc (p, x) that created the entry. The
// lexicographic order is exactly the reference's heapq tuple order, so the first non-stale pop of p carries
// the smallest such x (paper_notes §9: "push pairs (pos[p], p, x); the first non-stale pop wins").
struct HeapEntry {
    int pos, p, x;
    bool operator>(const HeapEntry& o) const { return std::tie(pos, p, x) > std::tie(o.pos, o.p, o.x); }
};
using TermHeap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>>;

// Stack level of RESEARCH_NOTES.md P1: a vertex x ∈ V_i and a cursor into its pos-sorted in-list.
struct Level {
    int x, ptr;
};

// Key of the max-residual policy queue: largest residual first, ties by smallest terminal index (§13.7).
struct ResidualKey {
    int64_t residual;
    int idx;
    bool operator<(const ResidualKey& o) const {  // "less" = lower priority
        if (residual != o.residual) return residual < o.residual;
        return idx > o.idx;
    }
};

DagResult fail(const std::string& msg) { return DagResult{"precondition_failed", msg, {}, {}}; }

}  // namespace

// [Alg 5] GLDAGPartition. See dag.hpp for the parameter contract.
DagResult dag_partition(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                        const std::vector<int64_t>& capacities, const std::vector<int64_t>& weights,
                        int policy, int variant, Stats& stats, Trace& trace, bool check_precondition) {
    Stats::Timer timer(stats, "dag");
    // ------------------------------------------------------------------ input validation (malformed => throw)
    if (n <= 0) throw std::invalid_argument("dag_partition: n must be positive");
    const int k = (int)terminals.size();
    if (k <= 0) throw std::invalid_argument("dag_partition: at least one terminal is required");
    if ((int)capacities.size() != k) throw std::invalid_argument("dag_partition: capacities must have one entry per terminal");
    if (!weights.empty() && (int)weights.size() != n) throw std::invalid_argument("dag_partition: weights must be empty or have length n");
    if (policy < 0 || policy > 2) throw std::invalid_argument("dag_partition: policy must be 0 (max residual), 1 (round robin) or 2 (first)");
    if (variant < 0 || variant > 1) throw std::invalid_argument("dag_partition: variant must be 0 (heap) or 1 (linear stack)");
    std::vector<int> term_index(n, -1);
    for (int i = 0; i < k; ++i) {
        const int t = terminals[i];
        if (t < 0 || t >= n) throw std::invalid_argument("dag_partition: terminal " + std::to_string(t) + " out of range");
        if (term_index[t] >= 0) throw std::invalid_argument("dag_partition: terminal " + std::to_string(t) + " listed twice");
        term_index[t] = i;
        if (capacities[i] < 0) throw std::invalid_argument("dag_partition: capacity of terminal " + std::to_string(t) + " is negative");
    }
    for (int v = 0; v < n; ++v)
        if (!weights.empty() && term_index[v] < 0 && weights[v] < 1)
            throw std::invalid_argument("dag_partition: non-terminal " + std::to_string(v) + " must have weight >= 1");

    // ------------------------------------------------------------------ out-CSR (paper §2 conventions: drop
    // arcs leaving a terminal, self-loops and duplicate arcs; the Python layer normalizes already).
    std::vector<int> out_start(n + 1, 0), out_deg(n, 0);
    for (const auto& a : arcs) {
        const int u = a.first, v = a.second;
        if (u < 0 || u >= n || v < 0 || v >= n)
            throw std::invalid_argument("dag_partition: arc (" + std::to_string(u) + "," + std::to_string(v) + ") out of range");
        if (u == v || term_index[u] >= 0) continue;
        ++out_start[u + 1];
    }
    for (int v = 0; v < n; ++v) out_start[v + 1] += out_start[v];
    std::vector<int> out_list(out_start[n]);
    {
        std::vector<int> fill(out_start.begin(), out_start.end() - 1);
        for (const auto& a : arcs) {
            const int u = a.first, v = a.second;
            if (u == v || term_index[u] >= 0) continue;
            out_list[fill[u]++] = v;
        }
        // duplicate merge with a stamp array: O(n + m), order of first occurrence kept.
        std::vector<int> stamp(n, -1);
        for (int u = 0; u < n; ++u) {
            int w = out_start[u];
            for (int p = out_start[u]; p < out_start[u + 1]; ++p) {
                const int v = out_list[p];
                if (stamp[v] == u) continue;
                stamp[v] = u;
                out_list[w++] = v;
            }
            out_deg[u] = w - out_start[u];
        }
    }
    auto out_begin = [&](int u) { return out_start[u]; };
    auto out_end = [&](int u) { return out_start[u] + out_deg[u]; };

    // ------------------------------------------------------------------ precondition [Lem 9.1]: out-degree >= k
    if (check_precondition) {
        for (int v = 0; v < n; ++v) {
            if (term_index[v] >= 0) continue;
            if (out_deg[v] < k)
                return fail("non-terminal " + std::to_string(v) + " has out-degree " + std::to_string(out_deg[v]) + " < k = " +
                            std::to_string(k) + "; the DAG is not k-T-connected [Lem 9.1]");
        }
    }

    // ------------------------------------------------------------------ canonical topological order [Def 9.2]:
    // Kahn's algorithm on the non-terminals with a min-heap on vertex id (deterministic), then the terminals.
    std::vector<int> indeg(n, 0);
    for (int u = 0; u < n; ++u)
        for (int p = out_begin(u); p < out_end(u); ++p)
            if (term_index[out_list[p]] < 0) ++indeg[out_list[p]];
    std::vector<int> order;
    order.reserve(n);
    {
        std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
        for (int v = 0; v < n; ++v)
            if (term_index[v] < 0 && indeg[v] == 0) ready.push(v);
        while (!ready.empty()) {
            const int v = ready.top();
            ready.pop();
            order.push_back(v);
            for (int p = out_begin(v); p < out_end(v); ++p) {
                const int x = out_list[p];
                if (term_index[x] >= 0) continue;
                if (--indeg[x] == 0) ready.push(x);
            }
        }
    }
    if ((int)order.size() != n - k) {
        for (int v = 0; v < n; ++v)
            if (term_index[v] < 0 && indeg[v] > 0)
                return fail("graph is not acyclic: vertex " + std::to_string(v) + " lies on or behind a directed cycle");
        throw std::logic_error("dag_partition: Kahn's algorithm lost vertices without a cycle");
    }
    for (int i = 0; i < k; ++i) order.push_back(terminals[i]);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[order[i]] = i;

    // ------------------------------------------------------------------ Σ w_v <= Σ c_t (paper_notes §9 precondition)
    int64_t total_w = 0, total_c = 0;
    for (int v = 0; v < n; ++v)
        if (term_index[v] < 0) total_w += weights.empty() ? 1 : weights[v];
    for (int i = 0; i < k; ++i) total_c += capacities[i];
    if (total_w > total_c)
        return fail("total non-terminal weight " + std::to_string(total_w) + " exceeds total capacity " + std::to_string(total_c) +
                    " (Σ w_v <= Σ c_t required, paper_notes §9)");

    // ------------------------------------------------------------------ in-lists sorted by canonical position:
    // a global counting sort (iterate tails in ≺-order), O(n + m). Variant 0 does not need the order but
    // shares the structure.
    std::vector<int> in_start(n + 1, 0);
    for (int u = 0; u < n; ++u)
        for (int p = out_begin(u); p < out_end(u); ++p) ++in_start[out_list[p] + 1];
    for (int v = 0; v < n; ++v) in_start[v + 1] += in_start[v];
    std::vector<int> in_list(in_start[n]);
    {
        std::vector<int> fill(in_start.begin(), in_start.end() - 1);
        for (int i = 0; i < n - k; ++i) {  // non-terminals in ≺-order (terminals have no out-arcs)
            const int u = order[i];
            for (int p = out_begin(u); p < out_end(u); ++p) in_list[fill[out_list[p]]++] = u;
        }
    }

    // ------------------------------------------------------------------ state of [Alg 5]
    std::vector<char> used(n, 0);
    std::vector<int> part_of(n, -1), parent(n, -1);
    std::vector<int64_t> residual(capacities);
    for (int i = 0; i < k; ++i) part_of[terminals[i]] = i;
    std::vector<int> active;  // ascending terminal indices (policies 1 and 2 index into it like the reference)
    std::vector<char> is_active(k, 0);
    std::priority_queue<ResidualKey> res_pq;  // policy 0, lazy deletion (stale = residual changed / inactive)
    for (int i = 0; i < k; ++i) {
        if (residual[i] > 0) {
            active.push_back(i);
            is_active[i] = 1;
            if (policy == 0) res_pq.push(ResidualKey{residual[i], i});
        }
    }
    int64_t rr = 0;  // round-robin counter (reference semantics: active[rr % len(active)], rr += 1)
    int r = n - k;
    int64_t done = 0;  // contractions performed by this call

    std::vector<TermHeap> heaps;
    std::vector<std::vector<Level>> stacks;
    if (variant == 0) {
        heaps.resize(k);
        for (int i = 0; i < k; ++i) {
            const int t = terminals[i];
            for (int p = in_start[t]; p < in_start[t + 1]; ++p) heaps[i].push(HeapEntry{pos[in_list[p]], in_list[p], t});
        }
    } else {
        stacks.resize(k);
        for (int i = 0; i < k; ++i) stacks[i].push_back(Level{terminals[i], in_start[terminals[i]]});
    }

    if (trace.enabled) {
        std::string arcs_json = "[";
        bool first = true;
        for (int u = 0; u < n; ++u)
            for (int p = out_begin(u); p < out_end(u); ++p) {
                if (!first) arcs_json += ',';
                first = false;
                arcs_json += "[" + std::to_string(u) + "," + std::to_string(out_list[p]) + "]";
            }
        arcs_json += ']';
        std::vector<int64_t> w_out(n, 0);
        for (int v = 0; v < n; ++v)
            if (term_index[v] < 0) w_out[v] = weights.empty() ? 1 : weights[v];
        trace.record("init", "{\"n\":" + std::to_string(n) + ",\"k\":" + std::to_string(k) + ",\"terminals\":" + vec_json(terminals) +
                                 ",\"capacities\":" + vec_json(capacities) + ",\"arcs\":" + arcs_json + ",\"weights\":" + vec_json(w_out) +
                                 ",\"order\":" + vec_json(order) + ",\"policy\":" + std::to_string(policy) + ",\"variant\":" +
                                 std::to_string(variant) + "}");
    }

    // choose the active terminal (§13.7); every policy is valid.
    auto select_terminal = [&]() -> int {
        if (active.empty()) throw std::runtime_error("dag_partition: no active terminal although non-terminals remain (Σ w <= Σ c violated)");
        if (policy == 2) return active[0];
        if (policy == 1) {
            const int i = active[(size_t)(rr % (int64_t)active.size())];
            ++rr;
            return i;
        }
        while (!res_pq.empty()) {
            const ResidualKey key = res_pq.top();
            if (is_active[key.idx] && key.residual == residual[key.idx]) return key.idx;
            res_pq.pop();  // stale
        }
        throw std::logic_error("dag_partition: residual queue empty while terminals are active");
    };
    auto deactivate = [&](int i) {
        is_active[i] = 0;
        active.erase(std::find(active.begin(), active.end(), i));
    };

    // ------------------------------------------------------------------ main loop of [Alg 5]
    while (r > 0) {
        const int i = select_terminal();
        int p = -1, x = -1;
        if (variant == 0) {
            // extract-min, skipping stale entries [Alg 5]; the entry's x is the parent (min over V_i by order).
            TermHeap& H = heaps[i];
            while (!H.empty()) {
                const HeapEntry e = H.top();
                H.pop();
                if (used[e.p]) continue;
                p = e.p;
                x = e.x;
                break;
            }
        } else {
            // P1: the minimum unused entry of H_i is the first unused element of the deepest non-exhausted level.
            std::vector<Level>& S = stacks[i];
            while (!S.empty()) {
                Level& L = S.back();
                const int end = in_start[L.x + 1];
                while (L.ptr < end && used[in_list[L.ptr]]) ++L.ptr;
                if (L.ptr == end) {
                    S.pop_back();
                    continue;
                }
                p = in_list[L.ptr++];
                break;
            }
            if (p >= 0) {
                // parent = smallest x ∈ V_i with (p, x) ∈ E: exactly the x of the minimum heap entry of p.
                for (int q = out_begin(p); q < out_end(p); ++q) {
                    const int y = out_list[q];
                    if (part_of[y] == i && (x < 0 || y < x)) x = y;
                }
                if (x < 0) throw std::logic_error("dag_partition: stack variant selected a vertex without an arc into the part");
            }
        }
        if (p < 0)
            throw std::runtime_error("[Lem 9.4] heap of active terminal " + std::to_string(terminals[i]) +
                                     " ran dry; the input is not k-T-connected");
        // contract p into t_i [Lem 9.3]: V_i += {p}, residual -= w_p, new pre-terminals = in-neighbours of p.
        used[p] = 1;
        part_of[p] = i;
        parent[p] = x;
        residual[i] -= weights.empty() ? 1 : weights[p];
        --r;
        ++done;
        if (variant == 0) {
            TermHeap& H = heaps[i];
            for (int q = in_start[p]; q < in_start[p + 1]; ++q) {
                const int u = in_list[q];
                if (!used[u]) H.push(HeapEntry{pos[u], u, p});  // used u would be a stale entry anyway
            }
        } else {
            stacks[i].push_back(Level{p, in_start[p]});
        }
        if (trace.enabled)
            trace.record("dag_contract", "{\"p\":" + std::to_string(p) + ",\"t\":" + std::to_string(terminals[i]) + ",\"parent\":" +
                                             std::to_string(x) + ",\"residual\":" + std::to_string(residual[i]) + "}");
        if (residual[i] <= 0) {
            deactivate(i);
        } else if (policy == 0) {
            res_pq.push(ResidualKey{residual[i], i});
        }
    }
    stats.contractions += done;
    stats.steps += done;

    if (trace.enabled) {
        std::vector<std::vector<int>> parts(k);
        for (int v = 0; v < n; ++v) parts[part_of[v]].push_back(v);
        std::string s = "{\"parts\":{";
        for (int i = 0; i < k; ++i) {
            if (i) s += ',';
            s += "\"" + std::to_string(terminals[i]) + "\":" + vec_json(parts[i]);
        }
        s += "}}";
        trace.record("done", s);
    }
    return DagResult{"ok", "ok", std::move(part_of), std::move(parent)};
}

}  // namespace glcore
