// Integral min-cost flow by successive shortest paths with Johnson potentials [Prop 5.4] (paper_notes §5).
//
// Network arcs are stored in pairs: arc id `a` (even) is the forward arc returned by add_arc, `a ^ 1` is its
// residual partner (capacity 0, cost -cost). `cap` of an E is the *residual* capacity, so the flow on a
// forward arc equals the residual capacity of its partner. Costs must be nonnegative, so the initial
// potentials are zero and every Dijkstra runs on nonnegative reduced costs c(u,v) + h(u) - h(v). Each
// augmentation pushes the bottleneck of the path (capped by the remaining requirement), so all flows stay
// integral ("Must be integral (SSP on integral data is)", paper_notes §5).
#include "mincostflow.hpp"

#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace glcore {

// [Prop 5.4] empty network on num_nodes nodes.
MinCostFlow::MinCostFlow(int num_nodes) : n_(num_nodes), adj_(num_nodes > 0 ? num_nodes : 0), potential_(num_nodes > 0 ? num_nodes : 0, 0) {
    if (num_nodes <= 0) throw std::invalid_argument("MinCostFlow: num_nodes must be positive, got " + std::to_string(num_nodes));
}

// [Prop 5.4] add the arc u -> v with capacity cap >= 0 and cost >= 0; returns its (even) arc id.
int MinCostFlow::add_arc(int u, int v, int64_t cap, int64_t cost) {
    if (u < 0 || u >= n_ || v < 0 || v >= n_)
        throw std::invalid_argument("MinCostFlow::add_arc: node out of range (" + std::to_string(u) + "," + std::to_string(v) +
                                    ") for " + std::to_string(n_) + " nodes");
    if (cap < 0) throw std::invalid_argument("MinCostFlow::add_arc: capacities must be nonnegative");
    if (cost < 0) throw std::invalid_argument("MinCostFlow::add_arc: costs must be nonnegative (zero initial potentials)");
    if (edges_.size() > size_t(std::numeric_limits<int>::max() - 2))
        throw std::length_error("MinCostFlow::add_arc: too many arcs");
    const int id = (int)edges_.size();
    edges_.push_back(E{v, cap, cost});
    adj_[u].push_back(id);
    edges_.push_back(E{u, 0, -cost});
    adj_[v].push_back(id + 1);
    return id;
}

// Flow on arc `arc`: for a forward arc the residual capacity of its partner; for a partner the negative of
// the forward flow (same convention as the reference glref.mincostflow.MinCostFlow.flow).
int64_t MinCostFlow::flow(int arc) const {
    if (arc < 0 || arc >= (int)edges_.size()) throw std::out_of_range("MinCostFlow::flow: arc id out of range");
    return (arc & 1) ? -edges_[arc].cap : edges_[arc ^ 1].cap;
}

// [Prop 5.4] successive shortest paths: repeat { Dijkstra on reduced costs; update potentials; augment by the
// bottleneck } until `required` units are sent or z is unreachable. Returns (value, cost); value < required
// means no flow of the requested value exists (FESAC fails in the split-assignment use).
std::pair<int64_t, int64_t> MinCostFlow::solve(int s, int z, int64_t required) {
    if (s < 0 || s >= n_ || z < 0 || z >= n_) throw std::invalid_argument("MinCostFlow::solve: source/sink out of range");
    if (s == z) throw std::invalid_argument("MinCostFlow::solve: source and sink must differ");
    if (required < 0) throw std::invalid_argument("MinCostFlow::solve: required must be nonnegative");
    const int64_t INF = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> dist(n_);
    std::vector<int> parent_arc(n_);
    // Binary heap keyed by (tentative distance, node): deterministic pop order, lazy deletion.
    using Item = std::pair<int64_t, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    int64_t value = 0, total_cost = 0;
    while (value < required) {
        // ---- Dijkstra with reduced costs (Johnson potentials keep them nonnegative).
        std::fill(dist.begin(), dist.end(), INF);
        std::fill(parent_arc.begin(), parent_arc.end(), -1);
        dist[s] = 0;
        pq.push(Item{0, s});
        while (!pq.empty()) {
            const Item it = pq.top();
            pq.pop();
            const int u = it.second;
            if (it.first > dist[u]) continue;  // stale entry
            const int64_t du = dist[u];
            for (int a : adj_[u]) {
                const E& e = edges_[a];
                if (e.cap <= 0) continue;
                const int64_t rc = e.cost + potential_[u] - potential_[e.to];  // >= 0 by the potential invariant
                if (rc < 0) throw std::logic_error("MinCostFlow::solve: negative reduced cost; potential invariant violated");
                const int64_t nd = du + rc;
                if (nd < dist[e.to]) {
                    dist[e.to] = nd;
                    parent_arc[e.to] = a;
                    pq.push(Item{nd, e.to});
                }
            }
        }
        if (dist[z] == INF) break;  // maximum flow reached below `required`
        // ---- potentials: h(v) += dist(v) for reachable v keeps every residual reduced cost nonnegative.
        for (int v = 0; v < n_; ++v)
            if (dist[v] != INF) potential_[v] += dist[v];
        // ---- bottleneck along the parent arcs (capped by the remaining requirement).
        int64_t amt = required - value;
        int64_t path_cost = 0;
        for (int v = z; v != s;) {
            const int a = parent_arc[v];
            const E& e = edges_[a];
            if (e.cap < amt) amt = e.cap;
            path_cost += e.cost;
            v = edges_[a ^ 1].to;
        }
        if (amt <= 0) throw std::logic_error("MinCostFlow::solve: zero bottleneck on a shortest path");
        for (int v = z; v != s;) {
            const int a = parent_arc[v];
            edges_[a].cap -= amt;
            edges_[a ^ 1].cap += amt;
            v = edges_[a ^ 1].to;
        }
        value += amt;
        int64_t inc;
        if (__builtin_mul_overflow(amt, path_cost, &inc) || __builtin_add_overflow(total_cost, inc, &total_cost))
            throw std::overflow_error("MinCostFlow::solve: total cost overflows int64");
    }
    return {value, total_cost};
}

}  // namespace glcore
