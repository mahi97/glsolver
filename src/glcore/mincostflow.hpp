// Min-cost flow (successive shortest paths with potentials) for split assignments [Prop 5.4].
#pragma once
#include <cstdint>
#include <vector>

namespace glcore {

class MinCostFlow {
public:
    explicit MinCostFlow(int num_nodes);
    int add_arc(int u, int v, int64_t cap, int64_t cost);  // returns arc id; flow(id) after solve
    // Sends up to `required` units from s to z at minimum cost. Returns (value, cost).
    std::pair<int64_t, int64_t> solve(int s, int z, int64_t required);
    int64_t flow(int arc) const;
private:
    struct E { int to; int64_t cap; int64_t cost; };
    int n_;
    std::vector<E> edges_;
    std::vector<std::vector<int>> adj_;
    std::vector<int64_t> potential_;
};

}  // namespace glcore
