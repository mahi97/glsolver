// GLDAGPartition [Alg 5] (paper_notes §9), O(m log n).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "stats.hpp"
#include "trace.hpp"

namespace glcore {

struct DagResult {
    std::string status;             // "ok" | "precondition_failed"
    std::string message;
    std::vector<int> assignment;    // part index per vertex, -1 if unassigned
    std::vector<int> parent;        // in-arborescence parent (original arc head) or -1
};

// policy: 0 = max residual capacity, 1 = round robin, 2 = first active.
DagResult dag_partition(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                        const std::vector<int64_t>& capacities, const std::vector<int64_t>& weights,
                        int policy, Stats& stats, Trace& trace, bool check_precondition);

}  // namespace glcore
