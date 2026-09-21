// GLDAGPartition [Alg 5] (paper_notes §9), O(m log n) with heaps or O(n + m) with the stack variant (RESEARCH_NOTES.md P1).
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

// policy:  0 = max residual capacity (ties by terminal index), 1 = round robin over the active terminals,
//          2 = first active terminal (paper_notes §13.7; every policy is valid).
// variant: 0 = one binary min-heap per terminal, exactly [Alg 5] (O(m log n), [Lem 9.6]);
//          1 = the linear stack-of-sorted-in-lists variant of RESEARCH_NOTES.md P1 (O(n + m), plus
//              O(n log k) for policy 0). Both variants produce identical assignments and parents.
// weights: empty for unit weights, else one entry per vertex (terminal entries ignored, non-terminals >= 1).
// check_precondition: test [Lem 9.1] (every non-terminal has out-degree >= k); acyclicity is always tested
// because the canonical order [Def 9.2] does not exist otherwise. Failures give status "precondition_failed"
// with the offending vertex in `message`; malformed input throws std::invalid_argument.
DagResult dag_partition(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                        const std::vector<int64_t>& capacities, const std::vector<int64_t>& weights,
                        int policy, int variant, Stats& stats, Trace& trace, bool check_precondition);

}  // namespace glcore
