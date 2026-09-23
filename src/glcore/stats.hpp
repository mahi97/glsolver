// Counters and timers reported through result.stats (docs/api.md).
#pragma once
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace glcore {

struct Stats {
    int64_t max_flow_calls = 0;       // full recomputations
    int64_t augment_calls = 0;        // single warm-started augmentations
    int64_t cut_calls = 0;            // reverse reachability computations
    int64_t matching_calls = 0;
    int64_t min_cost_flow_calls = 0;
    int64_t contractions = 0;
    int64_t deletions = 0;
    int64_t batched_deletions = 0;
    int64_t cycle_shifts = 0;
    int64_t terminal_removals = 0;
    int64_t roundings = 0;
    int64_t greedy_attempts = 0;
    int64_t greedy_successes = 0;
    int64_t shift_calls = 0;
    int64_t assignment_repairs = 0;   // witness recomputations
    int64_t steps = 0;
    // C1 diagnostics (RESEARCH_NOTES E5). flow_repairs: stored flows re-validated by evaluate_deletion
    // (summed over all calls, speculative O5 attempts included) - "repairs per deletion" is this over
    // `deletions`. NOT independent of augment_calls: each repaired flow drops the paths through the deleted
    // arcs and re-augments once per dropped path (plus at most one failing call), so where almost every
    // affected flow loses exactly one path (the Harary family) the two counters nearly coincide -- quote
    // them together as one signal, not as two. penalized_users_initial / _witness: the number of stored flows using an arc a pre-terminal
    // is likely to lose, summed over those arcs, measured right after compute_all (no witness yet, so
    // "likely to lose" = every out-arc of a pre-terminal that does not enter a terminal) and again right
    // after the initial witness (= every out-arc of a pre-terminal p other than (p, phi(p))).
    int64_t flow_repairs = 0;
    int64_t reroutes = 0;             // flows recomputed by the C1 re-routing pass
    int64_t penalized_users_initial = 0;
    int64_t penalized_users_witness = 0;
    std::map<std::string, double> time_seconds;  // per primitive
    std::vector<std::pair<int, int>> graph_size_over_time;  // (live non-terminals, live arcs) per step

    struct Timer {
        Stats& s; std::string key; std::chrono::steady_clock::time_point t0;
        Timer(Stats& s_, std::string k) : s(s_), key(std::move(k)), t0(std::chrono::steady_clock::now()) {}
        ~Timer() { s.time_seconds[key] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    };
};

}  // namespace glcore
