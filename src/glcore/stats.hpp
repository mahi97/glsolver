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
    std::map<std::string, double> time_seconds;  // per primitive
    std::vector<std::pair<int, int>> graph_size_over_time;  // (live non-terminals, live arcs) per step

    struct Timer {
        Stats& s; std::string key; std::chrono::steady_clock::time_point t0;
        Timer(Stats& s_, std::string k) : s(s_), key(std::move(k)), t0(std::chrono::steady_clock::now()) {}
        ~Timer() { s.time_seconds[key] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    };
};

}  // namespace glcore
