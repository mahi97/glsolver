// The general polynomial-time solvers (paper_notes §7, §8) with the exact optimizations of docs/optimizations.md.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "essential.hpp"
#include "graph.hpp"
#include "stats.hpp"
#include "trace.hpp"

namespace glcore {

struct SolverOptions {
    int threads = 1;
    int seed = 0;
    bool greedy_contraction = true;   // O5
    bool lazy_shift = true;           // O6
    bool batch_unused_arcs = true;    // O1 batch deletion of arcs used by no flow
    bool debug_asserts = false;       // A1–A8 (expensive)
    bool trace = false;
    bool record_cuts = false;
    bool check_precondition = true;   // report kappa < k vertices (informational) and FEAC failure
};

struct SolveResult {
    std::string status;               // "ok" | "precondition_failed"
    std::string message;
    std::vector<int> assignment;      // part index per vertex
    std::vector<int> parent;          // certificate: original out-neighbour in the same part, -1 for terminals
    std::vector<int> witness;         // final/initial witness phi (terminal index) per vertex or -1
    bool k_T_connected = false;       // all kappa == k at the start
};

class GLSolver {
public:
    GLSolver(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
             const std::vector<int64_t>& capacities, SolverOptions opt);
    SolveResult run();
    Stats& stats() { return stats_; }
    Trace& trace() { return trace_; }
private:
    Graph g_;
    Stats stats_;
    Trace trace_;
    SolverOptions opt_;
    EssentialOracle oracle_;
    std::vector<int64_t> cap_;         // by original terminal index
    std::vector<int> phi_;             // witness: terminal index per vertex, -1 if none/removed
    std::vector<int> parent_;
    std::vector<int> part_;            // final part index per vertex
    // operations
    bool step_remove_zero_terminal();
    bool step_contract_degree_one();
    bool step_greedy_contraction();    // O5
    void step_shift_assignment();      // [Alg 2] with O6
    bool find_initial_witness();
    void debug_check_witness(const char* where);
};

class GLWeightedSolver {
public:
    GLWeightedSolver(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                     const std::vector<int64_t>& capacities, const std::vector<int64_t>& weights, SolverOptions opt);
    SolveResult run();
    Stats& stats() { return stats_; }
    Trace& trace() { return trace_; }
private:
    Graph g_;
    Stats stats_;
    Trace trace_;
    SolverOptions opt_;
    EssentialOracle oracle_;
    std::vector<int64_t> cap_, w_;
    std::vector<int> parent_, part_;
    bool fesac_feasible(bool exact_cuts);
    bool step_remove_zero_terminal();
    bool step_contract_degree_one();
    bool step_matching_delete();       // (iii)
    void step_round_and_remove();      // (iv)
};

}  // namespace glcore
