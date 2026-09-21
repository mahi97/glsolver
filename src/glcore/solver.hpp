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

    // ---- private state of solver_general.cpp (paper_notes §7, docs/optimizations.md) ----
    std::vector<int> term_vertex_;     // original terminal index -> terminal vertex (inverse of terminal_index)
    std::vector<int> initial_witness_; // phi right after the witness search (reported in SolveResult::witness)
    std::string witness_message_;      // FEAC failure explanation
    // O(size) work lists: vertices whose out-degree changed (candidates for step (ii)), terminals whose
    // capacity reached 0 (step (i)), vertices that ever had an arc into a terminal (steps (iii-a)/(iii-b)).
    std::vector<int> deg_queue_;
    size_t deg_head_ = 0;
    std::vector<char> in_deg_queue_;
    std::vector<int> zero_terms_;      // terminal indices
    size_t zero_head_ = 0;
    std::vector<int> pt_candidates_;
    std::vector<char> in_pt_candidates_;
    // Dirty tracking for the (iii-a)/(iii-b) scans: a candidate's unused arcs / user score can only change
    // when a flow using one of its arcs changes (commit, contraction), when its phi changes, or when it
    // becomes a candidate; a terminal removal changes every flow (full rescan). Scans cost O(dirty).
    std::vector<char> cand_dirty_;
    std::vector<int> cand_dirty_list_;
    bool cand_all_dirty_ = true;
    std::vector<int64_t> cand_users_;         // cached O5 score per candidate (-1: not eligible)
    std::vector<int> cand_dsize_;
    std::vector<int> batch_arcs_;             // unused arcs found by the last refresh (O1 batch)
    std::vector<int64_t> greedy_skip_until_;  // per vertex: step number until which O5 skips it (backoff)
    std::vector<int> greedy_failures_;
    int greedy_credit_ = 0;                   // O5 throttle: attempts cost credit, successes earn it
    // Saturating matching reused across ShiftAssignment calls [Lem 7.8]: repaired in O(in-degree) when a
    // matched pair dies (contraction / deletion), recomputed from scratch only when a repair fails.
    std::vector<int> match_p_;                // terminal index -> matched pre-terminal or -1
    std::vector<int> matched_to_;             // vertex -> terminal index or -1
    // evaluate_deletion output slots, reused across steps (slot i = secondary arc e_i; scratch_slot_ = O1/O5)
    std::vector<std::vector<VertexFlow>> eval_slots_;
    std::vector<VertexFlow> scratch_slot_;
    int64_t size_sample_every_ = 1;

    bool step_batch_unused_arcs();     // O1 (iii-a)
    void contract_into(int p, int t);  // step (ii) body [Lem 7.4]
    void mark_candidate_dirty(int v);
    void mark_flow_arcs_dirty(int v);  // tails of the arcs of v's current stored flow
    void refresh_dirty_candidates();   // recompute batch_arcs_ / cand_users_ for the dirty candidates
    bool ensure_saturating_matching(); // reuse + repair, else recompute [Lem 7.8]
    bool deletion_keeps_witness(const std::vector<int>& affected, const std::vector<VertexFlow>& out);
    void delete_and_commit(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& out);
    void push_degree_changed(int v);
    void push_pt_candidate(int v);
    void sample_graph_size();
    void check_after_operation(const char* where);
    void trace_essential();
    std::string capacities_json() const;
    std::string phi_json() const;
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

    // ---- private state of solver_weighted.cpp (paper_notes §8, docs/optimizations.md) ----
    std::vector<int> term_vertex_;     // original terminal index -> terminal vertex
    // Carried split witness psi [Def 5.2]: per vertex the pairs (terminal index, units > 0); recv_[ti] is
    // Σ_v psi(v, t_i). psi is a FESAC witness [Def 5.3] of the current graph after every operation (see the
    // header comment of solver_weighted.cpp) and psi(v,t) > 0 implies t ∈ oracle_.ess(v) (P2 discipline).
    std::vector<std::vector<std::pair<int, int64_t>>> psi_;
    std::vector<int64_t> recv_;
    // O(size) work lists: vertices whose out-degree changed (step (ii) candidates), terminals whose
    // capacity reached 0 (step (i)), pre-terminals (steps (iii-a)/(iii-b)) with dirty tracking as in GLSolver.
    std::vector<int> deg_queue_;
    size_t deg_head_ = 0;
    std::vector<char> in_deg_queue_;
    std::vector<int> zero_terms_;      // terminal indices
    size_t zero_head_ = 0;
    std::vector<int> pt_candidates_;
    std::vector<char> in_pt_candidates_;
    std::vector<char> cand_dirty_;
    std::vector<int> cand_dirty_list_;
    bool cand_all_dirty_ = true;
    std::vector<int64_t> cand_users_;         // cached O5 score per candidate (-1: not eligible)
    std::vector<int> cand_dsize_;
    std::vector<int> cand_target_;            // arc (p, t) with psi(p,t) > 0 kept by O1/O5 (-1: none)
    std::vector<int> batch_arcs_;             // unused arcs found by the last refresh (O1 batch)
    std::vector<int64_t> greedy_skip_until_;  // per vertex: step number until which O5 skips it (backoff)
    std::vector<int> greedy_failures_;
    int greedy_credit_ = 0;                   // O5 throttle (see solver_general.cpp)
    // Saturating matching reused across steps (iii) [Lem 7.8]: repaired in O(in-degree), recomputed when needed.
    std::vector<int> match_p_;                // terminal index -> matched pre-terminal or -1
    std::vector<int> matched_to_;             // vertex -> terminal index or -1
    // evaluate_deletion output slots (slot i = secondary arc e_i; scratch_slot_ = O1/O5)
    std::vector<std::vector<VertexFlow>> eval_slots_;
    std::vector<VertexFlow> scratch_slot_;
    // Criticality cost table xi_v(t) [Def 6.2], rows only for the vertices affected by some secondary arc.
    std::vector<int> cost_row_of_;
    std::vector<std::vector<int64_t>> cost_rows_;
    std::vector<int> costed_vertices_;
    int64_t size_sample_every_ = 1;

    struct SplitOutcome {
        bool feasible = false;
        int64_t value = 0;
        int64_t cost = 0;
    };
    // [Prop 5.4] min-cost split assignment over the stored essential sets; with_costs uses the cost table,
    // adopt installs the solution as psi_. debug_only books the call as invariant verification.
    SplitOutcome solve_split_assignment(bool with_costs, bool adopt, bool debug_only);
    void build_cost_table(int kk, const std::vector<std::vector<int>>& aff);
    void clear_cost_table();
    bool step_batch_unused_arcs();     // O1 (iii-a)
    bool step_greedy_contraction();    // O5 (iii-b)
    void contract_into(int p, int t);  // step (ii) body [Lem 8.2]
    bool ensure_saturating_matching(); // reuse + repair, else recompute [Lem 7.8]
    // psi stays a witness of G \ D iff every pair with psi(v,t) > 0 keeps t essential (exact for dropped
    // kappa, O1 otherwise); special_p (if >= 0) is checked against the single terminal special_ti instead.
    bool psi_survives_deletion(const std::vector<int>& affected, const std::vector<VertexFlow>& out, int special_p, int special_ti);
    void delete_and_commit(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& out);
    int psi_target_arc(int p) const;   // alive arc (p, t) with the largest psi(p, t) > 0, or -1
    void set_psi_single(int p, int ti);
    int64_t psi_units(int v, int ti) const;
    void push_degree_changed(int v);
    void push_pt_candidate(int v);
    void mark_candidate_dirty(int v);
    void mark_flow_arcs_dirty(int v);
    void refresh_dirty_candidates();
    void sample_graph_size();
    void check_after_operation(const char* where);
    void debug_check_witness(const char* where);
    void trace_essential();
    std::string capacities_json() const;
    std::string psi_json() const;
};

}  // namespace glcore
