// Per-vertex essential-terminal certificates for all live non-terminals (docs/implementation.md, optimizations O1–O4).
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "flow.hpp"
#include "graph.hpp"

namespace glcore {

struct Stats;  // stats.hpp

class EssentialOracle {
public:
    EssentialOracle(Graph& g, Stats& stats, int threads, int seed);

    // Compute flows (and exact cuts) for every live non-terminal from scratch. Parallel over vertices.
    void compute_all();
    // Ensure vertex v has a validated flow for the current graph version (recompute from scratch if stale).
    const VertexFlow& flow(int v);
    // Certified essential terminals of v (subset of the truth; exact if cut_exact). Refresh with refresh_cut(v).
    const TermSet& ess(int v);
    void refresh_cut(int v);     // recompute the exact tightest cut for v (one reverse BFS)
    void refresh_all_cuts();     // for all live non-terminals (used before witness search)
    int kappa(int v);

    // Vertices whose stored flow uses arc a (exact: filtered by current flow version).
    std::vector<int> users_of_arc(int a);
    // For a set D of arcs to be deleted: for every vertex v whose flow uses an arc of D, compute the
    // warm-started flow in G \ D (docs/optimizations.md O2/O5) into `out[v]` (a VertexFlow valid for G \ D,
    // with exact cut when kappa dropped). Vertices not using D are not touched. Parallel.
    // Returns the list of affected vertices (sorted). With exact_cuts the tightest cut of G \ D is computed for
    // every affected vertex; otherwise only when kappa dropped (else the stored Ess stays a certified subset, O1/P2).
    std::vector<int> evaluate_deletion(const std::vector<int>& D, std::vector<VertexFlow>& out, bool exact_cuts = false);
    // Commit: the graph has been modified by deleting D; adopt the flows computed by evaluate_deletion.
    void commit_deletion(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& computed);
    // The graph has been modified by contracting p into t (already done on the Graph, as the LAST mutation):
    // translate flows (O3). With d^+(p) = 1 at contraction time every stored cut stays exact (§13.2); with
    // d^+(p) >= 2 the contraction also deleted arcs, so cuts are demoted to certified subsets (as after a
    // deletion) and kappa / the flows stay valid. Throws std::invalid_argument if the Graph's contraction
    // record does not describe (p, t).
    void after_contraction(int p, int t);
    // The graph has been modified by removing terminal t (already done): fix flows/cuts (O4).
    void after_terminal_removal(int t);
    // Vertex removed (rounding): drop its flow; flows passing through it are recomputed lazily.
    void after_vertex_removal(int v);

    Graph& graph() { return g_; }
private:
    Graph& g_;
    Stats& stats_;
    int threads_;
    int seed_;
    FlowEngine engine_;
    std::vector<VertexFlow> flows_;            // indexed by vertex
    std::vector<std::vector<std::pair<int, uint64_t>>> arc_users_;  // arc id -> (v, flow stamp)
    std::vector<uint64_t> flow_stamp_;         // per vertex, bumped when its flow changes
    void index_users(int v);

    // ---- validity discipline (see essential.cpp header) ----
    std::vector<char> valid_;                  // flows_[v] is a maximum flow of the current graph (when synced)
    uint64_t synced_version_ = 0;              // graph version the oracle has processed all mutations up to
    std::vector<std::vector<std::pair<int, uint64_t>>> vertex_users_;  // vertex x -> (v, stamp): flows through x
    std::vector<FlowEngine::ScratchPtr> scratch_;                      // one workspace per worker thread
    // Cut exactness: a stored exact cut is exact for the current graph iff it was computed in the current
    // "exact epoch"; the epoch advances on every mutation that can move tightest cuts of unaffected vertices
    // (arc deletion, vertex removal, contraction of a pre-terminal with d^+(p) >= 2, which deletes arcs) —
    // O(1) instead of demoting every flow. Contraction with d^+(p) = 1 keeps cuts (O3, §13.2).
    uint64_t exact_epoch_ = 1;
    std::vector<uint64_t> cut_epoch_;
    bool cut_is_exact(int v) const { return flows_[v].cut_exact && cut_epoch_[v] == exact_epoch_; }
    void mark_cut_exact(int v) { cut_epoch_[v] = exact_epoch_; }
    bool is_current(int v) const { return valid_[v] && synced_version_ == g_.version(); }
    void ensure_synced();                      // graph changed behind our back -> invalidate everything
    bool expect_mutations(uint64_t count);     // hooks: exactly `count` graph mutations since the last sync
    FlowEngine::Scratch& scratch(int worker);
    void run_parallel(size_t count, const std::function<void(size_t, int)>& fn);
    std::vector<int> users_of_arc_nosync(int a);
    std::vector<int> users_of_vertex_nosync(int x);
    void recompute_stale(const std::vector<int>& verts);  // parallel from-scratch flows for the stale ones
    // Stale bookkeeping in O(size): vertices invalidated one by one are queued; a global invalidation only
    // raises a flag (the O(n) sweep happens once, when the queue is drained).
    std::vector<int> stale_;
    bool all_stale_ = false;
    void invalidate(int v) { valid_[v] = 0; stale_.push_back(v); }
    void invalidate_all() { std::fill(valid_.begin(), valid_.end(), 0); stale_.clear(); all_stale_ = true; }
    void recompute_pending_stale();            // drain the queue (or sweep once after invalidate_all)
};

}  // namespace glcore
