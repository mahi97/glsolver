// Per-vertex essential-terminal certificates for all live non-terminals (docs/implementation.md, optimizations O1–O4).
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "flow.hpp"
#include "graph.hpp"
#include "threadpool.hpp"

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
    // The L/S/R side of every vertex in the exact tightest cut of v (size n; dead vertices read R), computed
    // by one reverse BFS on demand — the oracle stores only kappa, the paths and the Ess bitset (a stored
    // side[] per flow would cost n^2 bytes). Makes the cut of v exact as a side effect (like refresh_cut).
    void sides(int v, std::vector<Side>& out);

    // Vertices whose stored flow uses arc a (exact: the user index is maintained exactly), sorted.
    std::vector<int> users_of_arc(int a);
    // |users_of_arc(a)| in O(1).
    size_t num_users_of_arc(int a);
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
    // record does not describe (p, t). O(#flows through p), each translated in O(1) via the user index.
    void after_contraction(int p, int t);
    // The graph has been modified by removing terminal t (already done): fix flows/cuts (O4).
    void after_terminal_removal(int t);
    // Vertex removed (rounding): drop its flow; flows passing through it are recomputed lazily.
    void after_vertex_removal(int v);
    // C1 deletion-aware routing: hand the engine the solver's per-arc penalty array (nullptr = plain BFS).
    // The array is read by every augmenting search, including inside the parallel regions, so the owner may
    // only change it between oracle calls. Exactness: see FlowEngine::set_penalties.
    void set_penalties(const std::vector<unsigned char>* pen) { engine_.set_penalties(pen); }
    // C1 re-routing: recompute the stored flows of `verts` from scratch (in parallel, with exact cuts) under
    // the current penalties. Exact for the same reason compute_all is: a from-scratch maximum flow with its
    // tightest cut. kappa and Ess are unchanged, only which arcs the certificates use. Returns the number of
    // flows actually recomputed.
    size_t reroute(const std::vector<int>& verts);

    Graph& graph() { return g_; }
private:
    Graph& g_;
    Stats& stats_;
    int threads_;
    int seed_;
    FlowEngine engine_;
    std::vector<VertexFlow> flows_;            // indexed by vertex

    // ---- exact user index (see essential.cpp header) ----
    // One registry entry per arc of a registered flow's paths: reg_[v][i][j] describes paths[i][j] and
    // remembers its position in the two user lists, which store back-references (v, i, j); every removal is a
    // swap-remove that patches the moved entry's stored position. Memory = O(total live path length).
    struct Entry {
        int arc;        // the arc id (kept explicitly: paths may be rewritten in place before unregistering)
        int arc_slot;   // position of this flow's reference in arc_users_[arc]
        int vtx_slot;   // position of this flow's reference in vertex_users_[head(arc)]
    };
    struct Ref {
        int v, path, pos;  // reg_[v][path][pos]
    };
    std::vector<std::vector<std::vector<Entry>>> reg_;   // per vertex, per path (inner vectors keep capacity)
    std::vector<std::vector<Ref>> arc_users_;            // arc id -> flows whose paths use the arc
    std::vector<std::vector<Ref>> vertex_users_;         // vertex x -> flows whose paths enter x
    void register_flow(int v);      // index the current paths of flows_[v] (reg_[v] must be empty)
    void unregister_flow(int v);    // remove every entry of v from both lists, O(|entries of v|)
    void detach_arc_ref(const Entry& e);      // swap-remove the reference from arc_users_[e.arc]
    void detach_vertex_ref(const Entry& e);   // swap-remove the reference from vertex_users_[head(e.arc)]
    void ensure_arc_index_size();

    // ---- validity discipline (see essential.cpp header) ----
    std::vector<char> valid_;                  // flows_[v] is a maximum flow of the current graph (when synced)
    uint64_t synced_version_ = 0;              // graph version the oracle has processed all mutations up to
    std::vector<FlowEngine::ScratchPtr> scratch_;                      // one workspace per worker thread
    WorkerPool pool_;                                                  // persistent worker threads (O8)
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
    // raises a flag (the O(n) sweep happens once, when the queue is drained). Invalidation unregisters the
    // flow from the user index, so the index only ever holds valid flows.
    std::vector<int> stale_;
    bool all_stale_ = false;
    void invalidate(int v);
    void invalidate_all();
    void recompute_pending_stale();            // drain the queue (or sweep once after invalidate_all)
};

}  // namespace glcore
