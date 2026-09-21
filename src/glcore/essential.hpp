// Per-vertex essential-terminal certificates for all live non-terminals (docs/implementation.md, optimizations O1–O4).
#pragma once
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
    // Returns the list of affected vertices.
    std::vector<int> evaluate_deletion(const std::vector<int>& D, std::vector<VertexFlow>& out);
    // Commit: the graph has been modified by deleting D; adopt the flows computed by evaluate_deletion.
    void commit_deletion(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& computed);
    // The graph has been modified by contracting p into t (already done on the Graph): translate flows.
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
};

}  // namespace glcore
