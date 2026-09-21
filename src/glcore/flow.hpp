// Vertex-split unit-capacity flows for terminal connectivity (paper_notes §3.1, docs/optimizations.md O1–O4).
//
// The network H_v of [Prop 4.2] is never materialized. A VertexFlow stores, for a source vertex v, a family
// of kappa vertex-disjoint paths to distinct terminals as (a) per-arc flow flags and (b) per-vertex "through"
// flags, both restricted to the arcs/vertices touched (so reset is O(size)). The engine supports:
//   * from scratch:  compute_max_flow(v) -> kappa augmentations (BFS), O(k (n+m))
//   * warm start:    remove_path_through_arc / remove_path_to_terminal, then augment_once, O(n+m)
//   * tightest cut:  reach_sink() reverse BFS in the residual graph -> sides (L/S/R) and Ess bitset
// All operations read the *current* Graph; a VertexFlow is only meaningful for the graph version it was
// last validated against (see `version`). Paths are stored as arc ids in `path_arcs` (per path).
#pragma once
#include <memory>
#include <vector>

#include "bitset.hpp"
#include "graph.hpp"

namespace glcore {

enum class Side : char { L = 0, S = 1, R = 2 };

struct VertexFlow {
    int v = -1;
    int kappa = 0;
    uint64_t version = 0;                 // graph version the flow was last validated against
    std::vector<std::vector<int>> paths;  // paths[i] = arc ids from v to a terminal (in order)
    std::vector<int> path_terminal;       // terminal at the end of paths[i]
    // certified essential terminals (bitset over ORIGINAL terminal indices); always a subset of the true
    // Ess_G(v) for the current graph (docs/optimizations.md, "stale subset" discipline); exact when cut_exact.
    TermSet ess;
    bool cut_exact = false;
    std::vector<Side> side;               // tightest cut sides (only valid when cut_exact); size n
};

class FlowEngine {
public:
    explicit FlowEngine(const Graph& g);

    // Full recomputation for vertex v: BFS augmenting paths until no augmenting path (<= k iterations).
    // Fills paths, kappa, and (if compute_cut) the exact tightest cut, side[] and ess. Thread-safe when each
    // thread uses its own scratch (see Scratch below).
    struct Scratch;  // opaque per-thread workspace (queues, visited stamps, arc/vertex flags)
    Scratch* new_scratch() const;
    void free_scratch(Scratch*) const;
    // RAII owner for a Scratch (the deleter never needs the complete type).
    struct ScratchDeleter {
        const FlowEngine* engine = nullptr;
        void operator()(Scratch* s) const { if (engine && s) engine->free_scratch(s); }
    };
    using ScratchPtr = std::unique_ptr<Scratch, ScratchDeleter>;
    ScratchPtr make_scratch() const { return ScratchPtr(new_scratch(), ScratchDeleter{this}); }

    void compute_max_flow(int v, VertexFlow& f, Scratch& s, bool compute_cut) const;

    // Load f into the scratch flow flags (O(total path length)). Must be called before the warm-start ops.
    void load(const VertexFlow& f, Scratch& s) const;
    // Remove the path of f that uses arc a (no-op and return false if none uses it).
    bool remove_path_using_arc(VertexFlow& f, int a, Scratch& s) const;
    // Remove the path ending at terminal t (returns false if none).
    bool remove_path_to_terminal(VertexFlow& f, int t, Scratch& s) const;
    // One BFS augmentation in the residual network of the current graph, treating the arcs in `forbidden`
    // (alive arc ids) as deleted. Returns true if a path was found (kappa incremented, paths updated).
    bool augment_once(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden) const;
    // Reverse reachability from the sink in the residual graph of the current flow (must be maximum),
    // treating `forbidden` arcs as deleted: fills f.side, f.ess (exact), sets cut_exact = true.
    void compute_cut(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden) const;
    // Does the stored flow use arc a? O(1) via the scratch flags after load(), or O(size) without.
    static bool uses_arc(const VertexFlow& f, int a);
    // Translate paths across the contraction of p into t: any path ... x -(a1)-> p -(a2)-> t becomes
    // ... x -(a_new)-> t, where a_new is the redirected arc id (found via graph lookup). Paths not through p
    // are unchanged. Must be called after Graph::contract with the same arguments.
    void translate_after_contraction(VertexFlow& f, int p, int t) const;
    // True iff translate_after_contraction(f, p, t) is well defined: f.v != p and the path of f through p
    // (if any) continues along the arc (p,t) (O3 needs d^+(p) = 1 at contraction time; a flow whose path
    // leaves p elsewhere must be recomputed instead). Read-only, O(total path length).
    bool can_translate_after_contraction(const VertexFlow& f, int p, int t) const;

    const Graph& graph() const { return g_; }
private:
    const Graph& g_;
    void rebuild_paths(VertexFlow& f, Scratch& s) const;          // explicit paths from the flag state
    void remove_path(VertexFlow& f, size_t i, Scratch& s) const;  // clear flags of path i and drop it
};

}  // namespace glcore
