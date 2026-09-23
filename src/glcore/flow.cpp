// Implicit vertex-split unit-capacity flows and the tightest minimum cut (docs/paper_notes.md §3.1
// [Prop 4.2], docs/optimizations.md O1–O4).
//
// The network H_v of [Prop 4.2] is represented implicitly on top of the Graph:
//   * split node x_in = 2x, x_out = 2x+1; the split arc (x_in,x_out) has capacity 1 (flag through[x]);
//   * every graph arc (x,y) is the network arc (x_out, y_in) of capacity K > kappa (flow bit of arc_bits[a]);
//   * the source s feeds v_in with capacity K and v's split arc has capacity K, so v_in is never useful
//     for an augmenting path (its only exits are v_out and back to s) and never reachable from the sink in
//     a maximum flow (that would make s reachable). We therefore start every augmenting search at v_out,
//     never route flow into v, and treat v as R by construction; through[v] stays 0.
//   * every terminal t has the arc (t_out, z) of capacity K, never saturated: an augmenting search is
//     complete as soon as it reaches t_in for a terminal t with through[t] == 0 (t_in -> t_out -> z).
// Flow flags and through flags live in the per-thread Scratch and are restricted to the arcs/vertices
// touched (touched lists), so loading another VertexFlow costs O(size), never O(n+m). Visited marks use
// generation stamps (O(1) reset). Forbidden arcs (a set D about to be deleted, O2/O5) are marked in a
// per-scratch array for the duration of one search and treated as absent.
// next_arc[x] is the flow arc leaving x (x != v has at most one by the unit split capacity), maintained by
// every flag change, so that the explicit path lists are rebuilt in O(total path length) after an
// augmentation instead of scanning every out-list along the paths; the flags are re-derived from the
// rebuilt paths only when the augmentation left flow on a cycle (detected by counting flow arcs).
#include "flow.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace glcore {

struct FlowEngine::Scratch {
    int n = 0;
    // Per arc id, one byte holding two bits: kFlow (the loaded flow uses the arc) and kForbidden (transient
    // mark for one search). One array instead of two halves the random memory traffic of the BFS loops.
    std::vector<unsigned char> arc_bits;
    static constexpr unsigned char kFlow = 1, kForbidden = 2;
    std::vector<char> through;                // per vertex: 1 iff a loaded path passes through / ends at it
    std::vector<int> next_arc;                // per vertex: the flow arc leaving it (-1 if none; unused for v)
    std::vector<uint32_t> stamp;              // per split node: visited iff stamp == gen
    uint32_t gen = 0;
    std::vector<int> parent_node, parent_arc; // BFS tree: predecessor node and the arc used (-1 = split move)
    std::vector<int> queue;
    std::vector<int> queue_next;              // 0-1 search only: the nodes of the next penalty level
    std::vector<int> dist;                    // 0-1 search only: penalty distance (valid iff stamp == gen)
    std::vector<int> touched_arcs, touched_vertices;
    int num_flow_arcs = 0;                    // arcs carrying the flow bit
    int loaded_v = -1;                        // source vertex of the flow currently loaded (-1 = none)

    bool visited(int node) const { return stamp[node] == gen; }
    void mark(int node, int from, int arc) { stamp[node] = gen; parent_node[node] = from; parent_arc[node] = arc; }
    void next_gen() {
        if (++gen == 0) { std::fill(stamp.begin(), stamp.end(), 0u); gen = 1; }
    }
    bool has_flow(int a) const { return (arc_bits[a] & kFlow) != 0; }
    bool is_forbidden(int a) const { return (arc_bits[a] & kForbidden) != 0; }
    void set_arc(int a, int tail) {
        if (!has_flow(a)) { arc_bits[a] |= kFlow; touched_arcs.push_back(a); ++num_flow_arcs; }
        next_arc[tail] = a;
    }
    void clear_arc(int a, int tail) {
        if (has_flow(a)) { arc_bits[a] &= (unsigned char)~kFlow; --num_flow_arcs; }
        if (next_arc[tail] == a) next_arc[tail] = -1;
    }
    void set_through(int x) { if (!through[x]) { through[x] = 1; touched_vertices.push_back(x); } }
};

namespace {
inline int node_in(int x) { return 2 * x; }
inline int node_out(int x) { return 2 * x + 1; }
inline int node_vertex(int node) { return node >> 1; }
inline bool node_is_out(int node) { return (node & 1) != 0; }
}  // namespace

FlowEngine::FlowEngine(const Graph& g) : g_(g) {}

FlowEngine::Scratch* FlowEngine::new_scratch() const {
    auto* s = new Scratch;
    int n = g_.n();
    s->n = n;
    s->through.assign(n, 0);
    s->next_arc.assign(n, -1);
    s->stamp.assign(2 * n, 0);
    s->parent_node.assign(2 * n, -1);
    s->parent_arc.assign(2 * n, -1);
    s->arc_bits.assign(g_.num_arc_ids(), 0);
    s->queue.reserve(2 * n);
    return s;
}

void FlowEngine::free_scratch(Scratch* s) const { delete s; }

namespace {
// Arc ids grow with contractions: extend the per-arc arrays on demand (amortized O(1)).
void ensure_arc_capacity(FlowEngine::Scratch& s, const Graph& g) {
    size_t m = (size_t)g.num_arc_ids();
    if (s.arc_bits.size() < m) s.arc_bits.resize(m, 0);
}

// Clear every flag set since the last reset (O(touched)). Every vertex whose next_arc entry was set is the
// loaded source or has had its through flag set (hence is in touched_vertices): a path arc's tail is v or
// the head of the previous arc, and the augmentation sets an arc out of x only when x_out was reached from
// x_in (the flip then sets through[x]) or through a flow arc leaving x (x already on a path). So next_arc
// is reset through the touched-vertex list, without looking any arc up.
void reset_flags(FlowEngine::Scratch& s) {
    for (int a : s.touched_arcs) s.arc_bits[a] &= (unsigned char)~FlowEngine::Scratch::kFlow;
    for (int x : s.touched_vertices) {
        s.through[x] = 0;
        s.next_arc[x] = -1;
    }
    if (s.loaded_v >= 0) s.next_arc[s.loaded_v] = -1;
    s.touched_arcs.clear();
    s.touched_vertices.clear();
    s.num_flow_arcs = 0;
    s.loaded_v = -1;
}

void mark_forbidden(FlowEngine::Scratch& s, const std::vector<int>& D, const Graph& g) {
    for (int a : D) {
        if (a < 0 || a >= g.num_arc_ids()) throw std::invalid_argument("FlowEngine: forbidden arc id out of range");
        s.arc_bits[a] |= FlowEngine::Scratch::kForbidden;
    }
}
void unmark_forbidden(FlowEngine::Scratch& s, const std::vector<int>& D) {
    for (int a : D) s.arc_bits[a] &= (unsigned char)~FlowEngine::Scratch::kForbidden;
}

void require_loaded(const FlowEngine::Scratch& s, const VertexFlow& f, const char* where) {
    if (s.loaded_v != f.v || f.v < 0)
        throw std::logic_error(std::string("FlowEngine::") + where + ": call load() for this flow first");
}

void set_path_flags(FlowEngine::Scratch& s, const VertexFlow& f, const Graph& g) {
    const int m = g.num_arc_ids();
    for (const auto& path : f.paths) {
        for (int a : path) {
            if (a < 0 || a >= m) throw std::logic_error("FlowEngine::load: bad arc id in path");
            const Arc& e = g.arc(a);
            s.set_arc(a, e.tail);
            s.set_through(e.head);
        }
    }
}
}  // namespace

// Set the scratch flags to exactly the paths of f (previous contents reset in O(size of previous)).
void FlowEngine::load(const VertexFlow& f, Scratch& s) const {
    reset_flags(s);
    ensure_arc_capacity(s, g_);
    if (f.v < 0 || f.v >= g_.n()) throw std::invalid_argument("FlowEngine::load: flow has no source vertex");
    s.loaded_v = f.v;  // before the flags: reset_flags clears next_arc of the loaded source
    set_path_flags(s, f, g_);
}

// Rebuild the explicit path list of f from the flag state after an augmentation: from v follow, at every
// vertex, its unique out-arc with flow 1 (next_arc) until a terminal. Flow conservation with unit split
// capacities guarantees uniqueness; any flow left on cycles (not reachable from v) is detected by comparing
// the total path length with the number of flow arcs and discarded by re-setting the flags from the
// rebuilt paths, which keeps the same value. Paths come out vertex-disjoint and end at distinct terminals
// [Def 3.2]; verified under GLCORE_DEBUG_ASSERTS.
void FlowEngine::rebuild_paths(VertexFlow& f, Scratch& s) const {
    const int v = f.v;
    size_t np = 0;
    size_t total = 0;
    f.path_terminal.clear();
    for (int a0 : g_.out_arcs(v)) {
        if (!s.has_flow(a0)) continue;
        if (np == f.paths.size()) f.paths.emplace_back();
        std::vector<int>& path = f.paths[np++];
        path.clear();
        path.push_back(a0);
        int x = g_.arc(a0).head;
        int steps = 0;
        while (!g_.is_terminal(x)) {
            const int nxt = s.next_arc[x];
            if (nxt < 0 || !s.has_flow(nxt) || g_.arc(nxt).tail != x)
                throw std::logic_error("FlowEngine: flow conservation violated at vertex " + std::to_string(x));
            path.push_back(nxt);
            x = g_.arc(nxt).head;
            if (++steps > g_.n()) throw std::logic_error("FlowEngine: cycle while walking flow paths");
        }
        total += path.size();
        f.path_terminal.push_back(x);
    }
    f.paths.resize(np);  // keeps the capacity of the surviving path vectors
    if ((int)f.paths.size() != f.kappa)
        throw std::logic_error("FlowEngine: path count " + std::to_string(f.paths.size()) + " != kappa " +
                               std::to_string(f.kappa));
    // Flow on a cycle (unreachable from v): re-derive the flags from the paths, which drops it.
    if ((int)total != s.num_flow_arcs) {
        const int lv = s.loaded_v;
        reset_flags(s);
        s.loaded_v = lv;
        set_path_flags(s, f, g_);
    }
#ifdef GLCORE_DEBUG_ASSERTS
    {
        std::vector<char> seen(g_.n(), 0);
        for (size_t i = 0; i < f.paths.size(); ++i) {
            for (int a : f.paths[i]) {
                int y = g_.arc(a).head;
                if (seen[y]) throw std::logic_error("FlowEngine: paths are not vertex-disjoint");
                seen[y] = 1;
            }
            if (!g_.is_terminal(f.path_terminal[i])) throw std::logic_error("FlowEngine: path does not end at a terminal");
        }
        // every vertex other than v has at most one flow arc leaving it, and the flags match the paths
        int flow_arcs = 0;
        for (int a = 0; a < g_.num_arc_ids(); ++a) flow_arcs += s.has_flow(a) ? 1 : 0;
        if (flow_arcs != (int)total || flow_arcs != s.num_flow_arcs)
            throw std::logic_error("FlowEngine: flag state does not match the rebuilt paths");
        for (int x = 0; x < g_.n(); ++x) {
            if (x == v || !g_.live(x)) continue;
            int cnt = 0;
            for (int b : g_.out_arcs(x)) cnt += s.has_flow(b) ? 1 : 0;
            if (cnt > 1) throw std::logic_error("FlowEngine: two flow arcs leave vertex " + std::to_string(x));
            if (cnt == 1 && (s.next_arc[x] < 0 || !s.has_flow(s.next_arc[x])))
                throw std::logic_error("FlowEngine: next_arc out of sync at vertex " + std::to_string(x));
        }
    }
#endif
}

// One Edmonds–Karp step in the residual network of H_v restricted to the current graph minus `forbidden`
// (paper_notes §3.1; O2 warm start). Residual moves (see the file header for why v_in never appears):
//   from x_out: forward arc (x,y) alive, not forbidden, flow 0  -> y_in   (flow-1 arcs are dead ends);
//               reverse split (through[x] == 1, x != v)           -> x_in;
//   from x_in : forward split (through[x] == 0)                   -> x_out;
//               reverse arc (w,x) with flow 1, not forbidden       -> w_out.
// Reaching y_in for a terminal y with through[y] == 0 completes the path (y_in -> y_out -> z).
//
// Deletion-aware routing (C1, `set_penalties`). With Penalized = true the search is a 0-1 shortest-path
// search whose length is the total penalty of the FORWARD arcs of the path (a reverse residual move removes
// an arc from the flow and costs 0). Instead of a deque (whose push_front would turn the penalty-0 part of
// the search into a depth-first order) the frontier is two FIFOs: `queue` holds the current penalty level,
// `queue_next` the next one; a penalty-0 move appends to the current level, a penalty-1 move to the next.
// Levels are therefore searched in plain BFS order, which (a) breaks ties between equal-penalty paths exactly
// as the old BFS did and (b) makes the all-zero-penalty case bit-identical to it. A node may be relabelled
// when a cheaper level reaches it; `dist` holds its current label and a stale queue entry (dist != level) is
// skipped. The first target found by a penalty-0 move at the current level is optimal (every unsettled node
// has distance >= level); a target first reached by a penalty-1 move is remembered and returned when the
// current level is exhausted, which is again the first target in the order the plain BFS would have used.
template <bool Penalized>
int FlowEngine::search_target(const VertexFlow& f, Scratch& s) const {
    const int v = f.v;
    s.next_gen();
    s.queue.clear();
    const int src = node_out(v);
    s.mark(src, -1, -1);
    s.queue.push_back(src);
    const unsigned char* pen = nullptr;
    int pen_n = 0;
    int target = -1, pending = -1, level = 0;
    if constexpr (Penalized) {
        // The 0-1 arrays are sized here, not in new_scratch: the plain BFS never allocates them.
        if (s.dist.size() < (size_t)(2 * s.n)) s.dist.assign((size_t)(2 * s.n), 0);
        s.queue_next.clear();
        s.dist[src] = 0;
        pen = pen_->data();
        pen_n = (int)pen_->size();
    }
    // Label `to` at distance level + w if that is an improvement (or the first visit).
    auto relax = [&](int to, int from, int arc, int w) -> bool {
        if constexpr (Penalized) {
            if (s.visited(to) && s.dist[to] <= level + w) return false;
            s.mark(to, from, arc);
            s.dist[to] = level + w;
            return true;
        } else {
            (void)w;
            if (s.visited(to)) return false;
            s.mark(to, from, arc);
            return true;
        }
    };
    // The scan of one penalty level. `head` is declared INSIDE the level loop (it restarts at 0 on every
    // level: a level swaps in a fresh queue), which keeps the inner loop exactly the plain BFS loop of the
    // pre-C1 engine. This is a measured requirement, not a style choice: hoisting `head` out of the level
    // loop costs the plain BFS ~19 % (gcc 13.3 -O3, aarch64) although the emitted work is the same —
    // RESEARCH_NOTES E5-review. With Penalized = false the level loop itself collapses (the `else break;`).
    for (;;) {
        for (size_t head = 0; head < s.queue.size() && target < 0; ++head) {
            const int node = s.queue[head];
            if constexpr (Penalized) {
                if (s.dist[node] != level) continue;  // superseded by a cheaper label
            }
            const int x = node_vertex(node);
            if (node_is_out(node)) {
                for (int a : g_.out_arcs(x)) {
                    if (s.arc_bits[a]) continue;  // flow 1 (dead end) or forbidden
                    const int y = g_.arc(a).head;
                    if (y == v) continue;  // never route flow into the source
                    const int yin = node_in(y);
                    int w = 0;
                    if constexpr (Penalized) {
                        // A node already labelled at this level cannot be improved by any weight >= 0: skip it
                        // before reading the penalty byte, which is a second random access per arc (on dense
                        // graphs almost every arc leads to an already-labelled node).
                        if (s.visited(yin) && s.dist[yin] <= level) continue;
                        w = (a < pen_n) ? (int)pen[a] : 0;
                    }
                    if (!relax(yin, node, a, w)) continue;
                    if (!s.through[y] && g_.is_terminal(y)) {
                        if (w == 0) { target = yin; break; }
                        if constexpr (Penalized) {
                            if (pending < 0) pending = yin;
                        }
                        continue;
                    }
                    if constexpr (Penalized) (w == 0 ? s.queue : s.queue_next).push_back(yin);
                    else s.queue.push_back(yin);
                }
                if (target >= 0) break;
                if (x != v && s.through[x]) {
                    const int xin = node_in(x);
                    if (relax(xin, node, -1, 0)) s.queue.push_back(xin);
                }
            } else {
                if (!s.through[x]) {
                    const int xout = node_out(x);
                    if (relax(xout, node, -1, 0)) s.queue.push_back(xout);
                } else {
                    for (int a : g_.in_arcs(x)) {
                        if (s.arc_bits[a] != FlowEngine::Scratch::kFlow) continue;  // needs flow 1 and not forbidden
                        const int wout = node_out(g_.arc(a).tail);
                        if (relax(wout, node, a, 0)) s.queue.push_back(wout);
                    }
                }
            }
        }
        // Advance to the next penalty level (Penalized only; the plain BFS leaves the loop here).
        if constexpr (Penalized) {
            if (target >= 0) break;
            if (pending >= 0) { target = pending; break; }  // level exhausted: the penalty-1 target wins
            if (s.queue_next.empty()) break;
            s.queue.swap(s.queue_next);
            s.queue_next.clear();
            ++level;
        } else {
            break;
        }
    }
    return target;
}

// Keeping both searches out of augment_once is a measured requirement, not a style choice: see flow.hpp.
#if defined(__GNUC__) || defined(__clang__)
#define GLCORE_NOINLINE [[gnu::noinline]]
#else
#define GLCORE_NOINLINE
#endif
GLCORE_NOINLINE int FlowEngine::search_plain(const VertexFlow& f, Scratch& s) const {
    return search_target<false>(f, s);
}
GLCORE_NOINLINE int FlowEngine::search_penalized(const VertexFlow& f, Scratch& s) const {
    return search_target<true>(f, s);
}
#undef GLCORE_NOINLINE

bool FlowEngine::augment_once(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden) const {
    require_loaded(s, f, "augment_once");
    ensure_arc_capacity(s, g_);
    const int v = f.v;
    mark_forbidden(s, forbidden, g_);
    // Two instantiations: with pen_ == nullptr the plain BFS runs, with no penalty test in its loops; both
    // are called, never inlined here (flow.hpp explains the measurement that forced that).
    const int target = pen_ ? search_penalized(f, s) : search_plain(f, s);
    const int src = node_out(v);
    unmark_forbidden(s, forbidden);
    if (target < 0) return false;
    // Augment: flip the flags along the tree path target -> src. next_arc is kept consistent in either
    // order of "set the new out-arc" / "clear the old out-arc" of a vertex (clear_arc only resets it when it
    // still names the cleared arc).
    for (int node = target; node != src;) {
        const int pn = s.parent_node[node];
        const int pa = s.parent_arc[node];
        const int x = node_vertex(node);
        if (pa >= 0) {
            if (!node_is_out(node)) s.set_arc(pa, g_.arc(pa).tail);    // forward arc x_out -> y_in
            else s.clear_arc(pa, g_.arc(pa).tail);                    // reverse arc x_in -> w_out
        } else {
            if (node_is_out(node)) s.set_through(x); // forward split x_in -> x_out
            else s.through[x] = 0;                   // reverse split x_out -> x_in
        }
        node = pn;
    }
    s.set_through(node_vertex(target));              // the new terminal endpoint (t_out -> z)
    f.kappa += 1;
    rebuild_paths(f, s);
    return true;
}

// [Prop 4.2] kappa_G(v) and, optionally, the tightest minimum cut, from scratch: at most k augmentations.
void FlowEngine::compute_max_flow(int v, VertexFlow& f, Scratch& s, bool compute_cut_, bool with_sides) const {
    if (v < 0 || v >= g_.n() || !g_.live(v) || g_.is_terminal(v))
        throw std::invalid_argument("FlowEngine::compute_max_flow: " + std::to_string(v) + " is not a live non-terminal");
    f.v = v;
    f.kappa = 0;
    f.version = g_.version();
    f.paths.clear();
    f.path_terminal.clear();
    f.ess = TermSet(g_.k0());
    f.cut_exact = false;
    f.side.clear();
    load(f, s);
    static const std::vector<int> none;
    int guard = 0;
    while (augment_once(f, s, none)) {
        if (++guard > g_.k()) throw std::logic_error("FlowEngine: more augmentations than terminals");
    }
    if (compute_cut_) compute_cut(f, s, none, with_sides);
}

// [Prop 4.2] tightest minimum cut from a maximum flow: Reach = split nodes that can reach z in the residual
// graph (reverse BFS from every t_out, since (t_out,z) has capacity K > flow). Reverse residual rules:
//   y_out in Reach: y_in if through[y] == 0 (residual on the split arc);   w_in for out-arcs (y,w) with flow 1;
//   y_in  in Reach: y_out if through[y] == 1 (reverse of the split arc);   x_out for EVERY alive non-forbidden
//                   in-arc (x,y) (capacity K is never saturated).
// Sides: L if x_in reached, S if only x_out reached, R otherwise; v is R (v_out reachable would make s
// reachable, contradicting maximality — we check this and throw), |S| = kappa, T ⊆ L ∪ S. Ess = T ∩ S [Lem 4.1].
// x_in reached implies x_out reached (x_in is only entered through a flow arc, whose head has through == 1,
// or from x_out itself), so |S| = #reached out-nodes − #reached in-nodes and the side of every vertex is
// only materialized on request (with_sides).
void FlowEngine::compute_cut(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden, bool with_sides) const {
    require_loaded(s, f, "compute_cut");
    ensure_arc_capacity(s, g_);
    const int v = f.v;
    const int n = g_.n();
    mark_forbidden(s, forbidden, g_);
    s.next_gen();
    s.queue.clear();
    for (int t : g_.terminals()) {
        const int tout = node_out(t);
        s.mark(tout, -1, -1);
        s.queue.push_back(tout);
    }
    bool not_max = false;
    for (size_t head = 0; head < s.queue.size(); ++head) {
        const int node = s.queue[head];
        const int x = node_vertex(node);
        if (node_is_out(node)) {
            if (!s.through[x] && x != v) {
                const int xin = node_in(x);
                if (!s.visited(xin)) { s.mark(xin, node, -1); s.queue.push_back(xin); }
            }
            for (int a : g_.out_arcs(x)) {
                if (s.arc_bits[a] != FlowEngine::Scratch::kFlow) continue;  // flow 1 and not forbidden
                const int win = node_in(g_.arc(a).head);
                if (!s.visited(win)) { s.mark(win, node, a); s.queue.push_back(win); }
            }
        } else {
            if (s.through[x]) {
                const int xout = node_out(x);
                if (!s.visited(xout)) { s.mark(xout, node, -1); s.queue.push_back(xout); }
            }
            for (int a : g_.in_arcs(x)) {
                if (s.is_forbidden(a)) continue;
                const int w = g_.arc(a).tail;
                if (w == v) { not_max = true; continue; }
                const int wout = node_out(w);
                if (!s.visited(wout)) { s.mark(wout, node, a); s.queue.push_back(wout); }
            }
        }
    }
    unmark_forbidden(s, forbidden);
    if (not_max)
        throw std::logic_error("FlowEngine::compute_cut: the flow of vertex " + std::to_string(v) +
                               " is not maximum (an out-neighbour's in-node reaches the sink)");
    // The queue holds exactly the reached nodes (every mark is followed by a push).
    int cnt_in = 0, cnt_out = 0;
    for (int node : s.queue) {
        if (node_is_out(node)) ++cnt_out; else ++cnt_in;
    }
    const int cnt_s = cnt_out - cnt_in;
    if (cnt_s != f.kappa)
        throw std::logic_error("FlowEngine::compute_cut: |S| = " + std::to_string(cnt_s) + " != kappa = " +
                               std::to_string(f.kappa) + " for vertex " + std::to_string(v));
    if (f.ess.k != g_.k0() || f.ess.w.size() != (size_t)((g_.k0() + 63) / 64)) f.ess = TermSet(g_.k0());
    else std::fill(f.ess.w.begin(), f.ess.w.end(), 0u);
    for (int t : g_.terminals()) {
        // t_out is a BFS root; t is essential iff t_in is not reached (t ∈ S) [Lem 4.1]
        if (!s.visited(node_in(t))) f.ess.set(g_.terminal_index(t));
    }
    if (with_sides) {
        f.side.assign(n, Side::R);
        for (int x = 0; x < n; ++x) {
            if (!g_.live(x)) continue;
            const bool rin = s.visited(node_in(x)), rout = s.visited(node_out(x));
            if (rin) {
                if (!rout) throw std::logic_error("FlowEngine::compute_cut: x_in reached but x_out not [Prop 4.2]");
                f.side[x] = Side::L;
            } else if (rout) {
                f.side[x] = Side::S;
            }
        }
        for (int t : g_.terminals())
            if (f.side[t] == Side::R) throw std::logic_error("FlowEngine::compute_cut: terminal on the R side");
    } else {
        f.side.clear();
    }
    f.cut_exact = true;
}

// Clear the flags of path i and drop it from f (O(length)).
void FlowEngine::remove_path(VertexFlow& f, size_t i, Scratch& s) const {
    for (int a : f.paths[i]) {
        const Arc& e = g_.arc(a);
        s.clear_arc(a, e.tail);
        s.through[e.head] = 0;
    }
    f.paths.erase(f.paths.begin() + (std::ptrdiff_t)i);
    f.path_terminal.erase(f.path_terminal.begin() + (std::ptrdiff_t)i);
    f.kappa -= 1;
}

// O2: drop the unique path using arc a (paths are vertex-disjoint; if the tail of a is v itself the arc is
// the first arc of one path).
bool FlowEngine::remove_path_using_arc(VertexFlow& f, int a, Scratch& s) const {
    require_loaded(s, f, "remove_path_using_arc");
    for (size_t i = 0; i < f.paths.size(); ++i)
        for (int b : f.paths[i])
            if (b == a) { remove_path(f, i, s); return true; }
    return false;
}

// O2/O5: one pass over the paths with D marked in the forbidden array; the surviving paths keep their order
// (identical to removing the paths one arc of D at a time).
int FlowEngine::remove_paths_using_arcs(VertexFlow& f, const std::vector<int>& D, Scratch& s) const {
    require_loaded(s, f, "remove_paths_using_arcs");
    ensure_arc_capacity(s, g_);
    // A small D (the common O5/O6 case) is compared directly, a sequential scan of the path vectors; a
    // large one (dense graphs, |D| = d^+(p) - 1) is marked in the scratch bits.
    const bool small = D.size() <= 4;
    if (!small) mark_forbidden(s, D, g_);
    else
        for (int a : D)
            if (a < 0 || a >= g_.num_arc_ids()) throw std::invalid_argument("FlowEngine: forbidden arc id out of range");
    int removed = 0;
    size_t w = 0;
    for (size_t i = 0; i < f.paths.size(); ++i) {
        bool hit = false;
        if (small) {
            for (int b : f.paths[i]) {
                for (int a : D)
                    if (b == a) { hit = true; break; }
                if (hit) break;
            }
        } else {
            for (int b : f.paths[i])
                if (s.is_forbidden(b)) { hit = true; break; }
        }
        if (hit) {
            for (int a : f.paths[i]) {
                const Arc& e = g_.arc(a);
                s.clear_arc(a, e.tail);
                s.through[e.head] = 0;
            }
            ++removed;
            continue;
        }
        if (w != i) {
            f.paths[w].swap(f.paths[i]);
            f.path_terminal[w] = f.path_terminal[i];
        }
        ++w;
    }
    if (!small) unmark_forbidden(s, D);
    if (removed) {
        f.paths.resize(w);
        f.path_terminal.resize(w);
        f.kappa -= removed;
    }
    return removed;
}

// O4: drop the path ending at terminal t (at most one).
bool FlowEngine::remove_path_to_terminal(VertexFlow& f, int t, Scratch& s) const {
    require_loaded(s, f, "remove_path_to_terminal");
    for (size_t i = 0; i < f.paths.size(); ++i)
        if (f.path_terminal[i] == t) { remove_path(f, i, s); return true; }
    return false;
}

// O1 membership test on the explicit path lists.
bool FlowEngine::uses_arc(const VertexFlow& f, int a) {
    for (const auto& path : f.paths)
        for (int b : path)
            if (b == a) return true;
    return false;
}

namespace {
// Position (i, j) of the arc entering p on a path of f, or false if no path passes through p.
bool find_entry(const Graph& g, const VertexFlow& f, int p, size_t& i, size_t& j) {
    for (i = 0; i < f.paths.size(); ++i)
        for (j = 0; j < f.paths[i].size(); ++j)
            if (g.arc(f.paths[i][j]).head == p) return true;
    return false;
}
}  // namespace

bool FlowEngine::can_translate_after_contraction(const VertexFlow& f, int p, int t) const {
    if (f.v == p) return false;
    size_t i, j;
    if (!find_entry(g_, f, p, i, j)) return true;  // no path through p: nothing to translate
    const auto& path = f.paths[i];
    return j + 2 == path.size() && g_.arc(path[j + 1]).head == t;
}

// O3: after Graph::contract(p, t) the path ... x -(x,p)-> p -(p,t)-> t of f (if any) becomes ... x -(x,t)-> t,
// where (x,t) is the redirected arc (new id) or the pre-existing merged arc (old id). The other paths are
// untouched; kappa and Ess are unchanged (paper_notes §13.2). If f.v == p the flow is meaningless (p is
// dead) and the call is a no-op: callers drop it.
void FlowEngine::translate_after_contraction(VertexFlow& f, int p, int t) const {
    if (f.v == p) return;
    size_t i, j;
    if (!find_entry(g_, f, p, i, j)) return;  // paths are vertex-disjoint: at most one passes through p
    if (!translate_path_after_contraction(f, i, j, p, t))
        throw std::logic_error("FlowEngine::translate_after_contraction: the path through " + std::to_string(p) +
                               " does not continue to " + std::to_string(t) + " (contraction requires d^+(p) = 1 on flow arcs)");
}

bool FlowEngine::translate_path_after_contraction(VertexFlow& f, size_t i, size_t j, int p, int t) const {
    if (f.v == p) return false;
    if (i >= f.paths.size() || j >= f.paths[i].size() || g_.arc(f.paths[i][j]).head != p)
        throw std::logic_error("FlowEngine::translate_path_after_contraction: position (" + std::to_string(i) + "," +
                               std::to_string(j) + ") of the flow of " + std::to_string(f.v) + " does not enter " + std::to_string(p));
    std::vector<int>& path = f.paths[i];
    if (j + 2 != path.size() || g_.arc(path[j + 1]).head != t) return false;
    const int x = g_.arc(path[j]).tail;
    const int a_new = g_.find_arc(x, t);
    if (a_new < 0)
        throw std::logic_error("FlowEngine::translate_path_after_contraction: redirected arc (" + std::to_string(x) + "," +
                               std::to_string(t) + ") not found");
    if (f.path_terminal[i] != t) throw std::logic_error("FlowEngine::translate_path_after_contraction: path terminal mismatch");
    path[j] = a_new;
    path.pop_back();
    return true;
}

}  // namespace glcore
