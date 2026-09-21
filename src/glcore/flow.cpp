// Implicit vertex-split unit-capacity flows and the tightest minimum cut (docs/paper_notes.md §3.1
// [Prop 4.2], docs/optimizations.md O1–O4).
//
// The network H_v of [Prop 4.2] is represented implicitly on top of the Graph:
//   * split node x_in = 2x, x_out = 2x+1; the split arc (x_in,x_out) has capacity 1 (flag through[x]);
//   * every graph arc (x,y) is the network arc (x_out, y_in) of capacity K > kappa (flag arc_flow[a] in {0,1});
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
#include "flow.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace glcore {

struct FlowEngine::Scratch {
    int n = 0;
    std::vector<char> arc_flow;               // per arc id: 1 iff the loaded flow uses the arc
    std::vector<char> forbidden;              // per arc id: transient marks for one search
    std::vector<char> through;                // per vertex: 1 iff a loaded path passes through / ends at it
    std::vector<uint32_t> stamp;              // per split node: visited iff stamp == gen
    uint32_t gen = 0;
    std::vector<int> parent_node, parent_arc; // BFS tree: predecessor node and the arc used (-1 = split move)
    std::vector<int> queue;
    std::vector<int> touched_arcs, touched_vertices;
    int loaded_v = -1;                        // source vertex of the flow currently loaded (-1 = none)

    bool visited(int node) const { return stamp[node] == gen; }
    void mark(int node, int from, int arc) { stamp[node] = gen; parent_node[node] = from; parent_arc[node] = arc; }
    void next_gen() {
        if (++gen == 0) { std::fill(stamp.begin(), stamp.end(), 0u); gen = 1; }
    }
    void set_arc(int a) { if (!arc_flow[a]) { arc_flow[a] = 1; touched_arcs.push_back(a); } }
    void set_through(int x) { if (!through[x]) { through[x] = 1; touched_vertices.push_back(x); } }
    void reset_flags() {
        for (int a : touched_arcs) arc_flow[a] = 0;
        for (int x : touched_vertices) through[x] = 0;
        touched_arcs.clear();
        touched_vertices.clear();
        loaded_v = -1;
    }
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
    s->stamp.assign(2 * n, 0);
    s->parent_node.assign(2 * n, -1);
    s->parent_arc.assign(2 * n, -1);
    s->arc_flow.assign(g_.num_arc_ids(), 0);
    s->forbidden.assign(g_.num_arc_ids(), 0);
    s->queue.reserve(2 * n);
    return s;
}

void FlowEngine::free_scratch(Scratch* s) const { delete s; }

namespace {
// Arc ids grow with contractions: extend the per-arc arrays on demand (amortized O(1)).
void ensure_arc_capacity(FlowEngine::Scratch& s, const Graph& g) {
    size_t m = (size_t)g.num_arc_ids();
    if (s.arc_flow.size() < m) {
        s.arc_flow.resize(m, 0);
        s.forbidden.resize(m, 0);
    }
}

void mark_forbidden(FlowEngine::Scratch& s, const std::vector<int>& D, const Graph& g) {
    for (int a : D) {
        if (a < 0 || a >= g.num_arc_ids()) throw std::invalid_argument("FlowEngine: forbidden arc id out of range");
        s.forbidden[a] = 1;
    }
}
void unmark_forbidden(FlowEngine::Scratch& s, const std::vector<int>& D) {
    for (int a : D) s.forbidden[a] = 0;
}

void require_loaded(const FlowEngine::Scratch& s, const VertexFlow& f, const char* where) {
    if (s.loaded_v != f.v || f.v < 0)
        throw std::logic_error(std::string("FlowEngine::") + where + ": call load() for this flow first");
}
}  // namespace

// Set the scratch flags to exactly the paths of f (previous contents reset in O(size of previous)).
void FlowEngine::load(const VertexFlow& f, Scratch& s) const {
    s.reset_flags();
    ensure_arc_capacity(s, g_);
    if (f.v < 0 || f.v >= g_.n()) throw std::invalid_argument("FlowEngine::load: flow has no source vertex");
    for (const auto& path : f.paths) {
        for (int a : path) {
            if (a < 0 || a >= g_.num_arc_ids()) throw std::logic_error("FlowEngine::load: bad arc id in path");
            s.set_arc(a);
            s.set_through(g_.arc(a).head);
        }
    }
    s.loaded_v = f.v;
}

// Rebuild the explicit path list of f from the flag state after an augmentation: from v follow, at every
// vertex, its unique out-arc with flow 1 until a terminal. Flow conservation with unit split capacities
// guarantees uniqueness; any flow left on cycles (not reachable from v) is discarded by re-setting the
// flags from the rebuilt paths, which keeps the same value. Paths come out vertex-disjoint and end at
// distinct terminals [Def 3.2]; verified under GLCORE_DEBUG_ASSERTS.
void FlowEngine::rebuild_paths(VertexFlow& f, Scratch& s) const {
    const int v = f.v;
    f.paths.clear();
    f.path_terminal.clear();
    for (int a0 : g_.out_arcs(v)) {
        if (!s.arc_flow[a0]) continue;
        std::vector<int> path;
        path.push_back(a0);
        int x = g_.arc(a0).head;
        int steps = 0;
        while (!g_.is_terminal(x)) {
            int nxt = -1;
            for (int b : g_.out_arcs(x)) {
                if (!s.arc_flow[b]) continue;
                if (nxt >= 0) throw std::logic_error("FlowEngine: two flow arcs leave vertex " + std::to_string(x));
                nxt = b;
            }
            if (nxt < 0) throw std::logic_error("FlowEngine: flow conservation violated at vertex " + std::to_string(x));
            path.push_back(nxt);
            x = g_.arc(nxt).head;
            if (++steps > g_.n()) throw std::logic_error("FlowEngine: cycle while walking flow paths");
        }
        f.paths.push_back(std::move(path));
        f.path_terminal.push_back(x);
    }
    if ((int)f.paths.size() != f.kappa)
        throw std::logic_error("FlowEngine: path count " + std::to_string(f.paths.size()) + " != kappa " +
                               std::to_string(f.kappa));
    // Re-derive the flags from the paths (drops cycle flow, keeps touched lists tight).
    int lv = s.loaded_v;
    s.reset_flags();
    for (const auto& path : f.paths)
        for (int a : path) { s.set_arc(a); s.set_through(g_.arc(a).head); }
    s.loaded_v = lv;
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
bool FlowEngine::augment_once(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden) const {
    require_loaded(s, f, "augment_once");
    ensure_arc_capacity(s, g_);
    const int v = f.v;
    mark_forbidden(s, forbidden, g_);
    s.next_gen();
    s.queue.clear();
    const int src = node_out(v);
    s.mark(src, -1, -1);
    s.queue.push_back(src);
    int target = -1;
    for (size_t head = 0; head < s.queue.size() && target < 0; ++head) {
        const int node = s.queue[head];
        const int x = node_vertex(node);
        if (node_is_out(node)) {
            for (int a : g_.out_arcs(x)) {
                if (s.forbidden[a] || s.arc_flow[a]) continue;
                const int y = g_.arc(a).head;
                if (y == v) continue;  // never route flow into the source
                const int yin = node_in(y);
                if (s.visited(yin)) continue;
                s.mark(yin, node, a);
                if (!s.through[y] && g_.is_terminal(y)) { target = yin; break; }
                s.queue.push_back(yin);
            }
            if (target >= 0) break;
            if (x != v && s.through[x]) {
                const int xin = node_in(x);
                if (!s.visited(xin)) { s.mark(xin, node, -1); s.queue.push_back(xin); }
            }
        } else {
            if (!s.through[x]) {
                const int xout = node_out(x);
                if (!s.visited(xout)) { s.mark(xout, node, -1); s.queue.push_back(xout); }
            } else {
                for (int a : g_.in_arcs(x)) {
                    if (!s.arc_flow[a] || s.forbidden[a]) continue;
                    const int wout = node_out(g_.arc(a).tail);
                    if (!s.visited(wout)) { s.mark(wout, node, a); s.queue.push_back(wout); }
                }
            }
        }
    }
    unmark_forbidden(s, forbidden);
    if (target < 0) return false;
    // Augment: flip the flags along the tree path target -> src.
    for (int node = target; node != src;) {
        const int pn = s.parent_node[node];
        const int pa = s.parent_arc[node];
        const int x = node_vertex(node);
        if (pa >= 0) {
            if (!node_is_out(node)) s.set_arc(pa);   // forward arc x_out -> y_in
            else s.arc_flow[pa] = 0;                 // reverse arc x_in -> w_out
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
void FlowEngine::compute_max_flow(int v, VertexFlow& f, Scratch& s, bool compute_cut_) const {
    if (v < 0 || v >= g_.n() || !g_.live(v) || g_.is_terminal(v))
        throw std::invalid_argument("FlowEngine::compute_max_flow: " + std::to_string(v) + " is not a live non-terminal");
    f = VertexFlow{};
    f.v = v;
    f.version = g_.version();
    f.ess = TermSet(g_.k0());
    load(f, s);
    static const std::vector<int> none;
    int guard = 0;
    while (augment_once(f, s, none)) {
        if (++guard > g_.k()) throw std::logic_error("FlowEngine: more augmentations than terminals");
    }
    if (compute_cut_) compute_cut(f, s, none);
}

// [Prop 4.2] tightest minimum cut from a maximum flow: Reach = split nodes that can reach z in the residual
// graph (reverse BFS from every t_out, since (t_out,z) has capacity K > flow). Reverse residual rules:
//   y_out in Reach: y_in if through[y] == 0 (residual on the split arc);   w_in for out-arcs (y,w) with flow 1;
//   y_in  in Reach: y_out if through[y] == 1 (reverse of the split arc);   x_out for EVERY alive non-forbidden
//                   in-arc (x,y) (capacity K is never saturated).
// Sides: L if x_in reached, S if only x_out reached, R otherwise; v is R (v_out reachable would make s
// reachable, contradicting maximality — we check this and throw), |S| = kappa, T ⊆ L ∪ S. Ess = T ∩ S [Lem 4.1].
void FlowEngine::compute_cut(VertexFlow& f, Scratch& s, const std::vector<int>& forbidden) const {
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
                if (!s.arc_flow[a] || s.forbidden[a]) continue;
                const int win = node_in(g_.arc(a).head);
                if (!s.visited(win)) { s.mark(win, node, a); s.queue.push_back(win); }
            }
        } else {
            if (s.through[x]) {
                const int xout = node_out(x);
                if (!s.visited(xout)) { s.mark(xout, node, -1); s.queue.push_back(xout); }
            }
            for (int a : g_.in_arcs(x)) {
                if (s.forbidden[a]) continue;
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
    f.side.assign(n, Side::R);
    int cnt_s = 0;
    for (int x = 0; x < n; ++x) {
        if (!g_.live(x)) continue;
        const bool rin = s.visited(node_in(x)), rout = s.visited(node_out(x));
        if (rin) {
            if (!rout) throw std::logic_error("FlowEngine::compute_cut: x_in reached but x_out not [Prop 4.2]");
            f.side[x] = Side::L;
        } else if (rout) {
            f.side[x] = Side::S;
            ++cnt_s;
        }
    }
    if (cnt_s != f.kappa)
        throw std::logic_error("FlowEngine::compute_cut: |S| = " + std::to_string(cnt_s) + " != kappa = " +
                               std::to_string(f.kappa) + " for vertex " + std::to_string(v));
    f.ess = TermSet(g_.k0());
    for (int t : g_.terminals()) {
        if (f.side[t] == Side::R) throw std::logic_error("FlowEngine::compute_cut: terminal on the R side");
        if (f.side[t] == Side::S) f.ess.set(g_.terminal_index(t));
    }
    f.cut_exact = true;
}

// Clear the flags of path i and drop it from f (O(length)).
void FlowEngine::remove_path(VertexFlow& f, size_t i, Scratch& s) const {
    for (int a : f.paths[i]) {
        s.arc_flow[a] = 0;
        s.through[g_.arc(a).head] = 0;
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

bool FlowEngine::can_translate_after_contraction(const VertexFlow& f, int p, int t) const {
    if (f.v == p) return false;
    for (const auto& path : f.paths) {
        for (size_t j = 0; j < path.size(); ++j) {
            if (g_.arc(path[j]).head != p) continue;
            return j + 1 < path.size() && g_.arc(path[j + 1]).head == t;
        }
    }
    return true;  // no path through p: nothing to translate
}

// O3: after Graph::contract(p, t) the path ... x -(x,p)-> p -(p,t)-> t of f (if any) becomes ... x -(x,t)-> t,
// where (x,t) is the redirected arc (new id) or the pre-existing merged arc (old id). The other paths are
// untouched; kappa and Ess are unchanged (paper_notes §13.2). If f.v == p the flow is meaningless (p is
// dead) and the call is a no-op: callers drop it.
void FlowEngine::translate_after_contraction(VertexFlow& f, int p, int t) const {
    if (f.v == p) return;
    for (size_t i = 0; i < f.paths.size(); ++i) {
        auto& path = f.paths[i];
        for (size_t j = 0; j < path.size(); ++j) {
            if (g_.arc(path[j]).head != p) continue;
            if (j + 1 >= path.size() || g_.arc(path[j + 1]).head != t)
                throw std::logic_error("FlowEngine::translate_after_contraction: the path through " + std::to_string(p) +
                                       " does not continue to " + std::to_string(t) + " (contraction requires d^+(p) = 1 on flow arcs)");
            const int x = g_.arc(path[j]).tail;
            const int a_new = g_.find_arc(x, t);
            if (a_new < 0)
                throw std::logic_error("FlowEngine::translate_after_contraction: redirected arc (" + std::to_string(x) + "," +
                                       std::to_string(t) + ") not found");
            path[j] = a_new;
            path.erase(path.begin() + (std::ptrdiff_t)j + 1);
            if (f.path_terminal[i] != t) throw std::logic_error("FlowEngine::translate_after_contraction: path terminal mismatch");
            return;  // paths are vertex-disjoint: at most one passes through p
        }
    }
}

}  // namespace glcore
