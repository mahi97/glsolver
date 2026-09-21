// Mutable simple digraph with terminals (docs/paper_notes.md §2 conventions, [Def 2.1] contraction,
// §13.4 orig_head bookkeeping).
//
// Representation: arcs live in `arcs_` under stable ids; `out_[v]` / `in_[v]` hold the ids of the alive
// arcs only (unordered, swap-remove in O(1) through the per-arc position tables `pos_out_` / `pos_in_`);
// `arc_lookup_` maps (u,v) -> alive arc id for O(1) membership (needed by the duplicate merge of [Def 2.1]).
// Deleted vertices are tombstoned (`live_`), deleted arcs keep their id with alive == false so that
// external per-arc indices (flows, user lists) never need renumbering.
#include "graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace glcore {

namespace {
std::string arc_str(int u, int v) { return "(" + std::to_string(u) + "," + std::to_string(v) + ")"; }
}  // namespace

// [§2] Load an arc list: self-loops, arcs leaving a terminal and duplicate arcs are dropped (first
// occurrence kept). Arc ids are 0..m-1 in input order after filtering; orig_head == head initially (§13.4).
Graph::Graph(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals)
    : n_(n), terminals_(terminals) {
    if (n < 0) throw std::invalid_argument("Graph: n must be nonnegative");
    term_index_.assign(n, -1);
    live_.assign(n, 1);
    out_.assign(n, {});
    in_.assign(n, {});
    k0_ = (int)terminals.size();
    for (int i = 0; i < k0_; ++i) {
        int t = terminals[i];
        if (t < 0 || t >= n) throw std::invalid_argument("Graph: terminal " + std::to_string(t) + " out of range");
        if (term_index_[t] >= 0) throw std::invalid_argument("Graph: duplicate terminal " + std::to_string(t));
        term_index_[t] = i;
    }
    live_nonterminals_ = n - k0_;
    arcs_.reserve(arcs.size());
    for (const auto& uv : arcs) {
        int u = uv.first, v = uv.second;
        if (u < 0 || u >= n || v < 0 || v >= n)
            throw std::invalid_argument("Graph: arc " + arc_str(u, v) + " out of range");
        if (u == v) continue;                 // no self-loops
        if (term_index_[u] >= 0) continue;    // terminals have no out-arcs
        if (arc_lookup_.count(key(u, v))) continue;  // no parallel arcs (keep first)
        add_arc(u, v, v);
    }
}

int Graph::find_arc(int u, int v) const {
    auto it = arc_lookup_.find(key(u, v));
    return it == arc_lookup_.end() ? -1 : it->second;
}

// Append a new alive arc (u,v) and register it in the adjacency lists and the lookup table.
int Graph::add_arc(int u, int v, int orig_head) {
    int id = (int)arcs_.size();
    arcs_.push_back(Arc{u, v, true, orig_head});
    pos_out_.push_back((int)out_[u].size());
    out_[u].push_back(id);
    pos_in_.push_back((int)in_[v].size());
    in_[v].push_back(id);
    arc_lookup_.emplace(key(u, v), id);
    ++live_arcs_;
    return id;
}

// Swap-remove arc id a from an adjacency list (O(1) with the position tables; deterministic).
void Graph::erase_from(std::vector<int>& lst, int a) {
    bool is_out = (&lst == &out_[arcs_[a].tail]);
    std::vector<int>& pos = is_out ? pos_out_ : pos_in_;
    int p = pos[a];
    if (p < 0 || p >= (int)lst.size() || lst[p] != a)
        throw std::logic_error("Graph::erase_from: position table corrupt for arc " + std::to_string(a));
    int last = lst.back();
    lst[p] = last;
    pos[last] = p;
    lst.pop_back();
    pos[a] = -1;
}

// Mark arc a dead and unregister it; `erase_out` / `erase_in` say which adjacency lists to update (a
// caller clearing a whole list at once passes false for it). Does not touch the version counter.
void Graph::kill_arc(int a, bool erase_out, bool erase_in) {
    Arc& e = arcs_[a];
    if (!e.alive) throw std::logic_error("Graph::kill_arc: arc " + std::to_string(a) + " is already dead");
    if (erase_out) erase_from(out_[e.tail], a); else pos_out_[a] = -1;
    if (erase_in) erase_from(in_[e.head], a); else pos_in_[a] = -1;
    arc_lookup_.erase(key(e.tail, e.head));
    e.alive = false;
    --live_arcs_;
}

// Arc deletion (operation (iii) of [Alg 1] / the O1–O7 batch deletions).
void Graph::delete_arc(int a) {
    if (a < 0 || a >= (int)arcs_.size()) throw std::invalid_argument("Graph::delete_arc: bad arc id " + std::to_string(a));
    if (!arcs_[a].alive) throw std::invalid_argument("Graph::delete_arc: arc " + std::to_string(a) + " is not alive");
    kill_arc(a, true, true);
    ++version_;
}

// [Def 2.1] Contract pre-terminal p into terminal t along the alive arc (p,t). Every alive in-arc (u,p) is
// killed and, unless (u,t) already exists (duplicate merge: the existing arc and its orig_head are kept),
// replaced by a NEW arc (u,t) with orig_head = p (§13.4). All out-arcs of p are killed, p becomes dead.
// Returns parent[p] = orig_head of the contracted arc (§13.4).
int Graph::contract(int p, int t, std::vector<int>* created, std::vector<int>* deleted) {
    if (p < 0 || p >= n_ || !live_[p] || term_index_[p] >= 0)
        throw std::invalid_argument("Graph::contract: " + std::to_string(p) + " is not a live non-terminal");
    if (t < 0 || t >= n_ || !is_terminal(t))
        throw std::invalid_argument("Graph::contract: " + std::to_string(t) + " is not a live terminal");
    int a = find_arc(p, t);
    if (a < 0) throw std::invalid_argument("Graph::contract: arc " + arc_str(p, t) + " is required");
    int parent = arcs_[a].orig_head;
    // Redirect incoming arcs (u,p) -> (u,t). List order is deterministic, hence so are the new ids.
    std::vector<int> ins = in_[p];
    for (int b : ins) {
        int u = arcs_[b].tail;
        kill_arc(b, true, false);
        if (deleted) deleted->push_back(b);
        if (arc_lookup_.find(key(u, t)) == arc_lookup_.end()) {
            int c = add_arc(u, t, p);
            if (created) created->push_back(c);
        }
        // else: duplicate merged, keep the existing (u,t) and its orig_head
    }
    in_[p].clear();
    // Delete outgoing arcs of p (including (p,t)).
    std::vector<int> outs = out_[p];
    for (int b : outs) {
        kill_arc(b, false, true);
        if (deleted) deleted->push_back(b);
    }
    out_[p].clear();
    live_[p] = 0;
    --live_nonterminals_;
    ++version_;
    last_contraction_ = ContractionRecord{p, t, (int)outs.size(), version_};
    return parent;
}

// Operation (i) of [Alg 1]/[Alg 3]: delete terminal t and its in-arcs [Lem 7.2]. The original terminal
// index (terminal_index) is retained so bitsets over original indices stay meaningful.
void Graph::remove_terminal(int t) {
    if (t < 0 || t >= n_ || !is_terminal(t))
        throw std::invalid_argument("Graph::remove_terminal: " + std::to_string(t) + " is not a live terminal");
    if (!out_[t].empty()) throw std::logic_error("Graph::remove_terminal: terminal has out-arcs");
    std::vector<int> ins = in_[t];
    for (int b : ins) kill_arc(b, true, false);
    in_[t].clear();
    live_[t] = 0;
    terminals_.erase(std::find(terminals_.begin(), terminals_.end(), t));
    ++version_;
}

// Generic vertex deletion (RoundAndRemove [Alg 4]): kills all incident arcs, tombstones v.
void Graph::remove_vertex(int v) {
    if (v < 0 || v >= n_ || !live_[v])
        throw std::invalid_argument("Graph::remove_vertex: " + std::to_string(v) + " is not live");
    std::vector<int> outs = out_[v];
    for (int b : outs) kill_arc(b, false, true);
    out_[v].clear();
    std::vector<int> ins = in_[v];
    for (int b : ins) kill_arc(b, true, false);
    in_[v].clear();
    live_[v] = 0;
    if (term_index_[v] >= 0) {
        terminals_.erase(std::find(terminals_.begin(), terminals_.end(), v));
    } else {
        --live_nonterminals_;
    }
    ++version_;
}

// [Def 2.1] pre-terminal: live non-terminal with an arc into a live terminal.
bool Graph::is_pre_terminal(int v) const {
    return v >= 0 && v < n_ && live_[v] && term_index_[v] < 0 && terminal_out_arc(v) >= 0;
}

int Graph::terminal_out_arc(int v) const {
    for (int a : out_[v])
        if (is_terminal(arcs_[a].head)) return a;
    return -1;
}

// PT(G,T) in increasing vertex order.
std::vector<int> Graph::pre_terminals() const {
    std::vector<int> r;
    for (int v = 0; v < n_; ++v)
        if (is_pre_terminal(v)) r.push_back(v);
    return r;
}

// PT(G,S) [Def 2.1]: live non-terminals with an arc into S.
std::vector<int> Graph::pre_terminals_of(const std::vector<int>& S) const {
    std::vector<char> in_s(n_, 0);
    for (int t : S)
        if (t >= 0 && t < n_) in_s[t] = 1;
    std::vector<int> r;
    for (int v = 0; v < n_; ++v) {
        if (!live_[v] || term_index_[v] >= 0) continue;
        for (int a : out_[v])
            if (in_s[arcs_[a].head]) { r.push_back(v); break; }
    }
    return r;
}

std::vector<int> Graph::live_nonterminals() const {
    std::vector<int> r;
    r.reserve(live_nonterminals_);
    for (int v = 0; v < n_; ++v)
        if (live_[v] && term_index_[v] < 0) r.push_back(v);
    return r;
}

// Debug: adjacency / lookup / counter consistency and the conventions of §2. Throws std::logic_error.
void Graph::check_invariants() const {
    auto fail = [](const std::string& msg) { throw std::logic_error("Graph invariant violated: " + msg); };
    int alive = 0, lnt = 0;
    std::vector<int> seen_out(arcs_.size(), 0), seen_in(arcs_.size(), 0);
    for (int v = 0; v < n_; ++v) {
        if (!live_[v]) {
            if (!out_[v].empty() || !in_[v].empty()) fail("dead vertex " + std::to_string(v) + " has arcs");
            continue;
        }
        if (term_index_[v] < 0) ++lnt;
        if (term_index_[v] >= 0 && std::find(terminals_.begin(), terminals_.end(), v) == terminals_.end())
            fail("live terminal " + std::to_string(v) + " missing from terminals_");
        for (size_t i = 0; i < out_[v].size(); ++i) {
            int a = out_[v][i];
            if (a < 0 || a >= (int)arcs_.size()) fail("bad arc id in out list");
            const Arc& e = arcs_[a];
            if (!e.alive) fail("dead arc " + std::to_string(a) + " in out list of " + std::to_string(v));
            if (e.tail != v) fail("arc " + std::to_string(a) + " in wrong out list");
            if (pos_out_[a] != (int)i) fail("pos_out_ mismatch for arc " + std::to_string(a));
            if (term_index_[v] >= 0) fail("terminal " + std::to_string(v) + " has an out-arc");
            if (!live_[e.head]) fail("arc " + std::to_string(a) + " enters dead vertex");
            if (e.head == e.tail) fail("self-loop");
            if (is_terminal(e.head)) {
                if (e.orig_head < 0 || e.orig_head >= n_) fail("orig_head out of range");
            } else if (e.orig_head != e.head) {
                fail("orig_head != head for arc " + std::to_string(a) + " not entering a terminal");
            }
            auto it = arc_lookup_.find(key(e.tail, e.head));
            if (it == arc_lookup_.end() || it->second != a) fail("lookup mismatch for arc " + std::to_string(a));
            if (++seen_out[a] > 1) fail("arc listed twice in out lists");
        }
        for (size_t i = 0; i < in_[v].size(); ++i) {
            int a = in_[v][i];
            if (a < 0 || a >= (int)arcs_.size()) fail("bad arc id in in list");
            const Arc& e = arcs_[a];
            if (!e.alive) fail("dead arc " + std::to_string(a) + " in in list of " + std::to_string(v));
            if (e.head != v) fail("arc " + std::to_string(a) + " in wrong in list");
            if (pos_in_[a] != (int)i) fail("pos_in_ mismatch for arc " + std::to_string(a));
            if (++seen_in[a] > 1) fail("arc listed twice in in lists");
        }
    }
    for (size_t a = 0; a < arcs_.size(); ++a) {
        if (arcs_[a].alive) {
            ++alive;
            if (!seen_out[a] || !seen_in[a]) fail("alive arc " + std::to_string(a) + " missing from a list");
        } else if (seen_out[a] || seen_in[a]) {
            fail("dead arc " + std::to_string(a) + " present in a list");
        }
    }
    if (alive != live_arcs_) fail("live_arcs_ counter mismatch");
    if ((int)arc_lookup_.size() != live_arcs_) fail("lookup size mismatch");
    if (lnt != live_nonterminals_) fail("live_nonterminals_ counter mismatch");
    for (int t : terminals_) {
        if (t < 0 || t >= n_ || !live_[t] || term_index_[t] < 0) fail("terminals_ holds a non-live terminal");
        if (std::count(terminals_.begin(), terminals_.end(), t) != 1) fail("terminals_ has a duplicate");
    }
}

}  // namespace glcore
