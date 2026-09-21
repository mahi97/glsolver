// Mutable simple digraph with terminals (docs/paper_notes.md §2, §13.4).
#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace glcore {

// Arc ids are stable for the lifetime of a Graph: deleting an arc marks it dead; contraction creates
// new arc ids for redirected arcs (u,p)->(u,t). Adjacency lists hold arc ids.
struct Arc {
    int tail, head;
    bool alive;
    int orig_head;  // for arcs whose head is a terminal: the original head (paper_notes §13.4); else == head
};

class Graph {
public:
    Graph(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals);

    int n() const { return n_; }
    int k() const { return (int)terminals_.size(); }           // current number of terminals
    int k0() const { return (int)term_index_.size() ? k0_ : 0; } // original number of terminals
    const std::vector<int>& terminals() const { return terminals_; }  // live terminals, original order
    int terminal_index(int v) const { return term_index_[v]; }      // original index in [0,k0) or -1
    bool is_terminal(int v) const { return term_index_[v] >= 0 && live_[v]; }
    bool live(int v) const { return live_[v]; }
    const std::vector<char>& live_mask() const { return live_; }
    int num_live_nonterminals() const { return live_nonterminals_; }
    int num_live_arcs() const { return live_arcs_; }
    uint64_t version() const { return version_; }  // increments on every mutation

    const Arc& arc(int a) const { return arcs_[a]; }
    int num_arc_ids() const { return (int)arcs_.size(); }
    const std::vector<int>& out_arcs(int v) const { return out_[v]; }  // alive arc ids only
    const std::vector<int>& in_arcs(int v) const { return in_[v]; }
    int out_degree(int v) const { return (int)out_[v].size(); }
    int in_degree(int v) const { return (int)in_[v].size(); }
    int find_arc(int u, int v) const;          // alive arc id or -1
    bool has_arc(int u, int v) const { return find_arc(u, v) >= 0; }

    bool is_pre_terminal(int v) const;         // [Def 2.1]
    int terminal_out_arc(int v) const;         // some alive arc (v,t) with t terminal, or -1
    std::vector<int> pre_terminals() const;    // PT(G,T)
    std::vector<int> pre_terminals_of(const std::vector<int>& S) const;  // PT(G,S)
    std::vector<int> live_nonterminals() const;

    // ---- mutations (paper_notes §2) ----
    void delete_arc(int a);                    // by arc id
    // Contract pre-terminal p into terminal t along the alive arc a=(p,t) [Def 2.1].
    // Returns the parent (orig_head of a). Redirected arcs get new ids; duplicates are merged.
    // Newly created arc ids are appended to `created` (may be null).
    int contract(int p, int t, std::vector<int>* created = nullptr, std::vector<int>* deleted = nullptr);
    void remove_terminal(int t);               // operation (i)
    void remove_vertex(int v);                 // generic (rounding)

    void check_invariants() const;             // debug; throws std::logic_error

private:
    int n_;
    int k0_ = 0;
    std::vector<int> terminals_;
    std::vector<int> term_index_;
    std::vector<char> live_;
    std::vector<Arc> arcs_;
    std::vector<std::vector<int>> out_, in_;
    std::unordered_map<uint64_t, int> arc_lookup_;  // key(u,v) -> alive arc id
    int live_nonterminals_ = 0;
    int live_arcs_ = 0;
    uint64_t version_ = 0;

    static uint64_t key(int u, int v) { return (uint64_t(uint32_t(u)) << 32) | uint32_t(v); }
    int add_arc(int u, int v, int orig_head);
    void erase_from(std::vector<int>& lst, int a);
};

}  // namespace glcore
