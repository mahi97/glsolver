// Bipartite matching terminals -> pre-terminals [Lem 7.8] and the minimal Hall-deficient set [Lem 7.6]
// (docs/paper_notes.md §7.2, §8). Hopcroft–Karp on the bipartite graph {t ∈ S} × {in-neighbours of S};
// every in-neighbour of a terminal is a pre-terminal because terminals have no out-arcs (§2).
#include "matching.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace glcore {
namespace {

// Hopcroft–Karp with an iterative layered DFS (no recursion, so |S| may be large). Left = positions of S,
// right = compact ids of the pre-terminals. O(E sqrt(V)) on the bipartite graph of size E = Σ_{t∈S} in_deg(t).
struct HopcroftKarp {
    int L = 0, R = 0;
    std::vector<std::vector<int>> adj;  // left -> right compact ids
    std::vector<int> matchL, matchR, dist, it;
    static constexpr int INF = std::numeric_limits<int>::max();

    // BFS layering from the free left vertices; true if some free right vertex is reachable.
    bool bfs() {
        std::vector<int> queue;
        queue.reserve(L);
        for (int u = 0; u < L; ++u) {
            if (matchL[u] < 0) {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = INF;
            }
        }
        bool found = false;
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const int u = queue[qi];
            for (int r : adj[u]) {
                const int w = matchR[r];
                if (w < 0) {
                    found = true;
                } else if (dist[w] == INF) {
                    dist[w] = dist[u] + 1;
                    queue.push_back(w);
                }
            }
        }
        return found;
    }

    // Iterative DFS along the layering from `root`; augments and returns true if a free right vertex is found.
    bool dfs(int root) {
        std::vector<int> stack;
        stack.push_back(root);
        while (!stack.empty()) {
            const int u = stack.back();
            if (it[u] == (int)adj[u].size()) {
                dist[u] = INF;  // dead end for this phase
                stack.pop_back();
                continue;
            }
            const int r = adj[u][it[u]++];
            const int w = matchR[r];
            if (w < 0) {
                // augment along the stack: each left vertex on it advanced through adj[.][it[.]-1]
                for (int j = (int)stack.size() - 1; j >= 0; --j) {
                    const int uu = stack[j];
                    const int rr = adj[uu][it[uu] - 1];
                    matchL[uu] = rr;
                    matchR[rr] = uu;
                }
                return true;
            }
            if (dist[w] == dist[u] + 1) stack.push_back(w);
        }
        return false;
    }

    int run() {
        matchL.assign(L, -1);
        matchR.assign(R, -1);
        dist.assign(L, INF);
        it.assign(L, 0);
        int size = 0;
        while (bfs()) {
            std::fill(it.begin(), it.end(), 0);
            for (int u = 0; u < L; ++u)
                if (matchL[u] < 0 && dfs(u)) ++size;
        }
        return size;
    }
};

}  // namespace

// [Lem 7.8] matching from the terminals in S to distinct pre-terminals p with (p, t) ∈ E, saturating S.
// Returns matched pre-terminal per position of S, or an empty vector if S is not saturable.
std::vector<int> saturating_matching(const Graph& g, const std::vector<int>& S) {
    HopcroftKarp hk;
    hk.L = (int)S.size();
    hk.adj.resize(hk.L);
    std::unordered_map<int, int> compact;  // pre-terminal vertex id -> right id (O(size), not O(n))
    std::vector<int> right_vertex;
    for (int i = 0; i < hk.L; ++i) {
        const int t = S[i];
        if (t < 0 || t >= g.n() || !g.is_terminal(t))
            throw std::invalid_argument("saturating_matching: " + std::to_string(t) + " is not a live terminal");
        for (int a : g.in_arcs(t)) {
            const int p = g.arc(a).tail;
            auto ins = compact.emplace(p, (int)right_vertex.size());
            if (ins.second) right_vertex.push_back(p);
            hk.adj[i].push_back(ins.first->second);
        }
    }
    hk.R = (int)right_vertex.size();
    const int size = hk.run();
    if (size < hk.L) return {};
    std::vector<int> out(hk.L);
    for (int i = 0; i < hk.L; ++i) out[i] = right_vertex[hk.matchL[i]];
    return out;
}

// [Lem 7.8] saturating matching of all live terminals.
std::vector<int> saturating_matching(const Graph& g) { return saturating_matching(g, g.terminals()); }

// [Lem 7.6] inclusion-minimal nonempty S ⊆ T with no saturating matching (paper's procedure, paper_notes §8):
// start with S = T; while some S \ {t} is still unsaturable, drop t and restart the scan. Returns empty if T
// is saturable. Asserts |PT(G, S)| = |S| - 1 [Lem 7.6] on the result.
//
// This is literally glref.matching.minimal_hall_deficient_set_counted (same terminal order, same
// saturability answers, same restart discipline), so the two agree on every instance. The paper's singleton
// shortcut for a terminal without pre-terminal neighbour (paper_notes §8) is deliberately NOT applied: it
// would return a different (equally valid) minimal deficient set whenever another deficient set is scanned
// first, and RoundAndRemove [Alg 4] step (iv) would then pick different parts than the reference (the scan
// may drop an in-degree-0 terminal when the remaining set is still deficient, and end elsewhere).
std::vector<int> minimal_hall_deficient_set(const Graph& g) {
    std::vector<int> S = g.terminals();
    if (S.empty()) return {};
    if (!saturating_matching(g, S).empty()) return {};
    bool changed = true;
    while (changed) {
        changed = false;
        if (S.size() <= 1) break;  // the empty set is always saturable, so a singleton is minimal
        for (size_t j = 0; j < S.size(); ++j) {
            std::vector<int> S2;
            S2.reserve(S.size() - 1);
            for (size_t l = 0; l < S.size(); ++l)
                if (l != j) S2.push_back(S[l]);
            if (saturating_matching(g, S2).empty()) {
                S.swap(S2);
                changed = true;
                break;
            }
        }
    }
    const std::vector<int>& result = S;
    if (result.empty()) throw std::logic_error("minimal_hall_deficient_set: the empty set is always saturable");
    const size_t pt = g.pre_terminals_of(result).size();
    if (pt != result.size() - 1)
        throw std::logic_error("minimal_hall_deficient_set: [Lem 7.6] violated: |PT(G,S)| = " + std::to_string(pt) + " but |S| - 1 = " +
                               std::to_string(result.size() - 1));
    return result;
}

}  // namespace glcore
