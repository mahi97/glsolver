// Adversarial probe for the C1 deletion-aware ("avoid") routing of glcore::FlowEngine
// (RESEARCH_NOTES P4/E5, src/glcore/flow.cpp search_target<Penalized>).
//
// The pybind11 surface exposes no way to install a penalty array (neither `_core.tightest_cut` nor
// `_core.EssentialOracle` calls FlowEngine::set_penalties), so the 0-1 search can only be reached from
// Python end-to-end through the solvers.  This driver links flow.cpp + graph.cpp directly and attacks the
// search itself:
//
//   K1  kappa / Ess / L-S-R sides of a penalized maximum flow equal those of the plain BFS maximum flow.
//   K2  every stored path family is a family of VERTEX-DISJOINT simple v->T paths ending at distinct
//       terminals, made of alive arcs, of size exactly kappa (the invariant a 0-1 deque bug would break).
//   K3  augment_once returns true with penalties exactly when it returns true without them, from the SAME
//       loaded flow state (it never returns a non-augmenting path, and never misses an augmenting one).
//   K4  from an EMPTY flow the first augmenting path has minimum total penalty: its penalty equals the
//       reference Dijkstra distance from v to T in the digraph with arc weights = penalty
//       (from an empty flow the split residual graph is exactly the digraph).
//   K5  warm start with a forbidden set D: the warm-started kappa/Ess in G \ D equal a from-scratch
//       penalized computation on a copy of the graph with D deleted (same arc ids), and no stored path
//       uses an arc of D.
//   K6  relabelling stress: penalties built so a node is first reached at penalty level 1 and then
//       re-reached at level 0 (the case where a 0-1 search may finalize a node twice).
//
// Build:  g++ -std=c++20 -O2 -I src/glcore tests/cxx/routing_probe.cpp src/glcore/flow.cpp src/glcore/graph.cpp
// Run:    ./routing_probe <seed_lo> <seed_hi>   ->  "OK <checks>" (exit 0) or "FAIL: ..." (exit 1)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "flow.hpp"
#include "graph.hpp"

using namespace glcore;

static long long g_checks = 0;
static long long g_vertices = 0, g_paths_differ = 0, g_positive_pen = 0, g_warm = 0;

struct Failure {
    std::string msg;
};
static void fail(const std::string& m) { throw Failure{m}; }
static void need(bool ok, const std::string& m) {
    ++g_checks;
    if (!ok) fail(m);
}

// ---------------------------------------------------------------------------------------------------
// K2: the stored paths of a VertexFlow are kappa vertex-disjoint simple v->T paths on alive arcs.
// ---------------------------------------------------------------------------------------------------
static void validate_paths(const Graph& g, const VertexFlow& f, const std::vector<int>& forbidden,
                           const std::string& where) {
    need((int)f.paths.size() == f.kappa, where + ": #paths " + std::to_string(f.paths.size()) +
                                             " != kappa " + std::to_string(f.kappa));
    need(f.path_terminal.size() == f.paths.size(), where + ": path_terminal size mismatch");
    std::set<int> forb(forbidden.begin(), forbidden.end());
    std::set<int> used_vertices;   // interior + endpoint vertices, v excluded
    std::set<int> used_terminals;
    std::set<int> used_arcs;
    for (size_t i = 0; i < f.paths.size(); ++i) {
        const auto& p = f.paths[i];
        need(!p.empty(), where + ": empty path");
        int cur = f.v;
        std::set<int> on_this_path{f.v};
        for (size_t j = 0; j < p.size(); ++j) {
            const int a = p[j];
            need(a >= 0 && a < g.num_arc_ids(), where + ": arc id out of range");
            need(g.arc(a).alive, where + ": path uses a dead arc " + std::to_string(a));
            need(!forb.count(a), where + ": path uses a FORBIDDEN arc " + std::to_string(a));
            need(used_arcs.insert(a).second, where + ": arc " + std::to_string(a) + " used by two paths");
            need(g.arc(a).tail == cur, where + ": path is not a walk (arc tail != previous head)");
            cur = g.arc(a).head;
            // no repeated vertex inside one path (a 0-1 search that relabels a finalized node could
            // produce a parent chain with a repeated vertex)
            need(on_this_path.insert(cur).second,
                 where + ": path " + std::to_string(i) + " repeats vertex " + std::to_string(cur));
            const bool last = (j + 1 == p.size());
            need(last == g.is_terminal(cur),
                 where + ": terminal " + std::to_string(cur) + " is " + (last ? "not" : "") + " the last vertex");
            // vertex-disjointness across paths (v is shared by construction)
            need(used_vertices.insert(cur).second,
                 where + ": vertex " + std::to_string(cur) + " is used by two paths");
        }
        need(f.path_terminal[i] == cur, where + ": path_terminal does not match the last vertex");
        need(used_terminals.insert(cur).second, where + ": two paths end at the same terminal");
    }
}

// ---------------------------------------------------------------------------------------------------
// K4: reference minimum total penalty of a v->T path in the digraph (0-1 Dijkstra, independent code).
// ---------------------------------------------------------------------------------------------------
static int ref_min_penalty(const Graph& g, int v, const std::vector<unsigned char>& pen) {
    const int INF = 1 << 29;
    std::vector<int> d(g.n(), INF);
    d[v] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, v});
    int best = INF;
    while (!pq.empty()) {
        auto [dv, x] = pq.top();
        pq.pop();
        if (dv != d[x]) continue;
        if (g.is_terminal(x)) { best = std::min(best, dv); continue; }  // terminals are sinks
        for (int a : g.out_arcs(x)) {
            const int y = g.arc(a).head;
            const int w = a < (int)pen.size() ? (int)pen[a] : 0;
            if (dv + w < d[y]) { d[y] = dv + w; pq.push({d[y], y}); }
        }
    }
    return best;
}

static int path_penalty(const std::vector<int>& path, const std::vector<unsigned char>& pen) {
    int s = 0;
    for (int a : path) s += a < (int)pen.size() ? (int)pen[a] : 0;
    return s;
}

// ---------------------------------------------------------------------------------------------------
// instance generation
// ---------------------------------------------------------------------------------------------------
struct Inst {
    int n;
    std::vector<std::pair<int, int>> arcs;
    std::vector<int> terminals;
};

static Inst make_instance(std::mt19937& rng, int mode) {
    Inst in;
    const int n = (int)(rng() % 10) + 4;          // 4..13
    in.n = n;
    const int k = (int)(rng() % std::min(5, n - 1)) + 1;
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    in.terminals.assign(perm.begin(), perm.begin() + k);
    std::sort(in.terminals.begin(), in.terminals.end());
    std::set<std::pair<int, int>> seen;
    if (mode == 0) {  // dense-ish digraph
        const double p = 0.25 + 0.5 * (double)(rng() % 100) / 100.0;
        for (int u = 0; u < n; ++u)
            for (int w = 0; w < n; ++w)
                if (u != w && (double)(rng() % 1000) / 1000.0 < p) seen.insert({u, w});
    } else if (mode == 1) {  // undirected-as-digraph (both directions)
        const double p = 0.2 + 0.4 * (double)(rng() % 100) / 100.0;
        for (int u = 0; u < n; ++u)
            for (int w = u + 1; w < n; ++w)
                if ((double)(rng() % 1000) / 1000.0 < p) { seen.insert({u, w}); seen.insert({w, u}); }
    } else {  // sparse layered / DAG-ish: long paths, many ties
        for (int u = 0; u + 1 < n; ++u) seen.insert({u, u + 1});
        const int extra = (int)(rng() % (size_t)(2 * n));
        for (int i = 0; i < extra; ++i) {
            int u = (int)(rng() % (size_t)n), w = (int)(rng() % (size_t)n);
            if (u != w) seen.insert({u, w});
        }
    }
    // terminals are sinks in these tests only by convention: keep their out-arcs, the engine ignores them
    in.arcs.assign(seen.begin(), seen.end());
    return in;
}

static std::vector<unsigned char> make_penalties(const Graph& g, std::mt19937& rng, int mode) {
    std::vector<unsigned char> pen(g.num_arc_ids(), 0);
    if (mode == 0) return pen;                                   // all zero: must be bit-identical to BFS
    if (mode == 1) { for (auto& x : pen) x = 1; return pen; }     // all one: every path equally penalized
    if (mode == 2) {                                             // random
        for (auto& x : pen) x = (unsigned char)(rng() % 2);
        return pen;
    }
    if (mode == 3) {  // the solver's rule: 1 on the out-arcs of a pre-terminal that do not enter a terminal
        for (int a = 0; a < g.num_arc_ids(); ++a) {
            if (!g.arc(a).alive) continue;
            const int u = g.arc(a).tail;
            if (g.is_pre_terminal(u) && !g.is_terminal(g.arc(a).head)) pen[a] = 1;
        }
        return pen;
    }
    // mode 4 (K6): relabelling stress.  Penalize every arc leaving the low-numbered half of the vertices
    // and nothing else, so the first (cheap) frontier reaches the high half late and many nodes that were
    // first labelled at level 1 are re-reached at level 0.
    for (int a = 0; a < g.num_arc_ids(); ++a)
        if (g.arc(a).alive && g.arc(a).tail * 2 < g.n()) pen[a] = 1;
    return pen;
}

// ---------------------------------------------------------------------------------------------------
static void run_seed(unsigned seed) {
    std::mt19937 rng(seed);
    const Inst in = make_instance(rng, (int)(seed % 3));
    Graph g(in.n, in.arcs, in.terminals);
    const std::vector<unsigned char> pen = make_penalties(g, rng, (int)(seed % 5));

    FlowEngine plain(g);
    FlowEngine pnl(g);
    pnl.set_penalties(&pen);
    auto sp = plain.make_scratch();
    auto sq = pnl.make_scratch();
    static const std::vector<int> none;

    for (int v = 0; v < g.n(); ++v) {
        if (!g.live(v) || g.is_terminal(v)) continue;
        const std::string at = "seed " + std::to_string(seed) + " v " + std::to_string(v);

        ++g_vertices;
        VertexFlow f0, f1;
        plain.compute_max_flow(v, f0, *sp, true, true);
        pnl.compute_max_flow(v, f1, *sq, true, true);

        // K1
        need(f0.kappa == f1.kappa, at + ": kappa " + std::to_string(f1.kappa) + " != BFS " + std::to_string(f0.kappa));
        need(f0.ess == f1.ess, at + ": Ess differs between the two routings");
        need(f0.side == f1.side, at + ": the L/S/R sides differ between the two routings");
        if (f0.paths != f1.paths) ++g_paths_differ;
        // K2
        validate_paths(g, f0, none, at + " [bfs]");
        validate_paths(g, f1, none, at + " [avoid]");

        // K4: from an empty flow the first augmenting path must have minimum total penalty
        {
            VertexFlow e;
            e.v = v;
            e.kappa = 0;
            e.version = g.version();
            pnl.load(e, *sq);
            const bool got = pnl.augment_once(e, *sq, none);
            const int ref = ref_min_penalty(g, v, pen);
            need(got == (ref < (1 << 29)), at + ": augment_once disagrees with reachability of T");
            if (got) {
                validate_paths(g, e, none, at + " [first augmentation]");
                if (ref > 0) ++g_positive_pen;
                need(path_penalty(e.paths[0], pen) == ref,
                     at + ": first penalized path has penalty " + std::to_string(path_penalty(e.paths[0], pen)) +
                         ", reference minimum " + std::to_string(ref));
            }
        }

        // K3: from the SAME state the two searches agree on whether an augmenting path exists.
        {
            for (int trim = 0; trim < (int)f1.paths.size(); ++trim) {
                VertexFlow a = f1, b = f1;
                // drop the first `trim + 1` paths from both copies (identical state)
                for (int i = 0; i <= trim; ++i) {
                    a.paths.erase(a.paths.begin());
                    a.path_terminal.erase(a.path_terminal.begin());
                    b.paths.erase(b.paths.begin());
                    b.path_terminal.erase(b.path_terminal.begin());
                }
                a.kappa = (int)a.paths.size();
                b.kappa = (int)b.paths.size();
                plain.load(a, *sp);
                const bool ra = plain.augment_once(a, *sp, none);
                pnl.load(b, *sq);
                const bool rb = pnl.augment_once(b, *sq, none);
                need(ra == rb, at + ": augment_once returns " + std::to_string(rb) + " with penalties and " +
                                   std::to_string(ra) + " without, from the same state");
                if (rb) validate_paths(g, b, none, at + " [warm augment]");
                if (ra) validate_paths(g, a, none, at + " [warm augment bfs]");
            }
        }

        // K5: warm start around a forbidden set D, compared with a from-scratch run on G \ D.
        if (!f1.paths.empty()) {
            std::vector<int> D;
            for (const auto& p : f1.paths)
                if (rng() % 2) D.push_back(p[rng() % p.size()]);
            if (D.empty()) D.push_back(f1.paths[0][0]);
            std::sort(D.begin(), D.end());
            D.erase(std::unique(D.begin(), D.end()), D.end());

            ++g_warm;
            VertexFlow w = f1;
            pnl.load(w, *sq);
            pnl.remove_paths_using_arcs(w, D, *sq);
            int guard = 0;
            while (pnl.augment_once(w, *sq, D)) {
                if (++guard > g.k() + 2) fail(at + ": warm start does not terminate");
            }
            pnl.compute_cut(w, *sq, D, true);
            validate_paths(g, w, D, at + " [warm start in G\\D]");

            Graph h = g;
            for (int a : D) h.delete_arc(a);
            FlowEngine hp(h);
            hp.set_penalties(&pen);
            auto sh = hp.make_scratch();
            VertexFlow fh;
            hp.compute_max_flow(v, fh, *sh, true, true);
            need(fh.kappa == w.kappa, at + ": warm-started kappa " + std::to_string(w.kappa) +
                                          " != from-scratch kappa in G\\D " + std::to_string(fh.kappa));
            need(fh.ess == w.ess, at + ": warm-started Ess != from-scratch Ess in G\\D");

            FlowEngine hb(h);  // and the plain BFS on G \ D must agree too
            auto shb = hb.make_scratch();
            VertexFlow fb;
            hb.compute_max_flow(v, fb, *shb, true, true);
            need(fb.kappa == fh.kappa && fb.ess == fh.ess, at + ": G\\D kappa/Ess differ between routings");
        }
    }
}

int main(int argc, char** argv) {
    unsigned lo = 0, hi = 2000;
    if (argc > 1) lo = (unsigned)std::strtoul(argv[1], nullptr, 10);
    if (argc > 2) hi = (unsigned)std::strtoul(argv[2], nullptr, 10);
    try {
        for (unsigned s = lo; s < hi; ++s) run_seed(s);
    } catch (const Failure& f) {
        std::printf("FAIL: %s\n", f.msg.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: C++ exception: %s\n", e.what());
        return 1;
    }
    std::printf("OK checks=%lld vertices=%lld paths_differ=%lld positive_penalty_paths=%lld warm_starts=%lld\n",
                g_checks, g_vertices, g_paths_differ, g_positive_pen, g_warm);
    return 0;
}
