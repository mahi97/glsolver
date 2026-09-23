// GLWeightedSolver: GLWeightedPartition [Alg 3] with RoundAndRemove [Alg 4] and the min-cost split assignment
// [Prop 5.4] (docs/paper_notes.md §5, §8; docs/weighted_algorithm.md), plus the exact optimizations of
// docs/optimizations.md that carry over to split assignments unchanged:
//   O1 batch deletion of arcs used by no stored flow (step iii-a), O2 warm-started criticality (inside the
//   EssentialOracle), O3 no recomputation after contraction, O4 terminal removal, O5 greedy contraction
//   (step iii-b), O6-style laziness in step (iii-c), O8 parallel oracle, O9 bitsets.
//
// Witness discipline. The paper recomputes the split witness psi [Def 5.2] from scratch by one min-cost flow in
// every step (iii) (paper_notes §8, notes). This solver CARRIES psi and re-optimizes it only when the paper's
// argument needs the optimum; psi stays a FESAC witness [Def 5.3] of the current graph across every operation:
//   (i)   removal of t with c_t = 0: psi(., t) = 0, so the restriction is a witness [Lem 8.1] (old essential
//         terminals survive, [Lem 7.1]);
//   (ii)  contraction of p with d^+(p) = 1 along (p,t): Ess(p) = {t} forces psi(p,t) = w_p <= c_t, and psi \ {p}
//         witnesses the contracted graph with c_t - w_p [Lem 8.2] (Ess unchanged elsewhere, §13.2 / O3);
//   (iii) deletion of an arc set D such that psi(v,t) > 0 implies t ∈ Ess_{G\D}(v) for every v: the same psi is a
//         witness of G \ D. This is exactly the argument of [Lem 8.5] (and of §13.1 for FEAC): it needs no
//         minimality of psi. Only when EVERY secondary arc is critical for some pair with psi(v,t) > 0 is psi
//         replaced by a minimum-potential witness [Prop 5.4], for which [Lem 8.4] guarantees a non-critical
//         secondary arc (a one-unit cycle shift would otherwise lower the potential Φ̄ [Def weighted potential]);
//   (iv)  RoundAndRemove: the restriction of psi to the survivors is a witness [Lem 8.8] ([Lem 7.7]: no survivor
//         sends weight into the deficient set S; [Lem 8.7]: essentiality survives the removal).
// With opt.lazy_shift = false step (iii) always recomputes psi by the min-cost flow (the literal [Alg 3]).
//
// Certified-subset discipline (RESEARCH_NOTES.md P2): after every operation oracle_.ess(v) is a subset of the true
// Ess_G(v) that contains every t with psi(v,t) > 0 (initial psi over exact sets; a new psi is computed over the
// stored sets; a deletion keeps the set of an unaffected vertex, adopts for a kappa-restored one the set stored at
// COMMIT time — not the copy taken at evaluation time, which predates a refresh_all_cuts of step (iii) — and installs
// the exact set of a kappa-dropped one, which contains t when the deletion is non-critical for (v,t); O5 moves
// psi(p) onto a terminal already in the stored set; terminal removal and rounding recompute exact sets that
// contain old \ removed; contraction keeps them). delete_and_commit asserts the invariant. Criticality of e for
// such a pair is decided with the exact cut of G \ e (O2). The min-cost flow of step (iii) is first solved over the stored
// subsets; if it is infeasible or yields no non-critical secondary arc (both impossible over the exact sets,
// [Lem 8.5] / [Lem 8.4]) every cut is refreshed and the flow is solved once more before an invariant violation
// (std::logic_error) is reported.
#include "solver.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "matching.hpp"
#include "mincostflow.hpp"
#include "solver_common.hpp"

namespace glcore {

namespace {

std::string vs(int64_t v) { return std::to_string(v); }

int resolve_threads(int requested) {
    if (requested > 0) return requested;
    const unsigned hc = std::thread::hardware_concurrency();
    return std::min<int>(hc == 0 ? 1 : (int)hc, 16);
}

constexpr size_t kMaxSizeSamples = 10000;   // graph_size_over_time is thinned beyond this
constexpr int kGreedyCandidates = 3;         // O5: candidates tried per step at full credit
constexpr int kGreedyMaxBackoff = 64;        // O5: a failed candidate is skipped for <= this many steps
constexpr int kGreedyCreditMax = 48;         // O5 throttle, as in solver_general.cpp
constexpr int kGreedyCreditAttempt = 4;
constexpr int kGreedyCreditSuccess = 16;

// Primal-dual min-cost flow [Prop 5.4] for the split-assignment network (s -> v -> t -> z). Successive shortest
// paths would augment one unit at a time on unit-weight instances (|V\T| Dijkstra runs per flow); the primal-dual
// variant runs one Dijkstra on the reduced costs per distinct path length, then a Dinic max-flow restricted to
// the admissible arcs (reduced cost 0), so a zero-cost network costs one Dinic run. Same invariant as
// glcore::MinCostFlow (nonnegative reduced costs via Johnson potentials, augmentations only along shortest
// paths, hence a min-cost flow of its value at every moment); integral; deterministic (adjacency order).
// In debug mode the solver cross-checks (value, cost) against glcore::MinCostFlow on the same network.
class PrimalDualMinCostFlow {
public:
    explicit PrimalDualMinCostFlow(int num_nodes)
        : n_(num_nodes), adj_(num_nodes), pot_(num_nodes, 0), dist_(num_nodes), level_(num_nodes), it_(num_nodes) {
        if (num_nodes <= 0) throw std::invalid_argument("PrimalDualMinCostFlow: num_nodes must be positive");
    }

    int add_arc(int u, int v, int64_t cap, int64_t cost) {
        if (u < 0 || u >= n_ || v < 0 || v >= n_) throw std::out_of_range("PrimalDualMinCostFlow::add_arc: node out of range");
        if (cap < 0 || cost < 0) throw std::invalid_argument("PrimalDualMinCostFlow::add_arc: negative capacity or cost");
        const int id = (int)edges_.size();
        edges_.push_back(E{v, cap, cost});
        adj_[u].push_back(id);
        edges_.push_back(E{u, 0, -cost});
        adj_[v].push_back(id + 1);
        return id;
    }

    int64_t flow(int arc) const { return edges_[(size_t)arc ^ 1].cap; }  // forward arcs only

    // Sends up to `required` units from s to z at minimum cost; returns (value, cost).
    std::pair<int64_t, int64_t> solve(int s, int z, int64_t required) {
        if (s == z || s < 0 || z < 0 || s >= n_ || z >= n_) throw std::invalid_argument("PrimalDualMinCostFlow::solve: bad source/sink");
        int64_t value = 0;
        while (value < required) {
            dijkstra(s);
            if (dist_[z] == INF) break;
            for (int v = 0; v < n_; ++v)
                if (dist_[v] != INF) pot_[v] += dist_[v];
            // max-flow on the admissible arcs (reduced cost 0): every s-z path in it is a shortest path
            while (value < required && bfs(s, z)) {
                std::fill(it_.begin(), it_.end(), 0);
                while (value < required) {
                    const int64_t f = augment(s, z, required - value);
                    if (f == 0) break;
                    value += f;
                }
            }
        }
        int64_t cost = 0;
        for (size_t a = 0; a < edges_.size(); a += 2) {
            const int64_t f = edges_[a ^ 1].cap;
            if (f == 0) continue;
            int64_t inc;
            if (__builtin_mul_overflow(f, edges_[a].cost, &inc) || __builtin_add_overflow(cost, inc, &cost))
                throw std::overflow_error("PrimalDualMinCostFlow::solve: total cost overflows int64");
        }
        return {value, cost};
    }

private:
    struct E {
        int to;
        int64_t cap;
        int64_t cost;
    };
    static constexpr int64_t INF = std::numeric_limits<int64_t>::max();
    int n_;
    std::vector<E> edges_;
    std::vector<std::vector<int>> adj_;
    std::vector<int64_t> pot_, dist_;
    std::vector<int> level_, it_;
    std::vector<int> queue_, stack_nodes_, stack_arcs_;

    bool admissible(int u, int a) const {
        const E& e = edges_[a];
        return e.cap > 0 && e.cost + pot_[u] - pot_[e.to] == 0;
    }

    void dijkstra(int s) {
        std::fill(dist_.begin(), dist_.end(), INF);
        dist_[s] = 0;
        using Item = std::pair<int64_t, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
        pq.push(Item{0, s});
        while (!pq.empty()) {
            const Item top = pq.top();
            pq.pop();
            const int u = top.second;
            if (top.first > dist_[u]) continue;
            for (int a : adj_[u]) {
                const E& e = edges_[a];
                if (e.cap <= 0) continue;
                const int64_t rc = e.cost + pot_[u] - pot_[e.to];
                if (rc < 0) throw std::logic_error("PrimalDualMinCostFlow: negative reduced cost; potential invariant violated");
                const int64_t nd = top.first + rc;
                if (nd < dist_[e.to]) {
                    dist_[e.to] = nd;
                    pq.push(Item{nd, e.to});
                }
            }
        }
    }

    bool bfs(int s, int z) {
        std::fill(level_.begin(), level_.end(), -1);
        queue_.clear();
        queue_.push_back(s);
        level_[s] = 0;
        for (size_t h = 0; h < queue_.size(); ++h) {
            const int u = queue_[h];
            for (int a : adj_[u]) {
                const int v = edges_[a].to;
                if (level_[v] < 0 && admissible(u, a)) {
                    level_[v] = level_[u] + 1;
                    queue_.push_back(v);
                }
            }
        }
        return level_[z] >= 0;
    }

    // One augmenting path in the admissible level graph (iterative DFS with current-arc pointers).
    int64_t augment(int s, int z, int64_t limit) {
        stack_nodes_.clear();
        stack_arcs_.clear();
        stack_nodes_.push_back(s);
        while (!stack_nodes_.empty()) {
            const int u = stack_nodes_.back();
            if (u == z) {
                int64_t f = limit;
                for (int a : stack_arcs_) f = std::min(f, edges_[a].cap);
                for (int a : stack_arcs_) {
                    edges_[a].cap -= f;
                    edges_[(size_t)a ^ 1].cap += f;
                }
                return f;
            }
            bool advanced = false;
            for (; it_[u] < (int)adj_[u].size(); ++it_[u]) {
                const int a = adj_[u][it_[u]];
                const int v = edges_[a].to;
                if (level_[v] == level_[u] + 1 && admissible(u, a)) {
                    stack_nodes_.push_back(v);
                    stack_arcs_.push_back(a);
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                level_[u] = -1;  // dead end for this phase
                stack_nodes_.pop_back();
                if (!stack_arcs_.empty()) stack_arcs_.pop_back();
                if (!stack_nodes_.empty()) ++it_[stack_nodes_.back()];
            }
        }
        return 0;
    }
};

}  // namespace

// ------------------------------------------------------------------------------------------ construction

GLWeightedSolver::GLWeightedSolver(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                                   const std::vector<int64_t>& capacities, const std::vector<int64_t>& weights,
                                   SolverOptions opt)
    : g_(n, arcs, terminals), stats_(), trace_(), opt_(opt),
      oracle_(g_, stats_, resolve_threads(opt.threads), opt.seed),
      cap_(capacities), w_(weights), parent_(n, -1), part_(n, -1), term_vertex_(terminals) {
    opt_.threads = resolve_threads(opt.threads);
    trace_.enabled = opt_.trace;
    const int k = g_.k0();
    if (k < 1) throw std::invalid_argument("GLWeightedSolver: at least one terminal is required");
    if ((int)cap_.size() != k)
        throw std::invalid_argument("GLWeightedSolver: " + vs(cap_.size()) + " capacities for " + vs(k) + " terminals");
    if ((int)w_.size() != n)
        throw std::invalid_argument("GLWeightedSolver: " + vs(w_.size()) + " weights for " + vs(n) + " vertices");
    for (int i = 0; i < k; ++i) {
        if (cap_[i] < 0) throw std::invalid_argument("GLWeightedSolver: capacity of terminal " + vs(terminals[i]) + " is negative");
        part_[terminals[i]] = i;
    }
    for (int v = 0; v < n; ++v) {
        if (g_.terminal_index(v) >= 0) {
            w_[v] = 0;  // terminals carry no weight (§1.3)
        } else if (w_[v] < 1) {
            throw std::invalid_argument("GLWeightedSolver: weight of non-terminal " + vs(v) + " is " + vs(w_[v]) +
                                        "; weights must be positive integers [§1.3]");
        }
    }
    psi_.assign(n, {});
    recv_.assign(k, 0);
    in_deg_queue_.assign(n, 0);
    in_pt_candidates_.assign(n, 0);
    cand_dirty_.assign(n, 0);
    cand_users_.assign(n, -1);
    cand_dsize_.assign(n, 0);
    cand_target_.assign(n, -1);
    greedy_skip_until_.assign(n, 0);
    greedy_failures_.assign(n, 0);
    greedy_credit_ = kGreedyCreditMax;
    match_p_.assign(k, -1);
    matched_to_.assign(n, -1);
    cost_row_of_.assign(n, -1);
    reroute_pass_ = c1_reroute_enabled();  // C1, off by default (RESEARCH_NOTES E5)
}

// ------------------------------------------------------------------------------------------ small helpers

void GLWeightedSolver::push_degree_changed(int v) {
    if (v < 0 || v >= g_.n() || in_deg_queue_[v]) return;
    in_deg_queue_[v] = 1;
    deg_queue_.push_back(v);
}

void GLWeightedSolver::push_pt_candidate(int v) {
    if (v < 0 || v >= g_.n()) return;
    mark_candidate_dirty(v);
    if (in_pt_candidates_[v]) return;
    in_pt_candidates_[v] = 1;
    pt_candidates_.push_back(v);
}

// ------------------------------------------------------------------------------------------ C1 penalties

// Incremental entry point: a no-op with routing = "bfs" (see GLSolver::penalty_update_vertex).
void GLWeightedSolver::penalty_update_vertex(int p) {
    if (opt_.routing_avoid) penalty_set_vertex(p);
}

// As GLSolver::penalty_set_vertex, with the psi target arc [Def 5.2] in the role of (p, phi(p)): the
// algorithm keeps psi_target_arc(p) and may delete every other out-arc of a pre-terminal p (steps (iii-a),
// (iii-b) and the secondary arcs of step (iii)). Before psi exists (the initial compute_all) the out-arcs
// of a pre-terminal that do not enter a terminal are penalized. O(d^+(p)) plus the psi scan.
void GLWeightedSolver::penalty_set_vertex(int p) {
    if (p < 0 || p >= g_.n()) return;
    if (penalty_.size() < (size_t)g_.num_arc_ids()) penalty_.resize(g_.num_arc_ids(), 0);
    if (!g_.live(p) || g_.is_terminal(p)) {
        for (int a : g_.out_arcs(p)) penalty_[a] = 0;
        return;
    }
    const bool pre = g_.is_pre_terminal(p);
    const int a_psi = (have_witness_ && pre) ? psi_target_arc(p) : -1;
    for (int a : g_.out_arcs(p)) {
        unsigned char v = 0;
        if (pre) {
            if (have_witness_) v = (a == a_psi) ? 0 : 1;
            else v = g_.is_terminal(g_.arc(a).head) ? 0 : 1;
        }
        penalty_[a] = v;
    }
}

void GLWeightedSolver::penalty_recompute_all() {
    penalty_.assign(g_.num_arc_ids(), 0);
    for (int v : g_.live_nonterminals()) penalty_set_vertex(v);
}

// C1 re-routing pass, as in GLSolver (once, right after the initial split witness); like there it recomputes
// the offending flows from scratch, so their cuts come back exact.
void GLWeightedSolver::penalty_reroute_offenders() {
    if (!opt_.routing_avoid) return;
    if (!reroute_pass_) return;
    Stats::Timer timer(stats_, "c1_reroute");
    std::vector<int> offenders;
    for (int a = 0; a < g_.num_arc_ids(); ++a) {
        if (!g_.arc(a).alive || a >= (int)penalty_.size() || !penalty_[a]) continue;
        for (int v : oracle_.users_of_arc(a)) offenders.push_back(v);
    }
    std::sort(offenders.begin(), offenders.end());
    offenders.erase(std::unique(offenders.begin(), offenders.end()), offenders.end());
    oracle_.reroute(offenders);  // counts stats_.reroutes
}

int64_t GLWeightedSolver::penalized_users() {
    int64_t total = 0;
    for (int a = 0; a < g_.num_arc_ids(); ++a)
        if (g_.arc(a).alive && a < (int)penalty_.size() && penalty_[a]) total += (int64_t)oracle_.num_users_of_arc(a);
    return total;
}

void GLWeightedSolver::penalty_debug_check(const char* where) {
    if (!opt_.routing_avoid) return;  // nothing is maintained when the option is off
    std::vector<unsigned char> mine(penalty_);
    penalty_recompute_all();
    for (int a = 0; a < g_.num_arc_ids(); ++a) {
        if (!g_.arc(a).alive) continue;
        const unsigned char got = a < (int)mine.size() ? mine[a] : (unsigned char)0;
        if (got != penalty_[a])
            throw std::logic_error(std::string("C1 penalty out of sync after ") + where + ": arc " + vs(a) + " (" +
                                   vs(g_.arc(a).tail) + "," + vs(g_.arc(a).head) + ") is " + vs(got) + ", recomputed " +
                                   vs(penalty_[a]));
    }
}

void GLWeightedSolver::mark_candidate_dirty(int v) {
    if (v < 0 || v >= g_.n() || cand_dirty_[v]) return;
    cand_dirty_[v] = 1;
    cand_dirty_list_.push_back(v);
}

// The arcs of v's stored flow are about to lose (or have just gained) a user: their tails' unused-arc sets
// and O5 scores may change. O(total path length) — the oracle already spent that on the flow itself.
void GLWeightedSolver::mark_flow_arcs_dirty(int v) {
    if (!g_.live(v) || g_.is_terminal(v)) return;
    for (const auto& path : oracle_.flow(v).paths)
        for (int a : path) mark_candidate_dirty(g_.arc(a).tail);
}

// One (live non-terminals, live arcs) sample per step; thinned beyond kMaxSizeSamples (as in GLSolver).
void GLWeightedSolver::sample_graph_size() {
    if ((stats_.steps - 1) % size_sample_every_ != 0) return;
    stats_.graph_size_over_time.emplace_back(g_.num_live_nonterminals(), g_.num_live_arcs());
    if (stats_.graph_size_over_time.size() >= kMaxSizeSamples) {
        auto& s = stats_.graph_size_over_time;
        size_t w = 0;
        for (size_t i = 0; i < s.size(); i += 2) s[w++] = s[i];
        s.resize(w);
        size_sample_every_ *= 2;
    }
}

std::string GLWeightedSolver::capacities_json() const {
    JsonIntMap m;
    for (int t : g_.terminals()) m.add(t, vs(cap_[g_.terminal_index(t)]));
    return m.build();
}

// [[v, t, units], ...] sorted by (v, t) with t the terminal vertex (the reference's "psi" payload, §14).
std::string GLWeightedSolver::psi_json() const {
    std::string s = "[";
    bool first = true;
    for (int v = 0; v < g_.n(); ++v) {
        if (!g_.live(v) || g_.is_terminal(v) || psi_[v].empty()) continue;
        std::vector<std::pair<int, int64_t>> entries;
        for (const auto& e : psi_[v]) entries.emplace_back(term_vertex_[e.first], e.second);
        std::sort(entries.begin(), entries.end());
        for (const auto& e : entries) {
            if (!first) s += ',';
            first = false;
            s += "[" + vs(v) + "," + vs(e.first) + "," + vs(e.second) + "]";
        }
    }
    s += ']';
    return s;
}

int64_t GLWeightedSolver::psi_units(int v, int ti) const {
    for (const auto& e : psi_[v])
        if (e.first == ti) return e.second;
    return 0;
}

// The alive arc (p, t) into the terminal receiving the largest share of psi(p, .), or -1 (ties: smaller index).
int GLWeightedSolver::psi_target_arc(int p) const {
    int best = -1, best_ti = -1;
    int64_t best_units = 0;
    for (const auto& e : psi_[p]) {
        const int t = term_vertex_[e.first];
        if (!g_.is_terminal(t)) continue;
        const int a = g_.find_arc(p, t);
        if (a < 0) continue;
        if (best < 0 || e.second > best_units || (e.second == best_units && e.first < best_ti)) {
            best = a;
            best_ti = e.first;
            best_units = e.second;
        }
    }
    return best;
}

// psi(p, .) := {t_i: w_p} (the caller has checked that recv[t_i] - psi(p,t_i) + w_p <= c_{t_i}).
void GLWeightedSolver::set_psi_single(int p, int ti) {
    for (const auto& e : psi_[p]) recv_[e.first] -= e.second;
    psi_[p].clear();
    psi_[p].emplace_back(ti, w_[p]);
    recv_[ti] += w_[p];
    mark_candidate_dirty(p);
    penalty_update_vertex(p);  // C1: the kept arc of p is now (p, t_i)
}

// "essential" event (§14): exact sets (every cut refreshed first; trace mode is for small instances).
void GLWeightedSolver::trace_essential() {
    if (!trace_.enabled) return;
    oracle_.refresh_all_cuts();
    JsonIntMap ess, kappa, cuts;
    for (int v : g_.live_nonterminals()) {
        const VertexFlow& f = oracle_.flow(v);
        ess.add(v, termset_json(f.ess, term_vertex_));
        kappa.add(v, vs(f.kappa));
        if (opt_.record_cuts && f.cut_exact) {
            std::vector<Side> side;
            oracle_.sides(v, side);  // materialized on demand (one reverse BFS); never stored per flow
            std::vector<int> L, S, R;
            for (int x = 0; x < g_.n(); ++x) {
                if (!g_.live(x)) continue;
                (side[x] == Side::L ? L : (side[x] == Side::S ? S : R)).push_back(x);
            }
            cuts.add(v, JsonObject().raw("L", json_list(L)).raw("S", json_list(S)).raw("R", json_list(R)).build());
        }
    }
    JsonObject ev;
    ev.raw("ess", ess.build()).raw("kappa", kappa.build());
    if (opt_.record_cuts) ev.raw("cuts", cuts.build());
    trace_.record("essential", ev.build());
}

// ------------------------------------------------------------------------------------------ debug checks

// The carried psi is a split witness [Def 5.3] of the current graph: positive entries on live essential
// terminals (exact sets), Σ_t psi(v,t) = w_v, Σ_v psi(v,t) = recv_[t] <= c_t.
void GLWeightedSolver::debug_check_witness(const char* where) {
    // P2 first: the stored certified subsets (before any refresh makes them exact) contain every psi target.
    for (int v : g_.live_nonterminals())
        for (const auto& e : psi_[v])
            if (e.first >= 0 && e.first < g_.k0() && !oracle_.ess(v).test(e.first))
                throw std::logic_error(std::string("P2 violated after ") + where + ": psi(" + vs(v) + ", t_" + vs(e.first) + ") = " +
                                       vs(e.second) + " but terminal " + vs(term_vertex_[e.first]) + " is not in the certified essential subset of " +
                                       vs(v) + "; E = " + termset_json(oracle_.ess(v), term_vertex_));
    oracle_.refresh_all_cuts();
    const std::string at = std::string("split witness invariant violated after ") + where + ": ";
    std::vector<int64_t> got(g_.k0(), 0);
    for (int v : g_.live_nonterminals()) {
        int64_t sum = 0;
        for (const auto& e : psi_[v]) {
            const int ti = e.first;
            if (e.second <= 0) throw std::logic_error(at + "psi(" + vs(v) + ", t_" + vs(ti) + ") = " + vs(e.second) + " is not positive");
            if (ti < 0 || ti >= g_.k0() || !g_.is_terminal(term_vertex_[ti]))
                throw std::logic_error(at + "psi(" + vs(v) + ", t_" + vs(ti) + ") > 0 but t_" + vs(ti) + " is not a live terminal");
            if (!oracle_.ess(v).test(ti))
                throw std::logic_error(at + "psi(" + vs(v) + ", t_" + vs(ti) + ") = " + vs(e.second) + " but terminal " +
                                       vs(term_vertex_[ti]) + " is not essential for " + vs(v) + "; Ess = " +
                                       termset_json(oracle_.ess(v), term_vertex_));
            sum += e.second;
            got[ti] += e.second;
        }
        if (sum != w_[v]) throw std::logic_error(at + "vertex " + vs(v) + " sends " + vs(sum) + " units, weight is " + vs(w_[v]));
    }
    for (int t : g_.terminals()) {
        const int ti = g_.terminal_index(t);
        if (got[ti] != recv_[ti])
            throw std::logic_error(at + "terminal " + vs(t) + " receives " + vs(got[ti]) + " units but recv_ says " + vs(recv_[ti]));
        if (got[ti] > cap_[ti])
            throw std::logic_error(at + "terminal " + vs(t) + " receives " + vs(got[ti]) + " units, capacity " + vs(cap_[ti]));
    }
}

// debug_asserts: after every operation psi is re-verified and FESAC is re-tested with exact cuts.
void GLWeightedSolver::check_after_operation(const char* where) {
    if (!opt_.debug_asserts) return;
    Stats::Timer timer(stats_, "debug_checks");
    debug_check_witness(where);
    penalty_debug_check(where);  // C1: incremental maintenance vs. a full recomputation
    if (!fesac_feasible(true))
        throw std::logic_error(std::string("FESAC violated after ") + where + " [Lem 8.1/8.2/8.5/8.8]");
}

// ------------------------------------------------------------------------------------------ [Prop 5.4]

void GLWeightedSolver::clear_cost_table() {
    for (int v : costed_vertices_) {
        const int row = cost_row_of_[v];
        if (row >= 0) std::fill(cost_rows_[row].begin(), cost_rows_[row].end(), 0);
        cost_row_of_[v] = -1;
    }
    costed_vertices_.clear();
}

// xi_v(t) = |{i : e_i critical for (v,t)}| [Def 6.2] from the evaluations in eval_slots_: crit_i(v,t) holds iff
// kappa dropped under e_i (else nothing is critical, O1), t ∈ ess(v) (stored set) and t ∉ Ess_{G\e_i}(v)
// (exact cut, O2) [Def 6.1]. Rows are allocated only for vertices with a dropped kappa.
void GLWeightedSolver::build_cost_table(int kk, const std::vector<std::vector<int>>& aff) {
    clear_cost_table();
    const int k0 = g_.k0();
    for (int i = 0; i < kk; ++i) {
        for (int v : aff[i]) {
            if (!g_.live(v) || g_.is_terminal(v)) continue;
            const VertexFlow& f = eval_slots_[i][v];
            if (f.kappa >= oracle_.kappa(v)) continue;
            if (!f.cut_exact) throw std::logic_error("evaluate_deletion returned a dropped kappa without an exact cut for " + vs(v));
            int row = cost_row_of_[v];
            if (row < 0) {
                row = (int)costed_vertices_.size();
                if (row >= (int)cost_rows_.size()) cost_rows_.emplace_back(k0, 0);
                if ((int)cost_rows_[row].size() != k0) cost_rows_[row].assign(k0, 0);
                cost_row_of_[v] = row;
                costed_vertices_.push_back(v);
            }
            const TermSet& es = oracle_.ess(v);
            for (int ti = 0; ti < k0; ++ti)
                if (es.test(ti) && !f.ess.test(ti)) ++cost_rows_[row][ti];
        }
    }
}

// [Prop 5.4] / [Alg MinCostSplitAssignment]: s -> v (cap w_v, cost 0), v -> t (cap w_v, cost xi_v(t)) for t in
// the stored ess(v), t -> z (cap c_t, cost 0); a flow of value W = Σ w_v exists iff FESAC holds over these sets
// [Def 5.3]. With adopt the arc flows become psi (positive entries only) and recv_ is rebuilt.
GLWeightedSolver::SplitOutcome GLWeightedSolver::solve_split_assignment(bool with_costs, bool adopt, bool debug_only) {
    Stats::Timer timer(stats_, debug_only ? "debug_checks" : "min_cost_flow");
    const std::vector<int> verts = g_.live_nonterminals();
    const std::vector<int> terms = g_.terminals();
    const int L = (int)verts.size(), K = (int)terms.size(), k0 = g_.k0();
    const int s = 0, z = L + K + 1;
    PrimalDualMinCostFlow net(z + 1);
    std::unique_ptr<MinCostFlow> check;  // debug: the successive-shortest-paths implementation must agree
    if (opt_.debug_asserts) check = std::make_unique<MinCostFlow>(z + 1);
    std::vector<int> tnode(k0, -1);
    for (int j = 0; j < K; ++j) tnode[g_.terminal_index(terms[j])] = 1 + L + j;
    std::vector<std::vector<std::pair<int, int>>> arcs(L);  // (terminal index, arc id)
    int64_t W = 0;
    for (int i = 0; i < L; ++i) {
        const int v = verts[i];
        if (__builtin_add_overflow(W, w_[v], &W)) throw std::overflow_error("GLWeightedSolver: total weight overflows int64");
        net.add_arc(s, 1 + i, w_[v], 0);
        if (check) check->add_arc(s, 1 + i, w_[v], 0);
        const TermSet& es = oracle_.ess(v);
        const int64_t* row = (with_costs && cost_row_of_[v] >= 0) ? cost_rows_[cost_row_of_[v]].data() : nullptr;
        for (int ti = 0; ti < k0; ++ti) {
            if (!es.test(ti) || tnode[ti] < 0) continue;
            const int64_t cost = row ? row[ti] : 0;
            arcs[i].emplace_back(ti, net.add_arc(1 + i, tnode[ti], w_[v], cost));
            if (check) check->add_arc(1 + i, tnode[ti], w_[v], cost);
        }
    }
    for (int j = 0; j < K; ++j) {
        const int ti = g_.terminal_index(terms[j]);
        net.add_arc(tnode[ti], z, cap_[ti], 0);
        if (check) check->add_arc(tnode[ti], z, cap_[ti], 0);
    }
    const std::pair<int64_t, int64_t> r = net.solve(s, z, W);
    if (!debug_only) ++stats_.min_cost_flow_calls;
    if (check) {
        const std::pair<int64_t, int64_t> r2 = check->solve(s, z, W);
        if (r2 != r)
            throw std::logic_error("[Prop 5.4] primal-dual min-cost flow (value " + vs(r.first) + ", cost " + vs(r.second) +
                                   ") disagrees with successive shortest paths (value " + vs(r2.first) + ", cost " + vs(r2.second) + ")");
    }
    SplitOutcome out;
    out.value = r.first;
    out.cost = r.second;
    out.feasible = (r.first == W);
    if (out.feasible && adopt) {
        std::fill(recv_.begin(), recv_.end(), 0);
        for (int i = 0; i < L; ++i) {
            const int v = verts[i];
            psi_[v].clear();
            for (const auto& ta : arcs[i]) {
                const int64_t f = net.flow(ta.second);
                if (f > 0) {
                    psi_[v].emplace_back(ta.first, f);
                    recv_[ta.first] += f;
                }
            }
        }
        cand_all_dirty_ = true;  // psi changed everywhere: O1/O5 targets must be recomputed
        if (have_witness_ && opt_.routing_avoid) penalty_recompute_all();  // C1: every kept arc may have moved
    }
    return out;
}

// [Def 5.3] feasibility test by a zero-cost split assignment; with exact_cuts over the exact essential sets.
bool GLWeightedSolver::fesac_feasible(bool exact_cuts) {
    if (exact_cuts) oracle_.refresh_all_cuts();
    return solve_split_assignment(false, false, true).feasible;
}

// ------------------------------------------------------------------------------------------ deletions

// psi survives deleting D iff every pair with psi(v,t) > 0 keeps t essential in G \ D (paper_notes §13.1 /
// [Lem 8.5]): kappa unchanged -> the old certified set still holds (O1, generalized [Lem 4.3]); kappa dropped
// -> the oracle computed the exact cut of G \ D. Unaffected vertices are covered by O1. For special_p (O5: the
// pre-terminal about to be contracted) the concentrated psi'(p) = {special_ti: w_p} is checked instead.
bool GLWeightedSolver::psi_survives_deletion(const std::vector<int>& affected, const std::vector<VertexFlow>& out,
                                             int special_p, int special_ti) {
    for (int v : affected) {
        if (!g_.live(v) || g_.is_terminal(v)) continue;
        const VertexFlow& f = out[v];
        if (f.kappa == oracle_.kappa(v)) continue;
        if (!f.cut_exact) throw std::logic_error("evaluate_deletion returned a dropped kappa without an exact cut for " + vs(v));
        if (v == special_p) {
            if (!f.ess.test(special_ti))
                throw std::logic_error("O5: after deleting every other out-arc of " + vs(v) + " its single terminal t_" +
                                       vs(special_ti) + " is not essential (kappa " + vs(f.kappa) + ")");
            continue;
        }
        for (const auto& e : psi_[v])
            if (!f.ess.test(e.first)) return false;
    }
    return true;
}

// Delete the (already evaluated) arc set D and let the oracle adopt the flows of G \ D. Tails are queued for
// step (ii) (their out-degree dropped). Counters `deletions` / `batched_deletions` are kept by the oracle.
// P2: for an affected v whose kappa was restored, evaluate_deletion copied the subset stored at EVALUATION time;
// step (iii) may have refreshed every cut since (trace mode, or the exact retry) and computed psi over the exact
// sets, so the copy may lack a psi target. The set stored NOW is a certified subset of Ess_G(v) containing every
// psi target, hence a subset of Ess_{G\D}(v) (O1, generalized [Lem 4.3]); adopting it keeps the invariant.
void GLWeightedSolver::delete_and_commit(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& out) {
    Stats::Timer timer(stats_, "delete_commit");
    for (int v : affected) {
        mark_flow_arcs_dirty(v);  // the old paths through D are dropped
        if (g_.live(v) && !g_.is_terminal(v) && !out[v].cut_exact) out[v].ess = oracle_.ess(v);  // before the mutation
    }
    for (int a : D) {
        push_degree_changed(g_.arc(a).tail);
        mark_candidate_dirty(g_.arc(a).tail);
        g_.delete_arc(a);
    }
    if (opt_.routing_avoid) {
        // C1: the penalty of a SURVIVING arc (u, q) is [u is a pre-terminal] && [(u,q) is not the kept arc],
        // and both depend only on the arcs from u into terminals; so a tail needs no work unless one of the
        // deleted arcs entered a terminal (it may have been the kept arc, or u's last arc into a terminal).
        std::vector<int> tails;
        for (int a : D)
            if (g_.is_terminal(g_.arc(a).head)) tails.push_back(g_.arc(a).tail);
        std::sort(tails.begin(), tails.end());
        tails.erase(std::unique(tails.begin(), tails.end()), tails.end());
        for (int u : tails) penalty_update_vertex(u);
    }
    oracle_.commit_deletion(D, affected, out);
    for (int v : affected) {
        mark_flow_arcs_dirty(v);  // the rerouted paths gained users
        // P2 invariant: the certified subset stored for every live non-terminal contains every psi target.
        if (!g_.live(v) || g_.is_terminal(v)) continue;
        for (const auto& e : psi_[v])
            if (!oracle_.ess(v).test(e.first))
                throw std::logic_error("P2 violated after deleting " + vs(D.size()) + " arc(s): psi(" + vs(v) + ", t_" + vs(e.first) +
                                       ") = " + vs(e.second) + " but t_" + vs(e.first) + " (vertex " + vs(term_vertex_[e.first]) +
                                       ") is not in the certified essential set of " + vs(v) + "; E = " +
                                       termset_json(oracle_.ess(v), term_vertex_) + (oracle_.flow(v).cut_exact ? " (exact)" : " (subset)"));
    }
}

// ------------------------------------------------------------------------------------------ step (i)

// [Lem 8.1]: a terminal with c_t = 0 receives no weight (psi(., t) = 0), so removing it keeps FESAC with the
// same psi ([Lem 7.1]: old essential terminals survive). The oracle recomputes every cut exactly (O4; new
// essential terminals may appear, §13.3).
bool GLWeightedSolver::step_remove_zero_terminal() {
    while (zero_head_ < zero_terms_.size()) {
        const int ti = zero_terms_[zero_head_++];
        const int t = term_vertex_[ti];
        if (!g_.is_terminal(t) || cap_[ti] != 0) continue;
        if (recv_[ti] != 0)
            throw std::logic_error("[Lem 8.1] terminal " + vs(t) + " has capacity 0 but receives " + vs(recv_[ti]) + " units of psi");
        Stats::Timer timer(stats_, "terminal_removal");
        for (int a : g_.in_arcs(t)) push_degree_changed(g_.arc(a).tail);
        g_.remove_terminal(t);
        if (opt_.routing_avoid) penalty_recompute_all();  // C1: many vertices stop being pre-terminals at once
        oracle_.after_terminal_removal(t);  // counts stats_.terminal_removals
        cand_all_dirty_ = true;             // every flow may have been rerouted (O4)
        trace_.record("remove_terminal", JsonObject().num("t", t).raw("capacities", capacities_json()).build());
        trace_essential();
        check_after_operation("terminal removal");
        return true;
    }
    if (zero_head_ > 0) {
        zero_terms_.clear();
        zero_head_ = 0;
    }
    return false;
}

// ------------------------------------------------------------------------------------------ step (ii)

// Out-degree-1 pre-terminals come from the queue of vertices whose out-degree changed (initially every
// pre-terminal; later the tails of deleted arcs, the in-neighbours of contracted vertices and of removed
// terminals / vertices).
bool GLWeightedSolver::step_contract_degree_one() {
    while (deg_head_ < deg_queue_.size()) {
        const int p = deg_queue_[deg_head_++];
        in_deg_queue_[p] = 0;
        if (!g_.live(p) || g_.is_terminal(p) || g_.out_degree(p) != 1) continue;
        const int t = g_.arc(g_.out_arcs(p)[0]).head;
        if (!g_.is_terminal(t)) continue;  // not a pre-terminal; re-queued when its degree changes again
        contract_into(p, t);
        return true;
    }
    deg_queue_.clear();
    deg_head_ = 0;
    return false;
}

// [Lem 8.2] contract p (single arc (p,t)) into t and set c_t -= w_p. Ess(p) = {t} forces psi(p,t) = w_p <= c_t.
// kappa / Ess of every remaining vertex are unchanged (§13.2, O3), so nothing is recomputed (A7 checks it).
void GLWeightedSolver::contract_into(int p, int t) {
    Stats::Timer timer(stats_, "contract");
    const int ti = g_.terminal_index(t);
    if (psi_[p].size() != 1 || psi_[p][0].first != ti || psi_[p][0].second != w_[p])
        throw std::logic_error("[Lem 8.2] pre-terminal " + vs(p) + " has the single out-arc (" + vs(p) + "," + vs(t) +
                               ") so Ess(p) = {t} forces psi(p,t) = w_p = " + vs(w_[p]) + ", but psi(p, .) has " +
                               vs(psi_[p].size()) + " entries" + (psi_[p].empty() ? "" : " (first: t_" + vs(psi_[p][0].first) + " with " + vs(psi_[p][0].second) + " units)"));
    if (cap_[ti] < w_[p])
        throw std::logic_error("[Lem 8.2] c_" + vs(t) + " = " + vs(cap_[ti]) + " < w_" + vs(p) + " = " + vs(w_[p]) + " at contraction");
    // A7 snapshot (debug): exact kappa / Ess of every other live non-terminal before the contraction.
    std::vector<std::pair<int, std::pair<int, TermSet>>> before;
    if (opt_.debug_asserts) {
        oracle_.refresh_all_cuts();
        for (int v : g_.live_nonterminals())
            if (v != p) before.emplace_back(v, std::make_pair(oracle_.kappa(v), oracle_.ess(v)));
    }
    std::vector<int> preds;
    preds.reserve(g_.in_degree(p));
    for (int a : g_.in_arcs(p)) preds.push_back(g_.arc(a).tail);
    mark_flow_arcs_dirty(p);   // F_p disappears with p: its arcs lose a user
    mark_candidate_dirty(p);   // dropped from the candidate list at the next refresh
    parent_[p] = g_.contract(p, t);
    oracle_.after_contraction(p, t);  // O3; counts stats_.contractions
    penalty_update_vertex(p);                               // C1: p is dead, its in-neighbours gained (u,t)
    for (int u : preds) penalty_update_vertex(u);
    cap_[ti] -= w_[p];
    recv_[ti] -= w_[p];
    psi_[p].clear();
    if (cap_[ti] == 0) zero_terms_.push_back(ti);
    part_[p] = ti;
    for (int u : preds) {
        push_degree_changed(u);   // duplicate merge may have lowered d^+(u)
        push_pt_candidate(u);     // (u,p) became (u,t): u is a pre-terminal now
    }
    trace_.record("contract", JsonObject().num("p", p).num("t", t).num("parent", parent_[p]).raw("capacities", capacities_json()).build());
    if (opt_.debug_asserts) {
        Stats::Timer dt(stats_, "debug_checks");
        oracle_.refresh_all_cuts();
        for (const auto& e : before) {
            const int v = e.first;
            if (oracle_.kappa(v) != e.second.first || oracle_.ess(v) != e.second.second)
                throw std::logic_error("A7 violated: kappa/Ess of " + vs(v) + " changed by the contraction of " + vs(p) + " into " + vs(t) +
                                       " (kappa " + vs(e.second.first) + " -> " + vs(oracle_.kappa(v)) + ", Ess " +
                                       termset_json(e.second.second, term_vertex_) + " -> " + termset_json(oracle_.ess(v), term_vertex_) + ")");
        }
    }
    check_after_operation("contraction");
}

// ------------------------------------------------------------------------------------------ candidate scans

// Recompute, for every dirty candidate p (or all of them after a terminal removal / rounding / new psi):
// eligibility (live pre-terminal with an alive arc to a terminal receiving weight from p), the unused arcs of
// p other than that target arc (O1 batch) and the O5 score (flows using D_p = out(p) \ {target}).
void GLWeightedSolver::refresh_dirty_candidates() {
    std::vector<int> todo;
    if (cand_all_dirty_) {
        todo = pt_candidates_;
        for (int v : cand_dirty_list_) cand_dirty_[v] = 0;
        cand_dirty_list_.clear();
        cand_all_dirty_ = false;
    } else {
        todo.swap(cand_dirty_list_);
        for (int v : todo) cand_dirty_[v] = 0;
    }
    for (int p : todo) {
        if (!in_pt_candidates_[p]) continue;
        if (!g_.live(p) || g_.is_terminal(p) || !g_.is_pre_terminal(p)) {
            in_pt_candidates_[p] = 0;  // dropped; re-added (dirty) if a contraction makes it a pre-terminal again
            cand_users_[p] = -1;
            cand_target_[p] = -1;
            continue;
        }
        const int a_t = psi_target_arc(p);
        cand_target_[p] = a_t;
        if (a_t < 0) {
            cand_users_[p] = -1;
            continue;
        }
        int64_t users = 0;
        for (int a : g_.out_arcs(p)) {
            if (a == a_t) continue;
            const size_t u = oracle_.num_users_of_arc(a);
            users += (int64_t)u;
            if (u == 0 && opt_.batch_unused_arcs) batch_arcs_.push_back(a);
        }
        cand_users_[p] = g_.out_degree(p) >= 2 ? users : -1;
        cand_dsize_[p] = g_.out_degree(p) - 1;
    }
    size_t alive = 0;
    for (int p : pt_candidates_) alive += in_pt_candidates_[p];
    if (alive * 2 < pt_candidates_.size()) {
        size_t w = 0;
        for (int p : pt_candidates_)
            if (in_pt_candidates_[p]) pt_candidates_[w++] = p;
        pt_candidates_.resize(w);
    }
}

// ------------------------------------------------------------------------------------------ step (iii-a), O1

// D = alive arcs (p,q) of candidate pre-terminals p, other than p's target arc, used by no stored flow. By O1
// deleting all of D at once keeps every stored flow maximum and every certified set a subset of the new
// Ess, so psi stays a witness (no pair (v,t) can lose essentiality). Arcs that gained a user since the scan
// are caught by evaluate_deletion (affected non-empty) and the batch is then subject to the exact test.
bool GLWeightedSolver::step_batch_unused_arcs() {
    Stats::Timer timer(stats_, "batch");
    refresh_dirty_candidates();
    if (batch_arcs_.empty()) return false;
    std::vector<int> D;
    std::sort(batch_arcs_.begin(), batch_arcs_.end());
    batch_arcs_.erase(std::unique(batch_arcs_.begin(), batch_arcs_.end()), batch_arcs_.end());
    for (int a : batch_arcs_)
        if (g_.arc(a).alive) D.push_back(a);
    batch_arcs_.clear();
    if (D.empty()) return false;
    std::vector<int> affected = oracle_.evaluate_deletion(D, scratch_slot_, false);  // normally no affected vertex
    if (!psi_survives_deletion(affected, scratch_slot_, -1, -1)) return false;
    std::vector<std::pair<int, int>> pairs;
    if (trace_.enabled)
        for (int a : D) pairs.emplace_back(g_.arc(a).tail, g_.arc(a).head);
    delete_and_commit(D, affected, scratch_slot_);  // stats_.deletions += |D|, batched_deletions++ (oracle)
    trace_.record("delete_arcs", JsonObject().raw("arcs", json_pairs(pairs)).build());
    trace_essential();
    check_after_operation("batch deletion of unused arcs");
    return true;
}

// ------------------------------------------------------------------------------------------ step (iii-b), O5

// Try to make a pre-terminal p out-degree 1 by deleting D_p = out(p) \ {(p, t)} for a terminal t receiving
// weight from p: valid iff psi'(v,t') stays essential for every v in G \ D_p (§13.1 / [Lem 8.5]), where psi'
// moves all of p's weight onto t (possible iff recv[t] - psi(p,t) + w_p <= c_t; t stays essential for p as its
// only out-neighbour). Tested exactly on the affected vertices only (O1/O2). Candidates are ordered by (flows
// using D_p, |D_p|, p); a failed candidate backs off exponentially and a credit throttle bounds the work.
bool GLWeightedSolver::step_greedy_contraction() {
    Stats::Timer timer(stats_, "greedy");
    if (greedy_credit_ <= 0) {
        ++greedy_credit_;
        return false;
    }
    refresh_dirty_candidates();
    struct Cand {
        int64_t users;
        int dsize, p;
        bool operator<(const Cand& o) const { return std::tie(users, dsize, p) < std::tie(o.users, o.dsize, o.p); }
    };
    std::vector<Cand> cands;
    for (int p : pt_candidates_) {
        if (!in_pt_candidates_[p] || cand_users_[p] < 0 || !g_.live(p) || g_.is_terminal(p)) continue;
        if (greedy_skip_until_[p] > stats_.steps) continue;
        cands.push_back(Cand{cand_users_[p], cand_dsize_[p], p});
    }
    if (cands.empty()) return false;
    const int budget = greedy_credit_ >= 2 * kGreedyCreditSuccess ? kGreedyCandidates : (greedy_credit_ >= kGreedyCreditSuccess ? 2 : 1);
    const size_t tries = std::min<size_t>((size_t)budget, cands.size());
    std::partial_sort(cands.begin(), cands.begin() + (std::ptrdiff_t)tries, cands.end());
    auto fail = [&](int p) {
        const int f = ++greedy_failures_[p];
        greedy_skip_until_[p] = stats_.steps + std::min(kGreedyMaxBackoff, 1 << std::min(f, 6));
    };
    for (size_t c = 0; c < tries; ++c) {
        const int p = cands[c].p;
        const int a_t = cand_target_[p];
        if (a_t < 0 || !g_.arc(a_t).alive || g_.arc(a_t).tail != p || g_.out_degree(p) < 2) continue;  // stale score
        const int t = g_.arc(a_t).head;
        if (!g_.is_terminal(t)) continue;
        const int ti = g_.terminal_index(t);
        if (recv_[ti] - psi_units(p, ti) + w_[p] > cap_[ti]) {  // p's weight does not fit into t
            fail(p);
            continue;
        }
        ++stats_.greedy_attempts;
        greedy_credit_ -= kGreedyCreditAttempt;
        std::vector<int> D;
        std::vector<std::pair<int, int>> pairs;
        for (int a : g_.out_arcs(p))
            if (a != a_t) {
                D.push_back(a);
                pairs.emplace_back(p, g_.arc(a).head);
            }
        std::vector<int> affected = oracle_.evaluate_deletion(D, scratch_slot_, false);
        if (!psi_survives_deletion(affected, scratch_slot_, p, ti)) {
            fail(p);
            if (greedy_credit_ <= 0) break;
            continue;
        }
        greedy_credit_ = std::min(kGreedyCreditMax, greedy_credit_ + kGreedyCreditSuccess);
        set_psi_single(p, ti);                          // psi'(p) = {t: w_p}
        delete_and_commit(D, affected, scratch_slot_);  // stats_.deletions += |D| (oracle)
        ++stats_.greedy_successes;
        push_degree_changed(p);  // d^+(p) = 1 now: step (ii) contracts it next
        trace_.record("greedy_delete", JsonObject().num("p", p).num("t", t).raw("arcs", json_pairs(pairs)).build());
        trace_essential();
        check_after_operation("greedy deletion");
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------ matching [Lem 7.8]

// Keep the previous saturating matching: drop pairs whose pre-terminal died or whose arc was deleted, re-match
// each unmatched terminal with a free in-neighbour (augmenting path of length one), and only when that fails
// recompute from scratch (Hopcroft–Karp). Returns false iff no saturating matching exists (-> [Alg 4]).
bool GLWeightedSolver::ensure_saturating_matching() {
    Stats::Timer tm(stats_, "matching");
    const std::vector<int> terms = g_.terminals();
    for (int ti = 0; ti < g_.k0(); ++ti) {
        const int p = match_p_[ti];
        if (p < 0) continue;
        const int t = term_vertex_[ti];
        if (!g_.is_terminal(t) || !g_.live(p) || g_.is_terminal(p) || g_.find_arc(p, t) < 0) {
            if (matched_to_[p] == ti) matched_to_[p] = -1;
            match_p_[ti] = -1;
        }
    }
    bool complete = true;
    for (int t : terms) {
        const int ti = g_.terminal_index(t);
        if (match_p_[ti] >= 0) continue;
        for (int a : g_.in_arcs(t)) {
            const int p = g_.arc(a).tail;
            if (matched_to_[p] < 0) {
                matched_to_[p] = ti;
                match_p_[ti] = p;
                break;
            }
        }
        if (match_p_[ti] < 0) complete = false;
    }
    if (complete) return true;
    ++stats_.matching_calls;
    const std::vector<int> full = saturating_matching(g_);
    if (full.empty()) return false;
    std::fill(match_p_.begin(), match_p_.end(), -1);
    std::fill(matched_to_.begin(), matched_to_.end(), -1);
    for (size_t i = 0; i < terms.size(); ++i) {
        const int ti = g_.terminal_index(terms[i]);
        match_p_[ti] = full[i];
        matched_to_[full[i]] = ti;
    }
    return true;
}

// ------------------------------------------------------------------------------------------ step (iii), [Lem 8.5]

// A matching M = {(p_i, t_i)} saturating T [Lem 7.8], secondary arcs e_i = (p_i, q_i) != (p_i, t_i) (fewest stored
// flows first, ties by arc id; any choice is valid), criticality of e_i evaluated exactly for v ∈ U_i = users of
// e_i (O1/O2). Lazy mode deletes the first e_i that is critical for no pair with psi(v,t) > 0 (psi stays a
// witness, [Lem 8.5]); otherwise — or always in literal mode — psi is replaced by the minimum-potential witness
// of [Prop 5.4] with the costs xi [Def 6.2], for which [Lem 8.4] guarantees such an e_i. Returns false iff T has
// no saturating matching (then [Alg 4] applies).
bool GLWeightedSolver::step_matching_delete() {
    const std::vector<int> terms = g_.terminals();
    const int kk = (int)terms.size();
    if (kk == 0) throw std::logic_error("[Alg 3] no live terminal although non-terminals remain");
    if (opt_.debug_asserts) {
        for (int t : terms)
            if (cap_[g_.terminal_index(t)] <= 0)
                throw std::logic_error("[Alg 3] step (iii) requires c_t > 0 for every terminal; terminal " + vs(t) + " has capacity 0");
        for (int p : g_.pre_terminals())
            if (g_.out_degree(p) < 2)
                throw std::logic_error("[Alg 3] step (iii) requires d^+(p) >= 2 for every pre-terminal; " + vs(p) + " has out-degree " + vs(g_.out_degree(p)));
    }
    if (!ensure_saturating_matching()) return false;  // [Alg 4]
    std::vector<int> matched(kk);
    for (int i = 0; i < kk; ++i) matched[i] = match_p_[g_.terminal_index(terms[i])];
    if (opt_.debug_asserts) {  // A2: distinct pre-terminals, each with an arc to its terminal
        std::vector<int> ps(matched);
        std::sort(ps.begin(), ps.end());
        if (std::adjacent_find(ps.begin(), ps.end()) != ps.end()) throw std::logic_error("A2 violated: matching reuses a pre-terminal");
    }
    // ---- secondary arcs (fewest users, ties by smallest arc id)
    std::vector<int> sec(kk, -1);
    std::vector<size_t> num_users(kk, 0);
    std::vector<std::pair<int, int>> pairs, secondary;
    for (int i = 0; i < kk; ++i) {
        const int p = matched[i], t = terms[i];
        const int a_match = g_.find_arc(p, t);
        if (a_match < 0) throw std::logic_error("A2 violated: matched pair (" + vs(p) + "," + vs(t) + ") is not an arc");
        int best = -1;
        size_t best_users = 0;
        for (int a : g_.out_arcs(p)) {
            if (a == a_match) continue;
            const size_t u = oracle_.num_users_of_arc(a);
            if (best < 0 || u < best_users || (u == best_users && a < best)) {
                best = a;
                best_users = u;
            }
        }
        if (best < 0)
            throw std::logic_error("[Alg 3] matched pre-terminal " + vs(p) + " has out-degree 1; step (ii) must contract it first");
        sec[i] = best;
        num_users[i] = best_users;
        pairs.emplace_back(p, t);
        secondary.emplace_back(p, g_.arc(best).head);
    }
    trace_.record("matching", JsonObject().raw("pairs", json_pairs(pairs)).raw("secondary", json_pairs(secondary)).build());
    if ((int)eval_slots_.size() < g_.k0()) eval_slots_.resize(g_.k0());

    std::vector<std::vector<int>> aff(kk);
    std::vector<char> evaluated(kk, 0);
    auto evaluate = [&](int i) {
        if (evaluated[i]) return;
        Stats::Timer t(stats_, "criticality");
        aff[i] = oracle_.evaluate_deletion(std::vector<int>{sec[i]}, eval_slots_[i], false);
        evaluated[i] = 1;
        if (opt_.debug_asserts) {  // [Lem 7.10] e_i is never critical for (v, t_i)
            const int ti = g_.terminal_index(terms[i]);
            for (int v : aff[i]) {
                const VertexFlow& f = eval_slots_[i][v];
                if (f.kappa < oracle_.kappa(v) && oracle_.ess(v).test(ti) && !f.ess.test(ti))
                    throw std::logic_error("[Lem 7.10] e_" + vs(i) + " = (" + vs(g_.arc(sec[i]).tail) + "," + vs(g_.arc(sec[i]).head) +
                                           ") is critical for (" + vs(v) + ", t_" + vs(ti) + "), its own matching terminal");
            }
        }
    };
    // e_i critical for some pair (v, t) with psi(v,t) > 0 [Def 6.1]? kappa unchanged -> nothing critical (O1);
    // kappa dropped -> exact Ess_{G\e_i}(v) (O2).
    auto critical_for_psi = [&](int i) -> bool {
        for (int v : aff[i]) {
            const VertexFlow& f = eval_slots_[i][v];
            if (f.kappa >= oracle_.kappa(v)) continue;
            for (const auto& e : psi_[v])
                if (!f.ess.test(e.first)) return true;
        }
        return false;
    };
    auto delete_secondary = [&](int i) {
        const int u = g_.arc(sec[i]).tail, x = g_.arc(sec[i]).head;
        delete_and_commit(std::vector<int>{sec[i]}, aff[i], eval_slots_[i]);  // stats_.deletions++ (oracle)
        trace_.record("delete_arc", JsonObject().num("u", u).num("v", x).build());
        trace_essential();
        check_after_operation("deletion of a non-critical secondary arc");
    };
    std::vector<int> order(kk);
    std::iota(order.begin(), order.end(), 0);
    if (opt_.lazy_shift) {
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return num_users[a] < num_users[b]; });
        for (int i : order) {
            evaluate(i);
            if (!critical_for_psi(i)) {
                delete_secondary(i);
                return true;
            }
        }
    } else {
        for (int i : order) evaluate(i);
    }
    // ---- every secondary arc is critical for a pair carrying weight (or literal mode): [Prop 5.4] + [Lem 8.4]
    if (trace_.enabled) {
        oracle_.refresh_all_cuts();
        std::vector<std::array<int, 3>> triples;
        for (int i = 0; i < kk; ++i)
            for (int v : aff[i]) {
                const VertexFlow& f = eval_slots_[i][v];
                if (f.kappa >= oracle_.kappa(v)) continue;
                const TermSet& before = oracle_.ess(v);
                for (int ti = 0; ti < g_.k0(); ++ti)
                    if (before.test(ti) && !f.ess.test(ti)) triples.push_back({i, v, term_vertex_[ti]});
            }
        trace_.record("criticality", JsonObject().raw("crit", json_triples(triples)).build());
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (attempt == 1) oracle_.refresh_all_cuts();  // exact sets: [Lem 8.5] / [Lem 8.4] apply verbatim
        build_cost_table(kk, aff);
        const SplitOutcome r = solve_split_assignment(true, true, false);
        if (!r.feasible) {
            if (attempt == 0) continue;
            throw std::logic_error("[Lem 8.5] min-cost split assignment infeasible although FESAC holds: only " + vs(r.value) +
                                   " weight units can be assigned to essential terminals within the capacities (exact essential sets)");
        }
        trace_.record("min_cost_split", JsonObject().raw("psi", psi_json()).num("cost", r.cost).build());
        if (opt_.debug_asserts) debug_check_witness("min-cost split assignment");
        int e_nc = -1;
        for (int i = 0; i < kk; ++i)
            if (!critical_for_psi(i)) {
                e_nc = i;
                break;
            }
        if (e_nc >= 0) {
            delete_secondary(e_nc);
            return true;
        }
        if (attempt == 1)
            throw std::logic_error("[Lem 8.4] every secondary arc is critical for a pair with positive split weight although psi "
                                   "has minimum potential over the exact essential sets");
    }
    throw std::logic_error("[Alg 3] step (iii) fell through");  // unreachable
}

// ------------------------------------------------------------------------------------------ step (iv), [Alg 4]

// RoundAndRemove: S = inclusion-minimal Hall-deficient terminal set (|PT(G,S)| = |S| - 1, [Lem 7.6]), t_S = S[0],
// M' = matching from S \ {t_S} onto PT(G,S) saturating both sides; parts[t_S] = {t_S}, parts[t] = {t, M'(t)}
// [Lem 8.6]; delete PT(G,S) and S [Lem 8.8]. parent[M'(t)] = orig_head of the arc (M'(t), t) (§13.4).
void GLWeightedSolver::step_round_and_remove() {
    Stats::Timer timer(stats_, "rounding");
    if (opt_.debug_asserts) {
        for (int t : g_.terminals())
            if (cap_[g_.terminal_index(t)] <= 0) throw std::logic_error("RoundAndRemove requires positive capacities");
        for (int p : g_.pre_terminals())
            if (g_.out_degree(p) < 2) throw std::logic_error("RoundAndRemove requires pre-terminal out-degree >= 2");
    }
    const std::vector<int> S = minimal_hall_deficient_set(g_);  // <= |T|^2 + 1 matching tests (paper_notes §8)
    stats_.matching_calls += (int64_t)g_.k() * (int64_t)g_.k() + 1;
    if (S.empty()) throw std::logic_error("RoundAndRemove called although T has a saturating matching");
    const int t_S = S[0];
    std::vector<int> rest(S.begin() + 1, S.end());
    ++stats_.matching_calls;
    const std::vector<int> M = saturating_matching(g_, rest);
    if (!rest.empty() && M.empty())
        throw std::logic_error("[Lem 7.6] S \\ {t_S} = " + json_list(rest) + " has no saturating matching");
    std::vector<int> pt = g_.pre_terminals_of(S);
    std::vector<int> matched(M);
    std::sort(matched.begin(), matched.end());
    if (matched != pt || pt.size() != S.size() - 1)
        throw std::logic_error("[Lem 7.6] matched pre-terminals " + json_list(matched) + " differ from PT(G,S) = " + json_list(pt) +
                               " (|S| = " + vs(S.size()) + ")");
    std::vector<std::pair<int, int>> pairs;
    JsonIntMap caps_json, weights_json;
    for (int t : S) caps_json.add(t, vs(cap_[g_.terminal_index(t)]));
    for (size_t j = 0; j < rest.size(); ++j) {
        const int t = rest[j], p = M[j];
        const int a = g_.find_arc(p, t);
        if (a < 0) throw std::logic_error("[Lem 7.6] matched pair (" + vs(p) + "," + vs(t) + ") is not an arc");
        part_[p] = g_.terminal_index(t);
        parent_[p] = g_.arc(a).orig_head;
        pairs.emplace_back(t, p);
        weights_json.add(p, vs(w_[p]));
        for (const auto& e : psi_[p]) recv_[e.first] -= e.second;
        psi_[p].clear();
    }
    trace_.record("round_and_remove", JsonObject().raw("S", json_list(S)).raw("pairs", json_pairs(pairs)).num("t_S", t_S)
                                          .raw("capacities", caps_json.build()).raw("weights", weights_json.build()).build());
    for (int p : pt) {  // increasing vertex order, as the reference
        for (int a : g_.in_arcs(p)) {
            push_degree_changed(g_.arc(a).tail);
            push_pt_candidate(g_.arc(a).tail);  // keeps the candidate bookkeeping dirty for these tails
        }
        mark_candidate_dirty(p);
        std::vector<int> in_tails;
        for (int a : g_.in_arcs(p)) in_tails.push_back(g_.arc(a).tail);
        g_.remove_vertex(p);
        penalty_update_vertex(p);  // C1: p is dead; its in-neighbours may have stopped being pre-terminals
        for (int u : in_tails) penalty_update_vertex(u);
        oracle_.after_vertex_removal(p);
    }
    for (int t : S) {
        const int ti = g_.terminal_index(t);
        if (recv_[ti] != 0)  // [Lem 7.7]: no survivor sends weight into S
            throw std::logic_error("[Lem 7.7] terminal " + vs(t) + " of the deficient set still receives " + vs(recv_[ti]) +
                                   " units from surviving vertices");
        for (int a : g_.in_arcs(t)) push_degree_changed(g_.arc(a).tail);
        g_.remove_terminal(t);
        if (opt_.routing_avoid) penalty_recompute_all();  // C1
        oracle_.after_terminal_removal(t);  // counts stats_.terminal_removals; recomputes every cut (O4)
    }
    cand_all_dirty_ = true;
    ++stats_.roundings;
    trace_essential();
    check_after_operation("rounding");
}

// ------------------------------------------------------------------------------------------ run

SolveResult GLWeightedSolver::run() {
    Stats::Timer total(stats_, "total");
    SolveResult res;
    const int n = g_.n(), k = g_.k0();
    if (trace_.enabled) {
        std::vector<std::pair<int, int>> arcs;
        for (int a = 0; a < g_.num_arc_ids(); ++a)
            if (g_.arc(a).alive) arcs.emplace_back(g_.arc(a).tail, g_.arc(a).head);
        trace_.record("init", JsonObject().num("n", n).num("k", k).raw("terminals", json_list(term_vertex_))
                                  .raw("capacities", json_list(cap_)).raw("arcs", json_pairs(arcs)).raw("weights", json_list(w_)).build());
    }
    int64_t total_cap = 0, total_weight = 0;
    for (int64_t c : cap_)
        if (__builtin_add_overflow(total_cap, c, &total_cap)) throw std::overflow_error("GLWeightedSolver: total capacity overflows int64");
    for (int v : g_.live_nonterminals())
        if (__builtin_add_overflow(total_weight, w_[v], &total_weight)) throw std::overflow_error("GLWeightedSolver: total weight overflows int64");
    const std::string guarantee = "; the theorem's guarantee does not apply (a partition may still exist)";
    if (total_weight > total_cap) {
        res.status = "precondition_failed";
        res.message = "Flow-Essential Split-Assignment Condition fails: the weights sum to " + vs(total_weight) +
                      " but the capacities sum to " + vs(total_cap) + " [Def 5.3]" + guarantee;
        return res;
    }
    // C1 (RESEARCH_NOTES E5): the penalty array is built before the first flows and maintained in both
    // modes (the diagnostics must be comparable); only "avoid" hands it to the oracle. See GLSolver::run.
    penalty_recompute_all();
    if (opt_.routing_avoid) oracle_.set_penalties(&penalty_);
    {
        Stats::Timer t(stats_, "compute_all");
        oracle_.compute_all();
    }
    stats_.penalized_users_initial = penalized_users();
    res.k_T_connected = true;
    int bad = -1;
    std::vector<int> empty_ess;
    for (int v : g_.live_nonterminals()) {
        if (oracle_.kappa(v) != k) {
            res.k_T_connected = false;
            if (bad < 0) bad = v;
        }
        if (!oracle_.ess(v).any() && empty_ess.size() < 10) empty_ess.push_back(v);
    }
    trace_essential();
    {
        const SplitOutcome r = solve_split_assignment(false, true, false);  // exact sets right after compute_all
        if (!r.feasible) {
            res.status = "precondition_failed";
            res.message = "Flow-Essential Split-Assignment Condition fails: no split witness exists (only " + vs(r.value) + " of " +
                          vs(total_weight) + " weight units can be assigned to essential terminals within the capacities)";
            if (!empty_ess.empty()) res.message += "; vertices with no essential terminal: " + json_list(empty_ess);
            res.message += guarantee;
            if (opt_.check_precondition && bad >= 0)
                res.message += "; graph is not " + vs(k) + "-T-connected: kappa(" + vs(bad) + ") = " + vs(oracle_.kappa(bad));
            return res;
        }
        trace_.record("min_cost_split", JsonObject().raw("psi", psi_json()).num("cost", 0).build());
        have_witness_ = true;
        penalty_recompute_all();  // C1: the kept arc of every pre-terminal p is now its psi target
        penalty_reroute_offenders();
        stats_.penalized_users_witness = penalized_users();
    }
    for (int i = 0; i < k; ++i)
        if (cap_[i] == 0) zero_terms_.push_back(i);
    for (int v : g_.live_nonterminals())
        if (g_.is_pre_terminal(v)) {
            push_degree_changed(v);
            push_pt_candidate(v);
        }
    check_after_operation("initial split witness");

    // ---- [Alg 3] main loop: every iteration removes a terminal, a vertex or >= 1 arc
    while (g_.num_live_nonterminals() > 0) {
        ++stats_.steps;
        sample_graph_size();
        if (step_remove_zero_terminal()) continue;                              // (i)   [Lem 8.1]
        if (step_contract_degree_one()) continue;                               // (ii)  [Lem 8.2]
        if (opt_.batch_unused_arcs && step_batch_unused_arcs()) continue;       // (iii-a) O1
        if (opt_.greedy_contraction && step_greedy_contraction()) continue;     // (iii-b) O5
        if (step_matching_delete()) continue;                                   // (iii) [Lem 8.5]
        step_round_and_remove();                                                // (iv)  [Alg 4]
    }

    // ---- result: parts by index, in-arborescence parents (§13.4)
    for (int v = 0; v < n; ++v)
        if (part_[v] < 0) throw std::logic_error("GLWeightedSolver: vertex " + vs(v) + " was never assigned to a part");
    res.status = "ok";
    res.message = "ok";
    res.assignment = part_;
    res.parent = parent_;
    res.witness.assign(n, -1);  // split witnesses live in the trace ("min_cost_split"); empty on a non-ok status
                                // (as assignment / parent, the same convention as GLSolver::run)
    if (trace_.enabled) {
        std::vector<std::vector<int>> parts(k);
        for (int v = 0; v < n; ++v) parts[part_[v]].push_back(v);
        JsonIntMap pm, parents;
        for (int i = 0; i < k; ++i) pm.add(term_vertex_[i], json_list(parts[i]));
        for (int v = 0; v < n; ++v)
            if (parent_[v] >= 0) parents.add(v, vs(parent_[v]));
        trace_.record("done", JsonObject().raw("parts", pm.build()).raw("parents", parents.build()).build());
    }
    return res;
}

}  // namespace glcore
