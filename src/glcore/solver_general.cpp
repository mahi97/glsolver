// GLSolver: the unweighted polynomial-time algorithm GLPartition [Alg 1] with ShiftAssignment [Alg 2]
// (docs/paper_notes.md §7) and the exact optimizations of docs/optimizations.md:
//   O1 batch deletion of arcs used by no stored flow (step iii-a), O2 warm-started criticality (inside the
//   EssentialOracle), O3 no recomputation after contraction, O4 terminal removal, O5 greedy contraction
//   (step iii-b), O6 lazy ShiftAssignment restricted to the users U_i of each secondary arc, O8 parallel
//   oracle, O9 bitsets. The paper's ShiftAssignment (step iii-c) is the exact fallback that guarantees
//   progress [Lem 7.5, 7.11].
// Certified-subset discipline (RESEARCH_NOTES.md P2): after every operation oracle_.ess(v) is a subset of the
// true Ess_G(v) that contains phi[v]. Every decision uses exact cuts: criticality for (v, phi(v)) is decided
// with the exact cut of G \ e (O2), and after a cycle shift phi(v_i) := t_i the new terminal is certified with
// one exact cut (refresh_cut) [Lem 7.9]. A deletion commit adopts, for an affected vertex whose kappa was
// restored, the subset stored at COMMIT time (a subset of Ess_{G\D}(v) by O1) rather than the copy taken at
// evaluation time, which predates such a refresh (delete_and_commit); the invariant is asserted there.
// Invariant A1 (phi is a witness: phi[v] ∈ Ess_G(v), exact capacity counts) holds after every operation;
// with opt.debug_asserts it is re-verified after each one (A1–A8 of paper_notes §7.2).
#include "solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "matching.hpp"
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
// O5 throttle (credit in quarter units): an attempt costs 4, a success earns 16, and while the credit is
// exhausted one quarter is refilled per step, so a run of failures settles at one attempt every 4 steps
// while instances on which the greedy contraction keeps succeeding stay at kGreedyCandidates per step.
constexpr int kGreedyCreditMax = 48;
constexpr int kGreedyCreditAttempt = 4;
constexpr int kGreedyCreditSuccess = 16;
// E4 predictor (RESEARCH_NOTES.md E4): candidates ranked by the cached score are re-ranked among the best
// kGreedyPreselect by an exact safety test; a candidate with an unsafe user is vetoed after
// kGreedyRiskyFailures failed attempts unless it becomes safe (see step_greedy_contraction).
constexpr int kGreedyPreselect = 16;
constexpr int kGreedyRiskyFailures = 2;

// E4 research instrumentation of O5 (RESEARCH_NOTES.md E4): with the environment variable GLCORE_GREEDY_LOG set
// to a file name, every greedy attempt appends one CSV row of cheap pre-attempt features and its outcome.
// Off by default (one pointer test per attempt); never changes what the solver does.
struct GreedyLog {
    std::FILE* f = nullptr;
    GreedyLog() {
        const char* path = std::getenv("GLCORE_GREEDY_LOG");
        if (!path || !*path) return;
        f = std::fopen(path, "a");
        if (!f) return;
        std::fseek(f, 0, SEEK_END);
        if (std::ftell(f) == 0)
            std::fputs("step,live_nt,live_arcs,k_live,credit,rank,n_cands,p,t,failures_p,dsize,outdeg_p,indeg_p,kappa_p,"
                       "users_total,users_distinct,max_arc_users,term_arcs,users_on_term_arcs,risky_simple,risky_exact,"
                       "users_full,users_tight,min_head_indeg,self_users,ok,affected,dropped,failing,eval_us\n", f);
    }
    ~GreedyLog() { if (f) std::fclose(f); }
};
std::FILE* greedy_log() {
    static GreedyLog g;
    return g.f;
}

}  // namespace

// ------------------------------------------------------------------------------------------ construction

GLSolver::GLSolver(int n, const std::vector<std::pair<int, int>>& arcs, const std::vector<int>& terminals,
                   const std::vector<int64_t>& capacities, SolverOptions opt)
    : g_(n, arcs, terminals), stats_(), trace_(), opt_(opt),
      oracle_(g_, stats_, resolve_threads(opt.threads), opt.seed),
      cap_(capacities), phi_(n, -1), parent_(n, -1), part_(n, -1), term_vertex_(terminals) {
    opt_.threads = resolve_threads(opt.threads);
    trace_.enabled = opt_.trace;
    const int k = g_.k0();
    if (k < 1) throw std::invalid_argument("GLSolver: at least one terminal is required");
    if ((int)cap_.size() != k)
        throw std::invalid_argument("GLSolver: " + vs(cap_.size()) + " capacities for " + vs(k) + " terminals");
    for (int i = 0; i < k; ++i) {
        if (cap_[i] < 0) throw std::invalid_argument("GLSolver: capacity of terminal " + vs(terminals[i]) + " is negative");
        part_[terminals[i]] = i;
    }
    in_deg_queue_.assign(n, 0);
    in_pt_candidates_.assign(n, 0);
    cand_dirty_.assign(n, 0);
    cand_users_.assign(n, -1);
    cand_dsize_.assign(n, 0);
    greedy_skip_until_.assign(n, 0);
    greedy_failures_.assign(n, 0);
    greedy_credit_ = kGreedyCreditMax;
    match_p_.assign(k, -1);
    matched_to_.assign(n, -1);
}

// ------------------------------------------------------------------------------------------ small helpers

void GLSolver::push_degree_changed(int v) {
    if (v < 0 || v >= g_.n() || in_deg_queue_[v]) return;
    in_deg_queue_[v] = 1;
    deg_queue_.push_back(v);
}

void GLSolver::push_pt_candidate(int v) {
    if (v < 0 || v >= g_.n()) return;
    mark_candidate_dirty(v);
    if (in_pt_candidates_[v]) return;
    in_pt_candidates_[v] = 1;
    pt_candidates_.push_back(v);
}

void GLSolver::mark_candidate_dirty(int v) {
    if (v < 0 || v >= g_.n() || cand_dirty_[v]) return;
    cand_dirty_[v] = 1;
    cand_dirty_list_.push_back(v);
}

// The arcs of v's stored flow are about to lose (or have just gained) a user: their tails' unused-arc sets
// and O5 scores may change. O(total path length) — the oracle already spent that on the flow itself.
void GLSolver::mark_flow_arcs_dirty(int v) {
    if (!g_.live(v) || g_.is_terminal(v)) return;
    for (const auto& path : oracle_.flow(v).paths)
        for (int a : path) mark_candidate_dirty(g_.arc(a).tail);
}

// One (live non-terminals, live arcs) sample per step; beyond kMaxSizeSamples every other sample is dropped
// and the sampling interval doubles, so the series stays bounded and evenly spaced.
void GLSolver::sample_graph_size() {
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

std::string GLSolver::capacities_json() const {
    JsonIntMap m;
    for (int t : g_.terminals()) m.add(t, vs(cap_[g_.terminal_index(t)]));
    return m.build();
}

std::string GLSolver::phi_json() const {
    JsonIntMap m;
    for (int v = 0; v < g_.n(); ++v)
        if (g_.live(v) && !g_.is_terminal(v) && phi_[v] >= 0) m.add(v, vs(term_vertex_[phi_[v]]));
    return m.build();
}

// "essential" event (§14): emitted after the initial computation and after every recomputation of Ess/κ
// (terminal removal, arc deletion; §13.3). The stored sets are certified subsets (P2); the trace carries
// the exact sets, so every cut is refreshed first (trace mode is for small instances).
void GLSolver::trace_essential() {
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

// A1 (and A6/A8 after deletions / terminal removals): phi is a witness of the current graph [Def 5.1].
// First the P2 invariant on the stored certified subsets (before any refresh makes them exact).
void GLSolver::debug_check_witness(const char* where) {
    for (int v : g_.live_nonterminals()) {
        const int ti = phi_[v];
        if (ti >= 0 && ti < g_.k0() && !oracle_.ess(v).test(ti))
            throw std::logic_error(std::string("P2 violated after ") + where + ": phi(" + vs(v) + ") = t_" + vs(ti) + " (vertex " +
                                   vs(term_vertex_[ti]) + ") is not in the certified essential subset of " + vs(v) + "; E = " +
                                   termset_json(oracle_.ess(v), term_vertex_));
    }
    oracle_.refresh_all_cuts();
    std::vector<int64_t> counts(g_.k0(), 0);
    for (int v : g_.live_nonterminals()) {
        const int ti = phi_[v];
        if (ti < 0 || ti >= g_.k0() || !g_.is_terminal(term_vertex_[ti]))
            throw std::logic_error(std::string("A1 violated after ") + where + ": phi(" + vs(v) + ") = " + vs(ti) +
                                   " is not a live terminal index");
        if (!oracle_.ess(v).test(ti))
            throw std::logic_error(std::string("A1 violated after ") + where + ": phi(" + vs(v) + ") = t_" + vs(ti) +
                                   " (vertex " + vs(term_vertex_[ti]) + ") is not essential for " + vs(v) + "; Ess = " +
                                   termset_json(oracle_.ess(v), term_vertex_));
        ++counts[ti];
    }
    for (int t : g_.terminals()) {
        const int ti = g_.terminal_index(t);
        if (counts[ti] != cap_[ti])
            throw std::logic_error(std::string("A1 violated after ") + where + ": terminal " + vs(t) + " receives " +
                                   vs(counts[ti]) + " vertices, capacity " + vs(cap_[ti]));
    }
}

void GLSolver::check_after_operation(const char* where) {
    if (opt_.debug_asserts) debug_check_witness(where);
}

// The witness survives deleting D iff phi(v) stays essential for every affected v (paper_notes §13.1):
// kappa unchanged -> the old certified set still holds (O1, generalized [Lem 4.3]); kappa dropped -> the
// oracle computed the exact cut of G \ D. Unaffected vertices are covered by O1.
bool GLSolver::deletion_keeps_witness(const std::vector<int>& affected, const std::vector<VertexFlow>& out) {
    for (int v : affected) {
        if (!g_.live(v) || g_.is_terminal(v)) continue;
        const VertexFlow& f = out[v];
        const int kappa_now = oracle_.kappa(v);
        if (f.kappa == kappa_now) continue;
        if (!f.cut_exact) throw std::logic_error("evaluate_deletion returned a dropped kappa without an exact cut for " + vs(v));
        if (!f.ess.test(phi_[v])) return false;
    }
    return true;
}

// Delete the (already evaluated) arc set D and let the oracle adopt the flows of G \ D. Tails are queued for
// step (ii) (their out-degree dropped). Counters `deletions` / `batched_deletions` are kept by the oracle.
// P2: for an affected v whose kappa was restored, evaluate_deletion copied the subset stored at EVALUATION time
// (O1: it stays a subset of Ess_{G\D}(v)); in ShiftAssignment a cycle shift between evaluation and commit
// re-certifies the new phi(v) only in the oracle's stored set (refresh_cut), so the copy may lack phi(v). The
// set stored NOW is a certified subset of Ess_G(v) that contains phi(v), hence also a subset of Ess_{G\D}(v)
// (O1, generalized [Lem 4.3]); adopting it keeps phi[v] ∈ oracle_.ess(v) across the commit.
void GLSolver::delete_and_commit(const std::vector<int>& D, const std::vector<int>& affected, std::vector<VertexFlow>& out) {
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
    oracle_.commit_deletion(D, affected, out);
    for (int v : affected) {
        mark_flow_arcs_dirty(v);  // the rerouted paths gained users
        // P2 invariant: the certified subset stored for every live non-terminal contains its witness.
        if (g_.live(v) && !g_.is_terminal(v) && (phi_[v] < 0 || !oracle_.ess(v).test(phi_[v])))
            throw std::logic_error("P2 violated after deleting " + vs(D.size()) + " arc(s): phi(" + vs(v) + ") = " +
                                   (phi_[v] < 0 ? std::string("none") : "t_" + vs(phi_[v]) + " (vertex " + vs(term_vertex_[phi_[v]]) + ")") +
                                   " is not in the certified essential set of " + vs(v) + "; E = " + termset_json(oracle_.ess(v), term_vertex_) +
                                   (oracle_.flow(v).cut_exact ? " (exact)" : " (subset)"));
    }
}

// ------------------------------------------------------------------------------------------ initial witness

// paper_notes §7.3: bipartite flow non-terminals -> terminal slots over the exact essential sets [Def 5.1].
bool GLSolver::find_initial_witness() {
    const std::vector<int> verts = g_.live_nonterminals();
    std::vector<std::vector<int>> allowed(verts.size());
    std::vector<int> empty_ess;
    for (size_t i = 0; i < verts.size(); ++i) {
        allowed[i] = oracle_.ess(verts[i]).to_list();  // exact right after compute_all
        if (allowed[i].empty() && empty_ess.size() < 10) empty_ess.push_back(verts[i]);
    }
    const AssignmentResult r = saturating_assignment(allowed, cap_);
    if (!r.saturated) {
        witness_message_ = "Flow-Essential Assignment Condition fails: no witness assignment exists (only " + vs(r.value) +
                           " of " + vs(verts.size()) + " non-terminals can be assigned to an essential terminal within the capacities)";
        if (!empty_ess.empty()) witness_message_ += "; vertices with no essential terminal: " + json_list(empty_ess);
        witness_message_ += "; the theorem's guarantee does not apply (a partition may still exist)";
        return false;
    }
    for (size_t i = 0; i < verts.size(); ++i) phi_[verts[i]] = r.assignment[i];
    return true;
}

// ------------------------------------------------------------------------------------------ step (i)

// [Lem 7.2]: a terminal with c_t = 0 receives no vertex, so removing it keeps FEAC with the same phi
// ([Lem 7.1]: old essential terminals survive; certified subsets stay subsets). The oracle recomputes every
// cut exactly (O4; new essential terminals may appear, §13.3).
bool GLSolver::step_remove_zero_terminal() {
    while (zero_head_ < zero_terms_.size()) {
        const int ti = zero_terms_[zero_head_++];
        const int t = term_vertex_[ti];
        if (!g_.is_terminal(t) || cap_[ti] != 0) continue;
        Stats::Timer timer(stats_, "terminal_removal");
        for (int a : g_.in_arcs(t)) push_degree_changed(g_.arc(a).tail);
        g_.remove_terminal(t);
        oracle_.after_terminal_removal(t);  // counts stats_.terminal_removals
        cand_all_dirty_ = true;             // every flow may have been rerouted (O4); the matching entry of t
                                            // is dropped by ensure_saturating_matching's validation
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

// Out-degree-1 pre-terminals are found through the queue of vertices whose out-degree changed (initially all
// pre-terminals): a vertex only reaches out-degree 1 into a terminal by losing arcs (deletion: tails queued;
// terminal removal: in-neighbours queued) or by a contraction redirecting its arc onto a terminal
// (in-neighbours of the contracted vertex queued).
bool GLSolver::step_contract_degree_one() {
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

// [Lem 7.4] contract p (single arc (p,t)) into t, c_t -= 1, phi \ {p}. Ess(p) = {t} forces phi(p) = t.
// kappa / Ess of every remaining vertex are unchanged (§13.2, O3), so nothing is recomputed (A7 checks it).
void GLSolver::contract_into(int p, int t) {
    Stats::Timer timer(stats_, "contract");
    const int ti = g_.terminal_index(t);
    if (phi_[p] != ti)
        throw std::logic_error("[Lem 7.4] pre-terminal " + vs(p) + " has the single out-arc (" + vs(p) + "," + vs(t) +
                               ") but phi(" + vs(p) + ") = " + (phi_[p] < 0 ? std::string("none") : "t_" + vs(phi_[p]) + " (vertex " + vs(term_vertex_[phi_[p]]) + ")") +
                               "; Ess(p) = {t} forces phi(p) = t");
    if (cap_[ti] <= 0)
        throw std::logic_error("[Lem 7.4] contracting " + vs(p) + " into terminal " + vs(t) + " whose capacity is already 0");
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
    cap_[ti] -= 1;
    if (cap_[ti] == 0) zero_terms_.push_back(ti);
    part_[p] = ti;
    phi_[p] = -1;
    for (int u : preds) {
        push_degree_changed(u);   // duplicate merge may have lowered d^+(u)
        push_pt_candidate(u);     // (u,p) became (u,t): u is a pre-terminal now
    }
    trace_.record("contract", JsonObject().num("p", p).num("t", t).num("parent", parent_[p]).raw("capacities", capacities_json()).build());
    if (opt_.debug_asserts) {
        oracle_.refresh_all_cuts();
        for (const auto& e : before) {
            const int v = e.first;
            if (oracle_.kappa(v) != e.second.first || oracle_.ess(v) != e.second.second)
                throw std::logic_error("A7 violated: kappa/Ess of " + vs(v) + " changed by the contraction of " + vs(p) + " into " + vs(t) +
                                       " (kappa " + vs(e.second.first) + " -> " + vs(oracle_.kappa(v)) + ", Ess " +
                                       termset_json(e.second.second, term_vertex_) + " -> " + termset_json(oracle_.ess(v), term_vertex_) + ")");
        }
        debug_check_witness("contraction");
    }
}

// ------------------------------------------------------------------------------------------ candidate scans

// Recompute, for every dirty candidate p (or all of them after a terminal removal): eligibility (live
// pre-terminal; dropped otherwise), the unused arcs of p (O1 batch, only if (p, phi(p)) ∈ E) and the O5
// score (flows using D_p = out(p) \ {(p, phi(p))}). Cost O(Σ_{dirty p} d^+(p) + user-list compaction).
void GLSolver::refresh_dirty_candidates() {
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
            continue;
        }
        const int a_phi = g_.find_arc(p, term_vertex_[phi_[p]]);
        if (a_phi < 0) {
            cand_users_[p] = -1;
            continue;
        }
        int64_t users = 0;
        for (int a : g_.out_arcs(p)) {
            if (a == a_phi) continue;
            const size_t u = oracle_.num_users_of_arc(a);
            users += (int64_t)u;
            if (u == 0 && opt_.batch_unused_arcs) batch_arcs_.push_back(a);
        }
        cand_users_[p] = g_.out_degree(p) >= 2 ? users : -1;
        cand_dsize_[p] = g_.out_degree(p) - 1;
    }
    // compact the candidate list (dropped entries) once it is mostly dead
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

// D = alive arcs (p,q) of pre-terminals p with (p, phi(p)) ∈ E, q != phi(p), used by no stored flow. By O1
// deleting all of D at once keeps every phi(v) essential (no flow, hence no criticality, involves D).
bool GLSolver::step_batch_unused_arcs() {
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
    if (!deletion_keeps_witness(affected, scratch_slot_)) return false;
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

// Try to make a pre-terminal p with (p, phi(p)) ∈ E out-degree 1 by deleting D_p = out(p) \ {(p, phi(p))};
// valid iff phi(v) ∈ Ess_{G\D_p}(v) for every v (§13.1), tested exactly on the affected vertices only (O1/O2).
// Candidates are pre-selected by (flows using D_p, |D_p|, p) from the cached scores, then re-ranked by the E4
// predictor below; a failed candidate backs off exponentially and the credit throttle bounds the work spent
// on instances where O5 rarely succeeds.
//
// E4 predictor (RESEARCH_NOTES.md E4). A user v != p of an arc a = (p, q) in D_p is SAFE iff no path of F_v
// other than the one through a ends at t = phi(p). Deleting D_p then cannot lower kappa(v): remove the path
// through a, and its prefix up to p followed by the arc (p, t) is an augmenting path of the residual network
// of G \ D_p (the prefix is vertex-disjoint from the other paths, t is a terminal, hence a sink with no
// out-arcs, and no remaining path ends at t), so the warm-started re-augmentation of O2 restores kappa(v) and
// the certified subset survives (O1). The tail p itself always passes (Ess_{G\D_p}(p) = {t}). Hence a
// candidate whose users are all safe ("certain") passes the exact test with probability 1 — this is a
// theorem, not a heuristic, and the test costs O(sum of |users| * kappa) per candidate. Measured on random
// 4-regular graphs (14 474 logged attempts, n = 2 000): certain candidates succeed in 100 % of attempts, a
// candidate with an unsafe user in 16–25 %, and the success rate collapses with the candidate's own failure
// count (median 0 previous failures for successes, 7 for failures: the capped backoff keeps re-attempting
// chronic failures). Rule: among the kGreedyPreselect best-scored candidates try the certain ones first
// (fewest users), then the risky ones by (previous failures, users); a risky candidate is vetoed once it has
// failed more than kGreedyRiskyFailures times, until it becomes certain. On the logs this keeps 95–97 % of the
// successes and skips 71–81 % of the failures. Exactness is untouched: the predictor only chooses which
// FEAC-preserving deletion is *tested*; every deletion is still validated by evaluate_deletion (§13.1).
// GLCORE_GREEDY_PREDICTOR=0 in the environment disables the re-ranking and the veto (same-binary A/B).
bool GLSolver::step_greedy_contraction() {
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
    static const bool predictor_on = [] {
        const char* e = std::getenv("GLCORE_GREEDY_PREDICTOR");
        return !(e && *e == '0');
    }();
    // Unsafe users of D_p = out(p) \ {a_phi} (see the comment above): users v != p of an arc a of D_p with a
    // path of F_v ending at t that is not the path through a (paths end at distinct terminals: at most one).
    auto count_unsafe_users = [&](int p, int t, int a_phi) -> int {
        int unsafe = 0;
        for (int a : g_.out_arcs(p)) {
            if (a == a_phi) continue;
            for (int v : oracle_.users_of_arc(a)) {
                if (v == p) continue;
                const VertexFlow& f = oracle_.flow(v);
                for (size_t i = 0; i < f.paths.size(); ++i) {
                    if (f.path_terminal[i] != t) continue;
                    if (std::find(f.paths[i].begin(), f.paths[i].end(), a) == f.paths[i].end()) ++unsafe;
                    break;
                }
            }
        }
        return unsafe;
    };
    struct Ranked {
        int risky;      // 0: certain (every user safe), 1: some unsafe user
        int failures;   // previous failed attempts (0 for certain candidates: irrelevant)
        int64_t users;
        int dsize, p;
        bool operator<(const Ranked& o) const {
            return std::tie(risky, failures, users, dsize, p) < std::tie(o.risky, o.failures, o.users, o.dsize, o.p);
        }
    };
    const size_t pre = std::min<size_t>(predictor_on ? (size_t)kGreedyPreselect : (size_t)budget, cands.size());
    std::partial_sort(cands.begin(), cands.begin() + (std::ptrdiff_t)pre, cands.end());
    std::vector<Ranked> ranked;
    ranked.reserve(pre);
    for (size_t c = 0; c < pre; ++c) {
        const int p = cands[c].p;
        const int t = term_vertex_[phi_[p]];
        const int a_phi = g_.find_arc(p, t);
        if (a_phi < 0 || g_.out_degree(p) < 2) continue;  // stale score (cannot happen after a refresh)
        if (!predictor_on) {
            ranked.push_back(Ranked{0, 0, cands[c].users, cands[c].dsize, p});
            continue;
        }
        const int unsafe = count_unsafe_users(p, t, a_phi);
        if (unsafe > 0 && greedy_failures_[p] > kGreedyRiskyFailures) continue;  // veto (until it becomes certain)
        ranked.push_back(Ranked{unsafe > 0 ? 1 : 0, unsafe > 0 ? greedy_failures_[p] : 0, cands[c].users, cands[c].dsize, p});
    }
    if (predictor_on) std::sort(ranked.begin(), ranked.end());
    const size_t tries = std::min<size_t>((size_t)budget, ranked.size());
    for (size_t c = 0; c < tries; ++c) {
        const int p = ranked[c].p;
        const int t = term_vertex_[phi_[p]];
        const int a_phi = g_.find_arc(p, t);
        if (a_phi < 0 || g_.out_degree(p) < 2) continue;
        ++stats_.greedy_attempts;
        const int credit_before = greedy_credit_;
        greedy_credit_ -= kGreedyCreditAttempt;
        std::vector<int> D;
        std::vector<std::pair<int, int>> pairs;
        for (int a : g_.out_arcs(p))
            if (a != a_phi) {
                D.push_back(a);
                pairs.emplace_back(p, g_.arc(a).head);
            }
        // ---- E4 instrumentation (GLCORE_GREEDY_LOG): cheap features of this attempt, see GreedyLog
        std::FILE* glog = greedy_log();
        std::string feat;
        std::chrono::steady_clock::time_point eval_t0;
        if (glog) {
            int64_t users_total = 0, max_arc_users = 0, term_arcs = 0, users_on_term = 0, self_users = 0;
            int64_t risky_simple = 0, risky_exact = 0, users_full = 0, users_tight = 0;
            int min_head_indeg = -1;
            std::vector<int> distinct;
            for (int a : D) {
                const int q = g_.arc(a).head;
                const bool q_term = g_.is_terminal(q);
                if (q_term) ++term_arcs;
                const int hd = g_.in_degree(q);
                if (min_head_indeg < 0 || hd < min_head_indeg) min_head_indeg = hd;
                const std::vector<int> us = oracle_.users_of_arc(a);
                users_total += (int64_t)us.size();
                max_arc_users = std::max<int64_t>(max_arc_users, (int64_t)us.size());
                for (int v : us) {
                    if (v == p) { ++self_users; continue; }
                    distinct.push_back(v);
                    if (q_term) ++users_on_term;
                    const VertexFlow& f = oracle_.flow(v);
                    if (f.kappa == g_.k()) ++users_full;
                    if (g_.out_degree(v) == f.kappa) ++users_tight;
                    bool simple = false, exact = false;
                    for (size_t i = 0; i < f.paths.size(); ++i) {
                        if (f.path_terminal[i] != t) continue;
                        simple = true;
                        if (std::find(f.paths[i].begin(), f.paths[i].end(), a) == f.paths[i].end()) exact = true;
                    }
                    risky_simple += simple;
                    risky_exact += exact;
                }
            }
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            feat = vs(stats_.steps) + "," + vs(g_.num_live_nonterminals()) + "," + vs(g_.num_live_arcs()) + "," + vs(g_.k()) + "," +
                   vs(credit_before) + "," + vs((int64_t)c) + "," + vs((int64_t)cands.size()) + "," + vs(p) + "," + vs(t) + "," +
                   vs(greedy_failures_[p]) + "," + vs((int64_t)D.size()) + "," + vs(g_.out_degree(p)) + "," + vs(g_.in_degree(p)) + "," +
                   vs(oracle_.kappa(p)) + "," + vs(users_total) + "," + vs((int64_t)distinct.size()) + "," + vs(max_arc_users) + "," +
                   vs(term_arcs) + "," + vs(users_on_term) + "," + vs(risky_simple) + "," + vs(risky_exact) + "," + vs(users_full) + "," +
                   vs(users_tight) + "," + vs(min_head_indeg) + "," + vs(self_users);
            eval_t0 = std::chrono::steady_clock::now();
        }
        std::vector<int> affected = oracle_.evaluate_deletion(D, scratch_slot_, false);
        const bool keeps = deletion_keeps_witness(affected, scratch_slot_);
        if (glog) {
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - eval_t0).count();
            int64_t dropped = 0, failing = 0;
            for (int v : affected) {
                if (!g_.live(v) || g_.is_terminal(v)) continue;
                const VertexFlow& f = scratch_slot_[v];
                if (f.kappa < oracle_.kappa(v)) {
                    ++dropped;
                    if (!f.ess.test(phi_[v])) ++failing;
                }
            }
            std::fprintf(glog, "%s,%d,%lld,%lld,%lld,%.0f\n", feat.c_str(), keeps ? 1 : 0, (long long)affected.size(),
                         (long long)dropped, (long long)failing, us);
        }
        if (!keeps) {
            const int f = ++greedy_failures_[p];
            greedy_skip_until_[p] = stats_.steps + std::min(kGreedyMaxBackoff, 1 << std::min(f, 6));
            if (greedy_credit_ <= 0) break;
            continue;
        }
        greedy_credit_ = std::min(kGreedyCreditMax, greedy_credit_ + kGreedyCreditSuccess);
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

// Keep the previous saturating matching: drop pairs whose pre-terminal died or whose arc was deleted, try to
// re-match each unmatched terminal with a free in-neighbour (an augmenting path of length one, O(in-degree)),
// and only when that fails recompute the matching from scratch (Hopcroft–Karp). Returns false iff no
// saturating matching exists.
bool GLSolver::ensure_saturating_matching() {
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

// ------------------------------------------------------------------------------------------ step (iii-c), [Alg 2]

// ShiftAssignment [Alg 2] with O6: a matching M = {(p_i, t_i)} saturating T [Lem 7.8], secondary arcs
// e_i = (p_i, q_i) != (p_i, t_i) (fewest stored flows first, ties by arc id; any choice is valid), the
// criticality of e_i for (v, phi(v)) evaluated exactly for v ∈ U_i = users of e_i (O1/O2), cycle shifts of the
// reassignment graph until some e_i is critical for no pair (v, phi(v)) [Lem 7.11], then delete it [Lem 7.5].
void GLSolver::step_shift_assignment() {
    Stats::Timer timer(stats_, "shift");
    ++stats_.shift_calls;
    const std::vector<int> terms = g_.terminals();
    const int kk = (int)terms.size();
    if (kk == 0) throw std::logic_error("[Alg 2] no live terminal although non-terminals remain");
    if (opt_.debug_asserts) {
        for (int t : terms)
            if (cap_[g_.terminal_index(t)] <= 0)
                throw std::logic_error("[Alg 2] requires c_t > 0 for every terminal; terminal " + vs(t) + " has capacity 0");
        for (int p : g_.pre_terminals())
            if (g_.out_degree(p) < 2)
                throw std::logic_error("[Alg 2] requires d^+(p) >= 2 for every pre-terminal; " + vs(p) + " has out-degree " + vs(g_.out_degree(p)));
    }
    // ---- [Alg 2] line 1: saturating matching [Lem 7.8]
    if (!ensure_saturating_matching())
        throw std::logic_error("[Lem 7.8] no matching from the pre-terminals saturating the " + vs(kk) +
                               " live terminals although FEAC holds with positive capacities");
    std::vector<int> matched(kk);
    for (int i = 0; i < kk; ++i) matched[i] = match_p_[g_.terminal_index(terms[i])];
    if (opt_.debug_asserts) {  // A2: distinct pre-terminals, each with an arc to its terminal
        std::vector<int> ps(matched);
        std::sort(ps.begin(), ps.end());
        if (std::adjacent_find(ps.begin(), ps.end()) != ps.end()) throw std::logic_error("A2 violated: matching reuses a pre-terminal");
    }
    // ---- [Alg 2] line 2: secondary arcs (O6 heuristic: fewest users, ties by smallest arc id)
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
            throw std::logic_error("[Alg 2] matched pre-terminal " + vs(p) + " has out-degree 1; step (ii) must contract it first");
        sec[i] = best;
        num_users[i] = best_users;
        pairs.emplace_back(p, t);
        secondary.emplace_back(p, g_.arc(best).head);
    }
    trace_.record("matching", JsonObject().raw("pairs", json_pairs(pairs)).raw("secondary", json_pairs(secondary)).build());
    if ((int)eval_slots_.size() < g_.k0()) eval_slots_.resize(g_.k0());

    std::vector<std::vector<int>> aff(kk);
    std::vector<char> evaluated(kk, 0);
    // crit(e_i, v, phi(v)) [Def 6.1] for v ∈ U_i: phi(v) ∈ Ess_G(v) holds (A1), so e_i is critical iff
    // phi(v) ∉ Ess_{G\e_i}(v); when kappa did not drop, Ess_G(v) ⊆ Ess_{G\e_i}(v) (O2), else the cut is exact.
    auto crit = [&](int i, int v) -> bool {
        const VertexFlow& f = eval_slots_[i][v];
        return f.kappa < oracle_.kappa(v) && !f.ess.test(phi_[v]);
    };
    auto has_critical = [&](int i) -> bool {
        for (int v : aff[i])
            if (crit(i, v)) return true;
        return false;
    };
    auto evaluate = [&](int i) {
        if (evaluated[i]) return;
        aff[i] = oracle_.evaluate_deletion(std::vector<int>{sec[i]}, eval_slots_[i], false);
        evaluated[i] = 1;
    };
    auto delete_secondary = [&](int i) {
        const int u = g_.arc(sec[i]).tail, v = g_.arc(sec[i]).head;
        delete_and_commit(std::vector<int>{sec[i]}, aff[i], eval_slots_[i]);  // stats_.deletions++ (oracle)
        trace_.record("delete_arc", JsonObject().num("u", u).num("v", v).build());
        trace_essential();
        check_after_operation("deletion of a non-critical secondary arc");
    };
    // Evaluation order: lazy (O6) = fewest users first, stop at the first non-critical arc; literal = the
    // whole table first, then the first non-critical index. An arc used by no flow is non-critical for every
    // pair (O1) and comes first in both modes.
    std::vector<int> order(kk);
    std::iota(order.begin(), order.end(), 0);
    if (opt_.lazy_shift)
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return num_users[a] < num_users[b]; });
    else
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return (num_users[a] == 0) > (num_users[b] == 0); });
    if (opt_.lazy_shift) {
        for (int i : order) {
            evaluate(i);
            if (!has_critical(i)) {
                delete_secondary(i);
                return;
            }
        }
    } else {
        for (int i : order) evaluate(i);
        for (int i : order)
            if (!has_critical(i)) {
                delete_secondary(i);
                return;
            }
    }
    // ---- every secondary arc is critical for some (v, phi(v)): cycle shifts [Alg 2] lines 5–12
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
    auto potential = [&]() -> int64_t {  // [Def 6.3] Φ(φ) = Σ_v ξ_v(φ(v)) restricted to the entries that can be true
        int64_t s = 0;
        for (int i = 0; i < kk; ++i)
            for (int v : aff[i])
                if (crit(i, v)) ++s;
        return s;
    };
    const int64_t max_iter = (int64_t)kk * g_.num_live_nonterminals() + 1;
    std::vector<std::pair<int, int>> pred(g_.k0(), {-1, -1});  // terminal index -> (phi(v_i) index, v_i)
    std::vector<int> stamp(g_.k0(), 0);
    int gen = 0;
    for (int64_t iter = 0;; ++iter) {
        if (iter >= max_iter)
            throw std::logic_error("[Lem 7.11] more than k|V\\T| + 1 = " + vs(max_iter) + " cycle shifts without a non-critical secondary arc");
        // reassignment graph R: arc phi(v_i) -> t_i labelled v_i, v_i = first critical vertex of e_i
        for (int i = 0; i < kk; ++i) {
            int v_i = -1;
            for (int v : aff[i])
                if (crit(i, v)) { v_i = v; break; }
            if (v_i < 0) throw std::logic_error("[Alg 2] secondary arc e_" + vs(i) + " lost its critical vertex inside the shift loop");
            const int ti = g_.terminal_index(terms[i]);
            if (phi_[v_i] == ti)
                throw std::logic_error("[Lem 7.10] e_" + vs(i) + " is critical for (" + vs(v_i) + ", t_" + vs(ti) + ") — its own matching terminal");
            pred[ti] = {phi_[v_i], v_i};
        }
        // walk the unique in-arcs backwards until a terminal repeats (§13.5): a cycle of length >= 2
        ++gen;
        std::vector<int> walk;
        int cur = g_.terminal_index(terms[0]);
        while (stamp[cur] != gen) {
            stamp[cur] = gen;
            walk.push_back(cur);
            cur = pred[cur].first;
        }
        std::vector<int> cycle(std::find(walk.begin(), walk.end(), cur), walk.end());
        if (cycle.size() < 2) throw std::logic_error("[Lem 7.10] self-loop in the reassignment graph");
        const bool measure = opt_.debug_asserts || trace_.enabled;
        const int64_t before = measure ? potential() : 0;
        std::vector<std::array<int, 3>> changes;  // (v, old terminal, new terminal) as vertices
        for (int ti : cycle) {
            const int v = pred[ti].second, old = phi_[v];
            if (old != pred[ti].first) throw std::logic_error("[Alg 2] vertex " + vs(v) + " shifted twice in one cycle");
            phi_[v] = ti;  // [Lem 7.11]
            mark_candidate_dirty(v);  // (v, phi(v)) changed: its O1/O5 data must be recomputed
            changes.push_back({v, term_vertex_[old], term_vertex_[ti]});
        }
        // P2(b): certify t_i for v_i with one exact cut; [Lem 7.9] guarantees membership.
        for (int ti : cycle) {
            const int v = pred[ti].second;
            oracle_.refresh_cut(v);
            if (!oracle_.ess(v).test(ti))
                throw std::logic_error("[Lem 7.9] after the cycle shift t_" + vs(ti) + " (vertex " + vs(term_vertex_[ti]) +
                                       ") is not essential for " + vs(v) + "; Ess = " + termset_json(oracle_.ess(v), term_vertex_));
        }
        ++stats_.cycle_shifts;
        const int64_t after = measure ? potential() : 0;
        if (trace_.enabled) {
            std::vector<std::array<int, 3>> arcs;
            for (int i = 0; i < kk; ++i) {
                const int ti = g_.terminal_index(terms[i]);
                arcs.push_back({term_vertex_[pred[ti].first], terms[i], pred[ti].second});
            }
            std::vector<int> cyc;
            for (int ti : cycle) cyc.push_back(term_vertex_[ti]);
            trace_.record("reassignment_graph", JsonObject().raw("arcs", json_triples(arcs)).raw("cycle", json_list(cyc)).build());
            trace_.record("cycle_shift", JsonObject().raw("changes", json_triples(changes)).num("potential_before", before)
                                             .num("potential_after", after).raw("phi", phi_json()).build());
        }
        if (opt_.debug_asserts) {
            if (!(after < before))  // A5
                throw std::logic_error("[Lem 7.11] potential did not decrease: " + vs(before) + " -> " + vs(after));
            debug_check_witness("cycle shift");
        }
        for (int i : order)
            if (!has_critical(i)) {
                delete_secondary(i);
                return;
            }
    }
}

// ------------------------------------------------------------------------------------------ run

SolveResult GLSolver::run() {
    Stats::Timer total(stats_, "total");
    SolveResult res;
    const int n = g_.n(), k = g_.k0();
    if (trace_.enabled) {
        std::vector<std::pair<int, int>> arcs;
        for (int a = 0; a < g_.num_arc_ids(); ++a)
            if (g_.arc(a).alive) arcs.emplace_back(g_.arc(a).tail, g_.arc(a).head);
        trace_.record("init", JsonObject().num("n", n).num("k", k).raw("terminals", json_list(term_vertex_))
                                  .raw("capacities", json_list(cap_)).raw("arcs", json_pairs(arcs)).build());
    }
    int64_t total_cap = 0;
    for (int64_t c : cap_) total_cap += c;
    if (total_cap != g_.num_live_nonterminals()) {
        res.status = "precondition_failed";
        res.message = "Flow-Essential Assignment Condition fails: the capacities sum to " + vs(total_cap) + " but there are " +
                      vs(g_.num_live_nonterminals()) + " non-terminals [Def 5.1]";
        return res;
    }
    {
        Stats::Timer t(stats_, "compute_all");
        oracle_.compute_all();
    }
    res.k_T_connected = true;
    int bad = -1;
    for (int v : g_.live_nonterminals())
        if (oracle_.kappa(v) != k) {
            res.k_T_connected = false;
            if (bad < 0) bad = v;
        }
    trace_essential();
    {
        Stats::Timer t(stats_, "witness");
        if (!find_initial_witness()) {
            res.status = "precondition_failed";
            res.message = witness_message_;
            if (opt_.check_precondition && bad >= 0)
                res.message += "; graph is not " + vs(k) + "-T-connected: kappa(" + vs(bad) + ") = " + vs(oracle_.kappa(bad));
            return res;
        }
    }
    initial_witness_ = phi_;
    trace_.record("witness", JsonObject().raw("phi", phi_json()).build());
    for (int i = 0; i < k; ++i)
        if (cap_[i] == 0) zero_terms_.push_back(i);
    for (int v : g_.live_nonterminals())
        if (g_.is_pre_terminal(v)) {
            push_degree_changed(v);
            push_pt_candidate(v);
        }
    if (opt_.debug_asserts) debug_check_witness("initial witness");

    // ---- [Alg 1] main loop: every iteration removes a terminal, a vertex or >= 1 arc
    while (g_.num_live_nonterminals() > 0) {
        ++stats_.steps;
        sample_graph_size();
        if (step_remove_zero_terminal()) continue;                              // (i)   [Lem 7.2]
        if (step_contract_degree_one()) continue;                               // (ii)  [Lem 7.4]
        if (opt_.batch_unused_arcs && step_batch_unused_arcs()) continue;       // (iii-a) O1
        if (opt_.greedy_contraction && step_greedy_contraction()) continue;     // (iii-b) O5
        step_shift_assignment();                                                // (iii-c) [Alg 2], [Lem 7.5]
    }

    // ---- result: parts by index, in-arborescence parents (§13.4), the initial witness
    for (int v = 0; v < n; ++v)
        if (part_[v] < 0) throw std::logic_error("GLSolver: vertex " + vs(v) + " was never assigned to a part");
    res.status = "ok";
    res.message = "ok";
    res.assignment = part_;
    res.parent = parent_;
    res.witness.assign(n, -1);
    for (int v = 0; v < n; ++v)
        if (g_.terminal_index(v) < 0 && v < (int)initial_witness_.size()) res.witness[v] = initial_witness_[v];
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
