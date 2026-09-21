// EssentialOracle: per-vertex maximum flows, certified essential sets and tightest cuts for every live
// non-terminal (paper_notes §3.1 [Prop 4.2], §4 [Lem 4.1]; docs/optimizations.md O1–O4, O8; RESEARCH_NOTES P2).
//
// Validity discipline. `valid_[v]` says that flows_[v] is a maximum flow of the current graph and that
// flows_[v].ess is a certified subset of Ess_G(v) (exact when cut_exact), PROVIDED the oracle has processed
// every graph mutation: `synced_version_ == g.version()`. Each hook (commit_deletion / after_contraction /
// after_terminal_removal / after_vertex_removal) is called right after the corresponding mutation(s) and
// resynchronizes; a query that finds the graph version ahead of `synced_version_` concludes that the graph
// was changed behind the oracle's back and invalidates every flow (they are recomputed lazily). This makes
// "all unaffected flows stay valid" (O1, O3) an O(1) statement instead of an O(n) version bump per commit.
// The per-flow `version` field is refreshed on access and is informational.
//
// Indices. arc_users_[a] and vertex_users_[x] hold (v, stamp) entries; an entry is live iff
// flow_stamp_[v] == stamp and v's flow is current. index_users(v) bumps the stamp and appends the entries
// of the new paths; stale entries are compacted away when a list is read. Both indices cost O(total path
// length) — never O(n) per query.
//
// Parallelism (O8): std::thread workers with an atomic work counter; every worker owns one FlowEngine
// Scratch and writes only per-vertex slots, so results are deterministic regardless of the thread count;
// sequential post-processing runs in increasing vertex order. Stats counters are accumulated per worker.
#include "essential.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "stats.hpp"

namespace glcore {

namespace {
const std::vector<int> kNoForbidden;

std::string vstr(int v) { return std::to_string(v); }
}  // namespace

EssentialOracle::EssentialOracle(Graph& g, Stats& stats, int threads, int seed)
    : g_(g), stats_(stats), threads_(threads < 1 ? 1 : threads), seed_(seed), engine_(g),
      flows_(g.n()), arc_users_(g.num_arc_ids()), flow_stamp_(g.n(), 0), valid_(g.n(), 0),
      synced_version_(g.version()), vertex_users_(g.n()), cut_epoch_(g.n(), 0) {}

// ---------------------------------------------------------------------------------------------- helpers

FlowEngine::Scratch& EssentialOracle::scratch(int worker) {
    while ((int)scratch_.size() <= worker) scratch_.push_back(engine_.make_scratch());
    return *scratch_[worker];
}

void EssentialOracle::ensure_synced() {
    if (synced_version_ == g_.version()) return;
    invalidate_all();
    synced_version_ = g_.version();
}

// A hook expects exactly `count` mutations since the last sync; anything else means the caller performed
// other mutations without telling us, and the safe fallback is to invalidate everything.
bool EssentialOracle::expect_mutations(uint64_t count) {
    const bool ok = (g_.version() == synced_version_ + count);
    if (!ok) invalidate_all();
    synced_version_ = g_.version();  // the hook processes (or has invalidated) everything up to here
    return ok;
}

// Static work distribution would be unbalanced (flows differ in cost): workers grab indices from an atomic
// counter; the callback writes only into the slot of its index. Exceptions are captured and rethrown.
void EssentialOracle::run_parallel(size_t count, const std::function<void(size_t, int)>& fn) {
    if (count == 0) return;
    int nt = (int)std::min<size_t>((size_t)threads_, (count + 3) / 4);  // >= 4 items per thread
    if (nt < 1) nt = 1;
    scratch(nt - 1);  // allocate all workspaces up front (never inside the parallel region)
    if (nt == 1) {
        for (size_t i = 0; i < count; ++i) fn(i, 0);
        return;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex err_mutex;
    std::string err;
    auto worker = [&](int w) {
        try {
            for (;;) {
                if (failed.load(std::memory_order_relaxed)) break;
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) break;
                fn(i, w);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mutex);
            if (!failed.exchange(true)) err = e.what();
        } catch (...) {
            std::lock_guard<std::mutex> lock(err_mutex);
            if (!failed.exchange(true)) err = "unknown exception in worker thread";
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    for (int w = 1; w < nt; ++w) pool.emplace_back(worker, w);
    worker(0);
    for (auto& th : pool) th.join();
    if (failed) throw std::runtime_error("EssentialOracle parallel region failed: " + err);
}

// Record the arcs and vertices of v's current paths under a fresh stamp.
void EssentialOracle::index_users(int v) {
    if (arc_users_.size() < (size_t)g_.num_arc_ids()) arc_users_.resize(g_.num_arc_ids());
    const uint64_t st = ++flow_stamp_[v];
    for (const auto& path : flows_[v].paths) {
        for (int a : path) {
            arc_users_[a].emplace_back(v, st);
            vertex_users_[g_.arc(a).head].emplace_back(v, st);
        }
    }
}

// Compact a (v, stamp) list in place and return the live vertices (sorted).
std::vector<int> EssentialOracle::users_of_arc_nosync(int a) {
    std::vector<int> r;
    if (a < 0 || a >= (int)arc_users_.size()) return r;
    auto& lst = arc_users_[a];
    size_t w = 0;
    for (size_t i = 0; i < lst.size(); ++i) {
        const int v = lst[i].first;
        if (valid_[v] && flow_stamp_[v] == lst[i].second) {
            lst[w++] = lst[i];
            r.push_back(v);
        }
    }
    lst.resize(w);
    std::sort(r.begin(), r.end());
    return r;
}

std::vector<int> EssentialOracle::users_of_vertex_nosync(int x) {
    std::vector<int> r;
    if (x < 0 || x >= (int)vertex_users_.size()) return r;
    auto& lst = vertex_users_[x];
    size_t w = 0;
    for (size_t i = 0; i < lst.size(); ++i) {
        const int v = lst[i].first;
        if (valid_[v] && flow_stamp_[v] == lst[i].second) {
            lst[w++] = lst[i];
            r.push_back(v);
        }
    }
    lst.resize(w);
    std::sort(r.begin(), r.end());
    return r;
}

// From-scratch flows (with exact cuts) for every listed vertex that is not current; parallel, then indexed.
void EssentialOracle::recompute_stale(const std::vector<int>& verts) {
    std::vector<int> todo;
    for (int v : verts)
        if (!valid_[v]) todo.push_back(v);
    if (todo.empty()) return;
    run_parallel(todo.size(), [&](size_t i, int w) {
        const int v = todo[i];
        engine_.compute_max_flow(v, flows_[v], scratch(w), true);
    });
    stats_.max_flow_calls += (int64_t)todo.size();
    for (int v : todo) {
        valid_[v] = 1;
        mark_cut_exact(v);
        index_users(v);
    }
}

// Recompute exactly the flows invalidated since the last drain: O(#stale) unless a global invalidation
// happened (then one O(n) sweep).
void EssentialOracle::recompute_pending_stale() {
    std::vector<int> todo;
    if (all_stale_) {
        todo = g_.live_nonterminals();
        all_stale_ = false;
        stale_.clear();
    } else {
        std::sort(stale_.begin(), stale_.end());
        stale_.erase(std::unique(stale_.begin(), stale_.end()), stale_.end());
        for (int v : stale_)
            if (g_.live(v) && !g_.is_terminal(v) && !valid_[v]) todo.push_back(v);
        stale_.clear();
    }
    recompute_stale(todo);
}

// ------------------------------------------------------------------------------------------ public API

// [Prop 4.2] for every live non-terminal, from scratch (O8 parallel). Resynchronizes unconditionally.
void EssentialOracle::compute_all() {
    Stats::Timer timer(stats_, "essential.compute_all");
    invalidate_all();
    synced_version_ = g_.version();
    for (auto& lst : arc_users_) lst.clear();
    for (auto& lst : vertex_users_) lst.clear();
    for (int v = 0; v < g_.n(); ++v)
        if (!g_.live(v) || g_.is_terminal(v)) flows_[v] = VertexFlow{};
    recompute_pending_stale();
}

const VertexFlow& EssentialOracle::flow(int v) {
    if (v < 0 || v >= g_.n() || !g_.live(v) || g_.is_terminal(v))
        throw std::invalid_argument("EssentialOracle::flow: " + vstr(v) + " is not a live non-terminal");
    ensure_synced();
    if (!valid_[v]) {
        Stats::Timer timer(stats_, "essential.lazy_flow");
        engine_.compute_max_flow(v, flows_[v], scratch(0), true);
        ++stats_.max_flow_calls;
        valid_[v] = 1;
        mark_cut_exact(v);
        index_users(v);
    }
    flows_[v].version = g_.version();
    flows_[v].cut_exact = cut_is_exact(v);  // demote cuts computed before a cut-moving mutation
    return flows_[v];
}

const TermSet& EssentialOracle::ess(int v) { return flow(v).ess; }

int EssentialOracle::kappa(int v) { return flow(v).kappa; }

// One reverse reachability on the stored maximum flow (no forbidden arcs) -> exact tightest cut [Prop 4.2].
void EssentialOracle::refresh_cut(int v) {
    flow(v);  // ensures validity (a lazy recomputation already yields an exact cut)
    if (cut_is_exact(v)) return;
    FlowEngine::Scratch& s = scratch(0);
    engine_.load(flows_[v], s);
    engine_.compute_cut(flows_[v], s, kNoForbidden);
    mark_cut_exact(v);
    ++stats_.cut_calls;
}

// Exact cuts for all live non-terminals (parallel): stale flows are recomputed, the rest get one reverse BFS.
void EssentialOracle::refresh_all_cuts() {
    Stats::Timer timer(stats_, "essential.refresh_all_cuts");
    ensure_synced();
    recompute_pending_stale();
    const std::vector<int> verts = g_.live_nonterminals();
    std::vector<int> todo;
    for (int v : verts)
        if (!cut_is_exact(v)) todo.push_back(v);
    run_parallel(todo.size(), [&](size_t i, int w) {
        const int v = todo[i];
        FlowEngine::Scratch& s = scratch(w);
        engine_.load(flows_[v], s);
        engine_.compute_cut(flows_[v], s, kNoForbidden);
    });
    for (int v : todo) mark_cut_exact(v);
    stats_.cut_calls += (int64_t)todo.size();
}

// O1: the vertices whose stored (current) flow uses arc a.
std::vector<int> EssentialOracle::users_of_arc(int a) {
    ensure_synced();
    return users_of_arc_nosync(a);
}

// O2/O5 warm-started evaluation of deleting the arc set D, for every vertex whose flow uses an arc of D:
// drop the path(s) through D (one per arc of D used, several only if v is the tail of several arcs of D),
// re-augment with D forbidden until failure or until the old value is restored, then either the exact
// tightest cut of G \ D (kappa dropped, or exact_cuts) or the old certified subset (kappa restored: O1).
std::vector<int> EssentialOracle::evaluate_deletion(const std::vector<int>& D, std::vector<VertexFlow>& out,
                                                    bool exact_cuts) {
    Stats::Timer timer(stats_, "essential.evaluate_deletion");
    ensure_synced();
    for (int a : D) {
        if (a < 0 || a >= g_.num_arc_ids() || !g_.arc(a).alive)
            throw std::invalid_argument("EssentialOracle::evaluate_deletion: arc " + vstr(a) + " is not alive");
    }
    // Make every flow current first so that the user index is exact (stale flows would otherwise be
    // recomputed later, possibly through D, without ever being repaired). O(#stale).
    recompute_pending_stale();
    std::vector<int> affected;
    for (int a : D) {
        std::vector<int> u = users_of_arc_nosync(a);
        affected.insert(affected.end(), u.begin(), u.end());
    }
    std::sort(affected.begin(), affected.end());
    affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
    if (out.size() < (size_t)g_.n()) out.resize(g_.n());
    std::vector<int64_t> aug_calls(threads_ > 0 ? threads_ : 1, 0), cut_calls(aug_calls.size(), 0);
    run_parallel(affected.size(), [&](size_t i, int w) {
        const int v = affected[i];
        FlowEngine::Scratch& s = scratch(w);
        VertexFlow f = flows_[v];
        const int old_kappa = f.kappa;
        engine_.load(f, s);
        int removed = 0;
        for (int a : D)
            if (engine_.remove_path_using_arc(f, a, s)) ++removed;
        int restored = 0;
        while (restored < removed) {
            ++aug_calls[w];
            if (!engine_.augment_once(f, s, D)) break;
            ++restored;
        }
        if (f.kappa < old_kappa || exact_cuts) {
            engine_.compute_cut(f, s, D);
            ++cut_calls[w];
        } else {
            // kappa restored: Ess_G(v) ⊆ Ess_{G\D}(v) [Lem 4.3 generalized, O1]; keep the certified subset.
            f.cut_exact = false;
            f.side.clear();
        }
        f.version = g_.version();
        out[v] = std::move(f);
    });
    for (size_t w = 0; w < aug_calls.size(); ++w) {
        stats_.augment_calls += aug_calls[w];
        stats_.cut_calls += cut_calls[w];
    }
    return affected;
}

// The caller has deleted exactly the arcs of D: adopt the flows computed by evaluate_deletion. Flows of
// unaffected vertices avoid D and remain maximum with certified subsets (O1) — nothing to do for them.
void EssentialOracle::commit_deletion(const std::vector<int>& D, const std::vector<int>& affected,
                                      std::vector<VertexFlow>& computed) {
    Stats::Timer timer(stats_, "essential.commit_deletion");
    std::vector<int> uniq(D);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    for (int a : uniq) {
        if (a < 0 || a >= g_.num_arc_ids() || g_.arc(a).alive)
            throw std::invalid_argument("EssentialOracle::commit_deletion: arc " + vstr(a) + " has not been deleted");
    }
    const bool in_sync = expect_mutations(uniq.size());
    stats_.deletions += (int64_t)uniq.size();
    if (uniq.size() > 1) stats_.batched_deletions += 1;
    if (!in_sync) return;  // everything invalidated; the computed flows are not trusted
    ++exact_epoch_;        // tightest cuts of unaffected vertices may move (only Ess ⊆ is guaranteed, O1)
    for (int v : affected) {
        if (v < 0 || v >= g_.n() || (size_t)v >= computed.size() || computed[v].v != v)
            throw std::invalid_argument("EssentialOracle::commit_deletion: no computed flow for vertex " + vstr(v));
        if (!g_.live(v) || g_.is_terminal(v)) continue;
        flows_[v] = std::move(computed[v]);
        computed[v] = VertexFlow{};
        flows_[v].version = g_.version();
        valid_[v] = 1;
        if (flows_[v].cut_exact) mark_cut_exact(v);  // computed with D forbidden = exact in G \ D
        index_users(v);
    }
    // Safety net: a flow (re)computed between evaluate and commit may use D; recompute it in G \ D now.
    std::vector<int> extra;
    for (int a : uniq) {
        std::vector<int> u = users_of_arc_nosync(a);
        extra.insert(extra.end(), u.begin(), u.end());
    }
    if (!extra.empty()) {
        std::sort(extra.begin(), extra.end());
        extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
        for (int v : extra) invalidate(v);
        recompute_pending_stale();
    }
}

// O3: after Graph::contract(p, t) every flow through p is translated (kappa and Ess unchanged, §13.2); the
// flow of p itself is dropped. A flow whose path leaves p by another arc cannot be translated (contraction
// of a pre-terminal with d^+(p) > 1 on a flow arc): it is invalidated and recomputed lazily.
//
// Exactness. [Lem 7.3] / §13.2 ("kappa and Ess of every remaining vertex are unchanged") is proved for
// d^+(p) = 1, the only case of [Alg 1] step (ii). Graph::contract also accepts d^+(p) >= 2 [Def 2.1]; then
// the contraction is the contraction of G \ D for D = out(p) \ {(p,t)}, i.e. it also performs |D| arc
// deletions, and a deletion can move the tightest cut of a vertex whose flow avoids D (O1 only certifies
// kappa and Ess_G(v) ⊆ Ess_{G\D}(v)). Exactly as commit_deletion does, the exact epoch therefore advances
// in that case, demoting every stored cut to "certified subset" (refresh_cut / refresh_all_cuts restore
// exactness with one reverse BFS each). The out-degree at contraction time is read from the Graph's
// contraction record, since the hook runs after the mutation.
void EssentialOracle::after_contraction(int p, int t) {
    Stats::Timer timer(stats_, "essential.after_contraction");
    stats_.contractions += 1;
    if (!expect_mutations(1)) return;
    // A caller error below leaves one unprocessed mutation behind: invalidate everything before reporting it,
    // so the oracle stays usable (flows are recomputed lazily) even if the exception is caught.
    if (p < 0 || p >= g_.n() || g_.live(p)) {
        invalidate_all();
        throw std::invalid_argument("EssentialOracle::after_contraction: " + vstr(p) + " is still live");
    }
    const Graph::ContractionRecord& rec = g_.last_contraction();
    if (rec.version != g_.version() || rec.p != p || rec.t != t) {
        invalidate_all();
        throw std::invalid_argument("EssentialOracle::after_contraction(" + vstr(p) + "," + vstr(t) +
                                    "): the last graph mutation was not this contraction");
    }
    if (rec.out_degree != 1) ++exact_epoch_;  // |D| = d^+(p) - 1 >= 1 arc deletions: cuts may move (O1, not O3)
    valid_[p] = 0;
    flows_[p] = VertexFlow{};
    for (int u : users_of_vertex_nosync(p)) {
        if (engine_.can_translate_after_contraction(flows_[u], p, t)) {
            engine_.translate_after_contraction(flows_[u], p, t);
            flows_[u].version = g_.version();
            index_users(u);
        } else {
            invalidate(u);
        }
    }
    vertex_users_[p].clear();
}

// O4: after Graph::remove_terminal(t) drop the path ending at t (if any). The remainder has value kappa-1
// and is maximum in G \ t iff t was essential for v; a flow may end at a NON-essential t as well (another
// maximum family avoids t), so one warm-started augmentation is attempted before the reverse BFS that
// recomputes the tightest cut (new essential terminals can appear, paper_notes §13.3). Still O(n+m) per
// vertex; parallel over vertices. (docs/optimizations.md O4 omits the augmentation step.)
void EssentialOracle::after_terminal_removal(int t) {
    Stats::Timer timer(stats_, "essential.after_terminal_removal");
    stats_.terminal_removals += 1;
    if (!expect_mutations(1)) return;
    if (t < 0 || t >= g_.n() || g_.live(t) || g_.terminal_index(t) < 0) {
        invalidate_all();  // one unprocessed mutation: stay safe even if the caller catches the error
        throw std::invalid_argument("EssentialOracle::after_terminal_removal: " + vstr(t) + " is not a removed terminal");
    }
    std::vector<int> verts;
    for (int v : g_.live_nonterminals())
        if (valid_[v]) verts.push_back(v);
    std::vector<char> changed(g_.n(), 0);
    std::vector<int64_t> aug_calls(threads_ > 0 ? threads_ : 1, 0);
    run_parallel(verts.size(), [&](size_t i, int w) {
        const int v = verts[i];
        FlowEngine::Scratch& s = scratch(w);
        VertexFlow& f = flows_[v];
        engine_.load(f, s);
        if (engine_.remove_path_to_terminal(f, t, s)) {
            changed[v] = 1;
            ++aug_calls[w];
            engine_.augment_once(f, s, kNoForbidden);  // succeeds iff t was not essential for v
        }
        engine_.compute_cut(f, s, kNoForbidden);
        f.version = g_.version();
    });
    for (int64_t c : aug_calls) stats_.augment_calls += c;
    stats_.cut_calls += (int64_t)verts.size();
    for (int v : verts) {
        mark_cut_exact(v);
        if (changed[v]) index_users(v);
    }
    vertex_users_[t].clear();
}

// Generic vertex removal (rounding): drop v's flow, invalidate the flows through v (recomputed lazily) and
// demote every other cut to "certified subset" (kappa and the flow survive; Ess_G ⊆ Ess_{G\v} by the
// generalized [Lem 4.3], but the tightest cut may move).
void EssentialOracle::after_vertex_removal(int v) {
    Stats::Timer timer(stats_, "essential.after_vertex_removal");
    if (!expect_mutations(1)) return;
    if (v < 0 || v >= g_.n() || g_.live(v)) {
        invalidate_all();  // one unprocessed mutation: stay safe even if the caller catches the error
        throw std::invalid_argument("EssentialOracle::after_vertex_removal: " + vstr(v) + " is still live");
    }
    valid_[v] = 0;
    flows_[v] = VertexFlow{};
    for (int u : users_of_vertex_nosync(v)) invalidate(u);
    vertex_users_[v].clear();
    ++exact_epoch_;  // every other cut is demoted to "certified subset"
}

}  // namespace glcore
