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
// User index (exact). arc_users_[a] lists the flows whose paths use arc a and vertex_users_[x] the flows
// whose paths enter x, as back-references (v, i, j) to the registry entry reg_[v][i][j] of paths[i][j]; the
// entry stores the arc id and the positions of its two references. A flow is registered exactly while it
// is valid: every transition that replaces or drops a flow's paths first unregisters it — each of its
// entries is a swap-remove from the two lists in O(1), patching the position stored by the entry that was
// moved into the hole — so the index holds exactly the arcs of the valid flows and its memory is O(total
// live path length) at all times (the previous append-only index with lazy compaction grew by the whole
// path length of a flow at every change: RESEARCH_NOTES E2). A contraction (O3) rewrites the last two arcs
// of the path through p in place, so its index update is O(1) per affected flow as well; the flows through
// p are read off vertex_users_[p] without touching any other flow.
//
// Cut sides are never stored: compute_cut yields kappa / Ess from the reverse BFS alone and the L/S/R
// labels of all n vertices are materialized only by sides() (trace / record_cuts / diagnostics).
//
// Parallelism (O8): a persistent WorkerPool (created lazily, joined by the destructor) with an atomic work
// counter; every worker owns one FlowEngine Scratch and writes only per-vertex slots, so results are
// deterministic regardless of the thread count; sequential post-processing runs in increasing vertex
// order. Stats counters are accumulated per worker.
#include "essential.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "stats.hpp"

namespace glcore {

namespace {
const std::vector<int> kNoForbidden;

std::string vstr(int v) { return std::to_string(v); }
}  // namespace

EssentialOracle::EssentialOracle(Graph& g, Stats& stats, int threads, int seed)
    : g_(g), stats_(stats), threads_(threads < 1 ? 1 : threads), seed_(seed), engine_(g),
      flows_(g.n()), reg_(g.n()), arc_users_(g.num_arc_ids()), vertex_users_(g.n()), valid_(g.n(), 0),
      synced_version_(g.version()), cut_epoch_(g.n(), 0) {}

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

// Static work distribution would be unbalanced (flows differ in cost): workers grab indices from the pool's
// atomic counter; the callback writes only into the slot of its index. Exceptions are rethrown here.
void EssentialOracle::run_parallel(size_t count, const std::function<void(size_t, int)>& fn) {
    if (count == 0) return;
    int nt = (int)std::min<size_t>((size_t)threads_, (count + 3) / 4);  // >= 4 items per thread
    if (nt < 1) nt = 1;
    scratch(nt - 1);  // allocate all workspaces up front (never inside the parallel region)
    pool_.run(count, nt, fn);
}

// ------------------------------------------------------------------------------------------ user index

void EssentialOracle::ensure_arc_index_size() {
    if (arc_users_.size() < (size_t)g_.num_arc_ids()) arc_users_.resize(g_.num_arc_ids());
}

// Swap-remove one reference; the reference moved into the hole gets its stored position patched.
void EssentialOracle::detach_arc_ref(const Entry& e) {
    std::vector<Ref>& lst = arc_users_[e.arc];
    const int slot = e.arc_slot;
    const int last_slot = (int)lst.size() - 1;
    if (slot != last_slot) {
        const Ref moved = lst[last_slot];
        lst[slot] = moved;
        reg_[moved.v][moved.path][moved.pos].arc_slot = slot;
    }
    lst.pop_back();
}

void EssentialOracle::detach_vertex_ref(const Entry& e) {
    std::vector<Ref>& lst = vertex_users_[g_.arc(e.arc).head];
    const int slot = e.vtx_slot;
    const int last_slot = (int)lst.size() - 1;
    if (slot != last_slot) {
        const Ref moved = lst[last_slot];
        lst[slot] = moved;
        reg_[moved.v][moved.path][moved.pos].vtx_slot = slot;
    }
    lst.pop_back();
}

// Append one reference per path arc of flows_[v]; reg_[v] must hold no entries (unregister_flow first).
void EssentialOracle::register_flow(int v) {
    ensure_arc_index_size();
    const VertexFlow& f = flows_[v];
    std::vector<std::vector<Entry>>& rv = reg_[v];
    rv.resize(f.paths.size());  // the inner vectors keep their capacity across re-registrations
    for (size_t i = 0; i < f.paths.size(); ++i) {
        const std::vector<int>& path = f.paths[i];
        std::vector<Entry>& ri = rv[i];
        if (!ri.empty()) throw std::logic_error("EssentialOracle: flow " + vstr(v) + " is already registered");
        ri.reserve(path.size());
        for (size_t j = 0; j < path.size(); ++j) {
            const int a = path[j];
            const int h = g_.arc(a).head;
            ri.push_back(Entry{a, (int)arc_users_[a].size(), (int)vertex_users_[h].size()});
            arc_users_[a].push_back(Ref{v, (int)i, (int)j});
            vertex_users_[h].push_back(Ref{v, (int)i, (int)j});
        }
    }
}

// Remove every reference of v (O(1) each); a flow that is not registered is a no-op.
void EssentialOracle::unregister_flow(int v) {
    for (std::vector<Entry>& ri : reg_[v]) {
        for (const Entry& e : ri) {
            detach_arc_ref(e);
            detach_vertex_ref(e);
        }
        ri.clear();
    }
}

void EssentialOracle::invalidate(int v) {
    unregister_flow(v);
    valid_[v] = 0;
    stale_.push_back(v);
}

void EssentialOracle::invalidate_all() {
    for (int v = 0; v < g_.n(); ++v) unregister_flow(v);
    std::fill(valid_.begin(), valid_.end(), 0);
    stale_.clear();
    all_stale_ = true;
}

// The live users of arc a (sorted; every registered flow is valid).
std::vector<int> EssentialOracle::users_of_arc_nosync(int a) {
    std::vector<int> r;
    if (a < 0 || a >= (int)arc_users_.size()) return r;
    const std::vector<Ref>& lst = arc_users_[a];
    r.reserve(lst.size());
    for (const Ref& x : lst) r.push_back(x.v);
    std::sort(r.begin(), r.end());
    return r;
}

// The flows whose paths pass through / end at x (sorted; at most one entry per flow: paths are vertex-disjoint).
std::vector<int> EssentialOracle::users_of_vertex_nosync(int x) {
    std::vector<int> r;
    if (x < 0 || x >= (int)vertex_users_.size()) return r;
    const std::vector<Ref>& lst = vertex_users_[x];
    r.reserve(lst.size());
    for (const Ref& e : lst) r.push_back(e.v);
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
        register_flow(v);
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
    invalidate_all();  // unregisters every flow: both user lists are empty afterwards
    synced_version_ = g_.version();
    ensure_arc_index_size();
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
        register_flow(v);
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

// The tightest cut of v with the side of every vertex: one reverse BFS (the sides are the same for every
// maximum flow of v, [Def 3.8] uniqueness), never stored.
void EssentialOracle::sides(int v, std::vector<Side>& out) {
    flow(v);
    FlowEngine::Scratch& s = scratch(0);
    engine_.load(flows_[v], s);
    engine_.compute_cut(flows_[v], s, kNoForbidden, true);
    mark_cut_exact(v);
    ++stats_.cut_calls;
    out.swap(flows_[v].side);
    std::vector<Side>().swap(flows_[v].side);
}

// O1: the vertices whose stored (current) flow uses arc a.
std::vector<int> EssentialOracle::users_of_arc(int a) {
    ensure_synced();
    return users_of_arc_nosync(a);
}

size_t EssentialOracle::num_users_of_arc(int a) {
    ensure_synced();
    if (a < 0 || a >= (int)arc_users_.size()) return 0;
    return arc_users_[a].size();
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
        if (a >= (int)arc_users_.size()) continue;
        for (const Ref& r : arc_users_[a]) affected.push_back(r.v);
    }
    std::sort(affected.begin(), affected.end());
    affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
    if (out.size() < (size_t)g_.n()) out.resize(g_.n());
    std::vector<int64_t> aug_calls(threads_ > 0 ? threads_ : 1, 0), cut_calls(aug_calls.size(), 0);
    run_parallel(affected.size(), [&](size_t i, int w) {
        const int v = affected[i];
        FlowEngine::Scratch& s = scratch(w);
        VertexFlow& f = out[v];
        f = flows_[v];  // copy-assignment reuses the slot's buffers
        const int old_kappa = f.kappa;
        engine_.load(f, s);
        const int removed = engine_.remove_paths_using_arcs(f, D, s);
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
        unregister_flow(v);
        flows_[v] = std::move(computed[v]);
        computed[v] = VertexFlow{};
        flows_[v].version = g_.version();
        valid_[v] = 1;
        if (flows_[v].cut_exact) mark_cut_exact(v);  // computed with D forbidden = exact in G \ D
        register_flow(v);
    }
    // Safety net: a flow (re)computed between evaluate and commit may use D; recompute it in G \ D now.
    std::vector<int> extra;
    for (int a : uniq) {
        if (a >= (int)arc_users_.size() || arc_users_[a].empty()) continue;
        for (const Ref& r : arc_users_[a]) extra.push_back(r.v);
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
//
// Cost: O(#flows through p). vertex_users_[p] names, for every such flow, the registry entry of the arc
// entering p, i.e. the position (i, j) of that arc on the flow's paths; the translation replaces
// paths[i][j] and pops paths[i][j+1] (the arc (p,t), last on the path), and the index is patched with two
// O(1) removals and two appends. No path is scanned.
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
    ensure_arc_index_size();                   // redirected arcs have new ids
    unregister_flow(p);
    valid_[p] = 0;
    flows_[p] = VertexFlow{};
    // Every flow through p has exactly one entry here (its arc into p); processing a flow only touches its
    // own registry positions, so the copied references stay valid throughout the loop.
    const std::vector<Ref> through(vertex_users_[p]);
    const uint64_t version = g_.version();
    for (const Ref& r : through) {
        const int u = r.v;
        VertexFlow& f = flows_[u];
        if (engine_.translate_path_after_contraction(f, (size_t)r.path, (size_t)r.pos, p, t)) {
            std::vector<Entry>& ri = reg_[u][r.path];
            if (ri.size() != (size_t)r.pos + 2)
                throw std::logic_error("EssentialOracle::after_contraction: user index out of sync for flow " + vstr(u));
            const Entry last = ri.back();      // the arc (p, t): gone
            detach_arc_ref(last);
            detach_vertex_ref(last);
            ri.pop_back();
            Entry& e = ri[r.pos];              // the arc into p: now the redirected arc into t
            detach_arc_ref(e);
            detach_vertex_ref(e);
            const int a_new = f.paths[r.path][r.pos];
            e.arc = a_new;
            e.arc_slot = (int)arc_users_[a_new].size();
            arc_users_[a_new].push_back(Ref{u, r.path, r.pos});
            e.vtx_slot = (int)vertex_users_[t].size();
            vertex_users_[t].push_back(Ref{u, r.path, r.pos});
            f.version = version;
        } else {
            invalidate(u);
        }
    }
    if (!vertex_users_[p].empty())
        throw std::logic_error("EssentialOracle::after_contraction: flows still registered through " + vstr(p));
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
        if (changed[v]) {
            unregister_flow(v);  // the registry entries name the OLD arcs, so this is exact
            register_flow(v);
        }
    }
    if (!vertex_users_[t].empty())
        throw std::logic_error("EssentialOracle::after_terminal_removal: flows still registered through " + vstr(t));
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
    unregister_flow(v);
    valid_[v] = 0;
    flows_[v] = VertexFlow{};
    for (int u : users_of_vertex_nosync(v)) invalidate(u);
    if (!vertex_users_[v].empty())
        throw std::logic_error("EssentialOracle::after_vertex_removal: flows still registered through " + vstr(v));
    ++exact_epoch_;  // every other cut is demoted to "certified subset"
}

}  // namespace glcore
