"""Adversarial review tests for the independent verifier ``glsolver.verify``
(and the I/O round trips of ``glsolver.io``).

Every test here either

* feeds the verifier a partition that is *invalid* and asserts that it is
  rejected (a false accept would silently corrupt the whole project's ground
  truth), or
* feeds it a partition that is *valid* by an independent computation and
  asserts that it is accepted, or
* checks a structural property of the module (purity of its imports).

Independent oracles are written with plain NetworkX calls (``nx.ancestors``,
``nx.is_connected``) and never with the verifier's own BFS helpers.
"""
from __future__ import annotations

import ast
import json
import pathlib
import random
from types import SimpleNamespace

import networkx as nx
import pytest

from glsolver.instance import Instance, make_instance
from glsolver.io import (
    instance_from_dict,
    instance_to_dict,
    load_edgelist,
    load_instance,
    load_solution,
    result_to_dict,
    save_instance,
    save_solution,
)
from glsolver.verify import verify_instance_parts, verify_partition

VERIFY_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "src" / "glsolver" / "verify.py"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def path_und(n: int, terminals: list[int], sizes: list[int]) -> Instance:
    """Undirected path ``0-1-...-(n-1)``."""
    return make_instance(n, [(i, i + 1) for i in range(n - 1)], terminals, sizes=sizes, directed=False)


def independent_verdict(inst: Instance, parts: list[list[int]]) -> bool:
    """Independent re-implementation of validity with NetworkX only
    ([Def connected-to], [Thm weighted-k-t-conn], paper_notes §1.1/§13.11)."""
    n, k = inst.n, inst.k
    if len(parts) != k:
        return False
    flat: list[int] = []
    for part in parts:
        for v in part:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n:
                return False
            flat.append(v)
    if sorted(flat) != list(range(n)):
        return False
    tset = set(inst.terminals)
    if inst.directed:
        digraph = nx.DiGraph()
        digraph.add_nodes_from(range(n))
        digraph.add_edges_from(inst.arcs)
    else:
        assert inst.undirected_edges is not None
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(inst.undirected_edges)
    for i, part in enumerate(parts):
        pset = set(part)
        t = inst.terminals[i]
        if t not in pset or pset & tset != {t}:
            return False
        if inst.weights is None:
            if len(pset) != inst.capacities[i] + 1:
                return False
        else:
            w_max = max((inst.weights[v] for v in range(n) if v not in tset), default=1)
            weight = sum(inst.weights[v] for v in pset if v not in tset)
            if weight > inst.capacities[i] + w_max - 1:
                return False
        if inst.directed:
            if nx.ancestors(digraph.subgraph(pset), t) | {t} != pset:
                return False
        elif not nx.is_connected(graph.subgraph(pset)):
            return False
    return True


# ---------------------------------------------------------------------------
# purity: the verifier must depend on nothing but the stdlib and Instance
# ---------------------------------------------------------------------------


def test_verify_module_imports_only_stdlib_and_instance():
    """docs/api.md: the verifier 'imports nothing from the solver modules'
    and uses only BFS on the original input.  Checked on the AST so that the
    package ``__init__`` (which imports the solver front-end) cannot mask it."""
    tree = ast.parse(VERIFY_SOURCE.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    allowed = {"__future__", "collections", "dataclasses", "numbers", "typing", "glsolver.instance"}
    assert modules <= allowed, modules - allowed
    # no dynamic imports either
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            raise AssertionError("verify.py must not use __import__")
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "import_module":
            raise AssertionError("verify.py must not use importlib.import_module")


# ---------------------------------------------------------------------------
# invalid partitions that MUST be rejected
# ---------------------------------------------------------------------------


def test_rejects_part_connected_only_through_other_part_undirected():
    # 0-1-4-2-3-0 : part 0 = {0,1,2} is linked only through vertex 4 (terminal of part 1)
    inst = make_instance(5, [(0, 1), (1, 4), (4, 2), (2, 3), (0, 3)], [0, 4], sizes=[3, 2], directed=False)
    assert independent_verdict(inst, [[0, 1, 2], [4, 3]]) is False
    report = verify_instance_parts(inst, [[0, 1, 2], [4, 3]])
    assert report.valid is False
    assert any("part 0 is not connected" in e and "[2]" in e for e in report.errors)
    # the genuinely valid partition of the same instance is accepted
    assert verify_instance_parts(inst, [[0, 1, 3], [4, 2]]).valid is True
    # a non-terminal (5) of the other part as the only bridge between 1 and 2
    inst = make_instance(
        6, [(0, 1), (1, 5), (5, 2), (2, 3), (3, 4), (5, 4)], [0, 4], sizes=[3, 3], directed=False
    )
    report = verify_instance_parts(inst, [[0, 1, 2], [4, 3, 5]])
    assert report.valid is False
    assert report.details["connectivity_ok"] == [False, True]
    assert report.details["unreachable"] == {0: [2]}


def test_rejects_part_connected_only_through_other_part_directed():
    # 2 -> 3 -> 0, 3 -> 1, 2 -> 1: putting 2 with t_0 but 3 with t_1 breaks part 0 [Def connected-to]
    inst = make_instance(4, [(2, 3), (3, 0), (3, 1), (2, 1)], [0, 1], [1, 1])
    report = verify_instance_parts(inst, [[0, 2], [1, 3]])
    assert report.valid is False
    assert report.details["connectivity_ok"] == [False, True]
    assert report.details["unreachable"] == {0: [2]}
    assert verify_instance_parts(inst, [[0, 3], [1, 2]]).valid is True


def test_rejects_directed_reachability_via_arcs_not_in_instance():
    # Input edge (0,1) leaves terminal 0 and is dropped at normalization; the
    # verifier must judge on inst.arcs, where 1 cannot reach 0.
    inst = make_instance(3, [(0, 1), (2, 0)], [0], [2], directed=True)
    assert inst.arcs == ((2, 0),)
    report = verify_instance_parts(inst, [[0, 1, 2]])
    assert report.valid is False
    assert report.details["unreachable"] == {0: [1]}
    # Reversed orientation: 0 <- 1 <- 2 is fine, 0 -> 1 -> 2 is not (arcs out of 0 dropped, 1->2 useless)
    good = make_instance(3, [(1, 0), (2, 1)], [0], [2])
    assert verify_instance_parts(good, [[0, 1, 2]]).valid is True
    bad = make_instance(3, [(1, 2), (2, 1)], [0], [2])  # 2-cycle, nothing reaches 0
    report = verify_instance_parts(bad, [[0, 1, 2]])
    assert report.valid is False
    assert report.details["unreachable"] == {0: [1, 2]}
    # A vertex with a path to the terminal that leaves the part and comes back is NOT connected
    inst = make_instance(4, [(3, 2), (2, 1), (1, 0), (3, 0)], [0], [3], directed=True)
    assert verify_instance_parts(inst, [[0, 1, 2, 3]]).valid is True
    inst2 = make_instance(5, [(3, 2), (2, 1), (1, 0), (2, 4)], [0, 4], [2, 1], directed=True)
    report = verify_instance_parts(inst2, [[0, 1, 3], [4, 2]])
    assert report.valid is False  # 3 -> 2 -> 1 -> 0 uses vertex 2 of part 1
    assert report.details["unreachable"] == {0: [3]}


def test_rejects_foreign_terminal_even_when_everything_else_is_fine():
    # cycle 0-1-2-3-0, terminals 0 and 2; sizes 3 and 1
    inst = make_instance(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [0, 2], sizes=[3, 1], directed=False)
    assert verify_instance_parts(inst, [[0, 1, 3], [2]]).valid is True
    report = verify_instance_parts(inst, [[0, 1, 2], [3]])
    assert report.valid is False
    assert any("part 0 contains terminal t_1 (vertex 2)" in e for e in report.errors)
    assert any("terminal t_1 (vertex 2) is not in part 1" in e for e in report.errors)
    # foreign terminal duplicated into a part that keeps the right size otherwise
    report = verify_instance_parts(inst, [[0, 1, 2], [2]])
    assert report.valid is False
    assert any("vertex 2 appears 2 times" in e for e in report.errors)
    assert any("vertex 3 is missing" in e for e in report.errors)
    # directed: a foreign terminal can never reach the part's terminal (it has no out-arcs)
    dinst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 0])
    report = verify_instance_parts(dinst, [[0, 1], [2]])
    assert report.valid is False
    assert any("part 0 contains terminal t_1" in e for e in report.errors)


def test_rejects_duplicate_vertex_even_when_sizes_match():
    inst = path_und(5, [0, 4], [2, 3])
    # duplicate inside one part, sizes correct, one vertex missing
    report = verify_instance_parts(inst, [[0, 1], [4, 3, 3]])
    assert report.valid is False
    assert any("vertex 3 appears 2 times" in e for e in report.errors)
    assert any("vertex 2 is missing" in e for e in report.errors)
    # duplicate across parts, sizes correct
    report = verify_instance_parts(inst, [[0, 1], [4, 3, 1]])
    assert report.valid is False
    assert any("vertex 1 appears 2 times in the partition (parts [0, 1])" in e for e in report.errors)
    # the terminal itself duplicated
    report = verify_instance_parts(inst, [[0, 0], [4, 3, 2]])
    assert report.valid is False
    assert any("vertex 0 appears 2 times" in e for e in report.errors)
    # duplicates as numpy ints are still duplicates
    np = pytest.importorskip("numpy")
    report = verify_instance_parts(inst, [[0, 1], [np.int64(4), np.int32(3), 3]])
    assert report.valid is False


def test_rejects_capacities_off_by_one_in_both_directions():
    # sizes given by the instance are 2 and 3 on a 5-path
    inst = path_und(5, [0, 4], [2, 3])
    assert verify_instance_parts(inst, [[0, 1], [4, 3, 2]]).valid is True
    report = verify_instance_parts(inst, [[0, 1, 2], [4, 3]])  # part 0 one too big, part 1 one too small
    assert report.valid is False
    assert sum("expected c_0 + 1 = 2" in e for e in report.errors) == 1
    assert sum("expected c_1 + 1 = 3" in e for e in report.errors) == 1
    # total count right but one part too small and the missing vertex omitted entirely
    report = verify_instance_parts(inst, [[0, 1], [4, 3]])
    assert report.valid is False
    # capacity-0 terminal must get a singleton part
    inst0 = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 0])
    assert verify_instance_parts(inst0, [[0, 2], [1]]).valid is True
    assert verify_instance_parts(inst0, [[0], [1, 2]]).valid is False
    # directed, c = (2, 0): giving the non-terminals to the wrong terminal is a size error
    dinst = make_instance(4, [(2, 0), (3, 0), (2, 1), (3, 1)], [0, 1], [2, 0])
    assert verify_instance_parts(dinst, [[0, 2, 3], [1]]).valid is True
    report = verify_instance_parts(dinst, [[0, 2], [1, 3]])
    assert report.valid is False
    assert any("part 0 has size 2, expected c_0 + 1 = 3" in e for e in report.errors)
    assert any("part 1 has size 2, expected c_1 + 1 = 1" in e for e in report.errors)


def test_weighted_bound_is_exact_at_the_boundary():
    """[Thm weighted-k-t-conn]: weight(V_t \\ T) <= c_t + w_max - 1, with w_max
    the global maximum non-terminal weight."""
    # terminals 0,1; weights 2 (w=3), 3 (w=1), 4 (w=2); everyone sees both terminals
    arcs = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
    inst = make_instance(5, arcs, [0, 1], [3, 3], weights=[0, 0, 3, 1, 2])
    # part 0 weight 5 == 3 + 3 - 1: accepted
    report = verify_instance_parts(inst, [[0, 2, 4], [1, 3]])
    assert report.valid is True and report.details["exceeds_capacity"] == [0]
    # part 0 weight 6 == bound + 1: rejected
    report = verify_instance_parts(inst, [[0, 2, 3, 4], [1]])
    assert report.valid is False
    assert any("exceeding the bound" in e for e in report.errors)
    # capacities (4, 2): part 1 weight 3 == 2 + 3 - 1 accepted, weight 4 rejected
    inst = make_instance(5, arcs, [0, 1], [4, 2], weights=[0, 0, 3, 1, 2])
    assert verify_instance_parts(inst, [[0, 2], [1, 3, 4]]).valid is True
    assert verify_instance_parts(inst, [[0, 3], [1, 2, 4]]).valid is False  # part 1 weight 5
    # w_max is global: the heavy vertex sits in part 0, yet part 1 may use the slack w_max - 1
    inst = make_instance(5, arcs, [0, 1], [5, 1], weights=[0, 0, 3, 1, 2])
    report = verify_instance_parts(inst, [[0, 2], [1, 3, 4]])  # part 1 weight 3 == 1 + 3 - 1
    assert report.valid is True
    assert report.details["w_max"] == 3
    # unit weights: the bound collapses to c_t exactly
    inst = make_instance(5, arcs, [0, 1], [2, 1], weights=[0, 0, 1, 1, 1])
    assert verify_instance_parts(inst, [[0, 2, 3], [1, 4]]).valid is True
    assert verify_instance_parts(inst, [[0, 2], [1, 3, 4]]).valid is False
    # weighted instances still need the cover / terminal checks
    inst = make_instance(5, arcs, [0, 1], [6, 6], weights=[0, 0, 3, 1, 2])
    assert verify_instance_parts(inst, [[0, 2, 3, 4], [1]]).valid is True
    assert verify_instance_parts(inst, [[0, 2, 3], [1]]).valid is False  # 4 missing
    assert verify_instance_parts(inst, [[0, 2, 3, 4, 1], [1]]).valid is False  # foreign terminal


def test_rejects_permuted_parts_and_terminal_swaps():
    inst = path_und(6, [0, 5], [3, 3])
    assert verify_instance_parts(inst, [[0, 1, 2], [5, 4, 3]]).valid is True
    report = verify_instance_parts(inst, [[5, 4, 3], [0, 1, 2]])  # parts swapped
    assert report.valid is False
    assert report.details["connectivity_ok"] == [False, False]
    # part i must be the part of terminals[i], even when both parts are connected and well sized
    inst = make_instance(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [1, 3], sizes=[2, 2], directed=False)
    assert verify_instance_parts(inst, [[1, 0], [3, 2]]).valid is True
    assert verify_instance_parts(inst, [[3, 2], [1, 0]]).valid is False


def test_undirected_connectivity_is_judged_on_original_edges():
    """paper_notes §1.1/§13.11: arcs out of terminals are dropped in ``arcs``
    but undirected connectivity of ``G[V_i]`` must use the original edges."""
    # terminal 0 in the *middle* of its part: 1 - 0 - 2, then 2 - 3 - 4
    inst = make_instance(5, [(1, 0), (0, 2), (2, 3), (3, 4)], [0, 4], sizes=[3, 2], directed=False)
    assert all(u != 0 for u, _ in inst.arcs)  # no arc leaves terminal 0
    report = verify_instance_parts(inst, [[0, 1, 2], [4, 3]])
    assert report.valid is True, report.errors
    # a star centred at the terminal: every leaf is adjacent only to t
    star = make_instance(5, [(0, 1), (0, 2), (0, 3), (4, 3)], [0, 4], sizes=[4, 1], directed=False)
    assert verify_instance_parts(star, [[0, 1, 2, 3], [4]]).valid is True
    star = make_instance(5, [(0, 1), (0, 2), (0, 3), (4, 3)], [0, 4], sizes=[3, 2], directed=False)
    assert verify_instance_parts(star, [[0, 1, 2], [4, 3]]).valid is True
    assert verify_instance_parts(star, [[0, 1, 3], [4, 2]]).valid is False  # 2 hangs off t_0 only
    # an edge between the two terminals is kept in undirected_edges and must not connect parts
    inst = make_instance(4, [(0, 1), (1, 2), (2, 3), (0, 3)], [0, 3], sizes=[2, 2], directed=False)
    assert (0, 3) in inst.undirected_edges
    assert verify_instance_parts(inst, [[0, 1], [3, 2]]).valid is True
    inst = make_instance(4, [(0, 1), (0, 3), (3, 2)], [0, 3], sizes=[2, 2], directed=False)
    assert verify_instance_parts(inst, [[0, 1], [3, 2]]).valid is True
    assert verify_instance_parts(inst, [[0, 2], [3, 1]]).valid is False  # 2 reaches 0 only via t_1
    # a hand-built inconsistent instance (edges say path, arcs empty) is never silently accepted
    hand = Instance(
        n=3, arcs=(), terminals=(0,), capacities=(2,), directed=False,
        undirected_edges=((0, 1), (1, 2)),
    )
    report = verify_instance_parts(hand, [[0, 1, 2]])
    assert report.valid is False
    assert any("internal inconsistency" in e for e in report.errors)


def test_undirected_partition_is_never_judged_on_directed_reachability_alone():
    # Sanity: on a genuine make_instance() undirected instance the undirected verdict and the
    # directed verdict coincide for parts without foreign terminals, so the internal
    # consistency check never fires on a valid partition (60 random cases).
    rng = random.Random(4242)
    for _ in range(60):
        n = rng.randint(3, 10)
        graph = nx.gnp_random_graph(n, rng.uniform(0.25, 0.8), seed=rng.randint(0, 10**6))
        k = rng.randint(1, min(3, n))
        terminals = rng.sample(range(n), k)
        parts = [[t] for t in terminals]
        for v in range(n):
            if v not in terminals:
                parts[rng.randrange(k)].append(v)
        inst = make_instance(n, list(graph.edges()), terminals, sizes=[len(p) for p in parts], directed=False)
        report = verify_instance_parts(inst, parts)
        assert not any("internal inconsistency" in e for e in report.errors), report.errors
        assert report.valid == independent_verdict(inst, parts)


def test_extra_or_missing_parts_are_rejected_even_if_the_rest_is_perfect():
    inst = path_und(5, [0, 4], [2, 3])
    assert verify_instance_parts(inst, [[0, 1], [4, 3, 2], []]).valid is False
    assert verify_instance_parts(inst, [[0, 1], [4, 3, 2], [1]]).valid is False
    assert verify_instance_parts(inst, [[0, 1]]).valid is False
    assert verify_instance_parts(inst, []).valid is False


# ---------------------------------------------------------------------------
# valid partitions that MUST be accepted
# ---------------------------------------------------------------------------


def _arborescence_partition(rng: random.Random, inst: Instance) -> list[list[int]] | None:
    """Multi-source reverse BFS from the terminals over ``inst.arcs``: every
    vertex joins the part of the terminal that discovered it, so every part
    carries a spanning in-arborescence [Def connected-to].  ``None`` when some
    vertex reaches no terminal."""
    in_adj = inst.in_adjacency()
    owner = {t: i for i, t in enumerate(inst.terminals)}
    frontier = list(inst.terminals)
    rng.shuffle(frontier)
    while frontier:
        x = frontier.pop(rng.randrange(len(frontier)))
        preds = list(in_adj[x])
        rng.shuffle(preds)
        for u in preds:
            if u not in owner:
                owner[u] = owner[x]
                frontier.append(u)
    if len(owner) != inst.n:
        return None
    parts: list[list[int]] = [[t] for t in inst.terminals]
    for v in range(inst.n):
        if v not in inst.terminals:
            parts[owner[v]].append(v)
    return parts


def test_random_arborescence_partitions_are_accepted_directed_and_undirected():
    rng = random.Random(2024)
    accepted = 0
    for _ in range(150):
        n = rng.randint(2, 12)
        k = rng.randint(1, min(4, n))
        directed = rng.random() < 0.5
        terminals = rng.sample(range(n), k)
        p = rng.uniform(0.2, 0.8)
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
        if not directed:
            edges = [(u, v) for u, v in edges if u < v]
        probe = make_instance(n, edges, terminals, [0] * (k - 1) + [n - k], directed=directed)
        parts = _arborescence_partition(rng, probe)
        if parts is None:
            continue
        sizes = [len(part) for part in parts]
        if rng.random() < 0.3:
            weights = [0 if v in terminals else rng.randint(1, 3) for v in range(n)]
            caps = [sum(weights[v] for v in part) for part in parts]
            inst = make_instance(n, edges, terminals, caps, weights=weights, directed=directed)
        else:
            inst = make_instance(n, edges, terminals, sizes=sizes, directed=directed)
        # shuffle the vertex order inside each part: order must not matter
        shuffled = [rng.sample(part, len(part)) for part in parts]
        report = verify_instance_parts(inst, shuffled)
        assert report.valid is True, (inst, shuffled, report.errors)
        assert independent_verdict(inst, shuffled) is True
        accepted += 1
    assert accepted >= 100


def test_all_input_forms_agree_on_the_same_valid_and_invalid_partition():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    inst = make_instance(5, edges, [0, 3], sizes=[2, 3], directed=False)
    graph = nx.Graph(edges)
    good = [[0, 1], [3, 2, 4]]
    bad = [[0, 2], [3, 1, 4]]
    verdicts_good = {
        verify_partition(inst, parts=good).valid,
        verify_partition(graph, [0, 3], sizes=[2, 3], parts=good).valid,
        verify_partition((5, edges), [0, 3], sizes=[2, 3], parts=good, directed=False).valid,
    }
    verdicts_bad = {
        verify_partition(inst, parts=bad).valid,
        verify_partition(graph, [0, 3], sizes=[2, 3], parts=bad).valid,
        verify_partition((5, edges), [0, 3], sizes=[2, 3], parts=bad, directed=False).valid,
    }
    assert verdicts_good == {True}
    assert verdicts_bad == {False}


# ---------------------------------------------------------------------------
# differential test against the independent NetworkX oracle
# ---------------------------------------------------------------------------


def _perturb(rng: random.Random, parts: list[list[int]], terminals: list[int]) -> list[list[int]]:
    k = len(parts)
    out = [list(p) for p in parts]
    mode = rng.choice(["none", "swap", "move", "drop", "dup", "foreign", "extra", "fewer", "rotate"])
    if mode == "swap" and k >= 2:
        i, j = rng.sample(range(k), 2)
        if len(out[i]) > 1 and len(out[j]) > 1:
            a = out[i].pop(rng.randrange(1, len(out[i])))
            b = out[j].pop(rng.randrange(1, len(out[j])))
            out[i].append(b)
            out[j].append(a)
    elif mode == "move" and k >= 2:
        i, j = rng.sample(range(k), 2)
        if len(out[i]) > 1:
            out[j].append(out[i].pop(rng.randrange(1, len(out[i]))))
    elif mode == "drop":
        i = rng.randrange(k)
        if len(out[i]) > 1:
            out[i].pop(rng.randrange(1, len(out[i])))
    elif mode == "dup":
        i = rng.randrange(k)
        out[i].append(rng.choice(out[rng.randrange(k)]))
    elif mode == "foreign" and k >= 2:
        i, j = rng.sample(range(k), 2)
        out[i].append(terminals[j])
    elif mode == "extra":
        out.append([])
    elif mode == "fewer":
        out = out[:-1]
    elif mode == "rotate" and k >= 2:
        out = out[1:] + out[:1]
    return out


def test_verifier_agrees_with_independent_oracle_on_random_partitions():
    rng = random.Random(777)
    n_valid = n_invalid = 0
    for trial in range(1500):
        n = rng.randint(2, 9)
        k = rng.randint(1, min(3, n))
        directed = rng.random() < 0.5
        terminals = rng.sample(range(n), k)
        p = rng.uniform(0.15, 0.8)
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < p]
        if not directed:
            edges = [(u, v) for u, v in edges if u < v]
        parts = [[t] for t in terminals]
        for v in range(n):
            if v not in terminals:
                parts[rng.randrange(k)].append(v)
        if rng.random() < 0.4:
            weights = [0 if v in terminals else rng.randint(1, 4) for v in range(n)]
            caps = [0] * k
            for _ in range(sum(weights) + rng.randint(0, 3)):
                caps[rng.randrange(k)] += 1
            inst = make_instance(n, edges, terminals, caps, weights=weights, directed=directed)
        else:
            sizes = [len(part) for part in parts]
            if k >= 2 and rng.random() < 0.3:  # capacities off by one, sum preserved
                i, j = rng.sample(range(k), 2)
                if sizes[i] > 1:
                    sizes[i] -= 1
                    sizes[j] += 1
            inst = make_instance(n, edges, terminals, sizes=sizes, directed=directed)
        candidate = _perturb(rng, parts, terminals)
        report = verify_instance_parts(inst, candidate)
        expected = independent_verdict(inst, candidate)
        assert report.valid == expected, (trial, inst, candidate, report.errors)
        assert report.valid == (report.errors == [])
        assert not any("internal inconsistency" in e for e in report.errors)
        n_valid += expected
        n_invalid += not expected
    assert n_valid > 200 and n_invalid > 200


# ---------------------------------------------------------------------------
# verify_partition label handling (NetworkX graphs)
# ---------------------------------------------------------------------------


def test_networkx_partition_naming_a_node_absent_from_the_graph_is_rejected():
    """A partition that names a node the graph does not contain is not a
    partition of the graph.  ``_map_label`` documents that unknown labels are
    passed through 'so that verify_instance_parts reports them'; an unknown
    *integer* label that happens to lie in ``0..n-1`` must be reported too,
    otherwise a partition of the wrong node set is accepted as valid."""
    graph = nx.path_graph([10, 11, 12, 13])  # internal ids 0..3
    ok = verify_partition(graph, [10, 13], sizes=[2, 2], parts=[[10, 11], [13, 12]])
    assert ok.valid is True
    report = verify_partition(graph, [10, 13], sizes=[2, 2], parts=[[10, 1], [13, 2]])
    assert report.valid is False, (
        "nodes 1 and 2 are not in the graph (nodes 11 and 12 are missing), "
        f"yet the partition was accepted: {report.errors}"
    )


# ---------------------------------------------------------------------------
# I/O round trips (glsolver.io)
# ---------------------------------------------------------------------------


def _weighted_undirected_with_meta() -> Instance:
    # includes a terminal-terminal edge (0,3) and an edge listed terminal-first
    return make_instance(
        6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3)], [0, 3], [4, 3],
        weights=[0, 1, 2, 7, 3, 1], directed=False, name="w-und",
        meta={"family": "x", "seed": 3, "kappa": 2, "nested": {"a": [1, 2.5, None, True], "b": "s"}},
    )


def _directed_weighted_with_terminal_out_arcs() -> Instance:
    return make_instance(
        5, [(2, 0), (2, 1), (3, 0), (3, 1), (4, 3), (0, 4), (1, 0)], [0, 1], [4, 2],
        weights=[0, 0, 3, 1, 2], directed=True, name="dw", meta={"note": "arcs (0,4),(1,0) dropped"},
    )


@pytest.mark.parametrize("factory", [_weighted_undirected_with_meta, _directed_weighted_with_terminal_out_arcs])
def test_json_round_trip_preserves_every_field(tmp_path, factory):
    inst = factory()
    d = instance_to_dict(inst)
    text = json.dumps(d)  # must be plain JSON
    back = instance_from_dict(json.loads(text))
    for attr in ("n", "arcs", "terminals", "capacities", "weights", "directed", "undirected_edges", "name"):
        assert getattr(back, attr) == getattr(inst, attr), attr
    assert back.meta == inst.meta  # Instance.__eq__ ignores meta, so compare explicitly
    assert back == inst
    # file level, twice (idempotent), and a saved-then-loaded-then-saved file is byte identical
    path = tmp_path / "inst.json"
    save_instance(inst, path)
    loaded = load_instance(path)
    assert loaded == inst and loaded.meta == inst.meta and loaded.weights == inst.weights
    save_instance(loaded, tmp_path / "again.json")
    assert (tmp_path / "again.json").read_bytes() == path.read_bytes()
    # the verifier gives the same verdict on the original and the reloaded instance
    parts = [[0, 1, 2], [3, 4, 5]] if not inst.directed else [[0, 2, 4, 3], [1]]
    assert verify_instance_parts(loaded, parts).valid == verify_instance_parts(inst, parts).valid


def test_json_round_trip_keeps_terminal_terminal_edge_and_terminal_weights_zero():
    inst = _weighted_undirected_with_meta()
    d = instance_to_dict(inst)
    assert [0, 3] in d["edges"]
    assert d["weights"][0] == 0 and d["weights"][3] == 0
    assert d["weights"] == [0, 1, 2, 0, 3, 1]  # terminal 3's input weight 7 was forced to 0 on creation
    back = instance_from_dict(d)
    assert back.undirected_edges == inst.undirected_edges
    assert back.total_weight == inst.total_weight == 7
    assert back.w_max == inst.w_max == 3


def test_meta_is_json_normalised_but_otherwise_preserved():
    """JSON has no tuples/sets/int keys; those are normalised on the way out
    and everything JSON-representable survives unchanged."""
    np = pytest.importorskip("numpy")
    inst = make_instance(
        3, [(1, 0), (2, 0)], [0], [2], directed=True,
        meta={"shape": (2, 3), "tags": {"b", "a"}, 7: "int-key", "np": np.int64(4), "arr": np.array([1, 2]),
              "plain": {"x": [1, 2, {"y": None}], "f": 0.5, "t": True}},
    )
    back = instance_from_dict(json.loads(json.dumps(instance_to_dict(inst))))
    assert back.meta == {
        "shape": [2, 3], "tags": ["a", "b"], "7": "int-key", "np": 4, "arr": [1, 2],
        "plain": {"x": [1, 2, {"y": None}], "f": 0.5, "t": True},
    }
    assert back == inst


def test_sizes_form_and_capacities_form_load_to_identical_instances(tmp_path):
    base = {"name": "p", "directed": False, "n": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]], "terminals": [0, 4]}
    (tmp_path / "s.json").write_text(json.dumps(dict(base, sizes=[2, 3])))
    (tmp_path / "c.json").write_text(json.dumps(dict(base, capacities=[1, 2])))
    a, b = load_instance(tmp_path / "s.json"), load_instance(tmp_path / "c.json")
    assert a == b and a.capacities == (1, 2) and a.sizes == (2, 3)
    assert instance_to_dict(a)["capacities"] == [1, 2] and "sizes" not in instance_to_dict(a)


def test_edgelist_matches_json_for_the_same_graph(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("0 1\n1 2 0.5\n2 3\n3 0\n# done\n")
    from_list = load_edgelist(path, [0, 2], sizes=[2, 2], name="g")
    from_json = instance_from_dict({
        "name": "g", "directed": False, "n": 4, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "terminals": [0, 2], "sizes": [2, 2],
    })
    assert from_list == from_json
    d_from_list = load_edgelist(path, [0, 2], [1, 1], directed=True, name="g")
    assert d_from_list.arcs == ((1, 2), (3, 0))  # (0,1) and (2,3) leave terminals
    assert d_from_list.directed is True


def test_solution_round_trip_preserves_parts_assignment_and_certificate(tmp_path):
    inst = _weighted_undirected_with_meta()
    result = SimpleNamespace(
        parts=[[0, 1, 2], [3, 4, 5]], assignment=[0, 0, 0, 1, 1, 1], algorithm="weighted", status="ok",
        valid=True, runtime=0.25, stats={"max_flow_calls": 2, "time_total": 0.25},
        certificate={"parents": {1: 0, 2: 1, 4: 3, 5: 4}, "witness": {1: {0: 1}, 2: {0: 2}}},
        message="", instance=inst,
    )
    d = result_to_dict(result)
    save_solution(tmp_path / "sol.json", d)
    back = load_solution(tmp_path / "sol.json")
    assert back == d
    assert back["instance"] == "w-und"
    assert back["parts"] == [[0, 1, 2], [3, 4, 5]]
    assert back["assignment"] == [0, 0, 0, 1, 1, 1]
    assert back["certificate"]["parents"] == {"1": 0, "2": 1, "4": 3, "5": 4}
    assert back["certificate"]["witness"] == {"1": {"0": 1}, "2": {"0": 2}}
    # the parts stored in the file still verify against the reloaded instance
    assert verify_instance_parts(inst, back["parts"]).valid is True
