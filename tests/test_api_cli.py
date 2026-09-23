"""API and CLI tests (dispatcher, result object, precondition semantics, CLI subcommands)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from glsolver import GLResult, Instance, core_available, glpartition, make_instance, partition
from glsolver.api import choose_algorithm
from glsolver.generators import harary_graph, paper_running_example, random_kT_connected_dag
from glsolver.io import load_instance, save_instance
from glsolver.testing.minimize import minimize_instance
from glsolver.testing.registry import available_solvers, solvers_for


def test_networkx_undirected_partition_labels():
    G = nx.relabel_nodes(nx.petersen_graph(), {i: f"v{i}" for i in range(10)})
    res = partition(G, terminals=["v0", "v5", "v7"], sizes=[3, 3, 4])
    assert isinstance(res, GLResult)
    assert res.status == "ok" and res.valid is True
    labels = res.parts_labels
    assert sorted(sum(labels, [])) == sorted(G.nodes())
    assert "v0" in labels[0] and "v5" in labels[1] and "v7" in labels[2]
    assert res.runtime >= 0 and "peak_rss_mb" in res.stats


def test_tuple_input_and_capacities():
    res = glpartition((4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]), [0, 2], [1, 1], directed=False)
    assert res.valid is True and sorted(map(len, res.parts)) == [2, 2]


def test_auto_dispatch_names():
    assert choose_algorithm(paper_running_example()) in ("general", "reference")
    dag = random_kT_connected_dag(20, 3, seed=1)
    assert choose_algorithm(dag) in ("dag", "reference-dag")


def test_precondition_failed_is_not_infeasible():
    # vertex 3 cannot reach any terminal -> FEAC fails; the solver must not claim non-existence
    inst = make_instance(4, [(2, 0), (2, 1), (3, 2)], [0, 1], [1, 1], directed=True)
    res = glpartition(inst, algorithm="reference")
    assert res.status == "precondition_failed"
    assert "no witness" in res.message or "Flow-Essential" in res.message
    # the oracle can decide: here a partition actually exists? {0,2},{1,3}: 3->2 not in part 1... 3 has arc only to 2
    ora = glpartition(inst, algorithm="bruteforce")
    assert ora.status in ("ok", "infeasible")


def test_registry_lists_reference_backends():
    names = set(available_solvers(include_oracles=True))
    assert {"reference", "reference-weighted", "reference-dag", "bruteforce"} <= names
    if core_available():
        assert {"general", "weighted", "dag"} <= names
    inst = harary_graph(12, 3, seed=0)
    assert "reference-dag" not in solvers_for(inst)


def test_minimizer_shrinks_while_predicate_holds():
    inst = harary_graph(14, 3, seed=2)
    # predicate: instance still has >= 8 arcs (trivially shrinkable); minimizer must keep it valid
    small = minimize_instance(inst, lambda i: i.m >= 8, max_rounds=5)
    small.validate()
    assert small.m < inst.m


@pytest.mark.parametrize("algo", ["reference", "bruteforce", "ilp"])
def test_all_backends_agree_on_paper_example(algo):
    res = glpartition(paper_running_example(), algorithm=algo)
    assert res.status == "ok" and res.valid is True


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "glsolver.cli", *args], capture_output=True, text=True)


def test_cli_roundtrip(tmp_path: Path):
    inst = harary_graph(16, 3, seed=5)
    p = tmp_path / "inst.json"
    save_instance(inst, p)
    sol = tmp_path / "sol.json"
    r = _run_cli("solve", str(p), "--stats", "--algorithm", "reference", "-o", str(sol))
    assert r.returncode == 0, r.stderr
    assert "verifier=VALID" in r.stdout
    data = json.loads(sol.read_text())
    assert data["status"] == "ok" and data["valid"] is True
    r2 = _run_cli("verify", str(p), str(sol))
    assert r2.returncode == 0 and r2.stdout.startswith("VALID")
    r3 = _run_cli("verify", str(p), "--parts", "0,1;2,3;4")
    assert r3.returncode == 1 and "INVALID" in r3.stdout
    r4 = _run_cli("inspect", str(p), "--preconditions")
    assert r4.returncode == 0 and '"k_T_connected": true' in r4.stdout


def test_cli_edgelist_shorthand(tmp_path: Path):
    e = tmp_path / "g.txt"
    e.write_text("0 1\n1 2\n2 3\n3 0\n0 2\n1 3\n")
    r = _run_cli(str(e), "--terminals", "0,2", "--sizes", "2,2")
    assert r.returncode == 0 and "verifier=VALID" in r.stdout


def test_cli_generate(tmp_path: Path):
    out = tmp_path / "gen.json"
    r = _run_cli("generate", "harary_graph", "--n", "20", "--k", "3", "--seed", "1", "-o", str(out))
    assert r.returncode == 0, r.stderr
    inst = load_instance(out)
    assert isinstance(inst, Instance) and inst.n == 20 and inst.k == 3


# ---------------------------------------------------------------------------
# numpy fast path (RESEARCH_NOTES.md E4): array-backed instances, vectorized normalization, array bindings,
# lazy certificate
# ---------------------------------------------------------------------------
import copy  # noqa: E402
import dataclasses  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402

from glsolver.api import _assignment_from_parts, _parts_from_assignment  # noqa: E402
from glsolver.instance import normalize_arcs, normalize_arcs_array  # noqa: E402


def _normalize_arcs_reference(n, edges, terminals, directed):
    """The historical element-wise normalization (kept verbatim as the oracle for the vectorized one)."""
    tset = set(terminals)
    seen = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of range for n={n}")
        if u == v:
            continue
        cand = [(u, v), (v, u)] if not directed else [(u, v)]
        for a, b in cand:
            if a in tset:
                continue
            seen.add((a, b))
    return tuple(sorted(seen))


def test_instance_array_storage_is_lazy_and_equivalent():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (2, 2), (1, 0)]
    a = make_instance(4, edges, [0, 2], [1, 1], directed=False)
    b = make_instance(4, np.array(edges, dtype=np.int64), [0, 2], [1, 1], directed=False)
    # the array is the storage: int32, (m, 2), read-only; the tuple form is materialized on demand
    assert a.__dict__["_arcs_tuple"] is None and a.arc_array.dtype == np.int32 and a.arc_array.shape == (a.m, 2)
    assert not a.arc_array.flags.writeable
    assert a.m == 6 and a.undirected_edge_array.shape == (6, 2)
    assert a.arcs == ((1, 0), (1, 2), (1, 3), (3, 0), (3, 1), (3, 2)) and a.__dict__["_arcs_tuple"] is not None
    c = Instance(4, tuple(a.arcs), (0, 2), (1, 1), directed=False, undirected_edges=a.undirected_edges)
    assert a == b == c and hash(a) == hash(b) == hash(c)
    assert a.undirected_edges == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert isinstance(a.arcs[0][0], int) and isinstance(a.undirected_edges[0][1], int)
    # tuple-backed instances expose the same array view
    assert np.array_equal(c.arc_array, a.arc_array) and np.array_equal(c.undirected_edge_array, a.undirected_edge_array)
    assert c.m == 6 and a.out_adjacency() == c.out_adjacency() and a.in_adjacency() == c.in_adjacency()
    # dataclass machinery, copies and pickles see the canonical tuple form
    d = dataclasses.replace(a, name="renamed")
    assert d.arcs == a.arcs and d.name == "renamed" and d.m == a.m
    for clone in (copy.deepcopy(b), pickle.loads(pickle.dumps(b)), copy.copy(b)):
        assert clone == b and clone.m == b.m and clone.arcs == b.arcs and clone.undirected_edges == b.undirected_edges
    assert "arcs=((1, 0)" in repr(a)
    # a writeable caller array is never frozen by the instance
    src = np.array([[2, 0], [3, 1]], dtype=np.int32)
    inst = Instance(4, src, (0, 1), (1, 1))
    assert src.flags.writeable and not inst.arc_array.flags.writeable and inst.arcs == ((2, 0), (3, 1))
    inst.validate()


def test_array_backed_validate_reports_the_same_errors():
    def bad(arcs, terminals=(0, 1), n=4, caps=(1, 1)):
        with pytest.raises(ValueError) as exc:
            Instance(n, np.array(arcs, dtype=np.int32).reshape(-1, 2), terminals, caps).validate()
        with pytest.raises(ValueError) as exc_ref:
            Instance(n, tuple(map(tuple, arcs)), terminals, caps).validate()
        assert str(exc.value) == str(exc_ref.value)
        return str(exc.value)

    assert "duplicate arcs" in bad([(2, 0), (2, 0), (3, 1)])
    assert "self-loop at 2" in bad([(2, 2), (3, 1)])
    assert "leaves a terminal" in bad([(0, 2), (3, 1)])
    assert "out of range" in bad([(2, 7), (3, 1)])
    assert "out of range" in bad([(2, 0), (-1, 1)])
    assert "self-loop" in bad([(2, 0), (3, 3), (3, 3)])  # first offending arc in stored order wins, as before
    Instance(4, np.zeros((0, 2), dtype=np.int32), (0, 1, 2, 3), (0, 0, 0, 0)).validate()


@pytest.mark.parametrize("directed", [True, False])
def test_vectorized_normalization_matches_the_element_wise_reference(directed):
    rng = random.Random(7)
    for trial in range(200):
        n = rng.randint(1, 12)
        k = rng.randint(1, n)
        terminals = rng.sample(range(n), k)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 30))]
        if trial % 3 == 0:
            edges = [(u, v, 99) for u, v in edges]  # a third field is ignored, as int(e[0]), int(e[1]) did
        ref = _normalize_arcs_reference(n, edges, terminals, directed)
        assert normalize_arcs(n, edges, terminals, directed) == ref
        arr = normalize_arcs_array(n, np.array(edges).reshape(-1, 3 if trial % 3 == 0 else 2), terminals, directed)
        assert arr.dtype == np.int32 and tuple(map(tuple, arr.tolist())) == ref
        assert normalize_arcs(n, iter(edges), terminals, directed) == ref  # generic iterables still work
        assert normalize_arcs(n, [(float(u), float(v)) for u, v, *_ in edges], terminals, directed) == ref
    with pytest.raises(ValueError, match=r"edge \(3,5\) out of range for n=4"):
        normalize_arcs(4, [(0, 1), (3, 5)], [0], True)
    with pytest.raises(ValueError, match=r"edge \(3,5\) out of range for n=4"):
        make_instance(4, np.array([[0, 1], [3, 5]]), [0], [3])
    assert normalize_arcs(3, [], [0], False) == () and normalize_arcs(3, np.zeros((0, 2), int), [0], False) == ()


def test_make_instance_from_arrays_matches_lists_end_to_end():
    inst = harary_graph(60, 4, seed=3)
    ue = np.asarray(inst.undirected_edge_array)
    again = make_instance(inst.n, ue, inst.terminals, inst.capacities, directed=False, name=inst.name)
    assert again == inst and again.m == inst.m
    shuffled = ue[np.random.default_rng(1).permutation(len(ue))][:, ::-1]  # reversed, permuted view
    assert make_instance(inst.n, shuffled, inst.terminals, inst.capacities, directed=False, name=inst.name) == inst
    if core_available("general"):
        r1 = glpartition(inst, algorithm="general")
        r2 = glpartition(again, algorithm="general")
        assert r1.valid is True and r1.parts == r2.parts and r1.certificate == r2.certificate


def test_parts_and_assignment_helpers_match_the_loops():
    rng = random.Random(11)
    for _ in range(50):
        n, k = rng.randint(1, 40), rng.randint(1, 5)
        inst = Instance(n, (), tuple(range(k)), tuple([0] * k))  # only n / terminals are used by the helpers
        assignment = [rng.randint(-1, k - 1) for _ in range(n)]
        parts_ref = [[] for _ in range(k)]
        for v, i in enumerate(assignment):
            if i >= 0:
                parts_ref[i].append(v)
        parts = _parts_from_assignment(inst, assignment)
        assert parts == [sorted(p) for p in parts_ref]
        assert all(isinstance(v, int) for p in parts for v in p)
        parts_in = [rng.sample(range(-2, n + 2), rng.randint(0, min(n, 6))) for _ in range(k)]
        a_ref = [-1] * n
        for i, p in enumerate(parts_in):
            for v in p:
                if 0 <= v < n:
                    a_ref[v] = i
        assert _assignment_from_parts(inst, parts_in) == a_ref
    with pytest.raises(IndexError):
        _parts_from_assignment(Instance(3, (), (0,), (0,)), [0, 1, 0])


@pytest.mark.skipif(not core_available(), reason="C++ core not built")
def test_core_bindings_accept_numpy_arc_arrays():
    from glsolver import _core

    inst = harary_graph(40, 3, seed=2)
    arcs_list = [list(a) for a in inst.arcs]
    ref = _core.solve_general(inst.n, arcs_list, list(inst.terminals), list(inst.capacities), {"threads": 1})
    keys = ("status", "assignment", "parent", "witness", "k_T_connected")
    for arr in (inst.arc_array, inst.arc_array.astype(np.int64), np.ascontiguousarray(inst.arc_array[::-1]),
                np.asfortranarray(inst.arc_array), inst.arc_array.astype(np.uint16)):
        out = _core.solve_general(inst.n, arr, list(inst.terminals), list(inst.capacities), {"threads": 1})
        assert {key: out[key] for key in keys} == {key: ref[key] for key in keys}, arr.dtype
        assert out["stats"]["contractions"] == ref["stats"]["contractions"]
    w = [0 if v in inst.terminals else 1 for v in range(inst.n)]
    refw = _core.solve_weighted(inst.n, arcs_list, list(inst.terminals), list(inst.capacities), w, {"threads": 1})
    outw = _core.solve_weighted(inst.n, inst.arc_array, list(inst.terminals), list(inst.capacities), w, {"threads": 1})
    assert {key: outw[key] for key in keys} == {key: refw[key] for key in keys}
    assert _core.saturating_matching(inst.n, inst.arc_array, list(inst.terminals)) == \
        _core.saturating_matching(inst.n, arcs_list, list(inst.terminals))
    assert _core.minimal_hall_deficient_set(inst.n, inst.arc_array, list(inst.terminals)) == \
        _core.minimal_hall_deficient_set(inst.n, arcs_list, list(inst.terminals))
    dag = random_kT_connected_dag(30, 3, seed=4)
    d_ref = _core.dag_partition(dag.n, [list(a) for a in dag.arcs], list(dag.terminals), list(dag.capacities))
    d_arr = _core.dag_partition(dag.n, dag.arc_array, list(dag.terminals), list(dag.capacities))
    assert d_arr["status"] == d_ref["status"] == "ok" and d_arr["assignment"] == d_ref["assignment"]
    assert d_arr["parent"] == d_ref["parent"]
    # empty arrays are fine; type / shape problems are Python exceptions (as a wrong argument type always was);
    # value problems are status "error" (as for lists)
    assert _core.solve_general(2, np.zeros((0, 2), dtype=np.int32), [0, 1], [0, 0], {})["status"] == "ok"
    assert _core.solve_general(2, np.zeros(0, dtype=np.int64), [0, 1], [0, 0], {})["status"] == "ok"
    with pytest.raises(ValueError):
        _core.solve_general(inst.n, inst.arc_array.reshape(-1, 3)[:2], list(inst.terminals), list(inst.capacities), {})
    with pytest.raises(TypeError):
        _core.solve_general(inst.n, inst.arc_array.astype(np.float64), list(inst.terminals), list(inst.capacities), {})
    with pytest.raises(TypeError):
        _core.solve_general(inst.n, "0 1 1 2", list(inst.terminals), list(inst.capacities), {})
    with pytest.raises(TypeError):
        _core.solve_general(inst.n, [(0, 1, 2)], list(inst.terminals), list(inst.capacities), {})
    huge = np.array([[2, 0], [2**40, 1]], dtype=np.int64)
    out = _core.solve_general(3, huge, [0, 1], [1, 0], {})
    assert out["status"] == "error" and "does not fit" in out["message"]
    out = _core.solve_general(3, np.array([[2, 0], [2, 9]], dtype=np.int32), [0, 1], [1, 0], {})
    assert out["status"] == "error" and "out of range" in out["message"]


@pytest.mark.skipif(not core_available("general"), reason="C++ core not built")
def test_core_certificate_is_built_lazily_and_serializes():
    inst = harary_graph(50, 4, seed=1)
    res = glpartition(inst, algorithm="general")
    assert res.valid is True and callable(res.__dict__["_certificate"])  # not built yet
    parents = res.certificate["parents"]
    assert not callable(res.__dict__["_certificate"]) and res.certificate is res.certificate
    original = set(inst.arcs)
    assert set(parents) == set(range(inst.n)) - set(inst.terminals)
    assert all((v, p) in original and res.assignment[v] == res.assignment[p] for v, p in parents.items())
    assert set(res.certificate["witness"]) == set(parents) and set(res.certificate["witness"].values()) <= set(inst.terminals)
    d = res.to_dict()
    assert json.loads(json.dumps(d))["certificate"]["parents"] == {str(v): p for v, p in parents.items()}
    ref = glpartition(inst, algorithm="reference")
    assert isinstance(ref.certificate, dict) and set(ref.certificate["parents"]) == set(parents)
    res.certificate = {"parents": {}}  # plain assignment still works (the field keeps its documented type)
    assert res.certificate == {"parents": {}}
    failed = glpartition(make_instance(4, [(2, 0), (2, 1), (3, 2)], [0, 1], [1, 1]), algorithm="general")
    assert failed.status == "precondition_failed" and failed.certificate == {}
