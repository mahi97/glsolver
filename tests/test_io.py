"""Tests for ``glsolver.io`` (instance / solution files)."""
from __future__ import annotations

import json
from types import SimpleNamespace

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


def undirected_grid() -> Instance:
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 5)]  # 0-5 joins two terminals
    return make_instance(
        6, edges, [0, 5], sizes=[3, 3], directed=False, name="grid", meta={"family": "grid", "seed": 7}
    )


def directed_weighted() -> Instance:
    arcs = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 3), (0, 4)]  # (0,4) leaves a terminal: dropped
    return make_instance(5, arcs, [0, 1], [4, 2], weights=[0, 0, 3, 1, 2], directed=True, name="w")


# ---------------------------------------------------------------------------
# dict round trips
# ---------------------------------------------------------------------------


def test_instance_to_dict_schema_undirected():
    inst = undirected_grid()
    d = instance_to_dict(inst)
    assert set(d) == {"name", "directed", "n", "edges", "terminals", "capacities", "weights", "meta"}
    assert d["directed"] is False
    assert d["n"] == 6
    assert d["terminals"] == [0, 5]
    assert d["capacities"] == [2, 2]
    assert d["weights"] is None
    assert d["meta"] == {"family": "grid", "seed": 7}
    # undirected edge list, including the terminal-terminal edge (0,5)
    assert [0, 5] in d["edges"]
    assert len(d["edges"]) == 8
    json.dumps(d)  # plain JSON types only


def test_instance_to_dict_schema_directed_weighted():
    inst = directed_weighted()
    d = instance_to_dict(inst)
    assert d["directed"] is True
    assert d["weights"] == [0, 0, 3, 1, 2]
    assert [0, 4] not in d["edges"]  # arc out of terminal was dropped at normalization
    assert d["edges"] == [list(a) for a in inst.arcs]


@pytest.mark.parametrize("factory", [undirected_grid, directed_weighted])
def test_dict_round_trip(factory):
    inst = factory()
    back = instance_from_dict(instance_to_dict(inst))
    assert back == inst
    assert back.meta == inst.meta
    assert back.undirected_edges == inst.undirected_edges
    assert back.name == inst.name


def test_instance_from_dict_sizes_form_and_defaults():
    d = {"n": 4, "edges": [[0, 1], [1, 2], [2, 3]], "terminals": [0, 3], "sizes": [2, 2]}
    inst = instance_from_dict(d)  # directed missing -> undirected; weights/meta/name optional
    assert inst.directed is False
    assert inst.capacities == (1, 1)
    assert inst.sizes == (2, 2)
    assert inst.weights is None
    assert inst.name == ""
    assert inst.meta == {}
    assert inst.undirected_edges == ((0, 1), (1, 2), (2, 3))
    assert inst == make_instance(4, [(0, 1), (1, 2), (2, 3)], [0, 3], sizes=[2, 2], directed=False)


def test_instance_from_dict_requires_exactly_one_of_capacities_sizes():
    base = {"n": 3, "edges": [[0, 1], [1, 2]], "terminals": [0, 2], "directed": False}
    with pytest.raises(ValueError):
        instance_from_dict(dict(base))
    with pytest.raises(ValueError):
        instance_from_dict(dict(base, capacities=[1, 0], sizes=[2, 1]))
    with pytest.raises(ValueError):
        instance_from_dict({"n": 3, "terminals": [0]})  # missing edges
    with pytest.raises(TypeError):
        instance_from_dict([1, 2, 3])
    assert instance_from_dict(dict(base, capacities=[1, 0])).capacities == (1, 0)
    assert instance_from_dict(dict(base, sizes=[2, 1], weights=None)).capacities == (1, 0)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [undirected_grid, directed_weighted])
def test_save_load_instance(tmp_path, factory):
    inst = factory()
    path = tmp_path / "inst.json"
    save_instance(inst, path)
    text = path.read_text()
    assert text.endswith("\n")
    data = json.loads(text)
    assert data == instance_to_dict(inst)
    assert load_instance(path) == inst
    # deterministic output: saving twice gives byte-identical files
    save_instance(inst, tmp_path / "again.json")
    assert (tmp_path / "again.json").read_text() == text
    # sorted keys -> deterministic key order
    keys = list(data)
    assert keys == sorted(keys)
    assert text.startswith("{\n \"capacities\"")  # indent=1


def test_load_instance_sizes_form(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "name": "p", "directed": False, "n": 4,
        "edges": [[0, 1], [1, 2], [2, 3]], "terminals": [0, 3], "sizes": [1, 3],
    }))
    inst = load_instance(path)
    assert inst.name == "p"
    assert inst.capacities == (0, 2)


def test_load_edgelist(tmp_path):
    path = tmp_path / "g.edgelist"
    path.write_text(
        "# a comment line\n"
        "\n"
        "0 1\n"
        "1 2 7.5\n"
        "2 3   # trailing comment\n"
        "3,4\n"
        "   \n"
        "4 0\n"
    )
    inst = load_edgelist(path, [0, 2], sizes=[2, 3])
    assert inst.n == 5
    assert inst.directed is False
    assert inst.name == "g"
    assert inst.undirected_edges == ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
    assert inst.capacities == (1, 2)
    # directed, explicit n and name, weights
    inst = load_edgelist(path, [0, 2], [4, 2], directed=True, n=7, name="seven", weights=[0, 1, 0, 1, 2, 1, 1])
    assert inst.n == 7
    assert inst.directed is True
    assert inst.name == "seven"
    assert inst.arcs == ((1, 2), (3, 4), (4, 0))  # (0,1) and (2,3) leave terminals
    assert inst.weights == (0, 1, 0, 1, 2, 1, 1)
    # n defaults to 1 + max id over edges AND terminals
    inst = load_edgelist(path, [0, 6], sizes=[5, 2])
    assert inst.n == 7


def test_load_edgelist_rejects_bad_lines(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 1\n2\n")
    with pytest.raises(ValueError):
        load_edgelist(path, [0, 2], sizes=[2, 1])
    path.write_text("0 -1\n")
    with pytest.raises(ValueError):
        load_edgelist(path, [0], sizes=[1])


# ---------------------------------------------------------------------------
# solutions
# ---------------------------------------------------------------------------


def test_solution_round_trip(tmp_path):
    solution = {
        "instance": "grid", "algorithm": "general", "status": "ok",
        "parts": [[0, 1, 2], [5, 4, 3]], "assignment": [0, 0, 0, 1, 1, 1],
        "valid": True, "runtime": 0.0123, "stats": {"max_flow_calls": 3},
        "certificate": {"parents": {"1": 0, "2": 1, "3": 4, "4": 5}},
        "message": "",
    }
    path = tmp_path / "sol.json"
    save_solution(path, solution)
    assert load_solution(path) == solution
    assert path.read_text().endswith("\n")
    with pytest.raises(TypeError):
        save_solution(path, ["not", "a", "dict"])


def test_result_to_dict_duck_typed(tmp_path):
    inst = undirected_grid()
    result = SimpleNamespace(
        parts=[(0, 1, 2), (5, 4, 3)],
        assignment=[0, 0, 0, 1, 1, 1],
        algorithm="general",
        status="ok",
        valid=True,
        runtime=0.5,
        stats={"max_flow_calls": 3, "time_total": 0.5, "graph_size_over_time": [(9, 12), (8, 10)]},
        certificate={"parents": {1: 0, 2: 1, 3: 4, 4: 5}, "witness": {1: 0, 4: 5}},
        message="",
        instance=inst,
    )
    d = result_to_dict(result)
    assert d["instance"] == "grid"
    assert d["algorithm"] == "general"
    assert d["status"] == "ok"
    assert d["parts"] == [[0, 1, 2], [5, 4, 3]]
    assert d["assignment"] == [0, 0, 0, 1, 1, 1]
    assert d["valid"] is True
    assert d["runtime"] == 0.5
    assert d["stats"]["max_flow_calls"] == 3
    assert d["stats"]["graph_size_over_time"] == [[9, 12], [8, 10]]
    assert d["certificate"] == {"parents": {"1": 0, "2": 1, "3": 4, "4": 5}, "witness": {"1": 0, "4": 5}}
    assert d["message"] == ""
    json.dumps(d)
    save_solution(tmp_path / "r.json", d)
    assert load_solution(tmp_path / "r.json") == d


def test_result_to_dict_missing_attributes_and_string_instance():
    d = result_to_dict(SimpleNamespace(status="precondition_failed", message="kappa(3) = 2 < 3", instance="x"))
    assert d["instance"] == "x"
    assert d["status"] == "precondition_failed"
    assert d["message"] == "kappa(3) = 2 < 3"
    assert d["parts"] is None
    assert d["assignment"] is None
    assert d["valid"] is None
    assert d["runtime"] is None
    assert d["stats"] == {}
    assert d["certificate"] == {}
    assert d["algorithm"] is None
    json.dumps(d)


def test_result_to_dict_numpy_values():
    np = pytest.importorskip("numpy")
    result = SimpleNamespace(
        parts=[np.array([0, 1])], assignment=np.array([0, 0]), algorithm="dag", status="ok",
        valid=np.bool_(True), runtime=np.float64(0.25), stats={"peak_rss_mb": np.float32(1.5)},
        certificate={"parents": {np.int64(1): np.int64(0)}}, message="", instance=None,
    )
    d = result_to_dict(result)
    assert d["parts"] == [[0, 1]]
    assert d["assignment"] == [0, 0]
    assert d["valid"] is True
    assert d["runtime"] == 0.25
    assert d["certificate"] == {"parents": {"1": 0}}
    assert d["instance"] == ""
    json.dumps(d)
