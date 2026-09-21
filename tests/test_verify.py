"""Tests for the independent verifier ``glsolver.verify``."""
from __future__ import annotations

import random

import networkx as nx
import pytest

from glsolver.instance import Instance, make_instance
from glsolver.verify import VerificationReport, verify_instance_parts, verify_partition

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

# Paper Figure 1 running example: t1,t2,t3 = 0,1,2; v4..v9 = 3..8.
RUNNING_ARCS = [
    (7, 3), (7, 4), (4, 3), (3, 0), (3, 1), (4, 1),
    (8, 5), (8, 6), (5, 6), (5, 1), (6, 1), (6, 2),
]
RUNNING_PARTS = [[0, 3, 7], [1, 4, 5], [2, 6, 8]]


def running_example() -> Instance:
    return make_instance(9, RUNNING_ARCS, [0, 1, 2], [2, 2, 2], directed=True)


def path_instance(n: int = 5) -> Instance:
    edges = [(i, i + 1) for i in range(n - 1)]
    return make_instance(n, edges, [0, n - 1], sizes=[2, n - 2], directed=False)


def cycle_instance(n: int = 6) -> Instance:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return make_instance(n, edges, [0, n // 2], sizes=[n // 2, n - n // 2], directed=False)


def assert_valid(report: VerificationReport) -> None:
    assert report.errors == []
    assert report.valid is True
    assert all(report.details["connectivity_ok"])


def assert_error(report: VerificationReport, *substrings: str) -> None:
    assert report.valid is False
    assert report.errors, "expected at least one error"
    for sub in substrings:
        assert any(sub in e for e in report.errors), f"{sub!r} not found in {report.errors}"


# ---------------------------------------------------------------------------
# valid partitions
# ---------------------------------------------------------------------------


def test_report_dataclass_shape():
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3, 2]])
    assert isinstance(report, VerificationReport)
    for key in ("n", "k", "part_sizes", "bound_used", "connectivity_ok", "checked"):
        assert key in report.details
    assert report.details["n"] == 5
    assert report.details["k"] == 2
    assert report.details["part_sizes"] == [2, 3]
    assert report.details["bound_used"] == "exact"
    assert "part_weights" not in report.details  # unweighted
    assert set(report.details["checked"]) >= {
        "part_count", "vertex_ids", "cover", "terminal_membership", "sizes", "connectivity"
    }


def test_undirected_path_valid():
    assert_valid(verify_instance_parts(path_instance(), [[0, 1], [4, 3, 2]]))
    # order of vertices inside a part is irrelevant
    assert_valid(verify_instance_parts(path_instance(), [[1, 0], [2, 3, 4]]))


def test_undirected_cycle_valid():
    inst = cycle_instance(6)  # terminals 0 and 3, sizes 3 and 3
    assert_valid(verify_instance_parts(inst, [[0, 1, 2], [3, 4, 5]]))
    assert_valid(verify_instance_parts(inst, [[5, 0, 1], [2, 3, 4]]))


def test_undirected_grid_valid_via_networkx_labels():
    grid = nx.grid_2d_graph(3, 3)
    terminals = [(0, 0), (1, 0), (2, 0)]
    parts = [[(r, c) for c in range(3)] for r in range(3)]  # rows
    report = verify_partition(grid, terminals, sizes=[3, 3, 3], parts=parts)
    assert_valid(report)
    # columns are connected too but do not contain the right terminals
    cols = [[(r, c) for r in range(3)] for c in range(3)]
    report = verify_partition(grid, terminals, sizes=[3, 3, 3], parts=cols)
    assert_error(report, "terminal t_1", "is not in part 1", "part 0 contains terminal t_1")


def test_directed_running_example_valid():
    report = verify_instance_parts(running_example(), RUNNING_PARTS)
    assert_valid(report)
    assert report.details["directed"] is True
    assert report.details["connectivity_ok"] == [True, True, True]


def test_directed_singleton_parts_and_k1():
    # k = 1: a directed in-tree towards the single terminal 0
    inst = make_instance(4, [(1, 0), (2, 1), (3, 1)], [0], [3])
    assert_valid(verify_instance_parts(inst, [[0, 1, 2, 3]]))
    # capacity-0 terminal => singleton part
    inst = make_instance(3, [(2, 0), (2, 1)], [0, 1], [1, 0])
    assert_valid(verify_instance_parts(inst, [[0, 2], [1]]))


# ---------------------------------------------------------------------------
# every error class
# ---------------------------------------------------------------------------


def test_error_wrong_number_of_parts():
    inst = path_instance()
    assert_error(verify_instance_parts(inst, [[0, 1], [4, 3], [2]]), "expected 2 parts", "got 3")
    assert_error(verify_instance_parts(inst, [[0, 1, 2, 3, 4]]), "expected 2 parts", "got 1")


def test_error_missing_vertex():
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3]])
    assert_error(report, "vertex 2 is missing")
    # the size error is collected as well (all errors, not just the first)
    assert_error(report, "part 1 has size 2, expected c_1 + 1 = 3")


def test_error_duplicate_vertex():
    report = verify_instance_parts(path_instance(), [[0, 1, 2], [4, 3, 2]])
    assert_error(report, "vertex 2 appears 2 times", "parts [0, 1]")
    report = verify_instance_parts(path_instance(), [[0, 1, 1], [4, 3, 2]])
    assert_error(report, "vertex 1 appears 2 times", "parts [0]")


def test_error_out_of_range_and_non_integer_ids():
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3, 99]])
    assert_error(report, "part 1 contains out-of-range vertex id 99", "vertex 2 is missing")
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3, "2"]])
    assert_error(report, "part 1 contains non-integer vertex id '2'")
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3, -1]])
    assert_error(report, "out-of-range vertex id -1")
    report = verify_instance_parts(path_instance(), [[0, 1], [4, 3, 2.0]])
    assert_error(report, "non-integer vertex id 2.0")
    report = verify_instance_parts(path_instance(), [[0, True], [4, 3, 2]])
    assert_error(report, "non-integer vertex id True")


def test_error_terminal_not_in_its_part():
    report = verify_instance_parts(path_instance(), [[4, 3, 2], [0, 1]])
    assert_error(
        report,
        "terminal t_0 (vertex 0) is not in part 0",
        "terminal t_1 (vertex 4) is not in part 1",
        "part 0 contains terminal t_1 (vertex 4)",
        "part 1 contains terminal t_0 (vertex 0)",
    )
    assert report.details["connectivity_ok"] == [False, False]


def test_error_part_contains_other_terminal():
    inst = make_instance(4, [(0, 1), (1, 2), (2, 3)], [0, 3], sizes=[3, 1], directed=False)
    report = verify_instance_parts(inst, [[0, 1, 2, 3], [3]])
    assert_error(report, "part 0 contains terminal t_1 (vertex 3)", "vertex 3 appears 2 times")


def test_error_size_mismatch():
    report = verify_instance_parts(path_instance(), [[0, 1, 2], [4, 3]])
    assert_error(
        report,
        "part 0 has size 3, expected c_0 + 1 = 2",
        "part 1 has size 2, expected c_1 + 1 = 3",
    )
    # sizes are right per part in total but the exact-size rule is still per part
    inst = make_instance(4, [(0, 1), (1, 2), (2, 3)], [0, 3], sizes=[1, 3], directed=False)
    assert_error(verify_instance_parts(inst, [[0, 1], [3, 2]]), "part 0 has size 2, expected c_0 + 1 = 1")


def test_error_undirected_disconnected_part():
    report = verify_instance_parts(path_instance(), [[0, 1, 3], [4, 2]])
    assert_error(
        report,
        "part 0 is not connected: vertices [3] are not connected to terminal t_0 (vertex 0)",
        "part 1 is not connected: vertices [2] are not connected to terminal t_1 (vertex 4)",
    )
    assert report.details["connectivity_ok"] == [False, False]
    assert report.details["unreachable"] == {0: [3], 1: [2]}
    assert not any("internal inconsistency" in e for e in report.errors)


def test_error_directed_unreachable_terminal():
    inst = running_example()
    # v5 (4) has arcs only to v4 (3) and t2 (1); neither is in part 0 => cannot reach t1.
    report = verify_instance_parts(inst, [[0, 4, 7], [1, 3, 5], [2, 6, 8]])
    assert_error(
        report,
        "part 0 is not connected to its terminal: vertices [4, 7] cannot reach terminal t_0 (vertex 0)",
    )
    assert report.details["connectivity_ok"] == [False, True, True]
    assert report.details["unreachable"] == {0: [4, 7]}
    assert len(report.errors) == 1


def test_directed_orientation_matters():
    # In the undirected sense {0,3,4} is connected (0-3, 3-4) but directed arcs are 4->3->0 only;
    # reversing them breaks reachability of the terminal.
    forward = make_instance(3, [(2, 1), (1, 0)], [0], [2])
    assert_valid(verify_instance_parts(forward, [[0, 1, 2]]))
    backward = make_instance(3, [(1, 2)], [0], [2])  # arcs out of terminal 0 are dropped anyway
    report = verify_instance_parts(backward, [[0, 1, 2]])
    assert_error(report, "vertices [1, 2] cannot reach terminal t_0")


def test_directed_path_must_stay_inside_part():
    # 2 -> 3 -> 0 and 2 -> 1: putting 2 with terminal 0 but 3 with terminal 1 breaks part 0.
    inst = make_instance(4, [(2, 3), (3, 0), (2, 1)], [0, 1], [1, 1])
    assert_valid(verify_instance_parts(inst, [[0, 3], [1, 2]]))
    report = verify_instance_parts(inst, [[0, 2], [1, 3]])
    assert_error(report, "vertices [2] cannot reach terminal t_0", "vertices [3] cannot reach terminal t_1")


def test_all_errors_are_collected():
    inst = path_instance()
    report = verify_instance_parts(inst, [[1, 1, 9], [4, "x"]])
    assert_error(
        report,
        "vertex 1 appears 2 times",
        "out-of-range vertex id 9",
        "non-integer vertex id 'x'",
        "vertex 0 is missing",
        "vertex 2 is missing",
        "vertex 3 is missing",
        "terminal t_0 (vertex 0) is not in part 0",
        "part 0 has size 3, expected c_0 + 1 = 2",
        "part 1 has size 2, expected c_1 + 1 = 3",
    )


# ---------------------------------------------------------------------------
# undirected vs directed connectivity agreement (paper_notes §1.1)
# ---------------------------------------------------------------------------


def test_undirected_and_directed_connectivity_agree_on_random_partitions():
    rng = random.Random(12345)
    for trial in range(60):
        n = rng.randint(3, 12)
        k = rng.randint(1, min(3, n))
        graph = nx.gnp_random_graph(n, rng.uniform(0.2, 0.7), seed=rng.randint(0, 10**6))
        edges = list(graph.edges())
        terminals = rng.sample(range(n), k)
        # random assignment of non-terminals to parts
        parts = [[t] for t in terminals]
        for v in range(n):
            if v not in terminals:
                parts[rng.randrange(k)].append(v)
        sizes = [len(p) for p in parts]
        inst = make_instance(n, edges, terminals, sizes=sizes, directed=False)
        report = verify_instance_parts(inst, parts)
        assert not any("internal inconsistency" in e for e in report.errors), report.errors
        # cross-check the verdict against networkx on the induced subgraphs
        for i, part in enumerate(parts):
            expected = nx.is_connected(graph.subgraph(part))
            assert report.details["connectivity_ok"][i] == expected, (trial, i, part)
        assert report.valid == all(report.details["connectivity_ok"])


# ---------------------------------------------------------------------------
# weighted instances [Thm weighted-k-t-conn]
# ---------------------------------------------------------------------------


def weighted_star() -> Instance:
    # terminals 0,1 ; non-terminals 2 (w=3), 3 (w=1), 4 (w=2); every non-terminal sees both terminals
    arcs = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
    return make_instance(5, arcs, [0, 1], [3, 3], weights=[0, 0, 3, 1, 2])


def test_weighted_within_capacity_valid():
    inst = weighted_star()
    report = verify_instance_parts(inst, [[0, 2], [1, 3, 4]])
    assert_valid(report)
    assert report.details["bound_used"] == "c_t + w_max - 1"
    assert report.details["part_weights"] == [3, 3]
    assert report.details["exceeds_capacity"] == []
    assert report.details["w_max"] == 3
    assert "weights" in report.details["checked"]


def test_weighted_exceeds_capacity_but_within_bound_is_valid():
    inst = weighted_star()  # w_max = 3, bound = c_t + 2 = 5
    report = verify_instance_parts(inst, [[0, 2, 4], [1, 3]])  # part 0 weight 5 > c_0 = 3
    assert_valid(report)
    assert report.details["part_weights"] == [5, 1]
    assert report.details["exceeds_capacity"] == [0]


def test_weighted_exceeds_bound_is_invalid():
    inst = weighted_star()
    report = verify_instance_parts(inst, [[0, 2, 3, 4], [1]])  # part 0 weight 6 > 5
    assert_error(report, "part 0 has non-terminal weight 6, exceeding the bound c_0 + w_max - 1 = 3 + 3 - 1 = 5")
    assert report.details["part_weights"] == [6, 0]
    assert report.details["exceeds_capacity"] == [0]
    # sizes are NOT checked in the weighted case
    assert not any("expected c_" in e for e in report.errors)


def test_weighted_unit_weights_behave_like_exact_bound():
    inst = make_instance(5, [(2, 0), (3, 0), (4, 1), (3, 1)], [0, 1], [2, 1], weights=[0, 0, 1, 1, 1])
    assert_valid(verify_instance_parts(inst, [[0, 2, 3], [1, 4]]))
    report = verify_instance_parts(inst, [[0, 2], [1, 3, 4]])  # part 1 weight 2 > c_1 + 0
    assert_error(report, "part 1 has non-terminal weight 2, exceeding the bound")


# ---------------------------------------------------------------------------
# input forms of verify_partition
# ---------------------------------------------------------------------------


def test_networkx_string_labels():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    report = verify_partition(graph, ["a", "c"], sizes=[2, 2], parts=[["a", "b"], ["c", "d"]])
    assert_valid(report)
    report = verify_partition(graph, ["a", "c"], sizes=[2, 2], parts=[["a", "d"], ["c", "b"]])
    assert_valid(report)
    report = verify_partition(graph, ["a", "c"], sizes=[3, 1], parts=[["a", "b", "d"], ["c"]])
    assert_valid(report)
    report = verify_partition(graph, ["a", "c"], sizes=[3, 1], parts=[["a", "b", "c"], ["d"]])
    assert_error(report, "part 0 contains terminal t_1", "terminal t_1 (vertex 2) is not in part 1")
    report = verify_partition(graph, ["a", "c"], sizes=[2, 2], parts=[["a", "zz"], ["c", "d"]])
    assert_error(report, "unknown node label 'zz'", "is missing")


def test_networkx_digraph_and_capacities_positional():
    digraph = nx.DiGraph()
    digraph.add_edges_from([(u, v) for u, v in RUNNING_ARCS])
    # node order is insertion order, so labels != internal ids: parts are given in labels
    report = verify_partition(digraph, [0, 1, 2], [2, 2, 2], RUNNING_PARTS)
    assert_valid(report)
    labels = list(digraph.nodes())
    assert labels != sorted(labels)  # the mapping was actually exercised


def test_networkx_weights_dict():
    digraph = nx.DiGraph([(2, 0), (2, 1), (3, 0), (3, 1)])
    report = verify_partition(
        digraph, [0, 1], [2, 1], [[0, 2], [1, 3]], weights={2: 2, 3: 1}
    )
    assert_valid(report)
    assert report.details["part_weights"] == [2, 1]


def test_tuple_input_defaults_to_directed():
    report = verify_partition((9, RUNNING_ARCS), [0, 1, 2], [2, 2, 2], RUNNING_PARTS)
    assert_valid(report)
    assert report.details["directed"] is True
    # undirected tuple input
    report = verify_partition(
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]), [0, 4], sizes=[2, 3],
        parts=[[0, 1], [4, 3, 2]], directed=False,
    )
    assert_valid(report)
    assert report.details["directed"] is False
    report = verify_partition(
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]), [0, 4], sizes=[2, 3],
        parts=[[0, 2], [4, 3, 1]], directed=False,
    )
    assert_error(report, "part 0 is not connected", "part 1 is not connected")


def test_instance_input_rejects_extra_arguments():
    inst = path_instance()
    with pytest.raises(ValueError):
        verify_partition(inst, [0, 4], parts=[[0, 1], [4, 3, 2]])
    with pytest.raises(ValueError):
        verify_partition(inst, parts=None)
    with pytest.raises(ValueError):
        verify_partition((5, []), parts=[[0]])  # terminals missing
    with pytest.raises(TypeError):
        verify_partition("not a graph", [0], [0], [[0]])
    assert_valid(verify_partition(inst, parts=[[0, 1], [4, 3, 2]]))


def test_numpy_integer_ids_are_accepted():
    np = pytest.importorskip("numpy")
    inst = path_instance()
    parts = [np.array([0, 1]), np.array([4, 3, 2], dtype=np.int64)]
    assert_valid(verify_instance_parts(inst, parts))
