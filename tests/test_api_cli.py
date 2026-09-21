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
