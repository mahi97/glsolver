"""Instance catalogue for the competitive comparison.

Every instance is addressed by a small JSON-serialisable spec so that a worker
subprocess can rebuild it deterministically without shipping the graph:

    {"family": "harary", "n": 200, "k": 4, "seed": 1}
    {"family": "paper_running_example"}
    {"family": "counterexample", "copies": 1}

Only ``glsolver.generators`` (and ``glref.counterexample`` for the official
counterexample of paper_notes §10) are used, so the instances are exactly the
ones the project's own benchmarks use.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "reference") not in sys.path:
    sys.path.insert(0, str(_ROOT / "reference"))

import networkx as nx  # noqa: E402

import glsolver.generators as gen  # noqa: E402
from glsolver.instance import Instance  # noqa: E402

SIZES = (10, 20, 50, 100, 200, 500, 1000, 2000)
K = 4
SEED = 1
SCALED_FAMILIES = ("harary", "random_regular", "erdos_renyi", "sparse_k_connected")


def build_instance(spec: dict[str, Any]) -> Instance:
    fam = spec["family"]
    if fam == "counterexample":
        from glref.counterexample import build_counterexample_instance

        return build_counterexample_instance(int(spec.get("copies", 1)))
    if fam in ("paper_running_example", "paper_contract_counterexample", "paper_essential_example"):
        return getattr(gen, fam)()
    n, k, seed = int(spec["n"]), int(spec["k"]), int(spec["seed"])
    cat = gen.family_catalog()
    if fam not in cat:
        raise KeyError(f"unknown family {fam!r}")
    return cat[fam](n, k, seed)


def instance_id(spec: dict[str, Any]) -> str:
    fam = spec["family"]
    if fam == "counterexample":
        return f"counterexample(copies={spec.get('copies', 1)})"
    if fam.startswith("paper_"):
        return fam
    return f"{fam}/n={spec['n']}"


def all_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for fam in SCALED_FAMILIES:
        for n in SIZES:
            specs.append({"family": fam, "n": n, "k": K, "seed": SEED})
    specs.append({"family": "paper_running_example"})
    specs.append({"family": "paper_contract_counterexample"})
    specs.append({"family": "paper_essential_example"})
    specs.append({"family": "counterexample", "copies": 1})
    return specs


# ---------------------------------------------------------------------------
# helpers shared by the competitor adapters
# ---------------------------------------------------------------------------
def undirected_view(inst: Instance) -> nx.Graph:
    """The undirected graph a partitioner is handed.

    For an undirected instance this is exactly the input graph.  For a directed
    instance it is the *underlying* undirected graph (arc orientations dropped)
    -- the most generous thing we can give a tool that has no notion of
    direction.  Isolated vertices are kept so node ids stay ``0..n-1``.
    """
    G = nx.Graph()
    G.add_nodes_from(range(inst.n))
    if inst.undirected_edges is not None:
        G.add_edges_from(inst.undirected_edges)
    else:
        G.add_edges_from((int(u), int(v)) for u, v in inst.arcs)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def target_sizes(inst: Instance) -> list[int]:
    """Required part sizes ``n_i = c_i + 1`` (unweighted instances)."""
    return [int(c) + 1 for c in inst.capacities]


__all__ = [
    "SIZES", "K", "SEED", "SCALED_FAMILIES",
    "build_instance", "instance_id", "all_specs", "undirected_view", "target_sizes",
]
