# Instance file formats

All solvers, oracles, generators, the CLI and the benchmarks exchange instances
in one JSON schema (`*.json`) and optionally a plain edge-list format.

## JSON schema (canonical)

```json
{
  "name": "grid_5x5_k4_seed7",
  "directed": false,
  "n": 25,
  "edges": [[0,1],[1,2], ...],
  "terminals": [0, 4, 20, 24],
  "capacities": [5, 6, 6, 4],
  "weights": null,
  "meta": {"family": "grid", "seed": 7, "kappa": 2}
}
```

* `directed` — `false` for classical undirected GL. Undirected edges are
  unordered pairs; the loader symmetrizes them into arcs and drops arcs leaving
  terminals (paper §2).
* `capacities` — the paper's `c_i`: the number (or total weight) of
  **non-terminal** vertices for part `i`. For undirected/unweighted files you
  may instead give `"sizes"` (`n_i = c_i + 1`, including the terminal); exactly
  one of the two keys must be present.
* `weights` — `null` (unweighted) or a length-`n` array; terminal entries are
  ignored (forced to 0), non-terminals must be `>= 1`.
* `meta` — free-form provenance (generator family, seed, known connectivity…).

Solutions:

```json
{
  "instance": "grid_5x5_k4_seed7",
  "algorithm": "general",
  "status": "ok",
  "parts": [[0,1,2,5,6,7], [4,3,8,9,13,14,19], ...],
  "assignment": [0,0,0,1,1,0,...],
  "valid": true,
  "runtime": 0.0123,
  "stats": {...},
  "certificate": {"parents": {"1": 0, "2": 1, ...}}
}
```

`parts[i]` lists the vertices of `V_i` and always contains `terminals[i]`.
`assignment[v]` is the index `i` of the part containing `v`.

## Edge-list format (`*.edgelist`, `*.txt`)

One edge/arc per line, `u v` (optionally `u v w` for a weight, ignored).
Directedness, terminals, sizes/capacities and weights are passed on the CLI:

```
glsolve solve graph.edgelist --terminals 0,10,27,81 --sizes 25,25,25,25 [--directed] [--weights w.json]
```

Vertex ids in edge lists must be integers in `0..n-1` (n = 1 + max id unless
`--n` is given).
