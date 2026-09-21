# Curated examples

Small instances (`n ≤ 13`) whose traces show every operation of the
polynomial-time Győri–Lovász algorithm (arXiv 2608.30945) in readable frames.
Each file in `curated/` is a canonical JSON instance (`docs/instance_format.md`)
whose `meta` carries

* `viz_algorithm` — the trace-emitting backend that `render_all.py` runs
  (`reference`, `reference-weighted` or `reference-dag`);
* `description` — what the example illustrates;
* `expected` — properties of the reference trace that `render_all.py` and
  `tests/test_viz.py` assert (event count, number of cycle shifts, contractions,
  deletions, terminal removals, roundings, …).

Event counts include one `essential` event after every arc deletion, terminal
removal and rounding (the tracer re-emits `Ess`/`κ` whenever the solver
recomputes them, `docs/paper_notes.md §14`).

| file | n / k / arcs | events | what it shows |
|---|---|---|---|
| `paper_running_example.json` | 9 / 3 / 12 | 30 | the paper's running example (Fig. 4): matching + secondary arcs, **2 cycle shifts**, 3 deletions, 6 contractions, 2 terminal removals |
| `paper_essential_example.json` | 13 / 4 / 20 | 37 | the paper's Fig. 6: non-uniform essential sets (`Ess(v11) = {t1,t2}`, `Ess(v13) = T`, …), `c = (2,3,2,2)`; 1 cycle shift |
| `paper_contract_counterexample.json` | 9 / 3 / 18 (undirected) | 38 | the paper's Fig. 1: 3-*T*-connected but every contraction breaks it — 6 deletions must precede the 6 contractions; 0 shifts |
| `multi_cycle_shift.json` | 10 / 3 / 23 (undirected) | 65 | Harary `H_{3,10}` + 1 chord (`sparse_k_connected`, seed 0): ShiftAssignment performs **3 cycle shifts** (found by seeded search over Harary/random *k*-connected graphs, `n ≤ 14`) |
| `zero_capacity_removal.json` | 10 / 3 / 19 | 35 | the `paper_notes.md §13.3` gadget (`3→t1, 3→4, 4→t2, 4→t3`) embedded in a random digraph with `c(t2) = 1`: vertex 5 is contracted into `t2`, `t2` reaches capacity 0 and is **removed at step 4**, and **`t3` becomes a new essential terminal of vertex 3** (`Ess(3): {t1} → {t1,t3}`) |
| `weighted_rounding.json` | 8 / 3 / 15 (undirected, weights ≤ 2) | 44 | weighted variant of `H_{3,8}` (seed 0): min-cost split assignments `ψ` (proportional rings) and a final **RoundAndRemove** step (`roundings = 1`) |
| `small_dag.json` | 10 / 3 / 21 (DAG) | 9 | random 3-*T*-connected DAG (seed 1): `dag_contract` events only, no flows |

## Rendering them

```bash
cd /path/to/gs && source .venv/bin/activate
python examples/render_all.py                       # all examples → examples/output/<name>/
python examples/render_all.py --only small_dag --formats svg,html
python examples/render_all.py --fps 2 --out /tmp/gl_examples
```

For every example this writes `frames_svg/step_XXX.svg` (one frame per
trace event), `<name>.html` (the self-contained interactive viewer),
`<name>.gif` (animation), `final_partition.svg` (standalone partition, the
same function the benchmark dashboard uses) and `trace.json`, then prints a
summary table and writes `summary.json`; it exits with status 1 if a trace
violates the `expected` properties. `examples/output/` is generated, git-ignored
(`examples/.gitignore`) and can be deleted at any time.

The same outputs are available for any instance through the CLI:

```bash
glsolve visualize examples/curated/paper_running_example.json --format html -o paper.html
glsolve visualize examples/curated/paper_running_example.json --format svg --step 9 -o step9.svg
glsolve visualize examples/curated/weighted_rounding.json --format gif   # meta viz_algorithm = reference-weighted
```

Without `--algorithm` the CLI runs the instance's `meta["viz_algorithm"]`
(falling back to `reference`), i.e. the same backend as `render_all.py`.

See `visualization/README.md` for the drawing conventions and the Python API.

## How the searched examples were found

`multi_cycle_shift` and `weighted_rounding` come from a deterministic seeded
search (seeds 0–59 / 0–79, `n ≤ 14`, families `harary_graph`,
`sparse_k_connected`, `erdos_renyi_graph`, `random_regular_graph` and their
`weighted_variant`s) keeping the smallest instance with
`stats["cycle_shifts"] ≥ 3` resp. `stats["roundings"] ≥ 1` and all capacities
positive. `zero_capacity_removal` was found by embedding the §13.3 gadget in
random digraphs (seed 139 of a `random.Random(seed)` construction) and keeping
the first one where the replayed essential sets grow at the `remove_terminal`
event. All three are stored verbatim, so no search runs at test time.
