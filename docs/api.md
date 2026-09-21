# Python API contract

```python
from glsolver import partition, glpartition, verify_partition, Instance, make_instance

result = glpartition(
    graph,                      # networkx Graph/DiGraph | Instance | (n, edges)
    terminals,                  # list of terminals (node labels for networkx)
    capacities=None,            # paper convention c_i (non-terminal count / weight)
    sizes=None,                 # classical convention n_i (unweighted only); exactly one of capacities/sizes
    weights=None,               # None | sequence | dict for weighted instances
    directed=None,              # None → inferred from the graph type
    algorithm="auto",           # "auto" | "general" | "weighted" | "dag" | "reference" | "reference-weighted" | "bruteforce" | "ilp"
    verify=True,                # run the independent verifier on the output
    verify_preconditions="auto",# False | True | "auto": check k-T-connectivity / FEAC before solving
    trace=False,                # record the step-by-step trace for visualization
    seed=0,                     # tie-breaking seed for heuristic choices (determinism)
    time_limit=None,            # seconds, only honoured by oracles
)

partition(G, terminals, sizes=..., **kw)   # thin alias: classical undirected GL
```

`GLResult` fields:

| field | meaning |
|---|---|
| `parts` | `list[list[int]]`, `parts[i]` contains `terminals[i]` (internal ids; `result.parts_labels` maps back to networkx labels) |
| `assignment` | `list[int]`, part index per vertex |
| `algorithm` | name of the algorithm actually run |
| `status` | `"ok"`, `"precondition_failed"`, `"infeasible"` (oracles only), `"timeout"`, `"error"` |
| `message` | human-readable explanation, esp. for precondition failures |
| `valid` | `True/False` from the independent verifier, `None` if `verify=False` |
| `verification` | full `VerificationReport` |
| `runtime` | wall-clock seconds of the solve (excluding verification) |
| `stats` | dict of counters/timings: `max_flow_calls`, `min_cut_calls`, `matching_calls`, `min_cost_flow_calls`, `contractions`, `deletions`, `cycle_shifts`, `terminal_removals`, `roundings`, `assignment_repairs`, `time_*`, `peak_rss_mb`, `graph_size_over_time`; reference solvers also report `debug_max_flow_calls`, `debug_min_cost_flow_calls`, `time_debug_checks` (work done only by `debug=True` invariant checks, kept separate so counters are comparable across modes) |
| `certificate` | `{"parents": {v: parent_v}}` in-arborescence per part (original arcs), plus `"witness"` for the final witness when available |
| `trace` | list of trace events (docs/paper_notes.md §14) or `None` |
| `instance` | the normalized `Instance` |

Precondition semantics (docs/paper_notes.md §13.9):

* `status == "precondition_failed"` means *the theorem's guarantee does not
  apply* (FEAC/`k`-`T`-connectivity not satisfied); a partition may still exist.
  The message names the violating vertex / Hall-deficient set.
* The solver never claims that no partition exists; only the oracles
  (`bruteforce`, `ilp`) can return `"infeasible"`.

Independent verifier:

```python
report = verify_partition(instance_or_graph, terminals, capacities_or_sizes, parts,
                          weights=None, directed=None)
report.valid        # bool
report.errors       # list[str], empty iff valid
report.details      # per-part sizes/weights/connectivity, bound used
```

The verifier is pure Python, uses only BFS on the *original* input, and imports
nothing from the solver modules.
