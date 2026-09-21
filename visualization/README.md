# Visualization layer (`glsolver.viz`)

Step-by-step rendering of the polynomial-time Győri–Lovász algorithm from
the trace events of `docs/paper_notes.md §14`. The reference solvers
(`glref.unweighted`, `glref.weighted`, `glref.dag`) emit these events with
`glpartition(inst, algorithm="reference", trace=True).trace`; the C++
backends emit the same event types.

```python
from glsolver import glpartition
from glsolver.generators import paper_running_example
from glsolver.viz import TraceRenderer, render_html, animate_solution, partition_svg, visualize_counterexample

inst = paper_running_example()
res = glpartition(inst, algorithm="reference", trace=True)

r = TraceRenderer(inst, res.trace)          # or TraceRenderer(res)
r.save_frame(9, "step9.svg")                # svg / png / pdf by extension
r.render_all("frames/", fmt="png")          # one file per event
fig = r.frame(9)                            # matplotlib Figure
render_html(inst, res.trace, "paper.html")  # ONE self-contained file
animate_solution(res, path="paper.gif", fps=1)  # GIF (MP4 via imageio+ffmpeg, else GIF)
svg = partition_svg(inst, res.parts, width=400, parents=res.certificate["parents"])
visualize_counterexample(copies=1, path="counterexample.svg")
```

CLI (`glsolve visualize`):

```
glsolve visualize INSTANCE.json [--algorithm reference|reference-weighted|reference-dag|general|weighted|dag|auto]
                                [--format svg|png|pdf|html|gif|mp4] [--layout auto|grid|kamada|spring]
                                [--step I] [-o OUT]
```

`svg`/`png`/`pdf` without `--step` write every frame into a directory
(`OUT` or `<name>_viz/`); with `--step I` a single file. `html` writes one
file, `gif`/`mp4` an animation. Without `--algorithm` the instance's
`meta["viz_algorithm"]` is run (a note says so), else `reference`; a C++
backend that is not built is replaced by the corresponding reference solver
(a note is printed); `reference` on a weighted instance becomes
`reference-weighted`. SVG/PDF/PNG frames are byte-deterministic (no
timestamp, fixed `svg.hashsalt`), so they can be diffed across runs.

## Visual conventions (paper Figure 4, "Visual conventions")

| element | drawing |
|---|---|
| terminal `t_i` | rounded square, one colour per terminal (Okabe–Ito + Tol palette, 12 colours, then hatching); label `t1, t2, …` in the order of `inst.terminals` |
| non-terminal `v` | circle whose interior is split into sectors coloured by its **essential terminals** `Ess(v)` (`[Lem 4.1]`, from the latest `essential` event, see below); a thick outer ring in the colour of its **assigned terminal** `φ(v)` (witness); for the weighted algorithm the ring is split proportionally to the split witness `ψ(v,·)` |
| contracted / rounded vertex | stays in place, filled with its part's colour, dotted outline, smaller; its arborescence arc `(p, parent)` is drawn thick in the part's colour |
| removed terminal | faded, dashed outline (`part` column of the panel shows the finished size) |
| arcs | dark grey; arcs redirected by a contraction keep pointing at the vertex that absorbed them (which now belongs to the terminal's part); a pair of opposite arcs is drawn as two curved arcs |
| deleted arcs | light grey, dashed, remain visible; the arc deleted by the current event is orange with a ✕ |
| matching arcs `(p_i, t_i)` | blue, thick |
| secondary arcs `e_i = (p_i, q_i)` | red, thick, labelled `e_i` |
| ShiftAssignment context | matching arcs, secondary arcs, the criticality table and the reassignment inset are shown from the `matching` event up to **and including** the `delete_arc` frame that ends the call (so the reader can check that the deleted `e_i` is non-critical); the next event clears them |
| recomputed `Ess` (`essential` after a deletion / removal / rounding) | the vertices whose `Ess` or `κ` changed glow; the panel lists `Ess(v) {…}→{…}` and `κ(v) a→b` (or "unchanged") |
| reassignment graph `R` | inset: terminals as nodes on a circle, arc `φ(v_i) → t_i` labelled `v_i`, the chosen cycle in green |
| cycle shift | the changed vertices glow; the new ring colour is drawn and the previous colour lingers as a thin dashed outer ring; the panel shows `Φ` before → after |
| tightest min cut (events carrying `cuts`) | translucent hulls of `L` (blue), `S` (yellow), `R` (red); the HTML viewer has a selector for the vertex |
| side panel | step number, event type and a one-line explanation with the paper's lemma label, capacities and part sizes per terminal, live non-terminals/arcs, `φ`/`ψ`, potential, counters (contractions, deletions, cycle shifts, terminal removals, roundings, ShiftAssignment calls) |

Layout (`glsolver.viz.layout.compute_layout(inst, seed=0, method="auto")`)
is deterministic and computed **once per instance**, so every frame, the
HTML viewer and the animation use the same coordinates. Terminals sit on a
bottom row as in the paper's figures; `grid` puts the non-terminals on rows
by their directed distance to `T` (barycenter ordering), `kamada` runs
NetworkX `kamada_kawai_layout` on the undirected skeleton and pins the
terminals, `spring` is Fruchterman–Reingold with fixed terminals; `auto`
picks `grid` for small shallow instances (`n ≤ 24`, ≤ 5 rows) and `kamada`
otherwise.

## State replay

`glsolver.viz.render.replay(inst, trace)` returns one `StepState` per event
with the state *after* that event: current arcs (with draw endpoints),
deleted arcs, live vertices and terminals, parts, capacities, `φ`/`ψ`,
essential sets and `κ`, the matching/secondary arcs/criticality table of
the running ShiftAssignment call, the reassignment graph and cycle, the
changes and potential of a cycle shift, cuts, and counters. The potential
`Φ(φ) = Σ_v ξ_v(φ(v))` is computed from the `criticality` event when the
trace does not carry it (the reference records it only with `debug=True`).

The replay is driven by the trace alone — the viewer imports no solver
internals, so a C++ trace is drawn exactly as the C++ solver saw it. The
reference tracers emit an `essential` event after every recomputation of
`Ess`/`κ` (following `remove_terminal`, `delete_arc` and `round_and_remove`;
`docs/paper_notes.md §14`); the frame of the mutation already shows those
sets (the replay looks one event ahead) and the `essential` frame highlights
what changed. A trace from a backend that does not re-emit `essential` gets
its sectors drawn faded and the panel says "stale" from the mutation until
the next `essential` event. A trace whose `init` event belongs to another
instance (different `n` or terminals) or that names a non-terminal as a
terminal is rejected with `ValueError` rather than drawn in wrong colours.

## Outputs

* `TraceRenderer.frame(i)` → Matplotlib `Figure` (12×7 in, 100 dpi by default; Agg canvas, no pyplot state);
  `save_frame(i, path)` for `.svg`/`.png`/`.pdf`; `render_all(dir, fmt)`; `frame_image(i)` → `PIL.Image`.
* `render_html(inst, trace, path)` → one HTML file, no external resources: per-step state precomputed in
  Python and embedded as JSON; slider, prev/next/play buttons, event list (click to jump), keyboard
  `← → Home End Space`, hover tooltips with `Ess / κ / φ`.
* `animate_solution(inst_or_result, trace=None, path="out.gif", fps=1, algorithm="reference")` → GIF via
  Pillow (always); `.mp4` via `imageio` when `imageio-ffmpeg` is installed, otherwise a `RuntimeWarning`
  and a GIF next to the requested path (the returned path says which).
* `partition_svg(inst, parts, width=400)` → standalone SVG string of the final partition (used by the
  benchmark dashboard); `final_figure(inst, parts)` → the Matplotlib version.
* `visualize_counterexample(copies=1, path=...)` → abstracted figure of the official compact-connectivity
  counterexample (`docs/paper_notes.md §10`): the 9 terminals on a circle with the 36 bundles of 3
  pre-terminals on the chords, the paper's Figure 20 forcing gadget expanded in an inset with the targeted
  edge highlighted, and a text summary of the counts (also returned as a dict).

`examples/render_all.py` renders the curated examples of `examples/curated/`
(see `examples/README.md`).

## Dependencies

`matplotlib`, `pillow` (both in the `viz` extra), `networkx` (core
dependency) and optionally `imageio` + `imageio-ffmpeg` for MP4.
