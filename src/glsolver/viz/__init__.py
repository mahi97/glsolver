"""Visualization layer: step-by-step rendering of the algorithm from trace events.

* :mod:`glsolver.viz.layout` — deterministic layouts and the shared palette;
* :mod:`glsolver.viz.render` — :func:`replay` (state reconstruction),
  :class:`TraceRenderer` (Matplotlib frames: svg/png/pdf),
  :func:`partition_svg`, :func:`final_figure`;
* :mod:`glsolver.viz.html` — :func:`render_html` (one self-contained file);
* :mod:`glsolver.viz.animate` — :func:`animate_solution` (GIF, MP4 when ffmpeg is available);
* :mod:`glsolver.viz.counterexample` — :func:`visualize_counterexample`.

:func:`visualize_cli` backs ``glsolve visualize`` (see ``glsolver.cli``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from glsolver.api import glpartition
from glsolver.instance import Instance
from glsolver.viz.animate import animate_solution, resolve_algorithm, solve_with_trace
from glsolver.viz.counterexample import (
    counterexample_counts,
    counterexample_figure,
    counterexample_summary,
    visualize_counterexample,
)
from glsolver.viz.html import export_viewer_data, render_html
from glsolver.viz.layout import PALETTE, compute_layout, terminal_color
from glsolver.viz.render import (
    StepState,
    TraceRenderer,
    describe_event,
    final_figure,
    partition_svg,
    replay,
)

__all__ = [
    "PALETTE",
    "StepState",
    "TraceRenderer",
    "animate_solution",
    "compute_layout",
    "counterexample_counts",
    "counterexample_figure",
    "counterexample_summary",
    "describe_event",
    "export_viewer_data",
    "final_figure",
    "partition_svg",
    "render_html",
    "replay",
    "resolve_algorithm",
    "solve_with_trace",
    "terminal_color",
    "visualize_cli",
    "visualize_counterexample",
]

FORMATS = ("svg", "png", "pdf", "html", "gif", "mp4")


def visualize_cli(inst: Instance, args: Any) -> int:
    """Implementation of ``glsolve visualize INSTANCE [--algorithm A] [--format F] [--layout L] [--step I] [-o OUT]``.

    Without ``--algorithm`` the instance's ``meta["viz_algorithm"]`` (the
    backend the curated examples are documented with) is used, else
    ``reference``; ``resolve_algorithm`` then maps unavailable C++ cores and
    weighted instances to the matching reference solver.
    Runs the solver with ``trace=True`` and writes: for ``svg``/``png``/``pdf``
    every frame into a directory (``OUT`` or ``<name>_viz/``) or, with
    ``--step I``, the single file ``OUT`` (default ``<name>_step<I>.<fmt>``);
    for ``html`` one self-contained file; for ``gif``/``mp4`` an animation.
    Returns ``0`` on success, ``2`` when the solver did not finish with
    status ``ok`` (whatever trace exists is still rendered).
    """
    fmt = str(getattr(args, "format", "svg") or "svg").lower()
    if fmt not in FORMATS:
        raise SystemExit(f"unknown format {fmt!r}; choose from {FORMATS}")
    layout_method = str(getattr(args, "layout", "auto") or "auto")
    step = getattr(args, "step", None)
    output = getattr(args, "output", None)
    requested = getattr(args, "algorithm", None)
    if not requested:
        requested = str(inst.meta.get("viz_algorithm") or "reference")
        if inst.meta.get("viz_algorithm"):
            print(f"note: using the instance's meta['viz_algorithm'] = {requested!r} (pass --algorithm to override)")
    algo, note = resolve_algorithm(inst, str(requested))
    if note:
        print("note:", note)
    res = glpartition(inst, algorithm=algo, trace=True, seed=int(getattr(args, "seed", 0) or 0))
    print(res.summary())
    if res.status != "ok":
        print("message:", res.message)
    if not res.trace:
        print("no trace events were produced; nothing to render")
        return 2
    name = inst.name or Path(str(getattr(args, "instance", "instance"))).stem or "instance"
    renderer = TraceRenderer(inst, res.trace, layout_method=layout_method)
    if fmt in ("svg", "png", "pdf"):
        if step is None:
            outdir = Path(output) if output else Path(f"{name}_viz")
            paths = renderer.render_all(outdir, fmt)
            print(f"wrote {len(paths)} frames ({fmt}) to {outdir}/")
        else:
            if not 0 <= int(step) < len(renderer):
                raise SystemExit(f"--step must be in 0..{len(renderer) - 1} (trace has {len(renderer)} events)")
            out = Path(output) if output else Path(f"{name}_step{int(step):03d}.{fmt}")
            renderer.save_frame(int(step), out)
            print(f"wrote step {step} to {out}")
    elif fmt == "html":
        out = Path(output) if output else Path(f"{name}.html")
        render_html(inst, res.trace, out, renderer=renderer)
        print(f"wrote interactive viewer with {len(renderer)} steps to {out}")
    else:
        out = Path(output) if output else Path(f"{name}.{fmt}")
        written = animate_solution(inst, res.trace, out, fps=1, renderer=renderer)
        print(f"wrote animation ({len(renderer)} frames) to {written}")
    return 0 if res.status == "ok" else 2
