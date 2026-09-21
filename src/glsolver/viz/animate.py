"""Animated GIF / MP4 of a solver trace.

:func:`animate_solution` renders every frame of a trace with
:class:`glsolver.viz.render.TraceRenderer` and writes a GIF with Pillow
(always available) or an MP4 through ``imageio`` when its ffmpeg plugin
(``imageio-ffmpeg``) is installed; without it the function falls back to a
GIF next to the requested path and says so with a :class:`RuntimeWarning`.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from glsolver.api import CORE_ALGORITHMS, GLResult, choose_algorithm, core_available, glpartition
from glsolver.instance import Instance
from glsolver.viz.render import TraceRenderer

_CORE_TO_REFERENCE = {"general": "reference", "weighted": "reference-weighted", "dag": "reference-dag"}


def resolve_algorithm(inst: Instance, algorithm: str = "reference") -> tuple[str, str | None]:
    """Pick a trace-emitting backend for ``inst``.

    Returns ``(name, note)``: ``"auto"`` follows :func:`glsolver.api.choose_algorithm`;
    a C++ core name is mapped to the pure Python reference when the core is
    not built; ``"reference"`` on a weighted instance becomes
    ``"reference-weighted"``.  ``note`` explains a substitution (or is ``None``).
    """
    note = None
    algo = algorithm
    if algo == "auto":
        algo = choose_algorithm(inst)
    if algo in CORE_ALGORITHMS and not core_available():
        algo, note = _CORE_TO_REFERENCE[algo], f"C++ core not built: using {_CORE_TO_REFERENCE[algo]!r} instead of {algorithm!r}"
    if algo in ("reference", "general") and inst.is_weighted:
        algo, note = "reference-weighted", f"{algorithm!r} is unweighted; using 'reference-weighted' for the weighted instance"
    if algo in ("bruteforce", "ilp"):
        raise ValueError("the oracles do not emit traces; use a reference/core algorithm for visualization")
    return algo, note


def solve_with_trace(inst: Instance, algorithm: str = "reference", **kw: Any) -> GLResult:
    """Run ``glpartition`` with ``trace=True`` on a trace-emitting backend."""
    algo, _note = resolve_algorithm(inst, algorithm)
    return glpartition(inst, algorithm=algo, trace=True, **kw)


def animate_solution(
    inst: Any,
    trace: Any = None,
    path: str | Path = "out.gif",
    fps: float = 1.0,
    algorithm: str = "reference",
    *,
    renderer: TraceRenderer | None = None,
    hold_last: int = 2,
    colors: int = 128,
    **renderer_kw: Any,
) -> Path:
    """Render the trace of ``inst`` as an animation and return the written path.

    ``inst`` may be an :class:`Instance` (``trace`` is the event list, or
    ``None`` to run ``algorithm`` with ``trace=True`` first) or a
    :class:`GLResult` (its instance and trace are used).  ``path`` ending in
    ``.mp4`` is written with imageio/ffmpeg when available, otherwise a GIF
    is written to ``path.with_suffix('.gif')`` and a ``RuntimeWarning`` is
    issued.  ``fps`` frames per second; the last frame is shown ``hold_last``
    times longer (GIF) or repeated ``hold_last`` times (MP4).  Extra keyword arguments configure the
    :class:`TraceRenderer` (``layout``, ``layout_method``, ``figsize`` ...); combining them with an
    explicit ``renderer`` is a ``TypeError``.
    """
    if renderer is not None and renderer_kw:
        raise TypeError(f"animate_solution: renderer options {sorted(renderer_kw)} cannot be combined with an explicit renderer")
    if isinstance(inst, GLResult):
        if trace is None:
            trace = inst.trace
        inst = inst.instance
    if isinstance(trace, GLResult):
        trace = trace.trace
    if trace is None:
        res = solve_with_trace(inst, algorithm)
        if not res.trace:
            raise RuntimeError(f"backend {res.algorithm!r} produced no trace (status {res.status}: {res.message})")
        trace = res.trace
    renderer = renderer or TraceRenderer(inst, trace, **renderer_kw)
    frames = [renderer.frame_image(i) for i in range(len(renderer))]
    hold_last = max(1, int(hold_last))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        if _write_mp4(frames + [frames[-1]] * (hold_last - 1), path, fps):
            return path
        gif = path.with_suffix(".gif")
        warnings.warn(
            f"MP4 export needs imageio with ffmpeg (pip install imageio-ffmpeg); wrote {gif} instead",
            RuntimeWarning, stacklevel=2,
        )
        path = gif
    _write_gif(frames, path, fps, colors, hold_last)
    return path


def _write_gif(frames: list[Any], path: Path, fps: float, colors: int, hold_last: int = 1) -> None:
    """One GIF frame per trace event; the last frame lasts ``hold_last`` times longer."""
    from PIL import Image

    duration = max(20, int(round(1000.0 / max(fps, 1e-6))))
    durations = [duration] * (len(frames) - 1) + [duration * hold_last]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    pal[0].save(
        str(path), save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False,
        disposal=2,
    )


def _write_mp4(frames: list[Any], path: Path, fps: float) -> bool:
    """Try imageio + ffmpeg; return ``False`` (without writing) when unavailable."""
    try:
        import imageio.v2 as imageio  # type: ignore
        import numpy as np
    except Exception:
        return False
    try:
        writer = imageio.get_writer(str(path), fps=max(fps, 0.01), format="FFMPEG", codec="libx264", quality=8,
                                    macro_block_size=None)
    except Exception:
        return False
    try:
        with writer:
            for f in frames:
                writer.append_data(np.asarray(f.convert("RGB")))
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True
