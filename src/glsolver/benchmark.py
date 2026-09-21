"""Glue for ``glsolve benchmark``: locate the repository-level ``benchmarks`` package.

The benchmark runner lives in ``benchmarks/runner.py`` at the repository root
(it is not part of the installed wheel because it writes results into the
source tree).  :mod:`glsolver.cli` first tries ``benchmarks.runner`` and falls
back to this module, which finds the repository root of an editable install
and imports the runner from there.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "benchmarks" / "runner.py").exists():
            return parent
    return None


def run_from_cli(args: Any) -> int:
    """Delegate to :func:`benchmarks.runner.run_from_cli` (editable checkout required)."""
    try:
        from benchmarks.runner import run_from_cli as _run
    except ImportError:
        root = _repo_root()
        if root is None:
            raise SystemExit(
                "the benchmark runner needs a source checkout (benchmarks/runner.py); "
                "run `python -m benchmarks.runner` from the repository root"
            ) from None
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from benchmarks.runner import run_from_cli as _run
    return _run(args)


__all__ = ["run_from_cli"]
