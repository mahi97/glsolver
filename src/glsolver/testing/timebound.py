"""Wall-clock bound around a single solver call (docs/verification.md §4).

A solver bug that makes a loop non-terminating would otherwise *hang* the
whole test-suite instead of failing it.  :func:`time_bound` arms a
``SIGALRM`` timer and raises :class:`SolverTimeout` in the main thread when
it expires; :func:`call_bounded` is the one-call convenience wrapper.

:class:`SolverTimeout` derives from ``BaseException`` **on purpose**:

* :func:`glsolver.api.glpartition` converts every ``Exception`` raised by a
  backend into ``status == "error"``, so an ``Exception``-based timeout would
  be misreported as a solver error instead of a hang;
* Hypothesis shrinks every ``Exception`` (and each shrink attempt of a
  hanging instance would cost the full limit again) but re-raises other
  ``BaseException`` subclasses immediately, and pytest reports such an
  exception as an ordinary failed test.

Nested bounds are supported: on exit the outer timer is re-armed with its
remaining time.

Limitations (documented in docs/verification.md): ``SIGALRM`` handlers can
only be installed in the main thread and on POSIX; elsewhere the block runs
unbounded (``bounded`` in :func:`time_bound` says so).  A Python signal
handler runs only when the interpreter regains control, so a hang *inside* a
C extension (the C++ core, HiGHS) is reported when that call returns.
"""
from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from typing import Any, Callable, Iterator

#: environment variable holding the per-call limit (seconds) used by the harness
TIME_LIMIT_ENV = "GL_SOLVER_TIME_LIMIT"
#: default per-call limit; the exhaustive instances take milliseconds, so this
#: is a generous margin even for loaded CI machines
DEFAULT_TIME_LIMIT = 20.0


class SolverTimeout(BaseException):
    """A solver call exceeded its wall-clock bound (a hang, not an error)."""

    def __init__(self, seconds: float, what: str = "solver call") -> None:
        super().__init__(f"{what} did not return within {seconds:g} s (hang)")
        self.seconds = seconds
        self.what = what


def default_time_limit() -> float:
    """Per-call limit from ``$GL_SOLVER_TIME_LIMIT`` (``0`` disables the bound)."""
    raw = os.environ.get(TIME_LIMIT_ENV, "")
    if raw.strip() == "":
        return DEFAULT_TIME_LIMIT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{TIME_LIMIT_ENV}={raw!r} is not a number of seconds") from exc
    return max(0.0, value)


def can_bound() -> bool:
    """Whether a ``SIGALRM`` bound can be armed here (POSIX main thread)."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def time_bound(seconds: float | None, what: str = "solver call") -> Iterator[bool]:
    """Raise :class:`SolverTimeout` if the block runs longer than ``seconds``.

    ``seconds`` ``None`` or ``<= 0`` disables the bound.  The value yielded is
    ``True`` when a bound is actually armed (see :func:`can_bound`).
    """
    if not seconds or seconds <= 0 or not can_bound():
        yield False
        return
    state = {"armed": True}

    def handler(_signum: int, _frame: Any) -> None:
        if state["armed"]:
            raise SolverTimeout(seconds, what)

    previous_handler = signal.signal(signal.SIGALRM, handler)
    outer_remaining, _interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield True
    finally:
        state["armed"] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        for _ in range(2):  # let a trip that slipped in just before the cancel reach our (inert) handler
            pass
        signal.signal(signal.SIGALRM, previous_handler)
        if outer_remaining > 0:  # nested inside another bound: re-arm it with what is left
            left = outer_remaining - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(left, 1e-3))


def call_bounded(fn: Callable[..., Any], *args: Any, seconds: float | None,
                 what: str = "solver call", **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` under :func:`time_bound`."""
    with time_bound(seconds, what):
        return fn(*args, **kwargs)


__all__ = [
    "DEFAULT_TIME_LIMIT",
    "TIME_LIMIT_ENV",
    "SolverTimeout",
    "call_bounded",
    "can_bound",
    "default_time_limit",
    "time_bound",
]
