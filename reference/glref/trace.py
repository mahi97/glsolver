"""Trace recording for visualization and invariant debugging (paper_notes §14)."""
from __future__ import annotations

import json
from typing import Any


class Tracer:
    """Collects JSON-serializable events. ``Tracer(enabled=False)`` is a no-op."""

    def __init__(self, enabled: bool = True, record_cuts: bool = False) -> None:
        self.enabled = enabled
        self.record_cuts = record_cuts
        self.events: list[dict[str, Any]] = []

    def record(self, type_: str, **fields: Any) -> None:
        if not self.enabled:
            return
        ev: dict[str, Any] = {"type": type_}
        ev.update(fields)
        self.events.append(_jsonable(ev))

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.events, f, indent=1)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x
