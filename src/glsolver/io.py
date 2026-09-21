"""Instance and solution I/O following docs/instance_format.md.

Two formats are supported:

* the canonical JSON schema (``instance_to_dict`` / ``instance_from_dict``,
  ``save_instance`` / ``load_instance``): keys ``name``, ``directed``, ``n``,
  ``edges``, ``terminals``, ``capacities`` (or ``sizes``), ``weights`` and
  ``meta``; undirected instances store the undirected edge list, directed
  instances store the arc list;
* plain edge lists (``load_edgelist``): one ``u v`` (or ``u v w``, weight
  ignored) per line, blank lines and ``#`` comments skipped.

Solutions are plain dicts (``save_solution`` / ``load_solution``);
``result_to_dict`` turns any duck-typed result object into one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from glsolver.instance import Instance, make_instance

__all__ = [
    "instance_to_dict",
    "instance_from_dict",
    "load_instance",
    "save_instance",
    "load_edgelist",
    "save_solution",
    "load_solution",
    "result_to_dict",
]

_JSON_INDENT = 1


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback encoder: numpy scalars/arrays, sets, tuples, paths."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, tuple):
        return list(obj)
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy arrays and scalars
        return tolist()
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _json_key(key: Any) -> str:
    """Dict keys the way ``json.dumps`` would spell them (numpy ints included)."""
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    item = getattr(key, "item", None)
    if callable(item):  # numpy scalar
        key = item()
    return str(key)


def _jsonable(obj: Any) -> Any:
    """Deep-convert ``obj`` into plain JSON types (dicts get string keys)."""
    if isinstance(obj, dict):
        return {_json_key(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    return _jsonable(_json_default(obj))


def _dump(data: Any, path: str | Path) -> None:
    text = json.dumps(data, indent=_JSON_INDENT, sort_keys=True, default=_json_default)
    Path(path).write_text(text + "\n", encoding="utf-8")


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    """Serialize an :class:`Instance` into the canonical JSON dict.

    Undirected instances write their undirected edge list (so edges between two
    terminals survive the round trip); directed instances write the normalized
    arc list.  ``capacities`` is always used (never ``sizes``).
    """
    if inst.directed:
        edges = [[int(u), int(v)] for u, v in inst.arcs]
    elif inst.undirected_edges is not None:
        edges = [[int(u), int(v)] for u, v in inst.undirected_edges]
    else:  # hand-built undirected instance: recover edges from the symmetric arcs
        edges = [list(e) for e in sorted({(min(u, v), max(u, v)) for u, v in inst.arcs})]
    return {
        "name": str(inst.name),
        "directed": bool(inst.directed),
        "n": int(inst.n),
        "edges": edges,
        "terminals": [int(t) for t in inst.terminals],
        "capacities": [int(c) for c in inst.capacities],
        "weights": None if inst.weights is None else [int(w) for w in inst.weights],
        "meta": _jsonable(dict(inst.meta)),
    }


def instance_from_dict(d: dict[str, Any]) -> Instance:
    """Build a normalized :class:`Instance` from a canonical JSON dict.

    Exactly one of ``capacities`` / ``sizes`` must be present; ``weights`` may
    be missing or ``null``; ``name`` and ``meta`` are optional; ``directed``
    defaults to ``False`` (classical undirected GL) when absent.
    """
    if not isinstance(d, dict):
        raise TypeError("instance_from_dict expects a dict")
    missing = [key for key in ("n", "edges", "terminals") if key not in d]
    if missing:
        raise ValueError(f"instance dict is missing required keys: {missing}")
    capacities = d.get("capacities")
    sizes = d.get("sizes")
    if (capacities is None) == (sizes is None):
        raise ValueError("instance dict must contain exactly one of 'capacities' or 'sizes'")
    name = d.get("name") or ""
    meta = d.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("'meta' must be a JSON object")
    return make_instance(
        int(d["n"]),
        d["edges"],
        d["terminals"],
        capacities,
        sizes=sizes,
        weights=d.get("weights"),
        directed=bool(d.get("directed", False)),
        name=str(name),
        meta=dict(meta),
    )


def save_instance(inst: Instance, path: str | Path) -> None:
    """Write ``inst`` as JSON (indent 1, sorted keys, trailing newline)."""
    _dump(instance_to_dict(inst), path)


def load_instance(path: str | Path) -> Instance:
    """Read a canonical JSON instance file."""
    return instance_from_dict(_load(path))


def load_edgelist(
    path: str | Path,
    terminals: Sequence[int],
    capacities: Sequence[int] | None = None,
    *,
    sizes: Sequence[int] | None = None,
    directed: bool = False,
    n: int | None = None,
    weights: Sequence[int] | None = None,
    name: str = "",
) -> Instance:
    """Load a plain edge list (``u v`` or ``u v w`` per line; the third field is
    ignored; blank lines and lines starting with ``#`` are skipped; fields may
    be separated by whitespace or commas).

    ``n`` defaults to ``1 + max id`` over the edges and the terminals.  The
    instance name defaults to the file stem.
    """
    edges: list[tuple[int, int]] = []
    max_id = -1
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.replace(",", " ").split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{lineno}: expected 'u v' or 'u v w', got {line.rstrip()!r}")
            u, v = int(fields[0]), int(fields[1])
            if u < 0 or v < 0:
                raise ValueError(f"{path}:{lineno}: negative vertex id in {line.rstrip()!r}")
            edges.append((u, v))
            max_id = max(max_id, u, v)
    if n is None:
        n = 1 + max(max_id, max((int(t) for t in terminals), default=-1))
    return make_instance(
        int(n),
        edges,
        [int(t) for t in terminals],
        capacities,
        sizes=sizes,
        weights=weights,
        directed=bool(directed),
        name=name or Path(path).stem,
    )


# ---------------------------------------------------------------------------
# solutions
# ---------------------------------------------------------------------------


def save_solution(path: str | Path, solution: dict[str, Any]) -> None:
    """Write a solution dict (docs/instance_format.md) as JSON."""
    if not isinstance(solution, dict):
        raise TypeError("save_solution expects a dict; use result_to_dict() first")
    _dump(solution, path)


def load_solution(path: str | Path) -> dict[str, Any]:
    """Read a solution dict written by :func:`save_solution`."""
    data = _load(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: solution file must contain a JSON object")
    return data


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a duck-typed solver result into the solution dict of
    docs/instance_format.md.

    ``result`` may be any object exposing (some of) the attributes ``parts``,
    ``assignment``, ``algorithm``, ``status``, ``valid``, ``runtime``,
    ``stats``, ``certificate``, ``message`` and ``instance``; missing
    attributes become ``None`` (or ``{}``/``""``).  ``instance`` may be an
    :class:`Instance` (its ``name`` is stored) or a string.  Certificate keys
    (``parents``, ``witness`` ...) are converted to strings for JSON.
    """
    inst = getattr(result, "instance", None)
    if isinstance(inst, str):
        instance_name = inst
    elif inst is None:
        instance_name = ""
    else:
        instance_name = str(getattr(inst, "name", "") or "")

    parts = getattr(result, "parts", None)
    assignment = getattr(result, "assignment", None)
    runtime = getattr(result, "runtime", None)
    stats = getattr(result, "stats", None)
    certificate = getattr(result, "certificate", None)
    valid = getattr(result, "valid", None)
    return {
        "instance": instance_name,
        "algorithm": _opt_str(getattr(result, "algorithm", None)),
        "status": _opt_str(getattr(result, "status", None)),
        "message": str(getattr(result, "message", "") or ""),
        "parts": None if parts is None else [[int(v) for v in part] for part in parts],
        "assignment": None if assignment is None else [int(i) for i in assignment],
        "valid": None if valid is None else bool(valid),
        "runtime": None if runtime is None else float(runtime),
        "stats": _jsonable(dict(stats)) if stats else {},
        "certificate": _jsonable(dict(certificate)) if certificate else {},
    }


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)
