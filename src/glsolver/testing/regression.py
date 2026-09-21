"""Permanent store of regression instances under ``regression/``."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from glsolver.instance import Instance
from glsolver.io import instance_from_dict, instance_to_dict

REGRESSION_DIR = Path(__file__).resolve().parents[3] / "regression"


def instance_hash(inst: Instance) -> str:
    payload = json.dumps(instance_to_dict(inst), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def save_regression(inst: Instance, reason: str, name: str | None = None, extra: dict[str, Any] | None = None,
                    directory: Path | None = None) -> Path:
    """Write the instance (plus provenance) to ``regression/<name>.json``; idempotent per hash."""
    directory = directory or REGRESSION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    h = instance_hash(inst)
    name = name or f"{inst.meta.get('family', 'case')}_n{inst.n}_k{inst.k}_{h}"
    path = directory / f"{name}.json"
    d = instance_to_dict(inst)
    d["regression"] = {
        "reason": reason,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hash": h,
        **(extra or {}),
    }
    path.write_text(json.dumps(d, indent=1, sort_keys=True))
    return path


def load_all(directory: Path | None = None) -> Iterator[tuple[Path, Instance, dict[str, Any]]]:
    directory = directory or REGRESSION_DIR
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        d = json.loads(path.read_text())
        meta = d.pop("regression", {})
        yield path, instance_from_dict(d), meta
