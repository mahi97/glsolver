"""Regression store (``regression/*.json``, docs/verification.md).

Every instance stored with :func:`glsolver.testing.regression.save_regression`
is re-solved by every applicable backend.  The ``regression`` block of a file
may carry ``"expect"``: ``"ok"`` (default; every theorem solver must return a
valid partition) or ``"precondition_failed"`` (the theorem solvers must
report that status and never ``"infeasible"`` or a crash; the oracles may
answer anything but ``"error"``).

The store is seeded with three instances: the paper's running example, the
§13.3 terminal-removal example of docs/paper_notes.md and one random
FEAC-only digraph.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from glsolver.api import GLResult
from glsolver.instance import Instance
from glsolver.preconditions import check_preconditions
from glsolver.testing.registry import solvers_for
from glsolver.testing.regression import REGRESSION_DIR, instance_hash, load_all
from glsolver.verify import verify_instance_parts

ORACLES = ("bruteforce", "ilp")
ORACLE_MAX_N = 14
SEEDS = ("paper_running_example", "terminal_removal_13_3", "feac_only_random_seed1337")

_ENTRIES: list[tuple[Path, Instance, dict[str, Any]]] = list(load_all())


def _cases() -> list[Any]:
    out = []
    for path, inst, meta in _ENTRIES:
        for name in solvers_for(inst, include_oracles=True):
            marks = []
            if name in ORACLES and inst.n > ORACLE_MAX_N:
                marks.append(pytest.mark.skip(reason=f"{name}: n={inst.n} > {ORACLE_MAX_N}"))
            if name.startswith("reference") and inst.n > 300:
                marks.append(pytest.mark.slow)
            out.append(pytest.param(path, inst, meta, name, id=f"{path.stem}-{name}", marks=marks))
    return out


def _kw(name: str) -> dict[str, Any]:
    if name in ORACLES:
        return {"verify_preconditions": False, "time_limit": 120.0}
    if name.startswith("reference"):
        return {"debug": True}
    return {}


def test_store_is_seeded() -> None:
    names = {path.stem for path, _inst, _meta in _ENTRIES}
    assert set(SEEDS) <= names, f"missing seed instances: {set(SEEDS) - names}"
    for path, inst, meta in _ENTRIES:
        assert meta.get("reason"), f"{path.name}: regression files must carry a reason"
        assert meta.get("hash") == instance_hash(inst), f"{path.name}: stored hash does not match the instance"
        inst.validate()


def test_seed_instances_are_feac_only() -> None:
    by_name = {path.stem: inst for path, inst, _meta in _ENTRIES}
    for name in SEEDS:
        pre = check_preconditions(by_name[name], True)
        assert pre["feac"] is True and pre["k_T_connected"] is False, (name, pre["message"])


@pytest.mark.skipif(not _ENTRIES, reason=f"no regression instances under {REGRESSION_DIR}")
@pytest.mark.parametrize("path,inst,meta,name", _cases())
def test_regression_instance(path: Path, inst: Instance, meta: dict[str, Any], name: str, solvers) -> None:
    expect = meta.get("expect", "ok")
    res: GLResult = solvers[name](inst, **_kw(name))
    assert res.status != "error", f"{path.name} / {name}: crash: {res.message}"
    if expect == "ok":
        assert res.status == "ok", f"{path.name} / {name}: {res.status}: {res.message}"
        assert res.valid is True, f"{path.name} / {name}: {res.verification.errors if res.verification else ''}"
        assert verify_instance_parts(inst, res.parts).valid
        if inst.weights is None:
            assert [len(p) for p in res.parts] == [c + 1 for c in inst.capacities]
    elif expect == "precondition_failed":
        if name in ORACLES:
            assert res.status in ("ok", "infeasible", "timeout")
        else:
            assert res.status == "precondition_failed", f"{path.name} / {name}: {res.status}: {res.message}"
    else:
        pytest.fail(f"{path.name}: unknown expectation {expect!r}")
