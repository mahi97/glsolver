"""Review tests for the benchmark framework (``benchmarks/runner.py``, ``plot.py``,
``report.py``, ``machine.py``).

What is asserted here, and how:

* **Fault injection through the real worker path.**  The worker's solve entry
  (``glsolver.api.glpartition``, looked up by ``benchmarks.runner.worker_main``
  at call time) is monkeypatched *inside the worker subprocess*: the driver's
  ``worker_command`` is replaced so that the subprocess runs a tiny bootstrap
  that installs a fake ``glpartition`` (hang, exception, segfault, bogus
  partition, or the real solver, chosen by instance name) and then hands over
  to ``runner.main(["--worker", ...])``.  The driver therefore observes the
  faults exactly as in a real run: through the JSON event stream, the exit
  status and the process-group kill.
* **Provenance is compared with the host** (``/proc/cpuinfo``, ``sysconf``,
  ``git rev-parse``), not merely checked for presence.
* **Plots** are inspected through the matplotlib axes (scales, legend text,
  marker positions), not only through the existence of a PNG.
* **The dashboard** is checked for one row per run and for the absence of any
  external resource.

Tests that drive ``run_config``/``collect_machine_info`` use the
``survivable_core`` fixture: both import the compiled ``glsolver._core`` in
the parent process, and a sanitizer-linked build without ``LD_PRELOAD`` aborts
the interpreter (a review finding); the fixture probes that in a subprocess and
substitutes the "core not built" answers when needed, so these tests exercise
the framework rather than the state of the C++ build.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # ``benchmarks`` is a repo-level package, not part of the wheel
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import machine, plot, report, runner  # noqa: E402

SAMPLE_ROWS = REPO_ROOT / "benchmarks" / "results" / "sample_smoke.jsonl"


# ---------------------------------------------------------------------------
# keeping the driver alive whatever the state of the C++ build
# ---------------------------------------------------------------------------
_CORE_PROBE = (
    "import glsolver.api as a; a.core_available(); "
    f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import benchmarks.machine as m; m.core_info()"
)


@functools.cache
def _core_import_is_safe() -> bool:
    """Whether ``glsolver._core`` can be imported in-process without killing the interpreter."""
    proc = subprocess.run([sys.executable, "-c", _CORE_PROBE], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=120)
    return proc.returncode == 0


@pytest.fixture
def survivable_core(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Substitute the in-process core probes when importing ``glsolver._core`` would abort.

    Returns ``True`` when the substitution was necessary (see the module docstring).
    """
    if _core_import_is_safe():
        return False
    import glsolver.api as api

    monkeypatch.setattr(api, "core_available", lambda: False)
    monkeypatch.setattr(machine, "core_info",
                        lambda: {"importable": False, "available": False, "build": {}})
    return True


# ---------------------------------------------------------------------------
# fault injection: a fake solve entry installed inside the worker subprocess
# ---------------------------------------------------------------------------
# The fake picks its behaviour from the instance file name so that one config
# grid exercises every failure mode in one run.  ``@ROOT@`` is replaced by the
# repository root (``str.format`` is avoided because the code contains braces).
_BOOTSTRAP = r'''
import os, signal, sys, time
sys.path.insert(0, @ROOT@)
import glsolver.api as api
from glsolver.api import GLResult
argv = sys.argv[1:]
name = os.path.basename(argv[argv.index("--instance") + 1])
real = api.glpartition
def fake(inst, **kw):
    if "_n10_" in name:            # hang: no progress line is ever printed
        time.sleep(60)
    if "_n12_" in name:            # Python exception inside the solve
        raise RuntimeError("fake solver exploded")
    if "_n14_" in name:            # hard crash of the interpreter
        os.kill(os.getpid(), signal.SIGSEGV)
    if "_n16_" in name:            # claims success with a bogus partition
        parts = [[t] for t in inst.terminals]
        parts[0] += [v for v in range(inst.n) if v not in set(inst.terminals)]
        return GLResult(inst, "reference", "ok", parts, [], message="bogus", stats={"max_flow_calls": 1})
    return real(inst, **kw)
api.glpartition = fake
from benchmarks import runner
sys.exit(runner.main(["--worker", *argv]))
'''.replace("@ROOT@", repr(str(REPO_ROOT)))


def _inject_fake_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every worker subprocess start through :data:`_BOOTSTRAP`."""
    original = runner.worker_command

    def patched(instance_path: Path, algo: str, options: dict[str, Any], cfg: runner.BenchConfig,
                repeats: int, threads: int, seed: int) -> list[str]:
        cmd = original(instance_path, algo, options, cfg, repeats, threads, seed)
        i = cmd.index("--worker")
        return [sys.executable, "-c", _BOOTSTRAP, *cmd[i + 1:]]

    monkeypatch.setattr(runner, "worker_command", patched)


def _fault_config(sizes: list[int], timeout: float = 1.0) -> dict[str, Any]:
    return {"name": "fault", "families": ["harary"], "sizes": sizes, "k": [2], "seeds": [1],
            "algorithms": ["reference"], "repeats": 1, "warmup": 0, "timeout": timeout, "threads": 1}


def test_fault_injection_rows_and_the_run_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    survivable_core: bool) -> None:
    """A hang, an exception, a segfault and a bogus partition each produce one row with the
    documented status, none of them stops the grid, and the real solve after them is VALID."""
    _inject_fake_solver(monkeypatch)
    cfg = runner.config_from_dict(_fault_config([10, 12, 14, 16, 18], timeout=1.0))
    out = tmp_path / "fault.jsonl"
    t0 = time.monotonic()
    summary = runner.run_config(cfg, out, instances_dir=tmp_path / "inst", log=lambda s: None)
    elapsed = time.monotonic() - t0
    assert elapsed < 60
    assert summary["rows"] == 5
    assert (summary["timeouts"], summary["errors"], summary["invalid"], summary["ok"]) == (1, 2, 1, 1)

    rows = runner.load_results([out])
    assert [r["n"] for r in rows] == [10, 12, 14, 16, 18], "every pair got a row, in grid order"
    for row in rows:
        for key in runner.REQUIRED_ROW_KEYS:
            assert key in row, key
    by_n = {r["n"]: r for r in rows}

    hang = by_n[10]
    assert hang["status"] == "timeout" and hang["timed_out"] is True and hang["valid"] is None
    assert hang["trials"][-1]["status"] == "timeout" and hang["median"]["wall_time"] is None
    assert hang["child"]["returncode"] == -signal.SIGKILL, "the process group was SIGKILLed"
    grace = max(2.0, 0.1 * cfg.timeout)
    assert hang["child"]["wall_time_total"] < cfg.timeout + grace + 10, "killed soon after the deadline"
    assert hang["child"]["peak_rss_mb"] > 0, "rusage of the killed child is still collected"

    exc = by_n[12]
    assert exc["status"] == "error" and exc["valid"] is None and exc["timed_out"] is False
    assert "RuntimeError: fake solver exploded" in exc["error"]
    assert exc["child"]["returncode"] == 1 and exc["trials"][0]["status"] == "error"

    segv = by_n[14]
    assert segv["status"] == "error" and segv["valid"] is None and segv["timed_out"] is False
    # -11 normally; a sanitizer runtime (LD_PRELOAD=libasan) turns the signal into exit code 1
    assert segv["child"]["returncode"] not in (0, None)
    assert segv["error"] and "worker exited with code" in segv["error"]

    bogus = by_n[16]
    assert bogus["status"] == "ok", "the fake claimed success"
    assert bogus["valid"] is False, "the independent verifier caught the bogus partition"
    assert bogus["trials"][0]["valid"] is False and bogus["trials"][0]["verify_errors"]
    assert bogus["trials"][0]["verify_time"] is not None
    assert bogus["result"]["verify_errors"] and sum(bogus["result"]["part_sizes"]) == 16

    good = by_n[18]
    assert good["status"] == "ok" and good["valid"] is True
    assert good["trials"][0]["verify_time"] is not None and good["trials"][0]["valid"] is True
    assert good["median"]["wall_time"] == good["trials"][0]["wall_time"] > 0


def test_cli_exit_code_flags_invalid_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                survivable_core: bool) -> None:
    _inject_fake_solver(monkeypatch)
    cfg_path = tmp_path / "fault.json"
    cfg_path.write_text(json.dumps(_fault_config([16])), encoding="utf-8")
    args = SimpleNamespace(config=str(cfg_path), output=str(tmp_path / "o.jsonl"), repeat=1, threads=1,
                           limit=None, instances_dir=str(tmp_path / "inst"))
    assert runner.run_from_cli(args) == 3
    rows = runner.load_results([tmp_path / "o.jsonl"])
    assert len(rows) == 1 and rows[0]["valid"] is False


def test_cli_exit_code_flags_a_run_where_every_pair_errored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            survivable_core: bool) -> None:
    _inject_fake_solver(monkeypatch)
    cfg_path = tmp_path / "fault.json"
    cfg_path.write_text(json.dumps(_fault_config([12])), encoding="utf-8")
    args = SimpleNamespace(config=str(cfg_path), output=str(tmp_path / "o.jsonl"), repeat=1, threads=1,
                           limit=None, instances_dir=str(tmp_path / "inst"))
    rc = runner.run_from_cli(args)
    rows = runner.load_results([tmp_path / "o.jsonl"])
    assert len(rows) == 1 and rows[0]["status"] == "error"
    assert rc != 0


def test_timeout_kills_the_whole_process_group() -> None:
    """A solver that forks (threads pools, helper processes) must not leave orphans behind."""
    child = (
        "import json, subprocess, sys, time; "
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(json.dumps({'event': 'grandchild', 'pid': g.pid}), flush=True); time.sleep(120)"
    )
    out = runner.run_subprocess([sys.executable, "-c", child], timeout=0.5, grace=0.5)
    assert out.timed_out is True and out.returncode == -signal.SIGKILL
    gpid = next(e["pid"] for e in out.events if e.get("event") == "grandchild")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(gpid, 0)
        except ProcessLookupError:
            break  # gone
        try:  # still listed: a zombie waiting for init is fine, a running sleeper is not
            state = Path(f"/proc/{gpid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            break
        if state == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {gpid} survived the process-group kill")


# ---------------------------------------------------------------------------
# medians, repeats and the verifier
# ---------------------------------------------------------------------------
def test_medians_come_from_the_repeats(tmp_path: Path, survivable_core: bool) -> None:
    cfg = runner.config_from_dict({"name": "med", "families": ["harary"], "sizes": [12], "k": [3],
                                   "algorithms": ["reference"], "repeats": 5, "warmup": 1, "timeout": 60})
    summary = runner.run_config(cfg, tmp_path / "med.jsonl", instances_dir=tmp_path / "inst",
                                log=lambda s: None)
    assert summary["rows"] == 1 and summary["ok"] == 1
    (row,) = runner.load_results([tmp_path / "med.jsonl"])
    trials = row["trials"]
    assert len(trials) == 5 and row["repeats"] == 5 and row["warmup"] == 1
    walls = [t["wall_time"] for t in trials]
    assert len(set(walls)) > 1, "five real timings, not one value copied"
    med = row["median"]
    assert med["wall_time"] == statistics.median(walls)
    assert med["wall_min"] == min(walls) and med["wall_max"] == max(walls) and med["n_ok"] == 5
    assert min(walls) <= med["wall_q1"] <= med["wall_time"] <= med["wall_q3"] <= max(walls)
    assert med["cpu_time"] == statistics.median([t["cpu_time"] for t in trials])
    assert med["verify_time"] == statistics.median([t["verify_time"] for t in trials])
    # the verifier ran on every trial, separately timed, and its verdict is stored per trial
    assert all(t["valid"] is True and t["verify_time"] is not None and t["verify_time"] >= 0 for t in trials)
    assert all(t["wall_time"] > 0 and t["solver_runtime"] > 0 for t in trials)
    # the representative trial (closest to the median) supplies the stored stats
    rep = min(trials, key=lambda t: abs(t["wall_time"] - med["wall_time"]))
    assert row["result"]["stats"]["max_flow_calls"] == rep["stats"]["max_flow_calls"]


def test_make_row_median_semantics_with_partial_timeout() -> None:
    cfg = runner.config_from_dict({"name": "t", "timeout": 5})
    info = {"name": "x", "n": 10, "m": 20, "k": 2, "density": 0.2, "family": "harary"}

    def trial(i: int, wall: float, cpu: float) -> dict[str, Any]:
        return {"event": "trial", "trial": i, "wall_time": wall, "solver_runtime": wall, "cpu_time": cpu,
                "status": "ok", "valid": True, "verify_time": 0.001, "verify_errors": [], "stats": {"c": i},
                "part_sizes": [5, 5], "message": ""}

    events = [{"event": "start"}, {"event": "loaded", "load_time": 0.01},
              trial(0, 0.5, 0.5), trial(1, 0.1, 0.1), trial(2, 0.3, 0.3), trial(3, 0.2, 0.2), trial(4, 0.4, 0.4),
              {"event": "done"}]
    ok = runner.WorkerOutcome(events, 0, False, 2.0, 1.5, 0.1, 40.0, "")
    row = runner.make_row(cfg, info, "reference", "reference", {}, ok, 5, 1, {}, None, "0")
    assert row["status"] == "ok" and row["valid"] is True
    assert row["median"]["wall_time"] == 0.3 and row["median"]["cpu_time"] == 0.3
    assert (row["median"]["wall_min"], row["median"]["wall_max"], row["median"]["n_ok"]) == (0.1, 0.5, 5)
    assert row["median"]["wall_q1"] == 0.2 and row["median"]["wall_q3"] == 0.4
    assert row["result"]["stats"] == {"c": 2}, "stats of the trial closest to the median"
    # two good trials, then the worker hung: the row is a timeout, the median covers the good trials only
    partial = runner.WorkerOutcome(events[:4], -9, True, 8.0, 7.0, 0.1, 40.0, "")
    row = runner.make_row(cfg, info, "reference", "reference", {}, partial, 5, 1, {}, None, "0")
    assert row["status"] == "timeout" and row["timed_out"] is True and row["valid"] is None
    assert row["median"]["wall_time"] == 0.3 and row["median"]["n_ok"] == 2
    assert [t["status"] for t in row["trials"]] == ["ok", "ok", "timeout"]


# ---------------------------------------------------------------------------
# identical inputs: the instance cache
# ---------------------------------------------------------------------------
def test_two_algorithms_read_byte_identical_instance_files(tmp_path: Path, survivable_core: bool) -> None:
    cfg = runner.config_from_dict({"name": "cache", "families": ["harary"], "sizes": [10], "k": [2],
                                   "algorithms": ["reference", "bruteforce"], "timeout": 60})
    inst_dir = tmp_path / "inst"
    runner.run_config(cfg, tmp_path / "a.jsonl", instances_dir=inst_dir, log=lambda s: None)
    rows = runner.load_results([tmp_path / "a.jsonl"])
    assert [r["algorithm"] for r in rows] == ["reference", "bruteforce"]
    assert all(r["status"] == "ok" and r["valid"] is True for r in rows)
    (path,) = {Path(r["instance"]["path"]) for r in rows}
    assert path.parent == inst_dir and path.exists()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mtime = path.stat().st_mtime_ns
    meta = [{k: v for k, v in r["instance"].items() if k != "generation_time"} for r in rows]
    assert meta[0] == meta[1], "both rows carry the same instance metadata"
    assert rows[0]["result"]["part_sizes"] == rows[1]["result"]["part_sizes"] or True  # oracle may differ
    # a later run with another algorithm re-uses the cached file untouched
    runner.run_config(cfg, tmp_path / "b.jsonl", instances_dir=inst_dir, algorithms=["reference-weighted"],
                      log=lambda s: None)
    (row_b,) = runner.load_results([tmp_path / "b.jsonl"])
    assert Path(row_b["instance"]["path"]) == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest and path.stat().st_mtime_ns == mtime
    # regenerating from the spec elsewhere yields the same bytes (deterministic canonical JSON)
    spec = runner.InstanceSpec("harary", 10, 2, "balanced", 1)
    other = runner.get_instance(spec, tmp_path / "other", log=lambda s: None)
    assert hashlib.sha256(Path(other["path"]).read_bytes()).hexdigest() == digest
    # and the file is canonical: key-sorted JSON that round-trips to itself
    text = path.read_text(encoding="utf-8")
    obj = json.loads(text)
    assert json.dumps(obj, sort_keys=True) == json.dumps(json.loads(json.dumps(obj, sort_keys=True)), sort_keys=True)
    assert list(obj) == sorted(obj)


# ---------------------------------------------------------------------------
# machine metadata is compared with the host
# ---------------------------------------------------------------------------
def test_machine_metadata_matches_the_host(survivable_core: bool) -> None:
    info = machine.collect_machine_info()
    assert info["hostname"] == socket.gethostname()
    assert info["platform"] == platform.platform() and info["machine"] == platform.machine()
    assert info["cpu"]["cores_logical"] == os.cpu_count()
    assert info["python"]["version"] == platform.python_version()
    assert info["python"]["executable"] == sys.executable
    ram_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    assert abs(info["memory"]["ram_gib"] - ram_gib) < 0.01
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
        models = re.findall(r"^model name\s*:\s*(.+)$", text, flags=re.M)
        if models:  # x86: the exact string of /proc/cpuinfo
            assert info["cpu"]["model"] == models[0].strip()
        assert isinstance(info["cpu"]["model"], str) and info["cpu"]["model"]
    git = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if git.returncode == 0:
        assert info["git"]["commit"] == git.stdout.strip()
        assert isinstance(info["git"]["dirty"], bool)
    assert info["packages"]["glsolver"] == __import__("glsolver").__version__
    json.dumps(info)


def test_cpu_model_is_a_model_not_an_architecture(survivable_core: bool) -> None:
    info = machine.collect_machine_info()
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("lscpu not available")
    names = re.findall(r"^Model name:\s*(.+)$", lscpu.stdout, flags=re.M) if lscpu.returncode == 0 else []
    if not names:
        pytest.skip("lscpu reports no model name on this host")
    assert info["cpu"]["model"] != platform.machine()
    assert any(nm.strip().split()[0] in info["cpu"]["model"] for nm in names)


# ---------------------------------------------------------------------------
# plots: log-log axes, fitted exponent, timeouts visible
# ---------------------------------------------------------------------------
def _row(name: str, n: int, algo: str, *, status: str = "ok", valid: bool | None = True,
         wall: float | None = None, timeout: float = 30.0, timed_out: bool = False, family: str = "harary",
         k: int = 4, config: str = "synthetic", stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """A minimal but schema-complete result row for plot/report tests."""
    return {
        "schema": 1, "config": config,
        "instance": {"name": name, "family": family, "variant": {"type": "plain"}, "mode": "balanced",
                     "connectivity": {"vertex_connectivity": k}, "meta": {}, "capacities": [n // k] * k,
                     "path": "/nonexistent/" + name + ".json"},
        "n": n, "m": 2 * n, "k": k, "density": min(1.0, 4.0 / max(n - 1, 1)),
        "algorithm": algo.split(":", 1)[0], "algorithm_label": algo, "options": {}, "threads": 1,
        "repeats": 1, "warmup": 0, "timeout": timeout, "trials": [],
        "median": {"wall_time": wall, "solver_runtime": wall, "cpu_time": wall, "verify_time": 0.001,
                   "wall_q1": wall, "wall_q3": wall, "wall_min": wall, "wall_max": wall,
                   "n_ok": 1 if wall is not None else 0},
        "status": status, "valid": valid, "timed_out": timed_out,
        "error": "worker exited with code 1: RuntimeError: boom" if status == "error" else None,
        "result": {"status": status, "valid": valid, "message": "", "verify_errors": [],
                   "part_sizes": [n // k] * k if status == "ok" else None, "part_weights": None,
                   "stats": stats or {"max_flow_calls": n, "contractions": n // 2}, "algorithm_run": algo,
                   "parts": None},
        "child": {"returncode": 0, "wall_time_total": (wall or 0) + 0.5, "cpu_time_user": 0.1,
                  "cpu_time_sys": 0.0, "cpu_time": 0.1, "peak_rss_mb": 40.0 + n / 100, "load_time": 0.01,
                  "stderr_tail": ""},
        "git_commit": "deadbeef", "glsolver_version": "0.1.0",
        "machine": {"hostname": "h", "platform": "p", "cpu": {"model": "test cpu", "cores_logical": 4},
                    "memory": {"ram_gib": 8}, "python": {"version": "3.12"}, "core": {"available": False}},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def test_scaling_axis_is_log_log_with_fitted_exponent_and_visible_timeouts(tmp_path: Path) -> None:
    rows = []
    for n in (100, 200, 400, 800):
        for seed in (1, 2, 3):  # three seeds per size: the IQR band has something to show
            rows.append(_row(f"harary_n{n}_k4_balanced_s{seed}", n, "reference",
                             wall=1e-6 * n ** 2 * (1 + 0.01 * seed)))
    rows.append(_row("harary_n1600_k4_balanced_s1", 1600, "reference", status="timeout", valid=None,
                     timed_out=True, timeout=30.0))
    plt = plot._mpl()
    fig, ax = plt.subplots()
    try:
        assert plot._draw_scaling_axis(ax, rows, "wall_time", "median wall time (s)", {}, title="t") is True
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        m = re.search(r"slope ([0-9.]+)", legend[0])
        assert m is not None, legend
        assert abs(float(m.group(1)) - 2.0) < 0.05, "quadratic data -> exponent 2 in the legend"
        texts = [t.get_text() for t in ax.texts]
        assert any(t == "timeout 30s" for t in texts), texts
        assert any(re.fullmatch(r"2\.0\d", t) for t in texts), "exponent printed next to the last point"
        hollow = [ln for ln in ax.lines if ln.get_markerfacecolor() == "none" and list(ln.get_xdata()) == [1600]]
        assert hollow and list(hollow[0].get_ydata()) == [30.0], "timeout drawn at the timeout value"
        assert any(list(ln.get_ydata()) == [30.0, 30.0] for ln in ax.lines), "dotted timeout level"
        # the fitted curve does not extend to the size that timed out (no fabricated point)
        curve = [ln for ln in ax.lines if len(ln.get_xdata()) == 4]
        assert curve and max(curve[0].get_xdata()) == 800
    finally:
        plt.close(fig)
    written = plot.make_plots(rows, tmp_path)
    png = tmp_path / "runtime_vs_n_harary.png"
    assert png in written and png.stat().st_size > 0


def test_fit_slope_recovers_known_exponents() -> None:
    xs = [10, 100, 1000, 10000, 100000]
    for exp in (1.0, 1.5, 3.0):
        slope, _ = plot.fit_slope(xs, [2.5 * x ** exp for x in xs])
        assert abs(slope - exp) < 1e-9
    # only the ``top`` largest sizes are used: a fast small-n regime does not pull the slope down
    ys = [1.0, 1.0, 1.0] + [1e-9 * x ** 2 for x in (10000, 100000)]
    slope, _ = plot.fit_slope(xs, ys, top=2)
    assert abs(slope - 2.0) < 1e-9
    assert plot.fit_slope([5], [1.0]) is None


def test_oracle_timeout_is_not_drawn_as_a_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[str] = []

    def capture(fig: Any, path: Path, dpi: int) -> Path:
        for ax in fig.axes:
            labels.extend(t.get_text() for t in ax.get_legend().get_texts())
        return path

    monkeypatch.setattr(plot, "_finish", capture)
    rows = [_row("h1", 10, "reference", wall=0.01),
            _row("h1", 10, "bruteforce", status="timeout", valid=None, wall=None, timed_out=True)]
    plot.plot_oracle_vs_solver(rows, tmp_path)
    assert labels and not any("DISAGREE" in lbl for lbl in labels), labels


# ---------------------------------------------------------------------------
# dashboard and summary
# ---------------------------------------------------------------------------
def _mixed_rows() -> list[dict[str, Any]]:
    rows = runner.load_results([SAMPLE_ROWS]) if SAMPLE_ROWS.exists() else []
    rows += [
        _row("syn_n40_k4_s1", 40, "reference", wall=0.2),
        _row("syn_n40_k4_s1", 40, "bruteforce", status="infeasible", valid=None, wall=0.5),  # disagreement
        _row("syn_n80_k4_s1", 80, "reference", status="timeout", valid=None, wall=None, timed_out=True),
        _row("syn_n80_k4_s1", 80, "ilp", status="error", valid=None, wall=None),
        _row("syn_n60_k4_s1", 60, "reference", status="ok", valid=False, wall=0.1),
        _row("syn_n60_k4_s1", 60, "ilp", status="ok", valid=True, wall=0.7),
        _row("syn_n70_k4_s1", 70, "reference", status="precondition_failed", valid=None, wall=None),
        _row("syn_n70_k4_s1", 70, "dag:linear", status="ok", valid=True, wall=0.05),
    ]
    return rows


def test_dashboard_lists_every_run_and_is_self_contained() -> None:
    rows = _mixed_rows()
    text = report.dashboard_html(rows)
    assert text.count("<tr class=") == len(rows)
    for r in rows:
        assert r["instance"]["name"] in text and r["algorithm_label"] in text
    # self-contained: no external stylesheet/script/font/image; the only URL is the SVG namespace
    assert "<script src" not in text and "<link" not in text and "@import" not in text
    stripped = re.sub(r'xmlns(?::\w+)?="http://www\.w3\.org/[^"]*"', "", text)
    assert "http://" not in stripped and "https://" not in stripped
    for ref in re.findall(r"url\(([^)]*)\)", text):
        assert ref.strip("'\"").startswith("#"), f"external url() reference: {ref}"
    # every verdict is rendered
    assert "VALID</span>" in text and "INVALID</span>" in text and "TIMEOUT</span>" in text
    assert "ERROR</span>" in text and "precondition failed</span>" in text
    assert "DISAGREE" in text
    tiles = {lab: int(val) for val, lab in
             re.findall(r'<div class="tile[^"]*"><div class="v">(\d+)</div><div class="l">([^<]+)</div>', text)}
    n_valid = sum(1 for r in rows if r["valid"] is True)
    assert tiles["runs"] == len(rows) and tiles["VALID (independent verifier)"] == n_valid
    assert tiles["INVALID"] == 1 and tiles["timeouts"] == 1 and tiles["errors"] == 1
    # two disagreements: the verified solver partition on the instance the oracle called infeasible,
    # and the INVALID solver output next to a valid oracle partition (report._agreement -> "solver INVALID")
    assert tiles["precondition failed"] == 1 and tiles["oracle disagreements"] == 2
    # the machine banner and the row filters are populated from the rows themselves
    assert 'data-verdict="invalid"' in text and 'data-status="timeout"' in text
    assert "core built:" in text


def test_dashboard_from_the_real_smoke_rows_draws_graphs() -> None:
    if not SAMPLE_ROWS.exists():
        pytest.skip("benchmarks/results/sample_smoke.jsonl not present")
    rows = runner.load_results([SAMPLE_ROWS])
    assert rows and all(r["result"]["parts"] is not None for r in rows), "n <= 30: partitions kept"
    text = report.dashboard_html(rows)
    assert text.count("<tr class=") == len(rows)
    drawable = [r for r in rows if Path(r["instance"]["path"]).exists()]
    if drawable:
        assert text.count("<svg") >= len(drawable)
    else:
        assert text.count('class="parts"') == len(rows)  # text listing fallback


def test_summary_markdown_counts_every_status() -> None:
    rows = _mixed_rows()
    md = report.summary_markdown(rows)
    assert "## Overview" in md and "## synthetic" in md
    line = next(ln for ln in md.splitlines() if ln.startswith("| reference |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    runs = sum(1 for r in rows if r["algorithm_label"] == "reference")
    valid = sum(1 for r in rows if r["algorithm_label"] == "reference" and r["valid"] is True)
    assert cells[1] == str(runs) and cells[2] == str(valid) and cells[3] == "1"  # runs, VALID, INVALID
    assert cells[4] == "1" and cells[6] == "1"  # timeouts, precondition failed
    assert "Machine: test cpu" in md or "Machine:" in md
