"""Tests for the benchmark framework: runner (grid, cache, subprocess isolation, rows),
plot (PNG output) and report (markdown + verification dashboard)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # ``benchmarks`` is a repo-level package, not part of the wheel
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import machine, plot, report, runner  # noqa: E402

CONFIG_NAMES = ("smoke", "scaling", "families", "imbalance", "directed", "weighted", "dag",
                "connectivity_margin")


# ---------------------------------------------------------------------------
# machine info
# ---------------------------------------------------------------------------
def test_machine_info_is_json_and_complete():
    info = machine.collect_machine_info()
    for key in ("platform", "cpu", "memory", "python", "packages", "core", "git", "collected_at"):
        assert key in info
    assert info["cpu"]["cores_logical"] >= 1
    assert info["packages"]["glsolver"] is not None
    json.dumps(info)  # must be plain JSON


# ---------------------------------------------------------------------------
# configuration and grid expansion
# ---------------------------------------------------------------------------
def test_all_shipped_configs_parse_and_expand():
    for name in CONFIG_NAMES:
        cfg = runner.load_config(name)
        specs = runner.expand_grid(cfg)
        assert specs, name
        assert len({s.name for s in specs}) == len(specs), f"duplicate instance names in {name}"
        for algo in cfg.algorithms:
            assert algo in runner.KNOWN_ALGORITHMS


def test_grid_fractions_fixed_families_and_overrides():
    cfg = runner.config_from_dict({
        "name": "t", "sizes": [10, 40], "k": [2, 0.25], "seeds": [1],
        "families": ["harary", "paper_running_example", {"name": "counterexample", "copies": 1},
                     {"name": "erdos_renyi", "sizes": [12], "k": [3],
                      "variants": [{"type": "weighted", "w_max": 3}]}],
        "algorithms": ["reference"],
    })
    specs = runner.expand_grid(cfg)
    names = [s.name for s in specs]
    # fraction 0.25 of n=40 is k=10; of n=10 it is k=2 (deduplicated with the absolute k=2)
    assert "harary_n40_k10_balanced_s1" in names
    assert "harary_n10_k2_balanced_s1" in names
    assert names.count("harary_n10_k2_balanced_s1") == 1
    # fixed families ignore sizes/k and appear exactly once
    assert names.count("paper_running_example") == 1
    assert names.count("counterexample_c1") == 1
    # family-level overrides replace the grid lists
    er = [s for s in specs if s.family == "erdos_renyi"]
    assert [(s.n, s.k, s.variant_dict["type"]) for s in er] == [(12, 3, "weighted")]
    assert er[0].name == "erdos_renyi_n12_k3_balanced_s1_w3s0"


def test_eval_param_is_restricted():
    assert runner.eval_param("4*k", 10, 3) == 12
    assert runner.eval_param("max(3, k + 1)", 10, 2) == 3
    assert runner.eval_param("n // 2", 11, 2) == 5
    assert runner.eval_param(7, 1, 1) == 7
    with pytest.raises(ValueError):
        runner.eval_param("__import__('os').system('true')", 1, 1)
    with pytest.raises(ValueError):
        runner.eval_param("m + 1", 1, 1)


# ---------------------------------------------------------------------------
# instance generation and caching
# ---------------------------------------------------------------------------
def test_instance_cache_is_deterministic(tmp_path):
    spec = runner.InstanceSpec("harary", 12, 3, "balanced", 7)
    info1 = runner.get_instance(spec, tmp_path / "a", log=lambda s: None)
    info2 = runner.get_instance(spec, tmp_path / "b", log=lambda s: None)
    text1 = Path(info1["path"]).read_text()
    text2 = Path(info2["path"]).read_text()
    assert text1 == text2
    assert info1["n"] == 12 and info1["k"] == 3 and info1["m"] > 0
    assert 0 < info1["density"] <= 1
    assert info1["connectivity"]["vertex_connectivity"] == 3
    assert (tmp_path / "a" / f"{spec.name}.info.json").exists()
    # the cache is reused (same sidecar content)
    info3 = runner.get_instance(spec, tmp_path / "a", log=lambda s: None)
    assert info3["m"] == info1["m"]


def test_variants_modes_and_applicability(tmp_path):
    directed = runner.InstanceSpec("erdos_renyi", 14, 3, "extreme", 2, (("drop_fraction", 0.5), ("type", "directed")))
    info_d = runner.get_instance(directed, tmp_path, log=lambda s: None)
    assert info_d["directed"] is True and info_d["weighted"] is False
    assert 0 in info_d["capacities"]  # extreme mode: one part takes everything
    weighted = runner.InstanceSpec("harary", 14, 3, "balanced", 2, (("slack", 0), ("type", "weighted"), ("w_max", 4)))
    info_w = runner.get_instance(weighted, tmp_path, log=lambda s: None)
    assert info_w["weighted"] is True and info_w["w_max"] >= 1
    dag = runner.InstanceSpec("random_kT_dag", 14, 3, "balanced", 2)
    info_dag = runner.get_instance(dag, tmp_path, log=lambda s: None)
    assert info_dag["dag_kT_connected"] is True
    # applicability rules
    assert runner.algorithm_applies("reference", info_w, None, core_ok=False)[0] is False
    assert runner.algorithm_applies("reference-weighted", info_w, None, core_ok=False)[0] is True
    assert runner.algorithm_applies("reference-dag", info_d, None, core_ok=False)[0] is False
    assert runner.algorithm_applies("reference-dag", info_dag, None, core_ok=False)[0] is True
    assert runner.algorithm_applies("general", info_d, None, core_ok=False) == (False, "C++ core not built")
    assert runner.algorithm_applies("reference", info_d, {"max_n": 10}, core_ok=False)[0] is False


def test_generation_failure_is_skipped_not_raised(tmp_path):
    cfg = runner.config_from_dict({"name": "bad", "families": ["cycle"], "sizes": [10], "k": [4],
                                   "algorithms": ["reference"]})
    summary = runner.run_config(cfg, tmp_path / "bad.jsonl", instances_dir=tmp_path / "inst", log=lambda s: None)
    assert summary["rows"] == 0 and summary["skipped"] == 1
    assert any("generation failed" in str(v) for v in summary["skipped_reasons"].values())


# ---------------------------------------------------------------------------
# subprocess isolation
# ---------------------------------------------------------------------------
def test_subprocess_timeout_is_contained():
    t0 = time.monotonic()
    out = runner.run_subprocess(
        [sys.executable, "-c", "import time; print('{\"event\": \"start\"}', flush=True); time.sleep(60)"],
        timeout=0.3, grace=0.2,
    )
    assert out.timed_out is True
    assert time.monotonic() - t0 < 15
    assert out.events and out.events[0]["event"] == "start"
    assert out.peak_rss_mb > 0


def test_subprocess_crash_is_contained():
    out = runner.run_subprocess([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], timeout=10)
    assert out.timed_out is False and out.returncode == 3 and "boom" in out.stderr_tail
    seg = runner.run_subprocess([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"], timeout=10)
    # -11 normally; a sanitizer runtime (LD_PRELOAD=libasan) turns the signal into exit code 1
    assert seg.returncode is not None and seg.returncode != 0 and seg.timed_out is False


def test_make_row_records_timeouts_and_errors():
    cfg = runner.config_from_dict({"name": "t", "timeout": 5})
    info = {"name": "x", "n": 10, "m": 20, "k": 2, "density": 0.2, "family": "harary"}
    to = runner.WorkerOutcome([{"event": "start"}], -9, True, 6.0, 5.0, 0.1, 50.0, "")
    row = runner.make_row(cfg, info, "reference", "reference", {}, to, 1, 1, {}, None, "0")
    for key in runner.REQUIRED_ROW_KEYS:
        assert key in row
    assert row["status"] == "timeout" and row["timed_out"] is True and row["valid"] is None
    assert row["median"]["wall_time"] is None and row["trials"][-1]["status"] == "timeout"
    crash = runner.WorkerOutcome([{"event": "start"}], 1, False, 0.5, 0.4, 0.0, 30.0, "Traceback\nRuntimeError: x")
    row = runner.make_row(cfg, info, "reference", "reference", {}, crash, 1, 1, {}, None, "0")
    assert row["status"] == "error" and "RuntimeError" in row["error"] and row["valid"] is None


# ---------------------------------------------------------------------------
# end to end: smoke config on the reference backend -> plots -> report
# ---------------------------------------------------------------------------
def test_smoke_reference_end_to_end(tmp_path):
    out = tmp_path / "smoke.jsonl"
    args = SimpleNamespace(config="smoke", output=str(out), repeat=1, threads=1, limit=6,
                           algorithms="reference", instances_dir=str(tmp_path / "instances"))
    rc = runner.run_from_cli(args)
    assert rc == 0
    rows = runner.load_results([out])
    assert len(rows) == 6
    for row in rows:
        for key in runner.REQUIRED_ROW_KEYS:
            assert key in row, key
        assert row["status"] == "ok"
        assert row["valid"] is True, row.get("error")
        assert row["algorithm"] == "reference" and row["config"] == "smoke"
        assert row["n"] > 0 and row["m"] > 0 and row["k"] >= 2 and 0 < row["density"] <= 1
        assert len(row["trials"]) == 1
        trial = row["trials"][0]
        assert trial["wall_time"] > 0 and trial["verify_time"] is not None and trial["valid"] is True
        assert row["median"]["wall_time"] == trial["wall_time"]
        stats = row["result"]["stats"]
        assert stats["max_flow_calls"] > 0 and "time_essential" in stats
        assert len(stats["graph_size_over_time"]) <= runner.MAX_SIZE_SAMPLES
        assert sum(row["result"]["part_sizes"]) == row["n"]
        assert row["result"]["parts"] is not None  # n <= 30: partition kept for the dashboard
        assert row["child"]["peak_rss_mb"] > 0 and row["child"]["cpu_time"] >= 0
        assert row["machine"]["cpu"]["cores_logical"] >= 1
        assert row["glsolver_version"] and row["timestamp"]
        assert row["instance"]["family"] in ("harary", "erdos_renyi")
        assert Path(row["instance"]["path"]).exists()
        assert row["instance"]["sha256"] == runner.file_sha256(Path(row["instance"]["path"]))
        assert row["child"]["env"]["OMP_NUM_THREADS"] == "1" and row["grace"] == 2.0
        assert len(row["load_average"]) == 3 and row["machine"]["core"]["probe"] == "subprocess"
    # identical inputs: every row's instance file is the cached one
    assert len({r["instance"]["name"] for r in rows}) == 6

    pngs = plot.make_plots(rows, tmp_path / "plots")
    assert any(p.suffix == ".png" and p.stat().st_size > 0 for p in pngs)
    assert (tmp_path / "plots" / "runtime_vs_n_harary.png").exists()

    md = report.summary_markdown(rows)
    assert "| reference |" in md and "median wall" in md
    html_path = report.write_dashboard(rows, tmp_path / "dashboard.html")
    text = html_path.read_text(encoding="utf-8")
    assert "VALID" in text and "<table" in text and text.count("<tr class=") == 6
    assert "INVALID</span>" not in text
    assert "<svg" in text or 'class="parts"' in text  # inline drawing or text listing


def test_worker_cli_runs_standalone(tmp_path):
    """The worker must be runnable directly (no package install of ``benchmarks``)."""
    spec = runner.InstanceSpec("harary", 10, 2, "balanced", 1)
    info = runner.get_instance(spec, tmp_path, log=lambda s: None)
    cmd = [sys.executable, str(REPO_ROOT / "benchmarks" / "runner.py"), "--worker", "--instance", info["path"],
           "--algorithm", "reference", "--repeats", "2", "--warmup", "1", "--threads", "1", "--seed", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds.count("warmup") == 1 and kinds.count("trial") == 2 and kinds[-1] == "done"
    assert all(e["valid"] is True for e in events if e["event"] == "trial")


# ---------------------------------------------------------------------------
# the driver is isolated from the compiled core (probed out of process)
# ---------------------------------------------------------------------------
def test_core_probe_survives_a_core_that_aborts_on_import(tmp_path, monkeypatch):
    """A probe that dies like a sanitizer-linked core must yield 'not usable' + the headline,
    and a run that lists core algorithms must still produce its reference rows."""
    dying = ("import os, sys; sys.stderr.write('==7==ASan runtime does not come first in initial "
             "library list; preload it with LD_PRELOAD.\\n'); sys.stderr.flush(); os.abort()")
    monkeypatch.setattr(machine, "_core_probe_command", lambda: [sys.executable, "-c", dying])
    core = machine.core_info()
    assert core["available"] is False and core["importable"] is False and core["probe"] == "subprocess"
    assert core["returncode"] not in (0, None)
    assert "ASan runtime does not come first" in core["error"], core
    json.dumps(core)
    cfg = runner.config_from_dict({"name": "iso", "families": ["harary"], "sizes": [10], "k": [2],
                                   "algorithms": ["reference", "general"], "timeout": 60})
    summary = runner.run_config(cfg, tmp_path / "iso.jsonl", instances_dir=tmp_path / "inst", log=lambda s: None)
    assert summary["rows"] == 1 and summary["ok"] == 1 and summary["skipped"] == 1
    assert summary["core_available"] is False and "ASan" in summary["core_error"]
    (reason,) = summary["skipped_reasons"]
    assert reason.startswith("C++ core not usable: core probe exited with code") and "ASan" in reason
    (row,) = runner.load_results([tmp_path / "iso.jsonl"])
    assert row["algorithm"] == "reference" and row["valid"] is True
    assert row["machine"]["core"]["available"] is False and "ASan" in row["machine"]["core"]["error"]


def test_driver_never_imports_the_core_in_process(tmp_path):
    """Trap any in-process import of glsolver._core in the driver: the run must still succeed."""
    script = f"""
import importlib.abc, json, os, sys
class Trap(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "glsolver._core":
            sys.stderr.write("driver imported glsolver._core in-process\\n"); sys.stderr.flush(); os._exit(97)
        return None
sys.meta_path.insert(0, Trap())
sys.path.insert(0, {str(REPO_ROOT)!r})
from pathlib import Path
from benchmarks import runner
cfg = runner.config_from_dict({{"name": "trap", "families": ["harary"], "sizes": [10], "k": [2],
                               "algorithms": ["reference", "general"], "timeout": 60}})
s = runner.run_config(cfg, Path({str(tmp_path / "trap.jsonl")!r}), instances_dir=Path({str(tmp_path / "inst")!r}),
                      log=lambda m: None)
print(json.dumps(s))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                          cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "imported glsolver._core in-process" not in proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    rows = runner.load_results([tmp_path / "trap.jsonl"])
    # 'reference' always runs; 'general' runs too when the C++ core is usable (probed in a subprocess).
    core_ok = rows[0]["machine"]["core"]["available"]
    expected = 2 if core_ok else 1
    assert summary["rows"] == expected and summary["ok"] == expected, summary
    assert len(rows) == expected
    for row in rows:
        assert row["machine"]["core"]["probe"] == "subprocess"
        assert isinstance(row["machine"]["core"]["available"], bool)


# ---------------------------------------------------------------------------
# exit codes, crash summaries, per-row provenance, grace
# ---------------------------------------------------------------------------
def test_exit_code_distinguishes_invalid_errors_and_timeouts():
    base = {"rows": 4, "ok": 4, "invalid": 0, "timeouts": 0, "errors": 0, "skipped": 3, "precondition_failed": 1}
    assert runner.exit_code(base) == runner.EXIT_OK == 0
    assert runner.exit_code({**base, "errors": 1}) == runner.EXIT_FAILED_RUNS == 2
    assert runner.exit_code({**base, "timeouts": 1}) == 2
    assert runner.exit_code({**base, "invalid": 1}) == runner.EXIT_INVALID == 3
    assert runner.exit_code({**base, "invalid": 1, "errors": 5, "timeouts": 5}) == 3, "INVALID outranks the rest"


def test_stderr_summary_prefers_the_crash_headline():
    frames = "\n".join(f"    #{i} 0xfba13e16bb48 in thread_start clone3.S:{i}" for i in range(26))
    asan = "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x50\n" + frames
    assert machine.summarize_stderr(asan) == "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x50"
    assert machine.summarize_stderr("Traceback (most recent call last):\n  File x\nRuntimeError: x") == "RuntimeError: x"
    assert machine.summarize_stderr("==5==ASan runtime does not come first in initial library list").startswith("==5==")
    assert machine.summarize_stderr("Fatal Python error: Aborted\n\nCurrent thread") == "Fatal Python error: Aborted"
    assert machine.summarize_stderr("some warning\nlast line") == "last line"
    assert machine.summarize_stderr("") == "" and machine.summarize_stderr("x" * 500, limit=10) == "x" * 10
    cfg = runner.config_from_dict({"name": "t", "timeout": 5})
    info = {"name": "x", "n": 10, "m": 20, "k": 2, "density": 0.2, "family": "harary"}
    long_report = asan + "\n" + "\n".join(f"    #{i} more frames" for i in range(300))
    crash = runner.WorkerOutcome([{"event": "start"}], 1, False, 0.5, 0.4, 0.0, 30.0, long_report[-4000:],
                                 stderr_head=long_report[:4000])
    row = runner.make_row(cfg, info, "ilp", "ilp", {}, crash, 1, 1, {}, None, "0")
    assert row["error"] == "worker exited with code 1: ==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x50"
    assert row["child"]["stderr_head"].startswith("==1==ERROR: AddressSanitizer")
    assert len(row["child"]["stderr_head"]) <= runner.STDERR_HEAD_CHARS
    assert row["child"]["stderr_tail"].endswith("#299 more frames") and len(row["child"]["stderr_tail"]) <= runner.STDERR_TAIL_CHARS
    # a worker that died before reporting its environment: the row records what the driver exported
    assert row["child"]["env"]["OMP_NUM_THREADS"] == "1" and row["child"]["env"]["PYTHONHASHSEED"] is not None
    assert row["grace"] == 2.0 and len(row["load_average"]) == 3


def test_grace_period_is_configurable_and_recorded():
    assert runner.effective_grace(runner.config_from_dict({"name": "t", "timeout": 5})) == 2.0
    assert runner.effective_grace(runner.config_from_dict({"name": "t", "timeout": 600})) == 60.0
    assert runner.effective_grace(runner.config_from_dict({"name": "t", "timeout": 600, "grace": 0})) == 0.0
    cfg = runner.config_from_dict({"name": "t", "timeout": 4, "grace": 0.5})
    assert cfg.grace == 0.5 and runner.effective_grace(cfg) == 0.5
    info = {"name": "x", "n": 10, "m": 20, "k": 2, "density": 0.2, "family": "harary"}
    to = runner.WorkerOutcome([{"event": "start"}], -9, True, 4.6, 4.0, 0.1, 50.0, "")
    row = runner.make_row(cfg, info, "reference", "reference", {}, to, 1, 1, {}, None, "0", load_average=[1.0, 2.0, 3.0])
    assert row["grace"] == 0.5 and row["timeout"] == 4.0 and row["load_average"] == [1.0, 2.0, 3.0]
    assert row["trials"][-1]["message"] == "no progress for 4s (+0.5s grace); process group killed"
    # the kill really happens at timeout + grace, not at timeout
    t0 = time.monotonic()
    out = runner.run_subprocess([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.4, grace=0.6)
    elapsed = time.monotonic() - t0
    # The point is that the kill happens at timeout + grace rather than at
    # timeout: the lower bound is the assertion.  The upper bound only guards
    # against "never killed", so keep it generous — a loaded or emulated runner
    # can take seconds to schedule the signal, and this test must not turn into
    # a measurement of the machine.
    assert out.timed_out is True and elapsed >= 1.0, elapsed
    assert elapsed < 60.0, f"process was not killed promptly after timeout+grace: {elapsed:.1f}s"


def test_rows_record_the_worker_environment_and_a_per_row_load(tmp_path):
    cfg = runner.config_from_dict({"name": "env", "families": ["harary"], "sizes": [10, 12], "k": [2],
                                   "algorithms": ["reference"], "timeout": 60, "grace": 1.5, "threads": 1})
    summary = runner.run_config(cfg, tmp_path / "env.jsonl", instances_dir=tmp_path / "inst", log=lambda s: None)
    assert summary["rows"] == 2 and summary["ok"] == 2
    rows = runner.load_results([tmp_path / "env.jsonl"])
    expected_seed = os.environ.get("PYTHONHASHSEED", "0")
    for row in rows:
        env = row["child"]["env"]
        assert env == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "GLSOLVER_THREADS": "1", "PYTHONHASHSEED": expected_seed}, env
        assert row["grace"] == 1.5
        assert isinstance(row["load_average"], list) and len(row["load_average"]) == 3
        assert all(isinstance(x, float) and x >= 0 for x in row["load_average"])
        assert row["instance"]["sha256"] == runner.file_sha256(Path(row["instance"]["path"]))
        assert row["instance"]["generator_fingerprint"] == runner.generator_fingerprint()
    # the timestamps are per row (written after each pair), not one value copied
    assert rows[0]["timestamp"] <= rows[1]["timestamp"]


# ---------------------------------------------------------------------------
# instance cache: digests and generator fingerprint
# ---------------------------------------------------------------------------
def test_instance_cache_verifies_digest_and_generator_fingerprint(tmp_path):
    spec = runner.InstanceSpec("harary", 12, 3, "balanced", 7)
    logs: list[str] = []
    info = runner.get_instance(spec, tmp_path, log=logs.append)
    path, info_path = Path(info["path"]), tmp_path / f"{spec.name}.info.json"
    digest = runner.file_sha256(path)
    assert info["sha256"] == digest and len(digest) == 64
    assert info["generator_fingerprint"] == runner.generator_fingerprint() and info["glsolver_version"]
    assert json.loads(info_path.read_text())["sha256"] == digest
    # a clean hit rewrites nothing
    mtime = path.stat().st_mtime_ns
    logs.clear()
    assert runner.get_instance(spec, tmp_path, log=logs.append)["sha256"] == digest
    assert path.stat().st_mtime_ns == mtime and logs == []
    # a corrupted instance file is detected and regenerated to identical bytes
    path.write_text(path.read_text() + " ")
    assert runner.file_sha256(path) != digest
    logs.clear()
    info2 = runner.get_instance(spec, tmp_path, log=logs.append)
    assert any("regenerating" in m and "sha256" in m for m in logs), logs
    assert runner.file_sha256(path) == digest == info2["sha256"]
    # a sidecar produced by a different generator version is regenerated
    side = json.loads(info_path.read_text())
    side["generator_fingerprint"] = "0" * 64
    info_path.write_text(json.dumps(side))
    logs.clear()
    runner.get_instance(spec, tmp_path, log=logs.append)
    assert any("generator sources or glsolver version changed" in m for m in logs), logs
    assert json.loads(info_path.read_text())["generator_fingerprint"] == runner.generator_fingerprint()
    # a sidecar without a digest (older runner) is not trusted
    side = json.loads(info_path.read_text())
    del side["sha256"]
    info_path.write_text(json.dumps(side))
    logs.clear()
    runner.get_instance(spec, tmp_path, log=logs.append)
    assert any("no digest" in m for m in logs), logs
    assert json.loads(info_path.read_text())["sha256"] == digest
    # verification can be switched off (large caches): the corrupt file is then served as is
    path.write_text(path.read_text() + " ")
    cached, why = runner.cache_entry_status(path, info_path, verify_digest=False)
    assert cached is not None and why == ""
    assert runner.cache_entry_status(path, info_path)[1] == "instance file does not match its recorded sha256"
    assert runner.cache_entry_status(tmp_path / "nope.json", info_path) == (None, "not cached")
    # the fingerprint is a stable function of the installed sources
    assert runner.generator_fingerprint() == runner.generator_fingerprint() and len(runner.generator_fingerprint()) == 64


# ---------------------------------------------------------------------------
# figures: crashes are as visible as timeouts; oracle classes match the dashboard
# ---------------------------------------------------------------------------
def _syn_row(name: str, n: int, algo: str, *, k: int = 4, status: str = "ok", valid=True, wall=None,
             timeout: float = 30.0, timed_out: bool = False) -> dict:
    row = {
        "schema": 1, "config": "syn", "n": n, "m": 2 * n, "k": k, "density": 0.1,
        "instance": {"name": name, "family": "harary", "variant": {"type": "plain"}, "mode": "balanced",
                     "connectivity": {}, "meta": {}, "capacities": [n // k] * k, "path": "/nonexistent"},
        "algorithm": algo.split(":", 1)[0], "algorithm_label": algo, "options": {}, "threads": 1,
        "repeats": 1, "warmup": 0, "timeout": timeout, "grace": 3.0, "trials": [],
        "median": {"wall_time": wall, "solver_runtime": wall, "cpu_time": wall, "verify_time": None,
                   "wall_q1": wall, "wall_q3": wall, "wall_min": wall, "wall_max": wall, "n_ok": int(wall is not None)},
        "status": status, "valid": valid, "timed_out": timed_out, "error": None,
        "result": {"status": status, "valid": valid, "message": "", "verify_errors": [], "part_sizes": None,
                   "part_weights": None, "stats": {}, "algorithm_run": algo, "parts": None},
        "child": {"returncode": 0, "peak_rss_mb": 10.0, "stderr_tail": "", "env": {}},
        "git_commit": None, "glsolver_version": "0", "machine": {"cpu": {"model": "m"}}, "load_average": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    return row


def test_scaling_figures_mark_crashed_sizes(tmp_path):
    rows = [_syn_row(f"h{n}", n, "reference", wall=1e-6 * n ** 2) for n in (100, 200, 400)]
    rows.append(_syn_row("h800", 800, "reference", status="error", valid=None))
    rows.append(_syn_row("h1600", 1600, "reference", status="timeout", valid=None, timed_out=True))
    agg = plot.aggregate_by_x(rows, lambda r: r["n"])
    assert agg[800] == {"n_ok": 0, "n_timeout": 0, "n_error": 1, "n_rows": 1, "timeout_value": None, "error_level": 30.0}
    assert agg[1600]["n_timeout"] == 1 and agg[1600]["n_error"] == 0
    plt = plot._mpl()
    fig, ax = plt.subplots()
    try:
        assert plot._draw_scaling_axis(ax, rows, "wall_time", "y", {}, title="t") is True
        crosses = [ln for ln in ax.lines if ln.get_marker() == "x"]
        assert crosses and list(crosses[0].get_xdata()) == [800] and list(crosses[0].get_ydata()) == [30.0]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "reference (errors)" in legend, legend
        hollow = [ln for ln in ax.lines if ln.get_markerfacecolor() == "none" and list(ln.get_xdata()) == [1600]]
        assert hollow and list(hollow[0].get_ydata()) == [30.0], "timeouts still drawn at the timeout value"
    finally:
        plt.close(fig)
    # runtime vs k: a crashed k is marked too
    rows_k = [_syn_row(f"k{k}", 200, "reference", k=k, wall=0.01 * k) for k in (2, 4)]
    rows_k.append(_syn_row("k8", 200, "reference", k=8, status="error", valid=None))
    written = plot.plot_runtime_vs_k(rows_k, tmp_path)
    assert written and written[0].exists()
    fig, ax = plt.subplots()
    try:
        seen: dict = {}
        agg_k = plot.aggregate_by_x(rows_k, lambda r: r["k"])
        pend: list = []
        assert plot._draw_failure_markers(ax, agg_k, sorted(agg_k), plot._style("reference", seen), "reference",
                                          True, set(), pend) is True
        assert [list(ln.get_xdata()) for ln in ax.lines if ln.get_marker() == "x"] == [[8]] and pend == []
    finally:
        plt.close(fig)


def test_oracle_figure_classes_match_the_dashboard(tmp_path, monkeypatch):
    axes_seen: list = []
    monkeypatch.setattr(plot, "_finish", lambda fig, path, dpi: (axes_seen.extend(fig.axes), path)[1])
    rows = [
        _syn_row("a", 10, "reference", wall=0.01), _syn_row("a", 10, "bruteforce", wall=0.5),          # agree
        _syn_row("b", 10, "reference", wall=0.02),
        _syn_row("b", 10, "bruteforce", status="infeasible", valid=None, wall=0.6),                     # DISAGREE
        _syn_row("c", 10, "reference", wall=0.03),
        _syn_row("c", 10, "bruteforce", status="timeout", valid=None, timed_out=True),                  # oracle timeout
        _syn_row("d", 10, "reference", wall=0.04),
        _syn_row("d", 10, "bruteforce", status="error", valid=None),                                    # oracle error
    ]
    for solver, oracle in ((rows[0], rows[1]), (rows[2], rows[3]), (rows[4], rows[5]), (rows[6], rows[7])):
        cls = plot.classify_oracle_pair(solver, oracle)
        rep_cls, rep_text = report._agreement(solver, oracle)
        assert (cls == "disagree") == (rep_cls == "disagree")
        assert (cls == "agree") == (rep_cls == "agree")
        assert (cls == "oracle timeout") == (rep_text == "oracle timeout")
    plot.plot_oracle_vs_solver(rows, tmp_path)
    (ax,) = axes_seen
    legend = sorted(t.get_text() for t in ax.get_legend().get_texts())
    assert legend == ["bruteforce vs reference", "bruteforce vs reference (DISAGREE)",
                      "bruteforce vs reference (oracle error)", "bruteforce vs reference (oracle timeout)"], legend
    by_label = {ln.get_label(): ln for ln in ax.lines if not ln.get_label().startswith("_")}
    to = by_label["bruteforce vs reference (oracle timeout)"]
    assert list(to.get_ydata()) == [30.0] and to.get_markerfacecolor() == "none"
    assert to.get_markeredgecolor() == plot.MUTED, "grey, not the red DISAGREE edge"
    dis = by_label["bruteforce vs reference (DISAGREE)"]
    assert dis.get_markeredgecolor() == "#d03b3b" and list(dis.get_xdata()) == [0.02]
    assert any(t.get_text() == "oracle timeout 30s" for t in ax.texts)


# ---------------------------------------------------------------------------
# machine: ARM CPU identification, results dir ignore rules
# ---------------------------------------------------------------------------
_ARM_CPUINFO = "".join(
    f"processor\t: {i}\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd evtstrm aes sha2 crc32 atomics fphp "
    f"sve sve2 i8mm bf16\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x0\n"
    f"CPU part\t: {'0xd87' if i < 10 else '0xd85'}\nCPU revision\t: 1\n\n" for i in range(20)
)
_ARM_LSCPU = """Architecture:            aarch64
CPU(s):                  20
Vendor ID:               ARM
Model name:              Cortex-X925
Model:                   1
Thread(s) per core:      1
Core(s) per socket:      10
Socket(s):               1
CPU max MHz:             3900.0000
Model name:              Cortex-A725
Model:                   1
Thread(s) per core:      1
Core(s) per socket:      10
Socket(s):               1
CPU max MHz:             2808.0000
"""
_X86_CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
cpu MHz\t\t: 2900.000
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse4_2 avx avx2 avx512f bmi2 popcnt fma

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
cpu MHz\t\t: 2900.000
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu sse4_2 avx avx2 avx512f bmi2 popcnt fma
"""


def test_cpu_info_identifies_arm_cores_without_a_model_name_line():
    via_lscpu = machine.cpu_info(cpuinfo_text=_ARM_CPUINFO, lscpu_text=_ARM_LSCPU)
    assert via_lscpu["model"] == "10x Cortex-X925 + 10x Cortex-A725"
    assert via_lscpu["models"] == [{"name": "Cortex-X925", "cpus": 10, "max_mhz": 3900.0},
                                   {"name": "Cortex-A725", "cpus": 10, "max_mhz": 2808.0}]
    assert via_lscpu["flags_sample"] == ["aes", "asimd", "atomics", "bf16", "crc32", "fphp", "i8mm", "sha2", "sve", "sve2"]
    assert via_lscpu["implementer"] == "ARM"
    assert [(p["part"], p["cpus"], p["name"]) for p in via_lscpu["parts"]] == [
        ("0xd85", 10, "Cortex-X925"), ("0xd87", 10, "Cortex-A725")]
    via_midr = machine.cpu_info(cpuinfo_text=_ARM_CPUINFO, lscpu_text="")  # no lscpu on the host
    assert via_midr["model"] == "10x Cortex-X925 + 10x Cortex-A725"
    assert machine.arm_part_name(0x41, 0xD0C) == "Neoverse-N1"
    assert machine.arm_part_name(0x51, 0xFFF) == "Qualcomm part 0xfff"
    unknown = _ARM_CPUINFO.replace("0xd87", "0xfff").replace("0xd85", "0xffe")
    assert machine.cpu_info(cpuinfo_text=unknown, lscpu_text="")["model"] == "10x ARM part 0xffe + 10x ARM part 0xfff"
    x86 = machine.cpu_info(cpuinfo_text=_X86_CPUINFO, lscpu_text="")
    assert x86["model"] == "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz", "x86: the exact string is kept"
    assert x86["flags_sample"] == ["avx", "avx2", "avx512f", "bmi2", "fma", "popcnt", "sse4_2"]
    assert x86["cores_physical"] == 2 and x86["models"] is None
    host = machine.cpu_info()
    assert isinstance(host["model"], str) and host["model"]
    json.dumps(host)


def test_parse_lscpu_handles_cluster_wording():
    text = "Model name:  Cortex-A76\nThread(s) per core: 1\nCore(s) per cluster: 4\nCluster(s): 2\nCPU max MHz: 2400.0\n"
    assert machine.parse_lscpu_models(text) == [{"name": "Cortex-A76", "cpus": 8, "max_mhz": 2400.0}]
    assert machine.parse_lscpu_models("Architecture: x86_64\n") == []


def test_results_dir_is_gitignored_except_the_sample():
    try:
        probe = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not a git work tree")

    def ignored(rel: str) -> bool:
        res = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel], capture_output=True, timeout=20)
        return res.returncode == 0

    # Policy: raw rows (*.jsonl), summary.md, plots and exhaustive tallies are tracked (they are the
    # machine-readable evidence behind docs/benchmarks.md); the regenerable dashboard and the instance
    # cache are not.
    for rel in ("benchmarks/results/dashboard.html", "benchmarks/results/other.html", "benchmarks/instances/x.json"):
        assert ignored(rel), rel
    for rel in ("benchmarks/results/sample_smoke.jsonl", "benchmarks/results/summary.md",
                "benchmarks/results/plots/runtime_vs_n_all.png", "benchmarks/results/exhaustive_x.json",
                "benchmarks/runner.py"):
        assert not ignored(rel), rel


def test_cli_survives_a_core_whose_import_aborts(tmp_path):
    """No monkeypatching: a ``sitecustomize`` trap makes any import of ``glsolver._core`` abort the
    interpreter (as a sanitizer-linked build without LD_PRELOAD does).  The driver must finish its
    reference rows, exit 0, and record the probe's headline as the skip reason."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import importlib.abc, os, sys\n"
        "class Trap(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'glsolver._core':\n"
        "            sys.stderr.write('==4242==ERROR: AddressSanitizer: simulated abort on import\\n')\n"
        "            sys.stderr.flush(); os.abort()\n"
        "        return None\n"
        "sys.meta_path.insert(0, Trap())\n", encoding="utf-8")
    cfg = tmp_path / "trap.json"
    cfg.write_text(json.dumps({"name": "trap", "families": ["harary"], "sizes": [10], "k": [2],
                               "algorithms": ["reference", "general"], "timeout": 60}), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(site) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-m", "benchmarks.runner", str(cfg), "-o", str(tmp_path / "t.jsonl"),
                           "--instances-dir", str(tmp_path / "inst")], capture_output=True, text=True,
                          timeout=300, cwd=str(REPO_ROOT), env=env)
    assert proc.returncode == 0, proc.stderr
    assert "C++ core not usable: core probe exited with code" in proc.stderr and "AddressSanitizer" in proc.stderr
    (row,) = runner.load_results([tmp_path / "t.jsonl"])
    assert row["algorithm"] == "reference" and row["status"] == "ok" and row["valid"] is True
    assert row["machine"]["core"]["available"] is False
    assert "simulated abort on import" in row["machine"]["core"]["error"]
