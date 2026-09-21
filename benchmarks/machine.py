"""Hardware and software provenance for benchmark rows.

:func:`collect_machine_info` gathers everything a reader needs to interpret a
timing: platform, CPU model, core counts, nominal frequencies, RAM, the Python
and dependency versions, the ``glsolver`` version, whether the C++ core is
built/usable and any build information it exposes, the git commit of the
working tree, the current load average and thread-related environment
variables.  Every value is a plain JSON type; missing information is reported
as ``None`` rather than raising, because a benchmark run must not fail on an
exotic machine.

CPU identification (:func:`cpu_info`):

* x86: the exact ``model name`` line of ``/proc/cpuinfo`` plus a sample of the
  ISA flags;
* ARM (no ``model name`` line): the core types are taken from ``lscpu`` (one
  ``Model name`` per core type, e.g. ``10x Cortex-X925 + 10x Cortex-A725`` on a
  big.LITTLE part) or, without ``lscpu``, from the MIDR ``CPU implementer`` /
  ``CPU part`` values of ``/proc/cpuinfo`` and a table of known parts; the
  ``Features`` line supplies the flags; the raw implementer/part codes are kept;
* the nominal frequency is the maximum ``cpuinfo_max_freq`` of sysfs (per
  cluster on heterogeneous parts), falling back to ``cpu MHz`` and ``psutil``.

The compiled core is probed **out of process** (:func:`core_info`): a broken
or sanitizer-linked ``glsolver._core`` can abort the interpreter on import,
which must not kill the benchmark driver.  :func:`core_info_in_process` is the
body of that probe.

Usage::

    python -m benchmarks.machine --info      # pretty-print the dict as JSON
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "GLSOLVER_THREADS",
    "PYTHONHASHSEED",
)

# MIDR implementer codes (ARM ARM, "CPU implementer" of /proc/cpuinfo)
_ARM_IMPLEMENTERS = {
    0x41: "ARM", 0x42: "Broadcom", 0x43: "Cavium", 0x44: "DEC", 0x46: "Fujitsu", 0x48: "HiSilicon",
    0x49: "Infineon", 0x4D: "Motorola", 0x4E: "NVIDIA", 0x50: "APM", 0x51: "Qualcomm", 0x53: "Samsung",
    0x56: "Marvell", 0x61: "Apple", 0x66: "Faraday", 0x69: "Intel", 0x6D: "Microsoft", 0x70: "Phytium",
    0xC0: "Ampere",
}
# (implementer, part) -> core name; the ARM entries follow the kernel/lscpu tables
_ARM_PARTS = {
    (0x41, 0xD03): "Cortex-A53", (0x41, 0xD04): "Cortex-A35", (0x41, 0xD05): "Cortex-A55",
    (0x41, 0xD06): "Cortex-A65", (0x41, 0xD07): "Cortex-A57", (0x41, 0xD08): "Cortex-A72",
    (0x41, 0xD09): "Cortex-A73", (0x41, 0xD0A): "Cortex-A75", (0x41, 0xD0B): "Cortex-A76",
    (0x41, 0xD0C): "Neoverse-N1", (0x41, 0xD0D): "Cortex-A77", (0x41, 0xD0E): "Cortex-A76AE",
    (0x41, 0xD40): "Neoverse-V1", (0x41, 0xD41): "Cortex-A78", (0x41, 0xD42): "Cortex-A78AE",
    (0x41, 0xD43): "Cortex-A65AE", (0x41, 0xD44): "Cortex-X1", (0x41, 0xD46): "Cortex-A510",
    (0x41, 0xD47): "Cortex-A710", (0x41, 0xD48): "Cortex-X2", (0x41, 0xD49): "Neoverse-N2",
    (0x41, 0xD4A): "Neoverse-E1", (0x41, 0xD4B): "Cortex-A78C", (0x41, 0xD4C): "Cortex-X1C",
    (0x41, 0xD4D): "Cortex-A715", (0x41, 0xD4E): "Cortex-X3", (0x41, 0xD4F): "Neoverse-V2",
    (0x41, 0xD80): "Cortex-A520", (0x41, 0xD81): "Cortex-A720", (0x41, 0xD82): "Cortex-X4",
    (0x41, 0xD83): "Neoverse-V3AE", (0x41, 0xD84): "Neoverse-V3", (0x41, 0xD85): "Cortex-X925",
    (0x41, 0xD87): "Cortex-A725", (0x41, 0xD88): "Cortex-A520AE", (0x41, 0xD89): "Cortex-A720AE",
    (0x41, 0xD8E): "Neoverse-N3",
    (0x43, 0x0A1): "ThunderX", (0x43, 0x0AF): "ThunderX2", (0x43, 0x0B8): "ThunderX3",
    (0x46, 0x001): "A64FX", (0x48, 0xD01): "Kunpeng-920", (0x4E, 0x003): "Denver",
    (0x4E, 0x004): "Carmel", (0x50, 0x000): "X-Gene", (0x51, 0x001): "Oryon", (0x51, 0x800): "Kryo",
    (0x51, 0xC00): "Falkor", (0x51, 0xC01): "Saphira", (0xC0, 0xAC3): "Ampere-1",
    (0xC0, 0xAC4): "Ampere-1a",
}
_X86_FLAGS = {"avx", "avx2", "avx512f", "sse4_2", "bmi2", "popcnt", "fma"}
_ARM_FLAGS = {"asimd", "sve", "sve2", "atomics", "lse", "i8mm", "bf16", "fphp", "crc32", "aes", "sha2"}

# the first line of a crash report that a human wants to read (sanitizer
# headline, abort, fatal error, ...); Python tracebacks are handled separately
_CRASH_HEADLINE = re.compile(
    r"^(==\d+==|ERROR:|.*Sanitizer|.*: runtime error:|.*[Ff]atal( [Pp]ython)? error|"
    r"Segmentation fault|Aborted|.*Assertion .*failed|terminate called|.*Bus error|Killed)"
)
_PY_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b(: .*)?$")


def summarize_stderr(text: str, limit: int = 300) -> str:
    """The single most informative line of a process' stderr.

    A native crash or sanitizer report is summarised by its *first* matching
    headline (``==PID==ERROR: AddressSanitizer: ...``, ``Fatal ...``); a Python
    traceback by its *last* exception line (``RuntimeError: ...``); anything
    else by its last non-empty line.  Returns ``""`` for empty text.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in lines:
        if _CRASH_HEADLINE.match(ln):
            return ln[:limit]
    for ln in reversed(lines):
        if _PY_EXCEPTION_LINE.match(ln):
            return ln[:limit]
    return lines[-1][:limit]


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _lscpu_text() -> str | None:
    """Raw ``lscpu`` output (``None`` when the tool is missing or fails)."""
    try:
        res = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10,
                             env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 and res.stdout.strip() else None


def parse_lscpu_models(text: str) -> list[dict[str, Any]]:
    """``[{"name", "cpus", "max_mhz"}, ...]`` -- one entry per ``Model name`` block of ``lscpu``.

    util-linux prints one block per core type (big.LITTLE parts have several);
    the ``Core(s) per socket``/``Socket(s)`` (or ``per cluster``/``Cluster(s)``)
    and ``Thread(s) per core`` lines that follow a model name belong to it.
    """
    blocks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "Model name":
            cur = {"name": val, "raw": {}}
            blocks.append(cur)
        elif cur is not None:
            cur["raw"][key] = val
    def num(raw: dict[str, str], key: str) -> float | None:
        try:
            return float(raw[key])
        except (KeyError, ValueError):
            return None

    out: list[dict[str, Any]] = []
    for b in blocks:
        raw = b["raw"]
        cores = num(raw, "Core(s) per socket") or num(raw, "Core(s) per cluster")
        groups = num(raw, "Socket(s)") or num(raw, "Cluster(s)") or 1.0
        threads = num(raw, "Thread(s) per core") or 1.0
        cpus = int(cores * groups * threads) if cores else None
        out.append({"name": b["name"], "cpus": cpus, "max_mhz": num(raw, "CPU max MHz")})
    return out


def _arm_parts(cpuinfo: str) -> list[tuple[int, int, int]]:
    """``[(implementer, part, count), ...]`` from the MIDR fields of ``/proc/cpuinfo`` (ARM/Linux)."""
    impls = re.findall(r"^CPU implementer\s*:\s*(0x[0-9a-fA-F]+|\d+)", cpuinfo, flags=re.M)
    parts = re.findall(r"^CPU part\s*:\s*(0x[0-9a-fA-F]+|\d+)", cpuinfo, flags=re.M)
    if not impls or len(impls) != len(parts):
        return []
    counts = Counter((int(i, 0), int(p, 0)) for i, p in zip(impls, parts))
    return [(i, p, c) for (i, p), c in sorted(counts.items())]


def arm_part_name(implementer: int, part: int) -> str:
    """Human name of a MIDR ``(implementer, part)`` pair, or a descriptive fallback."""
    name = _ARM_PARTS.get((implementer, part))
    if name:
        return name
    vendor = _ARM_IMPLEMENTERS.get(implementer, f"implementer 0x{implementer:02x}")
    return f"{vendor} part 0x{part:03x}"


def _models_text(models: list[dict[str, Any]]) -> str:
    if len(models) == 1:
        return str(models[0]["name"])
    bits = []
    for m in models:
        bits.append(f"{m['cpus']}x {m['name']}" if m.get("cpus") else str(m["name"]))
    return " + ".join(bits)


def _sysfs_max_mhz() -> list[dict[str, Any]]:
    """``[{"max_mhz", "cpus"}, ...]`` from ``cpuinfo_max_freq`` of every CPU (distinct values, descending)."""
    counts: Counter[float] = Counter()
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        try:
            counts[round(int(p.read_text().strip()) / 1000.0, 1)] += 1
        except (OSError, ValueError):
            continue
    return [{"max_mhz": mhz, "cpus": n} for mhz, n in sorted(counts.items(), reverse=True)]


def cpu_info(cpuinfo_text: str | None = None, lscpu_text: str | None = None) -> dict[str, Any]:
    """CPU model(s), logical/physical core counts, nominal frequency and a flag sample.

    See the module docstring for the sources; ``cpuinfo_text``/``lscpu_text``
    override the host's ``/proc/cpuinfo`` and ``lscpu`` output (for tests).
    ``model`` is always a non-empty string; on heterogeneous parts it lists
    every core type with its CPU count and ``models`` holds the per-type detail.
    """
    info: dict[str, Any] = {
        "model": None,
        "models": None,
        "cores_logical": os.cpu_count(),
        "cores_physical": None,
        "mhz": None,
        "mhz_by_cluster": None,
        "flags_sample": None,
        "arch": platform.machine() or None,
        "implementer": None,
        "parts": None,
    }
    text = _read_text("/proc/cpuinfo") if cpuinfo_text is None else cpuinfo_text
    if text:
        models = re.findall(r"^model name\s*:\s*(.+)$", text, flags=re.M)
        if models:  # x86 and friends: the exact string
            info["model"] = models[0].strip()
            distinct = list(dict.fromkeys(m.strip() for m in models))
            if len(distinct) > 1:
                counts = Counter(m.strip() for m in models)
                info["models"] = [{"name": m, "cpus": counts[m], "max_mhz": None} for m in distinct]
                info["model"] = _models_text(info["models"])
        mhz = re.findall(r"^cpu MHz\s*:\s*([0-9.]+)", text, flags=re.M)
        if mhz:
            try:
                info["mhz"] = round(max(float(x) for x in mhz), 1)
            except ValueError:
                pass
        # physical cores = distinct (physical id, core id) pairs
        phys = re.findall(r"^physical id\s*:\s*(\d+)", text, flags=re.M)
        core = re.findall(r"^core id\s*:\s*(\d+)", text, flags=re.M)
        if phys and core and len(phys) == len(core):
            info["cores_physical"] = len(set(zip(phys, core)))
        flags = re.findall(r"^flags\s*:\s*(.+)$", text, flags=re.M)
        if flags:
            info["flags_sample"] = sorted(_X86_FLAGS & set(flags[0].split()))
        features = re.findall(r"^Features\s*:\s*(.+)$", text, flags=re.M)
        if features and info["flags_sample"] is None:
            info["flags_sample"] = sorted(_ARM_FLAGS & set(features[0].split()))
        parts = _arm_parts(text)
        if parts:
            info["implementer"] = _ARM_IMPLEMENTERS.get(parts[0][0], f"0x{parts[0][0]:02x}")
            info["parts"] = [{"implementer": f"0x{i:02x}", "part": f"0x{p:03x}", "cpus": c,
                              "name": arm_part_name(i, p)} for i, p, c in parts]
    if info["model"] is None:
        # ARM: lscpu names every core type; the MIDR table is the fallback
        ltext = _lscpu_text() if lscpu_text is None else lscpu_text
        models_l = parse_lscpu_models(ltext) if ltext else []
        if models_l:
            info["models"] = models_l
        elif info["parts"]:
            info["models"] = [{"name": p["name"], "cpus": p["cpus"], "max_mhz": None} for p in info["parts"]]
        if info["models"]:
            info["model"] = _models_text(info["models"])
    if info["model"] is None:
        info["model"] = platform.processor() or platform.machine() or None
    clusters = _sysfs_max_mhz()
    if clusters:
        info["mhz_by_cluster"] = clusters
        info["mhz"] = max(c["max_mhz"] for c in clusters)
    try:  # optional refinement
        import psutil  # type: ignore

        if info["cores_physical"] is None:
            info["cores_physical"] = psutil.cpu_count(logical=False)
        if info["mhz"] is None:
            freq = psutil.cpu_freq()
            if freq is not None:
                info["mhz"] = round(float(freq.max or freq.current), 1)
    except Exception:
        pass
    return info


def memory_info() -> dict[str, Any]:
    """Total physical RAM in GiB (``sysconf`` on POSIX, ``psutil`` otherwise)."""
    total: int | None = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        try:
            import psutil  # type: ignore

            total = int(psutil.virtual_memory().total)
        except Exception:
            total = None
    return {"ram_gib": None if total is None else round(total / 2**30, 2)}


def git_info(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit hash, branch and dirty flag of the working tree (``None`` outside git)."""
    out: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}

    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    out["commit"] = run("rev-parse", "HEAD")
    out["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    out["dirty"] = None if status is None else bool(status)
    return out


def _version_of(module: str) -> str | None:
    try:
        mod = __import__(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", None))


def core_info_in_process() -> dict[str, Any]:
    """Import ``glsolver._core`` **in this process** and describe it.

    This is the body of the out-of-process probe (:func:`core_info`); call it
    directly only where an aborting import is acceptable.  The core may expose
    ``build_info``, ``compiler``, ``compiler_flags``, ``build_type``,
    ``version`` or ``__version__``; each is copied when present (callables are
    called), everything else is ignored.
    """
    info: dict[str, Any] = {"importable": False, "available": False, "build": {}}
    try:
        from glsolver.api import core_available

        info["available"] = bool(core_available())
    except Exception as exc:
        info["available"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from glsolver import _core  # type: ignore
    except Exception as exc:
        info.setdefault("error", f"{type(exc).__name__}: {exc}")
        return info
    info["importable"] = True
    build: dict[str, Any] = {}
    for key in ("build_info", "compiler", "compiler_flags", "build_type", "version", "__version__",
                "debug_asserts", "sanitize", "openmp"):
        if not hasattr(_core, key):
            continue
        val = getattr(_core, key)
        try:
            if callable(val):
                val = val()
        except Exception:
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            build[key] = val
        elif isinstance(val, dict):
            build[key] = {str(k): v for k, v in val.items() if isinstance(v, (str, int, float, bool))}
        else:
            build[key] = str(val)
    info["build"] = build
    return info


_PROBE_SCRIPT = (
    "import json, sys; sys.path.insert(0, {root!r}); "
    "from benchmarks.machine import core_info_in_process; "
    "sys.stdout.write('\\n' + json.dumps(core_info_in_process()) + '\\n')"
)


def _core_probe_command() -> list[str]:
    """The subprocess that imports the core and prints :func:`core_info_in_process` as JSON."""
    return [sys.executable, "-c", _PROBE_SCRIPT.format(root=str(REPO_ROOT))]


def core_info(timeout: float = 120.0) -> dict[str, Any]:
    """Whether ``glsolver._core`` is importable/usable and what it says about its build.

    The core is imported in a **separate interpreter** (``sys.executable``, the
    caller's environment, so an ``LD_PRELOAD`` for a sanitizer build is
    honoured).  A probe that dies -- for example a sanitizer-linked build
    without its runtime preloaded, or an ABI mismatch that aborts on import --
    yields ``{"importable": False, "available": False, "error": <headline>}``
    instead of killing the caller.  ``probe`` records how the answer was
    obtained (``"subprocess"``) and ``returncode`` the probe's exit status.
    """
    base: dict[str, Any] = {"importable": False, "available": False, "build": {}, "probe": "subprocess"}
    try:
        proc = subprocess.run(_core_probe_command(), capture_output=True, text=True, timeout=timeout,
                              cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        return {**base, "returncode": None, "error": f"core probe timed out after {timeout:.0f}s"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {**base, "returncode": None, "error": f"core probe failed to start: {exc}"}
    if proc.returncode != 0:
        head = summarize_stderr(proc.stderr) or summarize_stderr(proc.stdout) or "no output"
        return {**base, "returncode": proc.returncode,
                "error": f"core probe exited with code {proc.returncode}: {head}"}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(info, dict):
                return {**base, **info, "returncode": 0}
    return {**base, "returncode": proc.returncode, "error": "core probe printed no JSON"}


def collect_machine_info() -> dict[str, Any]:
    """Collect the provenance dict stored in every benchmark row (all JSON types).

    ``env`` holds the thread-related variables of the *calling* process; the
    values the benchmark worker actually saw are stored per row by the runner
    (``child.env``).  ``load_average``/``collected_at`` describe the moment of
    collection; the runner records a per-row load average as well.
    """
    uname = platform.uname()
    load: list[float] | None
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "cpu": cpu_info(),
        "memory": memory_info(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "glsolver": _version_of("glsolver"),
            "numpy": _version_of("numpy"),
            "networkx": _version_of("networkx"),
            "scipy": _version_of("scipy"),
            "matplotlib": _version_of("matplotlib"),
        },
        "core": core_info(),
        "git": git_info(),
        "load_average": load,
        "env": {key: os.environ.get(key) for key in _THREAD_ENV},
        "collected_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m benchmarks.machine [--info]`` prints the collected dict as JSON."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in ("--info", "-i"):
        print(__doc__)
        return 2
    print(json.dumps(collect_machine_info(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
