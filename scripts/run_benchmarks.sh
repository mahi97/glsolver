#!/usr/bin/env bash
# Run the benchmark suite end to end: smoke first (fails fast), then the
# scaling / families / imbalance / directed / weighted / dag /
# connectivity_margin grids, then the figures and the verification dashboard.
#
#   scripts/run_benchmarks.sh                  # everything, default settings
#   CONFIGS="smoke scaling" scripts/run_benchmarks.sh --repeat 5 --threads 1
#   QUICK=1 scripts/run_benchmarks.sh          # --limit 4 instances per config (sanity check)
#
# Extra arguments are passed to every `benchmarks.runner` invocation
# (e.g. --repeat, --threads, --limit, --resume, --fresh, --algorithms).
# Results: benchmarks/results/<config>.jsonl, plots in benchmarks/results/plots/,
# tables in benchmarks/results/summary.md, dashboard in benchmarks/results/dashboard.html.
#
# Gate: the runner exits 3 when a partition was INVALID and 2 when any pair
# crashed or timed out (0 otherwise; skipped pairs and precondition failures are
# not failures).  The smoke grid must exit 0 -- every executed pair ok and VALID
# -- or this script stops with that code before anything expensive starts.  For
# the other configs a non-zero code is reported and the run continues.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PY=${PYTHON:-python}
CONFIGS=${CONFIGS:-"smoke scaling families imbalance directed weighted dag connectivity_margin"}
EXTRA=("$@")
if [ "${QUICK:-0}" = "1" ]; then
  EXTRA+=(--limit 4)
fi

# reproducible, quiet numerics; one solver thread unless the caller overrides it
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export MPLBACKEND=Agg

echo "== machine =="
$PY -m benchmarks.machine --info

for cfg in $CONFIGS; do
  echo
  echo "== config: $cfg =="
  if [ "$cfg" = "smoke" ]; then
    # the smoke grid must be green (exit 0: no INVALID, error or timeout rows)
    # before anything expensive starts
    rc=0
    $PY -m benchmarks.runner smoke --fresh "${EXTRA[@]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
      case "$rc" in
        3) echo "smoke benchmark produced INVALID partitions (exit $rc); aborting" >&2 ;;
        2) echo "smoke benchmark had crashed or timed-out pairs (exit $rc); aborting" >&2 ;;
        *) echo "smoke benchmark failed (exit $rc); aborting" >&2 ;;
      esac
      exit "$rc"
    fi
  else
    rc=0
    $PY -m benchmarks.runner "$cfg" --resume "${EXTRA[@]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "warning: config $cfg exited $rc (3 = INVALID partitions, 2 = crashed/timed-out pairs; see dashboard)" >&2
    fi
  fi
done

echo
echo "== plots =="
$PY -m benchmarks.plot
echo
echo "== report =="
$PY -m benchmarks.report || true
echo "dashboard: benchmarks/results/dashboard.html"
