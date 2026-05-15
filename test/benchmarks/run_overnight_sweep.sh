#!/bin/bash
# Overnight multi-version sweep for sm_120 (RTX 5070 Ti / RTX 3080-class).
#
# Designed to be approved once and left running unattended. Captures all
# stdout/stderr to a log file under the dated output dir so you can tail
# it from another terminal.
#
# Invoke (foreground; ctrl-Z + bg + disown if you want to leave the
# terminal):
#
#     ./test/benchmarks/run_overnight_sweep.sh
#
# Or background+detach in one shot:
#
#     nohup ./test/benchmarks/run_overnight_sweep.sh \
#         > /tmp/overnight_sweep.log 2>&1 &
#     disown
#
# Expected wall time: ~2-3 hours on a quiet machine (~80% of that is
# pinocchio per-algo subprocess JIT compiles).
#
# Pre-flight recommendations:
# - Close Zoom / Slack huddles / video calls. Pinocchio is CPU-bound;
#   contention skews its absolute numbers by 5-10×.
# - Optional: `sudo ./test/benchmarks/setCPU.sh` to lock CPU governor
#   to performance. Not required, but tightens variance.
# - Ensure ~500 MB free disk under .pytest_cache/grid_cuda/ for the
#   compiled binary cache.

set -uo pipefail   # NOT -e: we want to continue past individual
                   # (column, robot, base) failures and still produce a
                   # partial report.

# --- Discover repo root by chasing __file__ ---------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATE="$(date +%Y%m%d_%H%M)"
OUTDIR="test/benchmarks/results/comparison_sm120_overnight_${DATE}"
REPORT="test/benchmarks/benchmark_multi_version_sm120_overnight_${DATE}.md"
LOG="${OUTDIR}/run.log"

mkdir -p "$OUTDIR"

# Mirror everything to LOG + stdout. Process-substitution `tee` survives
# nohup and ctrl-Z/bg/disown.
exec > >(tee -a "$LOG") 2>&1

echo "==================================================================="
echo "Overnight multi-version sweep"
echo "==================================================================="
echo "Started:    $(date)"
echo "Repo root:  $REPO_ROOT"
echo "Output dir: $OUTDIR"
echo "Report:     $REPORT"
echo "Log:        $LOG"
echo "Host:       $(hostname)  user=$(whoami)"
echo "GPU:        $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo '(nvidia-smi unavailable)')"
echo "==================================================================="
echo

# Confirm Python + MathDx are usable before launching anything heavy.
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: $PYTHON not executable. Set PYTHON env var to a usable interpreter." >&2
    exit 1
fi

export MATHDX_ROOT="${MATHDX_ROOT:-/opt/nvidia/mathdx/25.12}"
if [[ ! -f "$MATHDX_ROOT/include/cublasdx.hpp" ]]; then
    echo "WARNING: $MATHDX_ROOT/include/cublasdx.hpp not found." >&2
    echo "         glass-nvidia column will be skipped by the pre-flight check." >&2
fi

# ---------------------------------------------------------------------
# Main sweep: 6 columns × 3 robots × 2 bases.
#
# Flags chosen for sm_120 correctness:
# - NO --no-licm-barrier: anti-LICM machinery must be active so single-
#   call timings reflect real work (sm_8x uses --cicc-opt-level 2 instead,
#   which preserves anti-LICM; sm_120 doesn't need that workaround).
# - NO --cicc-opt-level: sm_120 doesn't hang at cicc -O3, and -O3
#   produces measurably faster SASS than -O2 on this GPU.
# - --single-call-iters 50000 / --batch-iters 500: bumped from defaults
#   for tighter medians overnight.
# - MJX go2/g1 typically fail on current JAX/MJX pins (`_update_constraint`
#   rejects float-indexed array access). The harness reports the failure
#   per (column, robot, base) and continues. iiwa14 MJX usually works.
# ---------------------------------------------------------------------

echo
echo "=== Multi-version sweep ==="
"$PYTHON" test/benchmarks/run_multi_version.py \
    --columns glass glass_nvidia pre_glass pinocchio frax mjx \
    --robots iiwa14 go2 g1 \
    --bases fixed floating \
    --single-call-iters 50000 \
    --batch-iters 500 \
    --output-dir "$OUTDIR" \
    --report "$REPORT"
SWEEP_RC=$?
echo
echo "Multi-version sweep exit code: $SWEEP_RC"

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------
echo
echo "==================================================================="
echo "Sweep complete at $(date)"
echo "==================================================================="
JSON_COUNT=$(find "$OUTDIR" -name "*.json" -type f 2>/dev/null | wc -l)
echo "JSON files produced: $JSON_COUNT"
echo "Output dir:          $OUTDIR"
echo "Report:              $REPORT"
echo "Full log:            $LOG"

if [[ $JSON_COUNT -eq 0 ]]; then
    echo
    echo "WARNING: no JSON results were produced. Check the log above for"
    echo "         pre-flight or compile failures."
    exit 1
fi

# Exit success if we produced any results at all — the report itself
# will show which (column, robot, base) cells came back empty.
exit 0
