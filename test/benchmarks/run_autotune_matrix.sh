#!/bin/bash
# Autotune-matrix sweep wrapper (A1a Phase 1). Chains:
#   1. quiet-GPU precheck (warn if other CUDA processes are running — timing
#      must be ISOLATED; see docs/agent_debugging_guide.md §7/§8).
#   2. run_multi_version.py --autotune-threads --tiers shared lite minimal
#      (parallel BUILD then SERIAL MEASURE; emits results/autotune_best_<host>.json
#      + per-(robot,base) JSON with schema-2 algo_picks).
#   3. collect_kernel_limits.py per (robot, base)  — deterministic, read-only.
#   4. build_autotune_matrix.py  — join + §1c guard -> autotune_matrix_<host>.json.
#
# MEASUREMENT-ONLY: no codegen/kernel/binding edits anywhere in this chain.
#
# Invoke:
#     ./test/benchmarks/run_autotune_matrix.sh [robot ...]
# Default robots: iiwa14 go2 g1  (h2_plus omitted by default — its 2nd-order
# kernels compile 20-40 min each; add it explicitly for an overnight run).
#
# Background+detach:
#     nohup ./test/benchmarks/run_autotune_matrix.sh > /tmp/autotune_matrix.log 2>&1 &
#     disown

set -uo pipefail   # NOT -e: continue past individual cell failures.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export PATH="/usr/local/cuda/bin:$PATH"

ROBOTS=("$@")
if [[ ${#ROBOTS[@]} -eq 0 ]]; then
    ROBOTS=(iiwa14 go2 g1)
fi
BASES=(fixed floating)

DATE="$(date +%Y%m%d_%H%M)"
OUTDIR="test/benchmarks/results/autotune_matrix_${DATE}"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "==================================================================="
echo "Autotune-matrix sweep (A1a Phase 1)"
echo "==================================================================="
echo "Started:  $(date)"
echo "Robots:   ${ROBOTS[*]}"
echo "Bases:    ${BASES[*]}"
echo "Host:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo '(nvidia-smi unavailable)')"
echo "Output:   $OUTDIR"
echo "==================================================================="

if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: $PYTHON not executable. Set PYTHON env var." >&2
    exit 1
fi

# --- 1) quiet-GPU precheck (timing must be isolated) ------------------
echo
echo "=== quiet-GPU precheck ==="
OTHER_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
if [[ -n "$OTHER_PROCS" ]]; then
    echo "WARNING: other CUDA processes are running — timing will be skewed:"
    echo "$OTHER_PROCS"
    echo "(Proceeding anyway; for a real measurement run, quiesce the GPU first.)"
else
    echo "GPU quiet (no other compute apps)."
fi

# --- 2) sweep: parallel BUILD + serial MEASURE, all tiers -------------
echo
echo "=== sweep (run_multi_version.py --autotune-threads --tiers shared lite minimal) ==="
"$PYTHON" test/benchmarks/run_multi_version.py \
    --columns glass \
    --robots "${ROBOTS[@]}" \
    --bases "${BASES[@]}" \
    --autotune-threads \
    --tiers shared lite minimal \
    --output-dir "$OUTDIR"
echo "sweep exit code: $?"

# --- 3) collect kernel limits (read-only, correctness-class) ----------
echo
echo "=== collect_kernel_limits.py ==="
for robot in "${ROBOTS[@]}"; do
    for base in "${BASES[@]}"; do
        echo "--- limits: $robot $base ---"
        "$PYTHON" test/benchmarks/collect_kernel_limits.py \
            --robot "$robot" --base "$base" --no-recompile \
            || echo "WARN: limits failed for $robot $base"
    done
done

# --- 4) build the matrix (join + §1c guard) ---------------------------
echo
echo "=== build_autotune_matrix.py ==="
"$PYTHON" test/benchmarks/build_autotune_matrix.py \
    --picks "$OUTDIR"/*_grid*.json "$OUTDIR"/*_glass*.json 2>/dev/null \
    || "$PYTHON" test/benchmarks/build_autotune_matrix.py

echo
echo "==================================================================="
echo "Done at $(date). Matrix: test/benchmarks/results/autotune_matrix_$(hostname).json"
echo "Log: $LOG"
echo "==================================================================="
