#!/usr/bin/env bash
# Refresh the LIVE launch_configs/<robot>/<gpu>.json picks from a tier-sweep, WITHOUT
# the day-long recompile. Pipeline (see launch_configs/README.md):
#
#   sweep (cached, split) ── host bases ──> harvest ──> bake ──> [ffi] ──> diff
#
#   stage 0 (optional, GPU/overnight): a cached-binary host autotune sweep that writes
#           results/tier_sweep_phased_<TS>/.../*_grid_glass.json. Run it yourself with
#           `bash test/benchmarks/run_tier_sweep_phased.sh overnight 4` (it cache-HITS
#           the content-addressed binaries -> NO recompile) and pass --sweep-dir, OR let
#           this script pick the latest tier_sweep_phased_* dir.
#   stage 1: tools/sweep_to_autotune_best.py   -> ONE clean autotune_best (this run only).
#   stage 2: tools/autotune_to_launch_config.py -> bake `bases` per robot (symbol->short
#            key, per-algo merge, preserves ffi_bases). GPU-free.
#   stage 3 (--with-ffi, GPU/overnight): test/benchmarks/autotune_ffi.py -> `ffi_bases`.
#            Big robots (g1/h2_plus) get a non-SO .so subset (SO ffi falls back to host).
#   stage 4: print git diff of launch_configs/ (NEVER commits).
#
# WHY NOT tools/autotune_robot.sh: it builds a FULL monolithic per-tier binary, so g1/h2_plus
# hit the ~hour SO recompile (only the SPLIT noSO/SO binaries are cached). The sweep above
# reuses those cached split binaries.
#
# Usage:
#   bash tools/refresh_launch_configs.sh                      # latest sweep, bake bases only
#   bash tools/refresh_launch_configs.sh --sweep-dir DIR      # bake from a specific sweep
#   bash tools/refresh_launch_configs.sh --with-ffi           # also (re)tune ffi_bases (GPU)
#   DRY_RUN=1 bash tools/refresh_launch_configs.sh --with-ffi # print commands only
set -uo pipefail
REPO_ROOT="/home/plancher/Desktop/GRiD"
cd "$REPO_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
PY="$REPO_ROOT/.venv/bin/python"

DRY_RUN="${DRY_RUN:-0}"
WITH_FFI=0
SWEEP_DIR=""
GPU_KEY="${GPU_KEY:-rtx5090_sm120}"
CUDA_ARCH="${CUDA_ARCH:-sm_120}"
GPU_NAME="${GPU_NAME:-NVIDIA GeForce RTX 5090}"
AUTOTUNE_N="${AUTOTUNE_N:-256}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-ffi) WITH_FFI=1; shift ;;
    --sweep-dir) SWEEP_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$SWEEP_DIR" ] || SWEEP_DIR="$(ls -dt test/benchmarks/results/tier_sweep_phased_* 2>/dev/null | head -1)"
[ -n "$SWEEP_DIR" ] && [ -d "$SWEEP_DIR" ] || { echo "ERROR: no sweep dir (pass --sweep-dir)" >&2; exit 1; }

# big robots whose grid_rbd ffi .so must EXCLUDE the SO kernels (else day-long nvcc).
is_big() { case "$1" in g1|h2_plus) return 0 ;; *) return 1 ;; esac }
NONSO_SYMBOLS=(inverse_dynamics minv forward_dynamics aba crba \
  inverse_dynamics_gradient forward_dynamics_gradient \
  end_effector_pose end_effector_pose_gradient end_effector_pose_hessian \
  integrator integrator_gradient integrator_with_gradient)

run() { echo "+ $*"; [ "$DRY_RUN" = "1" ] && return 0; "$@"; }

TS="$(date +%Y%m%d_%H%M)"
BEST="$SWEEP_DIR/autotune_best_refresh_${TS}.json"
echo "########## refresh_launch_configs  sweep=$SWEEP_DIR  with_ffi=$WITH_FFI  $(date) ##########"

# stage 1: harvest this sweep's picks into ONE clean autotune_best
# (always runs even under DRY_RUN — GPU-free, and the bake/ffi discovery needs it)
echo "+ $PY tools/sweep_to_autotune_best.py --sweep-dir $SWEEP_DIR --out $BEST"
"$PY" tools/sweep_to_autotune_best.py --sweep-dir "$SWEEP_DIR" --out "$BEST" || exit 1

# discover (robot, base) pairs present in the harvested best
mapfile -t PAIRS < <("$PY" - "$BEST" <<'PYEOF'
import json,sys
best=json.load(open(sys.argv[1]))["best"]
for r in sorted(best):
    for b in sorted(best[r]):
        print(f"{r} {b}")
PYEOF
)
[ "${#PAIRS[@]}" -gt 0 ] || { echo "ERROR: no robots harvested from $SWEEP_DIR" >&2; exit 1; }

# stage 2: bake host `bases` per (robot, base)  [GPU-free]
for p in "${PAIRS[@]}"; do
  set -- $p; r="$1"; b="$2"
  echo "--- [$r/$b] bake bases ---"
  run "$PY" tools/autotune_to_launch_config.py --robot "$r" --bases "$b" \
      --gpu-key "$GPU_KEY" --cuda-arch "$CUDA_ARCH" --gpu-name "$GPU_NAME" \
      --autotune-N "$AUTOTUNE_N" --best "$BEST"
done

# stage 3: ffi_bases  [GPU + .so compile; opt-in]
if [ "$WITH_FFI" = "1" ]; then
  for p in "${PAIRS[@]}"; do
    set -- $p; r="$1"; b="$2"
    echo "--- [$r/$b] ffi_bases ---"
    if is_big "$r"; then
      run "$PY" test/benchmarks/autotune_ffi.py --robot "$r" --base "$b" \
          --build-algos "${NONSO_SYMBOLS[@]}"
    else
      run "$PY" test/benchmarks/autotune_ffi.py --robot "$r" --base "$b"
    fi
  done
fi

echo "########## DONE $(date) — review diffs, commit manually ##########"
git -C "$REPO_ROOT" status -s launch_configs/
git -C "$REPO_ROOT" diff --stat launch_configs/
