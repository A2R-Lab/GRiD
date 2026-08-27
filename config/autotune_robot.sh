#!/bin/bash
# Autotune GRiD launch configs for YOUR robot + GPU, then write a
# config/launch_configs/<robot>/<gpu>.json override that codegen bakes into
# grid_launch_config.cuh (A1 launch-config feature).
#
# WHAT IT DOES
#   1. Detects your GPU (name + compute capability) and derives the GPU key
#      <model>_<arch>, e.g. rtx5090_sm120.
#   2. Runs the RAM-SAFE autotune sweep (per_algo_bench.py --mode autotune
#      --stage sweep) for the given robot + bases, single-call timing OFF
#      (B8 default; it's hard to time and needs the rdc shim — batch N=256
#      timing is what we tune on). Serial build (--compile-jobs 1, one
#      per-algo TU at a time) so the big-robot SO TUs never OOM the box.
#   3. Converts the swept winners into config/launch_configs/<robot>/<gpu>.json in the
#      documented schema (via config/autotune_to_launch_config.py).
#   4. Prints next steps: re-codegen + rebuild to pick up the values, and how to
#      PR the JSON to crowdsource the matrix.
#
# USAGE
#   bash config/autotune_robot.sh <robot> [base ...]
#       <robot>     robot id (iiwa14 | go2 | g1 | h2_plus | ...; must be a codegen robot)
#       [base ...]  one or more of: fixed floating   (default: fixed floating)
#
#   GPU_KEY=<model>_<arch>   override GPU key if auto-detection fails (e.g. a40_sm86)
#   AUTOTUNE_N=<int>         batch size to tune at (default 256)
#   AUTOTUNE_DRY_RUN=1       skip the actual sweep (detection + plumbing check only)
#
# EXAMPLES
#   bash config/autotune_robot.sh iiwa14                 # fixed + floating
#   bash config/autotune_robot.sh go2 floating           # floating only
#   GPU_KEY=a40_sm86 bash config/autotune_robot.sh g1     # forced GPU key
#
# Run on a QUIET GPU — timing must be isolated (close other GPU workloads).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PATH=/usr/local/cuda/bin:$PATH

# Pick a python: prefer the repo venv, fall back to python3.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
[ -n "$PY" ] || { echo "ERROR: no python found (.venv/bin/python or python3)." >&2; exit 1; }

# ---- args ----------------------------------------------------------------
if [ "$#" -lt 1 ]; then
  echo "Usage: bash config/autotune_robot.sh <robot> [fixed floating]" >&2
  echo "  e.g. bash config/autotune_robot.sh iiwa14 fixed floating" >&2
  exit 2
fi
ROBOT="$1"; shift
BASES=("$@")
[ "${#BASES[@]}" -gt 0 ] || BASES=(fixed floating)
for b in "${BASES[@]}"; do
  case "$b" in
    fixed|floating) ;;
    *) echo "ERROR: base '$b' must be 'fixed' or 'floating'." >&2; exit 2 ;;
  esac
done

AUTOTUNE_N="${AUTOTUNE_N:-256}"

# ---- GPU detection -------------------------------------------------------
# gpu_name: full nvidia-smi name (kept verbatim in the JSON).
# arch:     compute_cap "12.0" -> "120" -> sm_120.
# model:    name lowercased, "nvidia"/"geforce" dropped, non-alnum stripped
#           -> "NVIDIA GeForce RTX 5090" -> "rtx5090".
# GPU_KEY = <model>_sm<arch>. Override with the GPU_KEY env var if this fails.
detect_gpu() {
  local smi name cc arch model
  smi="$(command -v nvidia-smi || true)"
  if [ -z "$smi" ]; then
    echo "ERROR: nvidia-smi not found — cannot auto-detect the GPU." >&2
    echo "       Set GPU_KEY=<model>_<arch> (e.g. GPU_KEY=rtx5090_sm120) and re-run." >&2
    return 1
  fi
  name="$("$smi" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/[[:space:]]*$//')"
  cc="$("$smi" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')"
  if [ -z "$name" ] || [ -z "$cc" ]; then
    echo "ERROR: GPU name/compute_cap query failed (name='$name' cc='$cc')." >&2
    echo "       Set GPU_KEY=<model>_<arch> (e.g. GPU_KEY=rtx5090_sm120) and re-run." >&2
    return 1
  fi
  GPU_NAME="$name"
  local ccd="${cc//./}"          # "12.0" -> "120"
  CUDA_ARCH="sm_${ccd}"          # cuda_arch field:  sm_120 (with underscore)
  # model key: lowercase, drop vendor words, keep [a-z0-9].
  model="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
  model="${model//nvidia/}"
  model="${model//geforce/}"
  model="$(printf '%s' "$model" | tr -cd 'a-z0-9')"
  [ -n "$model" ] || model="gpu"
  DETECTED_KEY="${model}_sm${ccd}"   # GPU key / filename: rtx5090_sm120 (no _)
  return 0
}

GPU_NAME=""
CUDA_ARCH=""
DETECTED_KEY=""
if [ -n "${GPU_KEY:-}" ]; then
  # User forced a key — it is authoritative for the filename AND cuda_arch
  # (derived from the key's _smXXX suffix, e.g. rtx5090_sm120 -> sm_120). We
  # still try detection only to fill in a human gpu_name for the JSON.
  detect_gpu || true
  if [[ "$GPU_KEY" == *_sm* ]]; then
    CUDA_ARCH="sm_${GPU_KEY##*_sm}"
  else
    CUDA_ARCH="${CUDA_ARCH:-sm_unknown}"
  fi
  [ -n "$GPU_NAME" ] || GPU_NAME="$GPU_KEY"
else
  detect_gpu || exit 1
  GPU_KEY="$DETECTED_KEY"
fi

echo "=== autotune_robot: robot=$ROBOT bases=[${BASES[*]}] ==="
echo "    GPU_KEY   = $GPU_KEY"
echo "    gpu_name  = $GPU_NAME"
echo "    cuda_arch = $CUDA_ARCH"
echo "    autotune_N= $AUTOTUNE_N"
echo "    output    = config/launch_configs/$ROBOT/$GPU_KEY.json"

HOST="$(python3 -c 'import platform;print(platform.node().replace(" ","_"))' 2>/dev/null || hostname)"
BEST_FILE="$REPO_ROOT/test/benchmarks/results/autotune_best_${HOST}.json"

# ---- run the autotune sweep (RAM-safe serial; single-timing OFF) ---------
# One per_algo_bench.py invocation per base (the per-exe cutover: small per-algo TUs, one exe/process,
# crash-isolated -- replaces the monolithic run.py path). Each writes a per-cell *_grid_glass.json with
# algo_picks into $SWEEPDIR; sweep_to_autotune_best.py then merges just THIS run's valid picks into a
# clean autotune_best_<host>.json (no stale-entry accumulation). --compile-jobs 1 => one TU at a time.
SWEEPDIR="$REPO_ROOT/test/benchmarks/results/autotune_sweep_${ROBOT}"
mkdir -p "$SWEEPDIR"
if [ "${AUTOTUNE_DRY_RUN:-0}" = "1" ]; then
  echo "=== AUTOTUNE_DRY_RUN=1: skipping the real sweep (no nvcc build) ==="
else
  for b in "${BASES[@]}"; do
    echo "=== [$ROBOT/$b] autotune sweep START $(date)  mem_avail=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}')MB ==="
    "$PY" test/benchmarks/per_algo_bench.py \
        --robot "$ROBOT" --base "$b" \
        --mode autotune --stage sweep --autotune-N "$AUTOTUNE_N" \
        --compile-jobs 1 \
        --output "$SWEEPDIR/${ROBOT}_${b}_grid_glass.json" \
      || { echo "ERROR: autotune sweep failed for $ROBOT/$b." >&2; exit 1; }
    echo "=== [$ROBOT/$b] autotune sweep DONE $(date) ==="
  done

  # Merge this run's per-cell picks -> a fresh autotune_best_<host>.json for the bake step.
  "$PY" config/sweep_to_autotune_best.py --sweep-dir "$SWEEPDIR" --out "$BEST_FILE" \
    || { echo "ERROR: autotune_best merge failed." >&2; exit 1; }

  if [ ! -f "$BEST_FILE" ]; then
    echo "ERROR: expected autotune artifact not found: $BEST_FILE" >&2
    echo "       The sweep produced no picks. Check the output above." >&2
    exit 1
  fi
fi

# ---- convert winners -> config/launch_configs/<robot>/<gpu>.json ----------------
if [ "${AUTOTUNE_DRY_RUN:-0}" = "1" ] && [ ! -f "$BEST_FILE" ]; then
  echo "=== AUTOTUNE_DRY_RUN: no autotune_best file to convert (expected in a dry run) ==="
else
  "$PY" config/autotune_to_launch_config.py \
      --robot "$ROBOT" --bases "${BASES[@]}" \
      --gpu-key "$GPU_KEY" --cuda-arch "$CUDA_ARCH" \
      --gpu-name "$GPU_NAME" --autotune-N "$AUTOTUNE_N" \
      --best "$BEST_FILE" \
    || { echo "ERROR: launch_config conversion failed." >&2; exit 1; }
fi

# ---- next steps ----------------------------------------------------------
cat <<EOF

=== DONE: config/launch_configs/$ROBOT/$GPU_KEY.json written ===

Next steps:
  1. Re-run codegen + rebuild so the host launchers pick up your tuned values
     (codegen bakes config/launch_configs/<robot>/<gpu>.json into grid_launch_config.cuh):

         grid-generate <urdf> ...      # your usual codegen for $ROBOT
         # then rebuild your GRiD / bindings as usual

  2. (Optional, please do!) PR the JSON to crowdsource the matrix:
         git add config/launch_configs/$ROBOT/$GPU_KEY.json
     See config/launch_configs/README.md for the contribution checklist (GPU model,
     driver/CUDA version, robot DoF/base in the PR description).

Note: single-call timing is OFF by default (B8) — it is hard to time and needs
the rdc shim. The autotune tunes on batch (N=$AUTOTUNE_N) timing, which is what
the host launch config should optimize for.
EOF
