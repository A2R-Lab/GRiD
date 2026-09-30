#!/usr/bin/env bash
# Serial competitive-baseline capture for the A1b competitive re-sweep.
# Baselines: pinocchio (CODEGEN/cppADCodeGen, the fast CPU path) + the GPU libs
# mjx, frax, mujoco_warp, and cuRobo. Runs ONE capture at a time (timing isolation)
# across the requested robots x bases. Adapters already sweep N in {16,32,64,128,256,1024}.
# Usage: bash test/benchmarks/run_competitive_gpu_baselines.sh [outdir]
set -u
cd "$(dirname "$0")/../.."            # repo root
source .venv/bin/activate

OUTDIR="${1:-test/benchmarks/results/competitive_20260613}"
mkdir -p "$OUTDIR"
ROBOTS="${ROBOTS:-iiwa14 go2 g1}"
BASES="${BASES:-fixed floating}"
# cuRobo ships only the g1 config (fixed-base); restrict to avoid argparse choices errors.
CUROBO_ROBOTS="${CUROBO_ROBOTS:-g1}"
CUROBO_BASES="${CUROBO_BASES:-fixed}"
MASTER="$OUTDIR/capture.log"
echo "=== competitive baseline capture $(date) ===" | tee "$MASTER"
echo "outdir=$OUTDIR robots=[$ROBOTS] bases=[$BASES]" | tee -a "$MASTER"

run_one() {                          # baseline robot base extra-args...
  local bl="$1" robot="$2" base="$3"; shift 3
  local out="$OUTDIR/${robot}_${base}_${bl}.json"
  local log="$OUTDIR/${robot}_${base}_${bl}.log"
  echo "--- [$bl] $robot $base -> $out" | tee -a "$MASTER"
  local t0=$SECONDS
  python "test/benchmarks/baselines/${bl}/run.py" --robot "$robot" --base "$base" \
      --output "$out" "$@" > "$log" 2>&1
  local rc=$?
  echo "    rc=$rc  ($((SECONDS-t0))s)" | tee -a "$MASTER"
  [ $rc -ne 0 ] && echo "    !! see $log" | tee -a "$MASTER"
}

_in_list() { case " $2 " in *" $1 "*) return 0;; *) return 1;; esac; }

for robot in $ROBOTS; do
  for base in $BASES; do
    run_one pinocchio   "$robot" "$base"               # CPU codegen (fast path)
    run_one mjx         "$robot" "$base"
    run_one frax        "$robot" "$base" --device both
    run_one mujoco_warp "$robot" "$base"
    if _in_list "$robot" "$CUROBO_ROBOTS" && _in_list "$base" "$CUROBO_BASES"; then
      run_one curobo    "$robot" "$base"
    fi
  done
done
echo "=== done $(date) ===" | tee -a "$MASTER"
