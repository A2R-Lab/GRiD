#!/usr/bin/env bash
# Serial GPU competitive-baseline capture (mjx + frax) for the A1b competitive re-sweep.
# Runs ONE capture at a time (timing isolation) across the requested robots x bases.
# Usage: bash test/benchmarks/run_competitive_gpu_baselines.sh [outdir]
set -u
cd "$(dirname "$0")/../.."            # repo root
source .venv/bin/activate

OUTDIR="${1:-test/benchmarks/results/competitive_20260613}"
mkdir -p "$OUTDIR"
ROBOTS="${ROBOTS:-iiwa14 go2 g1}"
BASES="${BASES:-fixed floating}"
MASTER="$OUTDIR/capture.log"
echo "=== competitive GPU baseline capture $(date) ===" | tee "$MASTER"
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

for robot in $ROBOTS; do
  for base in $BASES; do
    run_one mjx  "$robot" "$base"
    run_one frax "$robot" "$base" --device both
  done
done
echo "=== done $(date) ===" | tee -a "$MASTER"
