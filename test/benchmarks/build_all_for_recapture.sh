#!/usr/bin/env bash
# Pre-compile ALL binaries the clean competitive re-capture needs, into cache, so the
# timing run is pure --no-recompile (build phase != timing phase). RAM-safe serial
# (GRID_COMPILE_WORKERS=1; big-robot SO compiles are 24-36GB each). No timing here.
set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate
LOG=/tmp/build_all_recapture.log
echo "=== build-all start $(date) ===" | tee "$LOG"

ROBOTS="${ROBOTS:-iiwa14 go2 g1}"   # competitive set (H2+ has no competitors → skip here)
BASES="${BASES:-fixed floating}"
TIERS="${TIERS:-shared lite minimal}"  # all tiers so the re-capture picks best per algo

# 1) GRiD C++ harness binaries (full algo set, N=1024 already in source), per tier.
for robot in $ROBOTS; do
  for base in $BASES; do
    for tier in $TIERS; do
      echo "--- GRiD compile $robot $base tier=$tier $(date)" | tee -a "$LOG"
      t0=$SECONDS
      GRID_COMPILE_WORKERS=1 python test/benchmarks/baselines/grid/run.py \
          --robot "$robot" --base "$base" --tier "$tier" --compile-only --compile-workers 1 \
          >> "$LOG" 2>&1
      echo "    rc=$? ($((SECONDS-t0))s)" | tee -a "$LOG"
    done
  done
done

# 2) Pinocchio binary (fast; codegen JIT happens at capture time, but warm the .exe build).
for robot in $ROBOTS; do
  echo "--- pin compile $robot $(date)" | tee -a "$LOG"
  python test/benchmarks/baselines/pinocchio/run.py --robot "$robot" --base fixed \
      --algos crba --output /tmp/_pinwarm_${robot}.json >> "$LOG" 2>&1 || true
done

echo "=== build-all GRiD+pin done $(date) ===" | tee -a "$LOG"
echo "NOTE: grid_rbd bindings (wrapper column) build separately — see build_bindings step." | tee -a "$LOG"
