#!/bin/bash
# A1b OVERNIGHT serial sweep — RAM-SAFE for big robots (g1/h1_2).
#
# Big-robot SO monolithic TUs (single_main + batch_main) use ~24-36 GB cicc EACH.
# GRID_COMPILE_WORKERS=1 + --build-jobs 1 → exactly ONE TU compiles at a time
# (~36 GB peak = safe on a 62 GB box). Slow but unattended-safe.
#
# Per-robot sequential into ONE $ROOT so partial progress is preserved: if it
# doesn't reach h1_2, iiwa14/go2/g1 picks are already written + the matrix builds
# from whatever landed. Content-addressed cache (.pytest_cache/grid_cuda) means
# already-built cells are instant on (re)run.
set -uo pipefail
cd /home/plancher/Desktop/GRiD
export PATH=/usr/local/cuda/bin:$PATH
export GRID_COMPILE_WORKERS=1
ROBOTS="${A1B_ROBOTS:-iiwa14 go2 g1 h1_2}"
ROOT="test/benchmarks/results/a1b_overnight_$(date +%Y%m%d_%H%M)"
mkdir -p "$ROOT"

echo "=== A1b OVERNIGHT START $(date)  robots=[$ROBOTS]  GRID_COMPILE_WORKERS=1 build_jobs=1  root=$ROOT ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null)"

for r in $ROBOTS; do
  echo "=== [$r] START $(date)  mem_avail=$(free -m | awk '/^Mem:/{print $7}')MB ==="
  .venv/bin/python test/benchmarks/run_multi_version.py \
      --columns glass pinocchio \
      --robots "$r" --bases fixed floating \
      --autotune-threads --tiers shared lite minimal \
      --build-jobs 1 \
      --output-dir "$ROOT" || echo "WARN: sweep failed for $r (continuing)"
  echo "=== [$r] DONE $(date)  picks now: $(ls "$ROOT"/*_grid*.json 2>/dev/null | wc -l) ==="
done

echo "=== collect_kernel_limits ==="
for r in $ROBOTS; do for b in fixed floating; do
  .venv/bin/python test/benchmarks/collect_kernel_limits.py --robot "$r" --base "$b" --no-recompile || echo "WARN limits $r $b"
done; done

echo "=== build_autotune_matrix ==="
.venv/bin/python test/benchmarks/build_autotune_matrix.py --picks "$ROOT"/*_grid*.json "$ROOT"/*_glass*.json 2>/dev/null \
  || .venv/bin/python test/benchmarks/build_autotune_matrix.py

echo "=== A1B_OVERNIGHT_DONE $(date)  root=$ROOT ==="
