#!/bin/bash
# A1b autotune + competitive sweep — CANONICAL launch.
#
# RAM SAFETY (learned 2026-06-12): the SO-kernel compiles (idsva_so / fdsva_so on
# g1 / h1_2) are now ~6-7 GB of cicc EACH. The default auto build-jobs (~8 parallel)
# spikes RAM toward the 62 GB limit → crashes VS Code → tears down the snap.code
# cgroup → kills the sweep. So CAP --build-jobs at 4 (≈28 GB peak; bump to 6 ≈42 GB
# only if RAM headroom looks comfortable). It did NOT crash with the older/smaller
# compiles — only the large SO TUs at full 8-way.
#
# NOTE: build/measure runs inside the VS Code snap cgroup, so a VS Code crash kills
# the sweep. Best run when the desktop is idle. The run.py content-addressed cache
# (.pytest_cache/grid_cuda) persists across restarts, so if it dies and you relaunch
# WITHOUT changing codegen (.py) or the bench .cu files, it resumes from cached cells.
#
# Usage:   bash test/benchmarks/run_a1b_sweep.sh [BUILD_JOBS]   (default 4)
set -uo pipefail
cd /home/plancher/Desktop/GRiD
export PATH=/usr/local/cuda/bin:$PATH
BUILD_JOBS="${1:-4}"
# A1B_ROBOTS env overrides the robot set. NOTE: h1_2's SO monolithic compile uses
# ~36 GB RAM (single cicc) — exclude it unless build-jobs=1 + the box has the RAM,
# or until the subset-timing harness (B7) lets us build it dynamics-only.
ROBOTS="${A1B_ROBOTS:-iiwa14 go2 g1 h1_2}"
OUTDIR="test/benchmarks/results/a1b_sweep_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTDIR"

echo "=== A1b sweep START $(date)  build_jobs=$BUILD_JOBS  robots=[$ROBOTS]  outdir=$OUTDIR ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null)"

.venv/bin/python test/benchmarks/run_multi_version.py \
    --columns glass pinocchio \
    --robots $ROBOTS \
    --bases fixed floating \
    --autotune-threads \
    --tiers shared lite minimal \
    --build-jobs "$BUILD_JOBS" \
    --output-dir "$OUTDIR"
echo "=== sweep exit code: $? at $(date) ==="

echo "=== collect_kernel_limits ==="
for r in $ROBOTS; do for b in fixed floating; do
  .venv/bin/python test/benchmarks/collect_kernel_limits.py --robot "$r" --base "$b" --no-recompile || echo "WARN limits $r $b"
done; done

echo "=== build_autotune_matrix ==="
.venv/bin/python test/benchmarks/build_autotune_matrix.py --picks "$OUTDIR"/*_grid*.json "$OUTDIR"/*_glass*.json 2>/dev/null \
  || .venv/bin/python test/benchmarks/build_autotune_matrix.py

echo "=== A1B_PIPELINE_DONE $(date)  outdir=$OUTDIR ==="
