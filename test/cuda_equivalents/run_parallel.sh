#!/bin/bash
# C.2: parallel CUDA equivalence run, sized by free RAM (~5 GB / compile).
# Per memory `feedback_parallel_equivalence_testing.md`: these tests are
# correctness-only + per-(robot,base) parallel; safe to run concurrent.
#
# Usage:
#   ./test/cuda_equivalents/run_parallel.sh                    # default: smoke tier
#   ./test/cuda_equivalents/run_parallel.sh -k iiwa14          # filter to one robot
#   GRID_PARALLEL_GB_PER_JOB=4 ./test/cuda_equivalents/run_parallel.sh
#
# Auto-sizing logic:
#   N = min(physical_cores, max(1, floor(free_gb / GB_PER_JOB)))
# Default GB_PER_JOB = 5 (matches typical nvcc + cudafe peak per equivalence runner).

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="/usr/local/cuda/bin:$(pwd)/.venv/bin:${PATH}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

GB_PER_JOB="${GRID_PARALLEL_GB_PER_JOB:-5}"
CORES=$(nproc --all)
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
JOBS=$(( FREE_GB / GB_PER_JOB ))
if (( JOBS < 1 )); then JOBS=1; fi
if (( JOBS > CORES )); then JOBS=$CORES; fi

echo "[$(date +%H:%M:%S)] parallel equivalence: free=${FREE_GB}G cores=${CORES} GB/job=${GB_PER_JOB} -> -n ${JOBS}"

# pytest-xdist auto-shards by test case. The CUDA tests are parametrized over
# (robot, base, thread_count) so xdist gets per-(robot, base) parallelism for
# free.  We use loadgroup balancing so each worker stays warm across multiple
# thread-count variants of the same robot (header cache hit).
exec pytest \
    -p xdist \
    -n "$JOBS" \
    --dist loadgroup \
    -q --no-header --tb=short \
    test/cuda_equivalents/test_cuda_executable_equivalence.py \
    "$@"
