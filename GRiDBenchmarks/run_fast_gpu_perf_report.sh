#!/usr/bin/env bash
set -euo pipefail

# One-shot performance collection for a faster GPU machine.
#
# This script is intentionally not part of the overnight correctness run on the
# current slow laptop. Run it later on a stable fast workstation after the CUDA
# correctness sweep is green.
#
# Example:
#   bash GRiDBenchmarks/run_fast_gpu_perf_report.sh /path/to/robot.urdf fixed iiwa14 all auto float --save
#   bash GRiDBenchmarks/run_fast_gpu_perf_report.sh /path/to/robot.urdf floating g1 all GRID_SPILL_DA_DF_OUTPUT float --save

if [[ $# -lt 6 ]]; then
    cat >&2 <<'USAGE'
Usage:
  bash GRiDBenchmarks/run_fast_gpu_perf_report.sh URDF_PATH BASE_MODE ROBOT PROFILE FALLBACK_TIER PRECISION [--save]

BASE_MODE: fixed | floating
ROBOT: robot label for the baseline key, e.g. iiwa14, go2, g1
PROFILE: codegen profile label, e.g. all, dynamics, dynamics-gradients
FALLBACK_TIER: auto, GRID_SHARED_FULL, GRID_SPILL_DA_DF_OUTPUT, etc.
PRECISION: float | double

Set GRID_CUDA_ARCH if the fast machine is not sm_86 and timeGRiD.py has not
yet been taught to auto-detect the arch.
USAGE
    exit 2
fi

URDF_PATH="$1"
BASE_MODE="$2"
ROBOT="$3"
PROFILE="$4"
FALLBACK_TIER="$5"
PRECISION="$6"
SAVE_FLAG="${7:-}"

FLOATING_ARG=()
if [[ "${BASE_MODE}" == "floating" ]]; then
    FLOATING_ARG=(-f)
elif [[ "${BASE_MODE}" != "fixed" ]]; then
    echo "BASE_MODE must be fixed or floating, got: ${BASE_MODE}" >&2
    exit 2
fi

REPORT_ARGS=(
    .venv/bin/python
    GRiDBenchmarks/perf_regression_report.py
    --robot "${ROBOT}"
    --base-mode "${BASE_MODE}"
    --profile "${PROFILE}"
    --fallback-tier "${FALLBACK_TIER}"
    --precision "${PRECISION}"
)

if [[ "${SAVE_FLAG}" == "--save" ]]; then
    REPORT_ARGS+=(--save)
elif [[ -n "${SAVE_FLAG}" ]]; then
    echo "unknown optional argument: ${SAVE_FLAG}" >&2
    exit 2
fi

"${REPORT_ARGS[@]}" -- .venv/bin/python GRiDBenchmarks/timeGRiD.py "${URDF_PATH}" "${FLOATING_ARG[@]}"
