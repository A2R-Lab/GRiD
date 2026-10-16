#!/usr/bin/env bash
# Generate the iiwa14 header, build fused_control_step.cu with nvcc, check that
# the fused and layered paths agree, and time both.
#
#   bash examples/cuda/fused_control_step/build_and_time.sh [--batch N] [--repeats R]
#        [--threads T] [--json PATH] [--header PATH]
#
# --json PATH writes {"example": "fused_control_step", "batch": N, "fused_us": x,
# "layered_us": y, "repeats": R, "arch": "sm_xx"}; the timings are medians of
# cudaEvent-timed repeats after warm-up. The exit status is non-zero when the
# correctness check fails. The CUDA architecture comes from GRID_CUDA_ARCH
# ("120", "12.0", "sm_120" or "compute_120"), else from nvidia-smi, else the
# generator's default of 86. --header PATH skips the Python generation step and
# builds against that header (nvcc is then the only tool the script runs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
EX_DIR="examples/cuda/fused_control_step"

BATCH=256; REPEATS=200; THREADS=128; JSON=""; HEADER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --batch)   BATCH="$2";   shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --json)    JSON="$2";    shift 2 ;;
        --header)  HEADER="$2";  shift 2 ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Architecture: the same override and detection order as grid_codegen/cuda_env.py.
arch_digits() {
    if [ -n "${GRID_CUDA_ARCH:-}" ]; then
        echo "$GRID_CUDA_ARCH" | sed -e 's/^sm_//' -e 's/^compute_//' -e 's/\.//g'
    elif command -v nvidia-smi >/dev/null 2>&1 &&
         cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')" &&
         [ -n "$cap" ]; then
        echo "$cap" | sed 's/\.//g'
    else
        echo 86
    fi
}
ARCH="sm_$(arch_digits)"

if [ -z "$HEADER" ]; then
    HEADER="$EX_DIR/grid.cuh"
    echo ">> [1/3] generating $HEADER (inverse_dynamics only)"
    "$PY" examples/cuda/gen_iiwa14_header.py --output "$HEADER" >/dev/null
else
    echo ">> [1/3] using header $HEADER"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/fused_control_step"

echo ">> [2/3] compiling with nvcc ($ARCH)"
"$NVCC" -arch="$ARCH" -std=c++17 -O2 -I "$(dirname "$HEADER")" \
    "$EX_DIR/fused_control_step.cu" -o "$BIN"

echo ">> [3/3] running: batch=$BATCH repeats=$REPEATS threads=$THREADS"
ARGS=(--batch "$BATCH" --repeats "$REPEATS" --threads "$THREADS" --arch "$ARCH")
[ -n "$JSON" ] && ARGS+=(--json "$JSON")
"$BIN" "${ARGS[@]}"
