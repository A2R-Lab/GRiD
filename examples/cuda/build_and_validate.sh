#!/usr/bin/env bash
# Build + run + validate the GRiD inverse_dynamics CUDA example end to end.
# Run from the repo root:  bash examples/cuda/build_and_validate.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-.venv/bin/python}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${GRID_SM_ARCH:-sm_120}"          # RTX 5090; override for your GPU
EX_DIR="examples/cuda"
HEADER="$EX_DIR/grid.cuh"
BIN="$(mktemp -u /tmp/grid_id_example.XXXXXX)"
OUT="$(mktemp -u /tmp/grid_id_out.XXXXXX)"

echo "=== Flagship: inverse_dynamics (_device / _inner / batched) ==="
echo ">> [1/4] generating $HEADER (inverse_dynamics only)"
PYTHONPATH=. "$PY" "$EX_DIR/gen_iiwa14_header.py" --output "$HEADER" >/dev/null

echo ">> [2/4] compiling with nvcc ($ARCH)"
"$NVCC" -arch="$ARCH" -std=c++17 -I "$EX_DIR" -I GLASS/include \
    "$EX_DIR/inverse_dynamics_kernel_example.cu" -o "$BIN"

echo ">> [3/4] running"
"$BIN" | tee "$OUT"

echo ">> [4/4] validating vs RBDReference.inverse_dynamics"
PYTHONPATH=. "$PY" "$EX_DIR/validate.py" < "$OUT" 2>/dev/null

rm -f "$BIN" "$OUT"

echo
echo "=== Second-order: idsva_so via the _host surface ==="
SO_HEADER="$EX_DIR/grid_so.cuh"
SO_BIN="$(mktemp -u /tmp/grid_so_example.XXXXXX)"
SO_OUT="$(mktemp -u /tmp/grid_so_out.XXXXXX)"

echo ">> [1/4] generating $SO_HEADER (idsva_so_body_frame only)"
PYTHONPATH=. "$PY" - "$SO_HEADER" >/dev/null <<'PYGEN'
import sys
from robot_descriptions import iiwa14_description
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
robot = URDFParser().parse(iiwa14_description.URDF_PATH, floating_base=False)
GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
    algorithm_list=["idsva_so_body_frame"], output_path=sys.argv[1])
PYGEN

echo ">> [2/4] compiling with nvcc ($ARCH)"
"$NVCC" -arch="$ARCH" -std=c++17 -I "$EX_DIR" -I GLASS/include \
    "$EX_DIR/idsva_so_host_example.cu" -o "$SO_BIN"

echo ">> [3/4] running"
"$SO_BIN" > "$SO_OUT"
head -2 "$SO_OUT"

echo ">> [4/4] validating vs RBDReference.idsva_so_body_frame"
PYTHONPATH=. "$PY" "$EX_DIR/validate_so.py" < "$SO_OUT" 2>/dev/null

rm -f "$SO_BIN" "$SO_OUT"
echo ">> done."
