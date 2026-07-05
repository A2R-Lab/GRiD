#!/usr/bin/env bash
# Generate the signed GPU-proof receipt for GRiD's CUDA + wrapper equivalence
# suites. Run this on a machine with a real GPU and a quiet box (the run compiles
# every generated kernel cold — hours on first run, cached thereafter).
#
#   test/run_gpu_proof.sh                 # full receipt -> gpu-proof.json
#   PYTEST_ARGS="-k iiwa14" test/run_gpu_proof.sh   # scoped dry run
#
# The receipt records outcomes for every test carrying the gpu_proof marker
# (auto-applied to cuda_equivalence + python_wrappers by test/conftest.py) and
# signs the code fingerprint with your local SSH key. CI verifies it CPU-only.
#
# IMPORTANT: pass EXPLICIT test paths, never a bare `pytest test/` — a bare run
# would descend into the pytest-gpu-proof submodule's own suite and (harmlessly,
# via collect_ignore) waste collection, and would also pick up the CPU-only lane
# tests that belong in ordinary CI, not the GPU receipt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Refuse to sign a dirty tree: the fingerprint can't see submodule content, so a
# clean tree is what makes the receipt's commit SHA an honest description of the
# code under test (mirrors gpu-proof-policy.yaml allow_dirty:false).
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before signing a receipt" >&2
    echo "       (the fingerprint cannot descend into the codegen/GLASS submodules;" >&2
    echo "        a clean tree is what pins them via the receipt's commit SHA)." >&2
    exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"

exec "$PYTHON" -m pytest \
    test/cuda_equivalents \
    test/python_wrappers \
    -m gpu_proof \
    --gpu-proof-enable \
    --gpu-proof-out=gpu-proof.json \
    ${PYTEST_ARGS:-}
