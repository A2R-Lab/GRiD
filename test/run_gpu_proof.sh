#!/usr/bin/env bash
# Generate the signed GPU-proof receipt for GRiD's CUDA + wrapper equivalence
# suites. Run this on a machine with a real GPU and a quiet box.
#
# Receipt SCOPE is tiered so the receipt is never an all-or-nothing barrier —
# the fingerprint + signature + commit-SHA proof are identical regardless of how
# many tests the receipt attests; a smaller scope is a valid (narrower) receipt:
#
#   SCOPE=smoke test/run_gpu_proof.sh     # ~2 robots, cached cells — minutes; proves the plumbing
#   SCOPE=curated test/run_gpu_proof.sh   # representative robot set — tens of minutes
#   SCOPE=full  test/run_gpu_proof.sh     # every gpu_proof test — hours cold, the nightly job (DEFAULT)
#   PYTEST_ARGS="-k go2" test/run_gpu_proof.sh   # ad-hoc scope on top of SCOPE
#
# Ship code first, tighten coverage later: CI verifies whatever receipt is
# committed (and skips if none), so a smoke receipt can land with the code and a
# full receipt can replace it after an overnight run. Re-running just re-signs
# gpu-proof.json in place.
#
# The receipt records outcomes for every test carrying the gpu_proof marker
# (auto-applied to cuda_equivalence + python_wrappers by test/conftest.py) and
# signs the code fingerprint with your local SSH key. CI verifies it CPU-only.
#
# IMPORTANT: pass EXPLICIT test paths, never a bare `pytest test/` — a bare run
# would pick up the CPU-only lane tests that belong in ordinary CI, not the GPU
# receipt. (The plugin is the pytest-gpu-proof PyPI package now, installed via
# install/requirements-dev.txt — no vendored submodule to collect.)
#
# SPLIT=1 — crash-isolated receipt path (schema-2 shards). python_wrappers runs
# through test/run_split_suite.py (per-module pytest subprocesses: one module's
# SIGABRT/exit() can no longer eat the rest of the suite's outcomes), each module
# emitting a shard receipt; cuda_equivalents (already subprocess-isolated
# internally) runs as ONE extra shard; everything merges + re-signs into the same
# repo-root gpu-proof.json the monolithic path writes, so CI verify is unchanged.
#   SPLIT=1 test/run_gpu_proof.sh
#   SPLIT=1 SPLIT_CARRY_FROM=old-gpu-proof.json test/run_gpu_proof.sh
#     (carry unchanged modules' shards from an older receipt — verifier accepts
#      carried shards only under a policy with allow_carried: true)
# SCOPE tiers apply to both paths (forwarded as -k to every shard invocation).
set -euo pipefail

# SCOPE -> a -k expression narrowing the gpu_proof test set. Empty = full suite.
SCOPE="${SCOPE:-full}"
case "$SCOPE" in
    smoke)   SCOPE_K="iiwa14 or (go2 and floating)" ;;
    curated) SCOPE_K="iiwa14 or go2 or g1 or h1_2" ;;
    full)    SCOPE_K="" ;;
    *) echo "ERROR: unknown SCOPE='$SCOPE' (use smoke|curated|full)" >&2; exit 2 ;;
esac

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

K_ARGS=()
if [[ -n "$SCOPE_K" ]]; then K_ARGS=(-k "$SCOPE_K"); fi

echo "[run_gpu_proof] SCOPE=$SCOPE  SPLIT=${SPLIT:-0}  ${SCOPE_K:+(-k \"$SCOPE_K\")}  ${PYTEST_ARGS:+PYTEST_ARGS=$PYTEST_ARGS}"

if [[ "${SPLIT:-0}" = "1" ]]; then
    OUT_DIR="test/.split_suite/receipt_$(date +%Y%m%d_%H%M%S)"

    # Shard 1..N: python_wrappers, one shard per module (crash-isolated). The
    # driver merges its own shards into $OUT_DIR/gpu-proof.json. rc captured
    # explicitly — a pipe or early exec would mask which leg failed.
    CARRY_ARGS=()
    if [[ -n "${SPLIT_CARRY_FROM:-}" ]]; then
        CARRY_ARGS=(--receipts-carry-from "$SPLIT_CARRY_FROM")
    fi
    rc_wrappers=0
    "$PYTHON" test/run_split_suite.py --receipts "${CARRY_ARGS[@]}" \
        --out "$OUT_DIR" -- "${K_ARGS[@]}" ${PYTEST_ARGS:-} || rc_wrappers=$?

    # Shard N+1: cuda_equivalents as a single shard (its runners are already
    # per-exe subprocess-isolated; one shard keeps the receipt's coverage equal
    # to the monolithic path — a SPLIT receipt must never attest LESS).
    rc_cuda=0
    "$PYTHON" -m pytest test/cuda_equivalents -m gpu_proof "${K_ARGS[@]}" \
        --gpu-proof-enable \
        --gpu-proof-out="$OUT_DIR/receipts/cuda_equivalents.json" \
        --gpu-proof-shard=cuda_equivalents \
        --gpu-proof-shard-fingerprint-paths=test/cuda_equivalents \
        ${PYTEST_ARGS:-} || rc_cuda=$?

    if [[ $rc_wrappers -ne 0 || $rc_cuda -ne 0 ]]; then
        echo "ERROR: SPLIT legs failed (python_wrappers rc=$rc_wrappers, cuda_equivalents rc=$rc_cuda);" >&2
        echo "       shards left in $OUT_DIR for triage — NOT merging a failing receipt." >&2
        exit 1
    fi

    # Final merge + re-sign into the repo-root receipt CI reads.
    GPU_PROOF="$(dirname "$PYTHON")/gpu-proof"
    "$GPU_PROOF" merge \
        --out gpu-proof.json --repo . \
        "$OUT_DIR/gpu-proof.json" "$OUT_DIR/receipts/cuda_equivalents.json"
    echo "[run_gpu_proof] SPLIT receipt written to gpu-proof.json (shards in $OUT_DIR)"
    exit 0
fi

exec "$PYTHON" -m pytest \
    test/cuda_equivalents \
    test/python_wrappers \
    -m gpu_proof \
    "${K_ARGS[@]}" \
    --gpu-proof-enable \
    --gpu-proof-out=gpu-proof.json \
    ${PYTEST_ARGS:-}
