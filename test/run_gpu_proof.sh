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
# SPLIT=1 — crash-isolated, PAUSABLE receipt path (schema-2 shards), one
# driver invocation covering BOTH suites: python_wrappers as per-module shards
# AND cuda_equivalents as GRANULAR node-id shards bounded to ~2h each (bin-
# packed by test/run_split_suite.py; rolling durations calibrate the packing).
# Everything merges + re-signs into the same repo-root gpu-proof.json the
# monolithic path writes, so CI verify is unchanged.
#   SPLIT=1 test/run_gpu_proof.sh
#   SPLIT=1 SPLIT_RESUME=test/.split_suite/receipt_<stamp> test/run_gpu_proof.sh
#     (continue an interrupted/paused run — completed shards are never re-run)
#   SPLIT=1 SPLIT_CARRY_FROM=old-gpu-proof.json test/run_gpu_proof.sh
#     (carry unchanged modules' shards from an older receipt — verifier accepts
#      carried shards only under a policy with allow_carried: true)
# To pause a running SPLIT pass: `touch <out_dir>/PAUSE` (stops cleanly between
# shards, ≤ one shard's latency), or SIGINT/SIGTERM the driver (stops within
# the shard); resume with SPLIT_RESUME.
# SCOPE narrows only the cuda side (via --cuda-k at COLLECTION time — the
# wrapper modules are iiwa14-based without robot names in their test IDs, so a
# robot -k would silently deselect whole modules; 12/25 under curated, 2026-08-09).
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
    # ONE driver invocation covers both domains: per-module wrapper shards +
    # granular cuda node-id shards (≤ ~2h each). The driver merges ALL shard
    # receipts into $OUT_DIR/gpu-proof.json itself; rc captured explicitly —
    # a pipe or early exec would mask which shard failed.
    if [[ -n "${SPLIT_RESUME:-}" ]]; then
        OUT_DIR="$SPLIT_RESUME"
        RESUME_ARGS=(--resume "$OUT_DIR")
    else
        OUT_DIR="test/.split_suite/receipt_$(date +%Y%m%d_%H%M%S)"
        RESUME_ARGS=()
    fi
    CARRY_ARGS=()
    if [[ -n "${SPLIT_CARRY_FROM:-}" ]]; then
        CARRY_ARGS=(--receipts-carry-from "$SPLIT_CARRY_FROM")
    fi
    CUDA_K_ARGS=()
    if [[ -n "$SCOPE_K" ]]; then CUDA_K_ARGS=(--cuda-k "$SCOPE_K"); fi

    rc=0
    "$PYTHON" test/run_split_suite.py --receipts --domains wrappers,cuda \
        "${CUDA_K_ARGS[@]}" "${CARRY_ARGS[@]}" "${RESUME_ARGS[@]}" \
        --out "$OUT_DIR" -- ${PYTEST_ARGS:-} || rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "SPLIT driver rc=$rc (3=paused, 130=interrupted, 1=shard failures);" >&2
        echo "  shards/ledger in $OUT_DIR — continue with:" >&2
        echo "  SPLIT=1 SPLIT_RESUME=$OUT_DIR test/run_gpu_proof.sh" >&2
        exit 1
    fi

    # The driver already merged + re-signed; publish to the repo-root path CI reads.
    cp "$OUT_DIR/gpu-proof.json" gpu-proof.json
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
