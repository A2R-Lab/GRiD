#!/usr/bin/env bash
# Generate the signed GPU-proof receipt for GRiD's CUDA + wrapper equivalence
# suites. Run this on a machine with a real GPU and a quiet box.
#
# ONE entry point story: test/run_validation.py is the LAUNCHER (plan ->
# --execute -> commit evidence -> --finalize; --full for a release receipt),
# test/run_split_suite.py is the DRIVER it runs, and this script is a THIN
# WRAPPER over that driver for ad-hoc runs. All three write gpu-proof.json
# on the primary architecture.
#
# Receipts by architecture. test/gpu-proof-receipts.yaml maps the compute
# capability of the GPU in this box to a receipt path. The primary row
# (sm_120, the RTX 5090) records gpu-proof.json at the repo root: the one
# receipt CI verifies and the one that defines test/gpu-proof-tests.txt.
# Secondary rows (sm_87 Jetson Orin, sm_72 Jetson Xavier) record under
# receipts/ (receipts/gpu-proof-sm87.json) and document those architectures;
# a run on a secondary architecture never writes gpu-proof.json. The driver
# resolves the row first (`run_split_suite.py --print-receipt-target`) and a
# GPU without a row stops here with a message to add one. To record on a
# Jetson: install the dev requirements, then on a clean tree run
#   SPLIT=1 test/run_gpu_proof.sh
# and commit receipts/gpu-proof-sm87.json together with its header-key
# ledger (receipts/gpu-proof-sm87-header-keys.json). SPLIT_REFRESH=1 on a
# secondary architecture refreshes that architecture's own receipt; a carry
# or refresh from a receipt recorded on another architecture is refused.
# `make verify-all` verifies every receipt on disk, and
# `test/receipt_integrity.py --table` prints their compatibility-page rows.
# The launcher test/run_validation.py is primary-only.
#
# Main requires a receipt whose collection equals test/gpu-proof-tests.txt
# EXACTLY (policy required_test_manifest). A narrowed run is a local
# diagnostic that proves the plumbing, never a candidate receipt:
#   SCOPE=smoke test/run_gpu_proof.sh            # -k "iiwa14 or (go2 and floating)"
#   SCOPE=curated test/run_gpu_proof.sh          # -k "iiwa14 or go2 or g1 or h1_2"
#   PYTEST_ARGS="-k go2" test/run_gpu_proof.sh   # ad-hoc -k on top of SCOPE
#   (SCOPE=full, the default, is the whole gpu_proof set.)
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
#   SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh
#     (the EVERYDAY workflow when CI's verify-receipt goes red: re-run ONLY the
#      shards whose fingerprints changed vs the COMMITTED receipt of this
#      architecture (gpu-proof.json on the primary; the table's path on a
#      secondary) and
#      carry the rest — soundness-gated by codegen_neutrality's covering
#      matrix + robot-asset gate + per-shard header-key replay. Requires a
#      CLEAN tree. GRID_SPLIT_REFRESH_DRY=1 previews the stale/carried plan.)
# To pause a running SPLIT pass: `touch <out_dir>/PAUSE` (stops cleanly between
# shards, ≤ one shard's latency), or SIGINT/SIGTERM the driver (stops within
# the shard); resume with SPLIT_RESUME.
# SCOPE narrows only the cuda side (via --cuda-k at COLLECTION time — the
# wrapper modules are iiwa14-based without robot names in their test IDs, so a
# robot -k would silently deselect whole modules).
# Verify like CI does — WITH the skip baseline (a local verify without it
# disagrees with CI):
#   gpu-proof verify --receipt gpu-proof.json --policy test/gpu-proof-policy.yaml \
#     --expected-skips test/gpu-proof-expected-skips.txt
# (release: --policy test/gpu-proof-policy-release.yaml refuses carried shards;
#  a secondary receipt: its path and its expected-skips file from the table).
set -euo pipefail

# All knobs are ENV VARS (SCOPE/SPLIT/PYTEST_ARGS/...), not flags. Guard the
# arg list so `--help` prints the header instead of silently starting a
# multi-hour GPU pass (N2.4 ergonomics, 2026-09-08).
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '2,78p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg '$1' — this script is configured via env vars (SCOPE=, SPLIT=, PYTEST_ARGS=, ...); see --help" >&2; exit 2 ;;
  esac
done

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

# Refuse to sign a dirty tree: peer gitlinks pin commits, not uncommitted peer
# content. Correctness paths are fingerprinted; commit ancestry is not a
# substitute for those hashes (mirrors allow_dirty:false in both policies).
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before signing a receipt" >&2
    echo "       (peer gitlinks cannot attest uncommitted peer content;" >&2
    echo "        commit all correctness inputs before recording)." >&2
    exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"

# Which receipt this box records (test/gpu-proof-receipts.yaml, keyed on the
# detected compute capability): "gpu-proof.json" on the primary architecture,
# a path under receipts/ on a secondary one. An architecture without a row
# stops here, before any GPU work.
RECEIPT_TARGET="$("$PYTHON" test/run_split_suite.py --print-receipt-target)" || exit 2
IFS=$'\t' read -r RECEIPT_ROLE RECEIPT_ARCH RECEIPT_PATH RECEIPT_LABEL <<< "$RECEIPT_TARGET"
if [[ "$RECEIPT_ROLE" != "primary" ]]; then
    echo "[run_gpu_proof] secondary architecture sm_${RECEIPT_ARCH} (${RECEIPT_LABEL}): receipt ${RECEIPT_PATH}; gpu-proof.json is not written"
    mkdir -p "$(dirname "$RECEIPT_PATH")"
fi

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
    # SPLIT_REFRESH=1 (or =<receipt path>): shard-level receipt refresh — only
    # shards whose test files changed re-run; the rest carry from the previous
    # receipt (everyday policy accepts carried shards; the RELEASE policy
    # test/gpu-proof-policy-release.yaml refuses them, so releases still take
    # one fresh full pass). Requires SCOPE=full (a narrowed refresh would
    # attest less than the shards it replaces). SPLIT_REFRESH=1 means the
    # committed receipt of this architecture ($RECEIPT_PATH).
    REFRESH_ARGS=()
    if [[ -n "${SPLIT_REFRESH:-}" ]]; then
        if [[ -n "${SPLIT_CARRY_FROM:-}" ]]; then
            echo "ERROR: SPLIT_REFRESH and SPLIT_CARRY_FROM are mutually exclusive" >&2
            exit 2
        fi
        if [[ -n "$SCOPE_K" ]]; then
            echo "ERROR: SPLIT_REFRESH requires SCOPE=full" >&2
            exit 2
        fi
        REFRESH_SRC="$SPLIT_REFRESH"
        [[ "$REFRESH_SRC" == "1" ]] && REFRESH_SRC="$RECEIPT_PATH"
        REFRESH_ARGS=(--refresh-from "$REFRESH_SRC")
    fi
    CUDA_K_ARGS=()
    if [[ -n "$SCOPE_K" ]]; then CUDA_K_ARGS=(--cuda-k "$SCOPE_K"); fi

    rc=0
    # -u: unbuffered — the driver's section headers otherwise sit in a pipe
    # buffer until the first per-shard line flushes, making a healthy run look
    # hung to anyone tailing the log.
    "$PYTHON" -u test/run_split_suite.py --receipts --domains wrappers,cuda \
        "${CUDA_K_ARGS[@]}" "${CARRY_ARGS[@]}" "${REFRESH_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --out "$OUT_DIR" -- ${PYTEST_ARGS:-} || rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "SPLIT driver rc=$rc (3=paused, 130=interrupted, 1=shard failures);" >&2
        echo "  shards/ledger in $OUT_DIR — continue with:" >&2
        echo "  SPLIT=1 SPLIT_RESUME=$OUT_DIR test/run_gpu_proof.sh" >&2
        exit 1
    fi

    # The driver already merged + re-signed; publish to this architecture's
    # path (the repo-root gpu-proof.json CI reads on the primary, a path
    # under receipts/ on a secondary). A refresh with NOTHING stale merges
    # nothing (the committed receipt already covers the tree) — that is
    # success, not a missing file.
    if [[ -f "$OUT_DIR/gpu-proof.json" ]]; then
        cp "$OUT_DIR/gpu-proof.json" "$RECEIPT_PATH"
        echo "[run_gpu_proof] SPLIT receipt written to $RECEIPT_PATH (shards in $OUT_DIR)"
    else
        echo "[run_gpu_proof] nothing stale — committed $RECEIPT_PATH already covers this tree"
    fi
    exit 0
fi

exec "$PYTHON" -m pytest \
    test/cuda_equivalents \
    test/python_wrappers \
    -m gpu_proof \
    "${K_ARGS[@]}" \
    --gpu-proof-enable \
    --gpu-proof-out="$RECEIPT_PATH" \
    ${PYTEST_ARGS:-}
