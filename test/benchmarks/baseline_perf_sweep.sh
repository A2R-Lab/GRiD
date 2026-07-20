#!/usr/bin/env bash
# Overnight BASELINE perf capture for CURRENT modernizing-tests (post-GLASS-bump).
#
# Purpose (user 2026-07-06): current modernizing-tests IS the baseline, and right
# now we only A/B GRiD against its own FUTURE self — so capture GRiD's OWN
# per-algo/robot/tier timings across many robots. FUTURE perf work (deferred
# warp::inv / syev / congruence adoption, etc.) re-runs this same sweep and diffs.
# glass column ONLY: no pre_glass (we already beat it), no competitors (mjx/warp
# add hours and aren't the point of an against-ourselves baseline).
#
# Runs unattended overnight. Phased so the reliable data lands first; big-floating
# SO-wall robots are LAST so a partial finish still yields a broad baseline.
#
#   nohup bash tools/baseline_perf_sweep.sh > /tmp/baseline_perf_sweep.log 2>&1 &
set -uo pipefail
cd /home/plancher/Desktop/GRiD
export PATH=/usr/local/cuda/bin:$PATH
PY=.venv/bin/python
STAMP="$(date +%Y%m%d_%H%M)"
OUT="test/benchmarks/results/baseline_${STAMP}"
mkdir -p "$OUT"
echo "########## BASELINE PERF SWEEP (glass-only) ${STAMP} ##########"

# --- GPU-idle guard: timing must not contend with another agent -------------
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . | tr -d ' ')
NPROC=${NPROC:-0}
echo "GPU: util=${UTIL}%  mem=${MEM}MiB  compute_procs=${NPROC}"
if [ "${UTIL:-0}" -gt 5 ] || [ "${MEM:-0}" -gt 2000 ] || [ "${NPROC:-0}" -gt 0 ]; then
  echo "ABORT: GPU busy (another agent?). Timing needs an isolated GPU." >&2
  exit 1
fi

run () {  # run <phase-label> <extra args...>
  local label="$1"; shift
  echo "=== [${label}] START $(date) ==="
  # --build-jobs 3: SO compiles ~6-7GB each; 3 keeps us < ~21GB (box has ~25GB free).
  $PY test/benchmarks/run_multi_version.py \
      --columns glass \
      --build-jobs 3 \
      --output-dir "$OUT" \
      --report "$OUT/report_${label}.md" \
      "$@" || echo "WARN: phase ${label} exited non-zero (continuing)"
  echo "=== [${label}] END   $(date) ==="
  # Re-merge after every phase so a partial overnight still leaves a usable report.
  $PY test/benchmarks/merge_sweep_report.py --sweep-dir "$OUT" || true
}

# --- Phase 1: FIXED-base, all robots (fast, reliable — the core baseline) ----
run p1_fixed        --robots iiwa14 go2 g1 h2_plus baxter --fixed-only

# --- Phase 2: FLOATING small/medium (fast) ----------------------------------
run p2_float_small  --robots iiwa14 go2 baxter --bases floating

# --- Phase 3: FLOATING big robots LAST (SO-wall, hours each; overflow) -------
run p3_float_big    --robots g1 h2_plus --bases floating

echo "########## DONE ${STAMP} ##########"
echo "BASELINE saved under: $OUT (merged report: $OUT/benchmark_multi_version.md or report_*.md)"
echo "These GRiD 'glass' numbers are the reference future perf work diffs against."
