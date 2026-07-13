#!/usr/bin/env bash
# OVERNIGHT TIMING QUEUE — everything still owed on the timing side, one approval.
#
# WHY EACH ITEM IS HERE
#   1. go2-floating RE-MEASURE. Its shared-tier binary used to die on fdsva_so's shared-memory OOB
#      (fixed: GCG d884585 / parent 54c36f5, debugging-guide §1t). Because gpuErrchk exit()s and the
#      batch binary runs every algo in ONE process, that kernel's death killed the whole shared-tier
#      binary -> EVERY algo on this cell lost its shared-tier probes (shared=0 / lite=240 / minimal=192)
#      and the autotune fell back to lite+minimal => PESSIMISTIC picks (GRiD under-reporting itself).
#      All 3 tier binaries are ALREADY BUILT + content-cached, so this is a pure measure pass (~20 min).
#      ⚠ ONLY this cell needs it — every other (robot,base) had healthy shared probes:
#         g1_fixed 255, g1_floating 224, go2_fixed 170, iiwa14_fixed 204, iiwa14_floating 208.
#   2. collect_kernel_limits for the 4 cells that never finished (go2 fixed/floating, g1 fixed/floating).
#      iiwa14 fixed+floating already landed. This leg is SLOW (63-123 min/cell on iiwa14; g1 will be
#      worse) — it's why we killed it at midday. It feeds build_autotune_matrix's --kernel-limits.
#   3. Rebuild the autotune matrix + re-run the competitive analysis so the published table finally
#      reflects (a) the post-Inc6 GRiD column and (b) the un-pessimized go2-floating picks.
#
# TIMING ISOLATION IS THE WHOLE POINT: this script does NO compiles (everything is cache-warm) and
# refuses to start if anything else is on the GPU. Do not run other agents/builds while it is up.
#
# Usage: bash test/benchmarks/run_overnight_timing_queue.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/local/cuda/bin:$PATH
source .venv/bin/activate

ROOT="${ROOT:-test/benchmarks/results/grid_recapture_20260712_2131}"
COMPETITORS="${COMPETITORS:-test/benchmarks/results/competitive_20260712}"
LOG="$ROOT/overnight_timing_queue.log"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() { local t0=$SECONDS; "$@" >>"$LOG" 2>&1; local rc=$?; say "    rc=$rc ($((SECONDS-t0))s)"; return $rc; }

say "=== OVERNIGHT TIMING QUEUE START ==="
say "RAM: $(free -g | awk '/^Mem:/{print $7"GB avail"}')"

# --- HARD GUARD: timing must be isolated -----------------------------------------------------
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  say "!! ABORT: another process is on the GPU. Timing must be isolated (quiet the other agent)."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG"
  exit 1
fi
say "GPU is idle -> safe to time."

# ---------------------------------------------------------------- 1. go2-floating re-measure
say ""
say "=== [1/3] go2-floating RE-MEASURE (post fdsva_so fix; binaries cache-warm, NO compiles) ==="
step .venv/bin/python test/benchmarks/run_multi_version.py \
    --columns glass --robots go2 --bases floating \
    --autotune-threads --tiers shared lite minimal \
    --build-jobs 1 --no-recompile \
    --output-dir "$ROOT"

say "  verifying the shared tier actually came back:"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json
f = 'test/benchmarks/results/grid_recapture_20260712_2131/go2_floating_grid_glass.json'
d = json.load(open(f))
def find(o):
    if isinstance(o, dict):
        if 'algo_picks' in o: return o['algo_picks']
        for v in o.values():
            r = find(v)
            if r: return r
    return None
picks = find(d) or {}
tiers = {}
for a, p in picks.items():
    for t, sw in (p.get('sweep') or {}).items():
        tiers[t] = tiers.get(t, 0) + len(sw)
print("    probes per tier:", tiers)
print("    SHARED RECOVERED ✔" if tiers.get('shared', 0) > 0 else "    !! SHARED STILL ZERO — investigate")
fd = picks.get('fdsva_so')
if fd:
    print(f"    fdsva_so pick: tier={fd.get('tier_optimal')} threads={fd.get('threads_optimal')} us={fd.get('us_at_optimal')}")
PY

# ---------------------------------------------------------------- 2. kernel limits (the slow leg)
say ""
say "=== [2/3] collect_kernel_limits — the 4 cells that never finished (SLOW: hours) ==="
for cell in "go2 fixed" "go2 floating" "g1 fixed" "g1 floating"; do
  set -- $cell
  say "  [limits] $1 $2"
  step .venv/bin/python test/benchmarks/collect_kernel_limits.py --robot "$1" --base "$2" --no-recompile
done

# ---------------------------------------------------------------- 3. rebuild table
say ""
say "=== [3/3] rebuild autotune matrix + re-run competitive analysis ==="
step .venv/bin/python test/benchmarks/build_autotune_matrix.py --picks "$ROOT"/*_grid*.json

NEW_AUTOTUNE="$(ls -t test/benchmarks/results/autotune_best_*.json 2>/dev/null | head -1)"
say "  autotune_best -> $NEW_AUTOTUNE"
step .venv/bin/python test/benchmarks/analyze_competitive.py \
    --autotune "$NEW_AUTOTUNE" \
    --results "$COMPETITORS" "$ROOT"

say ""
say "=== OVERNIGHT TIMING QUEUE DONE ==="
say "  analysis : $COMPETITORS/ANALYSIS.md"
say "  log      : $LOG"
say "  NOTE grep the log for '[autotune] WARN' — a failed probe is DROPPED (never recorded as fast,"
say "       per §1c), but it silently costs coverage. Zero WARNs is the goal now that §1t is fixed."
