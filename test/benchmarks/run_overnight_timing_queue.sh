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
#   4. runtime-param A/B (hardware co-design): what does runtime-mutability actually COST? See leg 3.
#
# TIMING ISOLATION IS THE WHOLE POINT: this script refuses to start if anything else is on the GPU.
# Do not run other agents/builds while it is up.
# ⚠ Legs 1-2 are pure measure (cache-warm). Leg 3 DOES COMPILE (8 headers, cache-MISS by construction —
#   the runtime-param variants are part of the header cache key). Serial, GRID_COMPILE_WORKERS=1.
# NOTE leg 3b (multi_target) also compiles. It was CUT on 2026-07-14 and is now FIXED + RE-ENABLED
#   (parent 2dcedb2 — the monolithic timing TUs had no measure block; PER_ALGO_SPECS alone never fired).
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

# ---------------------------------------------------------------- 3. runtime-param A/B (hardware co-design)
# THE QUESTION: what does runtime-mutability actually COST? Each variant sources one class of model
# params from a MUTABLE device table instead of baking it as a compile-time literal. SPARSITY stays
# baked in every case, and each is BIT-IDENTICAL to the baked path until you call its set_*_params()
# mutator — so the ONLY thing being measured here is the loss of the compiler's VALUE-folding.
#
# HYPOTHESIS TO TEST: inertia + joint-dynamics should be ~free (cold-ish loads, hoisted out of the hot
# path), while runtime_transform should be the one that BITES — its loads land per-cell INSIDE the hot
# X-recompute. Codegen diff on iiwa14-fixed backs this up: transform changes 1440 lines vs inertia's
# 378 and joint-dynamics' 216.
#
# WHAT THE ANSWER DECIDES: whether runtime params stay an OPT-IN codegen variant (a separate artifact
# per robot — zero cost when off, which is what protects the perf story) or can just become the
# DEFAULT (one artifact, pay the table-read everywhere). Cannot be answered without these numbers.
#
# ⚠ UNLIKE legs 1-2 THIS LEG COMPILES (4 variants x 2 robots = 8 headers, all cache-MISS by
#   construction — the variants are in the header cache key). Serial, GRID_COMPILE_WORKERS=1.
#   Build and measure are NOT interleaved per variant: we time each variant right after its own build,
#   but nothing else is on the GPU, so the isolation that matters (no CONCURRENT timing) holds.
say ""
say "=== [3/4] runtime-param A/B — what does hardware-co-design mutability actually cost? ==="
RTDIR="$ROOT/runtime_params_ab"
mkdir -p "$RTDIR"
for cell in "iiwa14 fixed" "go2 floating"; do
  set -- $cell
  for variant in baked runtime-inertia runtime-transform runtime-joint-dynamics; do
    flag=""; [ "$variant" = "baked" ] || flag="--$variant"
    say "  [rt-ab] $1 $2 / $variant   mem_avail=$(free -g | awk '/^Mem:/{print $7}')GB"
    GRID_COMPILE_WORKERS=1 step .venv/bin/python test/benchmarks/baselines/grid/run.py \
        --robot "$1" --base "$2" --compile-workers 1 $flag \
        --output "$RTDIR/${1}_${2}_${variant//-/_}.json"
  done
done

say "  runtime-param A/B table (baked = 1.00x baseline; >1 means the table-read COSTS time):"
.venv/bin/python test/benchmarks/compare_runtime_param_ab.py --dir "$RTDIR" 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------- 3b. multi_target (FIXED + RE-ENABLED)
# WAS DISABLED 2026-07-14 because no MULTI_TARGET_POSITION row ever reached the results JSON.
# ROOT CAUSE (fixed, parent 2dcedb2): PER_ALGO_SPECS drives the PER-ALGO TU path, but a default run uses
# the HAND-WRITTEN MONOLITHIC timeGRiD_{batch,single}.cu -- which had no multi_target measure block. The
# specs were necessary but not sufficient. Both paths now have it; verified on iiwa14-fixed (N=34):
# multi_target_position 18.22us compute-only, _gradient 25.73us (vs end_effector_pose 7.22us @ 1 target
# -- sublinear, as expected: the shared FK chain-up amortizes over the batch, only extraction scales).
#
# ⚠ THE NUMBER IS ONLY MEANINGFUL WITH ITS BATCH SIZE. Cost scales with the target count, so the batch is
#   the robot's own COLLISION SPHERIZATION (collision is multi_target's real consumer -> multi_target and
#   config_free then describe the SAME geometry). The N is printed + lands in the JSON. NEVER quote bare.
# ⚠ NOT a W/L cell: no competitor has a multi_target counterpart -> capability-lead, like config_free.
say ""
say "=== [3b/4] multi_target_position{,_gradient} — first-ever timing (batch = collision spherization) ==="
MTDIR="$ROOT/multi_target"
mkdir -p "$MTDIR"
for cell in "iiwa14 fixed" "go2 floating"; do
  set -- $cell
  say "  [mt] $1 $2   mem_avail=$(free -g | awk '/^Mem:/{print $7}')GB"
  GRID_COMPILE_WORKERS=1 step .venv/bin/python test/benchmarks/baselines/grid/run.py \
      --robot "$1" --base "$2" --multi-target-from-collision --compile-workers 1 \
      --output "$MTDIR/${1}_${2}_multi_target.json"
done
say "  (grep the log for 'multi_target batch from collision spherization: N=' — that N is the batch size"
say "   these µs are FOR. A multi_target number quoted without its N is meaningless.)"

# ---------------------------------------------------------------- 4. rebuild table
say ""
say "=== [4/4] rebuild autotune matrix + re-run competitive analysis ==="
# NOTE: only the BAKED captures feed the published table (glob excludes runtime_params_ab/).
step .venv/bin/python test/benchmarks/build_autotune_matrix.py --picks "$ROOT"/*_grid*.json

NEW_AUTOTUNE="$(ls -t test/benchmarks/results/autotune_best_*.json 2>/dev/null | head -1)"
say "  autotune_best -> $NEW_AUTOTUNE"
step .venv/bin/python test/benchmarks/analyze_competitive.py \
    --autotune "$NEW_AUTOTUNE" \
    --results "$COMPETITORS" "$ROOT"

say ""
say "=== OVERNIGHT TIMING QUEUE DONE ==="
say "  analysis    : $COMPETITORS/ANALYSIS.md"
say "  runtime A/B : $RTDIR/  (+ the table printed above)"
say "  log         : $LOG"
say "  NOTE grep the log for '[autotune] WARN' — a failed probe is DROPPED (never recorded as fast,"
say "       per §1c), but it silently costs coverage. Zero WARNs is the goal now that §1t is fixed."
