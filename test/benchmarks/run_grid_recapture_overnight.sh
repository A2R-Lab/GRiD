#!/usr/bin/env bash
# GRiD FULL RE-CAPTURE — all three timing regimes, one approval, unattended-safe.
#
# WHY: the competitor matrix (results/competitive_20260712, 25/25 legs, all clean) is done, but the
# GRiD column feeding it is autotune_best_<host>.json dated 2026-07-02 -- BEFORE Inc6 (floating
# reduction determinism: crba -8%, minv -3.6%, id_grad -3.3%, fd_grad -1.9%), the surgical-spill
# work, and the Inc3 arena fold. So the current table UNDERSTATES GRiD. Recompile + recapture.
#
# THE THREE REGIMES (and what each costs):
#   1. compute-only  (GPU-resident: state already on device -- the MPC/RL design point)
#   2. with-mem      (numpy-in -> result-out, H2D/D2H included)
#      -> 1 and 2 are the SAME BINARY and the SAME RUN. One grid_glass capture emits both
#         (batch_256_compute_only_us + batch_256_with_mem_us). Regime 2 is FREE.
#   3. wrapper/FFI   (jax/torch via the grid_rbd bindings)
#      -> NOT regime 2 + a constant. The FFI launch path has a DIFFERENT thread optimum
#         (autotune_ffi.py header: iiwa14/fd/N=256, host-optimal 128thr = 49.7us through FFI,
#         FFI-optimal 768thr = 30.2us). It needs its OWN autotune. Same generated header, but a
#         second build artifact (the grid_rbd extension).
#
# PHASE SEPARATION IS THE POINT: builds may run concurrently; TIMING NEVER DOES.
#   Phase 1 BUILD   -- compile-only, no GPU timing. Parallel ONLY for the small robots.
#   Phase 2 MEASURE -- strictly serial, one process on the GPU, --no-recompile (pure cache hits).
#   Phase 3 ANALYZE -- CPU only.
#
# RAM SAFETY (the thing that crashes the box):
#   Big-robot SO monolithic TUs use ~24-36 GB of cicc EACH. On a 62 GB
#   box that means g1 compiles STRICTLY ONE TU AT A TIME. iiwa14/go2 are small enough to overlap 2.
#   Every compile runs GRID_COMPILE_WORKERS=1 --build-jobs 1 regardless.
#
# Resumable: the binary cache (.pytest_cache/grid_cuda) is content-keyed, so re-running skips
# anything already built. Safe to re-invoke after an interruption.
#
# Usage: bash test/benchmarks/run_grid_recapture_overnight.sh [outdir]
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/local/cuda/bin:$PATH
source .venv/bin/activate

ROOT="${1:-test/benchmarks/results/grid_recapture_$(date +%Y%m%d_%H%M)}"
mkdir -p "$ROOT"
LOG="$ROOT/recapture.log"

SMALL="${SMALL:-iiwa14 go2}"      # safe to overlap 2 compiles
BIG="${BIG:-g1}"                  # STRICTLY serial (24-36 GB cicc per TU)
BASES="${BASES:-fixed floating}"
TIERS="${TIERS:-shared lite minimal}"
COMPETITORS="${COMPETITORS:-test/benchmarks/results/competitive_20260712}"
PAR="${PAR:-2}"                   # max concurrent compiles for SMALL robots

# --- THREAD GRID: use run.py's DEFAULT. Do NOT override. ------------------------------------
# Threads are a RUNTIME knob (GRID_AUTOTUNE_THREAD_COUNT, read by grid_timing_dimms at first
# launch) -- sweeping them costs extra timed runs but ZERO recompiles. Only the TIER is
# compile-time (__launch_bounds__ + smem layout), and Phase 1 builds all three tiers anyway.
#
# run.py's DEFAULT_AUTOTUNE_THREAD_GRID is ALREADY the wide one:
#     (32, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024)
# (The `--help` string still says '96,128,192,256,320,384' -- that text is STALE; the A1a
# widening reached the warp floor 32 and the hw ceiling 1024 precisely because "clamping at 384
# hid the genuine large-robot optima" -- the fast regime scales UP with robot size: go2>=512,
# g1>=640. See project_grid_jax_ffi_thread_pathology.)
# _clip_grid_to_cap DROPS (does not time) any probe above a tier's launch_bounds cap
# (SHARED=MAX_PERF_LEVEL_THREADS, LITE=min(2x,768), MINIMAL=1024) -- so no failed launch, and no
# §1c "failed-launch-reads-fastest" trap. A one-level refinement probes around each winner.
#
# ⚠ Passing a narrower custom grid here would SILENTLY DROP the 32/64 warp-floor probes. Leave
# it unset unless you have a specific reason.
THREAD_GRID="${THREAD_GRID:-}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() { local t0=$SECONDS; "$@" >>"$LOG" 2>&1; local rc=$?; say "    rc=$rc ($((SECONDS-t0))s)  :: $*"; return $rc; }

say "=== GRiD RE-CAPTURE START ==="
say "root=$ROOT  small=[$SMALL] big=[$BIG] bases=[$BASES] tiers=[$TIERS] par=$PAR"
say "GPU: $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null)"
say "RAM: $(free -g | awk '/^Mem:/{print $7"GB avail of "$2"GB"}')"

# --- guard: never time on a busy GPU -------------------------------------------------------
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  say "!! ABORT: another process is on the GPU. Timing must be isolated."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$LOG"
  exit 1
fi

# ============================================================ PHASE 1: BUILD (no timing)
say ""
say "=== PHASE 1: BUILD (compile-only; no GPU timing) ==="

build_cell() {  # robot base tier
  local r="$1" b="$2" t="$3"
  # per-exe cutover: build this tier's per-algo solo exes via the wrapper (--compile-only). NO --build-dir:
  # the wrapper's content cache is LOCAL to its build-dir, so both this build and the PHASE-2 measure (which
  # calls the wrapper through run_multi_version) must share the wrapper's default dir (results/per_algo_<r>_<b>)
  # for the measure to cache-hit. --tier shared builds the suffix-less exe the autotune 'shared' tier reuses.
  python test/benchmarks/per_algo_bench.py \
      --robot "$r" --base "$b" --tier "$t" --compile-only --compile-jobs 1 >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)]     built $r/$b/$t rc=$?" >> "$LOG"
}

# small robots: overlap up to $PAR compiles
for r in $SMALL; do for b in $BASES; do for t in $TIERS; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  say "  [build] $r $b $t (parallel slot)"
  build_cell "$r" "$b" "$t" &
done; done; done
wait
say "  small-robot builds done"

# big robots: STRICTLY serial -- one 24-36 GB TU at a time
for r in $BIG; do for b in $BASES; do for t in $TIERS; do
  say "  [build] $r $b $t (SERIAL, big-robot)  mem_avail=$(free -g | awk '/^Mem:/{print $7}')GB"
  build_cell "$r" "$b" "$t"
done; done; done
say "  big-robot builds done"

# wrapper artifact: same generated headers, second build target
say "  [build] grid_rbd bindings (wrapper/FFI regime)"
step pip install -e ".[jax]"

# ============================================================ PHASE 2: MEASURE (serial, isolated)
say ""
say "=== PHASE 2: MEASURE (STRICTLY SERIAL -- one process on the GPU) ==="
say "    regimes 1+2 (compute-only AND with-mem) come from the SAME run"

# --no-recompile is LOAD-BEARING: it guarantees a compile can never sneak into the timing
# phase (Phase 1 already populated the content-keyed cache, so every cell is an instant hit).
for r in $SMALL $BIG; do
  say "  [measure] $r  mem_avail=$(free -g | awk '/^Mem:/{print $7}')GB"
  # THREAD_GRID intentionally unset by default -> run.py's wide (32..1024) default is used.
  GRID_ARG=(); [ -n "$THREAD_GRID" ] && GRID_ARG=(--autotune-thread-grid "$THREAD_GRID")
  step .venv/bin/python test/benchmarks/run_multi_version.py \
      --columns glass \
      --robots "$r" --bases $BASES \
      --autotune-threads --tiers $TIERS "${GRID_ARG[@]}" \
      --build-jobs 1 --no-recompile \
      --output-dir "$ROOT"
done

say "  [measure] kernel limits"
for r in $SMALL $BIG; do for b in $BASES; do
  step .venv/bin/python test/benchmarks/collect_kernel_limits.py --robot "$r" --base "$b" --no-recompile
done; done

# regime 3: the FFI path has its OWN thread optimum -- must be autotuned separately.
# NOTE: autotune_ffi.py takes --base {fixed,floating,both} and has NO --output; it writes its
# picks into launch_configs itself. --dry-run would only print.
say "  [measure] regime 3: wrapper/FFI autotune (grid_rbd)"
for r in $SMALL $BIG; do
  step .venv/bin/python test/benchmarks/autotune_ffi.py --robot "$r" --base both
done

# ============================================================ PHASE 3: ANALYZE (cpu only)
say ""
say "=== PHASE 3: ANALYZE ==="
step .venv/bin/python test/benchmarks/build_autotune_matrix.py --picks "$ROOT"/*_grid*.json

NEW_AUTOTUNE="$(ls -t test/benchmarks/results/autotune_best_*.json 2>/dev/null | head -1)"
say "  autotune_best -> $NEW_AUTOTUNE"

step .venv/bin/python test/benchmarks/analyze_competitive.py \
    --autotune "$NEW_AUTOTUNE" \
    --results "$COMPETITORS" "$ROOT"

say ""
say "=== GRID RE-CAPTURE DONE ==="
say "  grid_glass captures : $(ls "$ROOT"/*grid_glass*.json 2>/dev/null | wc -l)"
say "  ffi captures        : $(ls "$ROOT"/ffi_*.json 2>/dev/null | wc -l)"
say "  analysis            : $COMPETITORS/ANALYSIS.md"
say "  log                 : $LOG"
