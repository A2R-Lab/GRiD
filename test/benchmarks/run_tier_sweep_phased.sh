#!/bin/bash
# ============================================================================
# PHASED tier perf sweep — SO compiled SEPARATELY so it can't crash the box.
# ============================================================================
# Design (user-agreed 2026-06-26): the ONLY compile wall + RAM-crash risk in the
# whole sweep is the second-order kernels (idsva_so / fdsva_so). A single floating
# g1 SO translation unit is ~6-7 GB of cicc and can take 60+ min; h2_plus SO is
# ~36 GB single-TU. Building those alongside everything else (or 8-way) spikes RAM
# toward the 62 GB limit -> crashes VS Code -> tears down the snap.code cgroup ->
# kills the sweep (see run_a1b_sweep.sh header + memory build_ram_so_compiles).
#
# So we split the sweep into two phases that NEVER share a compile:
#
#   PHASE 1 (no-SO): every first-order / gradient / de-gated algorithm. Compiles
#            fast, no crash risk. This is the FULL validation surface for the
#            surgical-spill campaign (all 5 de-gates -- centroidal, fd_du, coriolis,
#            osc_inertia, f_ext -- and the GLASS v2 migration are first-order; NONE
#            touch SO). Run this first and to completion.
#
#   PHASE 2 (SO-only): idsva_so + fdsva_so (+ their first-order deps, which are
#            cheap). Compiled in their OWN header, ONE robot/base/tier at a time,
#            build-jobs <=2 (1 for h2_plus), with a `free -g` RAM gate before each
#            build. SO has its own (pre-existing, separate) spill rungs worth timing
#            but is lower priority than Phase 1 and is the part that can crash, so it
#            runs last, slow and careful.
#
# The split is driven by GRID_BENCH_ALGORITHM_LIST (comma-separated; regenerates the
# bench header to exactly that algo set -- see test/benchmarks/baselines/grid/run.py
# and GRiDCodeGenerator _normalize_codegen_algorithms).
#
# ROBOT / BASE MATRIX (user-chosen 2026-06-26): per-robot SINGLE base (no redundant
# fixed x floating cartesian). h1_2 DROPPED.
#     fixed-base   : iiwa14, baxter
#     floating-base: go2, g1, h2_plus
# BATCH SIZES: 32 and 256.
#
# ISOLATION (memory safe_dev_and_timing_methodology): timing must be SERIAL on a
# QUIET gpu -- no concurrent CPU/compiles during a measured cell. run_multi_version
# already builds-then-times per cell; do NOT run other heavy work while this runs.
# Best launched on an idle desktop. The run.py content-addressed cache
# (.pytest_cache/grid_cuda) resumes across restarts if codegen/.cu are unchanged.
#
# Usage:   bash test/benchmarks/run_tier_sweep_phased.sh [PHASE] [BUILD_JOBS]
#            PHASE      = 1 | 2 | all   (default all -> phase 1 then phase 2)
#            BUILD_JOBS = phase-1 build parallelism (default 4; phase 2 forces <=2)
#          BUILD_ONLY=1 bash test/benchmarks/run_tier_sweep_phased.sh [PHASE] [BUILD_JOBS]
#            -> PRE-COMPILE every cell's binaries into the cache and STOP before timing
#               (warm now; time later with the SAME command minus BUILD_ONLY on a quiet
#               GPU -> pure cache-hit). Drops the pinocchio column during pre-build.
#
# PREREQS TO VERIFY ON FIRST REAL RUN (not yet exercised end-to-end):
#   (a) GRID_BENCH_ALGORITHM_LIST exported here reaches run.py through
#       run_multi_version's subprocess (it inherits parent env by default; confirm
#       the generated header's "Codegen profile:" note lists the intended set).
#   (b) batch sizes 32/256 wiring: run_multi_version currently prints single/N=16/
#       N=256 summaries. Confirm/extend the batch-N knob (BATCH_SIZES below) maps to
#       a real flag before trusting the 32-column. Until then 256 is the safe column.
# ============================================================================
set -uo pipefail
cd /home/plancher/Desktop/GRiD
export PATH=/usr/local/cuda/bin:$PATH

PHASE="${1:-all}"
BUILD_JOBS="${2:-4}"
TS="$(date +%Y%m%d_%H%M)"
OUTROOT="test/benchmarks/results/tier_sweep_phased_${TS}"
mkdir -p "$OUTROOT"

# --- algorithm sets ---------------------------------------------------------
# Phase 1 = `all` MINUS the SO trio, PLUS osc_inertia (opt-in; de-gate #4).
NOSO_ALGOS="inverse_dynamics,minv,forward_dynamics,inverse_dynamics_gradient,forward_dynamics_gradient,aba,crba,end_effector_pose,end_effector_pose_gradient,end_effector_pose_hessian,integrator,integrator_gradient,integrator_with_gradient,f_ext_gradient,inverse_dynamics_regressor,forward_dynamics_parameter_gradient,kinetic_energy_regressor,potential_energy_regressor,com,ccrba,energy,generalized_gravity,nonlinear_effects,coriolis_matrix,dccrba,cmm_time_variation,osc_inertia,frame_jacobian,frame_jacobian_dot"
# Phase 2 = the SO kernels + their first-order deps (deps are cheap; SO is the wall).
# integrator family is REQUIRED: the SO algos pull in integrator_gradient transitively,
# whose floating mjx-output path calls grid_dIntegrate_q_block (defined by base `integrator`,
# _integrator.py:244). Without integrator in the set the SO-only header emits the caller but
# not the helper -> "type name is not allowed" on floating (fixed-base never hits that path;
# the full 'all' header always has integrator). [codegen dep-graph gap backlogged]
SO_ALGOS="inverse_dynamics,minv,forward_dynamics,inverse_dynamics_gradient,forward_dynamics_gradient,integrator,integrator_gradient,integrator_with_gradient,idsva_so_body_frame,fdsva_so,idsva_so_world_frame"

# --- shared config ----------------------------------------------------------
# Robot lists are env-overridable. baxter is REGISTERED (commit 1226853: run.py LOCAL_URDF +
# left_endpoint EE; run_multi_version ROBOTS/EE_FRAMES/GRID_ONLY) and verified fixed-base.
FIXED_ROBOTS="${FIXED_ROBOTS:-iiwa14 baxter}"
FLOATING_ROBOTS="${FLOATING_ROBOTS:-go2 g1 h2_plus}"
TIERS="shared lite minimal"
BATCH_SIZES="32 256"   # see PREREQ (b)

# BUILD_ONLY=1 -> pre-compile every cell's binaries into the content-addressed cache
# and STOP before timing (warm the cache now; time later on a quiet GPU with the same
# command minus --build-only). In build-only mode we also drop the pinocchio column
# (nothing to pre-build there) so the pre-build is pure GRiD compile.
BUILD_ONLY="${BUILD_ONLY:-0}"

run_cell() {  # $1=phase-tag $2=algos $3=robots $4=base $5=build_jobs
  local tag="$1" algos="$2" robots="$3" base="$4" bj="$5"
  local out="$OUTROOT/${tag}_${base}"
  mkdir -p "$out"
  local extra=() cols=(glass pinocchio)
  if [ "$BUILD_ONLY" = "1" ]; then extra=(--build-only); cols=(glass); fi
  echo "=== [$tag/$base]$([ "$BUILD_ONLY" = "1" ] && echo ' BUILD-ONLY') robots=[$robots] build_jobs=$bj  $(date) ==="
  echo "    free -g: $(free -g | awk '/Mem:/{print "used="$3" free="$4" avail="$7}')"
  echo "    GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
  GRID_BENCH_ALGORITHM_LIST="$algos" \
  .venv/bin/python test/benchmarks/run_multi_version.py \
      --columns "${cols[@]}" \
      --robots $robots \
      --bases "$base" \
      --autotune-threads \
      --tiers $TIERS \
      --build-jobs "$bj" \
      "${extra[@]}" \
      --output-dir "$out" 2>&1 | tee "$out/sweep.log"
}

phase1() {
  echo "########## PHASE 1 (no-SO) START $(date)  build_jobs=$BUILD_JOBS ##########"
  run_cell "p1_noSO" "$NOSO_ALGOS" "$FIXED_ROBOTS"    fixed    "$BUILD_JOBS"
  run_cell "p1_noSO" "$NOSO_ALGOS" "$FLOATING_ROBOTS" floating "$BUILD_JOBS"
  echo "########## PHASE 1 DONE $(date) ##########"
}

phase2() {
  # SO ONLY, serial, RAM-gated. build-jobs <=2; h2_plus forced to 1.
  local bj=2
  echo "########## PHASE 2 (SO-only) START $(date)  build_jobs<=$bj ##########"
  run_cell "p2_SO" "$SO_ALGOS" "$FIXED_ROBOTS" fixed    "$bj"
  # floating SO minus h2_plus (which is the ~36 GB single-TU monster — solo @1, last).
  local fl_no_h2=$(echo "$FLOATING_ROBOTS" | tr ' ' '\n' | grep -v h2_plus | tr '\n' ' ')
  [ -n "${fl_no_h2// }" ] && run_cell "p2_SO" "$SO_ALGOS" "$fl_no_h2" floating "$bj"
  if echo "$FLOATING_ROBOTS" | grep -qw h2_plus; then
    echo "--- h2_plus SO is the heaviest single compile (~36 GB). build-jobs=1, solo. ---"
    run_cell "p2_SO" "$SO_ALGOS" "h2_plus"     floating 1
  fi
  echo "########## PHASE 2 DONE $(date) ##########"
}

echo "=== PHASED tier sweep  phase=$PHASE  outroot=$OUTROOT  $(date) ==="
case "$PHASE" in
  1)   phase1 ;;
  2)   phase2 ;;
  all) phase1; phase2 ;;
  *)   echo "unknown PHASE '$PHASE' (use 1|2|all)"; exit 2 ;;
esac
echo "=== PHASED tier sweep COMPLETE  $(date)  results in $OUTROOT ==="
