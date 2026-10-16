#!/usr/bin/env bash
# timing_night.sh: one resumable overnight timing run on a quiet GPU.
#
# Stages, each with a done marker under <out>/markers/ so a rerun or
# --resume skips completed work:
#   gaps      the gap cells: Frax floating-base RNEA/FD gradients (go2, g1),
#             the grid_cuda end-effector pose gradient and the BARD
#             end-effector pose gradient, through the release collector.
#   release   the release re-collection of test/benchmarks/release/README.md
#             (core and wrappers at 300 samples, the table at 30 samples under
#             the fp32-fd-warnings policy, the CPU/tensor baselines on the
#             core operations), one fresh capture directory each.
#   tutorial  examples/cuda/fused_control_step/build_and_time.sh when present.
#   report    a CPU-only diagnostic export of every capture of this run.
# The GPU-idle guard of overnight_batch.sh runs before every stage.
#
# Launch (the maintainer coordinates other projects' timing before this):
#   nohup bash test/benchmarks/timing_night.sh > <out>/night.log 2>&1 &
# Continue an interrupted run:
#   bash test/benchmarks/timing_night.sh --resume test/benchmarks/results/timing_night_<date>
#
# Options:
#   --out DIR        output directory (default test/benchmarks/results/timing_night_<UTC date>)
#   --resume DIR     continue an existing run directory
#   --stages LIST    comma-separated subset of: gaps,release,tutorial,report
#   --dry-run        print the plan and the job counts; run nothing, write nothing
#   --help           this text
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The interpreter with the dev stack: $PYTHON when set, else the repository venv,
# else the python3 on PATH (CI installs the stack into the runner's interpreter).
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PATH=/usr/local/cuda/bin:$PATH
ALL_STAGES="gaps release tutorial report"
STAGES="$ALL_STAGES"
OUT=""
RESUME=""
DRY=0

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --out) OUT="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --stages) STAGES="${2//,/ }"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "ERROR: unknown argument '$1'; see --help" >&2; exit 2 ;;
  esac
done
for s in $STAGES; do
  case " $ALL_STAGES " in *" $s "*) ;; *) echo "ERROR: unknown stage '$s' (choose from: $ALL_STAGES)" >&2; exit 2 ;; esac
done
if [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { echo "ERROR: --resume directory does not exist: $RESUME" >&2; exit 2; }
  OUT="$RESUME"
fi
[ -n "$OUT" ] || OUT="$ROOT/test/benchmarks/results/timing_night_$(date -u +%Y%m%d)"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
MARKERS="$OUT/markers"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# One line of ownership: the box-wide timing lock belongs to the maintainer, not to this script.
echo "timing_night: the maintainer is responsible for coordinating other projects' timing before launch; this script holds no lock."
log "root=$ROOT out=$OUT stages='$STAGES' dry_run=$DRY"

# --- GPU-idle guard (same thresholds as overnight_batch.sh) -------------------
gpu_guard() {
  local util mem nproc
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)  # grep -c prints 0 itself
  log "GPU: util=${util:-?}%  mem=${mem:-?}MiB  compute_procs=${nproc:-?}"
  if [ "${util:-0}" -gt 5 ] || [ "${mem:-0}" -gt 2000 ] || [ "${nproc:-0}" -gt 0 ]; then
    log "ABORT: GPU is busy (another user). Timing needs an isolated GPU; rerun with --resume $OUT when clear."
    exit 1
  fi
}

# --- the jobs -------------------------------------------------------------------
# collect <capture-name> <collector arguments...>
COLLECT="$PY -m test.benchmarks.release.collect"
CPU_THREADS="$(nproc 2>/dev/null || echo 8)"

gap_jobs() {
  echo "gap-frax-floating-gradients|--stage table --robots go2 g1 --backends frax --operations inverse_dynamics_gradient forward_dynamics_gradient --accuracy-policy fp32-fd-warnings"
  echo "gap-grid-cuda-ee-gradient|--stage table --backends grid_cuda --operations end_effector_pose_gradient"
  echo "gap-bard-ee-gradient|--stage table --backends bard --operations end_effector_pose_gradient"
}

release_jobs() {
  echo "release-core|--stage core --iterations 300 --cpu-threads $CPU_THREADS"
  echo "release-wrappers|--stage wrappers --iterations 300"
  echo "release-table|--stage table --accuracy-policy fp32-fd-warnings --operations minv forward_dynamics forward_dynamics_gradient fdsva_so end_effector_pose end_effector_pose_gradient end_effector_pose_hessian crba nonlinear_effects generalized_gravity ccrba coriolis_matrix"
  echo "release-table-core-baselines|--stage table --operations inverse_dynamics inverse_dynamics_gradient idsva_so --backends mujoco_cpu bard frax"
}

plan_counts() {
  # Print the collector's dry plan as one line: planned jobs, runnable jobs, repeats, batches.
  # shellcheck disable=SC2086
  (cd "$ROOT" && $COLLECT $1) | "$PY" -c '
import json, sys
p = json.load(sys.stdin)
runnable = sum(1 for j in p["jobs"] if not j["unavailable"])
print("jobs=%d runnable=%d repeats=%d batches=%s iterations=%d warm_seconds=%s"
      % (len(p["jobs"]), runnable, p["repeats"], p["batches"], p["iterations"], p["warm_seconds"]))'
}

run_collect() {
  local name="$1" args="$2" dir="$OUT/$name"
  if [ -e "$MARKERS/$name.done" ]; then log "skip $name (done)"; return 0; fi
  gpu_guard
  if [ -d "$dir" ]; then
    local aside="$dir.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
    log "$name: setting aside the interrupted capture as $aside (never merged)"
    mv "$dir" "$aside"
  fi
  log "start $name: collect $args --execute --output $dir"
  # shellcheck disable=SC2086
  (cd "$ROOT" && $COLLECT $args --execute --output "$dir") > "$OUT/$name.log" 2>&1
  local rc=$?
  tail -3 "$OUT/$name.log" | sed "s/^/  $name: /"
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) $name" > "$MARKERS/$name.done"
    log "done $name"
  else
    log "FAILED $name (rc=$rc); the capture stays in $dir for inspection and is retried on --resume"
    STAGE_FAILED=1
  fi
  return "$rc"
}

run_stage() {
  local stage="$1"; shift
  case " $STAGES " in *" $stage "*) ;; *) log "stage $stage not selected"; return 0 ;; esac
  if [ -e "$MARKERS/stage-$stage.done" ]; then log "stage $stage already complete"; return 0; fi
  STAGE_FAILED=0
  log "=== stage $stage ==="
  "$@"
  if [ "$STAGE_FAILED" -eq 0 ]; then
    echo "$(date -u +%FT%TZ)" > "$MARKERS/stage-$stage.done"
    log "=== stage $stage complete ==="
  else
    log "=== stage $stage incomplete ==="
  fi
}

stage_gaps() {
  while IFS='|' read -r name args; do run_collect "$name" "$args"; done < <(gap_jobs)
}

stage_release() {
  while IFS='|' read -r name args; do run_collect "$name" "$args"; done < <(release_jobs)
}

TUTORIAL="$ROOT/examples/cuda/fused_control_step/build_and_time.sh"
stage_tutorial() {
  if [ -e "$MARKERS/tutorial.done" ]; then log "skip tutorial (done)"; return 0; fi
  if [ ! -f "$TUTORIAL" ]; then
    log "tutorial: skipped: example absent ($TUTORIAL)"
    echo "skipped: example absent" > "$OUT/fused_control_step.status"
    echo "$(date -u +%FT%TZ) skipped: example absent" > "$MARKERS/tutorial.done"
    return 0
  fi
  gpu_guard
  log "start tutorial: bash $TUTORIAL --json $OUT/fused_control_step.json"
  (cd "$ROOT" && bash "$TUTORIAL" --json "$OUT/fused_control_step.json") > "$OUT/tutorial.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$OUT/fused_control_step.json" ]; then
    echo "completed" > "$OUT/fused_control_step.status"
    echo "$(date -u +%FT%TZ)" > "$MARKERS/tutorial.done"
    log "done tutorial"
  else
    echo "failed: rc=$rc" > "$OUT/fused_control_step.status"
    log "FAILED tutorial (rc=$rc); see $OUT/tutorial.log"
    STAGE_FAILED=1
  fi
}

stage_report() {
  # CPU only: a diagnostic report over every completed capture of this run; not website data.
  local captures=()
  while IFS='|' read -r name _; do [ -e "$MARKERS/$name.done" ] && captures+=("$OUT/$name"); done < <(gap_jobs; release_jobs)
  if [ "${#captures[@]}" -eq 0 ]; then log "report: no completed capture to export"; STAGE_FAILED=1; return 1; fi
  rm -rf "$OUT/report"
  log "start report over ${#captures[@]} captures"
  (cd "$ROOT" && "$PY" -m test.benchmarks.release.report "${captures[@]}" --output "$OUT/report") > "$OUT/report.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then log "done report: $OUT/report/index.html"; else log "FAILED report (rc=$rc); see $OUT/report.log"; STAGE_FAILED=1; fi
}

write_provenance() {
  [ -s "$OUT/provenance.json" ] && return 0
  (cd "$ROOT" && "$PY" - "$OUT" "$STAGES" <<'EOF'
import json, sys
from test.benchmarks.release.collect import provenance
from test.benchmarks.release.protocol import write_json
out, stages = sys.argv[1], sys.argv[2].split()
write_json(f"{out}/provenance.json", {"schema": 1, "purpose": "timing_night", "stages": stages, "output": out, **provenance()})
EOF
  )
}

write_summary() {
  {
    echo "timing_night summary  $(date -u +%FT%TZ)"
    echo "output: $OUT"
    echo "stages requested: $STAGES"
    echo
    for s in $ALL_STAGES; do
      if [ -e "$MARKERS/stage-$s.done" ]; then echo "stage $s: complete"; else echo "stage $s: incomplete or not selected"; fi
    done
    echo
    while IFS='|' read -r name _; do
      if [ -e "$MARKERS/$name.done" ]; then
        echo "capture $name: complete"
        grep -h "Strict validated cells\|Capture:" "$OUT/$name.log" 2>/dev/null | sed 's/^/  /'
      elif [ -d "$OUT/$name" ]; then echo "capture $name: incomplete (see $name.log)"
      else echo "capture $name: not run"; fi
    done < <(gap_jobs; release_jobs)
    echo
    echo "tutorial microbench: $(cat "$OUT/fused_control_step.status" 2>/dev/null || echo "not run")"
    echo "report: $([ -f "$OUT/report/index.html" ] && echo "$OUT/report/index.html" || echo "not exported")"
    echo
    echo "Next steps (maintainer): review the captures, point docs/release_dataset.json at the ones that"
    echo "replace published data, then run .venv/bin/python docs/release_pipeline.py --output <scratch>."
  } > "$OUT/summary.txt"
  log "summary: $OUT/summary.txt"
}

# --- dry run ----------------------------------------------------------------------
if [ "$DRY" -eq 1 ]; then
  echo "DRY RUN: nothing is executed and nothing is written."
  echo "output directory: $OUT"
  echo "stages: $STAGES"
  for s in $STAGES; do
    case "$s" in
      gaps|release)
        echo "stage $s:"
        while IFS='|' read -r name args; do
          echo "  $name: collect $args"
          counts="$(plan_counts "$args")" || { echo "dry plan failed for $name (interpreter $PY)" >&2; exit 1; }
          echo "    $counts"
        done < <(if [ "$s" = gaps ]; then gap_jobs; else release_jobs; fi) ;;
      tutorial)
        if [ -f "$TUTORIAL" ]; then echo "stage tutorial: bash $TUTORIAL --json $OUT/fused_control_step.json"
        else echo "stage tutorial: skipped: example absent ($TUTORIAL)"; fi ;;
      report) echo "stage report: CPU-only diagnostic export of this run's captures to $OUT/report" ;;
    esac
  done
  exit 0
fi

# --- real run ---------------------------------------------------------------------
mkdir -p "$OUT" "$MARKERS"
write_provenance
run_stage gaps stage_gaps
run_stage release stage_release
run_stage tutorial stage_tutorial
run_stage report stage_report
write_summary
incomplete=0
for s in $STAGES; do [ -e "$MARKERS/stage-$s.done" ] || incomplete=1; done
if [ "$incomplete" -eq 0 ]; then log "all selected stages complete"; exit 0; fi
log "some stages are incomplete; rerun with --resume $OUT"
exit 1
