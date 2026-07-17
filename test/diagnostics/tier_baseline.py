"""P1 v2: Empirical resource-tier baseline for the cuBLASDx-removal v2.0 work.

For each (algorithm × robot) combination, capture three resource axes:

  1. Per-thread register count at __launch_bounds__(MAX_PERF_LEVEL_THREADS) — the
     perf tier today.
  2. Per-thread register count at __launch_bounds__(1024) — what TIER_MINIMAL
     looks like.
  3. Per-block dynamic shared memory bytes (queried from the constexpr
     *_DYNAMIC_SHARED_MEM_BYTES<float>() emitted in grid.cuh; ptxas only
     reports static smem, not dynamic, so we have to run a tiny host
     program per robot to extract these values).

Also captures spill stores at each tier — the strongest signal for "this
algorithm needs a real downgrade (not just an alias)" because nvcc
spilling to local memory is what makes the relaxed-launch_bounds path
slow.

Reuses the bench's `timeGRiD_batch.cu` as the kernel-instantiation driver,
but generates floating-base headers with `enable_floating_second_order=True`
so the SO kernels are emitted and the `#if GRID_HAS_FDSVA_SO` gates fire.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python test/diagnostics/tier_baseline.py

The script writes its report to test/diagnostics/results/tier_baseline.md
by default; override with the `TIER_BASELINE_REPORT` env var.
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys
from pathlib import Path

# Repo root is two levels up from test/diagnostics/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

URDF_DIR = Path.home() / ".cache/robot_descriptions"
ROBOTS = [
    ("iiwa14_fixed",   URDF_DIR / "drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf", False),
    ("go2_fixed",      URDF_DIR / "unitree_ros/robots/go2_description/urdf/go2_description.urdf", False),
    ("g1_fixed",       URDF_DIR / "unitree_ros/robots/g1_description/g1_29dof.urdf", False),
    ("g1_floating",    URDF_DIR / "unitree_ros/robots/g1_description/g1_29dof.urdf", True),
    # H2+ (Unitree, nv=75 fixed / 81 floating) is the large-robot scaling target
    # that retired the redundant h1_2. Vendored locally (GRiD-internal, not in
    # robot_descriptions), so it loads from robot_assets/.
    ("h2_plus_fixed",     REPO_ROOT / "robot_assets/h2_plus.urdf", False),
    ("h2_plus_floating",  REPO_ROOT / "robot_assets/h2_plus.urdf", True),
]

WORK = Path(os.environ.get("TIER_BASELINE_WORK", "/tmp/tier_baseline_v2"))
NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "120"
# The per-exe bench cutover retired the monolithic timeGRiD_batch.cu. Any TU that #includes grid.cuh and
# instantiates every kernel is an equivalent ptxas -v driver; a per_algo_bench solo TU does exactly that
# (its run_all_tests -> init_grid_kernel_attrs references every kernel). Materialize one into WORK.
BENCH_GRID_DIR = REPO_ROOT / "test/benchmarks/baselines/grid"
sys.path.insert(0, str(REPO_ROOT / "test/benchmarks"))
from per_algo_bench import _solo_batch_tu_source  # noqa: E402
WORK.mkdir(parents=True, exist_ok=True)
BATCH_CU = WORK / "instantiate_all_kernels.cu"
BATCH_CU.write_text(_solo_batch_tu_source("inverse_dynamics"))

# Friendly label per kernel name. Names match emitted symbols in grid.cuh.
KERNEL_LABELS = {
    "inverse_dynamics_kernel":                  "ID",
    "minv_kernel":                       "Minv",
    "forward_dynamics_kernel":                  "FD",
    "aba_kernel":                               "ABA",
    "crba_kernel":                              "CRBA",
    "end_effector_pose_kernel":                 "EE_POSE",
    "end_effector_pose_gradient_kernel":        "EE_POSE_GRAD",
    "end_effector_pose_hessian_kernel": "EE_POSE_HESS",
    "inverse_dynamics_gradient_kernel":         "ID_DU",
    "forward_dynamics_gradient_kernel":         "FD_DU",
    "idsva_so_body_frame_kernel":               "IDSVA_SO_B",
    "idsva_so_world_frame_kernel":              "IDSVA_SO_W",
    "fdsva_so_kernel":                          "FDSVA_SO",
}

# Per-kernel constexpr smem-bytes macros emitted by the codegen.
SMEM_MACROS = {
    "ID":            "INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES",
    "Minv":          "MINV_DYNAMIC_SHARED_MEM_BYTES",
    "FD":            "FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES",
    "ABA":           "ABA_DYNAMIC_SHARED_MEM_BYTES",
    "CRBA":          "CRBA_DYNAMIC_SHARED_MEM_BYTES",
    "EE_POSE":       "END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES",
    "EE_POSE_GRAD":  "END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    "EE_POSE_HESS":  "END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES",
    "ID_DU":         "INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    "FD_DU":         "FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
    "IDSVA_SO_B":    "IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    "IDSVA_SO_W":    "IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES",
    "FDSVA_SO":      "FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES",
}


def gen_grid_cuh(robot_label: str, urdf: Path, floating: bool, out_dir: Path) -> Path:
    """Regenerate grid.cuh for one robot config. Forces full SO emission on
    floating-base robots so the diagnostic can characterize those kernels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid.cuh"
    code = f"""
import sys
sys.path.insert(0, "{REPO_ROOT}")
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
p = URDFParser()
r = p.parse("{urdf}", floating_base={floating})
cg = GRiDCodeGenerator(r, 0, FILE_NAMESPACE="grid")
cg.gen_all_code(output_path="{out_path}", enable_floating_second_order=True,
                enable_idsva_so_world_frame=True)
"""
    rc = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/python"), "-c", code],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"codegen failed for {robot_label}:\n{rc.stdout}\n{rc.stderr}")
    return out_path


def detect_emitted_kernels(grid_cuh: Path) -> set[str]:
    """Return the set of friendly kernel labels actually defined in this
    grid.cuh."""
    text = grid_cuh.read_text()
    found = set()
    for kname, label in KERNEL_LABELS.items():
        if re.search(rf"\bvoid\s+{kname}\s*\(", text):
            found.add(label)
    return found


def compile_with_bounds(grid_cuh: Path, label: str, launch_bounds_override: int | None,
                         build_dir: Path) -> tuple[str, bool]:
    """Compile timeGRiD_batch.cu with -Xptxas -v. Optionally patches
    grid.cuh's __launch_bounds__(MAX_PERF_LEVEL_THREADS) → __launch_bounds__(N).
    Returns (nvcc stderr, success)."""
    build_dir.mkdir(parents=True, exist_ok=True)

    if launch_bounds_override is not None:
        patched = build_dir / "grid.cuh"
        text = grid_cuh.read_text()
        text = re.sub(
            r"__launch_bounds__\(MAX_PERF_LEVEL_THREADS\)",
            f"__launch_bounds__({launch_bounds_override})",
            text,
        )
        patched.write_text(text)
        active_cuh = patched
    else:
        active_cuh = grid_cuh

    obj = build_dir / f"batch_{label}.o"
    cmd = [
        NVCC, "-std=c++17", "-c", "-o", str(obj), str(BATCH_CU),
        f"-DGRID_HEADER_FILE=\"{active_cuh}\"",
        f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
        "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
        f"-I{BATCH_CU.parent}",
        f"-I{BENCH_GRID_DIR}",   # timeGRiD_common.h (the solo TU #includes it)
        f"-I{active_cuh.parent}",
        f"-I{REPO_ROOT / 'GLASS'}",
        f"-I{REPO_ROOT / 'GLASS/src'}",
        "-DGRID_TIMING_TEST_ITERS=1",
        "-Xptxas", "-v",
        "-Wno-deprecated-gpu-targets",
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    (build_dir / f"stderr_{label}.log").write_text(rc.stderr)
    return rc.stderr, rc.returncode == 0


def parse_ptxas(stderr: str) -> dict:
    """Pull per-kernel R / smem / spill stores from `nvcc -Xptxas -v` output."""
    pat_entry = re.compile(r"Compiling entry function '(_Z[^']+)'")
    pat_frame = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores")
    pat_used  = re.compile(r"Used (\d+) registers(?:, (\d+) bytes smem)?")

    info = {}
    current = None
    for line in stderr.splitlines():
        m = pat_entry.search(line)
        if m:
            current = m.group(1)
            info[current] = {"mangled": current}
            continue
        m = pat_frame.search(line)
        if m and current:
            info[current]["spill_st"] = int(m.group(2))
            continue
        m = pat_used.search(line)
        if m and current:
            info[current]["registers"] = int(m.group(1))
            info[current]["smem_static"] = int(m.group(2) or 0)
            continue

    # Map mangled name → friendly label; keep worst-case (highest register
    # count) variant per label when multiple specializations exist.
    out = {}
    for mangled, d in info.items():
        for kernel_name, label in KERNEL_LABELS.items():
            if kernel_name in mangled:
                if label not in out or d.get("registers", 0) > out[label].get("registers", 0):
                    out[label] = d
                break
    return out


SMEM_DUMP_TEMPLATE = r"""
#include "grid.cuh"
#include <cstdio>

int main() {
    using T = float;
    std::printf("MAX_PERF_LEVEL_THREADS %d\n", (int)grid::MAX_PERF_LEVEL_THREADS);
%MACRO_CALLS%
    return 0;
}
"""


def dump_smem_bytes(grid_cuh: Path, emitted_labels: set[str], build_dir: Path) -> tuple[dict[str, int], int]:
    """Compile + run a tiny .cu that prints constexpr smem-bytes for every
    kernel actually emitted. Returns ({label: bytes}, max_perf_level_threads)."""
    build_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    for label, macro in SMEM_MACROS.items():
        if label not in emitted_labels:
            continue
        calls.append(f'    std::printf("{label} %zu\\n", grid::{macro}<T>());')
    source = SMEM_DUMP_TEMPLATE.replace("%MACRO_CALLS%", "\n".join(calls))
    src = build_dir / "smem_dump.cu"
    src.write_text(source)

    binary = build_dir / "smem_dump"
    cmd = [NVCC, "-std=c++17", "-o", str(binary), str(src),
           f"-I{grid_cuh.parent}",
           f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
           "-O0", "-Wno-deprecated-gpu-targets"]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    (build_dir / "compile.log").write_text(rc.stderr)
    if rc.returncode != 0:
        return {}, 0
    rc = subprocess.run([str(binary)], capture_output=True, text=True)
    out = {}
    sug = 0
    for line in rc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            if parts[0] == "MAX_PERF_LEVEL_THREADS":
                try: sug = int(parts[1])
                except ValueError: pass
            else:
                try: out[parts[0]] = int(parts[1])
                except ValueError: pass
    return out, sug


def needs_real_downgrade(perf: dict | None, relax: dict | None, smem_bytes: int) -> str:
    """Decide whether a (kernel, robot) cell needs real downgrade code vs
    free-aliasing TIER_LITE/MINIMAL to TIER_SHARED.

    Signals (tightened to avoid false positives):
      A. relax.sp ≥ 500 AND relax.sp > 2 × max(perf.sp, 100):
         heavy absolute spill cost AND meaningful growth from perf.
      B. smem ≥ 80 KB: already 80% of sm_120's 100 KB per-block cap;
         outer-kernel inline users hit smem pressure even at perf.
    """
    if perf is None or relax is None:
        return "?"
    psp = perf.get("spill_st", 0)
    rsp = relax.get("spill_st", 0)
    spill_signal = rsp >= 500 and rsp > 2 * max(psp, 100)
    smem_signal = smem_bytes >= 81920
    if spill_signal and smem_signal:
        return "YES (spill+smem)"
    if spill_signal:
        return "YES (spill)"
    if smem_signal:
        return "YES (smem)"
    return "no"


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    results = []   # list of (robot, kernel_label, perf, relax, smem_bytes)
    sug = {}       # robot -> MAX_PERF_LEVEL_THREADS
    for robot_label, urdf, floating in ROBOTS:
        if not urdf.exists():
            print(f"[skip] {robot_label}: URDF not present at {urdf}")
            continue
        print(f"\n=== {robot_label} ===")
        robot_dir = WORK / robot_label
        try:
            grid_cuh = gen_grid_cuh(robot_label, urdf, floating, robot_dir)
        except Exception as e:
            print(f"  [codegen] FAILED: {e}")
            continue

        emitted = detect_emitted_kernels(grid_cuh)
        print(f"  grid.cuh ({grid_cuh.stat().st_size} bytes) — kernels emitted: {sorted(emitted)}")

        smem, suggested = dump_smem_bytes(grid_cuh, emitted, robot_dir / "smem")
        sug[robot_label] = suggested
        print(f"  smem dump: {len(smem)} entries, MAX_PERF_LEVEL_THREADS={suggested}")

        # Perf-tier compile (default __launch_bounds__(MAX_PERF_LEVEL_THREADS))
        stderr, ok = compile_with_bounds(grid_cuh, "perf", None, robot_dir / "build_perf")
        if not ok:
            print(f"  [perf-compile] FAILED — see {robot_dir}/build_perf/stderr_perf.log")
        perf = parse_ptxas(stderr)

        # Relaxed-tier compile (launch_bounds=1024)
        stderr, ok = compile_with_bounds(grid_cuh, "relax", 1024, robot_dir / "build_relax")
        if not ok:
            print(f"  [relax-compile] FAILED — see {robot_dir}/build_relax/stderr_relax.log")
        relax = parse_ptxas(stderr)

        for label in KERNEL_LABELS.values():
            if label not in emitted:
                continue
            results.append((robot_label, label, perf.get(label), relax.get(label), smem.get(label, -1)))

    # Render report
    lines = [
        "# Tier-baseline report v2 (P1)",
        "",
        "Each row: one kernel × one robot config. **R** = registers/thread,",
        "**sp** = local-memory spill stores (bytes), **smem** = per-block",
        "dynamic shared memory bytes (computed at codegen via the",
        "`*_DYNAMIC_SHARED_MEM_BYTES<float>()` constexpr; ptxas's static-smem",
        "report is unhelpful here because GRiD uses `extern __shared__`).",
        "",
        "**perf** = `__launch_bounds__(MAX_PERF_LEVEL_THREADS)` (the default);",
        "**relax** = `__launch_bounds__(1024)` (what TIER_MINIMAL looks like).",
        "",
        "**Decision predicate**:",
        "- `YES (spill)`: relax.sp ≥ 500 AND relax.sp > 2 × max(perf.sp, 100).",
        "  Real perf cliff at relaxed bounds.",
        "- `YES (smem)`: smem ≥ 80 KB (~80% of the sm_120 100 KB per-block cap).",
        "  Outer-kernel inline users hit smem pressure even at perf — they want",
        "  a smem-axis downgrade variant.",
        "- `no`: free-alias TIER_LITE/MINIMAL to TIER_SHARED; no body changes needed.",
        "",
        "**Per-robot `MAX_PERF_LEVEL_THREADS`**: " +
            ", ".join(f"{r}={n}" for r, n in sug.items()),
        "",
        "| Robot | Kernel | perf R/sp | relax R/sp | smem | downgrade? |",
        "|---|---|---|---|---:|---|",
    ]

    needs = {"YES (spill)": [], "YES (smem)": [], "YES (spill+smem)": []}
    for robot, label, p, r, smem_bytes in results:
        pr  = p.get("registers", "?") if p else "?"
        psp = p.get("spill_st", "?") if p else "?"
        rr  = r.get("registers", "?") if r else "?"
        rsp = r.get("spill_st", "?") if r else "?"
        verdict = needs_real_downgrade(p, r, smem_bytes if smem_bytes >= 0 else 0)
        if verdict.startswith("YES"):
            needs[verdict].append(f"{robot}/{label}")
        smem_cell = f"{smem_bytes}" if smem_bytes >= 0 else "?"
        lines.append(f"| {robot} | {label} | R={pr},sp={psp} | R={rr},sp={rsp} | {smem_cell} | {verdict} |")

    lines += ["", "## Cells needing real downgrade", ""]
    any_yes = False
    for kind, cells in needs.items():
        if not cells:
            continue
        any_yes = True
        lines.append(f"**{kind}** ({len(cells)}):")
        for c in cells:
            lines.append(f"  - {c}")
        lines.append("")
    if not any_yes:
        lines.append("**None** — every emitted (kernel, robot) cell can free-alias.")

    default_report = Path(__file__).resolve().parent / "results" / "tier_baseline.md"
    report = Path(os.environ.get("TIER_BASELINE_REPORT", str(default_report)))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {report}")
    for kind, cells in needs.items():
        if cells:
            print(f"  {kind}: {cells}")


if __name__ == "__main__":
    main()
