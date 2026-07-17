"""Legacy ptxas -v diagnostic — compare GRiD HEAD (GLASS linalg) vs the
pre-GLASS reference at commit d2c0d18 for the four regression cells:

  Pattern 1: go2_fixed FD + ABA  (1.11×-1.18× regression at N=16/256)
  Pattern 2: g1_fixed  EE_POSE_GRADIENT (1.14-1.18× at N=16/256)

Generates grid.cuh from both repos, compiles each with `-Xptxas -v`,
parses register counts + spill stores + smem, writes diff report.

Requires a sibling worktree at $GRID_PRE_GLASS_REPO (default:
sibling of this repo named GRiD-A2R-pre-glass) checked out at the
pre-GLASS commit. Same setup as
test/benchmarks/run_multi_version.py's --columns pre_glass path.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python test/diagnostics/glass_vs_pre_glass_ptxas.py

This is historical / archival. The cuBLASDx removal v2.0 work no
longer needs this comparison; kept for future codegen regressions
where comparing-against-pre-glass might still be illuminating.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HEAD_REPO       = Path(__file__).resolve().parents[2]
PRE_GLASS_REPO  = Path(os.environ.get("GRID_PRE_GLASS_REPO",
                                       str(HEAD_REPO.parent / "GRiD-A2R-pre-glass")))
URDF_DIR        = Path.home() / ".cache/robot_descriptions"
URDFS = {
    "go2":    URDF_DIR / "unitree_ros/robots/go2_description/urdf/go2_description.urdf",
    "g1":     URDF_DIR / "unitree_ros/robots/g1_description/g1_29dof.urdf",
    "iiwa14": URDF_DIR / "drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf",
}

CUDA_ARCH = "120"
NVCC = "/usr/local/cuda/bin/nvcc"

# The per-exe bench cutover retired HEAD's monolithic timeGRiD_batch.cu. A per_algo_bench solo TU is an
# equivalent ptxas -v driver (it instantiates every kernel via init_grid_kernel_attrs). Materialize one for
# the HEAD_glass variant; the pre_glass variant still uses its frozen worktree's timeGRiD.cu, unchanged.
HEAD_BENCH_DIR = HEAD_REPO / "test/benchmarks/baselines/grid"
sys.path.insert(0, str(HEAD_REPO / "test/benchmarks"))
from per_algo_bench import _solo_batch_tu_source  # noqa: E402
_HEAD_DRIVER = Path("/tmp/glass_vs_pre_glass_head_driver.cu")
_HEAD_DRIVER.write_text(_solo_batch_tu_source("inverse_dynamics"))


def gen_grid_cuh(repo: Path, robot: str, urdf: Path, out_dir: Path) -> None:
    """Run codegen using the given repo's URDFParser + GRiDCodeGenerator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    code = f"""
import sys
sys.path.insert(0, "{repo}")
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
p = URDFParser()
r = p.parse("{urdf}", floating_base=False)
cg = GRiDCodeGenerator(r, 0, FILE_NAMESPACE="grid")
cg.gen_all_code(output_path="{out_dir}/grid.cuh")
"""
    venv_py = HEAD_REPO / ".venv/bin/python"
    rc = subprocess.run([str(venv_py), "-c", code], capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(
            f"codegen failed for {robot} with {repo}:\n{rc.stdout}\n{rc.stderr}")


def compile_ptxas_v(grid_cuh: Path, batch_cu: Path, glass_root: Path | None,
                    linalg_backend: str, build_dir: Path, ee_frame: str = "") -> str:
    """Compile timeGRiD_batch.cu with -Xptxas -v. Returns captured stderr."""
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "batch.o"

    cmd = [
        NVCC, "-std=c++17", "-c", "-o", str(obj), str(batch_cu),
        f"-DGRID_HEADER_FILE=\"{grid_cuh}\"",
        f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}",
        "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
        f"-I{batch_cu.parent}",
        f"-I{HEAD_BENCH_DIR}",   # timeGRiD_common.h for the HEAD solo driver (pre_glass -I wins for its own)
        f"-I{grid_cuh.parent}",
        "-Xptxas", "-v",
        "-Wno-deprecated-gpu-targets",
    ]
    if linalg_backend == "glass":
        cmd.append("-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS")
        if glass_root:
            cmd.extend([f"-I{glass_root}", f"-I{glass_root / 'src'}"])
    elif linalg_backend == "pre_glass":
        # pre_glass has its own builtin linalg (no GLASS); no extra defines.
        pass

    # GRID_TIMING_TEST_ITERS macro to make the .cu instantiate; default in the
    # batch source is 1, but its templates only fire when called. The TU also
    # needs IS_TIMING. Let's just match what the bench passes.
    cmd.append("-DGRID_TIMING_TEST_ITERS=1")

    rc = subprocess.run(cmd, capture_output=True, text=True)
    log = (build_dir / "ptxas.stderr.log")
    log.write_text(rc.stderr)
    if rc.returncode != 0:
        # Save stdout too for debugging.
        (build_dir / "compile.stdout.log").write_text(rc.stdout)
        raise RuntimeError(
            f"nvcc failed compiling {batch_cu.name} with backend {linalg_backend}: "
            f"see {log} (rc={rc.returncode}); stderr head:\n{rc.stderr[:2000]}")
    return rc.stderr


KERNEL_PATTERNS = {
    "forward_dynamics_kernel": "FD",
    "aba_kernel": "ABA",
    "end_effector_pose_gradient_kernel": "EE_POSE_GRAD",
    "inverse_dynamics_kernel": "ID",
    "crba_kernel": "CRBA",
    "minv_kernel": "Minv",
}


def parse_ptxas(stderr: str) -> dict[str, dict[str, int]]:
    """Pull register count / spill / smem for the kernels we care about."""
    out = {}
    # ptxas -v emits lines like:
    #   ptxas info    : Compiling entry function '_Z31inverse_dynamics_kernelIfEvPT_PKS0_iPKN4grid10robotModelIS0_EES0_i' for 'sm_120'
    #   ptxas info    : Function properties for _Z31inverse_dynamics_kernelIfE...
    #       512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    #   ptxas info    : Used 64 registers, 8192 bytes smem, 408 bytes cmem[0]
    pat_entry = re.compile(r"Compiling entry function '(_Z[^']+)'")
    pat_props = re.compile(r"Function properties for (_Z\w+)")
    pat_frame = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
    pat_used  = re.compile(r"Used (\d+) registers(?:, (\d+) bytes smem)?(?:, (\d+) bytes cmem)?")

    lines = stderr.splitlines()
    current = None
    info = {}
    for line in lines:
        m = pat_entry.search(line)
        if m: current = m.group(1); info[current] = {"mangled": current}; continue
        m = pat_frame.search(line)
        if m and current:
            info[current]["stack"]  = int(m.group(1))
            info[current]["spill_st"] = int(m.group(2))
            info[current]["spill_ld"] = int(m.group(3))
            continue
        m = pat_used.search(line)
        if m and current:
            info[current]["registers"] = int(m.group(1))
            info[current]["smem"]      = int(m.group(2) or 0)
            info[current]["cmem"]      = int(m.group(3) or 0)
            continue

    # Map mangled name → kernel label
    for mangled, d in info.items():
        for needle, label in KERNEL_PATTERNS.items():
            if needle in mangled:
                key = label
                # Multiple template variants — preserve mangled to distinguish
                if key in out:
                    # Pick the worst register count (likely the timing-mode variant)
                    if d.get("registers", 0) > out[key].get("registers", 0):
                        out[key] = d
                else:
                    out[key] = d
                break
    return out


def main():
    glass_root = HEAD_REPO / "GLASS"
    work = Path("/tmp/ptxas_diag")
    work.mkdir(exist_ok=True)
    results = {}

    cells = [
        ("go2",     URDFS["go2"]),
        ("g1",      URDFS["g1"]),
        ("iiwa14",  URDFS["iiwa14"]),  # control: known not-regressed for FD/ABA at batch
    ]

    for robot, urdf in cells:
        if not urdf.exists():
            print(f"  [skip] {robot}: URDF {urdf} not present")
            continue
        print(f"\n=== {robot} (URDF: {urdf}) ===")
        for variant_name, repo, batch_cu, linalg in [
            ("HEAD_glass",
             HEAD_REPO,
             _HEAD_DRIVER,
             "glass"),
            ("pre_glass",
             PRE_GLASS_REPO,
             # pre_glass had a single timeGRiD.cu — not split into _single/_batch
             PRE_GLASS_REPO / "test/benchmarks/baselines/grid/timeGRiD.cu",
             "pre_glass"),
        ]:
            if not batch_cu.exists():
                print(f"  [skip] {variant_name}: batch_cu not found at {batch_cu}")
                continue
            label = f"{robot}_{variant_name}"
            cell_dir = work / label
            cell_dir.mkdir(exist_ok=True)
            try:
                gen_grid_cuh(repo, robot, urdf, cell_dir)
                glass_root_local = (repo / "GLASS") if (repo / "GLASS").exists() else None
                stderr = compile_ptxas_v(
                    cell_dir / "grid.cuh",
                    batch_cu,
                    glass_root_local,
                    linalg,
                    cell_dir / "build",
                )
                info = parse_ptxas(stderr)
                results[label] = info
                print(f"  [{label}] {len(info)} kernels parsed; "
                      f"FD={info.get('FD',{}).get('registers','?')} "
                      f"ABA={info.get('ABA',{}).get('registers','?')} "
                      f"EE_GRAD={info.get('EE_POSE_GRAD',{}).get('registers','?')}")
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")
                results[label] = {"error": str(e)}

    # Diff table
    report = ["# ptxas -v diagnostic\n"]
    report.append("Per-kernel register count (R), shared mem (S, bytes), "
                  "spill stores (sp). Lower R is usually better for occupancy.\n")
    report.append("| Robot | Kernel | HEAD_glass | pre_glass | Δ regs | Δ smem | Δ spill |")
    report.append("|---|---|---|---|---|---|---|")
    for robot, _ in cells:
        h = results.get(f"{robot}_HEAD_glass", {})
        p = results.get(f"{robot}_pre_glass", {})
        if "error" in h or "error" in p:
            report.append(f"| {robot} | (err) | {h.get('error','-')} | {p.get('error','-')} | | | |")
            continue
        for label in ("FD", "ABA", "EE_POSE_GRAD", "ID", "CRBA", "Minv"):
            hi = h.get(label, {})
            pi = p.get(label, {})
            hr, pr = hi.get("registers"), pi.get("registers")
            hs, ps = hi.get("smem"), pi.get("smem")
            hsp, psp = hi.get("spill_st"), pi.get("spill_st")
            d_r = f"{(hr-pr):+d}" if (hr is not None and pr is not None) else "-"
            d_s = f"{(hs-ps):+d}" if (hs is not None and ps is not None) else "-"
            d_sp = f"{(hsp-psp):+d}" if (hsp is not None and psp is not None) else "-"
            head_s = f"R={hr},S={hs},sp={hsp}" if hi else "-"
            pre_s  = f"R={pr},S={ps},sp={psp}" if pi else "-"
            report.append(f"| {robot} | {label} | {head_s} | {pre_s} | {d_r} | {d_s} | {d_sp} |")
        report.append("")

    default_report = Path(__file__).resolve().parent / "results" / "glass_vs_pre_glass.md"
    out_md = Path(os.environ.get("GLASS_VS_PRE_GLASS_REPORT", str(default_report)))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(report))
    print(f"\nReport: {out_md}")


if __name__ == "__main__":
    main()
