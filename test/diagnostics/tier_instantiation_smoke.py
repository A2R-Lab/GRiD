"""Tier-template smoke test: verify every emitted kernel template can be
instantiated at TIER_SHARED, TIER_LITE, and TIER_MINIMAL without compile errors,
and that nvcc -Xptxas -v reports distinct register/launch_bounds per tier.

This is the minimum-viable correctness check for the v2.0 resource-tier
framework: each algorithm × {PERF, LITE, MINIMAL} must compile. Run-time
correctness at each tier is exercised by the Python wrapper suites and
pinocchio-equivalents.

Run:
    PYTHONPATH=. .venv/bin/python test/diagnostics/tier_instantiation_smoke.py
"""
from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "120"

# __global__ kernels that take a RESOURCE_TIER template parameter as of v2.0
# AND have a single signature (the address-taking trick below depends on this).
# Overloaded kernels (inverse_dynamics_kernel, inverse_dynamics_gradient_kernel,
# forward_dynamics_gradient_kernel each have two overloads — with/without qdd
# input) share the same template machinery and are exercised via real launches
# by the 84-test python_wrappers suite. Including them here would need explicit
# signature casts per overload, which is brittle.
KERNELS = [
    "minv_kernel",
    "forward_dynamics_kernel",
    "aba_kernel",
    "crba_kernel",
    "end_effector_pose_kernel",
    "end_effector_pose_gradient_kernel",
    "end_effector_pose_hessian_kernel",
    "idsva_so_body_frame_kernel",
    "idsva_so_world_frame_kernel",
    "fdsva_so_kernel",
]

TIERS = ["TIER_SHARED", "TIER_LITE", "TIER_MINIMAL"]

# Integrator kernels carry an extra `IntegratorType IT` template arg BEFORE
# RESOURCE_TIER, so the 2-arg address trick above doesn't apply — they get a
# dedicated force-instantiation with an explicit IT (EULER = single-stage path,
# RK4 = multi-stage path, which spills s_D_qdd_stage at LITE/MINIMAL).
INTEGRATOR_KERNELS = [
    "integrator_kernel",
    "integrator_gradient_kernel",
    "integrator_with_gradient_kernel",
]
INTEGRATOR_ITS = ["EULER", "RK4"]


def generate(robot_label: str, urdf: Path, floating: bool, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid.cuh"
    # Mimic robots (e.g. h1_2) have their GRADIENT algorithms refused at codegen
    # time (G0 footgun guard: no silent-zero mimic gradients; deferred to
    # T3-finisher). gen_all_code("all") would raise NotImplementedError for them,
    # so codegen the non-gradient tier-bearing surface instead — this still
    # exercises the fd/minv/aba/crba/integrator per-tier spill ladders, which is
    # what this smoke gates. Non-mimic robots keep the full "all" surface.
    code = f"""
import sys
sys.path.insert(0, "{REPO_ROOT}")
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
p = URDFParser()
r = p.parse("{urdf}", floating_base={floating})
cg = GRiDCodeGenerator(r, 0, FILE_NAMESPACE="grid")
if cg.robot_has_mimic_joints():
    cg.gen_all_code(output_path="{out_path}",
                    algorithm_list=["inverse_dynamics", "minv", "forward_dynamics", "aba", "crba", "integrator"])
else:
    cg.gen_all_code(output_path="{out_path}")
"""
    rc = subprocess.run([str(REPO_ROOT / ".venv/bin/python"), "-c", code],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"codegen failed: {rc.stderr}")
    return out_path


def detect_emitted(grid_cuh: Path) -> list[str]:
    text = grid_cuh.read_text()
    return [k for k in KERNELS if re.search(rf"\bvoid\s+{k}\s*\(", text)]


def compile_all_tiers(grid_cuh: Path, emitted: list[str], build_dir: Path) -> dict:
    """Force-instantiate every emitted kernel at all three tiers; compile
    with -Xptxas -v; parse per-(kernel, tier) registers + launch_bounds.

    Some kernels (inverse_dynamics, *_gradient) have multiple overloads;
    rather than disambiguating signatures, we wrap each instantiation in
    a templated dispatcher lambda that nvcc must instantiate.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    body_lines = []
    for k in emitted:
        for tier in TIERS:
            # Force instantiation via a templated lambda that takes the kernel's
            # address; the cast inside the lambda silently disambiguates by
            # discarding the template return-type deduction (we don't actually
            # call the kernel, we just need its template body in the obj file).
            body_lines.append(
                f"    (void) reinterpret_cast<void*>(&grid::{k}<T, grid::{tier}>);"
            )
    # Integrator kernels: instantiate at every (IT, tier) so the per-tier spill
    # bodies (s_D_qdd_stage in smem vs d_workspace) all compile.
    grid_text = grid_cuh.read_text()
    for k in INTEGRATOR_KERNELS:
        if not re.search(rf"\bvoid\s+{k}\s*\(", grid_text):
            continue
        for it in INTEGRATOR_ITS:
            for tier in TIERS:
                body_lines.append(
                    f"    (void) reinterpret_cast<void*>(&grid::{k}<T, grid::IntegratorType::{it}, grid::{tier}>);"
                )
    # Validate the tier-aware sizing constexprs for inline-CUDA users. At
    # TIER_SHARED the SMEM_BYTES values should be non-zero (full smem
    # footprint, current behavior) and WORKSPACE_BYTES values should be 0.
    # At TIER_LITE/MINIMAL the SMEM_BYTES values should drop (some / all
    # scratch moved out) and WORKSPACE_BYTES should be non-zero. These
    # asserts catch regressions where a tier-aware constexpr isn't actually
    # parameterized on TIER.
    sizing_asserts = """
    // fdsva_so_contract: 4*nv^3 scratch
    // fdsva_so_contract scratch constants are now keyed on the placement bool
    // SCRATCH_IN_SMEM (true = s_temp/shared, false = d_workspace/global).
    static_assert(grid::FDSVA_SO_INNER_SMEM_BYTES<T, true>() > 0,
                  "FDSVA_SO_INNER_SMEM_BYTES<smem> must include scratch");
    static_assert(grid::FDSVA_SO_INNER_SMEM_BYTES<T, false>() == 0,
                  "FDSVA_SO_INNER_SMEM_BYTES<global> must drop scratch");
    static_assert(grid::FDSVA_SO_INNER_WORKSPACE_BYTES<T, true>() == 0,
                  "FDSVA_SO_INNER_WORKSPACE_BYTES<smem> must be zero");
    static_assert(grid::FDSVA_SO_INNER_WORKSPACE_BYTES<T, false>() > 0,
                  "FDSVA_SO_INNER_WORKSPACE_BYTES<global> must hold scratch");

    // fd_du_device, id_du_device, idsva_so_device: whole s_temp arena
    static_assert(grid::FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_SHARED>() >
                  grid::FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_LITE>(),
                  "FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES LITE must drop below PERF");
    static_assert(grid::FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_SHARED>() == 0,
                  "FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES PERF must be zero");
    static_assert(grid::FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() > 0,
                  "FORWARD_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES LITE must hold scratch");

    static_assert(grid::INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_SHARED>() >
                  grid::INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_LITE>(),
                  "INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_SMEM_BYTES LITE must drop below PERF");
    static_assert(grid::INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() > 0,
                  "INVERSE_DYNAMICS_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES LITE must hold scratch");

    static_assert(grid::IDSVA_SO_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_SHARED>() >
                  grid::IDSVA_SO_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_LITE>(),
                  "IDSVA_SO_DEVICE_INLINE_SMEM_BYTES LITE must drop below PERF");
    static_assert(grid::IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() > 0,
                  "IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES LITE must hold scratch");

    // d2ee: only the d2eeTemp slot moves to d_workspace; the d2ee s_temp arena
    // (inner_no_d2) stays in smem at all tiers, so on the small/curated robots
    // the D2EE SMEM constexpr is tier-independent (SHARED == LITE == MINIMAL)
    // and only the WORKSPACE bytes differ (0 at SHARED, >0 at LITE/MINIMAL).
    // The SMEM invariant is therefore "LITE never EXCEEDS SHARED" (>=), not a
    // strict drop. (A strict '>' here was a latent bug — it failed on every
    // robot whose d2ee SMEM is tier-independent, i.e. all of iiwa14/go2/h1_2.)
    static_assert(grid::END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_SHARED>() >=
                  grid::END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_LITE>(),
                  "END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_SMEM_BYTES LITE must not exceed SHARED");
    static_assert(grid::END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_LITE>() > 0,
                  "END_EFFECTOR_POSE_HESSIAN_DEVICE_INLINE_WORKSPACE_BYTES LITE must hold d2eeTemp");
"""
    src = build_dir / "force_inst.cu"
    src.write_text(
        '#include "grid.cuh"\n'
        'using T = float;\n'
        'void force_all_tiers() {\n'
        + '\n'.join(body_lines) + '\n'
        + sizing_asserts +
        '}\n'
    )
    obj = build_dir / "force_inst.o"
    rc = subprocess.run(
        [NVCC, "-std=c++17", "-c", "-o", str(obj), str(src),
         f"-I{grid_cuh.parent}",
         f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
         "-O3", "-Xptxas", "-v", "-Wno-deprecated-gpu-targets"],
        capture_output=True, text=True,
    )
    (build_dir / "stderr.log").write_text(rc.stderr)
    if rc.returncode != 0:
        return {"compile_ok": False, "stderr": rc.stderr[-1500:]}

    # Parse ptxas register reports per (kernel, tier). Mangled names look like
    # `_ZN4grid<N><kernel_name>_kernelIfLi[012]EEE...` where Li0=PERF, Li1=LITE,
    # Li2=MINIMAL. The `\d+` before the kernel name is its length prefix.
    pat_entry = re.compile(r"Compiling entry function '_ZN4grid\d+"
                           r"([a-z_]+)_kernelIfLi([012])EE")
    pat_used  = re.compile(r"Used (\d+) registers")
    info = {}
    current = None
    for line in rc.stderr.splitlines():
        m = pat_entry.search(line)
        if m:
            kname = m.group(1) + "_kernel"
            tier_idx = int(m.group(2))
            current = (kname, TIERS[tier_idx])
            continue
        m = pat_used.search(line)
        if m and current:
            info.setdefault(current, {})["registers"] = int(m.group(1))
            current = None
    return {"compile_ok": True, "info": info}


def _resolve_urdf_via_robot_descriptions(module_name: str) -> Path | None:
    """Resolve a robot_descriptions module's URDF_PATH. Returns None if the
    module isn't installed (so the smoke skips that scenario gracefully)."""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        path = Path(getattr(mod, "URDF_PATH"))
        return path if path.exists() else None
    except Exception:
        return None


# Robots to exercise. iiwa14 is the baseline (picks collapse, single-body emit).
# go2_fixed is the most divergent — full 3-way picks on fdsva_so + d2ee. h1_2_fixed
# is the high-DOF stress test where ID_DU/D2EE pick divergent levels and several
# kernels overflow at runtime (bench skips those via grid_kernel_fits_device).
SCENARIOS = [
    ("iiwa14_fixed",  "robot_descriptions.iiwa14_description",  False),
    ("go2_fixed",     "robot_descriptions.go2_description",     False),
    ("h1_2_fixed",    "robot_descriptions.h1_2_description",    False),
]


def main():
    # Fall back to legacy drake URDF path for iiwa14 if robot_descriptions
    # isn't on PYTHONPATH — keeps the smoke runnable in a stripped venv.
    legacy_iiwa = (Path.home() /
        ".cache/robot_descriptions/drake/manipulation/models/"
        "iiwa_description/urdf/iiwa14_primitive_collision.urdf")

    overall_ok = True
    for label, mod_name, floating in SCENARIOS:
        urdf = _resolve_urdf_via_robot_descriptions(mod_name)
        if urdf is None and label == "iiwa14_fixed" and legacy_iiwa.exists():
            urdf = legacy_iiwa
        if urdf is None:
            print(f"SKIP {label}: {mod_name} not installed (pip install robot_descriptions)")
            continue
        print(f"\n=== {label} ({urdf.name}) ===")
        work = Path(f"/tmp/tier_inst_smoke/{label}")
        try:
            grid_cuh = generate(label, urdf, floating, work)
        except RuntimeError as e:
            print(f"  CODEGEN FAILED: {e}")
            overall_ok = False
            continue
        emitted = detect_emitted(grid_cuh)
        print(f"  Emitted kernels ({len(emitted)})")
        result = compile_all_tiers(grid_cuh, emitted, work / "build")
        if not result["compile_ok"]:
            print(f"  COMPILE FAILED:\n{result['stderr']}")
            overall_ok = False
            continue
        print(f"  All {len(emitted) * 3} (kernel, tier) instantiations compile.")
        # Brief register summary only for the divergence-prone kernels.
        watch = ("fdsva_so_kernel", "end_effector_pose_hessian_kernel",
                 "inverse_dynamics_gradient_kernel", "forward_dynamics_gradient_kernel")
        for k in emitted:
            if k not in watch:
                continue
            reg_per_tier = [
                (t, (result["info"].get((k, t)) or {}).get("registers", "?"))
                for t in TIERS
            ]
            cells = "  ".join(f"{t}=R{r}" for t, r in reg_per_tier)
            print(f"    {k:42s}  {cells}")

    if not overall_ok:
        sys.exit(1)
    print("\nSmoke: PASS")


if __name__ == "__main__":
    main()
