"""Codegen + compile pipeline.

`generate_and_compile(urdf_path, options, target_dir)` does:

  1. Parse URDF with URDFParser.
  2. Run GRiDCodeGenerator.gen_all_code() to produce grid.cuh.
  3. Copy the robot-agnostic wrapper.cu boilerplate into target_dir.
  4. Invoke nvcc to compile (grid.cuh included by wrapper.cu) into robot.so.
  5. Write a meta.json next to robot.so capturing the per-robot constants
     (NUM_JOINTS / NUM_VEL / NUM_EES / floating_base) so the Python side
     can populate RobotHandle without re-importing the .so.

Errors raised by any step propagate as RuntimeError with the build.log
attached.
"""
from __future__ import annotations

import contextlib
import importlib.resources
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# Package logger. Quiet by default (no handler) — the host app opts in via
# logging.basicConfig() / its own handler. We use INFO for the compile notice
# so first-run register_robot() latency isn't mistaken for a hang.
_log = logging.getLogger("grid_rbd")


_NVCC_DEFAULT_FLAGS = [
    "-std=c++17",
    "-O3",
    "-ftz=true",
    "-prec-div=false",
    "-prec-sqrt=false",
    "--shared",
    "--compiler-options=-fPIC",
    # We INTENTIONALLY do NOT set -fvisibility=hidden here — the
    # Runner dlsym's our `extern "C"` symbols, which must be exported.
]


def find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc not found on PATH. grid-rbd requires the CUDA Toolkit at "
            "register_robot() time (used to compile the per-robot library)."
        )
    return nvcc


def repo_root() -> Path | None:
    """Locate the GRiD repo this package was installed from.

    For development installs (`pip install -e bindings/`), the parent of the
    package's parent IS the repo root. For sdist installs once we publish to
    PyPI, the repo isn't present and we ship the codegen submodules with the
    sdist; the path resolution is different. For now, only the editable path
    is implemented.
    """
    # bindings/grid_rbd/_compile.py  →  repo_root = .../bindings/..
    pkg = Path(__file__).resolve().parent
    candidate = pkg.parent.parent
    if (candidate / "GRiDCodeGenerator").exists() and (candidate / "URDFParser").exists():
        return candidate
    return None


def generate_grid_cuh(urdf_path: Path, options: dict[str, Any], out_path: Path) -> None:
    """Run URDFParser + GRiDCodeGenerator to produce grid.cuh at out_path.

    `options` carries the codegen-affecting knobs (floating_base, EE names,
    shared-mem target, linalg backend, etc.). Cosmetic options (cache_dir,
    force_rebuild) are filtered out by the caller before this is invoked.
    """
    # Ensure the GRiD submodules are importable. For editable installs, add
    # the repo root to sys.path so URDFParser / GRiDCodeGenerator resolve.
    root = repo_root()
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from URDFParser import URDFParser
    from GRiDCodeGenerator import GRiDCodeGenerator

    parser = URDFParser()
    robot = parser.parse(
        str(urdf_path),
        floating_base=options.get("floating_base", False),
    )

    file_namespace = options.get("file_namespace", "grid")
    debug_mode = options.get("debug_mode", 0)
    # fp64 (Phase 8): options["dtype"] in {"float32","float64"} selects the
    # compute precision. dtype="float64" flips the codegen shared-mem T-size to 8
    # so the spill-tier picks re-derive at the true fp64 footprint. Default
    # float32 -> codegen dtype "float" -> byte-identical fp32 path. The dtype is
    # in `options` so it re-keys the cache (fp32 vs fp64 .so coexist).
    codegen_dtype = "double" if options.get("dtype") == "float64" else "float"

    # Joint dynamics (viscous damping + Coulomb friction). options["use_joint_dynamics"]
    # (default absent/False) gates the codegen `USE_JOINT_DYNAMICS` flag, which emits
    # the joint-local bias `tau -= damping*qd + friction*sign(qd)` in the inverse_dynamics
    # / forward_dynamics / aba value paths. Emitted ONLY when the flag is on AND the robot
    # declares nonzero damping/friction; default-off keeps the header byte-identical (and
    # consistent with the bare-Pinocchio CUDA-equivalence oracle, which ignores
    # model.damping/friction). Injected into `options` (and thus the cache key) ONLY when
    # True, so a damped .so never collides with the historical no-op .so.
    use_joint_dynamics = bool(options.get("use_joint_dynamics", False))
    cg = GRiDCodeGenerator(robot, debug_mode, FILE_NAMESPACE=file_namespace,
                           dtype=codegen_dtype, USE_JOINT_DYNAMICS=use_joint_dynamics)

    # D.4 / Phase 5: runtime-mutable inertia table. options["runtime_inertia"]
    # (default absent/False) gates the codegen `runtime_inertia` flag (emits the
    # d_inertia_params table + on-device 6x6 rebuild + grid::set_inertia_params
    # host mutator). Default-off keeps the baked header byte-identical, so it is
    # injected into `options` (and thus the cache key) ONLY when True.
    runtime_inertia = bool(options.get("runtime_inertia", False))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Plumb ee_joint_names → fixed_target_name. The codegen expects a single
    # joint name string (it then keys the EE on that fixed joint); the
    # default empty string ⇒ codegen uses all leaf nodes (default EEs).
    # We expose `ee_joint_names` as a list because callers will frequently
    # want to think in those terms; if multiple are passed we currently
    # honor only the first (multi-target is a v2 concern). Lists matter for
    # the cache key — passing the same list always lands on the same .so.
    fixed_target_name = ""
    ee_joint_names = options.get("ee_joint_names") or []
    if ee_joint_names:
        fixed_target_name = ee_joint_names[0]

    # gen_all_code accepts output_path directly; redirect stdout to swallow
    # the chatty per-stage printouts. enable_floating_second_order=True so
    # idsva_so / fdsva_so are always available — user paid for the compile,
    # may as well include the methods.
    #
    # algorithm_list = the full "all" profile PLUS the opt-in frame_jacobian
    # family (frame_jacobian / frame_jacobian_dot / osc_inertia) AND
    # integrator_hessian (the plant_step_hessian s_d2AB surface). These are NOT
    # part of the default "all" profile (kept opt-in so the bench/default header
    # stays byte-identical), but the grid_rbd surface binds them, so we request
    # them explicitly here. The codegen emits `#define GRID_HAS_FRAME_JACOBIAN`
    # and `#define GRID_PLANT_HAS_STEP_HESSIAN`, which gate the wrapper's
    # corresponding C-ABI symbols. integrator_hessian pulls in fdsva_so +
    # integrator and is fixed-base only (the device fn static_asserts floating +
    # RK out; the kernel is simply not emitted for a floating base). mimic robots
    # refuse gradient algos inside gen_all_code; that refusal is unchanged (the
    # opt-in frame family is non-gradient, so this addition is mimic-safe).
    # Subset-build: by DEFAULT request the full "all" profile PLUS the opt-in
    # extras the grid_rbd surface binds (this is the byte-identical historical
    # list). When the caller threads a non-default `algorithm_list` through
    # `options` (register_robot(algorithm_list=...)), request exactly that set
    # instead — GRiDCodeGenerator._normalize_codegen_algorithms expands its
    # transitive deps, and the wrapper's per-algo GRID_HAS_* macros (emitted to 0
    # for the un-requested cores) ship clean rc=3 stubs for them. The default
    # (None) path keeps the SAME list, so a default register_robot is
    # byte-identical and reuses its existing .so (mirrors the inject-only-when-set
    # discipline of use_joint_dynamics / runtime_inertia / dtype).
    _DEFAULT_ALGORITHM_LIST = ["all", "frame_jacobian",
                               "frame_jacobian_dot", "osc_inertia",
                               "end_effector_pose_runtime",
                               "end_effector_pose_gradient_runtime",
                               "integrator_hessian"]
    requested_algos = options.get("algorithm_list")
    algorithm_list = list(requested_algos) if requested_algos else _DEFAULT_ALGORITHM_LIST
    with contextlib.redirect_stdout(io.StringIO()):
        cg.gen_all_code(
            output_path=str(out_path),
            fixed_target_name=fixed_target_name,
            algorithm_list=algorithm_list,
            enable_floating_second_order=True,
            enable_idsva_so_world_frame=options.get("floating_base", False),
            runtime_inertia=runtime_inertia,
        )

    if not out_path.exists():
        raise RuntimeError(
            f"GRiDCodeGenerator.gen_all_code() did not produce {out_path}"
        )

    # Capture per-robot constants so the Python side doesn't have to dlopen
    # the .so just to read NUM_JOINTS / NUM_VEL / NUM_EES.
    # joint_names (index == joint id) + leaf_jids let the handle's runtime-target
    # list API resolve ee_joint_names -> jids (mirrors RBDReference
    # select_end_effector_joints) without a robot model on the Python side.
    import math

    def _jsafe(v):
        # ±inf is not valid JSON; serialize unbounded / unspecified limits as null.
        if v is None:
            return None
        try:
            return None if math.isinf(float(v)) else float(v)
        except (TypeError, ValueError):
            return None

    joint_names = []
    # per-joint limits (index == jid), surfaced on the handle as metadata only
    # (not consumed by any kernel). pos = [lower, upper], vel/effort = scalar.
    joint_pos_limits = []
    joint_vel_limits = []
    joint_effort_limits = []
    for jid in range(robot.get_num_joints()):
        joint = robot.get_joint_by_id(jid)
        joint_names.append(joint.get_name() if joint is not None else "")
        lim = robot.get_joint_limits_by_id(jid) or []
        joint_pos_limits.append([_jsafe(lim[0]), _jsafe(lim[1])] if len(lim) == 2 else None)
        joint_vel_limits.append(_jsafe(robot.get_velocity_limit_by_id(jid)))
        joint_effort_limits.append(_jsafe(robot.get_effort_limit_by_id(jid)))
    meta = {
        "num_joints": robot.get_num_pos(),
        "num_vel": robot.get_num_vel(),
        "num_ees": robot.get_total_leaf_nodes(),
        "floating_base": bool(robot.floating_base),
        "joint_names": joint_names,
        "leaf_jids": [int(j) for j in robot.get_leaf_nodes()],
        "joint_pos_limits": joint_pos_limits,
        "joint_vel_limits": joint_vel_limits,
        "joint_effort_limits": joint_effort_limits,
    }
    # D.4 / Phase 5: when the mutable-inertia table is generated, persist the
    # BAKED 10-param-per-body table so the handle can expose it (fetch-then-mutate
    # via set_inertia_params). Layout mirrors gen_init_inertia_params /
    # init_inertia_params EXACTLY: bodies 1..N (the base body 0 is dropped, like
    # the I-region's Imats[1:]), each a length-10 [m, h(3)=m*c, I_O(6)] vector in
    # the frozen regressor basis. Flat row-major: pi[0..N-1] -> 10*N floats.
    if runtime_inertia:
        params = robot.get_inertia_params_ordered_by_id()[1:]  # drop base body
        meta["runtime_inertia"] = True
        meta["inertia_params"] = [[float(v) for v in pi] for pi in params]
    if use_joint_dynamics:
        meta["use_joint_dynamics"] = True
    return meta


def copy_wrapper_template(target_dir: Path) -> Path:
    """Copy the wrapper.cu boilerplate alongside grid.cuh."""
    src = Path(__file__).parent / "wrapper_template.cu"
    dst = target_dir / "wrapper.cu"
    shutil.copyfile(src, dst)
    return dst


def _jax_ffi_include_dir() -> Path | None:
    """Return JAX's FFI header include path if jax is installed, else None."""
    try:
        from jax import ffi as jax_ffi
        return Path(jax_ffi.include_dir())
    except Exception:
        return None


def _torch_build_flags() -> dict | None:
    """Return torch include/lib paths + ABI flag if torch is installed, else None.

    Mirrors _jax_ffi_include_dir(): the torch custom-op headers
    (torch/extension.h) bake the torch C++ ABI into the .so, so we must use
    torch's own include paths AND its _GLIBCXX_USE_CXX11_ABI setting, or the
    TORCH_LIBRARY symbols won't be ABI-compatible at torch.ops.load_library().
    """
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
        return {
            "includes": [Path(p) for p in include_paths()],
            "libdirs": [Path(p) for p in library_paths()],
            "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
            "version": torch.__version__,
        }
    except Exception:
        return None


def compile_so(
    wrapper_cu: Path,
    out_so: Path,
    cuda_arch: int,
    max_batch: int = 256,
    glass_root: Path | None = None,
    extra_flags: list[str] | None = None,
    enable_jax_ffi: bool = True,
    enable_torch: bool = True,
    torch_op_key: str | None = None,
    t_double: bool = False,
    runtime_inertia: bool = False,
) -> None:
    """Invoke nvcc to build wrapper.cu → robot.so.

    wrapper.cu must include "grid.cuh" from its own directory.

    When `enable_jax_ffi=True` (default) and JAX is installed, the .so will
    additionally export JAX FFI handler symbols (grid_rbd_jax_*). The
    Python side picks these up via dlsym in grid_rbd.jax.register_robot.
    If JAX isn't installed at compile time, the JAX FFI block is skipped
    (the .so is still fully functional via the plain C ABI).
    """
    nvcc = find_nvcc()
    arch = f"sm_{cuda_arch}"
    cmd = [nvcc] + _NVCC_DEFAULT_FLAGS + [
        f"-gencode=arch=compute_{cuda_arch},code={arch}",
        f"-DGRID_RBD_MAX_BATCH={max_batch}",
        f"-I{wrapper_cu.parent}",  # so #include "grid.cuh" resolves
        "-o", str(out_so),
        str(wrapper_cu),
    ]
    if glass_root:
        cmd.extend([f"-I{glass_root}", f"-I{glass_root / 'src'}"])

    # fp64 (Phase 8): flip the wrapper's `using T` to double. The grid.cuh must
    # have been generated with the matching dtype="double" codegen knob (so its
    # spill tiers are sized for sizeof(double)). JAX/torch FFI blocks are
    # suppressed in the wrapper for fp64 (#if !GRID_WRAPPER_T_DOUBLE), so we also
    # skip their -D flags below to avoid pulling their includes pointlessly.
    if t_double:
        cmd.append("-DGRID_WRAPPER_T_DOUBLE")
        enable_jax_ffi = False
        enable_torch = False

    # D.4 / Phase 5: runtime-mutable inertia. The grid.cuh must have been
    # generated with runtime_inertia=True (so grid::set_inertia_params exists);
    # this -D gates the wrapper's grid_rbd_set_inertia_params C-ABI symbol on it.
    if runtime_inertia:
        cmd.append("-DGRID_RBD_RUNTIME_INERTIA")

    # JAX FFI handlers: optionally enabled. When jax is available, point
    # nvcc at its FFI include dir and define GRID_RBD_WITH_JAX so the
    # wrapper template emits its handler block.
    if enable_jax_ffi:
        jax_inc = _jax_ffi_include_dir()
        if jax_inc and jax_inc.exists():
            cmd.extend([
                "-DGRID_RBD_WITH_JAX=1",
                f"-I{jax_inc}",
                "--expt-relaxed-constexpr",  # required by xla/ffi/api headers
            ])

    # PyTorch custom ops: optionally enabled. When torch is available, point
    # nvcc at its include/lib dirs, match its CXX11 ABI, and define
    # GRID_RBD_WITH_TORCH so the wrapper emits its op block. The op-library
    # name is keyed by the cache_key so two robots don't collide.
    if enable_torch:
        tflags = _torch_build_flags()
        if tflags is not None:
            cmd.append("-DGRID_RBD_WITH_TORCH=1")
            for inc in tflags["includes"]:
                cmd.append(f"-I{inc}")
            for ld in tflags["libdirs"]:
                # -rpath must go through the linker (nvcc rejects bare -Wl,...).
                cmd.extend([f"-L{ld}", "-Xlinker", f"-rpath,{ld}"])
            cmd.extend(["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"])
            cmd.append(f"-D_GLIBCXX_USE_CXX11_ABI={tflags['cxx11_abi']}")
            if "--expt-relaxed-constexpr" not in cmd:
                cmd.append("--expt-relaxed-constexpr")
            key = (torch_op_key or "default")
            # op-namespace token must be a valid C identifier (hex prefix is).
            cmd.append(f"-DGRID_RBD_TORCH_KEY={key}")

    if extra_flags:
        cmd.extend(extra_flags)

    log_path = out_so.with_suffix(".build.log")
    # First-run compile can take seconds–minutes (single-block fully-unrolled
    # kernels are cicc-bound). Emit a one-line notice so the wait isn't read as
    # a hang, and point at the live build log.
    _log.info("grid_rbd: compiling %s for %s (first run; nvcc, may take "
              "seconds–minutes) — log: %s", out_so.name, arch, log_path)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc failed (exit {result.returncode}). Build log:\n"
            + log_path.read_text()
        )
    _log.info("grid_rbd: built %s (build log: %s)", out_so.name, log_path)


def generate_and_compile(
    urdf_path: Path,
    options: dict[str, Any],
    target_dir: Path,
    cuda_arch: int,
    max_batch: int = 256,
) -> dict[str, Any]:
    """End-to-end: produce grid.cuh + wrapper.cu + robot.so in target_dir.

    Returns the meta dict (num_joints/num_vel/num_ees/floating_base).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    cuh_path = target_dir / "grid.cuh"
    meta = generate_grid_cuh(urdf_path, options, cuh_path)

    wrapper_cu = copy_wrapper_template(target_dir)

    # GLASS submodule path — only used in editable installs. sdist installs
    # ship the GLASS headers inside the package data (TODO).
    glass_root = None
    root = repo_root()
    if root and (root / "GLASS").exists():
        glass_root = root / "GLASS"

    so_path = target_dir / "robot.so"
    # The torch op-library namespace is keyed by the cache_key (== entry dir
    # name) so two robots registered in one process don't collide on op names.
    # Prefix with 'k' to guarantee a valid C identifier (hex may start 0-9).
    torch_op_key = "k" + target_dir.name[:12]
    t_double = options.get("dtype") == "float64"
    # Subset-build: the JAX / torch FFI handler blocks call grid::*_kernel for the
    # CORE algos directly and are NOT per-algo gated, so a reduced-profile header
    # (missing some core kernel) would fail to compile them. Those surfaces are the
    # full-profile fp32 path; the subset feature is a numpy-backend big-robot
    # compile-cost win. So when a non-default `algorithm_list` is requested, suppress
    # JAX/torch FFI for that .so (the numpy C-ABI is the subset surface). The DEFAULT
    # (no algorithm_list) build is unchanged — JAX/torch stay enabled, byte-identical.
    is_subset = bool(options.get("algorithm_list"))
    compile_so(wrapper_cu, so_path, cuda_arch=cuda_arch,
               max_batch=max_batch, glass_root=glass_root,
               torch_op_key=torch_op_key, t_double=t_double,
               enable_jax_ffi=not is_subset, enable_torch=not is_subset,
               runtime_inertia=bool(options.get("runtime_inertia", False)))

    # Persist meta.json
    meta["cuda_arch"] = cuda_arch
    meta["max_batch"] = max_batch
    # fp64 (Phase 8): record the element dtype so the handle/Runner picks the
    # matching numpy buffer type (RunnerF64 for float64) without dlopening.
    meta["dtype"] = "float64" if t_double else "float32"
    (target_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta
