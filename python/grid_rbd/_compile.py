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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


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

    For development installs (`pip install -e python/`), the parent of the
    package's parent IS the repo root. For sdist installs once we publish to
    PyPI, the repo isn't present and we ship the codegen submodules with the
    sdist; the path resolution is different. For now, only the editable path
    is implemented.
    """
    # python/grid_rbd/_compile.py  →  repo_root = .../python/..
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
    cg = GRiDCodeGenerator(robot, debug_mode, FILE_NAMESPACE=file_namespace)

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
    # family (frame_jacobian / frame_jacobian_dot / osc_inertia). These are NOT
    # part of the default "all" profile (kept opt-in so the bench/default header
    # stays byte-identical), but the grid_rbd surface binds them, so we request
    # them explicitly here. The codegen emits `#define GRID_HAS_FRAME_JACOBIAN`
    # which gates the wrapper's frame_jacobian C-ABI symbols. mimic robots refuse
    # gradient algos inside gen_all_code; that refusal is unchanged (the opt-in
    # frame family is non-gradient, so this addition is mimic-safe).
    with contextlib.redirect_stdout(io.StringIO()):
        cg.gen_all_code(
            output_path=str(out_path),
            fixed_target_name=fixed_target_name,
            algorithm_list=["all", "frame_jacobian",
                            "frame_jacobian_dot", "osc_inertia"],
            enable_floating_second_order=True,
            enable_idsva_so_world_frame=options.get("floating_base", False),
        )

    if not out_path.exists():
        raise RuntimeError(
            f"GRiDCodeGenerator.gen_all_code() did not produce {out_path}"
        )

    # Capture per-robot constants so the Python side doesn't have to dlopen
    # the .so just to read NUM_JOINTS / NUM_VEL / NUM_EES.
    return {
        "num_joints": robot.get_num_pos(),
        "num_vel": robot.get_num_vel(),
        "num_ees": robot.get_total_leaf_nodes(),
        "floating_base": bool(robot.floating_base),
    }


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
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc failed (exit {result.returncode}). Build log:\n"
            + log_path.read_text()
        )


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
    compile_so(wrapper_cu, so_path, cuda_arch=cuda_arch,
               max_batch=max_batch, glass_root=glass_root,
               torch_op_key=torch_op_key)

    # Persist meta.json
    meta["cuda_arch"] = cuda_arch
    meta["max_batch"] = max_batch
    (target_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta
