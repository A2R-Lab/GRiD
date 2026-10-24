"""Codegen + compile pipeline (two content-keyed stages).

`generate_sources(urdf_path, options, target_dir)`:
  1. Parse URDF with URDFParser.
  2. Run GRiDCodeGenerator.gen_all_code() to produce grid.cuh.
  3. Copy the robot-agnostic wrapper.cu boilerplate into target_dir.

`compile_sources(target_dir, meta, options, ...)`:
  4. Invoke nvcc to compile (grid.cuh included by wrapper.cu) into robot.so.
  5. Write a meta.json next to robot.so capturing the per-robot constants
     (NUM_JOINTS / NUM_VEL / NUM_EES / floating_base) so the Python side
     can populate RobotHandle without re-importing the .so.

warm_robot's two-stage store calls the halves directly so it can content-key
between them (codegen inputs -> grid.cuh key; source bytes -> .so key).
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
import re
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
    # (no --shared here: units are compiled with -c and host-linked into the .so)
    "--compiler-options=-fPIC",
    # We INTENTIONALLY do NOT set -fvisibility=hidden here — the
    # Runner dlsym's our `extern "C"` symbols, which must be exported.
]


def _resolve_launch_config_robot(urdf_path: str) -> str:
    """Map a URDF filename stem to its grid_assets/launch_configs/<robot> key.

    grid_assets/launch_configs/ is keyed by canonical robot id (iiwa14, go2, g1, h2_plus), but the
    URDF filename stem is often longer (iiwa14_primitive_collision, g1_29dof). Without
    this, the binding looked up grid_assets/launch_configs/iiwa14_primitive_collision/ (a MISS) and
    fell back to the conservative (TIER_SHARED, MAX_PERF_LEVEL_THREADS) default for every
    algo — i.e. it never applied the autotuned per-algo {tier, threads}. Match the stem
    exactly first, else by the longest grid_assets/launch_configs/<name> the stem starts with. Returns
    the bare stem when nothing matches (an un-tuned robot stays on the safe fallback)."""
    stem = Path(urdf_path).stem
    try:
        from grid_codegen.GRiDCodeGenerator import _launch_configs_dir
        lc = Path(_launch_configs_dir())
        if (lc / stem).is_dir():
            return stem
        cands = [d.name for d in lc.iterdir()
                 if d.is_dir() and (stem == d.name or stem.startswith(d.name))]
        if cands:
            return max(cands, key=len)  # longest = most specific match
    except Exception:
        pass
    return stem


def find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc not found on PATH: compiling a robot library needs the CUDA Toolkit at register_robot() time. "
            "Install the toolkit, or add its bin directory (usually /usr/local/cuda/bin) to PATH, and register again. "
            "A cache built elsewhere loads through grid_rbd.get_robot without nvcc."
        )
    return nvcc


def repo_root() -> Path | None:
    """Editable source root, or None for an installed wheel."""
    from grid_codegen.resources import checkout_root
    return checkout_root()


def configuration_layout_from_robot(robot):
    """Serialize independent (kind, q_start, v_start, nq, nv) joint blocks.

    Keep metadata generation here: _compile.py participates in BOTH cache keys,
    so a layout change cannot reuse an artifact's old metadata. Mimic joints
    share their source's coordinates and must not appear twice.
    """
    from ._configuration import validate_configuration_layout
    blocks = []
    for joint in robot.joints:
        if getattr(joint, "is_mimic", False) or joint.get_num_dof() == 0:
            continue
        qi = robot.get_joint_index_q(joint.jid)
        vi = robot.get_joint_index_v(joint.jid)
        qi = qi if isinstance(qi, list) else [qi]
        vi = vi if isinstance(vi, list) else [vi]
        kind = joint.jtype if joint.jtype in ("floating", "spherical") else "euclidean"
        blocks.append((kind, int(qi[0]), int(vi[0]), len(qi), len(vi)))
    return validate_configuration_layout(sorted(blocks, key=lambda b: b[1]),
                                         robot.get_num_pos(), robot.get_num_vel())


def generate_grid_cuh(urdf_path: Path, options: dict[str, Any], out_path: Path) -> None:
    """Run URDFParser + GRiDCodeGenerator to produce grid.cuh at out_path.

    `options` carries the codegen-affecting knobs (floating_base, EE names,
    shared-mem target, linalg backend, etc.). Cosmetic options (cache_dir,
    force_rebuild) are filtered out by the caller before this is invoked.
    """
    # Ensure the GRiD submodules are importable. For editable installs, add
    # the repo root to sys.path so URDFParser / GRiDCodeGenerator resolve.
    root = repo_root()
    if root:
        # repo root resolves GRiDCodeGenerator; external/ resolves the peer
        # submodules (URDFParser / RBDReference / GLASS live under external/).
        for _p in (str(root), str(root / "external")):
            if _p not in sys.path:
                sys.path.insert(0, _p)

    from URDFParser import URDFParser
    from grid_codegen import GRiDCodeGenerator

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
    # Smooth friction (2026-10-21): options["friction_model"] ("sign" default, "tanh")
    # and options["friction_width"] (the tanh width in rad/s: one scalar or one value
    # per v-slot) feed the codegen FRICTION_MODEL / FRICTION_WIDTH knobs. register_robot
    # puts them in options (and thus the cache key) only when non-default, so every
    # existing cache entry stays valid and the sign-model header is byte-identical.
    friction_model = str(options.get("friction_model", "sign"))
    friction_width = options.get("friction_width", 0.03)
    # A1b launch-config bake: the grid_assets/launch_configs/<robot>/<gpu>.json dir is keyed
    # by the CANONICAL robot id (e.g. "iiwa14", "go2", "g1"), which the URDF FILENAME
    # stem often EXCEEDS (iiwa14_primitive_collision, g1_29dof). Resolve the stem to its
    # launch_configs key (exact, else longest-prefix) so codegen bakes the autotuned
    # per-algo {tier,threads}; without this it MISSED and fell back to conservative
    # defaults. (No options override exists: warm_robot builds options from an
    # allowlist — thread a kwarg through it first if custom ids ever need one.)
    launch_config_robot = _resolve_launch_config_robot(urdf_path)
    # The binding IS the jax/torch FFI launch path, whose per-algo thread optimum
    # differs from the C++/host one (the SAME kernel: e.g. iiwa14 fd host-best=128 but
    # FFI-best=768 for batch-to-land). Bake the "ffi" profile (ffi_bases, autotune_ffi.py)
    # so adopters get the FFI-fast config by default; per-algo fallback to host `bases`
    # keeps un-FFI-tuned algos on the safe pick. Override with launch_config_profile.
    launch_config_profile = options.get("launch_config_profile", "ffi")
    # W15: the GPU profile is DEVICE-keyed (warm_robot resolves it from the
    # detected/explicit cuda_arch and puts it in the cache key); fall back to
    # the same selection here so a direct generate_grid_cuh call agrees.
    from grid_codegen.launch_config import select_launch_config_gpu
    launch_config_gpu = options.get("launch_config_gpu") or select_launch_config_gpu(
        launch_config_robot, options.get("cuda_arch"))
    cg = GRiDCodeGenerator(robot, debug_mode, FILE_NAMESPACE=file_namespace,
                           dtype=codegen_dtype, USE_JOINT_DYNAMICS=use_joint_dynamics,
                           FRICTION_MODEL=friction_model, FRICTION_WIDTH=friction_width,
                           LAUNCH_CONFIG_ROBOT=launch_config_robot,
                           LAUNCH_CONFIG_PROFILE=launch_config_profile,
                           LAUNCH_CONFIG_GPU=launch_config_gpu)

    # D.4 / Phase 5: runtime-mutable inertia table. options["runtime_inertia"]
    # (default absent/False) gates the codegen `runtime_inertia` flag (emits the
    # d_inertia_params table + on-device 6x6 rebuild + grid::set_inertia_params
    # host mutator). Default-off keeps the baked header byte-identical, so it is
    # injected into `options` (and thus the cache key) ONLY when True.
    runtime_inertia = bool(options.get("runtime_inertia", False))

    # runtime_transform (mirror of runtime_inertia): options["runtime_transform"]
    # gates the codegen `runtime_transform` flag (emits the d_transform_params
    # table + on-device Xfixed rebuild + grid::set_transform_params host mutator,
    # with the general-rpy DENSE X pattern baked). Default-off keeps the baked
    # header byte-identical; injected into the cache key ONLY when True.
    runtime_transform = bool(options.get("runtime_transform", False))
    # runtime_joint_dynamics (C5, mirror of runtime_inertia): options gates the
    # codegen flag (emits the d_joint_dynamics_params table + grid::set_joint_dynamics_params
    # host mutator; the id/fd/aba/*_gradient bias reads the folded per-v-slot coeff from
    # it). Default-off keeps the header byte-identical; in the cache key ONLY when True.
    runtime_joint_dynamics = bool(options.get("runtime_joint_dynamics", False))

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
        if len(ee_joint_names) > 1:
            # Explicit capability, not a silent drop (audit W08/W11): the baked
            # EE family is single-target; extra names are ignored here. The
            # runtime multi-target surface is end_effector_pose_runtime.
            import warnings
            warnings.warn(
                f"ee_joint_names={list(ee_joint_names)!r}: the baked end-effector "
                f"family is single-target; only {ee_joint_names[0]!r} is honored. "
                "Use handle.end_effector_pose_runtime(q, ee_joint_names=[...]) for "
                "several targets at once.", UserWarning, stacklevel=3)

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
    # integrator and emits for BOTH bases (floating routes to the SE(3)-retract
    # Hessian, gen_integrator_hessian_device_floating); only multi-stage RK
    # static_asserts out (clean-break deferral). mimic robots
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
    # enable_mujoco_kernels=False builds a PIN-ONLY .so: the mjx (MuJoCo
    # output-convention) C-ABI entry points are dropped and, crucially, the mjx
    # kernel twins are never instantiated. On a big floating-base humanoid those
    # twins are still the largest kernels -- the mjx second-order twins were far
    # larger than their pin counterparts (idsva_so_world_frame was 28x raw), but the
    # epilogues have since been block-parallelized (go2-floating: idsva_so 2.42x,
    # fdsva_so 1.41x pin). Opting out lets a pin-only consumer (GATO/PDDP
    # 2nd-order DDP, or anyone not using the MuJoCo convention) build a g1 .so in
    # ~33 min at ~11 GB peak instead of exhausting a 62 GB box. Default True keeps
    # the historical behavior byte-identical (inject-only-when-set discipline).
    #
    # Deliberately resolves to an EXPLICIT True/False here rather than passing None and
    # letting gen_all_code consult GRID_ENABLE_MUJOCO_KERNELS. The compiled .so is cached
    # under canonical_options(options); an env var that silently flipped the build without
    # changing that key would hand back a stale .so built the other way. Binding consumers
    # pass the option; the env var is a codegen-session convenience (the test suite).
    enable_mujoco_kernels = options.get("enable_mujoco_kernels", True)
    # Multi-contact f_ext (wrapper window 2): contact_frames is a list of URDF
    # FIXED-JOINT names naming the contact frames (same convention as the baked
    # f_ext_body family). Resolved HERE against the live parse (codegen is the
    # only place the URDF is open) into [{name, jid, offset}] specs; passed to
    # gen_all_code (bakes the f_ext_body* device family + fc plant controls +
    # GRID_HAS_CONTACT_FRAMES) and persisted to meta so the handle can echo the
    # registered frames at runtime. Default None = byte-identical build.
    cost_frames = options.get("cost_frames")
    contact_frame_names = options.get("contact_frames")
    contact_frame_specs = None
    if contact_frame_names:
        from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
        contact_frame_specs = contact_frames_from_urdf(robot, list(contact_frame_names))
    with contextlib.redirect_stdout(io.StringIO()):
        cg.gen_all_code(
            output_path=str(out_path),
            fixed_target_name=fixed_target_name,
            algorithm_list=algorithm_list,
            enable_floating_second_order=True,
            enable_mujoco_kernels=enable_mujoco_kernels,
            # Defer the world-frame emission decision to gen_all_code's default
            # (None -> _idsva_so_use_world_frame): world-frame for floating,
            # spherical, OR high-DOF fixed-base (NV >= threshold). Hardcoding
            # floating_base here dropped the world-frame inner on high-DOF
            # fixed-base robots (g1/h1_2/h2_plus) whose idsva_so_device still
            # dispatches there -> undefined idsva_so_world_frame_inner.
            enable_idsva_so_world_frame=None,
            cost_frames=cost_frames,
            runtime_inertia=runtime_inertia,
            runtime_transform=runtime_transform,
            runtime_joint_dynamics=runtime_joint_dynamics,
            enable_contact_runtime=bool(options.get("enable_contact_runtime", False)),
            contact_frames=contact_frame_specs,
            # The wrapper units compile against the plant block whatever the algorithm
            # list says, so a bindings header always carries it (E1 keeps the
            # kinematics-only drop to generator and CLI headers).
            emit_plant=True,
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
        # A single named fixed target exposes ONE EE (the named flange) — mirror the
        # codegen's GRID_RBD_NUM_EES so meta matches the .so's grid_rbd_num_ees().
        # "" / "all" keep the all-leaf count.
        "num_ees": (1 if (fixed_target_name and fixed_target_name != "all")
                    else robot.get_total_leaf_nodes()),
        "floating_base": bool(robot.floating_base),
        # Canonical launch-config robot key (URDF stem -> tuned-JSON dir). Lets the
        # handle locate grid_assets/launch_configs/<key>/<gpu>.json for the E6 profile overlay.
        "launch_config_robot": launch_config_robot,
        "launch_config_gpu": launch_config_gpu,   # the profile actually baked (W15)
        "joint_names": joint_names,
        "leaf_jids": [int(j) for j in robot.get_leaf_nodes()],
        "joint_pos_limits": joint_pos_limits,
        "joint_vel_limits": joint_vel_limits,
        "joint_effort_limits": joint_effort_limits,
        # A2 (N2.2, 2026-09-08): what this .so actually contains. algorithm_list
        # is the REQUESTED build list (the _DEFAULT full profile when the caller
        # passed None); generated_algorithms is the POST-dep-expansion emit set
        # the codegen actually produced — handle.capabilities() keys on it.
        "algorithm_list": list(algorithm_list),
        "generated_algorithms": sorted(cg.generated_algorithms),
        "enable_mujoco_kernels": bool(enable_mujoco_kernels),
    }
    meta["configuration_layout"] = configuration_layout_from_robot(robot)
    # Multi-contact f_ext: echo the registered contact frames (registration
    # order == the contact_fext f_c column order) so the handle can validate
    # shapes and report frames without reopening the URDF.
    if contact_frame_specs:
        meta["contact_frames"] = [
            {"name": c["name"], "jid": int(c["jid"]),
             "offset": [float(v) for v in c["offset"]]}
            for c in contact_frame_specs
        ]
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
        # attach_tool needs joint name -> inertia row. inertia_params is ordered by
        # get_links_ordered_by_id()[1:] (base dropped), so map each joint to the row
        # of its CHILD link (the body that joint moves). Robust vs jid != lid ordering.
        row_of_link = {l.get_name(): (r - 1)
                       for r, l in enumerate(robot.get_links_ordered_by_id())}  # base -> -1
        j2row = {}
        for jid in range(robot.get_num_joints()):
            jo = robot.get_joint_by_id(jid)
            if jo is None:
                continue
            row = row_of_link.get(jo.get_child())
            if row is not None and row >= 0:
                j2row[jo.get_name()] = int(row)
        meta["inertia_row_by_joint_name"] = j2row
    # runtime_transform: persist the BAKED 6-param-per-joint origin table so the
    # handle can fetch-then-mutate via set_transform_params. Layout mirrors
    # gen_init_transform_params / init_transform_params: ALL joints 0..NB-1, each
    # a [x,y,z,roll,pitch,yaw] vector. Flat: op[0..NB-1] -> 6*NB floats.
    if runtime_transform:
        oparams = robot.get_origin_params_ordered_by_id()
        meta["runtime_transform"] = True
        meta["transform_params"] = [[float(v) for v in op] for op in oparams]
    if use_joint_dynamics:
        meta["use_joint_dynamics"] = True
        meta["friction_model"] = friction_model
    if cost_frames:
        meta["cost_frames"] = [nm for nm, _, _ in cg._cost_frames]
    # runtime_joint_dynamics (C5): persist the BAKED alpha-folded per-v-slot
    # damping/friction (each length nv) so handle.joint_damping/.joint_friction echo
    # EXACTLY what init_joint_dynamics_params wrote, and set_joint_dynamics can fetch-
    # then-mutate. Built by the codegen's own fold (get_joint_dynamics_baked) so meta
    # and the device init agree by construction.
    if runtime_joint_dynamics:
        b_vslot, f_vslot = cg.get_joint_dynamics_baked()   # two length-nv folded lists
        meta["runtime_joint_dynamics"] = True
        meta["joint_damping"]  = [float(b) for b in b_vslot]
        meta["joint_friction"] = [float(f) for f in f_vslot]
        if cg._jd_tanh_active():
            # the table's third row: the per-v-slot tanh width the device table is
            # initialized with (set_joint_dynamics(friction_width=) fetch-then-mutate).
            meta["joint_friction_width"] = [float(w) for w in cg.get_friction_width_baked()]
    return meta


def copy_wrapper_template(target_dir: Path) -> Path:
    """Copy the wrapper.cu boilerplate alongside grid.cuh."""
    src = Path(__file__).parent / "wrapper_template.cu"
    dst = target_dir / "wrapper.cu"
    shutil.copyfile(src, dst)
    # The host-only header every unit of the multi-TU build includes (2026-10-09);
    # copied beside wrapper.cu so `-I<target_dir>` resolves it like grid.cuh.
    shutil.copyfile(Path(__file__).parent / "wrapper_common.cuh", target_dir / "wrapper_common.cuh")
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


def _required_cxx_std(torch_includes) -> str:
    """The C++ standard the wrapper must be compiled with when torch is present.

    Release check 2026-09-24: `pip install -e ".[torch]"` on a clean checkout pulls a
    torch whose ATen.h guard demands C++20 (`#error C++20 or later ... ATen`, torch
    2.14), while the wrapper compiled with -std=c++17 → every torch-enabled artifact
    failed to build. Parse the guard rather than pin a version so older (c++17) and
    newer torch both work. `GRID_RBD_CXX_STD` overrides (a cache-key knob).
    """
    import os
    import re
    forced = os.environ.get("GRID_RBD_CXX_STD")
    if forced:
        return forced
    need = 201703
    for inc in torch_includes:
        aten = Path(inc) / "ATen" / "ATen.h"
        if aten.exists():
            m = re.search(r"__cplusplus\s*<\s*(\d{6})L", aten.read_text(errors="ignore"))
            if m:
                need = max(need, int(m.group(1)))
            break
    return "c++20" if need >= 202002 else "c++17"


def _mjx_signature_flags(cuh_path: Path) -> list[str]:
    """Derive the wrapper's host-template SIGNATURE flags from the generated header.

    grid.cuh emits each HOST launcher either as ``<..., KIND, RESOURCE_TIER>``
    (fixed base — and, per algo, mimic/skew/spherical robots) or as
    ``<..., KIND, MUJOCO_OUTPUT, RESOURCE_TIER>`` (floating base). The extra
    template parameter is keyed on FLOATING-ness with PER-ALGO exceptions
    (fdsva_so/fd_gradient skip it on mimic/skew; id_gradient also on spherical)
    and does NOT depend on enable_mujoco_kernels — so the wrapper cannot infer
    the signature from GRID_RBD_WITH_MUJOCO (that macro is the mjx-KERNELS
    gate: enable_mujoco_kernels AND floating AND non-mimic/skew). Keying the
    signature switch on it broke every pin-only floating build (TIER landed in
    the MUJOCO_OUTPUT bool slot: hard error at TIER=2, silently-wrong
    MUJOCO_OUTPUT=true at TIER=1).

    Ground truth is the emitted header itself: for each host fn the wrapper
    launches with an explicit RESOURCE_TIER, scan its template line and emit
    ``-DGRID_RBD_SIG_MJX_<FN>`` iff it carries MUJOCO_OUTPUT. No codegen rule
    is duplicated here, so the flags can never drift from the header
    (agent_debugging_guide §1m: dispatcher predicate must match emission gate).
    """
    text = cuh_path.read_text()
    lines = text.splitlines()
    # The EE launchers are renamed per-target; resolve the actual fn names from
    # the header's own #define block.
    ee_defs = dict(re.findall(r"#define (GRID_RBD_EE_POSE\w*) (\w+)", text))
    # H6: the {SIG suffix -> host launcher fn} pairs derive from ABI_SPECS
    # (sig_mjx_macro + grid_symbol) — the table that also drives the wrapper's
    # generated bodies, so this scan and the C code can't disagree about which
    # launchers carry the MUJOCO_OUTPUT slot. EE launchers are per-target
    # macros; resolve their concrete fn from the header's own #define block.
    from grid_codegen.abi_specs import ABI_SPECS
    fns = {}
    for spec in ABI_SPECS.values():
        if not spec.sig_mjx_macro:
            continue
        suffix = spec.sig_mjx_macro.removeprefix("GRID_RBD_SIG_MJX_")
        sym = (spec.grid_symbol or ("grid::" + spec.key)).removeprefix("grid::")
        fns[suffix] = ee_defs.get(sym) if sym.startswith("GRID_RBD_") else sym
    flags: list[str] = []
    for suffix, fn in fns.items():
        if not fn:
            continue  # EE launcher not emitted for this build
        pat = re.compile(r"\bvoid " + re.escape(fn) + r"\(gridData")
        for i, line in enumerate(lines):
            if pat.search(line):
                # template line sits a couple of lines above (past __host__ etc.)
                for j in range(i - 1, max(i - 5, -1), -1):
                    if "template <" in lines[j]:
                        if "MUJOCO_OUTPUT" in lines[j]:
                            flags.append(f"-DGRID_RBD_SIG_MJX_{suffix}")
                        break
                break
    return flags


def compile_so(
    wrapper_cu: Path,
    out_so: Path,
    cuda_arch: int,
    max_batch: int = 256,
    glass_root: Path | None = None,
    extra_flags: list[str] | None = None,
    torch_op_key: str | None = None,
    t_double: bool = False,
    runtime_inertia: bool = False,
    runtime_transform: bool = False,
    runtime_joint_dynamics: bool = False,
    object_cache_dir: Path | None = None,
    reuse_objects: bool = True,
    cuda_archs: list[int] | None = None,
) -> None:
    """Invoke nvcc to build wrapper.cu → robot.so.

    wrapper.cu must include "grid.cuh" from its own directory.

    ``cuda_archs`` (default: ``GRID_RBD_CUDA_ARCHS``, see
    ``_cache.effective_cuda_archs``) builds a fat binary: one
    ``-gencode=arch=compute_X,code=sm_X`` per listed arch (plus ``cuda_arch``
    itself), with ``-DGRID_RBD_ARCH=cuda_arch`` unchanged (the arch the launch
    configs were selected for; the other arches run the same kernels with that
    tuning) and ``-DGRID_RBD_ARCHS=86:120`` (colon-separated) naming every
    embedded arch for the wrapper's device check. Unset = today's single-arch
    command, byte for byte.

    ``object_cache_dir`` enables the per-unit object cache (see _build_units):
    a unit whose exact compile inputs were compiled before links from the
    cached object instead of re-running nvcc; ``reuse_objects=False``
    (force_rebuild) recompiles every unit but still refreshes the cache.

    When JAX is installed, the .so additionally exports JAX FFI handler
    symbols (grid_rbd_jax_*); the Python side picks these up via dlsym in
    grid_rbd.jax.register_robot. Likewise the torch custom-op block compiles
    in iff torch is installed. Either absent at compile time just skips its
    block (the .so is still fully functional via the plain C ABI).
    """
    nvcc = find_nvcc()
    from . import _cache
    archs = _cache.effective_cuda_archs(cuda_arch, cuda_archs)
    gencode_archs = archs if archs is not None else [cuda_arch]
    arch = "+".join(f"sm_{a}" for a in gencode_archs)
    core_flags: list[str] = []   # flags only the core unit is compiled with
    cmd = [nvcc] + _NVCC_DEFAULT_FLAGS + [
        f"-gencode=arch=compute_{a},code=sm_{a}" for a in gencode_archs
    ] + [
        f"-DGRID_RBD_MAX_BATCH={max_batch}",
        f"-DGRID_RBD_ARCH={cuda_arch}",
        # Library embedding: generated gpuAssert must never cudaDeviceReset+exit()
        # the host interpreter — errors land in the sticky slot and surface as rc
        # from grid_rbd_init/close (direct-CUDA consumers keep the fail-fast
        # default; this define is bindings-builds only).
        "-DGRID_GPUERRCHK_NO_EXIT",
        f"-I{wrapper_cu.parent}",  # so #include "grid.cuh" resolves
    ]
    if archs is not None:
        # Fat binary: every embedded arch, for the wrapper's device-cc check
        # (GRID_RBD_ARCH alone names the tuning arch). Colon-separated: nvcc
        # splits a -D value on commas into several macros, and the token must
        # stay free of shell metacharacters; the template stringifies it.
        cmd.append("-DGRID_RBD_ARCHS=" + ":".join(str(a) for a in archs))
    if glass_root:
        cmd.extend([f"-I{glass_root}", f"-I{glass_root / 'src'}"])

    # fp64: flip the wrapper's `using T` to double. The grid.cuh must have been
    # generated with the matching dtype="double" codegen knob (so its spill tiers
    # are sized for sizeof(double)). Wave 2a: the jax/torch sections now follow T
    # (GRID_FFI_T / GRID_TORCH_DTYPE / data_ptr<T>), so their -D flags stay on —
    # an fp64 .so carries fp64 jax+torch surfaces.
    if t_double:
        cmd.append("-DGRID_WRAPPER_T_DOUBLE")

    # D.4 / Phase 5: runtime-mutable inertia. The grid.cuh must have been
    # generated with runtime_inertia=True (so grid::set_inertia_params exists);
    # this -D gates the wrapper's grid_rbd_set_inertia_params C-ABI symbol on it.
    if runtime_inertia:
        cmd.append("-DGRID_RBD_RUNTIME_INERTIA")

    # runtime_transform (mirror): the grid.cuh must have been generated with
    # runtime_transform=True (so grid::set_transform_params exists); this -D gates
    # the wrapper's grid_rbd_set_transform_params C-ABI symbol on it.
    if runtime_transform:
        cmd.append("-DGRID_RBD_RUNTIME_TRANSFORM")

    # runtime_joint_dynamics (C5, mirror): grid.cuh must have been generated with
    # runtime_joint_dynamics=True (so grid::set_joint_dynamics_params exists); this
    # -D gates the wrapper's grid_rbd_set_joint_dynamics_params C-ABI symbol on it.
    if runtime_joint_dynamics:
        cmd.append("-DGRID_RBD_RUNTIME_JOINT_DYNAMICS")

    # JAX FFI handlers: when jax is available, point nvcc at its FFI include
    # dir and define GRID_RBD_WITH_JAX so the wrapper template emits its
    # handler block.
    jax_inc = _jax_ffi_include_dir()
    if jax_inc and jax_inc.exists():
        cmd.extend([
            "-DGRID_RBD_WITH_JAX=1",
            f"-I{jax_inc}",
            "--expt-relaxed-constexpr",  # required by xla/ffi/api headers
        ])

    # PyTorch custom ops: when torch is available, point nvcc at its
    # include/lib dirs, match its CXX11 ABI, and define GRID_RBD_WITH_TORCH so
    # the wrapper emits its op block. The op-library name is keyed by the
    # cache_key so two robots don't collide.
    tflags = _torch_build_flags()
    if tflags is not None:
        cmd.append("-DGRID_RBD_WITH_TORCH=1")
        std = _required_cxx_std(tflags["includes"])
        cmd[:] = [f"-std={std}" if f == "-std=c++17" else f for f in cmd]
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
        # op-namespace token must be a valid C identifier (hex prefix is). Only
        # the core unit names the namespace (its TORCH_LIBRARY delegates to the
        # families' grid_rbd_torch_def_/impl_ functions), so the key is a
        # core-only flag: family objects stay identical across store entries.
        core_flags.append(f"-DGRID_RBD_TORCH_KEY={key}")

    if extra_flags:
        cmd.extend(extra_flags)

    _build_units(cmd, wrapper_cu.parent, out_so, arch, core_flags=core_flags,
                 object_cache_dir=object_cache_dir, reuse_objects=reuse_objects)


def _build_jobs() -> int:
    """Parallel unit compiles: GRID_RBD_BUILD_JOBS, else min(8, cores, free RAM / 3 GB)
    (a unit peaks at ~2.5 GB on a humanoid; 1 = sequential, still a smaller peak
    than the old single-unit build)."""
    env = os.environ.get("GRID_RBD_BUILD_JOBS")
    if env:
        return max(1, int(env))
    jobs = min(8, os.cpu_count() or 1)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                jobs = max(1, min(jobs, int(line.split()[1]) // (3 * 1024 * 1024)))
    except OSError:
        pass
    return jobs


def _split_compile_flags(unit_stem: str) -> list[str]:
    """Opt-in nvcc `--split-compile N` for the heaviest unit(s) (2026-10-10):
    GRID_RBD_SPLIT_COMPILE=N (0/unset = off) applies it to the units named in
    GRID_RBD_SPLIT_COMPILE_UNITS (comma list of family names, default
    `integrator_gradient` — the singleton that bounds every humanoid build).
    Measured 10-09: 3–4x wall on that unit, ~3x its RAM, and ptxas register
    allocation CHANGES (bit-different SASS), which is why it is opt-in and in
    the content key, never a default."""
    n = os.environ.get("GRID_RBD_SPLIT_COMPILE", "0").strip()
    if not n or n == "0":
        return []
    units = os.environ.get("GRID_RBD_SPLIT_COMPILE_UNITS", "integrator_gradient")
    wanted = {f"wrapper_{u.strip()}" for u in units.split(",") if u.strip()}
    return [f"--split-compile={int(n)}"] if unit_stem in wanted else []


def _unit_order(units: list[Path]) -> list[Path]:
    """Heavy families first (abi_specs.FAMILIES order), core last: the pool's
    makespan is the slowest unit, so start it first."""
    from grid_codegen.abi_specs import FAMILIES
    rank = {f"wrapper_{fam}.cu": i for i, fam in enumerate(FAMILIES)}
    return sorted(units, key=lambda p: rank.get(p.name, len(rank)))


def _unit_object_key(unit: Path, source_dir: Path, compile_cmd: list[str]) -> str:
    """Content key of one unit's object: exactly what nvcc sees compiling it —
    the unit text, the two headers every unit includes (grid.cuh,
    wrapper_common.cuh), the flag list with the staging dir's own -I removed
    (its CONTENT is hashed; its path differs per build), and the toolchain /
    ABI tags the store's content key uses (nvcc, host C++, GLASS, torch, jax).
    `-DGRID_RBD_TORCH_KEY` is in the core unit's flags only, so a family object
    is shared by every store entry with the same sources and flags."""
    from . import _cache
    import hashlib
    h = hashlib.sha256()
    src = str(source_dir)
    for a in compile_cmd[1:]:   # [0] = nvcc path (covered by _nvcc_identity)
        if a == f"-I{src}" or a.startswith(src + "/"):
            continue
        h.update(a.encode() + b"\0")
    for name in ("grid.cuh", "wrapper_common.cuh"):
        h.update((source_dir / name).read_bytes())
        h.update(b"\0")
    h.update(unit.read_bytes())
    h.update(f"nvcc={_cache._nvcc_identity()};host_cxx={_cache._host_cxx_tag()};"
             f"glass={_cache._glass_content_hash()};{_cache._torch_abi_tag()};"
             f"{_cache._jax_ffi_tag()};key_schema={_cache._KEY_SCHEMA}".encode())
    return h.hexdigest()


def _build_units(cmd: list[str], source_dir: Path, out_so: Path, arch: str,
                 core_flags: list[str] | None = None,
                 object_cache_dir: Path | None = None,
                 reuse_objects: bool = True) -> None:
    """Multi-TU build (2026-10-09): compile every `wrapper_*.cu` unit beside
    grid.cuh with `-c` in a bounded pool, then host-link the objects into
    robot.so. No -rdc: each unit's kernels are launched only from that unit.
    `cmd` is the full nvcc flag list; library/rpath flags are link-only;
    `core_flags` are added to the core unit's compile only (the torch op
    namespace key).

    Per-unit object cache (M4): with `object_cache_dir`, each unit's object is
    kept at `<object_cache_dir>/<unit>-<key>.o` under `_unit_object_key`; a
    unit whose key already exists links from the cached object with no nvcc
    (`reuse_objects=False` recompiles every unit and refreshes the cache). A
    wrapper edit therefore recompiles only the units whose text changed (core
    alone when only core text moved); a grid.cuh / wrapper_common.cuh / flag
    change recompiles every unit. Without a cache dir, objects are deleted
    after the link."""
    import concurrent.futures
    import threading
    import time

    units = _unit_order(sorted(source_dir.glob("wrapper_*.cu")))
    if not units:
        raise RuntimeError(f"no wrapper units beside {source_dir / 'grid.cuh'} (generate_sources writes them)")
    compile_cmd, link_flags = [], []
    i = 0
    while i < len(cmd):
        a = cmd[i]
        if a == "-Xlinker":
            link_flags.extend([a, cmd[i + 1]]); i += 2; continue
        if a.startswith(("-L", "-l")) and not a.startswith("-lineinfo"):
            link_flags.append(a)
        else:
            compile_cmd.append(a)
        i += 1
    nvcc = compile_cmd[0]
    jobs = _build_jobs()
    log_path = out_so.with_suffix(".build.log")
    build_dir = source_dir / "build"
    build_dir.mkdir(exist_ok=True)
    if object_cache_dir is not None:
        object_cache_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    _log.info("grid_rbd: compiling %s for %s (%d units, %d parallel; nvcc, may take "
              "seconds–minutes) — log: %s", out_so.name, arch, len(units), jobs, log_path)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(compile_cmd) + " -c <unit> -o <unit>.o   # x" + str(len(units))
                  + (" (+ " + " ".join(core_flags) + " for the core unit)" if core_flags else "") + "\n")
        log.write("$ " + " ".join([nvcc, "--shared", "-o", str(out_so), "<units>.o"] + link_flags) + "\n\n")

    def one(unit: Path) -> tuple[Path, int, float, str]:
        ucmd = compile_cmd + (list(core_flags or []) if unit.stem == "wrapper_core" else []) \
            + _split_compile_flags(unit.stem)
        cached = None
        if object_cache_dir is not None:
            cached = object_cache_dir / f"{unit.stem}-{_unit_object_key(unit, source_dir, ucmd)}.o"
            if reuse_objects and cached.is_file():
                os.utime(cached)   # a hit is a use: keeps it out of _prune_object_cache
                with lock, log_path.open("a") as log:
                    log.write(f"[{unit.stem}] cached {cached.name}\n")
                return cached, 0, 0.0, ""
        obj = build_dir / (unit.stem + ".o")
        ulog = build_dir / (unit.stem + ".log")
        t0 = time.monotonic()
        with ulog.open("w") as f:
            r = subprocess.run(ucmd + ["-c", str(unit), "-o", str(obj)], stdout=f, stderr=subprocess.STDOUT)
        secs = time.monotonic() - t0
        with lock, log_path.open("a") as log:
            log.write(f"[{unit.stem}] rc={r.returncode} {secs:.1f}s\n")
        if r.returncode == 0 and cached is not None:
            # publish atomically (another build may be producing the same key)
            tmp = cached.with_name(cached.name + f".{os.getpid()}.tmp")
            shutil.move(str(obj), str(tmp))
            os.replace(tmp, cached)
            obj = cached
        return obj, r.returncode, secs, ulog.read_text()

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(one, units))
    failed = [(obj, rc, out) for obj, rc, _, out in results if rc != 0]
    if failed:
        obj, rc, out = failed[0]
        raise RuntimeError(f"The CUDA compiler failed (nvcc exit {rc}) while building the {obj.stem} part of this robot's library. "
                f"Its output follows; an internal compiler error or out-of-memory usually means the build needs fewer "
                f"parallel units (GRID_RBD_BUILD_JOBS=1) or a smaller algorithm_list.\n{out}")
    objs = [str(obj) for obj, _, _, _ in results]
    link = [nvcc, "--shared", "-o", str(out_so)] + objs + link_flags
    t0 = time.monotonic()
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(link) + "\n")
        log.flush()
        r = subprocess.run(link, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[link] rc={r.returncode} {time.monotonic() - t0:.1f}s\n")
    if r.returncode != 0:
        raise RuntimeError(f"The CUDA linker failed (nvcc exit {r.returncode}) while producing this robot's library. The build log "
            f"follows; retry with force_rebuild=True to recompile every unit.\n" + log_path.read_text())
    for obj in objs:   # uncached objects: keep the store entry lean
        if object_cache_dir is None or Path(obj).parent != object_cache_dir:
            Path(obj).unlink(missing_ok=True)
    compiled = sum(1 for _, _, secs, _ in results if secs > 0)
    _log.info("grid_rbd: built %s from %d units (%d compiled, %d from the object cache; build log: %s)",
              out_so.name, len(units), compiled, len(units) - compiled, log_path)
    if object_cache_dir is not None:
        _prune_object_cache(object_cache_dir)


OBJECT_CACHE_MAX_AGE_DAYS = 0   # keep forever (user 2026-10-09); GRID_RBD_OBJECT_CACHE_DAYS=N opts in


def _prune_object_cache(object_cache_dir: Path, max_age_days: float = OBJECT_CACHE_MAX_AGE_DAYS) -> int:
    """Opt-in GC: drop cached unit objects not produced or linked from (mtime)
    in `max_age_days`. OFF by default (objects are kept forever, like store/
    entries — compile once, keep); `GRID_RBD_OBJECT_CACHE_DAYS=N` enables it.
    Every distinct build (robot x options x subset) adds its ~14 objects; an
    object is only ever a shortcut, so losing one costs that unit's compile and
    nothing else. Returns the number removed."""
    import time
    env = os.environ.get("GRID_RBD_OBJECT_CACHE_DAYS")
    if env is not None:
        max_age_days = float(env)
    if max_age_days <= 0:
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for obj in object_cache_dir.glob("wrapper_*.o"):
        try:
            if obj.stat().st_mtime < cutoff:
                obj.unlink()
                removed += 1
        except OSError:
            pass   # a concurrent build published/removed it
    if removed:
        _log.info("grid_rbd: pruned %d cached unit objects older than %g days from %s",
                  removed, max_age_days, object_cache_dir)
    return removed


def generate_sources(
    urdf_path: Path,
    options: dict[str, Any],
    target_dir: Path,
) -> dict[str, Any]:
    """The cheap CPU half: produce grid.cuh + wrapper.cu in target_dir.

    Split into its own stage (2026-08-19) so the two-stage
    content-addressed store can hash the generated sources and skip the nvcc
    half entirely when an identical build already exists.
    Returns the meta dict (num_joints/num_vel/num_ees/floating_base).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    meta = generate_grid_cuh(urdf_path, options, target_dir / "grid.cuh")
    wrapper = copy_wrapper_template(target_dir)
    # Multi-TU build: one unit per build family + the core, split from the
    # fenced template (bindings/grid_rbd/_units.py). wrapper.cu stays beside them
    # as the hashed source of truth; the units are what nvcc compiles.
    from . import _units
    for name, text in _units.split(wrapper.read_text()).items():
        (target_dir / f"wrapper_{name}.cu").write_text(text)
    return meta


def compile_sources(
    target_dir: Path,
    meta: dict[str, Any],
    options: dict[str, Any],
    cuda_arch: int,
    max_batch: int = 256,
    torch_op_key: str | None = None,
    object_cache_dir: Path | None = None,
    reuse_objects: bool = True,
) -> dict[str, Any]:
    """The nvcc half: compile target_dir's grid.cuh + wrapper units → robot.so
    and persist meta.json. Returns the completed meta dict. ``object_cache_dir``
    / ``reuse_objects`` drive the per-unit object cache (compile_so)."""
    cuh_path = target_dir / "grid.cuh"
    wrapper_cu = target_dir / "wrapper.cu"

    from grid_codegen.resources import resource_path
    glass_root = resource_path('GLASS')

    so_path = target_dir / "robot.so"
    # The torch op-library namespace is keyed by the cache_key (== the entry's
    # store-dir name; the two-stage flow passes it explicitly since it compiles
    # in a staging dir before publishing to store/<content_key>). Two robots
    # registered in one process therefore never collide on op names. Prefix
    # with 'k' to guarantee a valid C identifier (hex may start 0-9).
    if torch_op_key is None:
        torch_op_key = "k" + target_dir.name[:12]
    t_double = options.get("dtype") == "float64"
    # Subset-build: the JAX / torch FFI handler blocks are now per-CORE-algo gated
    # (each `grid::*_kernel`-calling handler + its def/impl sits inside
    # `#if GRID_HAS_<ALGO>`), mirroring the numpy C-ABI bodies. So a reduced-profile
    # header (missing some core kernel) compiles cleanly — the un-requested cores
    # drop out of the jax/torch surface, and the Python wrapper maps the resulting
    # missing-symbol AttributeError to the same clean "not built — add to
    # algorithm_list" subset error the numpy rc=3 path raises. JAX/torch are
    # therefore enabled by DEP AVAILABILITY only (compile_so adds the -D flags iff
    # the include dir / build flags are present), NOT by whether a subset was
    # requested. t_double no longer disables them: the jax/torch sections follow T
    # (Wave 2a), so an fp64 .so carries fp64 jax+torch surfaces.
    compile_so(wrapper_cu, so_path, cuda_arch=cuda_arch,
               max_batch=max_batch, glass_root=glass_root,
               torch_op_key=torch_op_key, t_double=t_double,
               runtime_inertia=bool(options.get("runtime_inertia", False)),
               runtime_transform=bool(options.get("runtime_transform", False)),
               runtime_joint_dynamics=bool(options.get("runtime_joint_dynamics", False)),
               object_cache_dir=object_cache_dir, reuse_objects=reuse_objects,
               # Host-template signature flags, derived from the header just
               # generated (floating builds carry a MUJOCO_OUTPUT template param
               # on most host launchers even when enable_mujoco_kernels=False).
               extra_flags=_mjx_signature_flags(cuh_path))

    # Persist meta.json
    meta["cuda_arch"] = cuda_arch
    # Multi-arch (opt-in, GRID_RBD_CUDA_ARCHS): every arch the fat binary
    # embeds. cuda_arch stays the arch whose launch configs it carries; the
    # others run the same kernels with that tuning (thread-count invariant, so
    # correct; performance may differ from an arch-native build). Absent from
    # a single-arch build's meta.
    from ._cache import effective_cuda_archs
    archs = effective_cuda_archs(cuda_arch)
    if archs is not None:
        meta["cuda_archs"] = archs
        meta["cuda_archs_tuning"] = (
            f"launch configs selected for sm_{cuda_arch}; the other embedded arches "
            f"run the same kernels with that tuning")
    meta["max_batch"] = max_batch
    # fp64 (Phase 8): record the element dtype so the handle/Runner picks the
    # matching numpy buffer type (RunnerF64 for float64) without dlopening.
    meta["dtype"] = "float64" if t_double else "float32"
    (target_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta
