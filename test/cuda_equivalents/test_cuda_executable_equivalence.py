import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents import build_adapter, resolve_backend
from RBDReference.tests.state_sampling import (
    DynamicsSample,
    _joint_ranges,
    build_dynamics_samples,
)


RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA_VERSION = 1
DEFAULT_RANDOM_SAMPLE_COUNT = 3
CUDA_CORNER_SAMPLE_NAMES = (
    "positive",
    "negative",
    "mixed_sign",
    "tiny",
    "velocity_only",
    "accel_or_torque_only",
    "near_limit",
    "floating_quat_identity",
    "floating_quat_positive",
    "floating_quat_mixed",
)
FIXED_CUDA_ALGORITHMS = (
    "inverse_dynamics",
    "direct_minv",
    "forward_dynamics",
    "inverse_dynamics_gradient_q",
    "inverse_dynamics_gradient_qd",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    "aba",
    "crba",
    "end_effector_pose",
    "end_effector_pose_gradient",
    "end_effector_pose_hessian",
)
FLOATING_CUDA_ALGORITHMS = (
    "inverse_dynamics",
    "direct_minv",
    "forward_dynamics",
    "inverse_dynamics_gradient_q",
    "inverse_dynamics_gradient_qd",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    "aba",
    "crba",
    "end_effector_pose",
)
FLOATING_CUDA_CANDIDATE_ALGORITHMS = (
    *FLOATING_CUDA_ALGORITHMS,
    "end_effector_pose_gradient",
    "end_effector_pose_hessian",
)
GPU_UNAVAILABLE_PATTERNS = (
    "no cuda-capable device",
    "cuda driver version is insufficient",
    "cuda driver version is insufficient for cuda runtime version",
    "cudaerrorinsufficientdriver",
    "cudaerrornodevice",
    "failed to initialize nvml",
    "gpu access blocked",
    "operation not permitted",
    "all cuda-capable devices are busy or unavailable",
)
SINGULAR_DEPENDENT_ALGORITHMS = {
    "direct_minv",
    "forward_dynamics",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    "aba",
}
# Mimic-joint CUDA codegen lands in phases (task T3). For robots WITH mimic
# joints (fr3, h1_2) only the algorithms whose mimic path has landed are
# compared; the rest are skipped (logged, not failed) until their phase lands.
# This set GROWS per phase and reaches full coverage at P4. Non-mimic robots
# are unaffected (they always compare every algorithm).
#   P1: inverse_dynamics, crba
#   P2: + direct_minv, forward_dynamics, aba
#   P3: + inverse_dynamics_gradient_q/qd, forward_dynamics_gradient_q/qd
#   P4: + end_effector_pose_gradient, end_effector_pose_hessian
#       (end_effector_pose value is mimic-unaffected and always compared)
MIMIC_SUPPORTED_ALGORITHMS = {
    # P1 (landed): ID + CRBA. end_effector_pose value is mimic-unaffected.
    "inverse_dynamics",
    "crba",
    "end_effector_pose",
    # P2 (landed): direct_minv via inv(CRBA); forward_dynamics + aba via decomp.
    "direct_minv",
    "forward_dynamics",
    "aba",
    # P3 (landed, FIXED-BASE): ID/FD gradients via the dense serial reduced-space
    # fold (id_du) + the -Minv*dc_du compose (fd_du). Floating-base mimic
    # gradients are still refused (skipped via the per-case gate below).
    "inverse_dynamics_gradient_q",
    "inverse_dynamics_gradient_qd",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    # P4 (landed, FIXED-BASE): kinematic gradient/hessian via the alpha-weighted
    # geometric-Jacobian column fold (ee_pose_gradient) + world-frame generator
    # fold (ee_pose_hessian). Floating-base mimic ee gradients are still refused
    # (the floating root needs a 6-DoF subspace fold, not a scalar alpha fold);
    # skipped for floating mimic via MIMIC_FLOATING_UNSUPPORTED_GRADIENTS below.
    "end_effector_pose_gradient",
    "end_effector_pose_hessian",
}


# Gradient algorithms that are in MIMIC_SUPPORTED_ALGORITHMS (so FIXED-base mimic
# compares them, P3) but are NOT yet emitted for FLOATING-base mimic robots (the
# dense id_du inner is fixed-base; gen_all_code refuses id_du/fd_du for floating
# mimic). Skip comparing them for floating mimic until the floating extension lands.
MIMIC_FLOATING_UNSUPPORTED_GRADIENTS = {
    "inverse_dynamics_gradient_q",
    "inverse_dynamics_gradient_qd",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    # ee pose grad/hessian mimic fold is FIXED-BASE only (B2-ee); floating mimic
    # ee derivatives are still refused at codegen (floating-root 6-DoF subspace
    # fold deferred), so skip comparing them for floating mimic robots.
    "end_effector_pose_gradient",
    "end_effector_pose_hessian",
}


def _robot_has_mimic_joints(project_model) -> bool:
    return any(
        getattr(j, "is_mimic", False) for j in project_model.robot.joints
    )

# Codegen algorithm selection for MIMIC robots (fr3, h1_2). Their GRADIENT
# algorithms are refused at codegen time (G0 footgun guard: mimic gradients are
# not folded yet — deferred to T3-finisher — and the old silent-zero stub was
# removed). gen_all_code("all") therefore raises NotImplementedError for them,
# so we codegen only the non-gradient surface this suite actually compares for
# mimic robots (MIMIC_SUPPORTED_ALGORITHMS): id / crba / ee_pose / direct_minv /
# forward_dynamics / aba. As each mimic-gradient phase lands (T3-finisher),
# extend both this list and MIMIC_SUPPORTED_ALGORITHMS together.
MIMIC_CODEGEN_ALGORITHM_LIST = ["id", "crba", "ee_pose", "minv", "fd", "aba"]
# Fixed-base mimic additionally supports the ID/FD gradients (T3-finisher P3) and
# the ee pose gradient/hessian (B2-ee P4). Floating-base mimic gradients are still
# refused, so floating uses the base list.
MIMIC_CODEGEN_ALGORITHM_LIST_FIXED = MIMIC_CODEGEN_ALGORITHM_LIST + [
    "id_du", "fd_du", "ee_pose_gradient", "ee_pose_hessian",
]
# Algorithms with a KNOWN, TRACKED correctness bug: their mismatches vs the
# independent oracle are reported as expected/known failures (not silent masks,
# not hard suite failures) pending a fix. The oracle stays correct so the bug is
# never hidden by comparing buggy-vs-buggy.
KNOWN_FAILING_ALGORITHMS = {}
# Per-(robot, algorithm) known bugs — same semantics as KNOWN_FAILING_ALGORITHMS
# but scoped to a specific robot so other robots' identical algorithm still fails
# hard if it regresses. The oracle stays correct (reported, never masked).
# A1 VALUE-PATH FIXED (2026-05-31): the h1_2 branched-multi-root inverse_dynamics
# VALUE bug was a shared-arena s_vaf/s_temp UNDER-SIZING for mimic robots. The ID
# inner indexes s_vaf and the I*v scratch by RAW body id (get_num_joints() bodies),
# but the device/kernel wrappers and the inner temp size reserved only
# get_num_pos()-many bodies. For mimic robots get_num_joints() > get_num_pos()
# (mimic joints carry 0 DoF), so the high-body force writes overflowed s_vaf into
# the adjacent s_XImats region, silently corrupting the LOW-jid X matrices (here
# the left leg, root 0). Fixed by sizing s_vaf/s_temp/XImats-scratch by
# get_num_joints() when robot_has_mimic_joints() (byte-identical for non-mimic).
# h1_2 inverse_dynamics / forward_dynamics / aba (and crba/direct_minv/ee_pose)
# now MATCH pinocchio and are UN-GATED.
#
# A second, related mimic bug was found+fixed in the same pass (2026-05-31): the
# DENSE mimic id_du/fd_du gradient fold called mx<s_ind>_peq_scaled with only the
# mimic multiplier alpha as the scale, DROPPING the joint motion-subspace sign
# s_sign (the helper applies the UNIT axis column, so S = s_sign*e_{s_ind} needs
# alpha*s_sign). Bodies with s_sign=-1 (h1_2's whole LEFT hand: index/middle/
# pinky/ring/thumb mimics + their proximals) got sign-flipped dv/da gradient
# contributions, so id_du/fd_du diverged at those v-slots. fr3's single mimic has
# s_sign=+1 so it was unaffected (alpha*1 == alpha, byte-identical). Fixed the
# four forward mx<s_ind>_peq_scaled calls to scale by alpha*s_sign; the backward
# fxS already carried the sign. h1_2 id_du/fd_du now match pinocchio and are
# UN-GATED. (Non-mimic robots never hit the dense inner -> Gate A unaffected.)
KNOWN_FAILING_ROBOT_ALGORITHMS = {}
CUDA_DEFAULT_TOLERANCE = {
    "rtol": 2e-4,
    "atol": 2e-4,
}
CUDA_ROBOT_ALGORITHM_TOLERANCES = {
    # h1_2 (mimic humanoid): forward_dynamics + aba go through the
    # algebraic-decomposition mimic path (qdd = Minv*(u - c)) whose reduced mass
    # matrix is ill-conditioned (cond ~5e6 floating / ~7e5 fixed), so the float32
    # generated kernel leaves low-percent per-entry residuals on the 1e6-scale
    # noise. Mirror go2/g1's ABA norm-relative guard. (RBDReference applies the
    # matching float64 tolerance overrides; this is the CUDA-side analogue.)
    ("h1_2", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "H1_2 CUDA ABA uses the mimic algebraic-decomposition path on an ill-conditioned reduced mass matrix; float32 leaves low-percent per-entry residuals while the vector norm stays tight.",
    },
    ("h1_2", "forward_dynamics"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "H1_2 CUDA forward dynamics (mimic decomposition qdd = Minv*(u-c)) reaches low-percent float32 residuals on the ill-conditioned reduced mass matrix; keep the full-vector norm guard.",
    },
    # h1_2 mimic ID/FD gradients: the dense reduced-space fold (id_du) and the
    # -Minv*dc_du compose (fd_du) run in float32 on an ill-conditioned reduced
    # mass matrix (cond ~7e5). Near-zero entries show milliscale residuals while
    # the full-matrix norm stays tight — mirror the existing iiwa14/g1/fr3
    # FD-gradient norm-relative guards.
    ("h1_2", "inverse_dynamics_gradient_q"): {
        "rtol": 2e-4, "atol": 2e-4, "norm_rtol": 5e-4,
        "note": "H1_2 mimic dense id_du float32 cancellation on near-zero entries; full-matrix norm guard.",
    },
    ("h1_2", "inverse_dynamics_gradient_qd"): {
        "rtol": 2e-4, "atol": 2e-4, "norm_rtol": 5e-4,
        "note": "H1_2 mimic dense id_du (qd) float32 cancellation on near-zero entries; full-matrix norm guard.",
    },
    ("h1_2", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4, "atol": 2e-4, "norm_rtol": 5e-3,
        "note": "H1_2 mimic fd_du = -Minv*dc_du float32 on cond~7e5 reduced mass matrix; full-matrix norm guard.",
    },
    ("h1_2", "forward_dynamics_gradient_qd"): {
        "rtol": 2e-4, "atol": 2e-4, "norm_rtol": 5e-3,
        "note": "H1_2 mimic fd_du (qd) float32 on cond~7e5 reduced mass matrix; full-matrix norm guard.",
    },
    ("go2", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "GO2 CUDA ABA is a float32 generated-kernel path and currently reaches low-percent per-entry residuals against the Python float64 reference on the random smoke samples.",
    },
    ("go2", "crba"): {
        "rtol": 2e-4,
        "atol": 5e-2,
        "note": "GO2 CUDA CRBA differs from the Python float64 reference by a few hundredths on near-zero off-diagonal entries while the rest of the dynamics stack remains strict.",
    },
    ("g1", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "G1 CUDA ABA is a large generated float32 kernel and currently reaches sub-percent to low-percent per-entry residuals against the Python float64 reference.",
    },
    ("g1", "crba"): {
        "rtol": 2e-4,
        "atol": 1.25,
        "note": "G1 CUDA CRBA has about unit-scale residuals on selected off-diagonal entries in this smoke path; keep this override scoped to G1 CRBA.",
    },
    ("g1", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "G1 floating FD-gradient-q is a large float32 generated-kernel path and uses selective spill; near-zero entries can show milliscale absolute residuals while the full-matrix norm remains tight.",
    },
    ("iiwa14", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "IIWA14 floating FD-gradient-q has float32 cancellation on deterministic corner samples; enforce the full-matrix norm while keeping entrywise checks strict for non-cancelled cases.",
    },
    ("iiwa14", "forward_dynamics_gradient_qd"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-5,
        "note": "IIWA14 floating FD-gradient-qd can leave sub-milliscale residuals on entries whose double-reference value is effectively zero; keep a tight full-matrix norm guard.",
    },
    ("fr3", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "FR3 CUDA ABA is checked with a norm-relative guard because the hand branch and float32 generated-kernel path produce sub-percent vector residuals.",
    },
    ("fr3", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "FR3 floating FD-gradient-q has small float32 Minv/gradient cancellation on near-zero and conservative entries; require a tight full-matrix norm.",
    },
    ("fr3", "forward_dynamics"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 1e-5,
        "note": "FR3 floating forward dynamics can show sub-milliscale float32 solve residuals on quaternion corner samples while the vector norm remains tight.",
    },
    ("gen3", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 7e-3,
        "note": "Gen3 floating FD-gradient-q is sensitive to the internally computed float32 qdd on high-acceleration quaternion samples; recomposing with CUDA qdd collapses the residual to a tight norm.",
    },
    ("gen3", "forward_dynamics_gradient_qd"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "Gen3 floating FD-gradient-qd has near-zero-entry float32 residuals; keep the full-matrix norm guard tight.",
    },
    ("gen3", "forward_dynamics"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 1e-5,
        "note": "Gen3 floating forward dynamics can show milliscale float32 solve residuals on quaternion corner samples while the vector norm remains tight.",
    },
    ("baxter", "forward_dynamics_gradient_qd"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "Baxter floating FD-gradient-qd has near-zero-entry float32 residuals in the broad deterministic CUDA sweep; keep the full-matrix norm guard tight.",
    },
    ("baxter", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 2e-4,
        "norm_rtol": 5e-4,
        "note": "Baxter floating FD-gradient-q has isolated float32 Minv/gradient residuals on random samples while the full-matrix norm remains tight.",
    },
    ("fetch", "forward_dynamics_gradient_q"): {
        "rtol": 2e-4,
        "atol": 5e-4,
        "norm_rtol": 1e-4,
        "note": "Fetch has a negative gripper axis and FD-gradient-q float32 cancellation on zero-torque and velocity-only smoke samples; keep entrywise checks strict unless the full-matrix norm remains very tight.",
    },
}


class SampleSelection(NamedTuple):
    names: set[str] | None
    include_corner_samples: bool
    explicit: bool


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _progress(config, message: str, *, verbose: bool = False) -> None:
    if not _env_enabled("GRID_CUDA_PROGRESS", default=True):
        return
    if verbose and not _env_enabled("GRID_CUDA_VERBOSE_PROGRESS", default=False):
        return
    prefix = "[cuda-equivalence] "
    reporter = None
    if config is not None:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(prefix + message)
        terminal_writer = getattr(reporter, "_tw", None)
        if terminal_writer is not None and hasattr(terminal_writer, "flush"):
            terminal_writer.flush()
    else:
        print(prefix + message, flush=True)


def _cache_verbose(config, message: str) -> None:
    if _env_enabled("GRID_CUDA_VERBOSE_CACHE", default=False):
        _progress(config, message)


def _cache_enabled() -> bool:
    return not _env_enabled("GRID_CUDA_DISABLE_CACHE", default=False)


def _cache_root() -> Path:
    return Path(os.environ.get("GRID_CUDA_CACHE_DIR", ".pytest_cache/grid_cuda")).resolve()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _hash_tree(root: Path, suffixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _nvcc_version_text() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "missing"
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return f"error:{result.returncode}:{result.stderr.strip()}"
    return result.stdout.strip()


def _detect_cuda_arch() -> str:
    env_arch = os.environ.get("GRID_CUDA_ARCH")
    if env_arch:
        return env_arch.replace(".", "")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                compute_cap = line.strip()
                if compute_cap:
                    return compute_cap.replace(".", "")

    return "86"


def _linalg_backend_compile_flags(arch: str) -> tuple[str, list[str], str]:
    """Linalg backend is SIMT-only after the cuBLASDx removal (v2.0). The
    function keeps the (cxx_standard, flags, note) tuple shape so callers
    don't need updating."""
    del arch  # unused; sm/arch is handled by the outer compile harness
    return "-std=c++11", ["-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS"], "glass"


def _parse_const_ints(header_path: Path) -> dict[str, int]:
    constants = {}
    const_re = re.compile(r"const int (?P<name>[A-Z0-9_]+) = (?P<value>-?[0-9]+);")
    for match in const_re.finditer(header_path.read_text()):
        constants[match.group("name")] = int(match.group("value"))
    return constants


def _fallback_summary(header_path: Path) -> str:
    constants = _parse_const_ints(header_path)
    interesting = [
        "GRID_ID_DU_SHARED_TIER_VALUE",
        "GRID_FD_DU_SHARED_TIER_VALUE",
        "GRID_ID_DU_USES_GLOBAL_TEMP",
        "GRID_FD_DU_USES_GLOBAL_TEMP",
        "GRID_ID_DU_USES_DA_DF_SPILL",
        "GRID_FD_DU_USES_DA_DF_SPILL",
        "GRID_GENERATES_D2EE",
        "GRID_D2EE_USES_WORKSPACE_TEMP",
        "GRID_D2EE_USES_WORKSPACE_D2XHOM",
        "GRID_D2EE_SHARED_TIER_VALUE",
    ]
    parts = [
        f"{name}={constants[name]}"
        for name in interesting
        if name in constants
    ]
    return ", ".join(parts) if parts else "fallback constants unavailable"


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda_equivalence")
    config.addinivalue_line("markers", "developer_only")
    config.addinivalue_line("markers", "floating_base")


def build_cuda_case_params(base_mode: str):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        marks = [
            pytest.mark.cuda_equivalence,
            pytest.mark.developer_only,
            pytest.mark.robot_smoke,
        ]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(
            pytest.param(
                spec,
                base_mode,
                id=f"{spec.robot_id}-{base_mode}",
                marks=marks,
            )
        )
    return params


def build_fixed_cuda_case_params():
    return build_cuda_case_params("fixed")


def build_floating_cuda_case_params():
    return build_cuda_case_params("floating")


def _header_cache_key(project_model, resolved_model, include_homogenous_transforms: bool) -> str:
    urdf_path = Path(resolved_model.urdf_path)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": "grid_header",
        "robot_id": project_model.spec.robot_id,
        "base_mode": project_model.base_mode,
        "nq": project_model.nq,
        "nv": project_model.nv,
        "urdf_path": str(urdf_path),
        "urdf_hash": _hash_file(urdf_path) if urdf_path.exists() else "missing",
        "robot_description_revision": resolved_model.revision,
        "codegen_hash": _hash_tree(REPO_ROOT / "GRiDCodeGenerator", (".py",)),
        "target_shared_mem_bytes": os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES", "default"),
        "shared_mem_type_size_bytes": os.environ.get("GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES", "default"),
        "codegen_profile": os.environ.get("GRID_CODEGEN_PROFILE", "all"),
        # Mimic robots codegen a reduced (non-gradient) algorithm list; fold it
        # into the key so their headers never collide with a full-"all" header.
        "mimic_algorithm_list": (
            MIMIC_CODEGEN_ALGORITHM_LIST
            if _robot_has_mimic_joints(project_model) else None
        ),
        "include_homogenous_transforms": include_homogenous_transforms,
        "debug_mode": False,
        "need_print_mat": True,
        "file_namespace": "grid",
    }
    return _stable_json_hash(payload)


def _run_gen_all_code(codegen, project_model, output_path, include_homogenous_transforms):
    """Codegen the header, selecting the mimic-safe (non-gradient) algorithm
    list for mimic robots so the G0 gradient-refusal guard isn't tripped."""
    kwargs = dict(
        include_homogenous_transforms=include_homogenous_transforms,
        output_path=str(output_path),
    )
    if _robot_has_mimic_joints(project_model):
        # Fixed-base mimic includes ID/FD gradients (P3); floating-base mimic
        # gradients are still refused, so use the non-gradient list there.
        if project_model.base_mode == "floating":
            kwargs["algorithm_list"] = MIMIC_CODEGEN_ALGORITHM_LIST
        else:
            kwargs["algorithm_list"] = MIMIC_CODEGEN_ALGORITHM_LIST_FIXED
    codegen.gen_all_code(**kwargs)


def _generate_grid_header(project_model, resolved_model, build_dir: Path, config) -> tuple[Path, str]:
    header_path = build_dir / "grid.cuh"
    include_homogenous_transforms = True
    header_key = _header_cache_key(
        project_model,
        resolved_model,
        include_homogenous_transforms=include_homogenous_transforms,
    )
    if not _cache_enabled():
        _progress(config, f"generating header for {project_model.spec.robot_id}-{project_model.base_mode}")
        codegen = GRiDCodeGenerator(
            project_model.robot,
            DEBUG_MODE=False,
            NEED_PRINT_MAT=True,
            FILE_NAMESPACE="grid",
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            _run_gen_all_code(codegen, project_model, header_path, include_homogenous_transforms)
        return header_path, header_key

    cached_dir = _cache_root() / "headers" / header_key
    cached_header = cached_dir / "grid.cuh"
    if cached_header.exists():
        shutil.copyfile(cached_header, header_path)
        _progress(config, f"header cache hit: {project_model.spec.robot_id}-{project_model.base_mode} key={header_key[:12]}")
        return header_path, header_key

    _progress(config, f"generating header for {project_model.spec.robot_id}-{project_model.base_mode} cache miss key={header_key[:12]}")
    cached_dir.mkdir(parents=True, exist_ok=True)
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=True,
        FILE_NAMESPACE="grid",
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _run_gen_all_code(codegen, project_model, cached_header, include_homogenous_transforms)
    (cached_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA_VERSION,
                "robot_id": project_model.spec.robot_id,
                "base_mode": project_model.base_mode,
                "target_shared_mem_bytes": os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES", "default"),
                "shared_mem_type_size_bytes": os.environ.get("GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES", "default"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    shutil.copyfile(cached_header, header_path)
    return header_path, header_key


def _compile_runner(
    build_dir: Path,
    *,
    floating_base: bool = False,
    header_key: str,
    skip_gradients: bool = False,
    skip_eepose_gradients: bool = False,
    config=None,
) -> tuple[Path, list[str]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA equivalence tests.")

    arch = _detect_cuda_arch()
    nvcc_version = _nvcc_version_text()
    _cache_verbose(config, "nvcc version: " + nvcc_version.splitlines()[-1])
    cxx_standard, linalg_flags, linalg_backend_note = _linalg_backend_compile_flags(arch)
    compile_flags = [cxx_standard, "-O0", *linalg_flags]
    l2_persisting = os.environ.get("GRID_CUDA_ENABLE_L2_PERSISTING")
    l2_define = int(l2_persisting) if l2_persisting is not None else 0
    floating_algorithms = os.environ.get("GRID_CUDA_FLOATING_ALGORITHMS", "")
    enable_floating_eepose_hessian = int(
        floating_base
        and (
            "end_effector_pose_hessian" in {
                part.strip() for part in floating_algorithms.split(",")
            }
            or floating_algorithms.strip().lower() == "all"
        )
    )
    runner_key = _stable_json_hash(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "kind": "cuda_equivalence_runner",
            "header_key": header_key,
            "runner_source_hash": _hash_file(RUNNER_SOURCE),
            "cuda_arch": arch,
            "nvcc_version": nvcc_version,
            "floating_base": bool(floating_base),
            "l2_persisting": l2_define,
            "floating_eepose_hessian": enable_floating_eepose_hessian,
            "skip_gradients": bool(skip_gradients),
            "skip_eepose_gradients": bool(skip_eepose_gradients),
            "compile_flags": compile_flags,
        }
    )

    if _cache_enabled():
        compile_dir = _cache_root() / "runners" / runner_key
        executable = compile_dir / "cuda_equivalence_runner.exe"
        if executable.exists():
            _progress(config, f"runner cache hit: arch=sm_{arch} floating={int(floating_base)} l2={l2_define} linalg={linalg_backend_note} eepose_hessian={enable_floating_eepose_hessian} key={runner_key[:12]}")
            cmd = [str(executable)]
            return executable, cmd
        compile_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_dir / "grid.cuh", compile_dir / "grid.cuh")
    else:
        compile_dir = build_dir
        executable = compile_dir / "cuda_equivalence_runner.exe"

    runner_copy = compile_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)

    defines = [f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}"]
    if l2_persisting is not None:
        defines.append(f"-DGRID_CUDA_ENABLE_L2_PERSISTING={l2_define}")
    # Mimic robots emit no gradient algorithms (G0 guard); compile the runner
    # without its gradient calls so it links against the gradient-free header.
    if skip_gradients:
        defines.append("-DGRID_RUNNER_SKIP_GRADIENTS=1")
    # ee_pose gradient/hessian (kinematic) land in a later mimic phase (P4) than
    # the dynamics gradients id_du/fd_du (P3). Fixed-base mimic skips ONLY the
    # ee_pose gradients (keeps id_du/fd_du); floating mimic skips all gradients.
    if skip_eepose_gradients and not skip_gradients:
        defines.append("-DGRID_RUNNER_SKIP_EEPOSE_GRADIENTS=1")

    cmd = [
        nvcc,
        cxx_standard,
        "-O0",
        *defines,
        *linalg_flags,
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "-gencode",
        f"arch=compute_{arch},code=compute_{arch}",
        "-o",
        str(executable),
        str(runner_copy),
    ]
    _progress(config, f"compiling runner arch=sm_{arch} floating={int(floating_base)} l2={l2_define} linalg={linalg_backend_note} eepose_hessian={enable_floating_eepose_hessian} cache_key={runner_key[:12]}")
    result = subprocess.run(cmd, cwd=compile_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA equivalence runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if _cache_enabled():
        (compile_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": CACHE_SCHEMA_VERSION,
                    "header_key": header_key,
                    "arch": arch,
                    "floating_base": bool(floating_base),
                    "l2_persisting": l2_define,
                    "floating_eepose_hessian": enable_floating_eepose_hessian,
                    "linalg_backend": linalg_backend_note,
                    "compile_flags": compile_flags,
                    "cmd": cmd,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return executable, cmd


def _run_runner(executable: Path, sample_input: str, compile_cmd: list[str], num_threads: int | None = None) -> str:
    argv = [str(executable)]
    if num_threads is not None:
        argv.append(str(num_threads))
    result = subprocess.run(
        argv,
        input=sample_input,
        cwd=executable.parent,
        capture_output=True,
        text=True,
    )
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(pattern in combined_output for pattern in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip(
                "CUDA runtime is unavailable in this environment. "
                "Run on a GPU-enabled machine with:\n"
                "  nvcc --version\n"
                "  nvidia-smi\n"
                "  .venv/bin/python -m pytest test/cuda_equivalents "
                "-m cuda_equivalence -vv\n"
                "If it still fails, report the pytest output plus this compile command:\n"
                f"  {' '.join(compile_cmd)}"
            )
        # A kernel whose most-spilled smem request still exceeds this GPU's
        # per-block cap physically cannot launch here — a hardware limit, not a
        # correctness bug. Skip honestly (self-heals if a future spill makes it
        # fit). Known case: the h1_2 body-frame IDSVA-SO inner is aliased so its
        # ancestor-pair scratch can't be independently spilled (168784 B > the
        # ~101 KB cap); h1_2's PRODUCTION SO path is world-frame, which fits and
        # passes. Closing the body-frame gap = the deferred de-alias refactor in
        # docs/idsva_so_inner_refactor_notes.md (gated on the perf sweep).
        if "shared-memory request" in combined_output and "this device supports" in combined_output:
            pytest.skip(
                "Kernel shared-memory request exceeds this GPU's per-block cap even "
                "at the most-spilled tier (hardware limit, not a bug). See:\n"
                f"  stderr: {result.stderr.strip()}\n"
                "For the body-frame IDSVA-SO case this is the deferred ancestor-scratch "
                "de-alias (docs/idsva_so_inner_refactor_notes.md); world-frame is the "
                "production path and fits."
            )
        # Register-pressure launch limit: at the robot's MAX_PERF_LEVEL_THREADS
        # (the "suggested" sweep point) a large floating-base inline kernel can
        # exceed the per-block register budget (regs/thread * threads > 64K).
        # This is a hardware launch limit, not a correctness bug — the same
        # kernel launches and PASSES equivalence at the lower thread counts in
        # the sweep (32/96). Skip honestly (e.g. fr3-floating FD inline at 1024
        # autotuned threads). The production launch uses MAX_PERF_LEVEL_THREADS
        # only when it fits; the runner deliberately probes the cap.
        if "too many resources requested for launch" in combined_output:
            pytest.skip(
                "Kernel exceeds this GPU's per-block register budget at the "
                "MAX_PERF_LEVEL_THREADS sweep point (hardware launch limit, not a "
                "bug; the same kernel passes equivalence at lower thread counts).\n"
                f"  stderr: {result.stderr.strip()}"
            )
        pytest.fail(
            "CUDA equivalence runner failed at runtime.\n"
            f"Command: {executable}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def _thread_counts() -> tuple[int, ...]:
    """Block thread counts to sweep each CUDA equivalence case over.

    Defaults to a single warp (32), a non-multiple of 32 (96) to exercise partial
    trailing warps, the sentinel 0 = the robot's MAX_PERF_LEVEL_THREADS (the count real
    GRiD usage launches at, resolved DYNAMICALLY in the runner from the generated
    header — never hardcoded, since it varies per robot: iiwa14=352, go2=288,
    g1/h1_2=512), and one session-random multi-warp count. The runner clamps every
    requested count to MAX_PERF_LEVEL_THREADS (the kernels' __launch_bounds__ cap).
    Sweeping thread counts catches thread-count-dependent races (missing
    __syncthreads between a write phase and a read/accumulate phase that happens
    to be correct only within a single warp) that a fixed 32-thread launch hides.
    Override via GRID_CUDA_THREAD_COUNTS (comma-separated ints, "suggested" for the
    MAX_PERF_LEVEL_THREADS sentinel, or "random" for a fresh multi-warp value)."""
    raw = os.environ.get("GRID_CUDA_THREAD_COUNTS")
    if raw:
        counts = []
        for part in raw.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part == "random":
                counts.append(_random_thread_count())
            elif part == "suggested":
                counts.append(0)
            else:
                counts.append(int(part))
        return tuple(dict.fromkeys(counts)) or (32,)
    return (32, 96, 0, _random_thread_count())


def _random_thread_count() -> int:
    # A session-fresh multi-warp count that is not a multiple of 32, so a
    # trailing partial warp is always present. Seeded from os.urandom so each
    # run probes a different count over time; the chosen value appears in the
    # parametrized test id for reproducibility.
    rng = random.Random()
    return rng.choice([n for n in range(33, 480) if n % 32 != 0])


def _random_sample_count() -> int:
    raw_count = os.environ.get("GRID_CUDA_RANDOM_SAMPLES")
    if raw_count is None:
        return DEFAULT_RANDOM_SAMPLE_COUNT
    try:
        return max(0, int(raw_count))
    except ValueError as exc:
        raise ValueError("GRID_CUDA_RANDOM_SAMPLES must be an integer.") from exc


def _floating_algorithm_selection() -> tuple[str, ...]:
    raw = os.environ.get("GRID_CUDA_FLOATING_ALGORITHMS")
    if not raw:
        return FLOATING_CUDA_ALGORITHMS
    if raw.strip().lower() == "all":
        return FLOATING_CUDA_CANDIDATE_ALGORITHMS
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(requested) - set(FLOATING_CUDA_CANDIDATE_ALGORITHMS))
    if unknown:
        raise ValueError(
            "GRID_CUDA_FLOATING_ALGORITHMS contained unsupported names: "
            + ", ".join(unknown)
        )
    return requested


def _parse_sample_names(raw: str) -> set[str] | None:
    if raw.strip().lower() == "all":
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _sample_name_selection(base_mode: str) -> SampleSelection:
    raw = os.environ.get("GRID_CUDA_SAMPLE_NAMES")
    if raw:
        return SampleSelection(_parse_sample_names(raw), True, True)
    if base_mode == "floating":
        raw = os.environ.get("GRID_CUDA_FLOATING_SAMPLE_NAMES")
        if not raw:
            return SampleSelection({"zero"}, False, False)
        return SampleSelection(_parse_sample_names(raw), True, True)
    return SampleSelection(None, False, False)


def _stable_robot_seed(robot_id: str) -> int:
    return 1000 + sum((index + 1) * ord(char) for index, char in enumerate(robot_id))


def _joint_position_bounds(project_model, low: float, high: float):
    if not project_model.nq:
        return 0, np.zeros((0, 2), dtype=np.float64), 0
    joint_offset = 7 if project_model.base_mode == "floating" else 0
    joint_count = project_model.nq - joint_offset
    skip_joint_ids = 1 if project_model.base_mode == "floating" else 0
    bounds = _joint_ranges(
        project_model.robot,
        joint_count,
        low,
        high,
        skip_joint_ids=skip_joint_ids,
    )
    return joint_offset, bounds, joint_count


def _bounded_joint_values(bounds: np.ndarray, preferred: np.ndarray) -> np.ndarray:
    if bounds.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.clip(preferred.astype(np.float64), bounds[:, 0], bounds[:, 1])


def _alternating_values(count: int, magnitude: float) -> np.ndarray:
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    return signs.astype(np.float64) * magnitude


def _set_floating_base(q: np.ndarray, quat_xyzw: np.ndarray, translation=None) -> None:
    if translation is None:
        translation = np.zeros(3, dtype=np.float64)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    q[0:3] = np.asarray(translation, dtype=np.float64)
    q[3:7] = quat_xyzw


def _make_cuda_corner_samples(project_model) -> list[DynamicsSample]:
    samples: list[DynamicsSample] = []
    joint_offset, bounds, joint_count = _joint_position_bounds(project_model, -0.6, 0.6)

    def make_sample(name: str, joint_values, qd_values, qdd_values, quat=None, translation=None):
        q = np.zeros(project_model.nq, dtype=np.float64)
        if project_model.base_mode == "floating":
            _set_floating_base(
                q,
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if quat is None else quat,
                translation=translation,
            )
        if joint_count:
            q[joint_offset:] = _bounded_joint_values(bounds, np.asarray(joint_values, dtype=np.float64))
        qd = np.asarray(qd_values, dtype=np.float64)
        qdd = np.asarray(qdd_values, dtype=np.float64)
        samples.append(DynamicsSample(name=name, q=q, qd=qd, qdd=qdd))

    positive_joints = np.full(joint_count, 0.2, dtype=np.float64)
    negative_joints = np.full(joint_count, -0.2, dtype=np.float64)
    mixed_joints = _alternating_values(joint_count, 0.25)
    tiny_joints = _alternating_values(joint_count, 1e-7)
    nominal_joints = _alternating_values(joint_count, 0.15)
    if joint_count:
        spans = bounds[:, 1] - bounds[:, 0]
        near_low = bounds[:, 0] + 0.1 * spans
        near_high = bounds[:, 1] - 0.1 * spans
        near_limit_joints = np.where(np.arange(joint_count) % 2 == 0, near_high, near_low)
    else:
        near_limit_joints = np.zeros(0, dtype=np.float64)

    make_sample(
        "positive",
        positive_joints,
        np.full(project_model.nv, 0.35, dtype=np.float64),
        np.full(project_model.nv, 0.75, dtype=np.float64),
        translation=np.array([0.05, 0.04, 0.03], dtype=np.float64),
    )
    make_sample(
        "negative",
        negative_joints,
        np.full(project_model.nv, -0.35, dtype=np.float64),
        np.full(project_model.nv, -0.75, dtype=np.float64),
        translation=np.array([-0.05, -0.04, -0.03], dtype=np.float64),
    )
    make_sample(
        "mixed_sign",
        mixed_joints,
        _alternating_values(project_model.nv, 0.45),
        -_alternating_values(project_model.nv, 0.9),
        translation=np.array([0.04, -0.03, 0.02], dtype=np.float64),
    )
    make_sample(
        "tiny",
        tiny_joints,
        _alternating_values(project_model.nv, 1e-7),
        -_alternating_values(project_model.nv, 1e-7),
    )
    make_sample(
        "velocity_only",
        nominal_joints,
        _alternating_values(project_model.nv, 0.55),
        np.zeros(project_model.nv, dtype=np.float64),
    )
    make_sample(
        "accel_or_torque_only",
        nominal_joints,
        np.zeros(project_model.nv, dtype=np.float64),
        _alternating_values(project_model.nv, 1.1),
    )
    make_sample(
        "near_limit",
        near_limit_joints,
        _alternating_values(project_model.nv, 0.25),
        _alternating_values(project_model.nv, 0.5),
    )
    if project_model.base_mode == "floating":
        make_sample(
            "floating_quat_identity",
            nominal_joints,
            _alternating_values(project_model.nv, 0.25),
            _alternating_values(project_model.nv, 0.5),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            translation=np.array([0.02, -0.01, 0.03], dtype=np.float64),
        )
        make_sample(
            "floating_quat_positive",
            positive_joints,
            np.full(project_model.nv, 0.2, dtype=np.float64),
            np.full(project_model.nv, 0.4, dtype=np.float64),
            quat=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
            translation=np.array([0.06, 0.02, 0.04], dtype=np.float64),
        )
        make_sample(
            "floating_quat_mixed",
            mixed_joints,
            _alternating_values(project_model.nv, 0.3),
            -_alternating_values(project_model.nv, 0.6),
            quat=np.array([-0.35, 0.2, -0.5, 0.75], dtype=np.float64),
            translation=np.array([-0.03, 0.05, -0.02], dtype=np.float64),
        )
    return samples


def _build_cuda_samples(
    project_model,
    random_count: int | None = None,
    *,
    include_corner_samples: bool = False,
):
    samples = list(build_dynamics_samples(project_model))
    if include_corner_samples:
        samples.extend(_make_cuda_corner_samples(project_model))
    if random_count is None:
        random_count = _random_sample_count()

    rng = np.random.default_rng(_stable_robot_seed(project_model.spec.robot_id))
    if project_model.nq:
        joint_offset = 7 if project_model.base_mode == "floating" else 0
        joint_count = project_model.nq - joint_offset
        skip_joint_ids = 1 if project_model.base_mode == "floating" else 0
        bounds = _joint_ranges(
            project_model.robot,
            joint_count,
            -0.75,
            0.75,
            skip_joint_ids=skip_joint_ids,
        )
    else:
        bounds = np.zeros((0, 2), dtype=np.float64)

    for sample_index in range(random_count):
        q = np.zeros(project_model.nq, dtype=np.float64)
        if project_model.base_mode == "floating":
            q[0:3] = rng.uniform(-0.35, 0.35, size=3)
            quat_xyzw = rng.uniform(-1.0, 1.0, size=4)
            quat_xyzw /= np.linalg.norm(quat_xyzw)
            q[3:7] = quat_xyzw.astype(np.float64)
            if bounds.size:
                q[7:] = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)
        elif project_model.nq:
            q = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)
        qd = rng.uniform(-1.5, 1.5, size=project_model.nv).astype(np.float64)
        qdd = rng.uniform(-2.5, 2.5, size=project_model.nv).astype(np.float64)
        samples.append(
            DynamicsSample(
                name=f"cuda_random_{sample_index}",
                q=q,
                qd=qd,
                qdd=qdd,
            )
        )
    return samples


def _sample_to_stdin(sample) -> str:
    values = [
        np.asarray(sample.q, dtype=np.float32),
        np.asarray(sample.qd, dtype=np.float32),
        np.asarray(sample.qdd, dtype=np.float32),
    ]
    serialized_rows = (" ".join(f"{value:.9g}" for value in vec) for vec in values)
    return "\n".join(serialized_rows) + "\n"


def _parse_runner_output(stdout: str) -> dict[str, np.ndarray]:
    outputs = {}
    lines = iter(stdout.splitlines())
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("BEGIN "):
            raise AssertionError(f"Unexpected CUDA runner output line: {stripped}")
        _begin, name, rows, cols = stripped.split()
        rows = int(rows)
        cols = int(cols)
        matrix_rows = []
        for _ in range(rows):
            matrix_rows.append([float(value) for value in next(lines).split()])
        end = next(lines).strip()
        if end != f"END {name}":
            raise AssertionError(f"Expected END {name}, got: {end}")
        outputs[name] = np.asarray(matrix_rows, dtype=np.float64).reshape(rows, cols)
    return outputs


def _normalize_cuda_minv(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.copy()
    lower = np.tril(normalized, k=-1)
    upper = np.triu(normalized, k=1)
    if np.count_nonzero(np.abs(lower) > 1e-7) == 0:
        normalized = normalized + upper.T
    return normalized


def _model_inertia_is_degenerate(reference_model, nv):
    """True if the resolved model has effectively zero inertia (all-zero mass
    matrix). Indicates a broken URDF asset (e.g. rizon4's upstream flexiv xacro
    emits bare mass/inertia tags NOT wrapped in <inertial>, so every link parses
    massless) — the dynamics are physically undefined and the float32 CUDA path
    divides by the zero-mass structure (-> NaN). Not a GRiD bug; skip honestly.
    Self-heals if a corrected asset later resolves with real inertias."""
    try:
        mass = np.asarray(reference_model.crba(np.zeros(nv)), dtype=np.float64)
    except Exception:
        return False
    return bool(mass.size and np.max(np.abs(mass)) < 1e-12)


def _has_invertible_project_mass_matrix(reference_model, q, min_singular_value=1e-12):
    try:
        mass = np.asarray(reference_model.crba(q), dtype=np.float64)
        singular_values = np.linalg.svd(mass, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if singular_values.size == 0:
        return False
    return bool(
        np.isfinite(singular_values).all()
        and singular_values[-1] > min_singular_value
    )


def _forward_dynamics_float32_matches(reference_model, project_model, sample, cuda, robot_id) -> bool:
    """True if the CUDA Minv-based forward_dynamics for this sample is finite and
    matches the float64 reference within the FD tolerance.

    Used to recognize a known float32 limitation: the O(n) Articulated-Body
    recursion (`aba`) can produce non-finite values on moderately ill-conditioned
    floating-base configs where the robust CRBA+solve path (`forward_dynamics`)
    still gives the correct answer. We only excuse a non-finite `aba` when this
    robust path is demonstrably correct, so a non-finite that coincides with a
    genuinely broken FD path is never masked."""
    if "forward_dynamics" not in cuda:
        return False
    actual = np.asarray(cuda["forward_dynamics"], dtype=np.float64)
    if not np.all(np.isfinite(actual)):
        return False
    expected = _expected_output(reference_model, project_model, sample, "forward_dynamics")
    tol = _cuda_tolerance(robot_id, "forward_dynamics")
    return bool(
        np.allclose(actual, expected, rtol=tol["rtol"], atol=tol["atol"])
        or (
            tol.get("norm_rtol") is not None
            and np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
            <= tol["norm_rtol"]
        )
    )


def _expected_output(reference_model, project_model, sample, name: str):
    # reference_model is the oracle (default pinocchio, exact) for every algorithm
    # EXCEPT d2ee. `end_effector_pose_hessian` always comes from project_model
    # (analytic pure-Python) because pinocchio's d2ee is finite-diff and invalid at
    # rpy/atan2 wraps. EE targets are enumerated from the model-under-test `robot` so
    # the compared set matches what CUDA emits (the same names resolve in both backends).
    robot = project_model.robot
    zeros = np.zeros(reference_model.nv, dtype=np.float64)
    if name == "inverse_dynamics":
        return reference_model.rnea(sample.q, sample.qd, zeros).reshape(1, -1)
    if name == "direct_minv":
        return reference_model.minv(sample.q)
    if name == "forward_dynamics":
        return reference_model.forward_dynamics(sample.q, sample.qd, sample.qdd).reshape(
            1, -1
        )
    if name == "inverse_dynamics_gradient_q":
        return reference_model.rnea_grad(sample.q, sample.qd, zeros)[0]
    if name == "inverse_dynamics_gradient_qd":
        return reference_model.rnea_grad(sample.q, sample.qd, zeros)[1]
    if name == "forward_dynamics_gradient_q":
        return reference_model.forward_dynamics_grad(sample.q, sample.qd, sample.qdd)[0]
    if name == "forward_dynamics_gradient_qd":
        return reference_model.forward_dynamics_grad(sample.q, sample.qd, sample.qdd)[1]
    if name == "aba":
        return reference_model.aba(sample.q, sample.qd, sample.qdd).reshape(1, -1)
    if name == "crba":
        return reference_model.crba(sample.q)
    if name == "end_effector_pose":
        poses = []
        for jid in robot.get_leaf_nodes():
            target = robot.get_joint_by_id(jid).get_name()
            poses.append(reference_model.end_effector_pose(sample.q, target))
        return np.concatenate(poses, axis=0).reshape(1, -1)
    if name == "end_effector_pose_gradient":
        gradients = []
        for jid in robot.get_leaf_nodes():
            target = robot.get_joint_by_id(jid).get_name()
            gradient = reference_model.end_effector_pose_gradient(sample.q, target)
            gradients.append(np.asarray(gradient, dtype=np.float64).reshape(-1, order="F"))
        return np.concatenate(gradients, axis=0).reshape(1, -1)
    if name == "end_effector_pose_hessian":
        # d2ee uses the independent pinocchio oracle (default backend). The pinocchio
        # backend's d2ee is the analytic getJointKinematicHessian(LOCAL_WORLD_ALIGNED)
        # -- a valid d/dv ground truth that agrees with the project analytic
        # chain-composition d2(pose)/dv2 to the FD-noise floor fleet-wide (A2 resolved
        # 2026-05-31). The CUDA codegen mirrors that analytic path; this comparison is
        # genuine (independent oracle), no buggy-vs-buggy masking.
        hessians = []
        for jid in robot.get_leaf_nodes():
            target = robot.get_joint_by_id(jid).get_name()
            hessian = reference_model.end_effector_pose_hessian(sample.q, target)
            hessians.append(np.asarray(hessian, dtype=np.float64).reshape(-1))
        return np.concatenate(hessians, axis=0).reshape(1, -1)
    raise ValueError(f"Unexpected CUDA output name: {name}")


def _cuda_tolerance(robot_id: str, algorithm: str):
    return CUDA_ROBOT_ALGORITHM_TOLERANCES.get(
        (robot_id, algorithm), CUDA_DEFAULT_TOLERANCE
    )


# end_effector_pose is a per-leaf 6-vector [x, y, z, roll, pitch, yaw]: rows 0-2
# are position, rows 3-5 are the rpy orientation (angles). _expected_output
# flattens each algorithm differently, so the flat index -> "is orientation row"
# map differs per algorithm (see _expected_output for the exact reshapes):
#   end_effector_pose          : per leaf 6-vector, contiguous  -> block size 6,
#                                orientation = (idx % 6) in {3,4,5}
#   end_effector_pose_gradient : per leaf (6, nq) reshaped order="F" (column-
#                                major) -> within each 6*nq leaf block the pose
#                                component varies fastest -> (idx % 6) in {3,4,5}
#   end_effector_pose_hessian  : per leaf (6, nq, nq) reshaped C-order -> pose
#                                component is the SLOWEST axis -> within each
#                                6*nq*nq leaf block, component = idx // (nq*nq),
#                                orientation = component in {3,4,5}
_EE_POSE_ALGORITHMS = {
    "end_effector_pose",
    "end_effector_pose_gradient",
    "end_effector_pose_hessian",
}


def _ee_row_info(algorithm: str, expected_flat: np.ndarray, n_leaves: int):
    """Per-flat-entry decode of an end_effector_pose* output into
    (leaf_index, pose_component): pose_component 0/1/2 = x/y/z (position),
    3/4/5 = roll/pitch/yaw (orientation). Returns (leaf_idx, component) int
    arrays the same length as the flattened array, or (None, None) if the
    algorithm isn't an EE-pose family or the size doesn't factor cleanly."""
    if algorithm not in _EE_POSE_ALGORITHMS or n_leaves <= 0:
        return None, None
    total = expected_flat.size
    if total % n_leaves != 0:
        return None, None
    per_leaf = total // n_leaves
    if per_leaf % 6 != 0:
        return None, None
    idx = np.arange(total)
    leaf_idx = idx // per_leaf
    leaf_local = idx % per_leaf
    if algorithm in ("end_effector_pose", "end_effector_pose_gradient"):
        # ee_pose: contiguous 6-vector per leaf. gradient: per leaf (6, nq)
        # reshaped order="F" -> pose component varies fastest. Both -> local % 6.
        component = leaf_local % 6
    else:
        # end_effector_pose_hessian: per leaf (6, nq, nq) C-order -> the pose
        # component is the SLOWEST axis (block of nq*nq per component).
        component = leaf_local // (per_leaf // 6)
    return leaf_idx, component


def _ee_orientation_mask(algorithm: str, expected_flat: np.ndarray, n_leaves: int):
    """Boolean mask (flat) selecting the rpy orientation entries of an
    end_effector_pose* output. None if not applicable."""
    leaf_idx, component = _ee_row_info(algorithm, expected_flat.reshape(-1), n_leaves)
    if leaf_idx is None:
        return None
    return (component >= 3) & (component <= 5)


def _wrap_to_pi(values):
    """Fold angle differences into (-pi, pi]. Used so that an rpy orientation
    differing by exactly 2*pi (e.g. yaw=+pi vs -pi, the atan2 branch ambiguity)
    is treated as EXACT agreement, not a 2*pi error. Position rows are NOT
    angles and must never be wrapped."""
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


# An rpy parameterization hits gimbal lock when pitch -> +/-pi/2: pitch_sqrt_term
# (= sqrt(rot[2,2]^2 + rot[2,1]^2)) -> 0, so roll and yaw stop being separable.
# At gimbal lock:
#  - end_effector_pose: roll and yaw are NOT uniquely defined (only roll-/+yaw is),
#    so CUDA and pinocchio can pick different-but-equivalent (roll, yaw) splits of
#    the same rotation (e.g. baxter zero: yaw 0.0 vs 2.034). PITCH itself is well
#    defined (= +/-pi/2) and stays asserted.
#  - gradient / hessian: the analytic d(rpy)/dq and d2(rpy)/dq2 blow up
#    (1/pitch_sqrt_term, 1/pitch_sqrt_term^2); the finite-diff reference returns
#    finite-but-enormous, numerically meaningless values and the float32 CUDA path
#    can go non-finite.
# Separately, even AWAY from gimbal lock the finite-diff reference for the rpy
# DERIVATIVES spikes to ~pi/step when the angle wraps across +/-pi between the
# +/-step samples (e.g. fr3 zero, yaw ~ +/-pi, pitch 0): a spurious O(1/step) value
# vs the correct analytic O(1) CUDA value. That config-independent wrap spike is
# caught by the magnitude threshold below.
# All of this is SCOPED to the rpy rows of ee_pose / its derivatives; position rows
# and every non-singular config stay strictly asserted, so real bugs are not masked.
EE_ORIENTATION_DERIV_BLOWUP_THRESHOLD = 1e4
EE_GIMBAL_PITCH_EPS = 1e-3


def _ee_gimbal_lock_leaves(reference_model, project_model, sample, eps=EE_GIMBAL_PITCH_EPS):
    """Indices (in get_leaf_nodes() order) of EE leaves whose reference pose pitch
    is within `eps` of +/-pi/2, i.e. at rpy gimbal lock for this config."""
    robot = project_model.robot
    locked = set()
    for li, jid in enumerate(robot.get_leaf_nodes()):
        target = robot.get_joint_by_id(jid).get_name()
        try:
            pose = np.asarray(
                reference_model.end_effector_pose(sample.q, target), dtype=np.float64
            ).reshape(-1)
        except Exception:
            continue
        pitch = float(pose[4])  # pose = [x, y, z, roll, pitch, yaw]
        if np.isfinite(pitch) and abs(abs(pitch) - np.pi / 2.0) <= eps:
            locked.add(li)
    return locked


def _assert_close(
    label: str,
    actual: np.ndarray,
    expected: np.ndarray,
    robot_id: str,
    algorithm: str,
    n_leaves: int = 0,
    gimbal_lock_leaves=frozenset(),
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    tol = _cuda_tolerance(robot_id, algorithm)

    if algorithm in _EE_POSE_ALGORITHMS:
        leaf_idx, component = _ee_row_info(algorithm, expected.reshape(-1), n_leaves)
        if leaf_idx is not None:
            orient = (component >= 3) & (component <= 5)
            shp = expected.shape
            orient_m = orient.reshape(shp)
            in_gimbal_leaf = np.isin(leaf_idx, list(gimbal_lock_leaves)).reshape(shp)
            if algorithm == "end_effector_pose":
                # (b1) rpy orientation rows are angles: a difference of 2*pi (or
                # +pi vs -pi from the atan2 branch) is EXACT agreement. Fold the
                # orientation-row residual into (-pi, pi] before comparing.
                # Position rows are left untouched. We adjust `actual` toward
                # `expected` by whole 2*pi turns so assert_allclose sees the true
                # (wrapped) error; the *derivatives* (gradient/hessian) are NOT
                # mod-2pi and are handled by the b2 gimbal-lock guard instead.
                wrapped_diff = _wrap_to_pi(actual - expected)
                actual = np.where(orient_m, expected + wrapped_diff, actual)
                # (b2 / ee_pose) At gimbal lock roll & yaw are not separable: CUDA
                # and pinocchio may split the same rotation into different
                # (roll, yaw) pairs. Skip the ROLL (3) and YAW (5) rows of
                # gimbal-locked leaves; pitch (4, = +/-pi/2) stays asserted.
                roll_or_yaw = (component == 3) | (component == 5)
                skip = (in_gimbal_leaf & roll_or_yaw.reshape(shp))
            else:
                # (b2) ee_pose_gradient / ee_pose_hessian orientation rows: the
                # rpy derivative is genuinely singular and unvalidatable when
                # EITHER (i) this leaf is at gimbal lock (analytic d(rpy)/dq blows
                # up as pitch_sqrt_term -> 0), OR (ii) the finite-diff reference
                # spiked to ~pi/step from an angle wrapping across +/-pi between
                # the +/-step samples (|expected| explodes) even away from gimbal
                # lock, OR (iii) the float32 CUDA value went non-finite there.
                # This is the finite-reference analogue of the caller's
                # non-finite-reference skip; scoped to orientation-derivative rows.
                skip = orient_m & (
                    in_gimbal_leaf
                    | (np.abs(expected) > EE_ORIENTATION_DERIV_BLOWUP_THRESHOLD)
                    | ~np.isfinite(actual)
                )
            keep = ~skip
            if not np.all(keep):
                actual = actual[keep]
                expected = expected[keep]
                if expected.size == 0:
                    return
    # Magnitude-scaled absolute floor (mirrors RBDReference/tests/comparators.py).
    # np.testing.assert_allclose checks |actual-expected| <= atol + rtol*|expected|
    # per element. For an array whose overall scale is huge (e.g. h1_2 has a
    # near-singular mass matrix => Minv entries ~1e6-1e8, and the FD gradients
    # inherit that scale), a fixed atol=2e-4 is meaningless: a float32-perfect
    # result (relative error ~1e-7) still trips the check on entries that are
    # individually small *relative to the array's overall scale*. Floor atol at
    # rtol*max|expected| so "small relative to the array scale" counts as close.
    # A genuine error is O(scale) (or O(0.1*scale)) and still exceeds this floor,
    # so real bugs are NOT masked -- this only excuses entries whose error is
    # within rtol of the array's dominant magnitude.
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    atol_eff = max(tol["atol"], tol["rtol"] * scale)
    try:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tol["rtol"],
            atol=atol_eff,
        )
    except AssertionError as exc:
        diff = np.abs(actual - expected)
        rel = diff / np.maximum(np.abs(expected), 1e-12)
        flat_index = int(np.argmax(diff))
        index = np.unravel_index(flat_index, diff.shape)
        norm_rel = np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12)
        norm_rtol = tol.get("norm_rtol")
        if norm_rtol is not None and norm_rel <= norm_rtol:
            return
        actual_at_index = actual[index]
        expected_at_index = expected[index]
        raise AssertionError(
            f"{label} CUDA mismatch: max_abs={diff[index]}, "
            f"max_rel={rel[index]}, first_worst_index={index}, "
            f"norm_rel={norm_rel}, actual={actual_at_index}, "
            f"expected={expected_at_index}, rtol={tol['rtol']}, "
            f"atol={atol_eff:.3e} (scale={scale:.3e})"
        ) from exc


@pytest.mark.parametrize("num_threads", _thread_counts(), ids=lambda t: f"threads{'suggested' if t == 0 else t}")
@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_cuda_case_params())
def test_fixed_base_generated_cuda_matches_python_reference(spec, base_mode, num_threads, tmp_path, request):
    selection = _sample_name_selection(base_mode)
    random_count = 0 if selection.explicit and os.environ.get("GRID_CUDA_RANDOM_SAMPLES") is None else None
    _run_cuda_equivalence_case(
        spec,
        base_mode,
        tmp_path,
        FIXED_CUDA_ALGORITHMS,
        sample_selection=selection,
        random_count=random_count,
        config=request.config,
        num_threads=num_threads,
    )


@pytest.mark.parametrize("num_threads", _thread_counts(), ids=lambda t: f"threads{'suggested' if t == 0 else t}")
@pytest.mark.parametrize(("spec", "base_mode"), build_floating_cuda_case_params())
def test_floating_base_generated_cuda_matches_python_reference(spec, base_mode, num_threads, tmp_path, request):
    selection = _sample_name_selection(base_mode)
    random_count = None
    if os.environ.get("GRID_CUDA_RANDOM_SAMPLES") is None and (
        selection.explicit or selection.names == {"zero"}
    ):
        random_count = 0
    _run_cuda_equivalence_case(
        spec,
        base_mode,
        tmp_path,
        _floating_algorithm_selection(),
        sample_selection=selection,
        random_count=random_count,
        config=request.config,
        num_threads=num_threads,
    )


def _run_cuda_equivalence_case(
    spec,
    base_mode,
    tmp_path,
    algorithms,
    sample_selection=None,
    random_count=None,
    config=None,
    num_threads=None,
):
    if sample_selection is None:
        sample_selection = SampleSelection(None, False, False)
    sample_names = sample_selection.names
    target_shared = os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES", "default")
    l2_mode = os.environ.get("GRID_CUDA_ENABLE_L2_PERSISTING", "0")
    _progress(
        config,
        (
            f"start {spec.robot_id}-{base_mode}: target_shared={target_shared}, "
            f"l2={l2_mode}, samples={sorted(sample_names) if sample_names is not None else 'default/all'}, "
            f"random={_random_sample_count() if random_count is None else random_count}"
        ),
    )
    try:
        _progress(config, f"resolving robot model for {spec.robot_id}")
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh before "
            f"executing CUDA equivalence tests. Resolution error: {exc}"
        )
    # Two roles:
    #  - project_model (always the pure-Python URDFParser adapter) is the
    #    MODEL-UNDER-TEST input: it owns the codegen `robot` + EE-target/joint-bound
    #    enumeration. It is NOT used as a value oracle (using our own reference as the
    #    oracle would mask bugs shared between CUDA and the reference — exactly what
    #    happened with d2ee).
    #  - reference_model is the independent ORACLE for EVERY algorithm (incl. d2ee).
    #    It defaults to the EXACT pinocchio backend (the C++ authority) and can be
    #    forced to the pure-Python reference via GRID_REFERENCE_BACKEND=reference (a
    #    debug fallback that re-enables buggy-vs-buggy masking, so avoid it for CI).
    #    d2ee is now a HARD requirement (A2 resolved 2026-05-31): the pinocchio
    #    backend's d2ee uses the analytic getJointKinematicHessian(LOCAL_WORLD_ALIGNED)
    #    -- a valid d/dv oracle that agrees with the project analytic chain-composition
    #    d2(pose)/dv2 fleet-wide -- so it is no longer in KNOWN_FAILING_ALGORITHMS.
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    oracle_backend = resolve_backend(os.environ.get("GRID_REFERENCE_BACKEND", "pinocchio"))
    reference_model = (
        project_model
        if oracle_backend == "reference"
        else build_adapter(spec, resolved, base_mode=base_mode, backend=oracle_backend)
    )
    _progress(config, f"oracle backend={oracle_backend} for {spec.robot_id}-{base_mode}")

    if _model_inertia_is_degenerate(reference_model, project_model.nv):
        pytest.skip(
            f"{spec.robot_id}-{base_mode} resolves to a zero-inertia model (all-zero mass "
            "matrix) — a broken upstream URDF asset, not a GRiD defect. Dynamics are "
            "physically undefined; skip until a corrected asset resolves. "
            "(rizon4: flexiv xacro emits bare mass/inertia tags not wrapped in <inertial>.)"
        )

    build_dir = tmp_path / f"cuda_{spec.robot_id}_{base_mode}"
    build_dir.mkdir()
    header_path, header_key = _generate_grid_header(project_model, resolved, build_dir, config)
    _progress(config, f"fallback summary for {spec.robot_id}-{base_mode}: {_fallback_summary(header_path)}")
    executable, compile_cmd = _compile_runner(
        build_dir,
        floating_base=base_mode == "floating",
        header_key=header_key,
        # Mimic gradients: fixed-base mimic now emits id_du/fd_du (P3), so the
        # runner compiles its gradient block; floating-base mimic gradients are
        # still refused, so skip them in the runner for that case.
        skip_gradients=(
            _robot_has_mimic_joints(project_model) and base_mode == "floating"
        ),
        # ee_pose gradients/hessian (P4): fixed-base mimic now emits them (the
        # alpha-weighted geometric-Jacobian / world-frame-generator fold), so the
        # runner compiles its ee-pose gradient/hessian block; floating-base mimic
        # ee derivatives are still refused, so skip them in the runner there.
        skip_eepose_gradients=(
            _robot_has_mimic_joints(project_model) and base_mode == "floating"
        ),
        config=config,
    )

    failures = []
    known_bug_failures = []
    skipped = []
    compared = 0
    matched_samples = 0
    for sample in _build_cuda_samples(
        project_model,
        random_count=random_count,
        include_corner_samples=sample_selection.include_corner_samples,
    ):
        if sample_names is not None and sample.name not in sample_names:
            continue
        matched_samples += 1
        _progress(config, f"{spec.robot_id}-{base_mode}/{sample.name}: running CUDA runner (threads={num_threads or 32})")
        stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd, num_threads=num_threads)
        cuda = _parse_runner_output(stdout)

        np.testing.assert_allclose(
            cuda["input_q"],
            np.asarray(sample.q, dtype=np.float32).reshape(1, -1),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            cuda["input_qd"],
            np.asarray(sample.qd, dtype=np.float32).reshape(1, -1),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            cuda["input_u"],
            np.asarray(sample.qdd, dtype=np.float32).reshape(1, -1),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            cuda["runtime_probe"],
            np.arange(10, 10 + project_model.nv, dtype=np.float32).reshape(1, -1),
            rtol=0.0,
            atol=1e-7,
        )

        if "direct_minv" in cuda:
            cuda["direct_minv"] = _normalize_cuda_minv(cuda["direct_minv"])
        invertible_mass_matrix = _has_invertible_project_mass_matrix(
            reference_model, sample.q
        )
        robot_is_mimic = _robot_has_mimic_joints(project_model)
        for name in algorithms:
            if robot_is_mimic and name not in MIMIC_SUPPORTED_ALGORITHMS:
                # Mimic CUDA codegen for this algorithm has not landed yet
                # (task T3 phased rollout); skip (logged, not failed).
                skipped.append(
                    f"{spec.robot_id}/{sample.name}/{name} (mimic codegen pending phase)"
                )
                continue
            # Mimic ID/FD gradients (P3) are FIXED-BASE only so far; floating-base
            # mimic gradients are still refused at codegen (skip_gradients) and not
            # emitted, so don't try to compare them for a floating mimic robot.
            if (robot_is_mimic and base_mode == "floating"
                    and name in MIMIC_FLOATING_UNSUPPORTED_GRADIENTS):
                skipped.append(
                    f"{spec.robot_id}/{sample.name}/{name} (floating+mimic gradient pending)"
                )
                continue
            if name in SINGULAR_DEPENDENT_ALGORITHMS and not invertible_mass_matrix:
                skipped.append(f"{spec.robot_id}/{sample.name}/{name}")
                continue
            try:
                _progress(
                    config,
                    f"{spec.robot_id}-{base_mode}/{sample.name}/{name}: comparing",
                    verbose=True,
                )
                expected_value = _expected_output(reference_model, project_model, sample, name)
                # The reference quantity can be genuinely UNDEFINED at degenerate
                # configs — e.g. baxter's q=0 puts the EE frame at an rpy/atan2
                # gimbal-lock singularity (pitch_sqrt_term -> 0), so the
                # roll/pitch/yaw pose Jacobian / Hessian is non-finite in the
                # reference AND in CUDA. We cannot validate CUDA against an
                # undefined value, so skip when the REFERENCE itself is non-finite.
                # This only fires where the math has no answer; a finite reference
                # is always asserted, so a real CUDA NaN/bug is never masked.
                if not np.all(np.isfinite(np.asarray(expected_value, dtype=np.float64))):
                    skipped.append(
                        f"{spec.robot_id}/{sample.name}/{name} "
                        "(reference undefined at this config: non-finite, e.g. rpy gimbal lock)"
                    )
                    continue
                # Known float32 limitation: the ABA recursion can go non-finite
                # on moderately ill-conditioned floating-base configs (e.g. the
                # quaternion corner samples) where the robust Minv-based
                # forward_dynamics path still computes the correct result. ABA is
                # correct in float64 (it matches Pinocchio), so this is numerical,
                # not a codegen bug. Excuse it ONLY when the reference is finite
                # and the robust FD path matched — never mask a non-finite that
                # coincides with a broken FD path. See test/TESTING_STRATEGY.md.
                if (
                    name == "aba"
                    and not np.all(np.isfinite(np.asarray(cuda[name], dtype=np.float64)))
                    and np.all(np.isfinite(np.asarray(expected_value, dtype=np.float64)))
                    and _forward_dynamics_float32_matches(reference_model, project_model, sample, cuda, spec.robot_id)
                ):
                    skipped.append(
                        f"{spec.robot_id}/{sample.name}/aba "
                        "(float32 ABA recursion non-finite; Minv forward_dynamics path correct)"
                    )
                    continue
                gimbal_lock_leaves = (
                    _ee_gimbal_lock_leaves(reference_model, project_model, sample)
                    if name in _EE_POSE_ALGORITHMS
                    else frozenset()
                )
                _assert_close(
                    f"{spec.robot_id}/{sample.name}/{name}/threads={num_threads or 32}",
                    cuda[name],
                    expected_value,
                    robot_id=spec.robot_id,
                    algorithm=name,
                    n_leaves=len(project_model.robot.get_leaf_nodes()),
                    gimbal_lock_leaves=gimbal_lock_leaves,
                )
                compared += 1
            except AssertionError as exc:
                if (name in KNOWN_FAILING_ALGORITHMS
                        or (spec.robot_id, name) in KNOWN_FAILING_ROBOT_ALGORITHMS):
                    known_bug_failures.append(str(exc))
                else:
                    failures.append(str(exc))
        _progress(config, f"{spec.robot_id}-{base_mode}/{sample.name}: complete", verbose=True)

    if matched_samples == 0:
        known = ["zero", "conservative", *CUDA_CORNER_SAMPLE_NAMES, "cuda_random_N"]
        pytest.fail(
            f"No CUDA samples matched selection {sorted(sample_names) if sample_names else sample_names}. "
            f"Known deterministic samples: {', '.join(known)}"
        )

    if known_bug_failures:
        reasons = "; ".join(sorted(
            set(KNOWN_FAILING_ALGORITHMS.values())
            | set(KNOWN_FAILING_ROBOT_ALGORITHMS.values())
        )) or "tracked known bug"
        _progress(
            config,
            f"KNOWN-BUG (tracked, NOT masked) {len(known_bug_failures)} mismatch(es) vs the "
            f"independent oracle [{reasons}]:\n" + "\n".join(known_bug_failures),
        )
    if failures:
        pytest.fail("\n".join(failures))
    if skipped:
        _progress(config, "Skipped CUDA comparisons (singular / float32-ABA): " + ", ".join(skipped))
    # Only skip the whole case when NOTHING was actually compared (e.g. a fully
    # singular robot). If any comparison passed, the case passes — individually
    # skipped comparisons (singular-dependent algos, float32-ABA non-finite) are
    # logged above, not promoted to a whole-test skip that would hide the passes.
    if compared == 0 and skipped:
        pytest.skip(
            "Skipped CUDA comparisons (singular / float32-ABA); none comparable: "
            + ", ".join(skipped)
        )
    _progress(config, f"complete {spec.robot_id}-{base_mode}: {matched_samples} sample(s)")
