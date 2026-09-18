"""CUDA-equivalence suite conftest: generate PIN-ONLY headers by default.

WHY (2026-07-24). Every test in this directory generates a `grid.cuh` and compiles it
with nvcc. On a floating-base, non-mimic robot the generator also instantiates the mjx
(`MUJOCO_OUTPUT=true`) twin of each kernel -- and those twins are enormous next to their
pin counterparts. Measured on g1-floating:

    idsva_so_world_frame      28.1x        forward/inverse dynamics, minv, crba ~1.0x
    fdsva_so                   4.5x
    inverse_dynamics_gradient  2.9x

~2.1M of the ~2.4M SASS lines in a humanoid build are mjx-only. That cost was being paid
by this suite on every floating cell.

It bought nothing: **no test in this directory exercises mjx.** (Audited 2026-07-24 --
the only `mujoco`/`MUJOCO` hits under `test/cuda_equivalents/` are four comments in .cu
runners; not one runner launches an mjx kernel and not one .py test asserts against the
MuJoCo output convention.) The suite was compiling the expensive half of the library and
then testing the other half. Same worst-of-both-worlds shape as the benchmark harness,
which compiled every mjx twin and never timed one.

So: default this suite to pin-only. Correctness coverage is unchanged, because there was
no mjx coverage to lose.

OPTING BACK IN. Wave B adds real mjx coverage. A test that genuinely exercises mjx must
pass `enable_mujoco_kernels=True` EXPLICITLY to its own `gen_all_code` call -- an explicit
argument always beats the env var, so such a test is self-contained and unaffected by this
file. Do not rely on the env default for mjx coverage; a test that silently generated a
pin-only header would pass vacuously.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _record_header_content_keys():
    """A4 (2026-09-11): record every generated grid.cuh's CONTENT hash.

    When ``GRID_HEADER_KEYS_OUT`` names a file (run_split_suite sets it per
    cuda shard under --receipts), append one JSON line per header this shard
    generates or reuses, so the shard's receipt gains a per-cell header
    content-key sidecar. Wave A' consumes these at refresh time: regenerate a
    cell's header CPU-side, compare content hashes, and re-run only the cells
    whose emitted bytes actually rotated (instead of staling the whole cuda
    domain on any codegen edit).

    Two capture layers, both patched from HERE (deliberately not from
    cuda_harness.py — that file is in every cuda shard's fingerprint, so an
    edit there would itself stale the whole domain; this conftest is not):
      - GRiDCodeGenerator.gen_all_code — every DIRECT per-test codegen call
        (the ~26 non-flagship modules), with the bound call kwargs as the
        best-effort recipe evidence;
      - cuda_harness._generate_grid_header — the flagship header path, which
        on a warm cache COPIES the header without calling gen_all_code (the
        layer above would miss cache hits).
    Recording is best-effort by design: a missing record makes Wave A'
    conservatively stale that cell, never silently carry it.
    """
    out = os.environ.get("GRID_HEADER_KEYS_OUT")
    if not out:
        yield
        return
    import hashlib
    import inspect
    import json
    from pathlib import Path

    from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

    def emit(record: dict) -> None:
        try:
            with open(out, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass

    def _hash(path) -> str | None:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return None

    orig_gen = GRiDCodeGenerator.gen_all_code
    sig = inspect.signature(orig_gen)

    # Env knobs that steer codegen output — snapshotted into every record so
    # the A' replayer regenerates under the SAME environment (spill tests
    # monkeypatch the smem knobs mid-session; the suite pins mjx kernels off).
    replay_env = ("GRID_ENABLE_MUJOCO_KERNELS", "GRID_CODEGEN_PROFILE",
                  "GRID_CUDA_TARGET_SHARED_MEM_BYTES",
                  "GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES")

    def _env_snapshot() -> dict:
        return {k: os.environ.get(k) for k in replay_env}

    def _urdf_sha(robot_name) -> str | None:
        try:
            from config import robot_urdf
            return _hash(robot_urdf(robot_name))
        except Exception:
            return None

    def gen_wrapper(self, *args, **kwargs):
        result = orig_gen(self, *args, **kwargs)
        try:
            bound = sig.bind(self, *args, **kwargs)
            call = {k: v for k, v in bound.arguments.items() if k != "self"}
            out_path = call.pop("output_path", None) or "grid.cuh"
            # JSON-representable kwargs stay STRUCTURED (the A' replayer can
            # feed them straight back to gen_all_code); everything else goes
            # to `opaque` as an address-stripped repr — evidence only, replay
            # needs a HEADER_RECIPES entry or conservatively stales the cell.
            import re
            clean, opaque = {}, {}
            for k, v in call.items():
                if isinstance(v, tuple):
                    v = list(v)
                try:
                    json.dumps(v)
                    clean[k] = v
                except (TypeError, ValueError):
                    opaque[k] = re.sub(r" at 0x[0-9a-f]+", "", repr(v))
            robot = getattr(self, "robot", None)
            name = getattr(robot, "name", None)
            emit({"kind": "direct",
                  "robot": name,
                  "floating": bool(getattr(robot, "floating_base", False)),
                  "kwargs": clean,
                  "opaque": opaque,
                  # generator-ctor state that shapes output (replayed as ctor
                  # kwargs; a truthy launch_config_robot is opaque -> no replay)
                  "codegen": {
                      "DEBUG_MODE": bool(getattr(self, "DEBUG_MODE", False)),
                      "gen_print_mat": bool(getattr(self, "gen_print_mat", False)),
                      "file_namespace": getattr(self, "file_namespace", "grid"),
                      "USE_JOINT_DYNAMICS": bool(getattr(self, "USE_JOINT_DYNAMICS", False)),
                      "MUJOCO_OUTPUT": bool(getattr(self, "MUJOCO_OUTPUT", False)),
                      "launch_config_profile": getattr(self, "launch_config_profile", "host"),
                      "runtime_joint_dynamics": bool(getattr(self, "runtime_joint_dynamics", False)),
                      "launch_config_robot": bool(getattr(self, "launch_config_robot", None)),
                      # fp64 (audit 2026-09-18): the ctor's dtype="double" folds into
                      # cuda_shared_mem_type_size_bytes (8 vs 4) and reshapes every
                      # arena/spill decision — the same emission-shaping class as
                      # GRID_ENABLE_MUJOCO_KERNELS (7.z14). Record the RESOLVED byte
                      # size (covers ctor dtype AND the env override in one value);
                      # replay reconstructs dtype from it.
                      "t_bytes": int(getattr(self, "cuda_shared_mem_type_size_bytes", 4)),
                  },
                  "env": _env_snapshot(),
                  "urdf_sha256": _urdf_sha(name),
                  "content_sha256": _hash(out_path)})
        except Exception:
            pass
        return result

    try:
        from test.cuda_equivalents import cuda_harness
    except ImportError:
        cuda_harness = None
    orig_flagship = getattr(cuda_harness, "_generate_grid_header", None)

    def flagship_wrapper(project_model, resolved_model, build_dir, config,
                         codegen_algorithm_list=None):
        header_path, header_key = orig_flagship(
            project_model, resolved_model, build_dir, config,
            codegen_algorithm_list=codegen_algorithm_list)
        try:
            emit({"kind": "flagship",
                  "robot": project_model.spec.robot_id,
                  "base_mode": project_model.base_mode,
                  "header_key": header_key,
                  "algorithm_list": (sorted(codegen_algorithm_list)
                                     if codegen_algorithm_list else None),
                  "env": _env_snapshot(),
                  "content_sha256": _hash(header_path)})
        except Exception:
            pass
        return header_path, header_key

    GRiDCodeGenerator.gen_all_code = gen_wrapper
    if orig_flagship is not None:
        cuda_harness._generate_grid_header = flagship_wrapper
    try:
        yield
    finally:
        GRiDCodeGenerator.gen_all_code = orig_gen
        if orig_flagship is not None:
            cuda_harness._generate_grid_header = orig_flagship


@pytest.fixture(scope="session", autouse=True)
def _pin_only_headers():
    """Default `gen_all_code` to `enable_mujoco_kernels=False` for this directory.

    Honors a caller-set `GRID_ENABLE_MUJOCO_KERNELS` (e.g. a deliberate
    `GRID_ENABLE_MUJOCO_KERNELS=1` sweep) rather than overriding it.
    """
    key = "GRID_ENABLE_MUJOCO_KERNELS"
    preset = os.environ.get(key)
    if preset is None:
        os.environ[key] = "0"
    try:
        yield
    finally:
        if preset is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = preset
