"""Consumer artifacts: a caller's CUDA sources compiled against a generated `grid.cuh` under
the store's rules (HJCD-IK ask G9, 2026-10-23).

A consumer such as an IK or trajectory-optimization solver builds one kernel translation unit
per robot against the header GRiD generates for that robot. `build_artifact` runs that build
the way `register_robot` builds its own wrapper: the sources are generated into a staging
directory, hashed with every compile input into a content key, compiled only if that key has
no artifact yet (each unit from the object cache when its exact inputs were compiled before),
and published atomically under a namespace of the cache directory that belongs to the
consumer:

    <cache_dir>/consumers/<consumer>/store/<content_key>/   grid.cuh, <artifact_name>, meta.json,
                                                            build_inputs.json, build.log
    <cache_dir>/consumers/<consumer>/bykey/<input_key>      pointer from the request to the content
    <cache_dir>/consumers/<consumer>/objects/               per-unit objects, keyed by their inputs
    <cache_dir>/consumers/<consumer>/manifest.json          name -> content key, under a lock

The identity covers the URDF bytes, the codegen options, the codegen tree, the consumer's
sources and headers by content, the defines and flags, the toolchain, the CUDA architecture
and the package version; a pointer is honoured only when the identity recorded beside the
artifact equals the current one. `load_artifact` returns a prepared artifact without a
toolchain, refusing one recorded for another architecture or key schema. The caller loads the
artifact itself (dlopen, ctypes, its own binding); GRiD never binds the consumer's symbols.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import _cache
from ._cache import (_KEY_SCHEMA, _arch_loadable, _sweep_dead_staging_dirs, canonical_options,
                     default_cache_dir, detect_cuda_arch, effective_cuda_archs, keymap_lookup,
                     keymap_record, package_version, read_build_inputs, stale_hit_reasons,
                     write_build_inputs)
from ._compile import _NVCC_DEFAULT_FLAGS, _build_jobs, find_nvcc, generate_grid_cuh
from ._store import entry_metadata, publish_entry, publish_object, staging_dir
from ._consumer_toolchain import resolve_consumer_toolchain
from ._consumer_dependencies import discover_dependencies
from ._model_artifact import (check_expected_id, file_digest, record_model_association,
                              validate_model_association)

_log = logging.getLogger("grid_rbd")

ARTIFACT_SCHEMA = 2
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# What a load must match: key schema, device, and the ABI for Python extensions. The
# toolchain, GLASS content and the codegen tree are provenance, recorded but not required to
# load a prepared artifact on a box that never had them.
LOAD_IDENTITY_KEYS = ("key_schema", "artifact_schema", "cuda_arch", "python_abi")
_SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx")


class ConsumerArtifactError(RuntimeError):
    """Base of the artifact errors."""


class ArtifactNotFoundError(ConsumerArtifactError, KeyError):
    """No artifact of that name in the consumer's manifest."""


class ArtifactIncompatibleError(ConsumerArtifactError):
    """The recorded artifact cannot be loaded here (another device or key schema)."""


@dataclass(frozen=True)
class ConsumerArtifact:
    consumer: str
    name: str
    path: Path            # the shared object
    header: Path          # the generated grid.cuh it was compiled against
    store_dir: Path
    content_key: str
    input_key: str | None
    cuda_arch: int
    meta: dict[str, Any]
    compiled: bool        # nvcc ran in this call (False: reused or loaded)

    @property
    def model_descriptor(self):
        """Validated coordinate metadata, not complete dynamics/artifact identity."""
        try:
            return validate_model_association(self.store_dir, self.meta, self.path.name)
        except ValueError as exc:
            raise ArtifactIncompatibleError(str(exc)) from exc


# ─── paths ──────────────────────────────────────────────────────────────────
def consumer_root(cache_dir: Path, consumer: str) -> Path:
    return cache_dir / "consumers" / consumer


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError(f"{what} must be a short identifier (letters, digits, '_', '.', '-'), got {value!r}")
    return value


def _artifact_module_hash() -> str:
    try:
        h = hashlib.sha256(Path(__file__).read_bytes())
        h.update(Path(__file__).with_name("_consumer_toolchain.py").read_bytes())
        h.update(Path(__file__).with_name("_consumer_dependencies.py").read_bytes())
        h.update(Path(__file__).with_name("_model_artifact.py").read_bytes())
        return h.hexdigest()
    except OSError:
        return ""


def _collision_include_dir() -> Path:
    import grid_codegen
    return Path(grid_codegen.__file__).resolve().parent / "collision"


def _glass_include_dirs() -> list[Path]:
    from grid_codegen.resources import resource_path
    root = Path(resource_path("GLASS"))
    return [root, root / "src"]


def artifact_identity(cuda_arch: int) -> dict[str, Any]:
    """Every artifact-shaping input that is not a caller argument, recorded beside each entry."""
    identity: dict[str, Any] = {
        "key_schema": _KEY_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "cuda_arch": int(cuda_arch),
        "nvcc": _cache._nvcc_identity(),
        "host_cxx": _cache._host_cxx_tag(),
        "glass_content": _cache._glass_content_hash(),
        "glass_commit": _cache._glass_tag(),
        "codegen_tree": _cache._codegen_source_hash(),
        "compile_py": _cache._compile_source_hash(),
        "artifacts_py": _artifact_module_hash(),
        "grid_rbd": package_version(),
        "generation_env": _cache._generation_env(),
    }
    archs = effective_cuda_archs(cuda_arch)
    if archs is not None:
        identity["cuda_archs"] = archs
    return identity


def _python_abi_identity() -> dict[str, Any]:
    """Interpreter ABI for declared extensions, without querying a compiler."""
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:2]),
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "debug": bool(sysconfig.get_config_var("Py_DEBUG")),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    }


def _validate_python_exports(path: Path, python_extension: bool) -> None:
    """Validate the pre-lookup declaration; inspecting a binary alone is too late
    to protect reuse. Consumer builds require nm, prepared loads do not."""
    result = subprocess.run(["nm", "-D", "--defined-only", str(path)],
                            capture_output=True, text=True, check=True)
    has_pyinit = any(line.split() and line.split()[-1].startswith("PyInit_")
                     for line in result.stdout.splitlines())
    if has_pyinit != python_extension:
        raise ValueError("python_extension declaration disagrees with exported PyInit_* symbols; "
                         "pass python_extension=True for a Python extension and False for a native library")


def _collision_mesh_identity(urdf_path: Path) -> list[dict[str, str]]:
    """Resolve with the spherizer's rules before considering a fast cache hit."""
    from grid_codegen.algorithms._spherize import _resolve_mesh_path
    dependencies = []
    for link in ET.parse(urdf_path).getroot().findall("link"):
        for mesh in link.findall("collision/geometry/mesh"):
            filename = mesh.get("filename", "")
            resolved = _resolve_mesh_path(filename, str(urdf_path.parent))
            if resolved is None:
                raise ValueError(f"collision mesh {filename!r} for link {link.get('name')!r} "
                                 "could not be resolved; provide the asset locally")
            path = Path(resolved).resolve()
            dependencies.append({"uri": filename, "path": str(path), "sha256": file_digest(path)})
    return dependencies


def _resolve_files(values: Sequence[str | Path], what: str, suffixes: tuple[str, ...] | None) -> list[Path]:
    out: list[Path] = []
    for v in values:
        p = Path(v).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{what} not found: {p}")
        if suffixes is not None and p.suffix not in suffixes:
            raise ValueError(f"{what} {p.name}: expected one of {suffixes}")
        out.append(p)
    if len({p.name for p in out}) != len(out):
        raise ValueError(f"{what} must have distinct file names (each becomes one object): {[p.name for p in out]}")
    return out


def _normalize_defines(defines: Sequence[str] | dict[str, Any] | None) -> list[str]:
    if not defines:
        return []
    if isinstance(defines, dict):
        items = [k if v is None else f"{k}={v}" for k, v in defines.items()]
    else:
        items = [str(d) for d in defines]
    for d in items:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(=.*)?$", d):
            raise ValueError(f"define {d!r} is not NAME or NAME=VALUE")
    return sorted(items)


# ─── code options (the subset of register_robot's that shape grid.cuh) ────────
def _code_options(urdf_path: Path, cuda_arch: int, *, floating_base, ee_joint_names, dtype, algorithm_list,
                  enable_mujoco_kernels, cost_frames, contact_frames, use_joint_dynamics, friction_model,
                  friction_width, runtime_inertia, runtime_transform, runtime_joint_dynamics, emit_plant,
                  collision, collision_res, collision_native, spherized_urdf, strict_collision,
                  launch_config_profile, runtime_armature: bool = False, armature=None) -> dict[str, Any]:
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
    opts: dict[str, Any] = {
        "floating_base": bool(floating_base),
        "ee_joint_names": list(ee_joint_names) if ee_joint_names else [],
        "cuda_arch": int(cuda_arch),
    }
    if dtype == "float64":
        opts["dtype"] = "float64"
    if algorithm_list is not None:
        if isinstance(algorithm_list, str):
            algos = [a.strip() for a in algorithm_list.replace(";", ",").split(",") if a.strip()]
        else:
            algos = [str(a).strip() for a in algorithm_list if str(a).strip()]
        if not algos:
            raise ValueError("algorithm_list must name at least one algorithm or profile")
        opts["algorithm_list"] = sorted(set(algos))
    if not enable_mujoco_kernels:
        opts["enable_mujoco_kernels"] = False
    if cost_frames:
        opts["cost_frames"] = [str(n) for n in cost_frames]
    if contact_frames:
        opts["contact_frames"] = [str(n) for n in contact_frames]
    if use_joint_dynamics:
        opts["use_joint_dynamics"] = True
    if friction_model not in ("sign", "tanh"):
        raise ValueError(f"friction_model must be 'sign' or 'tanh', got {friction_model!r}")
    if friction_model == "tanh" or friction_width is not None:
        if not use_joint_dynamics:
            raise ValueError("friction_model='tanh' / friction_width require use_joint_dynamics=True")
        if friction_model != "tanh":
            raise ValueError("friction_width applies to friction_model='tanh' only")
        opts["friction_model"] = "tanh"
        if friction_width is not None:
            try:
                opts["friction_width"] = [float(w) for w in friction_width]
            except TypeError:
                opts["friction_width"] = float(friction_width)
    for flag, value in (("runtime_inertia", runtime_inertia), ("runtime_transform", runtime_transform),
                        ("runtime_joint_dynamics", runtime_joint_dynamics), ("runtime_armature", runtime_armature)):
        if value:
            opts[flag] = True
    if armature is not None:
        from grid_codegen.armature import option_vector
        opts["armature"] = option_vector(armature)
    if emit_plant is not None:
        opts["emit_plant"] = bool(emit_plant)
    else:
        opts["emit_plant"] = None     # the generator's rule (E1) decides; recorded so it is keyed
    if sum(bool(x) for x in (collision, collision_native, spherized_urdf)) > 1:
        raise ValueError("pass one of collision=True, collision_native=True or spherized_urdf=")
    if spherized_urdf:
        sp = Path(spherized_urdf).expanduser().resolve()
        if not sp.is_file():
            raise FileNotFoundError(f"spherized_urdf not found: {sp}")
        opts["spherized_urdf"] = str(sp)
        opts["spherized_urdf_sha256"] = file_digest(sp)   # the key follows the model's content
    elif collision_native:
        opts["collision_native"] = True
        opts["collision_res"] = float(collision_res) if not isinstance(collision_res, (list, tuple)) \
            else [float(r) for r in collision_res]
    elif collision:
        opts["collision"] = True
        opts["collision_res"] = float(collision_res) if not isinstance(collision_res, (list, tuple)) \
            else [float(r) for r in collision_res]
    if collision or collision_native:
        opts["collision_meshes"] = _collision_mesh_identity(urdf_path)
        if not strict_collision:
            opts["strict_collision"] = False
    elif not strict_collision:
        raise ValueError("strict_collision=False applies to the spherizer (collision=True or "
                         "collision_native=True); a supplied spherized_urdf is validated, not spherized")
    if launch_config_profile is not None:
        opts["launch_config_profile"] = str(launch_config_profile)
    # The baked per-algorithm launch configuration is not derivable from the URDF bytes; fold
    # the resolved values in so a re-autotune re-keys the artifact (as register_robot does).
    try:
        from grid_codegen.GRiDCodeGenerator import load_launch_config
        from grid_codegen.launch_config import select_launch_config_gpu
        from ._compile import _resolve_launch_config_robot
        lc_robot = _resolve_launch_config_robot(str(urdf_path))
        lc_gpu = select_launch_config_gpu(lc_robot, cuda_arch)
        opts["launch_config_gpu"] = lc_gpu
        lc = load_launch_config(lc_robot, bool(floating_base), gpu=lc_gpu,
                                profile=opts.get("launch_config_profile", "ffi"))
        if lc:
            opts["launch_config"] = {k: dict(v) for k, v in sorted(lc.items())}
    except Exception:
        pass
    return opts


# ─── keys ───────────────────────────────────────────────────────────────────
def _input_key(urdf_bytes: bytes, code_options: dict[str, Any], sources: list[Path], headers: list[Path],
               include_dirs: list[Path], defines: list[str], extra_flags: list[str], link_flags: list[str],
               artifact_name: str, identity: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(urdf_bytes)
    h.update(canonical_options(code_options).encode())
    for p in sources:
        h.update(b"source:" + str(p).encode() + b"\0" + p.read_bytes() + b"\0")
    for p in headers:
        h.update(b"header:" + p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    h.update(("include_dirs=" + ";".join(str(d) for d in include_dirs)).encode())
    h.update(("defines=" + ";".join(defines)).encode())
    h.update(("extra_flags=" + ";".join(extra_flags)).encode())
    h.update(("link_flags=" + ";".join(link_flags)).encode())
    h.update(("artifact_name=" + artifact_name).encode())
    h.update(("identity=" + canonical_options(identity)).encode())
    return h.hexdigest()


def _toolchain_tag() -> str:
    return (f"nvcc={_cache._nvcc_identity()};host_cxx={_cache._host_cxx_tag()};"
            f"glass={_cache._glass_content_hash()};key_schema={_KEY_SCHEMA};artifact_schema={ARTIFACT_SCHEMA}")


def _hash_build_context(h, python_abi, toolchain, dependencies) -> None:
    """The declarations both build keys open with: the Python ABI, the consumer toolchain
    identity and the dependency capture, each canonically serialized."""
    h.update(("python_abi=" + canonical_options(python_abi)).encode())
    h.update(("consumer_toolchain=" + canonical_options(toolchain)).encode())
    h.update(("dependencies=" + canonical_options(dependencies)).encode())


def _hash_compile_flags(h, compile_cmd: list[str], gen_dir: Path) -> None:
    """Every nvcc argument except the staging directory's own paths, whose content is hashed."""
    src = str(gen_dir)
    for a in compile_cmd[1:]:
        if a == f"-I{src}" or a.startswith(src + "/"):
            continue
        h.update(a.encode() + b"\0")


def _content_key(gen_dir: Path, sources: list[Path], headers: list[Path], compile_cmd: list[str],
                 link_flags: list[str], artifact_name: str, *, python_abi, toolchain,
                 dependencies) -> str:
    """Exactly what nvcc sees: the generated header, the static collision header, the
    consumer's sources and headers, the flag list (the staging dir's own -I removed; its
    content is hashed), the link flags and the toolchain."""
    h = hashlib.sha256()
    _hash_build_context(h, python_abi, toolchain, dependencies)
    h.update((gen_dir / "grid.cuh").read_bytes())
    h.update(_collision_include_dir().joinpath("grid_collision_geometry.cuh").read_bytes())
    for p in sources:
        h.update(b"source:" + str(p).encode() + b"\0" + p.read_bytes() + b"\0")
    for p in headers:
        h.update(b"header:" + p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    _hash_compile_flags(h, compile_cmd, gen_dir)
    h.update(("link_flags=" + ";".join(link_flags) + ";artifact_name=" + artifact_name).encode())
    h.update(_toolchain_tag().encode())
    return h.hexdigest()


def _unit_object_key(unit: Path, gen_dir: Path, headers: list[Path], compile_cmd: list[str], *,
                     python_abi, toolchain, dependencies) -> str:
    h = hashlib.sha256()
    _hash_build_context(h, python_abi, toolchain, dependencies)
    # nvcc receives this exact resolved path: suffix selects the language and
    # __FILE__ exposes the whole path, not merely its basename.
    h.update(str(unit).encode() + b"\0")
    _hash_compile_flags(h, compile_cmd, gen_dir)
    h.update((gen_dir / "grid.cuh").read_bytes()); h.update(b"\0")
    h.update(_collision_include_dir().joinpath("grid_collision_geometry.cuh").read_bytes()); h.update(b"\0")
    for p in headers:
        h.update(p.read_bytes()); h.update(b"\0")
    h.update(unit.read_bytes())
    h.update(_toolchain_tag().encode())
    return h.hexdigest()


# ─── the compile half ─────────────────────────────────────────────────────────
def _compile_command(gen_dir: Path, cuda_arch: int, include_dirs: list[Path], defines: list[str],
                     toolchain) -> list[str]:
    """The nvcc command of every consumer unit; the toolchain carries the consumer's extra
    flags, the host compiler selection and the symbol isolation flags."""
    nvcc = find_nvcc()
    archs = effective_cuda_archs(cuda_arch)
    gencode_archs = archs if archs is not None else [cuda_arch]
    cmd = [nvcc] + toolchain.prepend + list(_NVCC_DEFAULT_FLAGS)
    cmd += [f"-gencode=arch=compute_{a},code=sm_{a}" for a in gencode_archs]
    cmd.append(f"-I{gen_dir}")
    cmd.append(f"-I{_collision_include_dir()}")
    for d in _glass_include_dirs():
        cmd.append(f"-I{d}")
    for d in include_dirs:
        cmd.append(f"-I{d}")
    for d in defines:
        cmd.append(f"-D{d}")
    cmd.extend(toolchain.compile_flags())
    return cmd


def _build(gen_dir: Path, sources: list[Path], headers: list[Path], compile_cmd: list[str], link_flags: list[str],
           out_so: Path, object_cache_dir: Path, reuse_objects: bool, *, validate_unit,
           python_abi, toolchain, dependencies) -> dict[str, Any]:
    """Compile each consumer source with -c (from the object cache when its key exists), then
    host-link the objects into `out_so`. Returns a record of what ran. `validate_unit(unit)`
    runs before a unit is compiled and before its object is published; it raises when the
    build inputs or that unit's dependencies changed."""
    nvcc = compile_cmd[0]
    jobs = _build_jobs()
    build_dir = gen_dir / "build"
    build_dir.mkdir(exist_ok=True)
    object_cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = gen_dir / "build.log"
    lock = threading.Lock()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(compile_cmd) + " -c <unit> -o <unit>.o   # x" + str(len(sources)) + "\n")
        log.write("$ " + " ".join([nvcc, "--shared", "-o", str(out_so), "<units>.o"] + link_flags) + "\n\n")
    record: dict[str, Any] = {"units": {}}

    def one(unit: Path) -> tuple[Path, int, float, str]:
        validate_unit(unit)
        key = _unit_object_key(unit, gen_dir, headers, compile_cmd, python_abi=python_abi,
                               toolchain=toolchain.identity, dependencies=dependencies["units"][str(unit)])
        cached = object_cache_dir / f"{unit.name}-{key}.o"
        if reuse_objects and cached.is_file():
            os.utime(cached)
            with lock, log_path.open("a") as log:
                log.write(f"[{unit.stem}] cached {cached.name}\n")
            record["units"][unit.name] = {"object": cached.name, "compiled": False}
            return cached, 0, 0.0, ""
        obj = build_dir / (unit.name + ".o")
        ulog = build_dir / (unit.name + ".log")
        t0 = time.monotonic()
        with ulog.open("w") as f:
            r = subprocess.run(compile_cmd + ["-c", str(unit), "-o", str(obj)], stdout=f,
                               stderr=subprocess.STDOUT, env=toolchain.environment())
        secs = time.monotonic() - t0
        with lock, log_path.open("a") as log:
            log.write(f"[{unit.stem}] rc={r.returncode} {secs:.1f}s\n")
        if r.returncode == 0:
            # Reject changed inputs before exposing an object under its old key,
            # not merely before publishing the final linked artifact.
            validate_unit(unit)
            publish_object(obj, cached)
            obj = cached
            record["units"][unit.name] = {"object": cached.name, "compiled": True, "seconds": round(secs, 1)}
        return obj, r.returncode, secs, ulog.read_text()

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(one, sources))
    failed = [(obj, rc, out) for obj, rc, _, out in results if rc != 0]
    if failed:
        obj, rc, out = failed[0]
        raise RuntimeError(f"nvcc exit {rc} compiling the consumer unit {obj.stem}:\n{out}")
    objs = [str(obj) for obj, _, _, _ in results]
    link = toolchain.link_command(nvcc, out_so, objs)
    t0 = time.monotonic()
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(link) + "\n")
        log.flush()
        r = subprocess.run(link, stdout=log, stderr=subprocess.STDOUT, env=toolchain.environment())
        log.write(f"[link] rc={r.returncode} {time.monotonic() - t0:.1f}s\n")
    if r.returncode != 0:
        raise RuntimeError(f"nvcc exit {r.returncode} linking the consumer artifact; build log:\n" + log_path.read_text())
    shutil.rmtree(build_dir, ignore_errors=True)
    record["compiled_units"] = sum(1 for _, _, secs, _ in results if secs > 0)
    return record


# ─── manifest (one per consumer, under a lock) ───────────────────────────────
def _schema_ok(record: dict[str, Any]) -> bool:
    """The record declares this artifact schema as an int (not a bool or a string)."""
    return type(record.get('artifact_schema')) is int and record['artifact_schema'] == ARTIFACT_SCHEMA


def _abi_disagreement(meta: dict[str, Any], recorded: dict[str, Any] | None,
                      python_extension: bool, python_abi) -> str | None:
    """Why the metadata or the identity record does not declare the requested extension
    ABI, or None when both agree with it."""
    if (meta.get("python_extension") is not python_extension or
            "python_abi" not in meta or meta["python_abi"] != python_abi):
        return "python_abi: metadata declaration disagrees with requested artifact"
    if recorded is not None and "python_abi" not in recorded:
        return "python_abi: no extension declaration recorded"
    if recorded is not None and recorded["python_abi"] != python_abi:
        return "python_abi: identity record disagrees with requested artifact"
    return None


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _load_manifest(root: Path) -> dict[str, Any]:
    path = _manifest_path(root)
    if not path.exists():
        return {"artifact_schema": ARTIFACT_SCHEMA, "artifacts": {}}
    data = json.loads(path.read_text())
    if data.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ConsumerArtifactError(
            f"{path} was written with artifact schema {data.get('artifact_schema')!r}; this grid_rbd expects "
            f"{ARTIFACT_SCHEMA}. Point cache_dir= at a fresh directory and build the artifacts again.")
    return data


def _manifest_register(root: Path, name: str, content_key: str, meta: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = _load_manifest(root)
            manifest["artifacts"][name] = {"content_key": content_key, **meta}
            tmp = _manifest_path(root).with_suffix(f".json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            tmp.replace(_manifest_path(root))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# ─── public operations ───────────────────────────────────────────────────────
def build_artifact(consumer: str, name: str, urdf_path: str | Path, *,
                   sources: Sequence[str | Path],
                   headers: Sequence[str | Path] = (),
                   include_dirs: Sequence[str | Path] = (),
                   defines: Sequence[str] | dict[str, Any] | None = None,
                   extra_flags: Sequence[str] = (),
                   link_flags: Sequence[str] = (),
                   artifact_name: str = "artifact.so",
                   python_extension: bool = False,
                   expected_descriptor_id: str | None = None,
                   cache_dir: str | Path | None = None,
                   cuda_arch: int | None = None,
                   force_rebuild: bool = False,
                   plan_only: bool = False,
                   floating_base: bool = False,
                   ee_joint_names: Sequence[str] | None = None,
                   dtype: str = "float32",
                   algorithm_list: Sequence[str] | str | None = None,
                   enable_mujoco_kernels: bool = True,
                   cost_frames: Sequence[str] | None = None,
                   contact_frames: Sequence[str] | None = None,
                   use_joint_dynamics: bool = False,
                   friction_model: str = "sign",
                   friction_width=None,
                   runtime_inertia: bool = False,
                   runtime_transform: bool = False,
                   runtime_joint_dynamics: bool = False,
                   runtime_armature: bool = False,
                   armature=None,
                   emit_plant: bool | None = None,
                   collision: bool = False,
                   collision_res: float | Sequence[float] = 0.05,
                   collision_native: bool = False,
                   spherized_urdf: str | Path | None = None,
                   strict_collision: bool = True,
                   launch_config_profile: str | None = None) -> ConsumerArtifact | dict[str, Any]:
    """Build (or reuse) the consumer's artifact for one robot and return where it is.

    `consumer` names the namespace in the cache directory (one per consuming project);
    `name` the artifact within it (one per robot and configuration). `sources` are the
    consumer's translation units, each compiled with `-c` against the generated `grid.cuh`
    (plus the static collision header and the GLASS include directories) and linked with
    `--shared` into `artifact_name`. Non-system dependencies are discovered with nvcc
    before reuse/build and hashed by path and contents; `headers` supplements that
    discovery with explicitly declared dependencies. System headers excluded by -MM
    are an environment assumption, not covered by compiler identity alone.
    `defines` are `NAME` or `NAME=VALUE`;
    `extra_flags` go to every unit compile, `link_flags` to the link.
    Declare `python_extension=True` for Python/pybind extensions before cache lookup.
    Their interpreter ABI participates in pointer, content and object identity and is
    checked on prepared loads. Native C libraries (the default) are Python-independent.
    Builds check the declaration against exported PyInit_* symbols using `nm`.
    `expected_descriptor_id` optionally checks prior CPU model inspection on
    cold and reused builds. Schema-2 artifacts bind validated coordinate
    metadata to the generated header and binary; older artifacts need rebuilding.

    The codegen keywords (`floating_base`, `ee_joint_names`, `dtype`, `algorithm_list`,
    `enable_mujoco_kernels`, `cost_frames`, `contact_frames`, the joint-dynamics and runtime
    knobs) mean what they mean on `register_robot`; `emit_plant` is the generator's plant-block
    switch (None: the generator's rule), and `collision`, `collision_res`, `collision_native`
    and `spherized_urdf` select the generated `grid_collision` namespace (a sphere model is
    validated against the URDF). `strict_collision` (default True) makes the spherizer's
    warnings errors on the `collision` and `collision_native` routes: a mesh whose voxel
    fill failed and would be replaced by a bounding-box cover, or a model with no spheres,
    stops the build instead of silently changing the clearance rows; False accepts them
    with the warning. `launch_config_profile` (default: the bindings' "ffi" profile) names
    the baked launch-configuration profile.

    `plan_only=True` reports what a build would use without generating or compiling;
    validating an existing entry still invokes dependency preprocessing. Cold plans
    cannot know dependencies until a generated header exists. `force_rebuild=True`
    recompiles every unit. Prepared `load_artifact` requires neither source nor compiler."""
    _check_name(consumer, "consumer")
    _check_name(name, "name")
    _check_name(artifact_name, "artifact_name")
    if not isinstance(python_extension, bool):
        raise TypeError("python_extension must be a bool")
    check_expected_id(expected_descriptor_id)
    urdf_p = Path(urdf_path).expanduser().resolve()
    if not urdf_p.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_p}")
    urdf_bytes = urdf_p.read_bytes()
    if expected_descriptor_id is not None:
        from ._model import describe_model
        actual = describe_model(urdf_p, floating_base=floating_base,
                                runtime_transform=runtime_transform)['descriptor_id']
        if actual != expected_descriptor_id:
            raise ValueError('expected_descriptor_id does not match requested model')
    srcs = _resolve_files(list(sources), "source", _SOURCE_SUFFIXES)
    if not srcs:
        raise ValueError("sources must name at least one translation unit")
    hdrs = _resolve_files(list(headers), "header", None)
    incs = [Path(d).expanduser().resolve() for d in include_dirs]
    for d in incs:
        if not d.is_dir():
            raise FileNotFoundError(f"include_dirs entry is not a directory: {d}")
    defs = _normalize_defines(defines)
    xflags = [str(f) for f in extra_flags]
    lflags = [str(f) for f in link_flags]
    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    root = consumer_root(cache_root, consumer)
    if cuda_arch is None:
        cuda_arch = detect_cuda_arch()
    if cuda_arch == 0:
        raise RuntimeError("Could not detect the GPU compute capability; pass cuda_arch= explicitly "
                           "(for example cuda_arch=120 for an sm_120 card).")
    code_options = _code_options(
        urdf_p, cuda_arch, floating_base=floating_base, ee_joint_names=ee_joint_names, dtype=dtype,
        algorithm_list=algorithm_list, enable_mujoco_kernels=enable_mujoco_kernels, cost_frames=cost_frames,
        contact_frames=contact_frames, use_joint_dynamics=use_joint_dynamics, friction_model=friction_model,
        friction_width=friction_width, runtime_inertia=runtime_inertia, runtime_transform=runtime_transform,
        runtime_joint_dynamics=runtime_joint_dynamics, runtime_armature=runtime_armature, armature=armature,
        emit_plant=emit_plant, collision=collision,
        collision_res=collision_res, collision_native=collision_native, spherized_urdf=spherized_urdf,
        strict_collision=strict_collision, launch_config_profile=launch_config_profile)
    identity = artifact_identity(cuda_arch)
    toolchain = resolve_consumer_toolchain(xflags, lflags)
    identity["host_cxx"] = toolchain.identity
    python_abi = _python_abi_identity() if python_extension else None
    identity["python_abi"] = python_abi
    input_key = _input_key(urdf_bytes, code_options, srcs, hdrs, incs, defs, xflags, lflags, artifact_name, identity)
    store = root / "store"

    def check_inputs_unchanged():
        toolchain.environment()
        current_options = dict(code_options)
        if collision or collision_native:
            current_options["collision_meshes"] = _collision_mesh_identity(urdf_p)
        if spherized_urdf:
            current_options["spherized_urdf_sha256"] = file_digest(Path(code_options["spherized_urdf"]))
        current_key = _input_key(urdf_p.read_bytes(), current_options, srcs, hdrs, incs,
                                 defs, xflags, lflags, artifact_name, identity)
        if current_key != input_key:
            raise RuntimeError("consumer build inputs changed during generation or compilation; "
                               "no artifact published; retry with immutable inputs")

    def _entry_state(entry_dir: Path) -> list[str]:
        meta = entry_metadata(entry_dir, artifact_name)
        if meta is None:
            return ["entry incomplete (artifact or meta.json missing)"]
        if not _schema_ok(meta):
            return ['artifact_schema: incompatible metadata; rebuild artifact']
        try:
            validate_model_association(entry_dir, meta, artifact_name, expected_descriptor_id)
        except ValueError as exc:
            return [str(exc)]
        recorded = read_build_inputs(entry_dir)
        if recorded is not None and not _schema_ok(recorded):
            return ['artifact_schema: incompatible identity record; rebuild artifact']
        disagreement = _abi_disagreement(meta, recorded, python_extension, python_abi)
        if disagreement:
            return [disagreement]
        stale = stale_hit_reasons(entry_dir, identity)
        if stale:
            return stale
        command = _compile_command(entry_dir, cuda_arch, incs, defs, toolchain)
        current = discover_dependencies(entry_dir, srcs, command, toolchain)
        check_inputs_unchanged()
        if meta.get("dependencies") != current:
            return ["dependencies: selected files or contents changed (or no capture recorded)"]
        return []

    if plan_only:
        mapped = keymap_lookup(root, input_key)
        entry_dir = store / mapped if mapped else None
        stale = _entry_state(entry_dir) if entry_dir is not None else None
        return {
            "consumer": consumer, "name": name, "cache_dir": str(cache_root), "cuda_arch": cuda_arch,
            "code_options": code_options, "identity": identity, "input_key": input_key,
            "sources": [str(p) for p in srcs], "headers": [str(p) for p in hdrs], "defines": defs,
            "content_key": mapped if (mapped and stale == []) else None,
            "cached": bool(mapped) and stale == [], "stale_reasons": stale,
            "store_dir": str(entry_dir) if (mapped and stale == []) else None,
            "would": "load" if (mapped and stale == []) else "generate+compile (the content key decides whether compilation is needed)",
        }

    if not force_rebuild:
        mapped = keymap_lookup(root, input_key)
        if mapped:
            entry_dir = store / mapped
            stale = _entry_state(entry_dir)
            if not stale:
                meta = json.loads((entry_dir / "meta.json").read_text())
                validate_model_association(entry_dir, meta, artifact_name, expected_descriptor_id)
                _manifest_register(root, name, mapped, meta)
                return ConsumerArtifact(consumer, name, entry_dir / artifact_name, entry_dir / "grid.cuh",
                                        entry_dir, mapped, input_key, cuda_arch, meta, False)
            _log.info("grid_rbd: artifact pointer for %s/%s rejected (%s); regenerating", consumer, name, "; ".join(stale))

    store.mkdir(parents=True, exist_ok=True)
    _sweep_dead_staging_dirs(store)
    gen_dir = staging_dir(store, input_key)
    compiled = False
    try:
        generate_grid_cuh(urdf_p, code_options, gen_dir / "grid.cuh")
        check_inputs_unchanged()
        compile_cmd = _compile_command(gen_dir, cuda_arch, incs, defs, toolchain)
        dependencies = discover_dependencies(gen_dir, srcs, compile_cmd, toolchain)

        def check_build_inputs():
            check_inputs_unchanged()
            if discover_dependencies(gen_dir, srcs, compile_cmd, toolchain) != dependencies:
                raise RuntimeError("consumer dependencies changed during build; no artifact published; "
                                   "retry with immutable inputs")

        def check_unit_inputs(unit):
            # Check the unit being exposed, not every unrelated unit around each
            # compile. Whole-build checks still bracket final publication.
            check_inputs_unchanged()
            current = discover_dependencies(gen_dir, [unit], compile_cmd, toolchain)
            if current["units"][str(unit)] != dependencies["units"][str(unit)]:
                raise RuntimeError("consumer dependencies changed during unit compilation; "
                                   "retry with immutable inputs")

        check_build_inputs()
        content_key = _content_key(gen_dir, srcs, hdrs, compile_cmd, lflags, artifact_name,
                                   python_abi=python_abi, toolchain=toolchain.identity,
                                   dependencies=dependencies)
        entry_dir = store / content_key
        cached_meta = entry_metadata(entry_dir, artifact_name)
        recorded = read_build_inputs(entry_dir) if cached_meta is not None else None
        # A sidecar can be damaged independently of the content-addressed binary.
        # Do not return a hit that load_artifact would reject, or relabel an
        # incompatible declaration. Rebuild and replace under the publication lock.
        replace = force_rebuild
        if cached_meta is not None:
            replace = replace or _abi_disagreement(cached_meta, recorded, python_extension, python_abi) is not None
            replace = replace or cached_meta.get("dependencies") != dependencies
            replace = replace or not _schema_ok(cached_meta) or (recorded is not None and (
                not _schema_ok(recorded) or recorded.get('key_schema') != _KEY_SCHEMA))
            try:
                validate_model_association(entry_dir, cached_meta, artifact_name, expected_descriptor_id)
            except ValueError:
                replace = True
        if cached_meta is not None and not replace:
            check_build_inputs()
            meta = cached_meta
            if read_build_inputs(entry_dir) is None:
                write_build_inputs(entry_dir, identity)
        else:
            _log.info("grid_rbd: building consumer artifact %s/%s (%d unit(s))", consumer, name, len(srcs))
            t0 = time.monotonic()
            record = _build(gen_dir, srcs, hdrs, compile_cmd, lflags, gen_dir / artifact_name,
                            root / "objects", reuse_objects=not force_rebuild, validate_unit=check_unit_inputs,
                            python_abi=python_abi, toolchain=toolchain, dependencies=dependencies)
            _validate_python_exports(gen_dir / artifact_name, python_extension)
            check_build_inputs()
            compiled = record["compiled_units"] > 0
            meta = {
                "consumer": consumer, "name": name, "artifact": artifact_name,
                "urdf": str(urdf_p), "urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
                "sources": {p.name: file_digest(p) for p in srcs},
                "source_paths": [str(p) for p in srcs],
                "headers": {p.name: file_digest(p) for p in hdrs},
                "include_dirs": [str(d) for d in incs], "defines": defs,
                "extra_flags": xflags, "link_flags": lflags,
                "python_extension": python_extension, "python_abi": python_abi,
                "consumer_toolchain": toolchain.identity,
                "dependencies": dependencies,
                "code_options": code_options, "cuda_arch": int(cuda_arch),
                "grid_rbd": package_version(), "artifact_schema": ARTIFACT_SCHEMA,
                "build_seconds": round(time.monotonic() - t0, 1), "build": record,
                "model_association": record_model_association(gen_dir, artifact_name),
            }
            archs = effective_cuda_archs(cuda_arch)
            if archs is not None:
                meta["cuda_archs"] = archs
            (gen_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
            write_build_inputs(gen_dir, identity)
            check_build_inputs()
            meta = publish_entry(gen_dir, entry_dir, artifact_name, force=replace)
        check_inputs_unchanged()
        validate_model_association(entry_dir, meta, artifact_name, expected_descriptor_id)
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
    keymap_record(root, input_key, content_key)
    _manifest_register(root, name, content_key, meta)
    return ConsumerArtifact(consumer, name, entry_dir / artifact_name, entry_dir / "grid.cuh",
                            entry_dir, content_key, input_key, cuda_arch, meta, compiled)


def load_artifact(consumer: str, name: str, cache_dir: str | Path | None = None,
                  cuda_arch: int | None = None, *, expected_descriptor_id: str | None = None) -> ConsumerArtifact:
    """The prepared artifact `name` of `consumer`, without a toolchain: the manifest names it,
    the entry must be whole and recorded for this device (or one of a fat build's devices) and
    key schema. Declared Python extensions must match this interpreter's ABI.
    With no detectable GPU and no `cuda_arch`, the recorded device is trusted."""
    _check_name(consumer, "consumer")
    _check_name(name, "name")
    check_expected_id(expected_descriptor_id)
    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    root = consumer_root(cache_root, consumer)
    entry = _load_manifest(root)["artifacts"].get(name) if _manifest_path(root).exists() else None
    if entry is None:
        raise ArtifactNotFoundError(f"no artifact {name!r} for consumer {consumer!r} under {cache_root}")
    entry_dir = root / "store" / entry["content_key"]
    artifact_name = entry.get("artifact", "artifact.so")
    meta = entry_metadata(entry_dir, artifact_name)
    if meta is None:
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} is incomplete at {entry_dir}")
    recorded = read_build_inputs(entry_dir)
    if recorded is None:
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} predates the identity record")
    if cuda_arch is None:
        cuda_arch = detect_cuda_arch()
    reasons = []
    if "python_abi" not in recorded:
        reasons.append("python_abi: no extension declaration recorded; rebuild the artifact")
    else:
        abi = recorded["python_abi"]
        if (abi is not None and not isinstance(abi, dict)) or (
                meta.get("python_extension") is not (abi is not None)) or (
                "python_abi" not in meta or meta["python_abi"] != abi):
            reasons.append("python_abi: metadata and identity declarations disagree; rebuild the artifact")
        elif abi is not None and abi != _python_abi_identity():
            reasons.append(f"python_abi: recorded {abi!r} != current {_python_abi_identity()!r}")
    if recorded.get("key_schema") != _KEY_SCHEMA:
        reasons.append(f"key_schema: recorded {recorded.get('key_schema')!r} != current {_KEY_SCHEMA!r}")
    if not _schema_ok(recorded):
        reasons.append(f"artifact_schema: recorded {recorded.get('artifact_schema')!r} != current {ARTIFACT_SCHEMA!r}")
    if not _schema_ok(meta):
        reasons.append('artifact_schema: metadata is incompatible; rebuild artifact')
    if cuda_arch and not _arch_loadable(recorded, int(cuda_arch)):
        reasons.append(f"cuda_arch: recorded {recorded.get('cuda_arch')!r} (archs {recorded.get('cuda_archs')!r}) != current {cuda_arch!r}")
    if reasons:
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} cannot be loaded here: " + "; ".join(reasons))
    try:
        validate_model_association(entry_dir, meta, artifact_name, expected_descriptor_id)
    except ValueError as exc:
        raise ArtifactIncompatibleError(str(exc)) from exc
    return ConsumerArtifact(consumer, name, entry_dir / artifact_name, entry_dir / "grid.cuh", entry_dir,
                            entry["content_key"], None, int(recorded.get("cuda_arch", cuda_arch or 0)), meta, False)


def list_artifacts(consumer: str, cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Every artifact the consumer's manifest names, with its metadata."""
    _check_name(consumer, "consumer")
    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    root = consumer_root(cache_root, consumer)
    if not _manifest_path(root).exists():
        return []
    return [{"name": n, **e} for n, e in _load_manifest(root)["artifacts"].items()]
