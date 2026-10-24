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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import _cache
from ._cache import (_KEY_SCHEMA, _arch_loadable, _sweep_dead_staging_dirs, canonical_options,
                     default_cache_dir, detect_cuda_arch, effective_cuda_archs, keymap_lookup,
                     keymap_record, package_version, read_build_inputs, stale_hit_reasons,
                     write_build_inputs)
from ._compile import _NVCC_DEFAULT_FLAGS, _build_jobs, find_nvcc, generate_grid_cuh

_log = logging.getLogger("grid_rbd")

ARTIFACT_SCHEMA = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# What a load must match: the key schema and the device the artifact was compiled for. The
# toolchain, GLASS content and the codegen tree are provenance, recorded but not required to
# load a prepared artifact on a box that never had them.
LOAD_IDENTITY_KEYS = ("key_schema", "artifact_schema", "cuda_arch")
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


# ─── paths ──────────────────────────────────────────────────────────────────
def consumer_root(cache_dir: Path, consumer: str) -> Path:
    return cache_dir / "consumers" / consumer


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError(f"{what} must be a short identifier (letters, digits, '_', '.', '-'), got {value!r}")
    return value


def _artifact_module_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                  collision, collision_res, collision_native, spherized_urdf, launch_config_profile) -> dict[str, Any]:
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
                        ("runtime_joint_dynamics", runtime_joint_dynamics)):
        if value:
            opts[flag] = True
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
        opts["spherized_urdf_sha256"] = _file_digest(sp)   # the key follows the model's content
    elif collision_native:
        opts["collision_native"] = True
        opts["collision_res"] = float(collision_res) if not isinstance(collision_res, (list, tuple)) \
            else [float(r) for r in collision_res]
    elif collision:
        opts["collision"] = True
        opts["collision_res"] = float(collision_res) if not isinstance(collision_res, (list, tuple)) \
            else [float(r) for r in collision_res]
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
        h.update(b"source:" + p.name.encode() + b"\0" + p.read_bytes() + b"\0")
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


def _content_key(gen_dir: Path, sources: list[Path], headers: list[Path], compile_cmd: list[str],
                 link_flags: list[str], artifact_name: str) -> str:
    """Exactly what nvcc sees: the generated header, the static collision header, the
    consumer's sources and headers, the flag list (the staging dir's own -I removed; its
    content is hashed), the link flags and the toolchain."""
    h = hashlib.sha256()
    h.update((gen_dir / "grid.cuh").read_bytes())
    h.update(_collision_include_dir().joinpath("grid_collision_geometry.cuh").read_bytes())
    for p in sources:
        h.update(b"source:" + p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    for p in headers:
        h.update(b"header:" + p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    src = str(gen_dir)
    for a in compile_cmd[1:]:
        if a == f"-I{src}" or a.startswith(src + "/"):
            continue
        h.update(a.encode() + b"\0")
    h.update(("link_flags=" + ";".join(link_flags) + ";artifact_name=" + artifact_name).encode())
    h.update(_toolchain_tag().encode())
    return h.hexdigest()


def _unit_object_key(unit: Path, gen_dir: Path, headers: list[Path], compile_cmd: list[str]) -> str:
    h = hashlib.sha256()
    src = str(gen_dir)
    for a in compile_cmd[1:]:
        if a == f"-I{src}" or a.startswith(src + "/"):
            continue
        h.update(a.encode() + b"\0")
    h.update((gen_dir / "grid.cuh").read_bytes()); h.update(b"\0")
    h.update(_collision_include_dir().joinpath("grid_collision_geometry.cuh").read_bytes()); h.update(b"\0")
    for p in headers:
        h.update(p.read_bytes()); h.update(b"\0")
    h.update(unit.read_bytes())
    h.update(_toolchain_tag().encode())
    return h.hexdigest()


# ─── the compile half ─────────────────────────────────────────────────────────
def _compile_command(gen_dir: Path, cuda_arch: int, include_dirs: list[Path], defines: list[str],
                     extra_flags: list[str]) -> list[str]:
    nvcc = find_nvcc()
    archs = effective_cuda_archs(cuda_arch)
    gencode_archs = archs if archs is not None else [cuda_arch]
    cmd = [nvcc] + list(_NVCC_DEFAULT_FLAGS) + [f"-gencode=arch=compute_{a},code=sm_{a}" for a in gencode_archs]
    cmd.append(f"-I{gen_dir}")
    cmd.append(f"-I{_collision_include_dir()}")
    for d in _glass_include_dirs():
        cmd.append(f"-I{d}")
    for d in include_dirs:
        cmd.append(f"-I{d}")
    for d in defines:
        cmd.append(f"-D{d}")
    cmd.extend(extra_flags)
    return cmd


def _build(gen_dir: Path, sources: list[Path], headers: list[Path], compile_cmd: list[str], link_flags: list[str],
           out_so: Path, object_cache_dir: Path, reuse_objects: bool) -> dict[str, Any]:
    """Compile each consumer source with -c (from the object cache when its key exists), then
    host-link the objects into `out_so`. Returns a record of what ran."""
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
        key = _unit_object_key(unit, gen_dir, headers, compile_cmd)
        cached = object_cache_dir / f"{unit.stem}-{key}.o"
        if reuse_objects and cached.is_file():
            os.utime(cached)
            with lock, log_path.open("a") as log:
                log.write(f"[{unit.stem}] cached {cached.name}\n")
            record["units"][unit.name] = {"object": cached.name, "compiled": False}
            return cached, 0, 0.0, ""
        obj = build_dir / (unit.stem + ".o")
        ulog = build_dir / (unit.stem + ".log")
        t0 = time.monotonic()
        with ulog.open("w") as f:
            r = subprocess.run(compile_cmd + ["-c", str(unit), "-o", str(obj)], stdout=f, stderr=subprocess.STDOUT)
        secs = time.monotonic() - t0
        with lock, log_path.open("a") as log:
            log.write(f"[{unit.stem}] rc={r.returncode} {secs:.1f}s\n")
        if r.returncode == 0:
            tmp = cached.with_name(cached.name + f".{os.getpid()}.tmp")
            shutil.move(str(obj), str(tmp))
            os.replace(tmp, cached)
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
    link = [nvcc, "--shared", "-o", str(out_so)] + objs + link_flags
    t0 = time.monotonic()
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(link) + "\n")
        log.flush()
        r = subprocess.run(link, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[link] rc={r.returncode} {time.monotonic() - t0:.1f}s\n")
    if r.returncode != 0:
        raise RuntimeError(f"nvcc exit {r.returncode} linking the consumer artifact; build log:\n" + log_path.read_text())
    shutil.rmtree(build_dir, ignore_errors=True)
    record["compiled_units"] = sum(1 for _, _, secs, _ in results if secs > 0)
    return record


# ─── manifest (one per consumer, under a lock) ───────────────────────────────
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
                   emit_plant: bool | None = None,
                   collision: bool = False,
                   collision_res: float | Sequence[float] = 0.05,
                   collision_native: bool = False,
                   spherized_urdf: str | Path | None = None,
                   launch_config_profile: str | None = None) -> ConsumerArtifact | dict[str, Any]:
    """Build (or reuse) the consumer's artifact for one robot and return where it is.

    `consumer` names the namespace in the cache directory (one per consuming project);
    `name` the artifact within it (one per robot and configuration). `sources` are the
    consumer's translation units, each compiled with `-c` against the generated `grid.cuh`
    (plus the static collision header and the GLASS include directories) and linked with
    `--shared` into `artifact_name`; `headers` are the consumer's own header files, folded
    into the identity by content (an include directory is folded by path only, so list the
    headers whose edits must re-key the build); `defines` are `NAME` or `NAME=VALUE`;
    `extra_flags` go to every unit compile, `link_flags` to the link.

    The codegen keywords (`floating_base`, `ee_joint_names`, `dtype`, `algorithm_list`,
    `enable_mujoco_kernels`, `cost_frames`, `contact_frames`, the joint-dynamics and runtime
    knobs) mean what they mean on `register_robot`; `emit_plant` is the generator's plant-block
    switch (None: the generator's rule), and `collision`, `collision_res`, `collision_native`
    and `spherized_urdf` select the generated `grid_collision` namespace (a sphere model is
    validated against the URDF). `launch_config_profile` (default: the bindings' "ffi"
    profile) names the baked launch-configuration profile.

    `plan_only=True` reports what a build would use and whether it is already there, without
    generating or compiling. `force_rebuild=True` recompiles every unit."""
    _check_name(consumer, "consumer")
    _check_name(name, "name")
    _check_name(artifact_name, "artifact_name")
    urdf_p = Path(urdf_path).expanduser().resolve()
    if not urdf_p.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_p}")
    urdf_bytes = urdf_p.read_bytes()
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
        runtime_joint_dynamics=runtime_joint_dynamics, emit_plant=emit_plant, collision=collision,
        collision_res=collision_res, collision_native=collision_native, spherized_urdf=spherized_urdf,
        launch_config_profile=launch_config_profile)
    identity = artifact_identity(cuda_arch)
    input_key = _input_key(urdf_bytes, code_options, srcs, hdrs, incs, defs, xflags, lflags, artifact_name, identity)
    store = root / "store"

    def _entry_state(entry_dir: Path) -> list[str]:
        if not ((entry_dir / artifact_name).exists() and (entry_dir / "meta.json").exists()):
            return ["entry incomplete (artifact or meta.json missing)"]
        return stale_hit_reasons(entry_dir, identity)

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
            "would": "load" if (mapped and stale == []) else "generate+compile (the content key decides whether nvcc runs)",
        }

    if not force_rebuild:
        mapped = keymap_lookup(root, input_key)
        if mapped:
            entry_dir = store / mapped
            stale = _entry_state(entry_dir)
            if not stale:
                meta = json.loads((entry_dir / "meta.json").read_text())
                _manifest_register(root, name, mapped, meta)
                return ConsumerArtifact(consumer, name, entry_dir / artifact_name, entry_dir / "grid.cuh",
                                        entry_dir, mapped, input_key, cuda_arch, meta, False)
            _log.info("grid_rbd: artifact pointer for %s/%s rejected (%s); regenerating", consumer, name, "; ".join(stale))

    store.mkdir(parents=True, exist_ok=True)
    _sweep_dead_staging_dirs(store)
    gen_dir = store / f".gen_{os.getpid()}_{input_key[:8]}"
    shutil.rmtree(gen_dir, ignore_errors=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    compiled = False
    try:
        generate_grid_cuh(urdf_p, code_options, gen_dir / "grid.cuh")
        compile_cmd = _compile_command(gen_dir, cuda_arch, incs, defs, xflags)
        content_key = _content_key(gen_dir, srcs, hdrs, compile_cmd, lflags, artifact_name)
        entry_dir = store / content_key
        if (entry_dir / artifact_name).exists() and (entry_dir / "meta.json").exists() and not force_rebuild:
            meta = json.loads((entry_dir / "meta.json").read_text())
            if read_build_inputs(entry_dir) is None:
                write_build_inputs(entry_dir, identity)
        else:
            _log.info("grid_rbd: building consumer artifact %s/%s (%d unit(s))", consumer, name, len(srcs))
            t0 = time.monotonic()
            record = _build(gen_dir, srcs, hdrs, compile_cmd, lflags, gen_dir / artifact_name,
                            root / "objects", reuse_objects=not force_rebuild)
            compiled = record["compiled_units"] > 0
            meta = {
                "consumer": consumer, "name": name, "artifact": artifact_name,
                "urdf": str(urdf_p), "urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
                "sources": {p.name: _file_digest(p) for p in srcs},
                "headers": {p.name: _file_digest(p) for p in hdrs},
                "include_dirs": [str(d) for d in incs], "defines": defs,
                "extra_flags": xflags, "link_flags": lflags,
                "code_options": code_options, "cuda_arch": int(cuda_arch),
                "grid_rbd": package_version(), "artifact_schema": ARTIFACT_SCHEMA,
                "build_seconds": round(time.monotonic() - t0, 1), "build": record,
            }
            archs = effective_cuda_archs(cuda_arch)
            if archs is not None:
                meta["cuda_archs"] = archs
            (gen_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
            write_build_inputs(gen_dir, identity)
            if force_rebuild and entry_dir.exists():
                shutil.rmtree(entry_dir)
            try:
                os.replace(gen_dir, entry_dir)
            except OSError:
                meta = json.loads((entry_dir / "meta.json").read_text())
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
    keymap_record(root, input_key, content_key)
    _manifest_register(root, name, content_key, meta)
    return ConsumerArtifact(consumer, name, entry_dir / artifact_name, entry_dir / "grid.cuh",
                            entry_dir, content_key, input_key, cuda_arch, meta, compiled)


def load_artifact(consumer: str, name: str, cache_dir: str | Path | None = None,
                  cuda_arch: int | None = None) -> ConsumerArtifact:
    """The prepared artifact `name` of `consumer`, without a toolchain: the manifest names it,
    the entry must be whole and recorded for this device (or one of a fat build's devices) and
    key schema. With no detectable GPU and no `cuda_arch`, the recorded device is trusted."""
    _check_name(consumer, "consumer")
    _check_name(name, "name")
    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    root = consumer_root(cache_root, consumer)
    entry = _load_manifest(root)["artifacts"].get(name) if _manifest_path(root).exists() else None
    if entry is None:
        raise ArtifactNotFoundError(f"no artifact {name!r} for consumer {consumer!r} under {cache_root}")
    entry_dir = root / "store" / entry["content_key"]
    artifact_name = entry.get("artifact", "artifact.so")
    if not (entry_dir / artifact_name).exists() or not (entry_dir / "meta.json").exists():
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} is incomplete at {entry_dir}")
    recorded = read_build_inputs(entry_dir)
    if recorded is None:
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} predates the identity record")
    if cuda_arch is None:
        cuda_arch = detect_cuda_arch()
    reasons = []
    if recorded.get("key_schema") != _KEY_SCHEMA:
        reasons.append(f"key_schema: recorded {recorded.get('key_schema')!r} != current {_KEY_SCHEMA!r}")
    if recorded.get("artifact_schema") != ARTIFACT_SCHEMA:
        reasons.append(f"artifact_schema: recorded {recorded.get('artifact_schema')!r} != current {ARTIFACT_SCHEMA!r}")
    if cuda_arch and not _arch_loadable(recorded, int(cuda_arch)):
        reasons.append(f"cuda_arch: recorded {recorded.get('cuda_arch')!r} (archs {recorded.get('cuda_archs')!r}) != current {cuda_arch!r}")
    if reasons:
        raise ArtifactIncompatibleError(f"artifact {name!r} of {consumer!r} cannot be loaded here: " + "; ".join(reasons))
    meta = json.loads((entry_dir / "meta.json").read_text())
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
