"""Per-host cache for compiled per-robot .so files (two-stage, content-keyed;
2026-08-19).

Layout:
    ~/.cache/grid-rbd/
    ├── manifest.json              # name -> content-key registry
    ├── bykey/                     # keymap: stage-1 input-key -> stage-2 content-key
    └── store/
        ├── <content_key>/
        │   ├── grid.cuh           # generated header
        │   ├── wrapper.cu         # boilerplate that exposes C ABI
        │   ├── robot.so           # compiled per-robot library
        │   ├── meta.json          # NUM_JOINTS / NUM_VEL / NUM_EES / options
        │   ├── build_inputs.json  # the build identity this entry was built under
        │   └── build.log
        └── ...

Two keys (see the "two-stage content-addressed store" section below):
  - STAGE-1 input key = sha256(urdf_bytes + canonical_json(options) +
    codegen-source-tree hash + grid_rbd_version + the BUILD IDENTITY below).
    The keymap maps it to a content key so the warm path (unchanged inputs)
    never regenerates.
  - STAGE-2 content key = sha256 of exactly what nvcc sees (generated
    grid.cuh + wrapper.cu bytes + compile-flag drivers + the toolchain part of
    the build identity); store/ dirs live under it. A codegen edit whose
    emitted bytes are identical re-runs only the cheap CPU generation half —
    never an nvcc rebuild.
  - BUILD IDENTITY (2026-09-22, audit W05) = every input that shapes the
    artifact but is not a caller option: cuda_arch, nvcc path+version, host
    C++ compiler, the CONTENT of the GLASS headers the generator vendors (not
    the submodule commit — a dirty checkout must re-key), the compile-flag
    module (_compile.py), wrapper_template.cu, torch/jax ABI tags, the
    generation-time env knobs (GRID_CUDA_TARGET_SHARED_MEM_BYTES,
    GRID_CUDA_TARGET_LITE_SHARED_MEM_BYTES, GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES,
    GRID_NO_LICM_BARRIER) and a key-schema number. `build_identity()` returns
    it as a readable dict; each store entry keeps a copy in build_inputs.json
    and a stage-1 hit is honoured ONLY if that record equals the current
    identity (a pointer recorded before the identity existed, or by a
    different toolchain, is a miss — never a silent stale .so).
CUDA arch is part of both keys so a multi-GPU user keeps separate .so files.

The manifest binds a human-friendly `name` to a content key. Re-registering the
same name with a different URDF or options overwrites the binding (the old
.so file lingers in store/ for manual GC; future v2 will add `grid-rbd gc`).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import platformdirs
    _DEFAULT_DIR = Path(platformdirs.user_cache_dir("grid-rbd"))
except ImportError:
    _DEFAULT_DIR = Path.home() / ".cache" / "grid-rbd"


_SCHEMA_VERSION = 1
# Bumped whenever the set of inputs folded into the keys changes: every bykey
# pointer recorded under an older schema becomes unreachable (a miss), the
# content-keyed store is untouched, and identical builds are still reused
# through the content key. Migration = selective rebuild, never a global wipe.
_KEY_SCHEMA = 2
_log = logging.getLogger("grid_rbd")


def default_cache_dir() -> Path:
    """Return the default cache directory; honors $GRID_RBD_CACHE_DIR."""
    override = os.environ.get("GRID_RBD_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DIR


def detect_cuda_arch() -> int:
    """Detect the compute capability of the first installed GPU (as int, e.g. 120 for sm_120).

    Returns 0 if no GPU is detectable; callers should error out then.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return int(major) * 10 + int(minor)
    except Exception:
        return 0


def package_version() -> str:
    """Version of the grid-rbd package; used as a cache-key input so generated
    code from older versions doesn't get reused after upgrade.

    Reads the package's own ``__version__`` (deterministic) rather than
    ``importlib.metadata.version("grid-rbd")``, which is ambiguous when a stale
    root ``GRiD-RBD`` dist also normalizes to ``grid-rbd`` (it would
    nondeterministically return 1.0.0 vs 0.1.0 by sys.path order, silently
    re-keying the compile cache and causing spurious recompiles)."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0-dev"


def _wrapper_template_hash() -> str:
    """sha256 of the bundled wrapper_template.cu so editing the wrapper
    invalidates the cache. The package version alone isn't enough for editable
    dev installs where the version doesn't bump on every edit."""
    path = Path(__file__).parent / "wrapper_template.cu"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def _codegen_source_hash() -> str:
    """sha256 of the codegen source (grid_codegen/ + URDFParser/ *.py) so that
    EDITING THE CODEGEN invalidates the binding cache. The generated grid.cuh is NOT
    in the cache key — only urdf_bytes + options + the resolved launch_config are — so
    without this, changing how GRiDCodeGenerator emits code (tier macros, host wrappers,
    launch_cfg, spill logic, …) would silently reuse a stale .so. Mirrors
    _wrapper_template_hash for the same reason (editable dev installs don't bump the
    package version per edit). Empty for an sdist install where the codegen package
    isn't present (the .so is shipped prebuilt; nothing to re-key against)."""
    # bindings/grid_rbd/_cache.py → repo_root = parent x3
    repo = Path(__file__).resolve().parent.parent.parent
    pkgs = [repo / "grid_codegen", repo / "external" / "URDFParser"]
    h = hashlib.sha256()
    found = False
    for pkg in pkgs:
        if not pkg.is_dir():
            continue
        for py in sorted(pkg.rglob("*.py")):
            try:
                h.update(py.relative_to(repo).as_posix().encode())
                h.update(py.read_bytes())
                found = True
            except OSError:
                continue
    # _compile.py drives the codegen INVOCATION (algorithm_list + the enable_*
    # flags passed to gen_all_code), which materially shapes the generated header
    # even though it lives outside the codegen packages. Without it, editing those
    # flags here silently reuses a stale .so (this is how an enable_idsva_so_world_frame
    # default change went unbuilt). Hash it explicitly; keep the rest of the binding
    # package out so runtime-only file churn doesn't force needless rebuilds.
    compile_py = Path(__file__).resolve().parent / "_compile.py"
    try:
        h.update(b"bindings/grid_rbd/_compile.py")
        h.update(compile_py.read_bytes())
        found = True
    except OSError:
        pass
    return h.hexdigest() if found else ""


def _torch_abi_tag() -> str:
    """Torch version + CXX11-ABI tag, mixed into the cache key so a torch-aware
    .so isn't reused across incompatible torch ABIs. Empty when torch is absent
    (a no-torch build is valid and shouldn't carry a torch tag)."""
    try:
        import torch
        return f"torch={torch.__version__},cxxabi={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    except Exception:
        return ""


def _jax_ffi_tag() -> str:
    """JAX-FFI availability + version tag, mixed into the cache key.

    `compile_so` emits the JAX FFI handler block (and links jax's FFI headers)
    only when jax is importable at compile time. Whether those `grid_rbd_jax_*`
    symbols exist therefore changes with jax's presence/version, so it must be
    part of the key: a .so built with NO jax installed lacks the FFI symbols,
    and a later jax-enabled session must NOT reuse it (the FFI dlsym would fail
    with a confusing 'symbol missing' error). Empty when jax is absent so a
    no-jax build doesn't carry a jax tag (mirrors `_torch_abi_tag`)."""
    try:
        import jax
        from jax import ffi as _jax_ffi  # noqa: F401  (probe FFI availability)
        return f"jaxffi={jax.__version__}"
    except Exception:
        return ""


def canonical_options(options: dict[str, Any]) -> str:
    """Canonical JSON serialization of compile options for hashing.

    Only options that affect generated code are included; cosmetic stuff
    (cache_dir, force_rebuild, etc.) is filtered by the caller.
    """
    return json.dumps(options, sort_keys=True, separators=(",", ":"))


def compute_cache_key(urdf_bytes: bytes, options: dict[str, Any], cuda_arch: int) -> str:
    """Compute the content-addressable cache key for a (urdf, options, arch)
    combination. The package version is mixed in so upgrades invalidate
    correctly."""
    h = hashlib.sha256()
    h.update(urdf_bytes)
    h.update(canonical_options(options).encode())
    h.update(f"arch={cuda_arch}".encode())
    h.update(f"grid_rbd={package_version()}".encode())
    h.update(f"wrapper={_wrapper_template_hash()}".encode())
    # Mix in the codegen source hash so editing GRiDCodeGenerator / URDFParser
    # re-keys the cache (the generated grid.cuh is not itself in the key). Empty
    # string for sdist installs (no codegen package) → no effect there.
    h.update(f"codegen={_codegen_source_hash()}".encode())
    # Mix in the torch ABI tag so a torch-aware build isn't reused under an
    # incompatible torch version (the TORCH_LIBRARY symbols bake in the ABI).
    # Empty string when torch is absent → no effect on no-torch builds.
    h.update(f"{_torch_abi_tag()}".encode())
    # Mix in the JAX-FFI tag so a .so built without jax (no FFI symbols) isn't
    # reused by a later jax-enabled session, and a jax-version bump that changes
    # the FFI ABI re-keys. Empty when jax is absent → no effect on no-jax builds.
    h.update(f"{_jax_ffi_tag()}".encode())
    # Audit W05 (2026-09-22): the toolchain / GLASS-content / env-knob identity.
    # Before this the stage-1 hit could hand back a .so built by another nvcc
    # or against other GLASS headers (the content key knew, but a bykey hit
    # never recomputed it). cuda_arch/torch/jax above are repeated inside the
    # identity dict; hashing them twice is harmless and keeps the dict whole.
    h.update(("identity=" + canonical_options(build_identity(cuda_arch))).encode())
    return h.hexdigest()


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def store_dir(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "store" / cache_key


# ─── two-stage content-addressed store (2026-08-19, user-ratified) ───────────
# compute_cache_key (above) is the STAGE-1 input key: it folds the whole
# grid_codegen/URDFParser source-tree hash, so ANY codegen edit misses — even
# one whose emitted sources are byte-identical. That made every codegen edit
# cost a full nvcc rebuild of every robot (a comment fix = hours of Phase A).
# The store is therefore CONTENT-keyed: the .so lives under a key hashed from
# the exact compile inputs (generated grid.cuh + wrapper.cu bytes + the
# compile-flag drivers), and a keymap file records input-key -> content-key so
# the warm fast path (unchanged inputs) never regenerates. A codegen edit only
# re-runs the cheap CPU generation half; nvcc re-runs only when the generated
# bytes actually change. The torch/jax op namespaces derive from cache_key[:12]
# on BOTH the compile and load sides, so the manifest records the CONTENT key
# and the .so is compiled with that namespace baked in.

_NVCC_VERSION_TAG: str | None = None


def _compile_source_hash() -> str:
    """sha256 of _compile.py — the compile-flag logic is a compile input."""
    try:
        return hashlib.sha256((Path(__file__).parent / "_compile.py").read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def _nvcc_version_tag() -> str:
    """nvcc --version text (cached per process). Closes the long-noted gap:
    the input key never folded the toolchain, so a CUDA upgrade could reuse a
    stale .so; the content key does fold it."""
    global _NVCC_VERSION_TAG
    if _NVCC_VERSION_TAG is None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            _NVCC_VERSION_TAG = "missing"
        else:
            try:
                out = subprocess.run([nvcc, "--version"], capture_output=True,
                                     text=True, timeout=30)
                _NVCC_VERSION_TAG = out.stdout.strip() or f"rc={out.returncode}"
            except Exception:
                _NVCC_VERSION_TAG = "error"
    return _NVCC_VERSION_TAG


def _glass_tag() -> str:
    """GLASS submodule HEAD (editable installs): its headers are -I compile
    inputs of wrapper.cu beyond what grid.cuh vendors."""
    root = Path(__file__).resolve().parents[2]
    glass = root / "external" / "GLASS"
    if not glass.exists():
        return ""
    try:
        return subprocess.check_output(
            ["git", "-C", str(glass), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


_HOST_CXX_TAG: str | None = None
_GLASS_CONTENT_HASH: str | None = None
# Env knobs GRiDCodeGenerator / the emission helpers read at GENERATION time
# (grep os.environ in grid_codegen). GRID_ENABLE_MUJOCO_KERNELS is deliberately
# absent: _compile.generate_sources resolves it to an explicit option (see the
# comment there). GRID_WORKSPACE_* are read by the EMITTED code at runtime.
GENERATION_ENV_KNOBS = (
    "GRID_CUDA_TARGET_SHARED_MEM_BYTES",
    "GRID_CUDA_TARGET_LITE_SHARED_MEM_BYTES",
    "GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES",
    "GRID_NO_LICM_BARRIER",
)
BUILD_INPUTS_FILE = "build_inputs.json"


def _host_cxx_tag() -> str:
    """Host compiler nvcc drives (-ccbin default = c++ on PATH): path + the
    first line of --version. Cached per process."""
    global _HOST_CXX_TAG
    if _HOST_CXX_TAG is None:
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if cxx is None:
            _HOST_CXX_TAG = "missing"
        else:
            try:
                out = subprocess.run([cxx, "--version"], capture_output=True,
                                     text=True, timeout=30)
                first = (out.stdout.strip().splitlines() or [f"rc={out.returncode}"])[0]
                _HOST_CXX_TAG = f"{cxx}:{first}"
            except Exception:
                _HOST_CXX_TAG = f"{cxx}:error"
    return _HOST_CXX_TAG


def _nvcc_identity() -> str:
    """Resolved nvcc path + its --version text: two toolkits on one box (or a
    PATH change) must not share artifacts."""
    return f"{shutil.which('nvcc') or 'missing'}:{_nvcc_version_tag()}"


def _glass_root() -> Path | None:
    root = Path(__file__).resolve().parents[2] / "external" / "GLASS"
    return root if root.exists() else None


def _glass_content_hash() -> str:
    """sha256 over the CONTENT of every GLASS header the generator can vendor
    (relative path + bytes; the top-level *.cuh and src/**; bench/docs/examples
    excluded). A dirty submodule checkout at the same commit re-keys — the
    commit label alone (see _glass_tag) cannot see that. Cached per process
    (~90 small files). Empty for an sdist install without the submodule."""
    global _GLASS_CONTENT_HASH
    if _GLASS_CONTENT_HASH is None:
        root = _glass_root()
        if root is None:
            _GLASS_CONTENT_HASH = ""
        else:
            h = hashlib.sha256()
            files = sorted(root.glob("*.cuh")) + sorted(
                p for p in (root / "src").rglob("*")
                if p.is_file() and p.suffix in (".cuh", ".h", ".hpp", ".cu", ".inl"))
            for f in files:
                try:
                    h.update(f.relative_to(root).as_posix().encode())
                    h.update(f.read_bytes())
                except OSError:
                    continue
            _GLASS_CONTENT_HASH = h.hexdigest()
    return _GLASS_CONTENT_HASH


def _generation_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in GENERATION_ENV_KNOBS}


def build_identity(cuda_arch: int) -> dict[str, Any]:
    """Every artifact-shaping input that is NOT a caller option, as a readable
    dict (persisted beside each store entry as build_inputs.json). Folded into
    the stage-1 key; its toolchain subset is folded into the content key."""
    return {
        "key_schema": _KEY_SCHEMA,
        "cuda_arch": int(cuda_arch),
        "nvcc": _nvcc_identity(),
        "host_cxx": _host_cxx_tag(),
        "glass_content": _glass_content_hash(),
        "glass_commit": _glass_tag(),  # label only; content decides
        "compile_py": _compile_source_hash(),
        "wrapper_template": _wrapper_template_hash(),
        "torch_abi": _torch_abi_tag(),
        "jax_ffi": _jax_ffi_tag(),
        "generation_env": _generation_env(),
    }


# Identity fields that matter for a stage-1 hit. `glass_commit` is a label
# (content decides) and generation_env is already reflected in the emitted
# bytes of the pointed-to entry — but both live inside the stage-1 KEY, so an
# honoured pointer implies they matched when it was recorded. The sidecar
# check is the belt to that suspender: it catches pointers recorded before
# the identity existed (no sidecar) or by a process whose identity did not
# include everything (older key schema).
def write_build_inputs(entry_dir: Path, identity: dict[str, Any]) -> None:
    tmp = entry_dir / f".{BUILD_INPUTS_FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(identity, indent=2, sort_keys=True))
    tmp.replace(entry_dir / BUILD_INPUTS_FILE)


def read_build_inputs(entry_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((entry_dir / BUILD_INPUTS_FILE).read_text())
    except (OSError, ValueError):
        return None


def stale_hit_reasons(entry_dir: Path, identity: dict[str, Any]) -> list[str]:
    """Why a stage-1 pointer to `entry_dir` must NOT be honoured: [] = sound.
    Missing sidecar = the entry predates the identity record (treated as a
    miss; the content key re-hits the same .so if it truly is identical)."""
    recorded = read_build_inputs(entry_dir)
    if recorded is None:
        return ["no build_inputs.json (entry predates the build-identity record)"]
    reasons = []
    for k in sorted(set(identity) | set(recorded)):
        if k == "glass_commit":
            continue
        if recorded.get(k) != identity.get(k):
            reasons.append(f"{k}: recorded {recorded.get(k)!r} != current {identity.get(k)!r}")
    return reasons


def compute_content_key(source_dir: Path, options: dict[str, Any],
                        cuda_arch: int, max_batch: int) -> str:
    """Content-addressed key for a generated-source dir: exactly the inputs
    nvcc sees (source bytes + flag drivers + toolchain + ABI tags)."""
    h = hashlib.sha256()
    h.update((source_dir / "grid.cuh").read_bytes())
    h.update((source_dir / "wrapper.cu").read_bytes())
    h.update(f"arch={cuda_arch};max_batch={max_batch}".encode())
    # Options that drive -D flags / template selection in compile_so (the rest
    # of `options` only shapes the generated bytes, already hashed above).
    for k in ("dtype", "runtime_inertia", "runtime_transform",
              "runtime_joint_dynamics"):
        h.update(f"{k}={options.get(k)!r};".encode())
    h.update(f"compile={_compile_source_hash()}".encode())
    h.update(f"{_torch_abi_tag()}".encode())
    h.update(f"{_jax_ffi_tag()}".encode())
    h.update(f"nvcc={_nvcc_identity()}".encode())
    h.update(f"host_cxx={_host_cxx_tag()}".encode())
    # Content, not commit (W05): the vendored GLASS bytes are inside grid.cuh
    # already, so this mostly guards the -I include path — but a dirty
    # submodule must never share a key with the clean commit it sits on.
    h.update(f"glass={_glass_content_hash()}".encode())
    h.update(f"key_schema={_KEY_SCHEMA}".encode())
    return h.hexdigest()


def keymap_lookup(cache_dir: Path, input_key: str) -> str | None:
    """input-key -> content-key pointer, or None if unrecorded."""
    path = cache_dir / "bykey" / input_key
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def keymap_record(cache_dir: Path, input_key: str, content_key: str) -> None:
    bykey = cache_dir / "bykey"
    bykey.mkdir(parents=True, exist_ok=True)
    tmp = bykey / f".{input_key}.{os.getpid()}.tmp"
    tmp.write_text(content_key + "\n")
    tmp.replace(bykey / input_key)  # atomic; last-writer-wins with same value


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    """Load the manifest JSON, returning an empty schema-conformant dict if
    none exists yet. Caller does not need to handle FileNotFoundError."""
    path = manifest_path(cache_dir)
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "robots": {}}
    with path.open() as f:
        data = json.load(f)
    if data.get("schema_version") != _SCHEMA_VERSION:
        # Future: handle migrations. For now, refuse to use older manifests.
        raise RuntimeError(
            f"manifest at {path} has schema_version "
            f"{data.get('schema_version')!r}, expected {_SCHEMA_VERSION}. "
            "Delete the cache to start fresh."
        )
    return data


def save_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(cache_dir)
    # pid-unique tmp: parallel writers (the split driver's RAM-aware compile
    # pool warms robots concurrently) must never interleave in one tmp file —
    # each rename then publishes a complete manifest (last-writer-wins).
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)  # atomic on POSIX


def manifest_register(
    cache_dir: Path,
    name: str,
    cache_key: str,
    meta: dict[str, Any],
) -> None:
    """Bind `name` to `cache_key` in the manifest, overwriting any prior
    binding under the same name.

    Audit W06 (2026-09-22): the read-modify-write is serialized by an
    advisory lock (manifest.lock beside the file) so concurrent registrations
    (the split driver's compile pool warms robots in parallel) cannot lose
    each other's bindings — before, two writers loaded the same snapshot and
    the second rename dropped the first's robot. save_manifest's tmp+rename
    still makes each publish atomic for readers, which take no lock."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "manifest.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = load_manifest(cache_dir)
            manifest["robots"][name] = {
                "cache_key": cache_key,
                **meta,
            }
            save_manifest(cache_dir, manifest)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def manifest_lookup(cache_dir: Path, name: str) -> dict[str, Any] | None:
    """Return manifest entry for `name`, or None if not registered."""
    manifest = load_manifest(cache_dir)
    return manifest.get("robots", {}).get(name)


def list_registered(cache_dir: Path) -> list[dict[str, Any]]:
    """List all registered robots with their metadata."""
    manifest = load_manifest(cache_dir)
    return [
        {"name": name, **entry}
        for name, entry in manifest.get("robots", {}).items()
    ]
