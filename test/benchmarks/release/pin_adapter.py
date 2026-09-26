"""Pinocchio codegen/analytical CPU adapter with materialized matched outputs.

Threading: one persistent C++ pool (release_pool.h) with an independent
model/data/codegen context per thread. Every cell is timed at each candidate
thread count — 1, the B//16 heuristic and the ceiling — and the BEST run mean
is the reported full-call time; every variant is kept in the capture. The
first collector split the batch from a Python executor, which cost 100-900 us
per call and made Pinocchio look 20x slower between B=16 and B=32.
"""
import ctypes
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import numpy as np
from .protocol import digest, ROOT, timed

OPS = ("inverse_dynamics", "inverse_dynamics_gradient", "idsva_so", "minv",
       "forward_dynamics", "forward_dynamics_gradient", "fdsva_so", "end_effector_pose")
CACHE = ROOT / "test/benchmarks/results/release-build-cache"


def _locked_build(library, command):
    with library.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(f".{os.getpid()}.tmp.so")
            subprocess.run(command(temporary), check=True)
            temporary.replace(library)


def cache_keys():
    """(codegen_key, bridge_key, dependencies): the JIT libraries depend on the
    code generators and their settings only; the bridge key adds the dispatch
    code around them so a threading change never invalidates G1 codegen."""
    from test.benchmarks.baselines.pinocchio.run import pinocchio_cflags, pinocchio_libs
    here = Path(__file__).parent
    helper = here.parent / "baselines/pinocchio/timePinocchio.cpp"
    codegen_sources = (helper, here / "pin_codegen_init.h",
        *sorted((here.parent / "baselines/util").rglob("*.h*")),
        here.parent / "baselines/pinocchio/ReusableThreads/ReusableThreads.h")
    bridge_sources = (here / "pin_bridge.cpp", here / "release_pool.h")
    flags = pinocchio_cflags() + pinocchio_libs()
    toolchain = subprocess.check_output(["g++", "--version"], text=True)
    pin_version = importlib.metadata.version("pin")
    codegen = {str(p): digest(p) for p in codegen_sources}
    bridge = {**codegen, **{str(p): digest(p) for p in bridge_sources}}
    codegen_key = hashlib.sha256((repr(codegen)+repr(flags)+toolchain+pin_version).encode()).hexdigest()[:20]
    bridge_key = hashlib.sha256((repr(bridge)+repr(flags)+toolchain+pin_version).encode()).hexdigest()[:20]
    return codegen_key, bridge_key, bridge, flags, pin_version


def jit_key(codegen_key, urdf, base):
    return hashlib.sha256((codegen_key+digest(urdf)+base).encode()).hexdigest()[:20]


class PinAdapter:
    def __init__(self, operation, fixture, directory, cpu_threads=1):
        from test.benchmarks.baselines.pinocchio.run import has_cppadcg
        if operation not in OPS:
            raise NotImplementedError("Matched analytical pose-coordinate derivative baseline not implemented")
        if not has_cppadcg():
            raise RuntimeError("Pinocchio CppADCodeGen unavailable; refusing to relabel a direct baseline as codegen")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        self.op, self.f = operation, fixture
        source = Path(__file__).with_name("pin_bridge.cpp")
        codegen_key, key, dependencies, flags, pin_version = cache_keys()
        CACHE.mkdir(parents=True, exist_ok=True)
        library = CACHE / f"pin-{key}.so"
        _locked_build(library, lambda out: ["g++", "-std=c++17", "-O2", "-DNDEBUG", "-DHAVE_CPPADCG", "-fPIC", "-shared",
            "-pthread", str(source), "-o", str(out), *flags])
        self.lib = ctypes.CDLL(str(library))
        self.lib.pin_release_error.restype = ctypes.c_char_p
        self.lib.pin_release_pool_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.pin_release_pool_create.restype = ctypes.c_void_p
        self.lib.pin_release_pool_threads.argtypes = [ctypes.c_void_p]
        self.lib.pin_release_pool_close.argtypes = [ctypes.c_void_p]
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.pin_release_pool_eval.argtypes = [ctypes.c_void_p, fp, fp, fp, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.lib.pin_release_pool_eval.restype = ctypes.c_int
        target = fixture.oracle.model.frames[fixture.oracle._resolve_frame_id(fixture.target)].name
        previous = Path.cwd()
        # Pinocchio gives RNEA, its derivatives, and Minv distinct library
        # filenames; FD compositions can safely reuse those same-model libs.
        jit_dir = CACHE / f"jit-{jit_key(codegen_key, fixture.urdf, fixture.base)}"
        jit_dir.mkdir(exist_ok=True)
        self.pool = None
        try:
            with (CACHE / (jit_dir.name + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                os.chdir(jit_dir)
                self.pool = self.lib.pin_release_pool_create(fixture.urdf.encode(), fixture.base == "floating",
                                                             OPS.index(operation), target.encode(), cpu_threads)
                if not self.pool:
                    raise RuntimeError(self.lib.pin_release_error().decode())
        finally:
            os.chdir(previous)
        self.ceiling = cpu_threads
        self.active = 1
        index = OPS.index(operation)
        self.metadata = {"backend": "pinocchio", "cpu_thread_ceiling": cpu_threads,
            "hardware_threads": os.cpu_count(),
            "thread_policy": "best run mean over candidate thread counts {1, B//16, ceiling}; persistent C++ pool, "
                             "slice 0 on the caller; independent native contexts; every variant recorded",
            "method": "codegen Minv/RNEA composition" if index in {4,5} else "codegen" if index in {0,1,3} else "analytical direct/composed",
            "dtype": "float32" if index in {0,1,3,4,5} else "float64",
            "output_storage_dtype": "float64", "library_sha256": digest(library),
            "codegen_compiler_options": "-O3 (no fast-math)",
            "codegen_constant_digits": 9,
            "bridge_dependencies": dependencies, "pin_version": pin_version,
            "codegen_cache_key": codegen_key,
            "jit_libraries": {p.name: digest(p) for p in sorted(jit_dir.glob("*.so"))},
            "allocation_policy": "persistent model/JIT/scratch; materialized host output per call",
            "note": "Warm full-call wall time includes batch submission; not a claim of optimal CPU threading."}

    def candidates(self, batch):
        return sorted({n for n in (1, max(1, batch // 16), self.ceiling) if 1 <= n <= min(self.ceiling, batch)})

    def prepare(self, batch):
        f = self.f
        # Map project coordinates into the oracle's Pinocchio model layout.
        q = np.ascontiguousarray([f.oracle._to_pin_q(x.astype(np.float64)) for x in f.q[:batch]], dtype=np.float32)
        v = np.ascontiguousarray([f.oracle._expand_project_v_to_pin(x.astype(np.float64)) for x in f.v[:batch]], dtype=np.float32)
        third = f.u if self.op in {"forward_dynamics", "forward_dynamics_gradient", "fdsva_so"} else f.a
        t = np.ascontiguousarray([f.oracle._expand_project_v_to_pin(x.astype(np.float64)) for x in third[:batch]], dtype=np.float32)
        n = f.nv
        size = 4*n**3 if self.op in {"idsva_so", "fdsva_so"} else 2*n*n if self.op.endswith("_gradient") else n*n if self.op == "minv" else 6 if self.op == "end_effector_pose" else n
        fp = ctypes.POINTER(ctypes.c_float)
        pointers = tuple(a.ctypes.data_as(fp) for a in (q, v, t))
        self.active = 1
        self.metadata["active_cpu_threads"] = self.active
        def call():
            out = np.empty((batch, size), np.float64)
            rc = self.lib.pin_release_pool_eval(self.pool, *pointers, batch,
                                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.active)
            if rc:
                raise RuntimeError(self.lib.pin_release_error().decode())
            if self.op in {"idsva_so", "fdsva_so"}:
                return tuple(out[:, k*n**3:(k+1)*n**3].reshape(batch,n,n,n) for k in range(4))
            if self.op.endswith("_gradient"):
                return out.reshape(batch,n,2*n)
            if self.op == "minv":
                return out.reshape(batch,n,n)
            return out
        self.host, self.resident = call, None
        self.sync, self.download = lambda x: None, lambda x: x
        self.batch = batch

    def time_host(self, warmups, iterations, warm_seconds=0.0):
        """Time every candidate thread count; report the best run mean and keep
        the rest. The post-timing checks then run at the selected count."""
        variants = {}
        for n in self.candidates(self.batch):
            self.active = n
            variants[str(n)] = timed(self.host, self.sync, warmups, iterations, warm_seconds)
        best = min(variants, key=lambda k: variants[k]["mean_us"])
        self.active = int(best)
        self.metadata["active_cpu_threads"] = self.active
        return variants[best], {"thread_variants": variants, "selected_threads": self.active}

    def normalize(self, out):
        return out

    def close(self):
        if self.pool:
            self.lib.pin_release_pool_close(self.pool)
            self.pool = None
