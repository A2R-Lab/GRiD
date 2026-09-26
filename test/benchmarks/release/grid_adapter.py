"""Real NumPy, JAX, PyTorch, native C-ABI and CUDA host-call paths sharing one build contract."""
from __future__ import annotations
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np
from .protocol import digest, ROOT, CORE, VECTOR_OPS, Q_ONLY_OPS, Q_QD_OPS

# Operation codes of kernel_bridge.cu (= the collector's OPS order).
KERNEL_OPS = {op: i for i, op in enumerate(("inverse_dynamics", "inverse_dynamics_gradient", "idsva_so", "minv",
    "forward_dynamics", "forward_dynamics_gradient", "fdsva_so", "end_effector_pose",
    "crba", "nonlinear_effects", "generalized_gravity", "ccrba", "coriolis_matrix"))}
assert tuple(KERNEL_OPS)[:3] == CORE


def tree_map(fn, value):
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    return fn(value)


def stats(samples):
    samples = np.asarray(samples, np.float64)
    return {"samples_us": samples.tolist(), "mean_us": float(samples.mean()),
            "median_us": float(np.median(samples)), "min_us": float(samples.min()),
            "max_us": float(samples.max())}


def build_locked(library, compile_command):
    """Content-keyed shared library under the release build cache, built once."""
    library.parent.mkdir(parents=True, exist_ok=True)
    with library.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(f".{os.getpid()}.tmp.so")
            subprocess.run(compile_command(temporary), check=True)
            temporary.replace(library)
    return library


class GridAdapter:
    def __init__(self, backend, operation, fixture, max_batch, build_dir, *, dtype="float32"):
        if dtype not in {"float32", "float64"} or (backend in {"grid_native", "grid_cuda"} and dtype != "float32"):
            raise ValueError("Expected fp32/fp64; the native timing bridges support fp32 only")
        import grid_rbd
        self.backend, self.op, self.fixture = backend, operation, fixture
        self.dtype = dtype
        interface = backend.removeprefix("grid_")
        if interface in {"native", "cuda"}:
            interface = "numpy"
        # Same compile contents and cache key for native/NumPy/JAX/PyTorch/CUDA.
        self.h = grid_rbd.register_robot("release_" + fixture.spec.robot_id + "_" + operation,
            fixture.urdf, floating_base=fixture.base == "floating", backend=interface,
            max_batch_size=max_batch, dtype=dtype,
            algorithm_list=["idsva_so_body_frame" if operation == "idsva_so" else operation],
            ee_joint_names=[fixture.target] if operation.startswith("end_effector_pose") else None,
            enable_mujoco_kernels=False, _profile_overlay=None)
        self.fn = getattr(self.h, operation)
        if interface == "jax":
            import jax
            if jax.default_backend() != "gpu":
                raise RuntimeError("JAX did not select a GPU; refusing a mislabeled GPU result")
            self.fn = jax.jit(self.fn)
        self.metadata = {"backend": backend, "dtype": dtype, "method": "analytical",
            "allocation_policy": "public API allocation; native includes host output allocation",
            "launch_policy": "same build; no profile overlay or new autotune",
            "build_metadata": getattr(self.h, "_base", self.h).meta,
            "library_sha256": digest(self.h._so_path) if hasattr(self.h, "_so_path") else None}
        self.build_dir = Path(build_dir)
        self.max_batch = max_batch
        # Arena fit at this max batch: a slot count below the batch means the
        # kernels grid-stride (correct, but fewer blocks in flight) — recorded
        # so the report can flag such cells.
        try:
            profile = self.h.device_profile
            self.metadata["workspace_slots"] = int(profile.get("workspace_slots", 0))
            self.metadata["arena_bytes"] = int(profile.get("arena_bytes", 0))
        except Exception as error:  # a handle without a default context yet
            self.metadata["workspace_slots_error"] = f"{type(error).__name__}: {error}"
        self.kernel = None
        if backend == "grid_cuda":
            if operation not in KERNEL_OPS:
                raise ValueError("CUDA host-call timing bridge does not cover " + operation)
            self.kernel = self.kernel_library()
            self.kernel_ctx = self.kernel.grid_kernel_create()
            if not self.kernel_ctx:
                raise RuntimeError("kernel bridge could not initialise the artifact's gridData/robotModel")
            if (self.kernel.grid_kernel_num_joints(), self.kernel.grid_kernel_num_vel()) != (fixture.nq, fixture.nv):
                raise RuntimeError("kernel bridge model dimensions differ from the fixture")
            self.metadata["threads_override"] = int(self.h.threads_per_block)
            self.metadata["allocation_policy"] = "generated gridData arena; outputs left in place (with_mem: pinned host buffer)"

    def prepare(self, batch):
        args = tuple(np.ascontiguousarray(a, dtype=self.dtype) for a in self.fixture.args(self.op, batch))
        if self.backend in {"grid_numpy", "grid_native", "grid_cuda"}:
            self.host = lambda: self.fn(*args)
            self.resident = None
            self.sync = lambda result: None
            self.download = lambda result: result
            if self.backend == "grid_cuda":
                self.resident = lambda: self.kernel_run(batch)
        elif self.backend == "grid_jax":
            import jax
            dev = tuple(jax.device_put(a) for a in args)
            jax.block_until_ready(dev)
            self.resident = lambda: self.fn(*dev)
            # Explicit copy and device_get: H2D + computation + complete D2H.
            self.host = lambda: jax.device_get(self.fn(*(jax.device_put(a.copy()) for a in args)))
            self.sync = jax.block_until_ready
            self.download = jax.device_get
        else:
            import torch
            dev = tuple(torch.from_numpy(a).to("cuda") for a in args)
            torch.cuda.synchronize()
            self.resident = lambda: self.fn(*dev)
            self.download = lambda result: tree_map(lambda a: a if isinstance(a, np.ndarray) else a.detach().cpu().numpy(), result)
            self.host = lambda: self.download(self.fn(*(torch.from_numpy(a).to("cuda") for a in args)))
            self.sync = lambda result: torch.cuda.synchronize()
        return args

    def normalize(self, result):
        result = self.download(result)
        if self.op in VECTOR_OPS:
            result = result[..., :self.fixture.nv]
        return result

    # ── native C ABI (host-to-host through the wrapper's extern "C" surface) ──
    def native_library(self):
        source = Path(__file__).with_name("native_bridge.cpp")
        compiler = subprocess.check_output(["g++", "--version"], text=True)
        key = hashlib.sha256((digest(source) + compiler + "-std=c++17 -O2 -shared -fPIC -ldl").encode()).hexdigest()[:20]
        cache = ROOT / "test/benchmarks/results/release-build-cache"
        lib = build_locked(cache / ("native-" + key + ".so"), lambda out: [
            "g++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-ldl", "-o", str(out)])
        self.metadata["native_bridge_sha256"] = digest(lib)
        return ctypes.CDLL(str(lib))

    def native_time(self, batch, warmups, iterations, warm_seconds=0.0):
        if self.op not in {"inverse_dynamics", "inverse_dynamics_gradient"}:
            raise ValueError("Native wrapper reference supports RNEA/grad RNEA only")
        bridge = self.native_library()
        fn = bridge.grid_release_time
        fp = ctypes.POINTER(ctypes.c_float)
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_longlong, fp, fp, fp,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                       ctypes.POINTER(ctypes.c_double), fp]
        fn.restype = ctypes.c_int
        args = self.fixture.args(self.op, batch)
        size = self.fixture.nq if self.op == "inverse_dynamics" else 2*self.fixture.nv**2
        raw = np.empty((batch, size), np.float32)
        samples = np.empty(iterations, np.float64)
        rc = fn(self.h._so_path.encode(), ("grid_rbd_"+self.op).encode(), self.h._runner.ctx_id(),
                *(a.ctypes.data_as(fp) for a in args), batch, raw.size, warmups, iterations, float(warm_seconds),
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), raw.ctypes.data_as(fp))
        if rc:
            raise RuntimeError(f"Native bridge failed with rc={rc}")
        return stats(samples), self.normalize(self.h._shape_out(self.op, raw))

    # ── CUDA host calls (generated grid.cuh entry points, no binding layer) ──
    def kernel_library(self):
        """Compile kernel_bridge.cu against THIS artifact's grid.cuh with the
        wrapper's nvcc flags; content-keyed on the header, the bridge, the flags
        and the compiler so a regenerated artifact never reuses a stale build."""
        from grid_rbd import _compile
        source = Path(__file__).with_name("kernel_bridge.cu")
        store = Path(self.h._so_path).parent
        header = store / "grid.cuh"
        if not header.exists():
            raise FileNotFoundError(f"artifact header missing next to the wrapper: {header}")
        meta = self.h.meta
        arch = int(meta["cuda_arch"])
        glass = ROOT / "external" / "GLASS"
        flags = [f for f in _compile._NVCC_DEFAULT_FLAGS] + [
            f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-DGRID_RBD_ARCH={arch}",
            f"-DGRID_KERNEL_MAX_BATCH={int(meta['max_batch'])}", f"-DGRID_KERNEL_NUM_EES={int(meta.get('num_ees', 1))}",
            f"-I{store}", f"-I{glass}", f"-I{glass / 'src'}",
            *_compile._mjx_signature_flags(header)]
        nvcc = _compile.find_nvcc()
        version = subprocess.check_output([nvcc, "--version"], text=True)
        key = hashlib.sha256((digest(source) + digest(header) + repr(flags) + version).encode()).hexdigest()[:20]
        cache = ROOT / "test/benchmarks/results/release-build-cache"
        lib = build_locked(cache / ("kernel-" + key + ".so"), lambda out: [nvcc, *flags, "-o", str(out), str(source)])
        self.metadata.update(kernel_bridge_sha256=digest(lib), kernel_bridge_flags=flags,
                             artifact_header_sha256=digest(header), nvcc_version=version.strip().splitlines()[-1])
        bridge = ctypes.CDLL(str(lib))
        fp = ctypes.POINTER(ctypes.c_float)
        dp = ctypes.POINTER(ctypes.c_double)
        bridge.grid_kernel_create.restype = ctypes.c_void_p
        bridge.grid_kernel_close.argtypes = [ctypes.c_void_p]
        bridge.grid_kernel_run.argtypes = [ctypes.c_void_p, ctypes.c_int, fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_float, fp]
        bridge.grid_kernel_run.restype = ctypes.c_int
        bridge.grid_kernel_time.argtypes = [ctypes.c_void_p, ctypes.c_int, fp, fp, fp, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_float, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                            dp, dp, fp, fp, ctypes.POINTER(ctypes.c_int)]
        bridge.grid_kernel_time.restype = ctypes.c_int
        if not bridge.grid_kernel_op_built(KERNEL_OPS[self.op]):
            raise RuntimeError(f"{self.op} is not built into the artifact header")
        return bridge

    def _kernel_inputs(self, batch):
        """(q, qd, third) as the bridge expects: q-only operations mirror q into
        the qd slot (as the wrapper's C ABI does) and pass zeros as third;
        (q, qd) operations pass zeros as third; the rest pass their third array."""
        fp = ctypes.POINTER(ctypes.c_float)
        args = tuple(np.ascontiguousarray(a, np.float32) for a in self.fixture.args(self.op, batch))
        if self.op in Q_ONLY_OPS:
            args = (args[0], args[0], np.zeros_like(args[0]))
        elif self.op in Q_QD_OPS:
            args = (args[0], args[1], np.zeros_like(args[1]))
        args = tuple(np.ascontiguousarray(a, np.float32) for a in args)
        self._kernel_keepalive = args
        return args, tuple(a.ctypes.data_as(fp) for a in args)

    def _kernel_output(self, batch):
        return np.empty((batch, self.kernel.grid_kernel_output_size(KERNEL_OPS[self.op])), np.float32)

    def _threads(self):
        return int(self.h.threads_per_block) if int(self.h.threads_per_block) >= 1 else 0

    def kernel_run(self, batch):
        _, pointers = self._kernel_inputs(batch)
        out = self._kernel_output(batch)
        fp = ctypes.POINTER(ctypes.c_float)
        rc = self.kernel.grid_kernel_run(self.kernel_ctx, KERNEL_OPS[self.op], *pointers, batch, self._threads(),
                                         -9.81, out.ctypes.data_as(fp))
        if rc:
            raise RuntimeError(f"kernel bridge run failed with rc={rc}")
        return self.h._shape_out(self.op, out)

    def kernel_time(self, batch, warmups, iterations, warm_seconds=0.0):
        _, pointers = self._kernel_inputs(batch)
        fp = ctypes.POINTER(ctypes.c_float)
        dp = ctypes.POINTER(ctypes.c_double)
        with_mem, compute = np.empty(iterations, np.float64), np.empty(iterations, np.float64)
        out_with_mem, out_compute = self._kernel_output(batch), self._kernel_output(batch)
        threads = ctypes.c_int(0)
        rc = self.kernel.grid_kernel_time(self.kernel_ctx, KERNEL_OPS[self.op], *pointers, batch, self._threads(),
                                          -9.81, float(warm_seconds), warmups, iterations,
                                          with_mem.ctypes.data_as(dp), compute.ctypes.data_as(dp),
                                          out_with_mem.ctypes.data_as(fp), out_compute.ctypes.data_as(fp),
                                          ctypes.byref(threads))
        if rc:
            raise RuntimeError(f"kernel bridge timing failed with rc={rc}")
        self.metadata["threads_per_block"] = threads.value
        return (stats(with_mem), stats(compute),
                self.normalize(self.h._shape_out(self.op, out_with_mem)),
                self.normalize(self.h._shape_out(self.op, out_compute)))

    def close(self):
        if self.kernel is not None and self.kernel_ctx:
            self.kernel.grid_kernel_close(self.kernel_ctx)
            self.kernel_ctx = None
