"""Native adapters for the bounded two-consumer experiment (no Python in samples)."""
import ctypes as C
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from .consumer_contract import ROBOTS, OPERATIONS, build_identity
from .grid_adapter import build_locked
from .protocol import ROOT, digest


def fixture_digest(fixture):
    return hashlib.sha256(b''.join(x.tobytes() for x in (fixture.q, fixture.v, fixture.a, fixture.u))).hexdigest()


def dependencies(compiler, source, flags):
    """Compiler-resolved includes, including system/nested/conditional headers."""
    import shlex
    output = subprocess.check_output([compiler, *flags, '-M', '-MT', 'consumer', str(source)], text=True)
    return [Path(p) for p in shlex.split(output.replace('\\\n', ' ').split(':', 1)[1])]


class ConsumerAdapter:
    def __init__(self, backend, operation, fixture, batch, directory, cpu_threads=1):
        if fixture.spec.robot_id not in ROBOTS or operation not in OPERATIONS:
            raise ValueError('outside first consumer slice')
        self.backend = backend
        self.session = self.context = None
        self.adapter = None
        self.output_width = fixture.nv if operation == 'inverse_dynamics' else 6
        self.batch = batch
        self.fp, self.dp = C.POINTER(C.c_float), C.POINTER(C.c_double)
        self.metadata = {}
        try:
            if backend == 'grid_cuda':
                from .grid_adapter import GridAdapter, KERNEL_OPS
                self.adapter = GridAdapter(backend, operation, fixture, batch, directory)
                self.adapter.prepare(batch)
                self.library = self.adapter.kernel
                self.library.grid_consumer_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
                self.library.grid_consumer_create.restype = C.c_void_p
                self.library.grid_consumer_close.argtypes = [C.c_void_p]
                self.context = self.library.grid_consumer_create(self.adapter.kernel_ctx, KERNEL_OPS[operation], self.adapter._threads())
                self.evaluate = self.library.grid_consumer_eval
                self.close_context = self.library.grid_consumer_close
            elif backend == 'pinocchio_plain':
                from .pin_adapter import PinAdapter
                self.adapter = PinAdapter(operation, fixture, directory, cpu_threads=cpu_threads, plain=True)
                self.library = self.adapter.lib
                self.library.pin_consumer_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
                self.library.pin_consumer_create.restype = C.c_void_p
                self.library.pin_consumer_close.argtypes = [C.c_void_p]
                self.context = self.library.pin_consumer_create(self.adapter.pool, batch, cpu_threads)
                self.evaluate = self.library.pin_consumer_eval
                self.close_context = self.library.pin_consumer_close
            else:
                raise ValueError('first slice supports grid_cuda and pinocchio_plain')
            if not self.context:
                raise RuntimeError('native consumer context creation failed')
            self.evaluate.argtypes = [C.c_void_p, self.fp, self.fp, C.c_int, C.c_bool]
            self.evaluate.restype = C.c_int
            source = Path(__file__).with_name('consumer_bridge.cpp')
            compiler = 'g++'
            flags = ['-std=c++17', '-O2', '-fPIC', '-shared', '-I/usr/local/cuda/include',
                     '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart']
            version = subprocess.check_output([compiler, '--version'], text=True)
            paths = dependencies(compiler, source, flags)
            key = build_identity(paths, flags, version, fixture_digest(fixture))
            library = build_locked(ROOT/'test/benchmarks/results/release-build-cache'/f'consumer-{key}.so',
                                   lambda out: [compiler, str(source), *flags, '-o', str(out)])
            self.driver = C.CDLL(str(library))
            self.driver.consumer_create.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p,
                                                     C.c_int, C.c_int, C.c_int, C.c_int, self.fp]
            self.driver.consumer_create.restype = C.c_int
            self.driver.consumer_once.argtypes = [C.c_void_p, C.c_int, self.fp]
            self.driver.consumer_measure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_double, C.c_int, self.dp, self.fp]
            self.driver.consumer_close.argtypes = [C.c_void_p]
            for name in ('consumer_once', 'consumer_measure', 'consumer_close'):
                getattr(self.driver, name).restype = C.c_int
            inputs = np.ascontiguousarray(np.concatenate(fixture.args(operation, batch), axis=1), dtype=np.float32)
            session = C.c_void_p()
            self._check(self.driver.consumer_create(C.byref(session), C.cast(self.evaluate, C.c_void_p), self.context,
                        backend == 'grid_cuda', batch, inputs.shape[1], self.output_width, inputs.ctypes.data_as(self.fp)))
            self.session = session
            self.metadata = {**self.adapter.metadata, 'consumer_bridge_sha256': digest(library),
                             'consumer_build_identity': key, 'consumer_fixture_sha256': fixture_digest(fixture),
                             'cpu_threads': cpu_threads if backend != 'grid_cuda' else None,
                             'boundary_io_allocation': 'persistent pinned host and device buffers',
                             'backend_internal_allocations': 'existing algorithm and pool behavior included',
                             'contract_precision': 'fp32 I/O; Pinocchio ID fp32, pose fp64; GRiD fp32'}
            if backend == 'pinocchio_plain':
                self.metadata.update(thread_policy='fixed ceiling capped by batch; no tuning',
                                     active_cpu_threads=min(cpu_threads, batch),
                                     output_storage_dtype='float32',
                                     allocation_policy='persistent consumer I/O and conversion buffers; existing pool internals included')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(rc):
        if rc:
            raise RuntimeError(f'native consumer boundary failed: {rc}')

    def once(self, boundary):
        out = np.empty((self.batch, self.output_width), np.float32)
        self._check(self.driver.consumer_once(self.session, int(boundary == 'gpu'), out.ctypes.data_as(self.fp)))
        return out

    def measure(self, boundary, warmups, iterations, warm_seconds):
        out = np.empty((self.batch, self.output_width), np.float32)
        samples = np.empty(iterations, np.float64)
        self._check(self.driver.consumer_measure(self.session, int(boundary == 'gpu'), warmups, warm_seconds,
                                                 iterations, samples.ctypes.data_as(self.dp), out.ctypes.data_as(self.fp)))
        return samples.tolist(), out

    def close(self):
        rc = 0
        if self.session:
            rc = self.driver.consumer_close(self.session)
            self.session = None
        if self.context:
            self.close_context(self.context)
            self.context = None
        if self.adapter:
            self.adapter.close()
            self.adapter = None
        self._check(rc)
