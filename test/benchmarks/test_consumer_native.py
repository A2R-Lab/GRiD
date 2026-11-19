"""Execute the real native boundary loop against a CPU-only CUDA test double."""
import ctypes as C
from pathlib import Path
import subprocess
import numpy as np
import pytest

SOURCE = Path(__file__).parent/'release/consumer_bridge.cpp'
FP = C.POINTER(C.c_float)
CALL = C.CFUNCTYPE(C.c_int, C.c_void_p, FP, FP, C.c_int, C.c_bool)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('consumer-native')
    (root/'cuda_runtime_api.h').write_text(r'''
#include <cstdlib>
#include <cstring>
#include <string>
using cudaError_t=int;
const int cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
static std::string events; static int fail=0,calls=0;
static int event(char c) { events+=c; return ++calls==fail?7:0; }
template<class T> int cudaMallocHost(T **p,size_t n) {int e=event('P'); if(!e)*p=(T*)malloc(n);return e;}
template<class T> int cudaMalloc(T **p,size_t n) {int e=event('A'); if(!e)*p=(T*)malloc(n);return e;}
static int cudaFree(void *p) {free(p);return event('F');}
static int cudaFreeHost(void *p) {free(p);return event('F');}
static int cudaMemcpy(void *d,const void *s,size_t n,cudaMemcpyKind kind) {
 int e=event(kind==cudaMemcpyHostToDevice?'H':kind==cudaMemcpyDeviceToHost?'D':'C');
 if(!e)memcpy(d,s,n); return e;
}
static int cudaDeviceSynchronize() {return event('S');}
extern "C" const char *trace() {return events.c_str();}
extern "C" void reset(int f) {events.clear();calls=0;fail=f;}
extern "C" void evaluate_mark() {events+='E';}
''')
    library = root/'test.so'
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-I'+str(root), str(SOURCE), '-o', str(library)], check=True)
    lib = C.CDLL(str(library))
    lib.consumer_create.argtypes = [C.POINTER(C.c_void_p), CALL, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, FP]
    lib.consumer_once.argtypes = [C.c_void_p, C.c_int, FP]
    lib.consumer_measure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_double, C.c_int, C.POINTER(C.c_double), FP]
    lib.consumer_close.argtypes = [C.c_void_p]
    lib.trace.restype = C.c_char_p
    return lib


def session(native, gpu=False):
    @CALL
    def evaluate(ctx, inputs, output, batch, device):
        native.evaluate_mark()
        for b in range(batch): output[b] = inputs[2*b]+inputs[2*b+1]
        return 0
    data = np.arange(8, dtype=np.float32)
    ptr = C.c_void_p()
    native.reset(0)
    assert native.consumer_create(C.byref(ptr), evaluate, 1, gpu, 4, 2, 1, data.ctypes.data_as(FP)) == 0
    native.reset(0)
    return ptr, evaluate


@pytest.mark.parametrize('gpu,on_gpu,trace', [(False,0,'E'), (False,1,'DSEHSD'), (True,0,'E'), (True,1,'ED')])
def test_native_order_and_no_allocation_inside_call(native, gpu, on_gpu, trace):
    ptr, callback = session(native, gpu)
    try:
        out = np.zeros(4,np.float32)
        assert native.consumer_once(ptr,on_gpu,out.ctypes.data_as(FP)) == 0
        np.testing.assert_array_equal(out,[1,5,9,13])
        assert native.trace().decode() == trace
    finally:
        native.reset(0); assert native.consumer_close(ptr) == 0


@pytest.mark.parametrize('fail', [1,2,3,4,5])
def test_every_transfer_and_completion_error_is_returned(native, fail):
    ptr, callback = session(native)
    native.reset(fail)
    try:
        out = np.zeros(4,np.float32)
        assert native.consumer_once(ptr,1,out.ctypes.data_as(FP)) == 107
    finally:
        native.reset(0); assert native.consumer_close(ptr) == 0


def test_native_loop_keeps_last_timed_result_and_checks_each_iteration(native):
    ptr, callback = session(native)
    try:
        out, samples = np.zeros(4,np.float32), np.zeros(3,np.float64)
        assert native.consumer_measure(ptr,1,1,0.,3,samples.ctypes.data_as(C.POINTER(C.c_double)),out.ctypes.data_as(FP)) == 0
        assert native.trace().decode() == 'DSEHS'*4+'D'
        np.testing.assert_array_equal(out,[1,5,9,13])
        # Fail during the second sample, not just the final download.
        native.reset(10)
        assert native.consumer_measure(ptr,1,1,0.,3,samples.ctypes.data_as(C.POINTER(C.c_double)),out.ctypes.data_as(FP)) == 107
    finally:
        native.reset(0); assert native.consumer_close(ptr) == 0
