#pragma once
#include <cuda_runtime_api.h>

// Where a consumer call completes. A mirrored kernel launch has no fence of its
// own. The generated compute-only host function returns after its own fence. Each
// boundary pays one fence after the last device work of the call, never two.
// `launch` and `check` return 0 or the bridge's error code; `copy` returns a CUDA error.
inline int consumer_cuda_rc(cudaError_t e) { return e==cudaSuccess ? 0 : 100+int(e); }

// Native layout: the result stays in the arena, so the launch is the last device work.
template<class Launch, class Check>
inline int consumer_native_call(bool launch_fences, Launch launch, Check check) {
    if(int rc=launch()) return rc;
    if(int rc=check()) return rc;
    return launch_fences ? 0 : consumer_cuda_rc(cudaDeviceSynchronize());
}

// Dense contract: the output copy follows the launch on either path and is fenced.
template<class Launch, class Check, class Copy>
inline int consumer_dense_call(Launch launch, Check check, Copy copy) {
    if(int rc=launch()) return rc;
    if(int rc=check()) return rc;
    if(int rc=consumer_cuda_rc(copy())) return rc;
    return consumer_cuda_rc(cudaDeviceSynchronize());
}
