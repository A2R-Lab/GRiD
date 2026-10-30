// G13: direct block pose/Jacobian and thread/warp FK for the slider fixture.
#include "grid.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#ifdef USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 2; } } while (0)

__global__ void evaluate(const Scalar *q, const grid::robotModel<Scalar> *model, Scalar *out) {
    __shared__ Scalar sq[grid::NUM_POS];
    __shared__ Scalar pose[6 * grid::NUM_EES];
    __shared__ Scalar jac[6 * grid::NUM_EES * grid::NUM_VEL];
    for (int i = threadIdx.x; i < grid::NUM_POS; i += blockDim.x) sq[i] = q[i];
    __syncthreads();
    grid::end_effector_pose_device_tool<Scalar>(pose, sq, model);
    __syncthreads();
    grid::end_effector_pose_gradient_device_tool<Scalar>(jac, sq, model);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 6; ++i) out[i] = pose[i];
        for (int i = 0; i < 6 * grid::NUM_VEL; ++i) out[6 + i] = jac[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    const int threads = atoi(argv[1]);
    const Scalar q = static_cast<Scalar>(strtod(argv[2], nullptr));
    static_assert(grid::NUM_POS == 1 && grid::NUM_VEL == 1 && grid::NUM_EES == 1);
    grid::robotModel<Scalar> *model = nullptr;
    CHECK(grid::init_robotModel_checked<Scalar>(&model));
    Scalar *dq = nullptr, *out = nullptr;
    CHECK(cudaMalloc(&dq, sizeof(Scalar)));
    CHECK(cudaMalloc(&out, 26 * sizeof(Scalar)));
    CHECK(cudaMemcpy(dq, &q, sizeof(Scalar), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out, 0xff, 26 * sizeof(Scalar)));
    size_t smem = std::max(grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<Scalar>(),
                          grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<Scalar>());
    CHECK(cudaFuncSetAttribute(evaluate, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    evaluate<<<1, threads, smem>>>(dq, model, out);
    CHECK(cudaGetLastError());
    // The last articulated frame is also evaluated through both FK primitives.
    grid::ee_pose_fk_batched_kernel<Scalar, false><<<1, threads>>>(out + 12, dq, 1, model, 1, grid::NUM_BODIES - 1);
    CHECK(cudaGetLastError());
    if (threads >= 32) {
        grid::ee_pose_fk_batched_kernel<Scalar, true><<<1, threads>>>(out + 19, dq, 1, model, 1, grid::NUM_BODIES - 1);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    Scalar host[26];
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    for (int i = 0; i < (threads >= 32 ? 26 : 19); ++i) printf("%.17g ", double(host[i]));
    puts("");
    CHECK(cudaFree(dq));
    CHECK(cudaFree(out));
    CHECK(grid::free_robotModel_checked<Scalar>(model));
    return 0;
}
