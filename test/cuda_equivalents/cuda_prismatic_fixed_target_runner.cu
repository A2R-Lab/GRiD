// G13: direct block pose/Jacobian and thread/warp FK for the slider fixture.
#include "grid.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 2; } } while (0)

__global__ void evaluate(const float *q, const grid::robotModel<float> *model, float *out) {
    __shared__ float sq[grid::NUM_POS];
    __shared__ float pose[6 * grid::NUM_EES];
    __shared__ float jac[6 * grid::NUM_EES * grid::NUM_VEL];
    for (int i = threadIdx.x; i < grid::NUM_POS; i += blockDim.x) sq[i] = q[i];
    __syncthreads();
    grid::end_effector_pose_device_tool<float>(pose, sq, model);
    __syncthreads();
    grid::end_effector_pose_gradient_device_tool<float>(jac, sq, model);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 6; ++i) out[i] = pose[i];
        for (int i = 0; i < 6 * grid::NUM_VEL; ++i) out[6 + i] = jac[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    const int threads = atoi(argv[1]);
    const float q = strtof(argv[2], nullptr);
    static_assert(grid::NUM_POS == 1 && grid::NUM_VEL == 1 && grid::NUM_EES == 1);
    grid::robotModel<float> *model = nullptr;
    CHECK(grid::init_robotModel_checked<float>(&model));
    float *dq = nullptr, *out = nullptr;
    CHECK(cudaMalloc(&dq, sizeof(float)));
    CHECK(cudaMalloc(&out, 26 * sizeof(float)));
    CHECK(cudaMemcpy(dq, &q, sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out, 0xff, 26 * sizeof(float)));
    size_t smem = std::max(grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<float>(),
                          grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<float>());
    CHECK(cudaFuncSetAttribute(evaluate, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    evaluate<<<1, threads, smem>>>(dq, model, out);
    CHECK(cudaGetLastError());
    // The last articulated frame is also evaluated through both FK primitives.
    grid::ee_pose_fk_batched_kernel<float, false><<<1, threads>>>(out + 12, dq, 1, model, 1, grid::NUM_BODIES - 1);
    CHECK(cudaGetLastError());
    if (threads >= 32) {
        grid::ee_pose_fk_batched_kernel<float, true><<<1, threads>>>(out + 19, dq, 1, model, 1, grid::NUM_BODIES - 1);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    float host[26];
    CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    for (int i = 0; i < (threads >= 32 ? 26 : 19); ++i) printf("%.9g ", host[i]);
    puts("");
    CHECK(cudaFree(dq));
    CHECK(cudaFree(out));
    CHECK(grid::free_robotModel_checked<float>(model));
    return 0;
}
