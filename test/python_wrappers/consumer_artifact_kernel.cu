// A consumer translation unit for the artifact build gate (HJCD-IK ask G9): one kernel over
// the generated header, exported through a C ABI the test dlopens with ctypes. It evaluates the
// end-effector pose for one configuration and, when the header carries the collision
// namespace, the collision-free verdict against an empty environment.
#include "grid.cuh"
#include <cuda_runtime.h>

using T = float;

__global__ void consumer_pose_kernel(const T *d_q, const grid::robotModel<T> *d_model, T *d_pose, int *d_free) {
    extern __shared__ T s_dyn[];
    (void)s_dyn;
    __shared__ T s_q[grid::NUM_POS];
    __shared__ T s_pose[6 * grid::NUM_EES];
    for (int i = threadIdx.x; i < grid::NUM_POS; i += blockDim.x) s_q[i] = d_q[i];
    __syncthreads();
    grid::end_effector_pose_device<T>(s_pose, s_q, d_model);
    __syncthreads();
    int is_free = -1;
#ifdef CONSUMER_HAS_COLLISION
    __shared__ T s_sphere_pos[3 * grid_collision::NUM_COLLISION_SPHERES];
    __shared__ T s_sphere_r[grid_collision::NUM_COLLISION_SPHERES];
    grid_collision::Environment<T> env{};
    // environment only (check_self = false, G10): the dense sphere model may touch itself at rest
    is_free = grid_collision::config_free<T>(s_q, d_model, env, s_sphere_pos, s_sphere_r, nullptr,
                                             static_cast<T>(0), false) ? 1 : 0;
#endif
    if (threadIdx.x == 0) {
        for (int i = 0; i < 6 * grid::NUM_EES; ++i) d_pose[i] = s_pose[i];
        *d_free = is_free;
    }
}

extern "C" int consumer_num_pos() { return grid::NUM_POS; }
extern "C" int consumer_num_ees() { return grid::NUM_EES; }
extern "C" int consumer_build_tag() { return CONSUMER_BUILD_TAG; }

// Evaluates the pose (6 per end effector) of `q` on the device; returns 0 on success,
// writes the collision verdict (1 free, 0 hit, -1 when the header has no collision namespace).
extern "C" int consumer_pose(const float *q, float *pose, int *is_free) {
    static const grid::robotModel<T> *d_model = nullptr;
    if (d_model == nullptr) d_model = grid::init_robotModel<T>();
    T *d_q = nullptr, *d_pose = nullptr; int *d_free = nullptr;
    if (cudaMalloc(&d_q, grid::NUM_POS * sizeof(T)) != cudaSuccess) return 1;
    if (cudaMalloc(&d_pose, 6 * grid::NUM_EES * sizeof(T)) != cudaSuccess) return 1;
    if (cudaMalloc(&d_free, sizeof(int)) != cudaSuccess) return 1;
    cudaMemcpy(d_q, q, grid::NUM_POS * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
#ifdef CONSUMER_HAS_COLLISION
    // the batched sphere extractor inside config_free has its own dynamic arena
    if (grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>() > smem) smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
#endif
    cudaFuncSetAttribute(consumer_pose_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    consumer_pose_kernel<<<1, 64, smem>>>(d_q, d_model, d_pose, d_free);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(pose, d_pose, 6 * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(is_free, d_free, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_q); cudaFree(d_pose); cudaFree(d_free);
    return e == cudaSuccess ? 0 : 2;
}
