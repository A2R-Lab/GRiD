// Restricted cooperative callers of the GLASS bodies embedded by GRiD.
// Each block owns its arena. No timing, aggregate registration, or aliasing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using T = TEST_SCALAR;
constexpr int N = 7, NN = N*N;
constexpr int SCRATCH = 2*NN + 2*N + 4;
constexpr int ARENA = 2*NN + N + SCRATCH + 2;
constexpr int INPUT = 3*NN + N, OUTPUT = N + 2*NN + 3;

__device__ __forceinline__ void operate(T* __restrict__ arena,
                                       const T* __restrict__ in,
                                       T* __restrict__ out) {
    T *a = arena + 1, *b = a + 2*NN, *scratch = b + N;
    for (int op = 0; op < 3; ++op) {
        for (int i = threadIdx.x; i < ARENA; i += blockDim.x) arena[i] = T(NAN);
        if (threadIdx.x == 0) arena[0] = arena[ARENA-1] = T(1234);
        __syncthreads();
        for (int i = threadIdx.x; i < NN; i += blockDim.x) {
            a[i] = in[op*NN+i];
            if (op == 2) a[NN+i] = T(i/N == i%N);
        }
        for (int i = threadIdx.x; i < N; i += blockDim.x) b[i] = in[3*NN+i];
        __syncthreads();
        if (op == 0) grid::glass::posv<T, N>(a, b);
        if (op == 1) grid::glass::eig_clamp<T, N>(a, T(0.125), scratch);
        if (op == 2) grid::glass::inv_pivoted<T, N>(a, scratch);
        __syncthreads();
        const int count = op == 0 ? N : NN;
        const int offset = op == 0 ? 0 : N + (op-1)*NN;
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            out[offset+i] = op == 0 ? b[i] : a[(op == 2 ? NN : 0)+i];
        if (threadIdx.x == 0)
            out[N+2*NN+op] = (arena[0] == T(1234) && arena[ARENA-1] == T(1234)) ? T(1234) : T(NAN);
        __syncthreads();
    }
}

template <bool GLOBAL>
__global__ void run(T* __restrict__ global, const T* __restrict__ in,
                    T* __restrict__ out) {
    __shared__ T shared[ARENA];
    operate(GLOBAL ? global + blockIdx.x*ARENA : shared,
            in + blockIdx.x*INPUT, out + blockIdx.x*OUTPUT);
}

#define CK(call) do { auto e=(call); if(e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while(0)

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const int threads = atoi(argv[1]);
    const bool global = atoi(argv[2]);
    if (threads < 1 || threads > 256) return 2;
    std::vector<T> in(2*INPUT), out(2*OUTPUT);
    for (auto& x : in) { double value; if (scanf("%lf", &value) != 1) return 2; x=T(value); }
    T *di, *dw, *dout;
    CK(cudaMalloc(&di, in.size()*sizeof(T)));
    CK(cudaMalloc(&dw, 2*ARENA*sizeof(T)));
    CK(cudaMalloc(&dout, out.size()*sizeof(T)));
    CK(cudaMemcpy(di, in.data(), in.size()*sizeof(T), cudaMemcpyHostToDevice));
    for (int repeat=0; repeat<3; ++repeat) {
        CK(cudaMemset(dout, 0xff, out.size()*sizeof(T)));
        if (global) run<true><<<2,threads>>>(dw,di,dout);
        else run<false><<<2,threads>>>(dw,di,dout);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dout, out.size()*sizeof(T), cudaMemcpyDeviceToHost));
        for (auto x : out) printf("%.17g ", double(x));
        printf("\n");
    }
    CK(cudaFree(di)); CK(cudaFree(dw)); CK(cudaFree(dout));
}
