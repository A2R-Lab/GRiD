# Writing your own CUDA kernel against the generated GRiD header

GRiD's value is the **generated `grid::` CUDA** it emits from a URDF. The
high-level `grid_rbd` Python wrapper hides that CUDA; this directory shows how to
call it yourself — the thing you do when you fuse GRiD dynamics into your own
MPC/RL/controls kernel.

> **Prefer a notebook?** [`../notebooks/07_inline_cuda.ipynb`](../notebooks/07_inline_cuda.ipynb)
> is the tutorial version of this directory: it generates `grid.cuh`, writes a
> small kernel, compiles it with `nvcc` inline, and validates vs `RBDReference`,
> all in one *Run All*. This directory is the fuller, scripted walkthrough.

The flagship walkthrough uses `inverse_dynamics` (the RNEA — the simplest
algorithm) on the fixed-base KUKA **iiwa14** (7 DoF). Everything here compiles and
runs on an RTX 5090 (`sm_120`) and is numerically validated against the
`RBDReference` numpy oracle.

## Files

| file | what it is |
|------|------------|
| `inverse_dynamics_kernel_example.cu` | the flagship — three hand-written kernels (`_device`, `_inner`, batched) |
| `idsva_so_host_example.cu` | a heavy second-order kernel (`idsva_so`) via the `_host` surface |
| `gen_iiwa14_header.py` | the one-line codegen invocation, wrapped in a CLI |
| `validate.py` / `validate_so.py` | diff the examples' output against the `RBDReference` oracle |
| `build_and_validate.sh` | generate → compile → run → validate, both examples, end to end |
| `wrapper_types.md` | the `_inner` → `_device` → `_kernel` → `_host` → batch surface tour |

## Quickstart

```bash
# from the repo root
bash examples/cuda/build_and_validate.sh
```

That prints the RNEA torques from each kernel and a per-block relative error vs
the oracle (`worst rel_err ≈ 2.7e-07` on this box, float32, tol 1e-4).

## Step by step

### 0. Generate the header for your robot (one line of Python)

```python
from robot_descriptions import iiwa14_description
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

robot = URDFParser().parse(iiwa14_description.URDF_PATH, floating_base=False)
GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
    algorithm_list=["inverse_dynamics"], output_path="grid.cuh")
```

`algorithm_list` restricts codegen to the kernels you need (smaller header,
faster `nvcc`). `gen_iiwa14_header.py` is exactly this, behind a `--output` flag.
Run it from the repo root (the editable install provides the imports).

### 1. Include it

```cpp
#include "grid.cuh"   // everything lives in namespace grid::
```

The header **vendors GLASS** (the SIMT linalg backend) inline, so you don't
strictly need `-I GLASS/include` — we pass it anyway for parity with the test
harness. `gpuErrchk` / `gpuErrchkKernel` are defined *inside* `grid.cuh`.

### 2. Reserve dynamic shared memory via the emitted macro

Each algorithm emits a `*_DYNAMIC_SHARED_MEM_BYTES<T>()` macro giving the dynamic
shared-memory arena size its kernels need. **Grep your generated header for the
exact name** — for `inverse_dynamics` the emitted names are:

| macro | used by |
|-------|---------|
| `grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()` | the generated `_kernel` / `_host` launchers |
| `grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()` | the `_device` auto-scratch wrapper |

> Note: the macro prefix matches the algorithm's verbose function name — the
> `inverse_dynamics` kernels really are reserved with
> `INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES`. Most algorithms follow the same
> rule (`FORWARD_DYNAMICS_`, `INVERSE_DYNAMICS_GRADIENT_`,
> `END_EFFECTOR_POSE_`, …); a handful keep their established proper names
> (`MINV_`, `ABA_`, `CRBA_`, `IDSVA_SO_`, …). When in doubt, grep the header you
> generated for `_DYNAMIC_SHARED_MEM_BYTES`.

### 3. Write the kernel — two surfaces, pick one

**`grid::inverse_dynamics_device<T>(...)` (easy).** It declares the
`extern __shared__` arena, carves out `s_vaf`/`s_XImats`/`s_temp`/linalg scratch,
runs `load_update_XImats_helpers`, then the inner. You hand it shared-memory
inputs/outputs only:

```cpp
grid::inverse_dynamics_device<T>(
    s_c, s_q, s_qd, s_qdd, d_robotModel, /*d_f_ext=*/nullptr, gravity);
```

Launch reserves `grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()`.

**`grid::inverse_dynamics_inner<T>(...)` (full control).** No scratch management —
*you* place every buffer (so you can share `s_XImats` with other fused algorithm
calls). You must (1) lay out the scratch, (2) call
`grid::load_update_XImats_helpers<T>(...)`, (3) call the inner:

```cpp
__shared__ T s_vaf[18*grid::NUM_JOINTS];     // v/a/f, 6 each per body
__shared__ T s_XImats[72*grid::NUM_JOINTS];  // 6x6 transform + inertia
__shared__ T s_temp[6*grid::NUM_JOINTS];     // RNEA helper scratch
grid::load_update_XImats_helpers<T>(s_XImats, s_q, /*topo=*/nullptr, d_robotModel, s_temp);
__syncthreads();
grid::inverse_dynamics_inner<T>(
    s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats,
    /*s_topology_helpers=*/nullptr, s_temp, /*d_f_ext=*/nullptr, gravity);
```

`s_topology_helpers` is `nullptr` here because `grid::TOPOLOGY_HELPERS_COUNT == 0`
for iiwa14; robots that need it allocate `TOPOLOGY_HELPERS_COUNT` ints.

See `wrapper_types.md` for when to reach for `_inner` vs `_device` vs `_kernel`
vs `_host`.

### 4. Register the smem and launch — then CHECK errors

```cpp
gpuErrchk(cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
my_kernel<<<1, threads, smem_bytes>>>(...);
gpuErrchk(cudaPeekAtLastError());     // catches launch-config failures
gpuErrchk(cudaDeviceSynchronize());   // catches in-kernel faults
```

You **must** register the opt-in dynamic shared memory before launching, or the
launch fails with `cudaErrorInvalidValue`. And always check `cudaGetLastError`
(`cudaPeekAtLastError`) after the launch — a silent launch failure zeros the
output, which masquerades as a (wrong) answer. This is the most common GRiD-kernel
footgun.

Keep `threads <= grid::MAX_PERF_LEVEL_THREADS`; the kernels carry
`__launch_bounds__` at that cap, and launching wider is an invalid config. One
warp (32) is plenty for a 7-DoF arm.

### 5. Batched — one block per robot/timestep

The GRiD design is **single-block per problem** (never split one robot across
blocks). To process a batch of `B` states, launch `B` blocks; block `k` handles
timestep `k`, with a grid-stride loop so it's robust to `B > gridDim.x`:

```cpp
my_batched_kernel<<<B, threads, smem_bytes>>>(d_cB, d_qB, d_qdB, d_qddB, B, ...);
```

The single-block design scales for free — the batched kernel just wraps the same
`_device` call in a `for (k = blockIdx.x; k < B; k += gridDim.x)` loop.

## Second-order example (the other end of the layering)

`idsva_so_host_example.cu` runs `idsva_so` (analytical second-order inverse
dynamics: `d2tau/dq`, `d2tau/dqd`, `d2tau/dvdq`, `dM/dq`) through the generated
**`_host`** wrapper — the layer you reach for when you just want to *call* a heavy
algorithm and let GRiD own the `gridData`, the host↔device memcpy, the L2-pinned
`d_workspace`, and the launch. It validates block-by-block at `worst rel_err ≈
2.7e-6` (float32, tol 1e-3). Its header uses the fixed-base second-order token:

```python
gen_all_code(algorithm_list=["idsva_so_body_frame"], output_path="grid_so.cuh")
```

See `wrapper_types.md` for why a heavy kernel wants `_host` while your own fused
kernel wants `_inner`/`_device`.

## Adapting to another robot / algorithm

1. Change the URDF in `gen_iiwa14_header.py` (and `floating_base=True` for a
   floating base — then `q` is length `NUM_JOINTS` while `qd`/`qdd`/torques are
   length `NUM_VEL`, and you load via the floating helpers; see
   `test/cuda_equivalents/cuda_equivalence_runner.cu` for the floating pattern).
2. Add your algorithm(s) to `algorithm_list`.
3. Grep the new header for the algorithm's `*_DYNAMIC_SHARED_MEM_BYTES` macro and
   `void <algo>_inner(` / `_device(` signatures — buffer order can differ per
   algorithm. Update the kernel accordingly.
