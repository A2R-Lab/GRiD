# GRiD wrapper-type tour: `_inner` → `_device` → `_kernel` → `_host` → batch

For every algorithm, the GRiD codegen emits the **same five layers** of surface,
each wrapping the one before. Picking the right layer is the main decision when
you write CUDA against `grid.cuh`. This tour uses `inverse_dynamics` (companion to
`inverse_dynamics_kernel_example.cu`); every other algorithm follows the identical
shape with its own macro prefix, which matches the verbose function name
(`FORWARD_DYNAMICS_`, `INVERSE_DYNAMICS_GRADIENT_`, …); a few keep their
established proper names (`MINV_`, `ABA_`, `CRBA_`, …).

## The layers (innermost → outermost)

### 1. `_inner` — fat logic, caller places ALL scratch  (`__device__`)
```cpp
inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd,
    s_XImats, s_topology_helpers, s_temp, d_f_ext, gravity);
```
The actual RNEA. It does **no** memory management: every buffer (`s_vaf`,
`s_XImats`, `s_temp`, topology helpers, linalg scratch) is a pointer **you** pass
in, all in shared memory, and you must have already populated `s_XImats` via
`load_update_XImats_helpers`. It does not even `extern __shared__`.

**Reach for it when:** you are writing your own fused kernel and want to *share*
scratch across several algorithm calls (e.g. compute `s_XImats` once, then run
RNEA *and* CRBA off it), or you have a bespoke shared-memory layout. Maximum
control, maximum responsibility — get a buffer size wrong and you corrupt a
neighbor silently (size per-body buffers by `NUM_JOINTS`/`NUM_BODIES`, not
`NUM_VEL`).

### 2. `_device` — auto-allocates its own smem scratch  (`__device__`)
```cpp
inverse_dynamics_device<T>(s_c, s_q, s_qd, s_qdd, d_robotModel, d_f_ext, gravity);
```
Wraps `_inner`. It declares the `extern __shared__` arena, carves out
`s_vaf`/`s_XImats`/`s_temp`/linalg scratch from it, calls
`load_update_XImats_helpers`, then `_inner`. You supply only inputs/outputs
(shared-memory pointers) + the model. The launch must reserve
`grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()`.

**Reach for it when:** you want to drop GRiD dynamics into a kernel *you* launch
(custom grid/stream/fusion at the launch level) but don't want to hand-manage the
algorithm's scratch. This is the **default choice for writing your own kernel** —
it's exactly Path A in the flagship example.

### 3. `_kernel` — launchable `__global__`
```cpp
inverse_dynamics_kernel<T><<<B, threads, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
    d_c, d_q_qd, stride_q_qd, d_qdd, d_f_ext, d_robotModel, gravity, NUM_TIMESTEPS);
```
A ready-made `__global__` that reads packed device inputs (`d_q_qd` with a
`stride_q_qd`), runs the per-timestep block loop (`for k in NUM_TIMESTEPS` at
block level), and writes packed device outputs. Carries
`__launch_bounds__(tier_max_threads<TIER>())`. Note its arena macro is
`INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES` (slightly larger than the `_device` one — it also
holds `s_q_qd`/`s_c`/`s_vaf` for the load/store).

**Reach for it when:** you want the batched launcher but are managing the device
buffers + memcpy yourself, and don't want the `gridData` bookkeeping.

### 4. `_host` — gridData orchestration + memcpy  (`__host__`)
```cpp
inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/true>(
    hd_data, d_robotModel, gravity, num_timesteps,
    block_dimms, thread_dimms, streams);
```
The top of the stack. Takes a `grid::gridData<T>` (from `init_gridData`) that
bundles all host+device input/output pointers, async-copies `h_*` → `d_*`,
launches `_kernel`, copies results `d_*` → `h_*`. Template flags select the
qdd-input and compressed-memory variants.

**Reach for it when:** you "just want to call one algorithm" from host code with
no CUDA plumbing — closest to what `grid_rbd` does under the hood. See the
`#else` (fixed-base) branch of `test/cuda_equivalents/cuda_equivalence_runner.cu`
for the full `init_grid` / `init_robotModel` / `init_gridData` / `close_grid`
lifecycle around these `_host` calls.

### 5. batch — one block per problem
Not a separate emitted symbol but the **launch convention**: GRiD is
single-block-per-problem (one robot/timestep per CUDA block, never split across
blocks). `_kernel` and `_host` bake this in via their `NUM_TIMESTEPS` block loop;
when you write your own kernel around `_device`, you express it as
`<<<B, threads, smem>>>` + a `for (k = blockIdx.x; k < B; k += gridDim.x)` grid
stride (Path C in the flagship example). The single-block design means batching is
free — more blocks, same per-block code.

## Decision guide

| your situation | use |
|----------------|-----|
| fusing GRiD into a bigger kernel, sharing `s_XImats`/scratch across algos | `_inner` |
| writing your **own** kernel, one algo, don't want to manage scratch | `_device` |
| want the batched `__global__` but own your device buffers + memcpy | `_kernel` |
| just call one algorithm from host, no CUDA plumbing | `_host` |
| many robots/timesteps | launch `B` blocks (built into `_kernel`/`_host`; grid-stride around `_device` in your own kernel) |

## How it maps to the flagship example

`inverse_dynamics_kernel_example.cu` demonstrates the two layers you write by
hand:
- **Path A** calls `inverse_dynamics_device` — layer 2.
- **Path B** calls `inverse_dynamics_inner` + `load_update_XImats_helpers` — layer
  1, showing the caller-owned scratch.
- **Path C** wraps `_device` in a `<<<B, …>>>` grid-stride kernel — the batch
  convention.

Layers 3–4 (`_kernel` / `_host`) are exercised end-to-end by
`test/cuda_equivalents/cuda_equivalence_runner.cu`, which is the place to look for
the full `gridData` lifecycle.
