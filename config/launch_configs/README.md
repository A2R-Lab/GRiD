# GRiD launch configs — crowdsourced per-(robot, GPU) autotuned launch parameters

GRiD kernels are single-block and **thread-count-invariant** (same result at any block size), so the optimal
`(resource_tier, threads_per_block)` for each algorithm is a pure *performance* choice that depends on the
**robot** (DoF/topology) and the **GPU**. This directory holds measured-optimal launch configs so GRiD defaults
to fast launches out of the box — instead of the register-clamped fallback that can be 100×+ too slow.

At codegen time, GRiD bakes the matching `config/launch_configs/<robot>/<gpu>.json` into the generated
`grid_launch_config.cuh`; the host kernel launchers (and therefore the python/jax/torch bindings) default their
launch config from it. If there's no entry for your (robot, GPU), GRiD falls back to a conservative default —
still correct, just not optimal.

## Layout
```
config/launch_configs/<robot>/<gpu>.json
```
- `<robot>` — the robot name (matches the URDF/codegen robot id, e.g. `iiwa14`, `go2`, `g1`).
- `<gpu>`   — a GPU key, `<model>_<arch>` lowercased, e.g. `rtx5090_sm120`.

## File format
```json
{
  "gpu": "rtx5090_sm120",
  "cuda_arch": "sm_120",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "autotune_N": 256,
  "source": "GRiD autotune sweep <date>",
  "bases": {
    "fixed":    { "crba": { "tier": "shared", "threads": 96, "us_at_optimal": 11.68 }, "...": {} },
    "floating": { "...": {} }
  }
}
```
`tier` ∈ {`shared`, `lite`, `minimal`}; `threads` is the optimal threads-per-block (single block). `us_at_optimal`
is the measured per-problem µs at N=`autotune_N` (informational).

A config may additionally carry **per-surface thread-pick overlays**: `ffi_bases` (jax FFI),
`pybind_bases` (numpy/pybind11), and `torch_bases` — same `{fixed|floating: {algo: {tier, threads}}}`
shape as `bases`, recorded by `test/benchmarks/autotune_ffi.py --surface {jax,numpy,torch}` (with
matching `ffi_meta`/`pybind_meta`/`torch_meta` provenance blocks). Each surface's bindings default
from its overlay when present, falling back to `bases`.

## Generate a config for YOUR robot / GPU
```
bash config/autotune_robot.sh <robot> [fixed floating]
```
This runs the GRiD autotune sweep (single-call timing off by default; RAM-safe serial build for big robots) and
writes `config/launch_configs/<robot>/<your_gpu>.json`. Re-run codegen + rebuild and the host launchers pick up your
values. (See the **"Autotune launch config for your robot / GPU"** section of
`docs/source/user_guide/tutorials/benchmarks.rst` for the full workflow.)

## Contribute a (robot, GPU) combo (please do! — this crowdsources a complete matrix)
1. Generate the config as above on a **quiet GPU** (timing must be isolated — close other GPU workloads).
2. Sanity-check the JSON against the format above; confirm `gpu`/`cuda_arch`/`gpu_name` are correct.
3. Open a PR adding `config/launch_configs/<robot>/<gpu>.json`. One file per (robot, GPU). Include in the PR
   description: GPU model, driver/CUDA version, and the robot's DoF/base. No code changes needed — codegen
   auto-discovers the file.

Currently seeded: **baxter, g1, go2, h1_2, h2_plus, iiwa14** on `rtx5090_sm120`. Note that
**h1_2 is retired** from the swept robot set (replaced by H2+); its legacy config is retained.
