# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-07-02  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: 3.9.0

All times in **µs**.

Columns:
- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only (pre_glass harness does not support floating-base).
- **glass**: GRiD HEAD with the pure-SIMT GLASS backend at the SHARED tier (formerly 'PERF'; max smem, lowest spill — full inner scratch in shared memory).
- **glass_lite**: GRiD HEAD at the LITE tier — partial spill of cold/large buffers to L2-pinned d_workspace; trades some throughput for ~50% smem headroom so more blocks fit per SM. `—` if the algorithm has a single tier. Under `--autotune-threads` this is the best-thread N=256 time from the collapsed autotune sweep (a single run autotunes all tiers); `—` in the single-call / N=16 sub-tables (the sweep tunes only the N=256 path).
- **glass_min**: GRiD HEAD at the MINIMAL tier — most aggressive spill so the kernel fits on lower-spec GPUs / leaves smem free for the caller. `—` if the algorithm has a single tier. Same autotune sourcing as glass_lite.
- **grid_best**: the autotuned global winner over (tier × thread-count) at **batch N=256 compute-only**, formatted `µs (tier@threads)`. Populated only when the sweep ran with `--autotune-threads`; `—` otherwise and in the single-call / N=16 sub-tables (the autotune tunes the N=256 path).
- **pin**: Pinocchio CPU reference (codegen where available).
- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only (id / fd / ee_pose / id_du); others render `—`.
- **mujoco_warp**: MuJoCo Warp (Warp-based MJX successor) GPU reference. Same MJCF + algo coverage as mjx; others render `—`.
- **frax_cpu / frax_gpu**: Frax (JAX) reference (https://github.com/danielpmorton/frax) timed separately on JAX's CPU and CUDA backends — Frax advertises both as fast. Subset of algos only (id / fd / crba / minv); others render `—`.
- **bard_cpu / bard_gpu**: BARD (PyTorch) reference (https://github.com/YueWang996/bard-pytorch-dynamics) timed separately on torch's CPU and CUDA backends. Subset of algos only (id / fd / crba); others render `—`. BARD times the full update_kinematics + algo pipeline per state.
- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; **< 1.00× = HEAD regressed**.

Each algorithm gets three sub-tables: **single-call**, **batch N=16**, **batch N=256**. Same backend columns + ratio in each. Values are median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: Pinocchio has no direct FDSVA_SO; the baseline is synthesized in-harness via the Singh/Carpentier chain rule (RNEA SO + ABA derivatives + Minv). This is what any downstream pinocchio user would write.

## Core Dynamics

### Inverse Dynamics (RNEA / Recursive Newton-Euler Algorithm)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.17 (codegen) | 388.06 | 866.41 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 0.33 (codegen) | 650.02 | 743.55 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1.01 (codegen) | 853.68 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.32 | — | — | — | 16.80 | 567.61 | 869.45 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 20.14 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 21.19 | — | — | — | 42.46 | 690.05 | 749.40 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 41.72 | — | — | — | 56.86 | 733.70 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.73 | 12.11 | 11.99 | 11.99 (minimal@128) | 51.12 | 448.79 | 878.81 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 31.78 | 20.14 | 19.84 | 19.84 (minimal@128) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 35.05 | 21.14 | 21.60 | 21.02 (shared@192) | 73.08 | 532.78 | 751.66 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 73.90 | 41.54 | 42.94 | 40.91 (shared@192) | 218.48 | 805.89 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Minv (M⁻¹, computed directly)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.29 (codegen) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 1.33 (codegen) | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 4.40 (codegen) | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.37 | — | — | — | 14.29 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 25.07 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 26.90 | — | — | — | 39.76 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 67.77 | — | — | — | 158.33 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.09 | 13.19 | 13.58 | 13.19 (lite@128) | 41.32 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 43.12 | 37.17 | 28.63 | 23.45 (shared@224) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 46.63 | 27.82 | 31.86 | 26.15 (shared@256) | 87.34 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 143.45 | 70.43 | 74.55 | 70.43 (lite@160) | 170.27 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.52 (codegen) | 668.11 | 1330.68 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 1.85 (codegen) | 687.40 | 1113.77 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 5.67 (codegen) | 843.48 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.15 | — | — | — | 17.66 | 411.25 | 1324.87 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 41.75 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 32.12 | — | — | — | 64.00 | 514.44 | 1113.60 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 88.98 | — | — | — | 183.00 | 670.34 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 27.20 | 17.11 | 18.73 | 16.95 (shared@128) | 58.05 | 382.35 | 1335.74 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 54.21 | 41.43 | 43.38 | 37.81 (shared@80) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 56.34 | 32.81 | 38.51 | 31.12 (shared@224) | 202.53 | 634.48 | 1315.48 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 183.52 | 89.48 | 93.37 | 89.48 (lite@128) | 303.18 | 774.05 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### ABA (Articulated Body Algorithm)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.38 (codegen) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 1.60 (codegen) | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 2.94 (codegen) | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.15 | — | — | — | 30.81 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 30.61 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 43.15 | — | — | — | 48.43 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 86.87 | — | — | — | 119.61 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.06 | 16.22 | 16.75 | 16.22 (shared@128) | 174.93 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 53.97 | 37.04 | 29.22 | 28.85 (shared@256) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 77.65 | 40.97 | 45.49 | 40.97 (lite@128) | 177.71 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 160.39 | 82.36 | 97.58 | 82.28 (shared@224) | 285.87 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.20 (codegen) | — | 102.28 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 0.47 (codegen) | — | 76.20 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1.67 (codegen) | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.48 | — | — | — | 10.56 | — | 102.98 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 18.45 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 13.88 | — | — | — | 25.25 | — | 75.37 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 35.02 | — | — | — | 35.51 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.18 | 11.77 | 12.63 | 11.70 (shared@96) | 24.51 | — | 103.32 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 30.83 | 18.37 | 18.93 | 18.37 (lite@224) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 22.92 | 15.63 | 16.22 | 14.92 (shared@192) | 42.97 | — | 76.10 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 64.28 | 37.69 | 40.26 | 36.08 (shared@224) | 148.27 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Gradients

### Inverse Dynamics Gradient (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 1.15 (codegen) | 603.69 | 41593.89 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 2.89 (codegen) | 917.95 | 91728.94 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 8.46 (codegen) | 1089.89 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.03 | — | — | — | 51.10 | 508.59 | 675079.71 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 33.08 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 39.01 | — | — | — | 99.92 | 456.02 | 1517746.42 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 102.17 | — | — | — | 256.15 | 1365.95 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 32.09 | 24.79 | 25.49 | 23.21 (shared@128) | 187.69 | 594.49 | 10891827.32 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 50.21 | 34.03 | 35.76 | 33.69 (shared@224) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 63.14 | 45.67 | 57.67 | 44.34 (shared@256) | 282.61 | 1144.03 | 24531933.88 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 202.13 | 168.21 | 174.95 | 168.21 (lite@384) | 364.06 | 2737.51 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics Gradient (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 2.34 (codegen) | 703.84 | 61054.12 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 5.67 (codegen) | 816.82 | 135540.52 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 17.78 (codegen) | 1161.76 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.99 | — | — | — | 62.71 | 721.60 | 991478.85 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 74.07 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 55.42 | — | — | — | 198.93 | 608.82 | 2164768.03 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 184.02 | — | — | — | 570.50 | 1137.34 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 53.10 | 46.62 | 36.37 | 32.67 (shared@128) | 103.44 | 615.83 | 15914608.24 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 89.51 | 80.03 | 59.16 | 59.16 (minimal@224) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 94.32 | 70.08 | 82.88 | 60.23 (shared@256) | 170.71 | 1187.06 | 40513155.56 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 385.39 | 251.56 | 263.72 | 251.56 (lite@320) | 639.63 | 3598.11 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### F_EXT_GRAD (∂tau/∂fext=-Jᵀ, ∂q̈/∂fext=M⁻¹Jᵀ)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### F_EXT_GRADIENT_DQ (∂(inverse_dynamics_gradient)/∂fext=-∂Jᵀ/∂q, fixed base)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Inverse Dynamics Regressor (Joint-torque Y; tau=Y·π, ∂tau/∂π)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics Parameter Gradient (∂q̈/∂π = -M⁻¹·Y)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Kinetic Energy Regressor (KE = y_KE·π, length 10·NB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Potential Energy Regressor (PE = y_PE·π, length 10·NB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 24.91 | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 51.97 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 40.38 | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 98.60 | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 35.21 | 24.43 | 23.90 | 23.90 (minimal@128) | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 63.42 | 46.06 | 37.44 | 37.44 (minimal@224) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 65.58 | 39.17 | 39.35 | 39.17 (lite@224) | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 195.05 | 101.77 | 101.79 | 101.77 (lite@128) | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 34.19 | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 75.28 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 64.79 | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 192.85 | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 53.66 | 33.08 | 33.31 | 33.08 (lite@192) | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 94.40 | 76.59 | 54.15 | 54.15 (minimal@256) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 115.71 | 80.14 | 80.20 | 80.14 (lite@320) | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 406.91 | 263.49 | 271.17 | 263.49 (lite@256) | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 34.63 | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 75.67 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 67.46 | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 194.54 | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 54.18 | 33.53 | 33.67 | 33.50 (shared@128) | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 96.16 | 76.65 | 54.54 | 54.54 (minimal@256) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 120.88 | 73.84 | 81.95 | 73.84 (lite@256) | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 418.71 | 274.71 | 262.82 | 262.82 (minimal@256) | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_Hessian (∂²x_{k+1}/∂z², z=[q,qd,u]; plant_step_hessian s_d2AB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Kinematics

### END_EFFECTOR_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.31 (direct) | 379.76 | 59.62 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 0.88 (direct) | 603.79 | 59.42 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1.01 (direct) | 684.54 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.98 | — | — | — | 9.98 | 386.66 | 60.84 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 8.46 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 23.58 | — | — | — | 18.19 | 274.43 | 59.70 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 28.57 | — | — | — | 58.48 | 323.73 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.26 | 7.24 | 7.10 | 7.07 (shared@128) | 28.59 | 508.05 | 62.10 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 9.17 | 8.61 | 8.78 | 8.61 (lite@128) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 41.55 | 23.69 | 23.88 | 23.69 (lite@256) | 53.16 | 297.68 | 60.93 | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 50.63 | 28.69 | 28.92 | 28.32 (shared@256) | 96.03 | 358.71 | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 0.34 (direct) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 0.54 (direct) | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1.15 (direct) | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.77 | — | — | — | 12.20 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 18.86 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 37.93 | — | — | — | 17.02 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 46.63 | — | — | — | 52.83 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.25 | 15.42 | 15.56 | 15.42 (lite@48) | 39.87 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 27.06 | 18.33 | 18.32 | 18.27 (shared@96) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 70.50 | 36.23 | 38.26 | 36.23 (lite@80) | 41.70 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 109.72 | 44.97 | 50.02 | 44.15 (shared@80) | 84.27 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_HESSIAN (2nd-order EE Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 1.42 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 2.89 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 7.17 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 9.93 | — | — | — | 30.45 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 15.04 | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 47.69 | — | — | — | 55.76 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 61.99 | — | — | — | 180.60 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.61 | 10.29 | 10.35 | 10.29 (lite@128) | 44.13 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | 23.79 | 17.75 | 16.24 | 16.24 (minimal@384) | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 50.01 | 45.19 | 48.43 | 45.19 (lite@256) | 85.82 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 119.12 | 98.02 | 98.68 | 98.02 (lite@384) | 315.05 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### FRAME_JACOBIAN (general-frame J: LOCAL/WORLD/LWA)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### FRAME_JACOBIAN_DOT (time derivative Jdot of the general-frame J)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### OSC_INERTIA (operational-space inertia Lambda = (J Minv J^T)^-1)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_RUNTIME (runtime target/offset pose [xyz;rpy])

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_GRADIENT_RUNTIME (runtime target/offset pose Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 7.43 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 26.17 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 77.03 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 147.08 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 425.12 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1797.63 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 391.34 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 838.39 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 2334.40 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 18.75 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 139.20 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 1087.19 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 304.40 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 2324.32 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 18912.11 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | 770.28 | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | 2482.43 | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | 25660.57 | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Centroidal

### Generalized Gravity g(q)=RNEA(q,0,0)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Nonlinear Effects c(q,qd)=RNEA(q,qd,0)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Energy (KE/PE/mechanical)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### CoM + CoM Jacobian

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### CCRBA (A, h)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Coriolis Matrix C(q,q̇)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### dCCRBA (∂A/∂q tensor, 6×NV×NV)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

### CMM Time Variation (Ȧ, 6×NV)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Plant

### Plant (cost/constraint/step primitives)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | mujoco_warp | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| baxter | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | fixed | — | — | — | — | — | — | — | — | — | — | — | — | — |
| h2_plus | floating | — | — | — | — | — | — | — | — | — | — | — | — | — |

