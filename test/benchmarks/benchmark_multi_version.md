# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-06-14  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: ?

All times in **µs**.

Columns:
- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only (pre_glass harness does not support floating-base).
- **glass**: GRiD HEAD with the pure-SIMT GLASS backend at the SHARED tier (formerly 'PERF'; max smem, lowest spill — full inner scratch in shared memory).
- **glass_lite**: GRiD HEAD at the LITE tier — partial spill of cold/large buffers to L2-pinned d_workspace; trades some throughput for ~50% smem headroom so more blocks fit per SM. `—` if the algorithm has a single tier. Under `--autotune-threads` this is the best-thread N=256 time from the collapsed autotune sweep (a single run autotunes all tiers); `—` in the single-call / N=16 sub-tables (the sweep tunes only the N=256 path).
- **glass_min**: GRiD HEAD at the MINIMAL tier — most aggressive spill so the kernel fits on lower-spec GPUs / leaves smem free for the caller. `—` if the algorithm has a single tier. Same autotune sourcing as glass_lite.
- **grid_best**: the autotuned global winner over (tier × thread-count) at **batch N=256 compute-only**, formatted `µs (tier@threads)`. Populated only when the sweep ran with `--autotune-threads`; `—` otherwise and in the single-call / N=16 sub-tables (the autotune tunes the N=256 path).
- **pin**: Pinocchio CPU reference (codegen where available).
- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only (id / fd / ee_pose / id_du); others render `—`.
- **frax_cpu / frax_gpu**: Frax (JAX) reference (https://github.com/danielpmorton/frax) timed separately on JAX's CPU and CUDA backends — Frax advertises both as fast. Subset of algos only (id / fd / crba / minv); others render `—`.
- **bard_cpu / bard_gpu**: BARD (PyTorch) reference (https://github.com/YueWang996/bard-pytorch-dynamics) timed separately on torch's CPU and CUDA backends. Subset of algos only (id / fd / crba); others render `—`. BARD times the full update_kinematics + algo pipeline per state.
- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; **< 1.00× = HEAD regressed**.

Each algorithm gets three sub-tables: **single-call**, **batch N=16**, **batch N=256**. Same backend columns + ratio in each. Values are median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: Pinocchio has no direct FDSVA_SO; the baseline is synthesized in-harness via the Singh/Carpentier chain rule (RNEA SO + ABA derivatives + Minv). This is what any downstream pinocchio user would write.

## Core Dynamics

### Inverse Dynamics (RNEA / Recursive Newton-Euler Algorithm)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.25 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 18.03 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 14.16 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 20.99 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 34.42 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 41.22 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.63 | 12.08 | 12.04 | 12.04 (minimal@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 29.00 | 17.83 | 17.90 | 17.79 (shared@160) | — | — | — | — | — | — | — |
| go2 | fixed | — | 14.69 | 14.20 | 14.15 | 14.15 (minimal@256) | — | — | — | — | — | — | — |
| go2 | floating | — | 34.95 | 21.15 | 21.57 | 20.99 (shared@224) | — | — | — | — | — | — | — |
| g1 | fixed | — | 57.64 | 31.06 | 31.43 | 31.06 (lite@320) | — | — | — | — | — | — | — |
| g1 | floating | — | 72.79 | 36.96 | 38.61 | 36.51 (shared@256) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Minv (M⁻¹, computed directly)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.35 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 23.82 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 16.12 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 26.99 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 52.59 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 67.73 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.02 | 13.31 | 13.58 | 13.31 (lite@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 39.78 | 24.41 | 28.76 | 22.46 (shared@160) | — | — | — | — | — | — | — |
| go2 | fixed | — | 26.05 | 16.21 | 18.34 | 16.21 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 46.08 | 27.75 | 31.95 | 26.14 (shared@224) | — | — | — | — | — | — | — |
| g1 | fixed | — | 81.29 | 56.19 | 51.05 | 48.90 (shared@128) | — | — | — | — | — | — | — |
| g1 | floating | — | 143.01 | 63.21 | 69.37 | 63.21 (lite@192) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.98 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 28.01 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 19.91 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 31.72 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 68.82 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 88.88 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.93 | 17.01 | 18.81 | 16.63 (shared@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 47.35 | 30.90 | 35.67 | 26.54 (shared@128) | — | — | — | — | — | — | — |
| go2 | fixed | — | 31.48 | 18.90 | 23.08 | 18.90 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 55.41 | 32.51 | 38.48 | 30.94 (shared@256) | — | — | — | — | — | — | — |
| g1 | fixed | — | 102.64 | 64.94 | 69.67 | 63.54 (shared@128) | — | — | — | — | — | — | — |
| g1 | floating | — | 182.50 | 81.41 | 86.79 | 81.41 (lite@192) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### ABA (Articulated Body Algorithm)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.88 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 31.90 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 19.18 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 42.49 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 52.21 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 84.92 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.77 | 16.00 | 16.74 | 16.00 (shared@112) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 55.93 | 30.91 | 33.65 | 30.54 (shared@128) | — | — | — | — | — | — | — |
| go2 | fixed | — | 31.35 | 18.82 | 20.60 | 18.82 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 76.57 | 40.57 | 45.05 | 40.57 (lite@128) | — | — | — | — | — | — | — |
| g1 | fixed | — | 92.92 | 64.06 | 50.18 | 49.13 (shared@128) | — | — | — | — | — | — | — |
| g1 | floating | — | 156.52 | 79.06 | 89.15 | 78.61 (shared@192) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.32 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 11.96 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 13.59 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 13.85 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 37.58 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 34.87 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.02 | 11.68 | 12.57 | 11.68 (lite@96) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 18.86 | 12.16 | 13.02 | 12.16 (lite@160) | — | — | — | — | — | — | — |
| go2 | fixed | — | 14.32 | 14.00 | 14.88 | 13.90 (shared@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 22.90 | 15.63 | 16.46 | 14.89 (shared@192) | — | — | — | — | — | — | — |
| g1 | fixed | — | 65.97 | 33.68 | 35.12 | 33.68 (lite@320) | — | — | — | — | — | — | — |
| g1 | floating | — | 63.91 | 37.16 | 38.28 | 35.18 (shared@224) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Gradients

### Inverse Dynamics Gradient (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.00 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 33.86 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 24.66 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 38.75 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 53.81 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 101.60 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 32.15 | 24.75 | 25.28 | 23.19 (shared@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 51.63 | 38.40 | 47.58 | 33.94 (shared@256) | — | — | — | — | — | — | — |
| go2 | fixed | — | 35.65 | 25.22 | 26.71 | 25.22 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 62.79 | 45.42 | 57.34 | 44.19 (shared@256) | — | — | — | — | — | — | — |
| g1 | fixed | — | 90.53 | 66.47 | 65.00 | 65.00 (minimal@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 202.11 | 168.40 | 174.72 | 168.40 (lite@384) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics Gradient (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.96 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 48.60 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 35.19 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 54.79 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 120.45 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 183.72 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 52.72 | 46.77 | 39.82 | 32.55 (shared@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 80.60 | 62.16 | 66.96 | 48.02 (shared@256) | — | — | — | — | — | — | — |
| go2 | fixed | — | 55.15 | 35.02 | 40.34 | 34.67 (shared@192) | — | — | — | — | — | — | — |
| go2 | floating | — | 92.98 | 69.81 | 84.47 | 60.07 (shared@224) | — | — | — | — | — | — | — |
| g1 | fixed | — | 242.87 | 152.77 | 125.43 | 125.43 (minimal@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 384.21 | 253.03 | 260.08 | 253.03 (lite@320) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### F_EXT_GRAD (∂tau/∂fext=-Jᵀ, ∂q̈/∂fext=M⁻¹Jᵀ)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### F_EXT_GRADIENT_DQ (∂(inverse_dynamics_gradient)/∂fext=-∂Jᵀ/∂q, fixed base)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Inverse Dynamics Regressor (Joint-torque Y; tau=Y·π, ∂tau/∂π)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Forward Dynamics Parameter Gradient (∂q̈/∂π = -M⁻¹·Y)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Kinetic Energy Regressor (KE = y_KE·π, length 10·NB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Potential Energy Regressor (PE = y_PE·π, length 10·NB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 24.25 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 36.71 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 27.34 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 40.09 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 77.23 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 88.04 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 34.74 | 24.23 | 23.92 | 23.92 (minimal@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 58.68 | 34.94 | 35.03 | 34.94 (lite@128) | — | — | — | — | — | — | — |
| go2 | fixed | — | 39.13 | 26.60 | 26.58 | 26.55 (shared@192) | — | — | — | — | — | — | — |
| go2 | floating | — | 64.87 | 39.10 | 39.39 | 39.01 (shared@224) | — | — | — | — | — | — | — |
| g1 | fixed | — | 106.80 | 68.37 | 56.98 | 56.98 (minimal@192) | — | — | — | — | — | — | — |
| g1 | floating | — | 177.66 | 92.83 | 91.80 | 91.80 (minimal@128) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.96 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 54.68 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 35.69 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 58.72 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 120.83 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 170.43 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 52.98 | 32.78 | 33.13 | 32.78 (lite@128) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 92.68 | 53.48 | 61.51 | 53.37 (shared@192) | — | — | — | — | — | — | — |
| go2 | fixed | — | 55.71 | 35.37 | 36.68 | 35.36 (shared@192) | — | — | — | — | — | — | — |
| go2 | floating | — | 101.33 | 80.43 | 81.29 | 80.43 (lite@288) | — | — | — | — | — | — | — |
| g1 | fixed | — | 225.62 | 101.69 | 102.82 | 101.69 (lite@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 369.85 | 245.79 | 245.71 | 245.71 (minimal@256) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 34.18 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 57.28 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 35.76 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 66.89 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 119.80 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 173.95 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 53.38 | 33.23 | 33.33 | 33.17 (shared@192) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 97.76 | 55.98 | 63.92 | 55.84 (shared@256) | — | — | — | — | — | — | — |
| go2 | fixed | — | 56.58 | 35.57 | 36.38 | 35.48 (shared@192) | — | — | — | — | — | — | — |
| go2 | floating | — | 120.00 | 81.67 | 82.74 | 81.67 (lite@288) | — | — | — | — | — | — | — |
| g1 | fixed | — | 245.82 | 102.13 | 103.03 | 102.13 (lite@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 408.90 | 254.31 | 247.49 | 247.49 (minimal@256) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Integrator_Hessian (∂²x_{k+1}/∂z², z=[q,qd,u]; plant_step_hessian s_d2AB)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Kinematics

### END_EFFECTOR_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.98 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 23.92 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 7.12 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 23.58 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 11.18 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 28.58 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.21 | 7.07 | 7.11 | 7.07 (lite@352) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 41.86 | 23.81 | 24.05 | 23.81 (lite@128) | — | — | — | — | — | — | — |
| go2 | fixed | — | 7.47 | 7.24 | 7.20 | 7.20 (minimal@96) | — | — | — | — | — | — | — |
| go2 | floating | — | 41.55 | 23.66 | 23.99 | 23.66 (lite@224) | — | — | — | — | — | — | — |
| g1 | fixed | — | 17.28 | 11.77 | 11.24 | 11.24 (minimal@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 51.02 | 28.82 | 29.25 | 28.46 (shared@224) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.77 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 33.99 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 17.80 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 37.59 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 25.22 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 46.66 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.24 | 15.57 | 15.66 | 15.57 (lite@48) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 55.91 | 33.33 | 33.89 | 32.93 (shared@80) | — | — | — | — | — | — | — |
| go2 | fixed | — | 22.13 | 16.57 | 16.76 | 16.57 (lite@96) | — | — | — | — | — | — | — |
| go2 | floating | — | 70.56 | 36.36 | 38.51 | 36.36 (lite@80) | — | — | — | — | — | — | — |
| g1 | fixed | — | 46.45 | 24.64 | 26.55 | 24.64 (lite@80) | — | — | — | — | — | — | — |
| g1 | floating | — | 107.67 | 45.45 | 50.05 | 44.28 (shared@80) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_HESSIAN (2nd-order EE Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.04 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 36.73 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 12.52 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 47.74 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 27.71 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 61.93 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.63 | 10.30 | 10.38 | 10.30 (lite@160) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 64.94 | 34.35 | 36.58 | 33.57 (shared@80) | — | — | — | — | — | — | — |
| go2 | fixed | — | 13.65 | 14.34 | 13.21 | 13.21 (minimal@256) | — | — | — | — | — | — | — |
| go2 | floating | — | 50.20 | 44.63 | 48.23 | 44.63 (lite@448) | — | — | — | — | — | — | — |
| g1 | fixed | — | 48.66 | 45.76 | 47.60 | 45.76 (lite@384) | — | — | — | — | — | — | — |
| g1 | floating | — | 120.17 | 97.85 | 100.05 | 97.85 (lite@384) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### FRAME_JACOBIAN (general-frame J: LOCAL/WORLD/LWA)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### FRAME_JACOBIAN_DOT (time derivative Jdot of the general-frame J)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### OSC_INERTIA (operational-space inertia Lambda = (J Minv J^T)^-1)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_RUNTIME (runtime target/offset pose [xyz;rpy])

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### END_EFFECTOR_POSE_GRADIENT_RUNTIME (runtime target/offset pose Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 37.26 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 238.29 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 50.55 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 329.71 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 478.21 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 975.03 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 69.32 | 48.16 | 65.54 | 48.16 (lite@352) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 470.81 | 246.83 | 327.92 | 246.30 (shared@96) | — | — | — | — | — | — | — |
| go2 | fixed | — | 97.41 | 69.51 | 86.81 | 69.51 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 670.31 | 521.45 | 448.38 | 383.68 (shared@96) | — | — | — | — | — | — | — |
| g1 | fixed | — | 1336.51 | 1539.77 | 1315.61 | 1218.13 (shared@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 2235.15 | 2145.19 | 1972.00 | 1623.72 (shared@128) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 37.18 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 50.47 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 478.15 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 69.34 | 48.09 | 65.63 | 48.09 (lite@352) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 97.37 | 69.59 | 86.71 | 69.59 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 1335.26 | 1539.56 | 1315.83 | 1217.91 (shared@256) | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 128.06 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 238.22 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 161.33 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 329.83 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 732.20 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 975.28 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 154.61 | 122.21 | 135.96 | 121.33 (shared@64) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 471.00 | 247.16 | 327.92 | 246.23 (shared@96) | — | — | — | — | — | — | — |
| go2 | fixed | — | 315.14 | 178.83 | 200.55 | 178.83 (lite@96) | — | — | — | — | — | — | — |
| go2 | floating | — | 670.22 | 521.21 | 448.57 | 383.43 (shared@96) | — | — | — | — | — | — | — |
| g1 | fixed | — | 1530.51 | 954.35 | 997.70 | 954.35 (lite@224) | — | — | — | — | — | — | — |
| g1 | floating | — | 2228.19 | 2144.37 | 1971.62 | 1622.54 (shared@128) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 73.10 | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 363.61 | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | 94.47 | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | 448.23 | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | 2652.04 | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | 6856.40 | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 130.72 | 114.68 | 121.03 | 90.15 (shared@192) | — | — | — | — | — | — | — |
| iiwa14 | floating | — | 706.94 | 644.10 | 499.27 | 417.77 (shared@192) | — | — | — | — | — | — | — |
| go2 | fixed | — | 173.91 | 144.00 | 162.80 | 144.00 (lite@224) | — | — | — | — | — | — | — |
| go2 | floating | — | 909.11 | 943.90 | 773.37 | 773.37 (minimal@288) | — | — | — | — | — | — | — |
| g1 | fixed | — | 7015.75 | 7323.35 | 7071.43 | 6165.44 (shared@256) | — | — | — | — | — | — | — |
| g1 | floating | — | 17772.07 | 10848.20 | 17169.23 | 10848.20 (lite@512) | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Centroidal

### Generalized Gravity g(q)=RNEA(q,0,0)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Nonlinear Effects c(q,qd)=RNEA(q,qd,0)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Energy (KE/PE/mechanical)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### CoM + CoM Jacobian

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### CCRBA (A, h)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### Coriolis Matrix C(q,q̇)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### dCCRBA (∂A/∂q tensor, 6×NV×NV)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

### CMM Time Variation (Ȧ, 6×NV)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

## Plant

### Plant (cost/constraint/step primitives)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best | pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — | — | — | — |

