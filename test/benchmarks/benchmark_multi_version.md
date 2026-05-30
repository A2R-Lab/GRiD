# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-05-30  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: ?

All times in **µs**.

Columns:
- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only (pre_glass harness does not support floating-base).
- **glass**: GRiD HEAD with the pure-SIMT GLASS backend at the PERF tier (max smem; lowest spill).
- **glass_lite**: GRiD HEAD at the LITE tier — partial spill of cold/large buffers to L2-pinned d_workspace; trades some throughput for ~50% smem headroom so more blocks fit per SM. `—` if the algorithm has a single tier.
- **glass_min**: GRiD HEAD at the MINIMAL tier — most aggressive spill so the kernel fits on lower-spec GPUs / leaves smem free for the caller. `—` if the algorithm has a single tier.
- **pin**: Pinocchio CPU reference (codegen where available).
- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only (id / fd / ee_pose / id_du); others render `—`.
- **frax_cpu / frax_gpu**: Frax (JAX) reference (https://github.com/danielpmorton/frax) timed separately on JAX's CPU and CUDA backends — Frax advertises both as fast. Subset of algos only (id / fd / crba / minv); others render `—`.
- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; **< 1.00× = HEAD regressed**.

Each algorithm gets three sub-tables: **single-call**, **batch N=16**, **batch N=256**. Same backend columns + ratio in each. Values are median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: Pinocchio has no direct FDSVA_SO; the baseline is synthesized in-harness via the Singh/Carpentier chain rule (RNEA SO + ABA derivatives + Minv). This is what any downstream pinocchio user would write.

## Core Dynamics

### ID (Inverse Dynamics)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.52 | 4.49 | 4.54 | — | — | — | — | — |
| iiwa14 | floating | — | 9.94 | 9.89 | 10.32 | — | — | — | — | — |
| go2 | fixed | — | 6.02 | 5.89 | 5.94 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.09 | 11.24 | 11.12 | — | — | — | — | — |
| iiwa14 | floating | — | 16.22 | 16.43 | 16.53 | — | — | — | — | — |
| go2 | fixed | — | 12.95 | 12.69 | 12.84 | — | — | — | — | — |
| go2 | floating | — | 18.99 | 19.31 | 19.44 | — | — | — | — | — |
| g1 | fixed | — | 30.90 | 31.20 | 31.01 | — | — | — | — | — |
| g1 | floating | — | 36.68 | 37.07 | 38.00 | — | — | — | — | — |
| h1_2 | fixed | — | 47.83 | 46.28 | 46.34 | — | — | — | — | — |
| h1_2 | floating | — | 52.03 | 52.26 | 53.66 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.41 | 11.31 | 11.42 | — | — | — | — | — |
| iiwa14 | floating | — | 25.55 | 26.69 | 17.71 | — | — | — | — | — |
| go2 | fixed | — | 13.37 | 13.13 | 13.21 | — | — | — | — | — |
| go2 | floating | — | 30.81 | 31.25 | 20.60 | — | — | — | — | — |
| g1 | fixed | — | 52.69 | 53.76 | 30.96 | — | — | — | — | — |
| g1 | floating | — | 64.64 | 65.10 | 40.53 | — | — | — | — | — |
| h1_2 | fixed | — | 49.96 | 82.77 | 46.86 | — | — | — | — | — |
| h1_2 | floating | — | 94.37 | 94.80 | 56.36 | — | — | — | — | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.19 | 6.24 | 6.39 | — | — | — | — | — |
| iiwa14 | floating | — | 13.88 | 14.58 | 15.31 | — | — | — | — | — |
| go2 | fixed | — | 8.05 | 8.13 | 8.74 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.22 | 12.21 | 12.38 | — | — | — | — | — |
| iiwa14 | floating | — | 21.38 | 22.56 | 24.57 | — | — | — | — | — |
| go2 | fixed | — | 14.78 | 14.76 | 15.41 | — | — | — | — | — |
| go2 | floating | — | 24.09 | 24.64 | 26.51 | — | — | — | — | — |
| g1 | fixed | — | 46.55 | 54.07 | 47.69 | — | — | — | — | — |
| g1 | floating | — | 59.98 | 56.54 | 61.85 | — | — | — | — | — |
| h1_2 | fixed | — | 67.97 | 75.85 | 71.69 | — | — | — | — | — |
| h1_2 | floating | — | 85.58 | 82.48 | 87.15 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.70 | 12.71 | 12.89 | — | — | — | — | — |
| iiwa14 | floating | — | 35.03 | 39.47 | 28.15 | — | — | — | — | — |
| go2 | fixed | — | 23.11 | 15.12 | 16.95 | — | — | — | — | — |
| go2 | floating | — | 40.34 | 42.66 | 29.71 | — | — | — | — | — |
| g1 | fixed | — | 76.87 | 102.03 | 69.40 | — | — | — | — | — |
| g1 | floating | — | 130.30 | 117.31 | 100.21 | — | — | — | — | — |
| h1_2 | fixed | — | 97.59 | 115.14 | 94.58 | — | — | — | — | — |
| h1_2 | floating | — | 126.67 | 174.21 | 133.17 | — | — | — | — | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.45 | 8.66 | 8.96 | — | — | — | — | — |
| iiwa14 | floating | — | 17.80 | 20.92 | 21.64 | — | — | — | — | — |
| go2 | fixed | — | 9.96 | 9.84 | 11.57 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.31 | 15.48 | 16.51 | — | — | — | — | — |
| iiwa14 | floating | — | 25.53 | 28.98 | 31.05 | — | — | — | — | — |
| go2 | fixed | — | 18.26 | 18.00 | 19.90 | — | — | — | — | — |
| go2 | floating | — | 28.72 | 29.58 | 32.98 | — | — | — | — | — |
| g1 | fixed | — | 61.98 | 61.85 | 65.57 | — | — | — | — | — |
| g1 | floating | — | 79.43 | 75.45 | 80.86 | — | — | — | — | — |
| h1_2 | fixed | — | 75.70 | 90.42 | 94.16 | — | — | — | — | — |
| h1_2 | floating | — | 111.45 | 105.85 | 110.58 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.66 | 16.31 | 19.13 | — | — | — | — | — |
| iiwa14 | floating | — | 42.25 | 52.32 | 35.07 | — | — | — | — | — |
| go2 | fixed | — | 28.15 | 17.47 | 21.42 | — | — | — | — | — |
| go2 | floating | — | 49.26 | 52.03 | 36.51 | — | — | — | — | — |
| g1 | fixed | — | 97.86 | 123.70 | 91.52 | — | — | — | — | — |
| g1 | floating | — | 168.53 | 159.73 | 129.13 | — | — | — | — | — |
| h1_2 | fixed | — | 142.90 | 179.56 | 125.14 | — | — | — | — | — |
| h1_2 | floating | — | 164.08 | 219.98 | 157.55 | — | — | — | — | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.23 | 8.35 | 9.11 | — | — | — | — | — |
| iiwa14 | floating | — | 21.93 | 21.20 | 23.73 | — | — | — | — | — |
| go2 | fixed | — | 10.32 | 10.37 | 11.58 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.31 | 14.29 | 14.51 | — | — | — | — | — |
| iiwa14 | floating | — | 27.95 | 27.97 | 30.90 | — | — | — | — | — |
| go2 | fixed | — | 17.60 | 17.75 | 19.50 | — | — | — | — | — |
| go2 | floating | — | 37.38 | 37.34 | 40.82 | — | — | — | — | — |
| g1 | fixed | — | 45.95 | 62.46 | 48.49 | — | — | — | — | — |
| g1 | floating | — | 73.27 | 72.55 | 82.35 | — | — | — | — | — |
| h1_2 | fixed | — | 77.67 | 96.45 | 90.62 | — | — | — | — | — |
| h1_2 | floating | — | 113.46 | 121.00 | 145.19 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.72 | 15.03 | 15.41 | — | — | — | — | — |
| iiwa14 | floating | — | 49.22 | 50.77 | 31.48 | — | — | — | — | — |
| go2 | fixed | — | 27.57 | 17.42 | 19.01 | — | — | — | — | — |
| go2 | floating | — | 67.64 | 68.62 | 41.46 | — | — | — | — | — |
| g1 | fixed | — | 83.98 | 78.31 | 49.47 | — | — | — | — | — |
| g1 | floating | — | 138.17 | 137.99 | 84.68 | — | — | — | — | — |
| h1_2 | fixed | — | 95.48 | 132.45 | 101.92 | — | — | — | — | — |
| h1_2 | floating | — | 216.73 | 233.19 | 149.38 | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.66 | 4.57 | 4.67 | — | — | — | — | — |
| iiwa14 | floating | — | 4.76 | 4.77 | 5.09 | — | — | — | — | — |
| go2 | fixed | — | 6.45 | 6.31 | 6.63 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.53 | 10.58 | 11.45 | — | — | — | — | — |
| iiwa14 | floating | — | 11.07 | 10.88 | 11.67 | — | — | — | — | — |
| go2 | fixed | — | 12.53 | 12.54 | 13.28 | — | — | — | — | — |
| go2 | floating | — | 12.65 | 12.67 | 13.15 | — | — | — | — | — |
| g1 | fixed | — | 33.54 | 34.29 | 33.48 | — | — | — | — | — |
| g1 | floating | — | 31.19 | 32.01 | 31.89 | — | — | — | — | — |
| h1_2 | fixed | — | 54.00 | 52.72 | 54.76 | — | — | — | — | — |
| h1_2 | floating | — | 48.97 | 51.66 | 54.41 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.05 | 11.10 | 12.50 | — | — | — | — | — |
| iiwa14 | floating | — | 16.91 | 16.89 | 12.96 | — | — | — | — | — |
| go2 | fixed | — | 13.16 | 13.18 | 14.02 | — | — | — | — | — |
| go2 | floating | — | 20.56 | 20.57 | 16.81 | — | — | — | — | — |
| g1 | fixed | — | 60.29 | 61.30 | 34.24 | — | — | — | — | — |
| g1 | floating | — | 57.52 | 59.34 | 42.61 | — | — | — | — | — |
| h1_2 | fixed | — | 100.14 | 97.09 | 56.11 | — | — | — | — | — |
| h1_2 | floating | — | 96.91 | 103.19 | 99.60 | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.60 | 7.53 | 8.54 | — | — | — | — | — |
| iiwa14 | floating | — | 15.99 | 17.49 | 17.31 | — | — | — | — | — |
| go2 | fixed | — | 8.90 | 8.78 | 9.16 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 21.84 | 21.74 | 22.78 | — | — | — | — | — |
| iiwa14 | floating | — | 31.08 | 33.50 | 37.16 | — | — | — | — | — |
| go2 | fixed | — | 23.39 | 23.20 | 23.19 | — | — | — | — | — |
| go2 | floating | — | 35.26 | 35.19 | 39.49 | — | — | — | — | — |
| g1 | fixed | — | 47.30 | 57.51 | 51.22 | — | — | — | — | — |
| g1 | floating | — | 90.20 | 88.22 | 100.32 | — | — | — | — | — |
| h1_2 | fixed | — | 94.06 | 78.24 | 82.82 | — | — | — | — | — |
| h1_2 | floating | — | 180.09 | 180.76 | 201.30 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.45 | 22.65 | 24.07 | — | — | — | — | — |
| iiwa14 | floating | — | 46.42 | 53.57 | 46.45 | — | — | — | — | — |
| go2 | fixed | — | 24.09 | 23.79 | 24.95 | — | — | — | — | — |
| go2 | floating | — | 55.45 | 56.29 | 53.84 | — | — | — | — | — |
| g1 | fixed | — | 78.35 | 67.81 | 60.47 | — | — | — | — | — |
| g1 | floating | — | 183.22 | 184.09 | 165.89 | — | — | — | — | — |
| h1_2 | fixed | — | 175.43 | 147.29 | 106.93 | — | — | — | — | — |
| h1_2 | floating | — | 431.44 | 433.66 | 456.92 | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.39 | 22.43 | 21.47 | — | — | — | — | — |
| iiwa14 | floating | — | 28.63 | 35.45 | 36.62 | — | — | — | — | — |
| go2 | fixed | — | 17.85 | 17.87 | 19.81 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.26 | 37.47 | 36.29 | — | — | — | — | — |
| iiwa14 | floating | — | 44.54 | 53.15 | 54.50 | — | — | — | — | — |
| go2 | fixed | — | 32.77 | 32.73 | 35.91 | — | — | — | — | — |
| go2 | floating | — | 49.58 | 56.58 | 62.40 | — | — | — | — | — |
| g1 | fixed | — | 105.44 | 116.84 | 105.75 | — | — | — | — | — |
| g1 | floating | — | 163.51 | 149.20 | 164.94 | — | — | — | — | — |
| h1_2 | fixed | — | 147.25 | 169.43 | 166.07 | — | — | — | — | — |
| h1_2 | floating | — | 269.30 | 288.52 | 293.04 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 47.69 | 37.91 | 37.37 | — | — | — | — | — |
| iiwa14 | floating | — | 73.09 | 93.50 | 63.71 | — | — | — | — | — |
| go2 | fixed | — | 50.43 | 32.27 | 37.73 | — | — | — | — | — |
| go2 | floating | — | 82.50 | 98.99 | 77.91 | — | — | — | — | — |
| g1 | fixed | — | 215.49 | 179.64 | 122.40 | — | — | — | — | — |
| g1 | floating | — | 349.22 | 304.02 | 249.83 | — | — | — | — | — |
| h1_2 | fixed | — | 289.96 | 342.69 | 324.53 | — | — | — | — | — |
| h1_2 | floating | — | 697.98 | 721.89 | 717.27 | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.79 | 8.79 | — | — | — | — | — | — |
| iiwa14 | floating | — | 18.88 | 18.91 | — | — | — | — | — | — |
| go2 | fixed | — | 10.11 | 10.11 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.67 | 22.69 | 22.30 | — | — | — | — | — |
| iiwa14 | floating | — | 33.68 | 33.64 | 33.74 | — | — | — | — | — |
| go2 | fixed | — | 25.58 | 25.52 | 25.55 | — | — | — | — | — |
| go2 | floating | — | 36.79 | 36.85 | 36.66 | — | — | — | — | — |
| g1 | fixed | — | 69.78 | 69.60 | 57.03 | — | — | — | — | — |
| g1 | floating | — | 87.36 | 90.03 | 89.66 | — | — | — | — | — |
| h1_2 | fixed | — | 83.79 | 83.87 | 84.16 | — | — | — | — | — |
| h1_2 | floating | — | 121.80 | 121.46 | 121.47 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.52 | 31.61 | 30.85 | — | — | — | — | — |
| iiwa14 | floating | — | 52.95 | 53.06 | 52.69 | — | — | — | — | — |
| go2 | fixed | — | 35.67 | 35.67 | 35.57 | — | — | — | — | — |
| go2 | floating | — | 58.77 | 58.66 | 58.93 | — | — | — | — | — |
| g1 | fixed | — | 106.60 | 105.15 | 97.80 | — | — | — | — | — |
| g1 | floating | — | 177.17 | 138.10 | 137.39 | — | — | — | — | — |
| h1_2 | fixed | — | 153.03 | 153.39 | 153.11 | — | — | — | — | — |
| h1_2 | floating | — | 179.68 | 179.14 | 179.82 | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.10 | 17.11 | — | — | — | — | — | — |
| iiwa14 | floating | — | 33.56 | 33.55 | — | — | — | — | — | — |
| go2 | fixed | — | 18.07 | 17.99 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.41 | 31.31 | 31.33 | — | — | — | — | — |
| iiwa14 | floating | — | 49.73 | 49.50 | 52.95 | — | — | — | — | — |
| go2 | fixed | — | 33.22 | 33.32 | 33.10 | — | — | — | — | — |
| go2 | floating | — | 53.28 | 61.38 | 60.35 | — | — | — | — | — |
| g1 | fixed | — | 107.04 | 80.22 | 80.18 | — | — | — | — | — |
| g1 | floating | — | 171.90 | 160.74 | 160.63 | — | — | — | — | — |
| h1_2 | fixed | — | 160.93 | 160.62 | 160.88 | — | — | — | — | — |
| h1_2 | floating | — | 307.11 | 306.81 | 306.45 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.04 | 48.09 | 48.52 | — | — | — | — | — |
| iiwa14 | floating | — | 82.42 | 82.93 | 60.33 | — | — | — | — | — |
| go2 | fixed | — | 51.10 | 50.89 | 50.77 | — | — | — | — | — |
| go2 | floating | — | 90.38 | 78.23 | 78.31 | — | — | — | — | — |
| g1 | fixed | — | 220.92 | 144.21 | 144.27 | — | — | — | — | — |
| g1 | floating | — | 370.37 | 266.75 | 258.07 | — | — | — | — | — |
| h1_2 | fixed | — | 329.41 | 324.82 | 329.51 | — | — | — | — | — |
| h1_2 | floating | — | 792.74 | 795.90 | 820.56 | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.03 | 17.02 | — | — | — | — | — | — |
| iiwa14 | floating | — | 35.65 | 35.64 | — | — | — | — | — | — |
| go2 | fixed | — | 18.23 | 18.23 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.64 | 31.61 | 31.65 | — | — | — | — | — |
| iiwa14 | floating | — | 52.35 | 52.27 | 55.41 | — | — | — | — | — |
| go2 | fixed | — | 33.40 | 33.45 | 33.07 | — | — | — | — | — |
| go2 | floating | — | 60.10 | 62.84 | 62.48 | — | — | — | — | — |
| g1 | fixed | — | 107.17 | 80.86 | 80.74 | — | — | — | — | — |
| g1 | floating | — | 172.95 | 166.45 | 166.58 | — | — | — | — | — |
| h1_2 | fixed | — | 160.13 | 160.41 | 160.06 | — | — | — | — | — |
| h1_2 | floating | — | 309.97 | 309.54 | 309.50 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.48 | 48.69 | 48.94 | — | — | — | — | — |
| iiwa14 | floating | — | 88.38 | 88.91 | 63.31 | — | — | — | — | — |
| go2 | fixed | — | 51.65 | 51.71 | 50.83 | — | — | — | — | — |
| go2 | floating | — | 107.76 | 78.88 | 80.05 | — | — | — | — | — |
| g1 | fixed | — | 221.11 | 145.40 | 145.49 | — | — | — | — | — |
| g1 | floating | — | 375.39 | 274.45 | 280.91 | — | — | — | — | — |
| h1_2 | fixed | — | 327.21 | 324.31 | 326.04 | — | — | — | — | — |
| h1_2 | floating | — | 814.94 | 810.95 | 804.34 | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.38 | 1.41 | 1.37 | — | — | — | — | — |
| iiwa14 | floating | — | 15.62 | 15.78 | 16.89 | — | — | — | — | — |
| go2 | fixed | — | 1.21 | 1.26 | 1.22 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.74 | 6.87 | 6.80 | — | — | — | — | — |
| iiwa14 | floating | — | 21.24 | 21.25 | 21.40 | — | — | — | — | — |
| go2 | fixed | — | 6.92 | 6.87 | 6.85 | — | — | — | — | — |
| go2 | floating | — | 21.00 | 20.72 | 21.13 | — | — | — | — | — |
| g1 | fixed | — | 10.35 | 10.38 | 10.38 | — | — | — | — | — |
| g1 | floating | — | 25.54 | 25.92 | 25.97 | — | — | — | — | — |
| h1_2 | fixed | — | 12.54 | 12.62 | 12.61 | — | — | — | — | — |
| h1_2 | floating | — | 28.16 | 28.59 | 29.52 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.64 | 7.15 | 6.97 | — | — | — | — | — |
| iiwa14 | floating | — | 36.88 | 37.16 | 21.66 | — | — | — | — | — |
| go2 | fixed | — | 7.28 | 7.20 | 7.15 | — | — | — | — | — |
| go2 | floating | — | 36.43 | 36.36 | 21.28 | — | — | — | — | — |
| g1 | fixed | — | 15.52 | 15.94 | 11.04 | — | — | — | — | — |
| g1 | floating | — | 44.64 | 45.40 | 26.60 | — | — | — | — | — |
| h1_2 | fixed | — | 19.73 | 20.00 | 13.50 | — | — | — | — | — |
| h1_2 | floating | — | 50.10 | 50.46 | 30.80 | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.74 | 1.84 | 2.08 | — | — | — | — | — |
| iiwa14 | floating | — | 16.27 | 16.53 | 17.53 | — | — | — | — | — |
| go2 | fixed | — | 1.63 | 1.70 | 2.09 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.37 | 14.36 | 14.44 | — | — | — | — | — |
| iiwa14 | floating | — | 29.32 | 29.31 | 29.75 | — | — | — | — | — |
| go2 | fixed | — | 14.53 | 14.53 | 14.69 | — | — | — | — | — |
| go2 | floating | — | 29.97 | 30.02 | 30.63 | — | — | — | — | — |
| g1 | fixed | — | 19.46 | 19.48 | 19.46 | — | — | — | — | — |
| g1 | floating | — | 35.13 | 35.58 | 36.79 | — | — | — | — | — |
| h1_2 | fixed | — | 26.34 | 26.50 | 27.28 | — | — | — | — | — |
| h1_2 | floating | — | 41.66 | 42.35 | 44.50 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.60 | 14.61 | 14.67 | — | — | — | — | — |
| iiwa14 | floating | — | 45.54 | 45.88 | 30.25 | — | — | — | — | — |
| go2 | fixed | — | 14.96 | 14.93 | 15.10 | — | — | — | — | — |
| go2 | floating | — | 45.77 | 45.88 | 30.42 | — | — | — | — | — |
| g1 | fixed | — | 26.17 | 26.14 | 20.09 | — | — | — | — | — |
| g1 | floating | — | 55.94 | 57.39 | 36.77 | — | — | — | — | — |
| h1_2 | fixed | — | 37.29 | 38.73 | 30.17 | — | — | — | — | — |
| h1_2 | floating | — | 70.93 | 71.82 | 48.26 | — | — | — | — | — |

### EE_POSE_HESSIAN (2nd-order EE Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.82 | 6.47 | — | — | — | — | — | — |
| iiwa14 | floating | — | 38.85 | 42.92 | — | — | — | — | — | — |
| go2 | fixed | — | 4.27 | 4.36 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.94 | 19.01 | 13.75 | — | — | — | — | — |
| iiwa14 | floating | — | 47.68 | 87.15 | 52.57 | — | — | — | — | — |
| go2 | fixed | — | 10.16 | 10.22 | 10.53 | — | — | — | — | — |
| go2 | floating | — | 72.61 | 72.46 | 72.11 | — | — | — | — | — |
| g1 | fixed | — | 63.81 | 56.10 | 70.86 | — | — | — | — | — |
| g1 | floating | — | 370.64 | 384.86 | 282.77 | — | — | — | — | — |
| h1_2 | fixed | — | 269.92 | 497.07 | 337.03 | — | — | — | — | — |
| h1_2 | floating | — | 1088.38 | 2135.21 | 1536.47 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.89 | 39.40 | 15.48 | — | — | — | — | — |
| iiwa14 | floating | — | 87.14 | 251.45 | 66.33 | — | — | — | — | — |
| go2 | fixed | — | 11.31 | 11.28 | 12.42 | — | — | — | — | — |
| go2 | floating | — | 133.04 | 132.64 | 118.28 | — | — | — | — | — |
| g1 | fixed | — | 115.45 | 146.32 | 196.48 | — | — | — | — | — |
| g1 | floating | — | 954.91 | 975.81 | 675.41 | — | — | — | — | — |
| h1_2 | fixed | — | 1317.16 | 1827.18 | 1543.84 | — | — | — | — | — |
| h1_2 | floating | — | 3186.23 | 5568.94 | 4161.71 | — | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 25.98 | 28.20 | — | — | — | — | — | — |
| iiwa14 | floating | — | 366.21 | 376.48 | — | — | — | — | — | — |
| go2 | fixed | — | 36.16 | 39.07 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.01 | 35.38 | 41.88 | — | — | — | — | — |
| iiwa14 | floating | — | 380.43 | 387.03 | 477.04 | — | — | — | — | — |
| go2 | fixed | — | 45.21 | 51.57 | 58.05 | — | — | — | — | — |
| go2 | floating | — | 564.11 | 602.10 | 673.77 | — | — | — | — | — |
| g1 | fixed | — | 1337.22 | 3726.77 | 3396.73 | — | — | — | — | — |
| g1 | floating | — | 1491.29 | 1693.68 | 1820.88 | — | — | — | — | — |
| h1_2 | fixed | — | 9751.45 | 20953.96 | 18497.66 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.28 | 43.22 | 56.58 | — | — | — | — | — |
| iiwa14 | floating | — | 751.24 | 770.26 | 520.93 | — | — | — | — | — |
| go2 | fixed | — | 87.75 | 70.17 | 80.32 | — | — | — | — | — |
| go2 | floating | — | 1133.04 | 1219.56 | 744.72 | — | — | — | — | — |
| g1 | fixed | — | 2995.48 | 6170.65 | 5521.87 | — | — | — | — | — |
| g1 | floating | — | 3280.82 | 3666.71 | 2593.35 | — | — | — | — | — |
| h1_2 | fixed | — | 21680.48 | 29478.80 | 25988.57 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.02 | 28.45 | — | — | — | — | — | — |
| iiwa14 | floating | — | 2633.17 | 2870.52 | — | — | — | — | — | — |
| go2 | fixed | — | 36.20 | 39.29 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.02 | 35.29 | 41.85 | — | — | — | — | — |
| iiwa14 | floating | — | 2692.13 | 3634.45 | 3109.23 | — | — | — | — | — |
| go2 | fixed | — | 44.99 | 51.58 | 58.02 | — | — | — | — | — |
| go2 | floating | — | 3946.92 | 4322.87 | 4815.09 | — | — | — | — | — |
| g1 | fixed | — | 1337.49 | 3726.20 | 3396.20 | — | — | — | — | — |
| g1 | floating | — | 29682.39 | 30955.81 | 35513.00 | — | — | — | — | — |
| h1_2 | fixed | — | 9750.02 | 20953.90 | 18488.58 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.29 | 43.22 | 56.60 | — | — | — | — | — |
| iiwa14 | floating | — | 5330.59 | 7154.59 | 6160.70 | — | — | — | — | — |
| go2 | fixed | — | 87.76 | 70.35 | 80.21 | — | — | — | — | — |
| go2 | floating | — | 7762.22 | 8471.44 | 9446.68 | — | — | — | — | — |
| g1 | fixed | — | 2994.99 | 6171.61 | 5530.41 | — | — | — | — | — |
| g1 | floating | — | 74672.13 | 78828.66 | 86091.38 | — | — | — | — | — |
| h1_2 | fixed | — | 21675.71 | 29480.35 | 25987.39 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 233.62 | 243.72 | — | — | — | — | — | — |
| iiwa14 | floating | — | 366.18 | 376.47 | — | — | — | — | — | — |
| go2 | fixed | — | 378.23 | 393.91 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 240.67 | 254.83 | 278.82 | — | — | — | — | — |
| iiwa14 | floating | — | 380.41 | 387.02 | 477.17 | — | — | — | — | — |
| go2 | fixed | — | 389.76 | 403.39 | 463.10 | — | — | — | — | — |
| go2 | floating | — | 564.09 | 601.95 | 673.87 | — | — | — | — | — |
| g1 | fixed | — | 1245.85 | 1231.15 | 1344.22 | — | — | — | — | — |
| g1 | floating | — | 1490.30 | 1693.29 | 1820.64 | — | — | — | — | — |
| h1_2 | fixed | — | 2446.99 | 2495.97 | 3610.45 | — | — | — | — | — |
| h1_2 | floating | — | 2961.77 | 2330.45 | 4700.15 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 476.92 | 275.09 | 295.29 | — | — | — | — | — |
| iiwa14 | floating | — | 751.26 | 770.16 | 521.04 | — | — | — | — | — |
| go2 | fixed | — | 772.38 | 425.56 | 480.80 | — | — | — | — | — |
| go2 | floating | — | 1133.19 | 1219.18 | 744.67 | — | — | — | — | — |
| g1 | fixed | — | 2551.82 | 2588.02 | 1752.82 | — | — | — | — | — |
| g1 | floating | — | 3280.03 | 3665.08 | 2593.16 | — | — | — | — | — |
| h1_2 | fixed | — | 6227.44 | 6343.95 | 5252.23 | — | — | — | — | — |
| h1_2 | floating | — | 7472.72 | 6183.03 | 7539.18 | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 49.65 | 74.25 | — | — | — | — | — | — |
| iiwa14 | floating | — | 520.25 | 657.66 | — | — | — | — | — | — |
| go2 | fixed | — | 67.49 | 95.17 | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 65.84 | 91.99 | 87.94 | — | — | — | — | — |
| iiwa14 | floating | — | 515.21 | 701.97 | 765.23 | — | — | — | — | — |
| go2 | fixed | — | 85.14 | 112.42 | 117.72 | — | — | — | — | — |
| go2 | floating | — | 707.55 | 911.17 | 1148.99 | — | — | — | — | — |
| g1 | fixed | — | 3660.42 | 8510.14 | 7585.23 | — | — | — | — | — |
| g1 | floating | — | 7478.39 | 5598.37 | 8320.78 | — | — | — | — | — |
| h1_2 | fixed | — | 28580.56 | 47417.30 | 45528.73 | — | — | — | — | — |
| h1_2 | floating | — | 33078.45 | 33298.67 | 33294.84 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_lite | glass_min | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:----------:|:---------:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 115.48 | 103.35 | 108.53 | — | — | — | — | — |
| iiwa14 | floating | — | 1010.12 | 1385.33 | 847.71 | — | — | — | — | — |
| go2 | fixed | — | 156.22 | 138.08 | 173.70 | — | — | — | — | — |
| go2 | floating | — | 1443.88 | 1861.12 | 1321.99 | — | — | — | — | — |
| g1 | fixed | — | 9261.78 | 13342.63 | 12185.94 | — | — | — | — | — |
| g1 | floating | — | 19575.59 | 13294.97 | 18584.11 | — | — | — | — | — |
| h1_2 | fixed | — | 81179.49 | 97333.93 | 94067.57 | — | — | — | — | — |
| h1_2 | floating | — | 165802.06 | 156408.73 | 156703.11 | — | — | — | — | — |

