# GRiD Multi-Version Benchmark Comparison

**Machine**: brian-2021Desktop  
**GPU**: NVIDIA GeForce RTX 3080 (cc 8.6, CUDA 12.6)  
**CPU**: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz  
**Date**: 2026-05-14  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: 3.9.0

All times in **µs**.

Columns:
- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only (pre_glass harness does not support floating-base).
- **glass**: GRiD HEAD with the pure-SIMT GLASS v2 backend.
- **glass_nv**: GRiD HEAD with the cuBLASDx-backed GLASS v2 backend.
- **pin**: Pinocchio CPU reference (codegen where available).
- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only (id / fd / ee_pose / id_du); others render `—`.
- **frax**: Frax (JAX) GPU reference (https://github.com/danielpmorton/frax). Subset of algos only (id / fd / crba / minv); others render `—`.
- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; **< 1.00× = HEAD regressed**.
- **glass_nv/glass**: N=256 compute-only ratio. **> 1.00× = cuBLASDx is faster**.

Each algorithm gets three sub-tables: **single-call**, **batch N=16**, **batch N=256**. Same 6 backend columns + ratios in each. Values are median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: No Pinocchio equivalent — GRiD numbers only.

## Core Dynamics

### ID (Inverse Dynamics)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 4.70 | 4.92 | 4.92 | 0.27 (codegen) | — | — | 1.04× | 1.00× |
| iiwa14 | floating | — | 23.98 | 23.98 | 0.36 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 4.76 | 6.17 | 6.17 | 0.33 (codegen) | — | — | 1.01× | 1.00× |
| go2 | floating | — | 29.33 | 29.32 | 0.76 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 14.06 | 17.90 | 18.26 | 1.13 (codegen) | — | — | 0.85× | 1.00× |
| g1 | floating | — | 52.68 | 52.66 | 1.32 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 9.61 | 8.58 | 8.59 | 11.91 | — | — | 1.12× | 1.00× |
| iiwa14 | floating | — | 28.63 | 28.69 | 13.17 | — | — | — | 1.00× |
| go2 | fixed | 9.80 | 9.79 | 9.90 | 12.65 | — | — | 1.00× | 0.99× |
| go2 | floating | — | 34.20 | 34.34 | 12.94 | — | — | — | 1.00× |
| g1 | fixed | 18.47 | 21.07 | 21.17 | 19.56 | — | — | 0.88× | 1.00× |
| g1 | floating | — | 57.93 | 58.05 | 17.44 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 12.13 | 11.66 | 11.69 | 47.62 | — | — | 1.04× | 1.00× |
| iiwa14 | floating | — | 101.78 | 101.93 | 53.53 | — | — | — | 1.00× |
| go2 | fixed | 14.96 | 14.81 | 14.88 | 70.25 | — | — | 1.01× | 1.00× |
| go2 | floating | — | 123.39 | 123.50 | 76.84 | — | — | — | 1.00× |
| g1 | fixed | 41.70 | 48.85 | 48.96 | 159.02 | — | — | 0.85× | 1.00× |
| g1 | floating | — | 220.62 | 220.30 | 130.13 | — | — | — | 1.00× |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 8.17 | 6.85 | 6.73 | 0.49 (codegen) | — | — | 1.70× | 1.00× |
| iiwa14 | floating | — | 41.93 | 41.90 | 1.09 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 8.51 | 7.75 | 7.38 | 0.54 (codegen) | — | — | 1.16× | 1.00× |
| go2 | floating | — | 45.77 | 45.77 | 1.61 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 26.47 | 21.61 | 23.31 | 2.91 (codegen) | — | — | 1.13× | 1.00× |
| g1 | floating | — | 73.92 | 73.92 | 6.15 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 12.43 | 10.49 | 10.50 | 12.59 | — | — | 1.18× | 1.00× |
| iiwa14 | floating | — | 46.72 | 46.79 | 16.85 | — | — | — | 1.00× |
| go2 | fixed | 12.94 | 11.68 | 11.67 | 14.69 | — | — | 1.11× | 1.00× |
| go2 | floating | — | 50.58 | 50.75 | 21.33 | — | — | — | 1.00× |
| g1 | fixed | 30.14 | 26.54 | 26.56 | 29.10 | — | — | 1.14× | 1.00× |
| g1 | floating | — | 79.13 | 79.25 | 55.44 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 22.89 | 13.49 | 13.47 | 61.39 | — | — | 1.70× | 1.00× |
| iiwa14 | floating | — | 173.58 | 173.99 | 120.52 | — | — | — | 1.00× |
| go2 | fixed | 19.31 | 16.70 | 16.77 | 81.76 | — | — | 1.16× | 1.00× |
| go2 | floating | — | 189.12 | 189.25 | 121.99 | — | — | — | 1.00× |
| g1 | fixed | 67.23 | 59.31 | 59.44 | 356.05 | — | — | 1.13× | 1.00× |
| g1 | floating | — | 306.13 | 306.12 | 593.87 | — | — | — | 1.00× |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 10.61 | 10.00 | 9.14 | 0.90 (codegen) | — | — | 1.07× | 1.52× |
| iiwa14 | floating | — | 46.11 | 46.11 | 1.63 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 9.98 | 10.70 | 9.90 | 1.02 (codegen) | — | — | 1.01× | 1.00× |
| go2 | floating | — | 53.24 | 53.24 | 2.63 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 33.41 | 33.01 | 33.49 | 5.18 (codegen) | — | — | 0.94× | 1.00× |
| g1 | floating | — | 90.48 | 90.48 | 9.70 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 14.64 | 13.47 | 13.39 | 14.80 | — | — | 1.09× | 1.01× |
| iiwa14 | floating | — | 50.70 | 50.73 | 17.92 | — | — | — | 1.00× |
| go2 | fixed | 14.53 | 14.77 | 14.88 | 14.85 | — | — | 0.98× | 0.99× |
| go2 | floating | — | 57.80 | 57.98 | 22.28 | — | — | — | 1.00× |
| g1 | fixed | 35.84 | 37.70 | 37.96 | 32.12 | — | — | 0.95× | 0.99× |
| g1 | floating | — | 96.01 | 95.92 | 52.05 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 27.91 | 26.02 | 17.13 | 63.16 | — | — | 1.07× | 1.52× |
| iiwa14 | floating | — | 189.81 | 190.22 | 106.76 | — | — | — | 1.00× |
| go2 | fixed | 20.83 | 20.59 | 20.65 | 65.18 | — | — | 1.01× | 1.00× |
| go2 | floating | — | 218.31 | 218.41 | 165.12 | — | — | — | 1.00× |
| g1 | fixed | 79.20 | 83.81 | 84.06 | 323.40 | — | — | 0.94× | 1.00× |
| g1 | floating | — | 372.82 | 372.91 | 593.18 | — | — | — | 1.00× |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 11.63 | 8.63 | 8.73 | 0.64 (codegen) | — | — | 1.13× | 1.00× |
| iiwa14 | floating | — | 64.52 | 64.52 | 1.02 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 9.61 | 9.93 | 9.62 | 0.77 (codegen) | — | — | 0.99× | 1.00× |
| go2 | floating | — | 76.16 | 76.16 | 1.62 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 29.78 | 28.33 | 28.35 | 3.02 (codegen) | — | — | 0.94× | 1.00× |
| g1 | floating | — | 118.59 | 118.56 | 3.98 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 15.81 | 12.90 | 12.84 | 16.19 | — | — | 1.23× | 1.00× |
| iiwa14 | floating | — | 68.97 | 68.98 | 19.04 | — | — | — | 1.00× |
| go2 | fixed | 14.33 | 14.49 | 14.54 | 19.53 | — | — | 0.99× | 1.00× |
| go2 | floating | — | 80.71 | 80.86 | 23.55 | — | — | — | 1.00× |
| g1 | fixed | 33.33 | 34.27 | 34.32 | 33.81 | — | — | 0.97× | 1.00× |
| g1 | floating | — | 123.67 | 124.81 | 36.44 | — | — | — | 0.99× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 20.04 | 17.74 | 17.67 | 98.57 | — | — | 1.13× | 1.00× |
| iiwa14 | floating | — | 263.64 | 264.08 | 141.65 | — | — | — | 1.00× |
| go2 | fixed | 19.84 | 20.07 | 20.16 | 154.37 | — | — | 0.99× | 1.00× |
| go2 | floating | — | 310.35 | 310.50 | 213.49 | — | — | — | 1.00× |
| g1 | fixed | 71.71 | 76.49 | 76.59 | 367.82 | — | — | 0.94× | 1.00× |
| g1 | floating | — | 486.04 | 485.63 | 366.68 | — | — | — | 1.00× |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 5.93 | 4.45 | 4.48 | 0.32 (codegen) | — | — | 1.18× | 1.00× |
| iiwa14 | floating | — | 28.85 | 28.85 | 0.58 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 7.20 | 5.00 | 4.99 | 0.40 (codegen) | — | — | 1.24× | 0.99× |
| go2 | floating | — | 32.78 | 32.81 | 0.89 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 22.34 | 15.79 | 15.82 | 1.76 (codegen) | — | — | 1.26× | 1.00× |
| g1 | floating | — | 78.26 | 78.25 | 2.50 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 10.56 | 9.11 | 8.91 | 12.51 | — | — | 1.16× | 1.02× |
| iiwa14 | floating | — | 33.50 | 33.94 | 12.19 | — | — | — | 0.99× |
| go2 | fixed | 11.98 | 9.67 | 9.77 | 11.68 | — | — | 1.24× | 0.99× |
| go2 | floating | — | 37.58 | 37.76 | 31.92 | — | — | — | 0.99× |
| g1 | fixed | 26.27 | 20.62 | 20.76 | 15.92 | — | — | 1.27× | 0.99× |
| g1 | floating | — | 83.55 | 83.91 | 18.16 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 13.65 | 11.58 | 11.63 | 42.35 | — | — | 1.18× | 1.00× |
| iiwa14 | floating | — | 121.29 | 121.40 | 49.61 | — | — | — | 1.00× |
| go2 | fixed | 17.69 | 14.24 | 14.37 | 56.73 | — | — | 1.24× | 0.99× |
| go2 | floating | — | 137.25 | 137.38 | 72.87 | — | — | — | 1.00× |
| g1 | fixed | 58.29 | 46.27 | 46.27 | 157.86 | — | — | 1.26× | 1.00× |
| g1 | floating | — | 323.84 | 323.87 | 176.26 | — | — | — | 1.00× |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 7.58 | 7.47 | 7.47 | 1.41 (codegen) | — | — | 1.08× | 1.00× |
| iiwa14 | floating | — | 32.11 | 32.11 | 2.70 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 7.45 | 8.38 | 8.38 | 1.60 (codegen) | — | — | 1.06× | 1.00× |
| go2 | floating | — | 41.83 | 41.96 | 4.06 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 26.56 | 29.63 | 29.62 | 8.98 (codegen) | — | — | 0.98× | 1.00× |
| g1 | floating | — | 121.87 | 122.02 | 13.84 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 13.03 | 11.83 | 11.80 | 18.71 | — | — | 1.10× | 1.00× |
| iiwa14 | floating | — | 36.80 | 36.86 | 21.99 | — | — | — | 1.00× |
| go2 | fixed | 13.41 | 12.97 | 13.00 | 23.25 | — | — | 1.03× | 1.00× |
| go2 | floating | — | 46.77 | 46.91 | 29.45 | — | — | — | 1.00× |
| g1 | fixed | 33.24 | 33.09 | 33.14 | 46.44 | — | — | 1.00× | 1.00× |
| g1 | floating | — | 127.21 | 127.47 | 53.37 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 18.28 | 16.89 | 16.90 | 138.77 | — | — | 1.08× | 1.00× |
| iiwa14 | floating | — | 134.30 | 134.40 | 183.97 | — | — | — | 1.00× |
| go2 | fixed | 21.14 | 19.98 | 20.07 | 211.11 | — | — | 1.06× | 1.00× |
| go2 | floating | — | 174.36 | 174.48 | 274.80 | — | — | — | 1.00× |
| g1 | fixed | 114.86 | 117.31 | 117.42 | 437.91 | — | — | 0.98× | 1.00× |
| g1 | floating | — | 637.94 | 634.93 | 466.04 | — | — | — | 1.00× |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 15.81 | 15.74 | 15.92 | 3.17 (codegen) | — | — | 1.03× | 1.00× |
| iiwa14 | floating | — | 60.05 | 60.04 | 5.13 (codegen) | — | — | — | 1.00× |
| go2 | fixed | 13.97 | 16.85 | 16.91 | 3.18 (codegen) | — | — | 0.97× | 1.00× |
| go2 | floating | — | 74.09 | 74.01 | 9.13 (codegen) | — | — | — | 1.00× |
| g1 | fixed | 55.80 | 63.36 | 62.30 | 19.91 (codegen) | — | — | 0.92× | 1.01× |
| g1 | floating | — | 182.67 | 182.79 | 31.02 (codegen) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 21.23 | 19.79 | 20.01 | 25.16 | — | — | 1.07× | 0.99× |
| iiwa14 | floating | — | 64.34 | 64.51 | 33.36 | — | — | — | 1.00× |
| go2 | fixed | 19.99 | 21.11 | 21.31 | 32.44 | — | — | 0.95× | 0.99× |
| go2 | floating | — | 78.53 | 78.66 | 49.94 | — | — | — | 1.00× |
| g1 | fixed | 61.96 | 65.85 | 65.33 | 90.18 | — | — | 0.94× | 1.01× |
| g1 | floating | — | 188.03 | 188.27 | 130.88 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 28.16 | 27.36 | 27.35 | 216.64 | — | — | 1.03× | 1.00× |
| iiwa14 | floating | — | 245.26 | 245.33 | 374.94 | — | — | — | 1.00× |
| go2 | fixed | 29.70 | 30.46 | 30.61 | 337.33 | — | — | 0.97× | 1.00× |
| go2 | floating | — | 302.13 | 302.14 | 378.71 | — | — | — | 1.00× |
| g1 | fixed | 231.18 | 251.44 | 248.70 | 790.69 | — | — | 0.92× | 1.01× |
| g1 | floating | — | 981.17 | 978.25 | 1128.53 | — | — | — | 1.00× |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 2.59 | 1.73 | 1.73 | 0.43 (direct) | — | — | 1.15× | 1.00× |
| iiwa14 | floating | — | 71.00 | 71.00 | 0.48 (direct) | — | — | — | 1.00× |
| go2 | fixed | 2.39 | 1.54 | 1.53 | 1.20 (direct) | — | — | 1.17× | 0.98× |
| go2 | floating | — | 70.96 | 70.96 | 1.29 (direct) | — | — | — | 1.00× |
| g1 | fixed | 4.85 | 3.55 | 3.55 | 1.78 (direct) | — | — | 1.17× | 0.99× |
| g1 | floating | — | 76.36 | 76.36 | 1.91 (direct) | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 7.17 | 5.99 | 5.97 | 12.73 | — | — | 1.20× | 1.00× |
| iiwa14 | floating | — | 75.88 | 75.59 | 13.09 | — | — | — | 1.00× |
| go2 | fixed | 7.01 | 5.96 | 6.06 | 12.83 | — | — | 1.17× | 0.98× |
| go2 | floating | — | 75.50 | 75.64 | 12.81 | — | — | — | 1.00× |
| g1 | fixed | 9.86 | 8.00 | 8.10 | 16.73 | — | — | 1.23× | 0.99× |
| g1 | floating | — | 81.07 | 81.30 | 15.91 | — | — | — | 1.00× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 7.45 | 6.48 | 6.47 | 32.23 | — | — | 1.15× | 1.00× |
| iiwa14 | floating | — | 289.65 | 290.04 | 33.40 | — | — | — | 1.00× |
| go2 | fixed | 7.58 | 6.49 | 6.65 | 66.82 | — | — | 1.17× | 0.98× |
| go2 | floating | — | 289.52 | 289.61 | 68.74 | — | — | — | 1.00× |
| g1 | fixed | 16.35 | 13.92 | 14.01 | 102.31 | — | — | 1.17× | 0.99× |
| g1 | floating | — | 314.40 | 314.42 | 73.00 | — | — | — | 1.00× |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 2.91 | 2.17 | 2.17 | 0.41 (direct) | — | — | 1.16× | 1.00× |
| iiwa14 | floating | — | 0.00 | 0.00 | 0.56 (direct) | — | — | — | 0.96× |
| go2 | fixed | 3.78 | 2.86 | 13.87 | 0.56 (direct) | — | — | 1.18× | 0.25× |
| go2 | floating | — | 0.00 | 0.09 | 0.68 (direct) | — | — | — | 0.88× |
| g1 | fixed | 15.85 | 12.94 | 137.00 | 1.35 (direct) | — | — | 1.18× | 0.08× |
| g1 | floating | — | 0.00 | 0.09 | 1.50 (direct) | — | — | — | 0.83× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 7.46 | 6.47 | 6.43 | 12.77 | — | — | 1.15× | 1.01× |
| iiwa14 | floating | — | 0.66 | 0.64 | 13.41 | — | — | — | 1.03× |
| go2 | fixed | 8.42 | 7.33 | 18.72 | 11.95 | — | — | 1.15× | 0.39× |
| go2 | floating | — | 0.64 | 0.76 | 13.21 | — | — | — | 0.83× |
| g1 | fixed | 20.58 | 17.44 | 140.58 | 16.33 | — | — | 1.18× | 0.12× |
| g1 | floating | — | 0.68 | 0.80 | 15.77 | — | — | — | 0.84× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 9.53 | 8.22 | 8.21 | 30.90 | — | — | 1.16× | 1.00× |
| iiwa14 | floating | — | 0.65 | 0.67 | 36.98 | — | — | — | 0.96× |
| go2 | fixed | 13.22 | 11.17 | 44.93 | 37.88 | — | — | 1.18× | 0.25× |
| go2 | floating | — | 0.66 | 0.76 | 46.02 | — | — | — | 0.88× |
| g1 | fixed | 56.21 | 47.63 | 585.54 | 84.98 | — | — | 1.18× | 0.08× |
| g1 | floating | — | 0.66 | 0.80 | 82.21 | — | — | — | 0.83× |

## Second-Order

### IDSVA_SO (2nd-order ID)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 10.82 (direct) | — | — | — | — |
| iiwa14 | floating | — | — | — | 31.94 (direct) | — | — | — | — |
| go2 | fixed | — | — | — | 14.06 (direct) | — | — | — | — |
| go2 | floating | — | — | — | 42.01 (direct) | — | — | — | — |
| g1 | fixed | — | — | — | 68.15 (direct) | — | — | — | — |
| g1 | floating | — | — | — | 140.86 (direct) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 58.19 | — | — | — | — |
| iiwa14 | floating | — | — | — | 137.91 | — | — | — | — |
| go2 | fixed | — | — | — | 68.50 | — | — | — | — |
| go2 | floating | — | — | — | 170.12 | — | — | — | — |
| g1 | fixed | — | — | — | 330.70 | — | — | — | — |
| g1 | floating | — | — | — | 678.88 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 601.81 | — | — | — | — |
| iiwa14 | floating | — | — | — | 1863.54 | — | — | — | — |
| go2 | fixed | — | — | — | 745.10 | — | — | — | — |
| go2 | floating | — | — | — | 2428.43 | — | — | — | — |
| g1 | fixed | — | — | — | 3080.84 | — | — | — | — |
| g1 | floating | — | — | — | 6578.98 | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

