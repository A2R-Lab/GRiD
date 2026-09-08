# GRiD Multi-Version Benchmark Comparison

**Machine**: ?  
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (cc 12.0, CUDA 12.9) (cc 12.0, CUDA ?)  
**CPU**: AMD Ryzen AI 9 HX 370 w/ Radeon 890M  
**Date**: 2026-05-13  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: ?

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
| iiwa14 | fixed | 13.81 | 4.93 | 5.12 | — | 3206.88 | 1976.66 | 2.69× | 1.11× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 27.26 | 7.71 | 7.91 | — | — | 2272.46 | 1.28× | 0.70× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 46.41 | 25.30 | 25.53 | — | — | 3065.15 | 1.88× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 45.82 | 60.91 | 55.55 | — | 3656.95 | 1297.82 | 0.75× | 1.10× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 73.40 | 59.70 | 72.39 | — | — | 2315.28 | 1.23× | 0.82× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 60.25 | 79.43 | 76.52 | — | — | 3467.72 | 0.76× | 1.04× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 178.23 | 66.28 | 59.61 | — | 4466.00 | 1678.02 | 2.69× | 1.11× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 84.23 | 65.66 | 93.63 | — | — | 2318.50 | 1.28× | 0.70× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 308.15 | 164.10 | 167.56 | — | — | 4088.73 | 1.88× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 26.81 | 6.81 | 6.74 | — | — | 1689.60 | 0.82× | 0.98× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 24.69 | 8.86 | 8.79 | — | — | 2633.10 | 1.24× | 0.65× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 25.31 | 0.01 | 0.01 | — | — | 3384.69 | 6.13× | 0.63× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 76.27 | 74.29 | 74.47 | — | — | 1837.41 | 1.03× | 1.00× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 53.86 | 103.64 | 78.98 | — | — | 2871.15 | 0.52× | 1.31× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 4.80 | 2.13 | 2.85 | — | — | 3874.64 | 2.25× | 0.75× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 83.45 | 102.34 | 104.51 | — | — | 1964.70 | 0.82× | 0.98× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 175.10 | 141.53 | 217.21 | — | — | 2823.35 | 1.24× | 0.65× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 13.19 | 2.15 | 3.39 | — | — | 3548.30 | 6.13× | 0.63× |
| g1 | floating | — | — | — | — | — | — | — | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 33.90 | 9.37 | 9.30 | — | 6564.44 | 1070.71 | 0.84× | 1.01× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 39.38 | 11.93 | 15.38 | — | — | 2477.59 | 1.06× | 0.72× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 30.59 | 0.01 | 0.01 | — | — | 3211.61 | 8.33× | 1.10× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 81.05 | 62.98 | 60.96 | — | 10576.52 | 1739.06 | 1.29× | 1.03× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 144.78 | 66.94 | 81.41 | — | — | 2406.66 | 2.16× | 0.82× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 3.86 | 1.95 | 1.97 | — | — | 4558.69 | 1.98× | 0.99× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 104.64 | 124.01 | 122.52 | — | 23713.80 | 1874.38 | 0.84× | 1.01× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 194.33 | 183.75 | 256.72 | — | — | 2724.00 | 1.06× | 0.72× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 20.37 | 2.45 | 2.22 | — | — | 4461.73 | 8.33× | 1.10× |
| g1 | floating | — | — | — | — | — | — | — | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 75.88 | 8.84 | 8.75 | — | — | — | 0.96× | 0.93× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 45.41 | 11.44 | 18.25 | — | — | — | 1.94× | 0.70× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 26.11 | 0.04 | 0.04 | — | — | — | 3.63× | 1.04× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 66.42 | 66.87 | 130.84 | — | — | — | 0.99× | 0.51× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 55.50 | 70.38 | 74.85 | — | — | — | 0.79× | 0.94× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 4.65 | 2.37 | 2.69 | — | — | — | 1.96× | 0.88× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 82.82 | 86.50 | 92.59 | — | — | — | 0.96× | 0.93× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 315.17 | 162.22 | 233.29 | — | — | — | 1.94× | 0.70× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 13.11 | 3.61 | 3.47 | — | — | — | 3.63× | 1.04× |
| g1 | floating | — | — | — | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 25.51 | 4.77 | 4.73 | — | — | 1249.28 | 0.70× | 1.01× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 32.61 | 7.02 | 11.21 | — | — | 2300.47 | 3.02× | 0.96× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 56.12 | 28.25 | 28.68 | — | — | 3152.06 | 2.23× | 0.96× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 77.37 | 57.59 | 54.53 | — | — | 1008.62 | 1.34× | 1.06× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 54.34 | 60.48 | 72.94 | — | — | 2228.10 | 0.90× | 0.83× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 76.17 | 100.83 | 82.36 | — | — | 3669.76 | 0.76× | 1.22× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 52.84 | 75.30 | 74.29 | — | — | 1072.32 | 0.70× | 1.01× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 206.34 | 68.27 | 70.80 | — | — | 2054.97 | 3.02× | 0.96× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 356.27 | 160.03 | 166.44 | — | — | 3178.67 | 2.23× | 0.96× |
| g1 | floating | — | — | — | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 10.07 | 7.79 | 7.73 | — | 2760.88 | — | 0.75× | 0.97× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 12.54 | 10.47 | 16.73 | — | — | — | 1.37× | 0.95× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 40.01 | 35.31 | 35.85 | — | — | — | 0.87× | 0.99× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 75.42 | 68.88 | 57.29 | — | 3417.78 | — | 1.09× | 1.20× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 52.76 | 62.36 | 85.03 | — | — | — | 0.85× | 0.73× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 72.08 | 93.20 | 95.42 | — | — | — | 0.77× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 89.60 | 120.25 | 124.52 | — | 12135.01 | — | 0.75× | 0.97× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 185.49 | 135.52 | 142.48 | — | — | — | 1.37× | 0.95× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 277.74 | 318.90 | 322.64 | — | — | — | 0.87× | 0.99× |
| g1 | floating | — | — | — | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 0.83 | 0.00 | 0.00 | — | — | — | 1.29× | 1.00× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 1.85 | 0.00 | 0.00 | — | — | — | 1.32× | 1.14× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 0.89 | 0.00 | 0.00 | — | — | — | 1.40× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 10.08 | 2.33 | 3.21 | — | — | — | 4.32× | 0.73× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 17.92 | 1.96 | 3.31 | — | — | — | 9.12× | 0.59× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 12.85 | 2.12 | 2.35 | — | — | — | 6.05× | 0.90× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 2.64 | 2.06 | 2.06 | — | — | — | 1.29× | 1.00× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 4.52 | 3.41 | 3.00 | — | — | — | 1.32× | 1.14× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 2.83 | 2.03 | 2.06 | — | — | — | 1.40× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 17.23 | 1.15 | 1.14 | — | 3248.20 | — | 0.64× | 1.13× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 20.13 | 1.03 | 1.64 | — | — | — | 2.44× | 0.97× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 12.48 | 4.75 | 4.83 | — | — | — | 0.86× | 0.88× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 41.67 | 57.37 | 50.32 | — | 4179.85 | — | 0.73× | 1.14× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 275.59 | 51.73 | 51.01 | — | — | — | 5.33× | 1.01× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 46.17 | 63.70 | 59.91 | — | — | — | 0.72× | 1.06× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 40.65 | 63.29 | 55.77 | — | 4238.12 | — | 0.64× | 1.13× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 141.55 | 58.08 | 59.88 | — | — | — | 2.44× | 0.97× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 84.34 | 98.30 | 112.17 | — | — | — | 0.86× | 0.88× |
| g1 | floating | — | — | — | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 8.19 | 1.34 | 1.33 | — | — | — | 0.64× | 1.19× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 81.63 | 1.77 | 2.82 | — | — | — | 2.77× | 1.19× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 19.99 | 12.02 | 12.21 | — | — | — | 0.88× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 41.66 | 85.79 | 45.25 | — | — | — | 0.49× | 1.90× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 208.06 | 52.63 | 57.31 | — | — | — | 3.95× | 0.92× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 52.12 | 72.07 | 68.58 | — | — | — | 0.72× | 1.05× |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 40.39 | 63.19 | 53.13 | — | — | — | 0.64× | 1.19× |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | 184.94 | 66.82 | 56.32 | — | — | — | 2.77× | 1.19× |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | 137.37 | 156.14 | 159.63 | — | — | — | 0.88× | 0.98× |
| g1 | floating | — | — | — | — | — | — | — | — |

## Second-Order

### IDSVA_SO (2nd-order ID)

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

