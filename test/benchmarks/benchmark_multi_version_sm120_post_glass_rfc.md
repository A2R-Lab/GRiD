# GRiD Multi-Version Benchmark Comparison

**Machine**: ?  
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (cc 12.0, CUDA 12.9) (cc 12.0, CUDA ?)  
**CPU**: AMD Ryzen AI 9 HX 370 (12 physical cores)  
**Date**: 2026-05-14  
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
| iiwa14 | fixed | 13.50 | 7.41 | 5.02 | 0.41 (codegen) | — | — | 0.68× | 1.02× |
| iiwa14 | floating | — | 10.22 | 10.19 | 0.58 (codegen) | — | — | — | 1.09× |
| go2 | fixed | 23.40 | 7.53 | 7.62 | 0.54 (codegen) | — | — | 1.16× | 0.69× |
| go2 | floating | — | 13.73 | 13.85 | 1.40 (codegen) | — | — | — | 1.01× |
| g1 | fixed | 45.89 | 24.40 | 24.44 | 2.96 (codegen) | — | — | 1.73× | 0.69× |
| g1 | floating | — | 31.15 | 31.17 | — | — | — | — | 0.90× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 64.04 | 56.67 | 48.14 | 273.95 | — | — | 1.13× | 1.18× |
| iiwa14 | floating | — | 59.20 | 60.81 | 356.00 | — | — | — | 0.97× |
| go2 | fixed | 138.49 | 50.97 | 66.72 | 325.69 | — | — | 2.72× | 0.76× |
| go2 | floating | — | 66.76 | 60.47 | 284.24 | — | — | — | 1.10× |
| g1 | fixed | 76.12 | 75.92 | 74.51 | 337.66 | — | — | 1.00× | 1.02× |
| g1 | floating | — | 94.08 | 83.34 | — | — | — | — | 1.13× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 70.45 | 103.68 | 101.42 | 784.90 | — | — | 0.68× | 1.02× |
| iiwa14 | floating | — | 125.29 | 115.17 | 887.65 | — | — | — | 1.09× |
| go2 | fixed | 88.28 | 76.40 | 110.20 | 930.48 | — | — | 1.16× | 0.69× |
| go2 | floating | — | 146.82 | 145.64 | 956.72 | — | — | — | 1.01× |
| g1 | fixed | 240.25 | 138.87 | 200.62 | 1054.46 | — | — | 1.73× | 0.69× |
| g1 | floating | — | 261.59 | 291.74 | — | — | — | — | 0.90× |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 23.52 | 10.39 | 6.52 | 1.10 (codegen) | — | — | 1.04× | 1.44× |
| iiwa14 | floating | — | 0.00 | 0.00 | 2.74 (codegen) | — | — | — | 0.99× |
| go2 | fixed | 29.23 | 8.41 | 8.41 | 0.93 (codegen) | — | — | 0.99× | 1.01× |
| go2 | floating | — | 32.57 | 32.75 | 4.56 (codegen) | — | — | — | 1.01× |
| g1 | fixed | 26.61 | 0.00 | 0.00 | 7.03 (codegen) | — | — | 0.92× | 1.53× |
| g1 | floating | — | 0.00 | 0.00 | — | — | — | — | 0.86× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 55.19 | 75.66 | 53.77 | 203.34 | — | — | 0.73× | 1.41× |
| iiwa14 | floating | — | 1.95 | 2.48 | 404.78 | — | — | — | 0.78× |
| go2 | fixed | 51.06 | 55.83 | 66.31 | 246.24 | — | — | 0.91× | 0.84× |
| go2 | floating | — | 85.07 | 89.22 | 393.88 | — | — | — | 0.95× |
| g1 | fixed | 7.46 | 2.10 | 3.19 | 473.59 | — | — | 3.55× | 0.66× |
| g1 | floating | — | 2.28 | 5.30 | — | — | — | — | 0.43× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 179.29 | 171.74 | 119.06 | 811.84 | — | — | 1.04× | 1.44× |
| iiwa14 | floating | — | 1.97 | 2.00 | 976.99 | — | — | — | 0.99× |
| go2 | fixed | 110.11 | 111.70 | 111.11 | 885.86 | — | — | 0.99× | 1.01× |
| go2 | floating | — | 286.37 | 284.20 | 1062.60 | — | — | — | 1.01× |
| g1 | fixed | 3.01 | 3.27 | 2.13 | 1225.17 | — | — | 0.92× | 1.53× |
| g1 | floating | — | 1.85 | 2.15 | — | — | — | — | 0.86× |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 22.72 | 14.36 | 9.03 | 2.37 (codegen) | — | — | 0.60× | 1.72× |
| iiwa14 | floating | — | 34.21 | 34.16 | 4.12 (codegen) | — | — | — | 1.08× |
| go2 | fixed | 35.12 | 11.29 | 11.28 | 2.02 (codegen) | — | — | 0.83× | 1.01× |
| go2 | floating | — | 0.00 | 0.00 | 5.99 (codegen) | — | — | — | 0.94× |
| g1 | fixed | 27.72 | 0.00 | 0.00 | 11.87 (codegen) | — | — | 1.87× | 0.67× |
| g1 | floating | — | 0.00 | 0.00 | — | — | — | — | 1.07× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 55.39 | 95.33 | 55.41 | 235.55 | — | — | 0.58× | 1.72× |
| iiwa14 | floating | — | 89.48 | 101.24 | 325.28 | — | — | — | 0.88× |
| go2 | fixed | 70.51 | 67.19 | 58.71 | 371.58 | — | — | 1.05× | 1.14× |
| go2 | floating | — | 2.03 | 2.14 | 392.30 | — | — | — | 0.95× |
| g1 | fixed | 3.06 | 1.92 | 2.04 | 437.03 | — | — | 1.60× | 0.94× |
| g1 | floating | — | 2.02 | 2.14 | — | — | — | — | 0.95× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 131.41 | 218.72 | 127.39 | 836.76 | — | — | 0.60× | 1.72× |
| iiwa14 | floating | — | 309.67 | 286.27 | 960.77 | — | — | — | 1.08× |
| go2 | fixed | 120.01 | 144.08 | 143.25 | 888.92 | — | — | 0.83× | 1.01× |
| go2 | floating | — | 1.90 | 2.03 | 1033.99 | — | — | — | 0.94× |
| g1 | fixed | 3.62 | 1.94 | 2.92 | 1174.88 | — | — | 1.87× | 0.67× |
| g1 | floating | — | 2.32 | 2.16 | — | — | — | — | 1.07× |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 92.55 | 13.54 | 8.52 | 1.57 (codegen) | — | — | 0.64× | 1.48× |
| iiwa14 | floating | — | 37.45 | 37.39 | 2.53 (codegen) | — | — | — | 1.04× |
| go2 | fixed | 34.03 | 10.85 | 10.84 | 1.75 (codegen) | — | — | 0.96× | 1.09× |
| go2 | floating | — | 47.73 | 47.93 | 4.34 (codegen) | — | — | — | 1.05× |
| g1 | fixed | 27.11 | 0.00 | 0.00 | 7.38 (codegen) | — | — | 1.28× | 0.91× |
| g1 | floating | — | 86.33 | 86.20 | — | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 115.67 | 63.90 | 55.84 | 292.00 | — | — | 1.81× | 1.14× |
| iiwa14 | floating | — | 116.67 | 115.51 | 358.33 | — | — | — | 1.01× |
| go2 | fixed | 240.27 | 86.17 | 68.28 | 339.90 | — | — | 2.79× | 1.26× |
| go2 | floating | — | 118.46 | 120.28 | 371.58 | — | — | — | 0.98× |
| g1 | fixed | 3.01 | 2.37 | 2.05 | 542.06 | — | — | 1.27× | 1.15× |
| g1 | floating | — | 159.63 | 161.05 | — | — | — | — | 0.99× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 74.47 | 116.68 | 78.79 | 863.58 | — | — | 0.64× | 1.48× |
| iiwa14 | floating | — | 433.64 | 418.61 | 1023.13 | — | — | — | 1.04× |
| go2 | fixed | 129.10 | 134.91 | 123.26 | 1069.83 | — | — | 0.96× | 1.09× |
| go2 | floating | — | 521.09 | 495.72 | 1048.21 | — | — | — | 1.05× |
| g1 | fixed | 2.42 | 1.89 | 2.09 | 1336.30 | — | — | 1.28× | 0.91× |
| g1 | floating | — | 708.51 | 706.47 | — | — | — | — | 1.00× |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 35.17 | 7.30 | 4.59 | 0.50 (codegen) | — | — | 0.90× | 1.37× |
| iiwa14 | floating | — | 13.64 | 13.64 | 1.24 (codegen) | — | — | — | 1.04× |
| go2 | fixed | 34.50 | 6.69 | 6.71 | 0.69 (codegen) | — | — | 0.97× | 0.90× |
| go2 | floating | — | 16.70 | 16.86 | 1.99 (codegen) | — | — | — | 0.90× |
| g1 | fixed | 53.77 | 27.49 | 27.50 | 5.57 (codegen) | — | — | 1.99× | 0.99× |
| g1 | floating | — | 55.04 | 54.94 | — | — | — | — | 1.04× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 53.87 | 60.25 | 49.12 | 263.84 | — | — | 0.89× | 1.23× |
| iiwa14 | floating | — | 65.96 | 69.26 | 318.49 | — | — | — | 0.95× |
| go2 | fixed | 117.73 | 74.64 | 56.39 | 243.75 | — | — | 1.58× | 1.32× |
| go2 | floating | — | 71.45 | 94.68 | 241.06 | — | — | — | 0.75× |
| g1 | fixed | 76.17 | 81.21 | 71.33 | 319.31 | — | — | 0.94× | 1.14× |
| g1 | floating | — | 118.97 | 105.80 | — | — | — | — | 1.12× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 77.03 | 85.59 | 62.26 | 771.01 | — | — | 0.90× | 1.37× |
| iiwa14 | floating | — | 151.64 | 146.28 | 875.93 | — | — | — | 1.04× |
| go2 | fixed | 64.07 | 65.99 | 73.08 | 899.92 | — | — | 0.97× | 0.90× |
| go2 | floating | — | 166.87 | 185.09 | 823.22 | — | — | — | 0.90× |
| g1 | fixed | 300.66 | 150.96 | 152.87 | 906.30 | — | — | 1.99× | 0.99× |
| g1 | floating | — | 436.93 | 422.00 | — | — | — | — | 1.04× |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 16.35 | 11.94 | 7.51 | 4.14 (codegen) | — | — | 0.72× | 1.62× |
| iiwa14 | floating | — | 15.29 | 15.36 | 6.35 (codegen) | — | — | — | 1.04× |
| go2 | fixed | 24.64 | 9.95 | 9.99 | 4.02 (codegen) | — | — | 0.90× | 0.89× |
| go2 | floating | — | 21.68 | 21.90 | 10.03 (codegen) | — | — | — | 1.02× |
| g1 | fixed | 37.73 | 34.30 | 34.42 | 19.77 (codegen) | — | — | 1.70× | 1.07× |
| g1 | floating | — | 75.57 | 75.43 | — | — | — | — | 1.00× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 72.98 | 65.93 | 53.50 | 402.92 | — | — | 1.11× | 1.23× |
| iiwa14 | floating | — | 64.82 | 80.10 | 401.65 | — | — | — | 0.81× |
| go2 | fixed | 70.12 | 60.36 | 58.17 | 483.33 | — | — | 1.16× | 1.04× |
| go2 | floating | — | 80.18 | 83.72 | 553.21 | — | — | — | 0.96× |
| g1 | fixed | 77.44 | 88.70 | 94.46 | 808.25 | — | — | 0.87× | 0.94× |
| g1 | floating | — | 140.61 | 137.12 | — | — | — | — | 1.03× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 122.99 | 171.74 | 106.12 | 1033.26 | — | — | 0.72× | 1.62× |
| iiwa14 | floating | — | 172.98 | 166.05 | 1147.40 | — | — | — | 1.04× |
| go2 | fixed | 120.22 | 133.61 | 150.82 | 1267.16 | — | — | 0.90× | 0.89× |
| go2 | floating | — | 209.39 | 205.44 | 1371.04 | — | — | — | 1.02× |
| g1 | fixed | 533.34 | 312.82 | 292.82 | 1790.75 | — | — | 1.70× | 1.07× |
| g1 | floating | — | 621.98 | 622.78 | — | — | — | — | 1.00× |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 1.30 | 0.00 | 0.00 | 6.70 (codegen) | — | — | 2.32× | 0.77× |
| iiwa14 | floating | — | 0.00 | 0.00 | 11.53 (codegen) | — | — | — | 0.96× |
| go2 | fixed | 1.18 | 0.00 | 0.00 | 6.93 (codegen) | — | — | 3.85× | 0.94× |
| go2 | floating | — | 0.00 | 0.00 | 22.05 (codegen) | — | — | — | 0.93× |
| g1 | fixed | 0.67 | 0.00 | 0.00 | 59.97 (codegen) | — | — | 2.63× | 1.34× |
| g1 | floating | — | 0.00 | 0.00 | — | — | — | — | 1.06× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 10.54 | 2.74 | 2.11 | 434.68 | — | — | 3.84× | 1.30× |
| iiwa14 | floating | — | 2.15 | 1.98 | 618.35 | — | — | — | 1.09× |
| go2 | fixed | 12.99 | 1.93 | 2.11 | 462.81 | — | — | 6.73× | 0.91× |
| go2 | floating | — | 2.56 | 2.28 | 677.85 | — | — | — | 1.12× |
| g1 | fixed | 9.30 | 2.00 | 2.30 | 1235.43 | — | — | 4.65× | 0.87× |
| g1 | floating | — | 1.93 | 2.18 | — | — | — | — | 0.89× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 4.84 | 2.09 | 2.69 | 1118.69 | — | — | 2.32× | 0.77× |
| iiwa14 | floating | — | 1.93 | 2.02 | 1695.91 | — | — | — | 0.96× |
| go2 | fixed | 7.88 | 2.05 | 2.18 | 1401.25 | — | — | 3.85× | 0.94× |
| go2 | floating | — | 2.04 | 2.20 | 1668.87 | — | — | — | 0.93× |
| g1 | fixed | 8.05 | 3.06 | 2.28 | 2547.00 | — | — | 2.63× | 1.34× |
| g1 | floating | — | 2.31 | 2.18 | — | — | — | — | 1.06× |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 14.47 | 1.76 | 1.10 | 0.85 (direct) | — | — | 0.80× | 1.24× |
| iiwa14 | floating | — | 25.25 | 25.33 | 1.14 (direct) | — | — | — | 1.02× |
| go2 | fixed | 16.08 | 0.97 | 0.98 | 2.17 (direct) | — | — | 0.93× | 0.93× |
| go2 | floating | — | 25.45 | 25.66 | 2.26 (direct) | — | — | — | 0.97× |
| g1 | fixed | 11.33 | 4.62 | 4.63 | 3.31 (direct) | — | — | 0.87× | 0.91× |
| g1 | floating | — | 29.39 | 29.80 | — | — | — | — | 0.89× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 135.52 | 52.24 | 50.58 | 259.79 | — | — | 2.59× | 1.03× |
| iiwa14 | floating | — | 83.22 | 84.51 | 260.71 | — | — | — | 0.98× |
| go2 | fixed | 44.05 | 44.21 | 53.20 | 294.82 | — | — | 1.00× | 0.83× |
| go2 | floating | — | 77.80 | 99.93 | 242.81 | — | — | — | 0.78× |
| g1 | fixed | 43.93 | 51.59 | 48.67 | 249.42 | — | — | 0.85× | 1.06× |
| g1 | floating | — | 79.63 | 90.09 | — | — | — | — | 0.88× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 56.69 | 71.12 | 57.54 | 819.58 | — | — | 0.80× | 1.24× |
| iiwa14 | floating | — | 145.40 | 143.19 | 835.77 | — | — | — | 1.02× |
| go2 | fixed | 42.09 | 45.47 | 49.09 | 921.67 | — | — | 0.93× | 0.93× |
| go2 | floating | — | 139.59 | 143.27 | 852.36 | — | — | — | 0.97× |
| g1 | fixed | 75.07 | 86.60 | 95.03 | 933.11 | — | — | 0.87× | 0.91× |
| g1 | floating | — | 246.14 | 275.66 | — | — | — | — | 0.89× |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 27.76 | 2.04 | 1.28 | 1.12 (direct) | — | — | 1.23× | 1.06× |
| iiwa14 | floating | — | 308.88 | 309.56 | 1.31 (direct) | — | — | — | 1.00× |
| go2 | fixed | 69.13 | 1.68 | 2.23 | 1.30 (direct) | — | — | 1.04× | 1.21× |
| go2 | floating | — | 310.50 | 305.98 | 1.63 (direct) | — | — | — | 1.02× |
| g1 | fixed | 19.68 | 11.69 | 13.75 | 3.83 (direct) | — | — | 1.38× | 0.88× |
| g1 | floating | — | 319.40 | 330.30 | — | — | — | — | 0.97× |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 116.59 | 49.63 | 50.45 | 256.05 | — | — | 2.35× | 0.98× |
| iiwa14 | floating | — | 400.59 | 420.02 | 217.13 | — | — | — | 0.95× |
| go2 | fixed | 42.12 | 45.43 | 47.32 | 274.58 | — | — | 0.93× | 0.96× |
| go2 | floating | — | 414.81 | 422.75 | 239.85 | — | — | — | 0.98× |
| g1 | fixed | 50.89 | 70.43 | 58.57 | 258.34 | — | — | 0.72× | 1.20× |
| g1 | floating | — | 446.83 | 428.02 | — | — | — | — | 1.04× |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | 75.08 | 61.04 | 57.51 | 796.49 | — | — | 1.23× | 1.06× |
| iiwa14 | floating | — | 1975.31 | 1971.75 | 791.79 | — | — | — | 1.00× |
| go2 | fixed | 64.55 | 62.23 | 51.24 | 904.49 | — | — | 1.04× | 1.21× |
| go2 | floating | — | 2029.08 | 1997.31 | 834.19 | — | — | — | 1.02× |
| g1 | fixed | 195.96 | 141.58 | 160.28 | 931.57 | — | — | 1.38× | 0.88× |
| g1 | floating | — | 2072.86 | 2147.86 | — | — | — | — | 0.97× |

## Second-Order

### IDSVA_SO (2nd-order ID)

**single-call**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 19.68 (direct) | — | — | — | — |
| iiwa14 | floating | — | — | — | 61.82 (direct) | — | — | — | — |
| go2 | fixed | — | — | — | 27.18 (direct) | — | — | — | — |
| go2 | floating | — | — | — | 84.05 (direct) | — | — | — | — |
| g1 | fixed | — | — | — | 131.03 (direct) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 596.95 | — | — | — | — |
| iiwa14 | floating | — | — | — | 1205.51 | — | — | — | — |
| go2 | fixed | — | — | — | 804.64 | — | — | — | — |
| go2 | floating | — | — | — | 1533.17 | — | — | — | — |
| g1 | fixed | — | — | — | 2382.35 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax | glass/pre | glass_nv/glass |
|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:|:---------:|:-------------:|
| iiwa14 | fixed | — | — | — | 1597.05 | — | — | — | — |
| iiwa14 | floating | — | — | — | 2953.40 | — | — | — | — |
| go2 | fixed | — | — | — | 1952.10 | — | — | — | — |
| go2 | floating | — | — | — | 3786.75 | — | — | — | — |
| g1 | fixed | — | — | — | 4990.36 | — | — | — | — |
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

