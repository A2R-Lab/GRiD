# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-05-24  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: 3.9.0

All times in **µs**.

Columns:
- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only (pre_glass harness does not support floating-base).
- **glass**: GRiD HEAD with the pure-SIMT GLASS backend.
- **pin**: Pinocchio CPU reference (codegen where available).
- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only (id / fd / ee_pose / id_du); others render `—`.
- **frax_cpu / frax_gpu**: Frax (JAX) reference (https://github.com/danielpmorton/frax) timed separately on JAX's CPU and CUDA backends — Frax advertises both as fast. Subset of algos only (id / fd / crba / minv); others render `—`.
- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; **< 1.00× = HEAD regressed**.

Each algorithm gets three sub-tables: **single-call**, **batch N=16**, **batch N=256**. Same backend columns + ratio in each. Values are median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: No Pinocchio equivalent — GRiD numbers only.

## Core Dynamics

### ID (Inverse Dynamics)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.63 | 0.16 (codegen) | — | 17.63 | 49.70 | — |
| iiwa14 | floating | — | 10.05 | 0.21 (codegen) | — | 25.68 | 58.56 | — |
| go2 | fixed | — | 6.89 | 0.23 (codegen) | — | 28.20 | 90.07 | — |
| go2 | floating | — | 13.24 | 0.28 (codegen) | — | 35.48 | 144.89 | — |
| g1 | fixed | — | 22.70 | 0.66 (codegen) | — | 47.85 | 231.74 | — |
| g1 | floating | — | 28.92 | 1.03 (codegen) | — | 55.09 | 244.37 | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.22 | 16.48 | — | 79.13 | 65.12 | — |
| iiwa14 | floating | — | 15.14 | 18.51 | — | 137.82 | 65.55 | — |
| go2 | fixed | — | 12.64 | 23.49 | — | 86.70 | 98.89 | — |
| go2 | floating | — | 18.80 | 26.52 | — | 175.41 | 104.52 | — |
| g1 | fixed | — | 29.11 | 53.11 | — | 285.19 | 192.90 | — |
| g1 | floating | — | 34.60 | 55.43 | — | 502.03 | 118.11 | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.76 | 73.00 | — | 830.97 | 59.26 | — |
| iiwa14 | floating | — | 25.19 | 36.70 | — | 2114.92 | 65.90 | — |
| go2 | fixed | — | 13.18 | 42.58 | — | 1390.49 | 115.98 | — |
| go2 | floating | — | 31.86 | 41.44 | — | 2247.70 | 102.74 | — |
| g1 | fixed | — | 52.23 | 63.17 | — | 3549.02 | 248.29 | — |
| g1 | floating | — | — | 73.58 | — | 3623.37 | 138.95 | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.46 | 0.29 (codegen) | — | 17.77 | 71.82 | — |
| iiwa14 | floating | — | 27.91 | 0.71 (codegen) | — | 30.08 | 106.36 | — |
| go2 | fixed | — | 7.99 | 0.34 (codegen) | — | 28.54 | 117.72 | — |
| go2 | floating | — | 32.13 | 1.28 (codegen) | — | 37.19 | 200.00 | — |
| g1 | fixed | — | 38.90 | 2.17 (codegen) | — | 50.77 | 224.14 | — |
| g1 | floating | — | 65.58 | 4.39 (codegen) | — | 71.24 | 261.93 | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.55 | 13.87 | — | 88.41 | 73.95 | — |
| iiwa14 | floating | — | 36.01 | 29.45 | — | 164.29 | 141.99 | — |
| go2 | fixed | — | 15.12 | 17.68 | — | 117.81 | 125.85 | — |
| go2 | floating | — | 39.94 | 39.56 | — | 371.68 | 134.67 | — |
| g1 | fixed | — | 59.34 | 73.59 | — | 411.55 | 308.84 | — |
| g1 | floating | — | 96.36 | 183.68 | — | 686.06 | 233.40 | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.26 | 34.61 | — | 1090.26 | 80.43 | — |
| iiwa14 | floating | — | 64.60 | 42.58 | — | 3077.92 | 102.74 | — |
| go2 | fixed | — | 23.44 | 46.90 | — | 1691.81 | 237.59 | — |
| go2 | floating | — | 72.97 | 62.30 | — | 3044.39 | 153.87 | — |
| g1 | fixed | — | 111.16 | 124.94 | — | 3745.42 | 251.58 | — |
| g1 | floating | — | — | 219.85 | — | 4876.96 | 249.30 | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.75 | 0.53 (codegen) | — | 20.02 | 73.36 | — |
| iiwa14 | floating | — | 33.00 | 1.25 (codegen) | — | 28.33 | 81.59 | — |
| go2 | fixed | — | 10.84 | 0.64 (codegen) | — | 30.72 | 100.84 | — |
| go2 | floating | — | 38.15 | 1.93 (codegen) | — | 36.67 | 149.63 | — |
| g1 | fixed | — | 53.06 | 3.52 (codegen) | — | 54.59 | 229.33 | — |
| g1 | floating | — | 82.51 | 5.66 (codegen) | — | 54.17 | 241.35 | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.58 | 16.84 | — | 90.16 | 90.28 | — |
| iiwa14 | floating | — | 39.58 | 30.39 | — | 205.06 | 87.59 | — |
| go2 | fixed | — | 18.84 | 21.26 | — | 113.47 | 122.43 | — |
| go2 | floating | — | 44.54 | 47.14 | — | 377.01 | 131.12 | — |
| g1 | fixed | — | 76.21 | 73.79 | — | 396.82 | 342.55 | — |
| g1 | floating | — | 115.49 | 124.01 | — | 597.50 | 199.02 | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.32 | 43.24 | — | 1106.54 | 86.75 | — |
| iiwa14 | floating | — | 72.73 | 55.23 | — | 2608.53 | 98.71 | — |
| go2 | fixed | — | 29.78 | 55.88 | — | 1707.48 | 365.11 | — |
| go2 | floating | — | 82.22 | 68.63 | — | 2894.33 | 143.89 | — |
| g1 | fixed | — | 136.37 | 108.63 | — | 4142.07 | 261.84 | — |
| g1 | floating | — | — | 276.83 | — | 4746.73 | 223.87 | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.22 | 0.37 (codegen) | — | — | — | — |
| iiwa14 | floating | — | 50.98 | 0.72 (codegen) | — | — | — | — |
| go2 | fixed | — | 10.30 | 0.46 (codegen) | — | — | — | — |
| go2 | floating | — | 60.79 | 1.35 (codegen) | — | — | — | — |
| g1 | fixed | — | 38.48 | 2.29 (codegen) | — | — | — | — |
| g1 | floating | — | 96.41 | 2.90 (codegen) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.27 | 29.35 | — | — | — | — |
| iiwa14 | floating | — | 57.02 | 36.52 | — | — | — | — |
| go2 | fixed | — | 17.57 | 43.80 | — | — | — | — |
| go2 | floating | — | 67.11 | 58.75 | — | — | — | — |
| g1 | fixed | — | 46.11 | 101.97 | — | — | — | — |
| g1 | floating | — | 103.23 | 114.83 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.02 | 69.25 | — | — | — | — |
| iiwa14 | floating | — | 107.86 | 99.69 | — | — | — | — |
| go2 | fixed | — | 27.53 | 74.07 | — | — | — | — |
| go2 | floating | — | 127.08 | 72.25 | — | — | — | — |
| g1 | fixed | — | 84.15 | 132.90 | — | — | — | — |
| g1 | floating | — | — | 141.36 | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.57 | 0.21 (codegen) | — | 16.30 | 47.43 | — |
| iiwa14 | floating | — | 13.59 | 0.35 (codegen) | — | 24.19 | 51.53 | — |
| go2 | fixed | — | 6.46 | 0.25 (codegen) | — | 26.52 | 72.41 | — |
| go2 | floating | — | 16.02 | 0.51 (codegen) | — | 31.95 | 146.49 | — |
| g1 | fixed | — | 26.90 | 1.18 (codegen) | — | 45.56 | 229.31 | — |
| g1 | floating | — | 51.62 | 1.70 (codegen) | — | 52.21 | 193.30 | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.49 | 10.45 | — | 76.41 | 54.26 | — |
| iiwa14 | floating | — | 19.40 | 12.79 | — | 152.24 | 57.12 | — |
| go2 | fixed | — | 12.50 | 12.53 | — | 94.19 | 78.47 | — |
| go2 | floating | — | 23.50 | 17.77 | — | 275.62 | 91.75 | — |
| g1 | fixed | — | 33.78 | 31.05 | — | 347.66 | 219.79 | — |
| g1 | floating | — | 57.04 | 34.97 | — | 593.31 | 116.82 | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.04 | 30.04 | — | 963.06 | 89.05 | — |
| iiwa14 | floating | — | 32.63 | 144.68 | — | 2374.89 | 57.53 | — |
| go2 | fixed | — | 13.13 | 29.35 | — | 1472.31 | 347.80 | — |
| go2 | floating | — | 39.49 | 38.96 | — | 2400.78 | 120.10 | — |
| g1 | fixed | — | 60.99 | 73.61 | — | 3534.35 | 228.02 | — |
| g1 | floating | — | — | 62.20 | — | 4043.34 | 135.64 | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.55 | 1.14 (codegen) | — | — | — | — |
| iiwa14 | floating | — | 15.32 | 2.71 (codegen) | — | — | — | — |
| go2 | fixed | — | 9.63 | 1.60 (codegen) | — | — | — | — |
| go2 | floating | — | 21.46 | 2.88 (codegen) | — | — | — | — |
| g1 | fixed | — | 31.98 | 6.07 (codegen) | — | — | — | — |
| g1 | floating | — | 69.35 | 8.37 (codegen) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.67 | 49.92 | — | — | — | — |
| iiwa14 | floating | — | 23.04 | 63.47 | — | — | — | — |
| go2 | fixed | — | 15.89 | 79.80 | — | — | — | — |
| go2 | floating | — | 27.99 | 104.88 | — | — | — | — |
| g1 | fixed | — | 38.79 | 229.74 | — | — | — | — |
| g1 | floating | — | 88.58 | 226.00 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.52 | 227.33 | — | — | — | — |
| iiwa14 | floating | — | 38.74 | 81.37 | — | — | — | — |
| go2 | fixed | — | 16.65 | 97.23 | — | — | — | — |
| go2 | floating | — | 49.68 | 197.79 | — | — | — | — |
| g1 | fixed | — | 71.57 | 403.48 | — | — | — | — |
| g1 | floating | — | — | 320.27 | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.44 | 2.34 (codegen) | — | — | — | — |
| iiwa14 | floating | — | 43.76 | 4.22 (codegen) | — | — | — | — |
| go2 | fixed | — | 19.55 | 2.69 (codegen) | — | — | — | — |
| go2 | floating | — | 51.39 | 5.63 (codegen) | — | — | — | — |
| g1 | fixed | — | 79.47 | 11.43 (codegen) | — | — | — | — |
| g1 | floating | — | 153.47 | 17.59 (codegen) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.04 | 62.05 | — | — | — | — |
| iiwa14 | floating | — | 50.37 | 102.66 | — | — | — | — |
| go2 | fixed | — | 26.08 | 104.33 | — | — | — | — |
| go2 | floating | — | 58.55 | 194.96 | — | — | — | — |
| g1 | fixed | — | 106.70 | 415.65 | — | — | — | — |
| g1 | floating | — | 193.45 | 560.93 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 40.36 | 83.03 | — | — | — | — |
| iiwa14 | floating | — | 94.33 | 117.62 | — | — | — | — |
| go2 | fixed | — | 45.46 | 288.16 | — | — | — | — |
| go2 | floating | — | 110.04 | 231.23 | — | — | — | — |
| g1 | fixed | — | 238.73 | 506.90 | — | — | — | — |
| g1 | floating | — | — | 630.88 | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.97 | — | — | — | — | — |
| iiwa14 | floating | — | 33.98 | — | — | — | — | — |
| go2 | fixed | — | 10.83 | — | — | — | — | — |
| go2 | floating | — | 38.64 | — | — | — | — | — |
| g1 | fixed | — | 54.31 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 21.89 | — | — | — | — | — |
| iiwa14 | floating | — | 48.04 | — | — | — | — | — |
| go2 | fixed | — | 26.34 | — | — | — | — | — |
| go2 | floating | — | 52.50 | — | — | — | — | — |
| g1 | fixed | — | 84.81 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.85 | — | — | — | — | — |
| iiwa14 | floating | — | 82.57 | — | — | — | — | — |
| go2 | fixed | — | 37.54 | — | — | — | — | — |
| go2 | floating | — | 91.23 | — | — | — | — | — |
| g1 | fixed | — | 149.14 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.79 | — | — | — | — | — |
| iiwa14 | floating | — | 50.35 | — | — | — | — | — |
| go2 | fixed | — | 19.88 | — | — | — | — | — |
| go2 | floating | — | 57.76 | — | — | — | — | — |
| g1 | fixed | — | 82.65 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 30.40 | — | — | — | — | — |
| iiwa14 | floating | — | 64.33 | — | — | — | — | — |
| go2 | fixed | — | 33.51 | — | — | — | — | — |
| go2 | floating | — | 71.42 | — | — | — | — | — |
| g1 | fixed | — | 116.39 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 47.76 | — | — | — | — | — |
| iiwa14 | floating | — | 114.55 | — | — | — | — | — |
| go2 | fixed | — | 53.16 | — | — | — | — | — |
| go2 | floating | — | 129.23 | — | — | — | — | — |
| g1 | fixed | — | 255.50 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.79 | — | — | — | — | — |
| iiwa14 | floating | — | 52.72 | — | — | — | — | — |
| go2 | fixed | — | 19.99 | — | — | — | — | — |
| go2 | floating | — | 59.97 | — | — | — | — | — |
| g1 | fixed | — | 82.76 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 30.91 | — | — | — | — | — |
| iiwa14 | floating | — | 66.76 | — | — | — | — | — |
| go2 | fixed | — | 33.87 | — | — | — | — | — |
| go2 | floating | — | 75.31 | — | — | — | — | — |
| g1 | fixed | — | 117.14 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.74 | — | — | — | — | — |
| iiwa14 | floating | — | 119.77 | — | — | — | — | — |
| go2 | fixed | — | 53.79 | — | — | — | — | — |
| go2 | floating | — | 137.10 | — | — | — | — | — |
| g1 | fixed | — | 259.14 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.38 | 0.32 (direct) | — | — | — | — |
| iiwa14 | floating | — | 15.66 | 0.34 (direct) | — | — | — | — |
| go2 | fixed | — | 1.21 | 0.79 (direct) | — | — | — | — |
| go2 | floating | — | 15.13 | 0.85 (direct) | — | — | — | — |
| g1 | fixed | — | 4.62 | 1.03 (direct) | — | — | — | — |
| g1 | floating | — | 18.76 | 1.14 (direct) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.77 | 10.21 | — | — | — | — |
| iiwa14 | floating | — | 21.33 | 10.07 | — | — | — | — |
| go2 | fixed | — | 6.90 | 17.61 | — | — | — | — |
| go2 | floating | — | 21.10 | 18.09 | — | — | — | — |
| g1 | fixed | — | 10.40 | 31.17 | — | — | — | — |
| g1 | floating | — | 25.62 | 32.19 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.99 | 29.93 | — | — | — | — |
| iiwa14 | floating | — | 37.43 | 28.92 | — | — | — | — |
| go2 | fixed | — | 7.24 | 46.75 | — | — | — | — |
| go2 | floating | — | 36.61 | 36.88 | — | — | — | — |
| g1 | fixed | — | 15.90 | 41.11 | — | — | — | — |
| g1 | floating | — | — | 58.10 | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.64 | 0.33 (direct) | — | — | — | — |
| iiwa14 | floating | — | 262.37 | 0.54 (direct) | — | — | — | — |
| go2 | fixed | — | 1.97 | 0.46 (direct) | — | — | — | — |
| go2 | floating | — | 262.88 | 0.56 (direct) | — | — | — | — |
| g1 | fixed | — | 13.63 | 1.01 (direct) | — | — | — | — |
| g1 | floating | — | 274.12 | 1.16 (direct) | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.23 | 10.22 | — | — | — | — |
| iiwa14 | floating | — | 270.45 | 12.20 | — | — | — | — |
| go2 | fixed | — | 7.79 | 12.49 | — | — | — | — |
| go2 | floating | — | 271.04 | 14.49 | — | — | — | — |
| g1 | fixed | — | 19.57 | 28.36 | — | — | — | — |
| g1 | floating | — | 280.34 | 33.64 | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.66 | 35.45 | — | — | — | — |
| iiwa14 | floating | — | 534.87 | 33.94 | — | — | — | — |
| go2 | fixed | — | 8.63 | 33.98 | — | — | — | — |
| go2 | floating | — | 535.74 | 38.77 | — | — | — | — |
| g1 | fixed | — | 34.24 | 49.44 | — | — | — | — |
| g1 | floating | — | — | 73.72 | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.03 | — | — | — | — | — |
| iiwa14 | floating | — | 366.54 | — | — | — | — | — |
| go2 | fixed | — | 36.10 | — | — | — | — | — |
| go2 | floating | — | 548.89 | — | — | — | — | — |
| g1 | fixed | — | 1298.79 | — | — | — | — | — |
| g1 | floating | — | 1414.53 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.01 | — | — | — | — | — |
| iiwa14 | floating | — | 381.11 | — | — | — | — | — |
| go2 | fixed | — | 44.94 | — | — | — | — | — |
| go2 | floating | — | 566.83 | — | — | — | — | — |
| g1 | fixed | — | 1339.22 | — | — | — | — | — |
| g1 | floating | — | 1494.12 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.44 | — | — | — | — | — |
| iiwa14 | floating | — | 754.93 | — | — | — | — | — |
| go2 | fixed | — | 87.37 | — | — | — | — | — |
| go2 | floating | — | 1135.45 | — | — | — | — | — |
| g1 | fixed | — | 2992.71 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.44 | — | — | — | — | — |
| iiwa14 | floating | — | 2637.60 | — | — | — | — | — |
| go2 | fixed | — | 36.12 | — | — | — | — | — |
| go2 | floating | — | 3936.82 | — | — | — | — | — |
| g1 | fixed | — | 1299.76 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.01 | — | — | — | — | — |
| iiwa14 | floating | — | 2695.25 | — | — | — | — | — |
| go2 | fixed | — | 44.92 | — | — | — | — | — |
| go2 | floating | — | 3978.96 | — | — | — | — | — |
| g1 | fixed | — | 1337.01 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.45 | — | — | — | — | — |
| iiwa14 | floating | — | 5333.01 | — | — | — | — | — |
| go2 | fixed | — | 87.33 | — | — | — | — | — |
| go2 | floating | — | 7793.54 | — | — | — | — | — |
| g1 | fixed | — | 2992.88 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 235.58 | — | — | — | — | — |
| iiwa14 | floating | — | 367.82 | — | — | — | — | — |
| go2 | fixed | — | 375.46 | — | — | — | — | — |
| go2 | floating | — | 550.73 | — | — | — | — | — |
| g1 | fixed | — | 1210.98 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 241.03 | — | — | — | — | — |
| iiwa14 | floating | — | 381.11 | — | — | — | — | — |
| go2 | fixed | — | 388.49 | — | — | — | — | — |
| go2 | floating | — | 566.73 | — | — | — | — | — |
| g1 | fixed | — | 1259.93 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 478.96 | — | — | — | — | — |
| iiwa14 | floating | — | 754.97 | — | — | — | — | — |
| go2 | fixed | — | 770.94 | — | — | — | — | — |
| go2 | floating | — | 1135.28 | — | — | — | — | — |
| g1 | fixed | — | 2538.78 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 49.55 | — | — | — | — | — |
| iiwa14 | floating | — | 529.40 | — | — | — | — | — |
| go2 | fixed | — | 64.78 | — | — | — | — | — |
| go2 | floating | — | 773.05 | — | — | — | — | — |
| g1 | fixed | — | 3234.90 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 55.71 | — | — | — | — | — |
| iiwa14 | floating | — | 528.33 | — | — | — | — | — |
| go2 | fixed | — | 82.46 | — | — | — | — | — |
| go2 | floating | — | 779.45 | — | — | — | — | — |
| g1 | fixed | — | 3683.64 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 105.48 | — | — | — | — | — |
| iiwa14 | floating | — | 1043.49 | — | — | — | — | — |
| go2 | fixed | — | 162.63 | — | — | — | — | — |
| go2 | floating | — | 1591.19 | — | — | — | — | — |
| g1 | fixed | — | 9255.97 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

