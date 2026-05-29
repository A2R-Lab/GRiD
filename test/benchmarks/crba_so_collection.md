# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-05-27  
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
| iiwa14 | fixed | — | 4.54 | 0.17 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 5.99 | 0.22 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 22.20 | 0.65 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 38.45 | 1.78 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.07 | 17.52 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 12.82 | 24.06 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 30.96 | 55.34 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 49.50 | 93.87 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.47 | 37.93 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 13.34 | 64.05 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 52.58 | 127.26 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 51.97 | 177.76 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.21 | 0.30 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 8.01 | 0.35 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 31.59 | 2.22 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 52.31 | 5.83 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.29 | 15.22 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 14.63 | 21.31 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 46.32 | 54.61 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 69.04 | 278.89 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.72 | 25.92 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 22.99 | 57.50 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 77.73 | 86.55 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 96.00 | 418.25 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.46 | 0.53 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 9.93 | 0.67 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 41.36 | 3.39 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 66.57 | 8.32 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.39 | 16.14 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 18.19 | 20.51 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 62.13 | 114.40 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 77.18 | 242.75 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.69 | 66.93 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 27.90 | 55.19 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 98.57 | 154.76 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 146.26 | 264.29 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.23 | 0.40 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 10.29 | 0.47 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 38.35 | 2.33 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 65.52 | 4.47 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.31 | 30.24 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 17.55 | 43.88 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 46.28 | 108.75 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 79.29 | 240.82 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.04 | 75.30 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 27.59 | 79.25 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 83.95 | 122.38 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 96.01 | 217.35 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.66 | 0.20 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 6.44 | 0.27 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 26.98 | 1.18 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 47.96 | 2.54 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.54 | 10.74 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 12.43 | 16.18 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 33.77 | 32.65 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 55.67 | 58.96 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.05 | 27.46 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 13.02 | 115.36 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 60.37 | 72.20 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 103.71 | 125.03 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.61 | 1.16 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 8.87 | 1.26 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 31.34 | 6.10 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 76.06 | 13.27 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 21.62 | 60.31 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 23.22 | 96.01 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 47.38 | 238.85 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 95.69 | 438.86 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.45 | 87.61 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 23.95 | 202.07 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 78.13 | 403.22 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 178.58 | 473.53 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.38 | 2.36 (codegen) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 17.71 | 2.53 (codegen) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 70.48 | 11.58 (codegen) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 139.39 | 32.58 (codegen) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.19 | 62.18 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 32.57 | 96.05 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 105.51 | 408.33 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 145.83 | 983.18 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 47.59 | 159.81 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 50.20 | 275.29 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 215.56 | 491.56 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 290.49 | 1103.41 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.79 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 10.08 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 42.31 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 67.07 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.65 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 25.44 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 69.97 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 85.23 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.33 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 35.54 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 106.75 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 155.71 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.01 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 18.01 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 79.27 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 150.01 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.49 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 33.08 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 107.46 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 162.17 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.08 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 50.89 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 222.02 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 325.15 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.06 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 18.16 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 79.29 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 150.14 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.76 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 33.35 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 107.42 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 161.95 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.41 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 51.70 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 222.28 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 326.87 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.38 | 0.32 (direct) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 1.21 | 0.85 (direct) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 4.62 | 1.33 (direct) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 7.77 | 1.78 (direct) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.73 | 12.72 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 6.86 | 17.91 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 10.40 | 43.83 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 13.97 | 95.15 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.97 | 27.78 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 7.21 | 58.62 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 15.44 | 79.87 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 23.21 | 109.52 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.64 | 0.34 (direct) | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 1.87 | 0.45 (direct) | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 12.39 | 1.03 (direct) | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 32.59 | 1.79 (direct) | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.39 | 10.48 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 15.96 | 12.53 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 29.81 | 41.31 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 48.00 | 74.34 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.79 | 40.10 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 16.34 | 35.43 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 42.23 | 83.24 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 85.50 | 107.88 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### EE_POSE_HESSIAN (2nd-order EE Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | — | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.01 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 36.02 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1299.29 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 5826.58 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.09 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 44.79 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1339.86 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 9793.85 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.38 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 87.73 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 3004.04 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 21682.43 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.02 | 7.50 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 36.07 | 10.86 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1291.22 | 43.58 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 5794.38 | 150.45 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.00 | 121.92 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 44.78 | 185.45 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1340.02 | 801.34 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 9791.23 | 2712.38 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.26 | 418.85 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 87.68 | 391.54 | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 3003.69 | 1198.01 | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 21683.18 | 8055.68 | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 234.30 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 377.56 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1194.52 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 2381.86 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 240.94 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 388.54 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1249.65 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 2454.27 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 477.08 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 770.30 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 2559.43 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 6227.67 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 49.71 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 67.22 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 3243.31 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 17706.14 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 58.70 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 77.55 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 3676.52 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 28672.41 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 108.50 | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | 148.48 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 9262.83 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |
| h1_2 | fixed | — | 81563.77 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

