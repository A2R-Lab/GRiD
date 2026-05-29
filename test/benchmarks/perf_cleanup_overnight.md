# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-05-27  
**Pre-glass ref**: `d2c0d18` (last benchmark-capable commit before GLASS v2 work)  
**Pinocchio**: ?

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
| iiwa14 | fixed | — | 4.54 | — | — | — | — | — |
| iiwa14 | floating | — | 9.89 | — | — | — | — | — |
| go2 | fixed | — | 6.02 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 22.16 | — | — | — | — | — |
| g1 | floating | — | 28.17 | — | — | — | — | — |
| h1_2 | fixed | — | 38.45 | — | — | — | — | — |
| h1_2 | floating | — | 44.17 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.17 | — | — | — | — | — |
| iiwa14 | floating | — | 16.18 | — | — | — | — | — |
| go2 | fixed | — | 12.94 | — | — | — | — | — |
| go2 | floating | — | 18.98 | — | — | — | — | — |
| g1 | fixed | — | 30.98 | — | — | — | — | — |
| g1 | floating | — | 36.63 | — | — | — | — | — |
| h1_2 | fixed | — | 49.38 | — | — | — | — | — |
| h1_2 | floating | — | 53.36 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.48 | — | — | — | — | — |
| iiwa14 | floating | — | 25.54 | — | — | — | — | — |
| go2 | fixed | — | 13.34 | — | — | — | — | — |
| go2 | floating | — | 31.11 | — | — | — | — | — |
| g1 | fixed | — | 53.13 | — | — | — | — | — |
| g1 | floating | — | 64.71 | — | — | — | — | — |
| h1_2 | fixed | — | 52.40 | — | — | — | — | — |
| h1_2 | floating | — | 96.81 | — | — | — | — | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.19 | — | — | — | — | — |
| iiwa14 | floating | — | 28.47 | — | — | — | — | — |
| go2 | fixed | — | 8.05 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 31.59 | — | — | — | — | — |
| g1 | floating | — | 58.83 | — | — | — | — | — |
| h1_2 | fixed | — | 52.33 | — | — | — | — | — |
| h1_2 | floating | — | 83.94 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.25 | — | — | — | — | — |
| iiwa14 | floating | — | 35.76 | — | — | — | — | — |
| go2 | fixed | — | 14.69 | — | — | — | — | — |
| go2 | floating | — | 38.54 | — | — | — | — | — |
| g1 | fixed | — | 46.36 | — | — | — | — | — |
| g1 | floating | — | 76.99 | — | — | — | — | — |
| h1_2 | fixed | — | 69.08 | — | — | — | — | — |
| h1_2 | floating | — | 104.05 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.69 | — | — | — | — | — |
| iiwa14 | floating | — | 64.36 | — | — | — | — | — |
| go2 | fixed | — | 23.52 | — | — | — | — | — |
| go2 | floating | — | 69.60 | — | — | — | — | — |
| g1 | fixed | — | 77.04 | — | — | — | — | — |
| g1 | floating | — | 163.93 | — | — | — | — | — |
| h1_2 | fixed | — | 95.16 | — | — | — | — | — |
| h1_2 | floating | — | 146.53 | — | — | — | — | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.45 | — | — | — | — | — |
| iiwa14 | floating | — | 32.10 | — | — | — | — | — |
| go2 | fixed | — | 9.96 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 41.37 | — | — | — | — | — |
| g1 | floating | — | 75.24 | — | — | — | — | — |
| h1_2 | fixed | — | 66.72 | — | — | — | — | — |
| h1_2 | floating | — | 107.84 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.35 | — | — | — | — | — |
| iiwa14 | floating | — | 39.84 | — | — | — | — | — |
| go2 | fixed | — | 18.12 | — | — | — | — | — |
| go2 | floating | — | 43.35 | — | — | — | — | — |
| g1 | fixed | — | 62.23 | — | — | — | — | — |
| g1 | floating | — | 97.74 | — | — | — | — | — |
| h1_2 | fixed | — | 77.25 | — | — | — | — | — |
| h1_2 | floating | — | 131.05 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.63 | — | — | — | — | — |
| iiwa14 | floating | — | 71.20 | — | — | — | — | — |
| go2 | fixed | — | 27.99 | — | — | — | — | — |
| go2 | floating | — | 79.32 | — | — | — | — | — |
| g1 | fixed | — | 97.11 | — | — | — | — | — |
| g1 | floating | — | 204.88 | — | — | — | — | — |
| h1_2 | fixed | — | 146.51 | — | — | — | — | — |
| h1_2 | floating | — | 182.51 | — | — | — | — | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.22 | — | — | — | — | — |
| iiwa14 | floating | — | 50.99 | — | — | — | — | — |
| go2 | fixed | — | 10.32 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 38.42 | — | — | — | — | — |
| g1 | floating | — | 96.58 | — | — | — | — | — |
| h1_2 | fixed | — | 65.44 | — | — | — | — | — |
| h1_2 | floating | — | 137.69 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.32 | — | — | — | — | — |
| iiwa14 | floating | — | 56.71 | — | — | — | — | — |
| go2 | fixed | — | 17.51 | — | — | — | — | — |
| go2 | floating | — | 66.79 | — | — | — | — | — |
| g1 | fixed | — | 46.29 | — | — | — | — | — |
| g1 | floating | — | 103.23 | — | — | — | — | — |
| h1_2 | fixed | — | 79.26 | — | — | — | — | — |
| h1_2 | floating | — | 145.91 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.01 | — | — | — | — | — |
| iiwa14 | floating | — | 107.26 | — | — | — | — | — |
| go2 | fixed | — | 27.67 | — | — | — | — | — |
| go2 | floating | — | 126.94 | — | — | — | — | — |
| g1 | fixed | — | 84.17 | — | — | — | — | — |
| g1 | floating | — | 197.91 | — | — | — | — | — |
| h1_2 | fixed | — | 95.62 | — | — | — | — | — |
| h1_2 | floating | — | 281.26 | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.20 | — | — | — | — | — |
| iiwa14 | floating | — | 13.57 | — | — | — | — | — |
| go2 | fixed | — | 7.98 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 43.20 | — | — | — | — | — |
| g1 | floating | — | 51.85 | — | — | — | — | — |
| h1_2 | fixed | — | 86.59 | — | — | — | — | — |
| h1_2 | floating | — | 91.96 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.97 | — | — | — | — | — |
| iiwa14 | floating | — | 19.38 | — | — | — | — | — |
| go2 | fixed | — | 21.52 | — | — | — | — | — |
| go2 | floating | — | 23.36 | — | — | — | — | — |
| g1 | fixed | — | 74.02 | — | — | — | — | — |
| g1 | floating | — | 57.25 | — | — | — | — | — |
| h1_2 | fixed | — | 132.65 | — | — | — | — | — |
| h1_2 | floating | — | 99.35 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 27.34 | — | — | — | — | — |
| iiwa14 | floating | — | 32.32 | — | — | — | — | — |
| go2 | fixed | — | 48.88 | — | — | — | — | — |
| go2 | floating | — | 39.68 | — | — | — | — | — |
| g1 | fixed | — | 297.67 | — | — | — | — | — |
| g1 | floating | — | 107.59 | — | — | — | — | — |
| h1_2 | fixed | — | 647.36 | — | — | — | — | — |
| h1_2 | floating | — | 191.26 | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.59 | — | — | — | — | — |
| iiwa14 | floating | — | 15.86 | — | — | — | — | — |
| go2 | fixed | — | 8.91 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 31.32 | — | — | — | — | — |
| g1 | floating | — | 70.48 | — | — | — | — | — |
| h1_2 | fixed | — | 76.11 | — | — | — | — | — |
| h1_2 | floating | — | 145.30 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 21.61 | — | — | — | — | — |
| iiwa14 | floating | — | 31.03 | — | — | — | — | — |
| go2 | fixed | — | 23.37 | — | — | — | — | — |
| go2 | floating | — | 35.32 | — | — | — | — | — |
| g1 | fixed | — | 47.74 | — | — | — | — | — |
| g1 | floating | — | 90.44 | — | — | — | — | — |
| h1_2 | fixed | — | 95.71 | — | — | — | — | — |
| h1_2 | floating | — | 182.00 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.39 | — | — | — | — | — |
| iiwa14 | floating | — | 46.41 | — | — | — | — | — |
| go2 | fixed | — | 24.13 | — | — | — | — | — |
| go2 | floating | — | 55.99 | — | — | — | — | — |
| g1 | fixed | — | 78.73 | — | — | — | — | — |
| g1 | floating | — | 183.30 | — | — | — | — | — |
| h1_2 | fixed | — | 178.72 | — | — | — | — | — |
| h1_2 | floating | — | 434.78 | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 16.49 | — | — | — | — | — |
| iiwa14 | floating | — | 43.38 | — | — | — | — | — |
| go2 | fixed | — | 17.81 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 70.59 | — | — | — | — | — |
| g1 | floating | — | 147.29 | — | — | — | — | — |
| h1_2 | fixed | — | 139.16 | — | — | — | — | — |
| h1_2 | floating | — | 231.02 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.34 | — | — | — | — | — |
| iiwa14 | floating | — | 59.28 | — | — | — | — | — |
| go2 | fixed | — | 32.78 | — | — | — | — | — |
| go2 | floating | — | 64.20 | — | — | — | — | — |
| g1 | fixed | — | 105.77 | — | — | — | — | — |
| g1 | floating | — | 181.24 | — | — | — | — | — |
| h1_2 | fixed | — | 145.94 | — | — | — | — | — |
| h1_2 | floating | — | 292.96 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 47.43 | — | — | — | — | — |
| iiwa14 | floating | — | 102.45 | — | — | — | — | — |
| go2 | fixed | — | 50.53 | — | — | — | — | — |
| go2 | floating | — | 112.00 | — | — | — | — | — |
| g1 | fixed | — | 215.13 | — | — | — | — | — |
| g1 | floating | — | 382.16 | — | — | — | — | — |
| h1_2 | fixed | — | 290.16 | — | — | — | — | — |
| h1_2 | floating | — | 739.36 | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.79 | — | — | — | — | — |
| iiwa14 | floating | — | 33.54 | — | — | — | — | — |
| go2 | fixed | — | 10.11 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 42.07 | — | — | — | — | — |
| g1 | floating | — | 76.24 | — | — | — | — | — |
| h1_2 | fixed | — | 67.11 | — | — | — | — | — |
| h1_2 | floating | — | 109.37 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 22.68 | — | — | — | — | — |
| iiwa14 | floating | — | 48.19 | — | — | — | — | — |
| go2 | fixed | — | 25.62 | — | — | — | — | — |
| go2 | floating | — | 51.41 | — | — | — | — | — |
| g1 | fixed | — | 70.08 | — | — | — | — | — |
| g1 | floating | — | 106.34 | — | — | — | — | — |
| h1_2 | fixed | — | 85.12 | — | — | — | — | — |
| h1_2 | floating | — | 141.75 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.51 | — | — | — | — | — |
| iiwa14 | floating | — | 81.88 | — | — | — | — | — |
| go2 | fixed | — | 35.70 | — | — | — | — | — |
| go2 | floating | — | 88.53 | — | — | — | — | — |
| g1 | fixed | — | 105.57 | — | — | — | — | — |
| g1 | floating | — | 213.96 | — | — | — | — | — |
| h1_2 | fixed | — | 155.79 | — | — | — | — | — |
| h1_2 | floating | — | 198.48 | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.06 | — | — | — | — | — |
| iiwa14 | floating | — | 49.30 | — | — | — | — | — |
| go2 | fixed | — | 18.01 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 79.62 | — | — | — | — | — |
| g1 | floating | — | 152.49 | — | — | — | — | — |
| h1_2 | fixed | — | 149.80 | — | — | — | — | — |
| h1_2 | floating | — | 284.88 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.64 | — | — | — | — | — |
| iiwa14 | floating | — | 65.45 | — | — | — | — | — |
| go2 | fixed | — | 33.23 | — | — | — | — | — |
| go2 | floating | — | 67.59 | — | — | — | — | — |
| g1 | fixed | — | 107.61 | — | — | — | — | — |
| g1 | floating | — | 190.78 | — | — | — | — | — |
| h1_2 | fixed | — | 161.88 | — | — | — | — | — |
| h1_2 | floating | — | 324.42 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 47.85 | — | — | — | — | — |
| iiwa14 | floating | — | 115.03 | — | — | — | — | — |
| go2 | fixed | — | 50.99 | — | — | — | — | — |
| go2 | floating | — | 119.75 | — | — | — | — | — |
| g1 | fixed | — | 221.26 | — | — | — | — | — |
| g1 | floating | — | 411.46 | — | — | — | — | — |
| h1_2 | fixed | — | 326.67 | — | — | — | — | — |
| h1_2 | floating | — | 845.33 | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.06 | — | — | — | — | — |
| iiwa14 | floating | — | 51.44 | — | — | — | — | — |
| go2 | fixed | — | 18.24 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 79.51 | — | — | — | — | — |
| g1 | floating | — | 153.16 | — | — | — | — | — |
| h1_2 | fixed | — | 149.97 | — | — | — | — | — |
| h1_2 | floating | — | 284.57 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 31.61 | — | — | — | — | — |
| iiwa14 | floating | — | 68.27 | — | — | — | — | — |
| go2 | fixed | — | 33.44 | — | — | — | — | — |
| go2 | floating | — | 75.23 | — | — | — | — | — |
| g1 | fixed | — | 107.57 | — | — | — | — | — |
| g1 | floating | — | 191.59 | — | — | — | — | — |
| h1_2 | fixed | — | 161.94 | — | — | — | — | — |
| h1_2 | floating | — | 334.58 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 48.36 | — | — | — | — | — |
| iiwa14 | floating | — | 121.03 | — | — | — | — | — |
| go2 | fixed | — | 51.63 | — | — | — | — | — |
| go2 | floating | — | 137.53 | — | — | — | — | — |
| g1 | fixed | — | 221.73 | — | — | — | — | — |
| g1 | floating | — | 414.15 | — | — | — | — | — |
| h1_2 | fixed | — | 326.56 | — | — | — | — | — |
| h1_2 | floating | — | 876.92 | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.38 | — | — | — | — | — |
| iiwa14 | floating | — | 15.48 | — | — | — | — | — |
| go2 | fixed | — | 1.21 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 4.63 | — | — | — | — | — |
| g1 | floating | — | 18.75 | — | — | — | — | — |
| h1_2 | fixed | — | 7.78 | — | — | — | — | — |
| h1_2 | floating | — | 22.36 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.74 | — | — | — | — | — |
| iiwa14 | floating | — | 21.25 | — | — | — | — | — |
| go2 | fixed | — | 6.89 | — | — | — | — | — |
| go2 | floating | — | 21.02 | — | — | — | — | — |
| g1 | fixed | — | 10.38 | — | — | — | — | — |
| g1 | floating | — | 25.61 | — | — | — | — | — |
| h1_2 | fixed | — | 14.05 | — | — | — | — | — |
| h1_2 | floating | — | 29.70 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.93 | — | — | — | — | — |
| iiwa14 | floating | — | 37.13 | — | — | — | — | — |
| go2 | fixed | — | 7.29 | — | — | — | — | — |
| go2 | floating | — | 36.93 | — | — | — | — | — |
| g1 | fixed | — | 15.88 | — | — | — | — | — |
| g1 | floating | — | 44.74 | — | — | — | — | — |
| h1_2 | fixed | — | 23.28 | — | — | — | — | — |
| h1_2 | floating | — | 52.52 | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.64 | — | — | — | — | — |
| iiwa14 | floating | — | 260.30 | — | — | — | — | — |
| go2 | fixed | — | 1.87 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 12.40 | — | — | — | — | — |
| g1 | floating | — | 284.68 | — | — | — | — | — |
| h1_2 | fixed | — | 32.69 | — | — | — | — | — |
| h1_2 | floating | — | 312.53 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.33 | — | — | — | — | — |
| iiwa14 | floating | — | 276.68 | — | — | — | — | — |
| go2 | fixed | — | 15.98 | — | — | — | — | — |
| go2 | floating | — | 278.87 | — | — | — | — | — |
| g1 | fixed | — | 29.87 | — | — | — | — | — |
| g1 | floating | — | 290.43 | — | — | — | — | — |
| h1_2 | fixed | — | 48.28 | — | — | — | — | — |
| h1_2 | floating | — | 317.75 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.75 | — | — | — | — | — |
| iiwa14 | floating | — | 541.80 | — | — | — | — | — |
| go2 | fixed | — | 16.49 | — | — | — | — | — |
| go2 | floating | — | 543.25 | — | — | — | — | — |
| g1 | fixed | — | 42.59 | — | — | — | — | — |
| g1 | floating | — | 563.17 | — | — | — | — | — |
| h1_2 | fixed | — | 85.13 | — | — | — | — | — |
| h1_2 | floating | — | 623.22 | — | — | — | — | — |

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
| iiwa14 | fixed | — | 26.13 | — | — | — | — | — |
| iiwa14 | floating | — | 366.29 | — | — | — | — | — |
| go2 | fixed | — | 36.39 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1287.83 | — | — | — | — | — |
| g1 | floating | — | 1428.09 | — | — | — | — | — |
| h1_2 | fixed | — | 5825.50 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 32.95 | — | — | — | — | — |
| iiwa14 | floating | — | 378.40 | — | — | — | — | — |
| go2 | fixed | — | 44.97 | — | — | — | — | — |
| go2 | floating | — | 564.08 | — | — | — | — | — |
| g1 | fixed | — | 1338.57 | — | — | — | — | — |
| g1 | floating | — | 1489.51 | — | — | — | — | — |
| h1_2 | fixed | — | 9811.73 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.56 | — | — | — | — | — |
| iiwa14 | floating | — | 752.21 | — | — | — | — | — |
| go2 | fixed | — | 87.82 | — | — | — | — | — |
| go2 | floating | — | 1134.03 | — | — | — | — | — |
| g1 | fixed | — | 3000.63 | — | — | — | — | — |
| g1 | floating | — | 3278.93 | — | — | — | — | — |
| h1_2 | fixed | — | 21686.66 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.01 | — | — | — | — | — |
| iiwa14 | floating | — | 2625.86 | — | — | — | — | — |
| go2 | fixed | — | 36.34 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1294.87 | — | — | — | — | — |
| g1 | floating | — | 28189.80 | — | — | — | — | — |
| h1_2 | fixed | — | 5815.47 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 32.98 | — | — | — | — | — |
| iiwa14 | floating | — | 2680.23 | — | — | — | — | — |
| go2 | fixed | — | 44.98 | — | — | — | — | — |
| go2 | floating | — | 3948.61 | — | — | — | — | — |
| g1 | fixed | — | 1338.66 | — | — | — | — | — |
| g1 | floating | — | 29784.00 | — | — | — | — | — |
| h1_2 | fixed | — | 9810.31 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.59 | — | — | — | — | — |
| iiwa14 | floating | — | 5328.21 | — | — | — | — | — |
| go2 | fixed | — | 87.89 | — | — | — | — | — |
| go2 | floating | — | 7766.31 | — | — | — | — | — |
| g1 | fixed | — | 3000.35 | — | — | — | — | — |
| g1 | floating | — | 74590.54 | — | — | — | — | — |
| h1_2 | fixed | — | 21658.47 | — | — | — | — | — |
| h1_2 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 234.15 | — | — | — | — | — |
| iiwa14 | floating | — | 365.99 | — | — | — | — | — |
| go2 | fixed | — | 377.56 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 1193.57 | — | — | — | — | — |
| g1 | floating | — | 1422.17 | — | — | — | — | — |
| h1_2 | fixed | — | 2400.14 | — | — | — | — | — |
| h1_2 | floating | — | 2798.52 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 240.72 | — | — | — | — | — |
| iiwa14 | floating | — | 378.44 | — | — | — | — | — |
| go2 | fixed | — | 389.73 | — | — | — | — | — |
| go2 | floating | — | 564.06 | — | — | — | — | — |
| g1 | fixed | — | 1259.57 | — | — | — | — | — |
| g1 | floating | — | 1489.45 | — | — | — | — | — |
| h1_2 | fixed | — | 2454.29 | — | — | — | — | — |
| h1_2 | floating | — | 2955.24 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 476.79 | — | — | — | — | — |
| iiwa14 | floating | — | 752.31 | — | — | — | — | — |
| go2 | fixed | — | 772.85 | — | — | — | — | — |
| go2 | floating | — | 1133.96 | — | — | — | — | — |
| g1 | fixed | — | 2561.25 | — | — | — | — | — |
| g1 | floating | — | 3277.75 | — | — | — | — | — |
| h1_2 | fixed | — | 6215.75 | — | — | — | — | — |
| h1_2 | floating | — | 7471.09 | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 49.80 | — | — | — | — | — |
| iiwa14 | floating | — | 544.42 | — | — | — | — | — |
| go2 | fixed | — | 67.44 | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | 3253.77 | — | — | — | — | — |
| g1 | floating | — | 6990.65 | — | — | — | — | — |
| h1_2 | fixed | — | 17727.66 | — | — | — | — | — |
| h1_2 | floating | — | 21498.79 | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 58.67 | — | — | — | — | — |
| iiwa14 | floating | — | 535.12 | — | — | — | — | — |
| go2 | fixed | — | 77.93 | — | — | — | — | — |
| go2 | floating | — | 733.42 | — | — | — | — | — |
| g1 | fixed | — | 3664.53 | — | — | — | — | — |
| g1 | floating | — | 7536.08 | — | — | — | — | — |
| h1_2 | fixed | — | 28692.36 | — | — | — | — | — |
| h1_2 | floating | — | 33277.51 | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 108.60 | — | — | — | — | — |
| iiwa14 | floating | — | 1060.95 | — | — | — | — | — |
| go2 | fixed | — | 149.27 | — | — | — | — | — |
| go2 | floating | — | 1479.54 | — | — | — | — | — |
| g1 | fixed | — | 9258.44 | — | — | — | — | — |
| g1 | floating | — | 19711.12 | — | — | — | — | — |
| h1_2 | fixed | — | 81726.30 | — | — | — | — | — |
| h1_2 | floating | — | 166229.40 | — | — | — | — | — |

