# GRiD Multi-Version Benchmark Comparison

**Machine**: plancher-omen-26  
**GPU**: NVIDIA GeForce RTX 5090 (cc 12.0, CUDA 13.2)  
**CPU**: Intel(R) Core(TM) Ultra 9 285K  
**Date**: 2026-05-23  
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
| iiwa14 | fixed | — | 4.62 | — | — | — | — | — |
| iiwa14 | floating | — | 10.07 | — | — | — | — | — |
| go2 | fixed | — | 6.89 | — | — | — | — | — |
| go2 | floating | — | 13.24 | — | — | — | — | — |
| g1 | fixed | — | 22.72 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.25 | — | — | — | — | — |
| iiwa14 | floating | — | 15.17 | — | — | — | — | — |
| go2 | fixed | — | 12.71 | — | — | — | — | — |
| go2 | floating | — | 18.81 | — | — | — | — | — |
| g1 | fixed | — | 29.08 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.76 | — | — | — | — | — |
| iiwa14 | floating | — | 24.87 | — | — | — | — | — |
| go2 | fixed | — | 13.23 | — | — | — | — | — |
| go2 | floating | — | 31.79 | — | — | — | — | — |
| g1 | fixed | — | 51.82 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### Minv (M⁻¹)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.45 | — | — | — | — | — |
| iiwa14 | floating | — | 27.91 | — | — | — | — | — |
| go2 | fixed | — | 8.00 | — | — | — | — | — |
| go2 | floating | — | 32.14 | — | — | — | — | — |
| g1 | fixed | — | 39.00 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 12.46 | — | — | — | — | — |
| iiwa14 | floating | — | 35.98 | — | — | — | — | — |
| go2 | fixed | — | 15.10 | — | — | — | — | — |
| go2 | floating | — | 40.00 | — | — | — | — | — |
| g1 | fixed | — | 59.22 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.25 | — | — | — | — | — |
| iiwa14 | floating | — | 64.15 | — | — | — | — | — |
| go2 | fixed | — | 23.44 | — | — | — | — | — |
| go2 | floating | — | 73.10 | — | — | — | — | — |
| g1 | fixed | — | 112.30 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### FD (Minv+RNEA)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.80 | — | — | — | — | — |
| iiwa14 | floating | — | 33.04 | — | — | — | — | — |
| go2 | fixed | — | 10.84 | — | — | — | — | — |
| go2 | floating | — | 38.19 | — | — | — | — | — |
| g1 | fixed | — | 53.10 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.60 | — | — | — | — | — |
| iiwa14 | floating | — | 39.58 | — | — | — | — | — |
| go2 | fixed | — | 18.90 | — | — | — | — | — |
| go2 | floating | — | 44.45 | — | — | — | — | — |
| g1 | fixed | — | 75.96 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 15.29 | — | — | — | — | — |
| iiwa14 | floating | — | 72.22 | — | — | — | — | — |
| go2 | fixed | — | 29.86 | — | — | — | — | — |
| go2 | floating | — | 82.30 | — | — | — | — | — |
| g1 | fixed | — | 136.51 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### ABA (Articulated Body)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 8.27 | — | — | — | — | — |
| iiwa14 | floating | — | 50.94 | — | — | — | — | — |
| go2 | fixed | — | 10.30 | — | — | — | — | — |
| go2 | floating | — | 60.85 | — | — | — | — | — |
| g1 | fixed | — | 38.48 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.38 | — | — | — | — | — |
| iiwa14 | floating | — | 56.82 | — | — | — | — | — |
| go2 | fixed | — | 17.71 | — | — | — | — | — |
| go2 | floating | — | 66.72 | — | — | — | — | — |
| g1 | fixed | — | 45.95 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.98 | — | — | — | — | — |
| iiwa14 | floating | — | 107.42 | — | — | — | — | — |
| go2 | fixed | — | 27.48 | — | — | — | — | — |
| go2 | floating | — | 126.89 | — | — | — | — | — |
| g1 | fixed | — | 84.11 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### CRBA

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 4.59 | — | — | — | — | — |
| iiwa14 | floating | — | 13.60 | — | — | — | — | — |
| go2 | fixed | — | 6.46 | — | — | — | — | — |
| go2 | floating | — | 16.03 | — | — | — | — | — |
| g1 | fixed | — | 26.91 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 10.46 | — | — | — | — | — |
| iiwa14 | floating | — | 19.28 | — | — | — | — | — |
| go2 | fixed | — | 12.52 | — | — | — | — | — |
| go2 | floating | — | 23.38 | — | — | — | — | — |
| g1 | fixed | — | 33.92 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 11.03 | — | — | — | — | — |
| iiwa14 | floating | — | 32.38 | — | — | — | — | — |
| go2 | fixed | — | 13.09 | — | — | — | — | — |
| go2 | floating | — | 39.51 | — | — | — | — | — |
| g1 | fixed | — | 60.72 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.57 | — | — | — | — | — |
| iiwa14 | floating | — | 16.07 | — | — | — | — | — |
| go2 | fixed | — | 9.64 | — | — | — | — | — |
| go2 | floating | — | 20.99 | — | — | — | — | — |
| g1 | fixed | — | 31.98 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 13.63 | — | — | — | — | — |
| iiwa14 | floating | — | 22.96 | — | — | — | — | — |
| go2 | fixed | — | 15.87 | — | — | — | — | — |
| go2 | floating | — | 27.88 | — | — | — | — | — |
| g1 | fixed | — | 38.63 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 14.51 | — | — | — | — | — |
| iiwa14 | floating | — | 38.44 | — | — | — | — | — |
| go2 | fixed | — | 16.58 | — | — | — | — | — |
| go2 | floating | — | 49.64 | — | — | — | — | — |
| g1 | fixed | — | 71.19 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 17.49 | — | — | — | — | — |
| iiwa14 | floating | — | 43.78 | — | — | — | — | — |
| go2 | fixed | — | 19.57 | — | — | — | — | — |
| go2 | floating | — | 51.35 | — | — | — | — | — |
| g1 | fixed | — | 79.54 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 23.16 | — | — | — | — | — |
| iiwa14 | floating | — | 50.13 | — | — | — | — | — |
| go2 | fixed | — | 26.01 | — | — | — | — | — |
| go2 | floating | — | 58.36 | — | — | — | — | — |
| g1 | fixed | — | 106.26 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 40.37 | — | — | — | — | — |
| iiwa14 | floating | — | 93.83 | — | — | — | — | — |
| go2 | fixed | — | 45.32 | — | — | — | — | — |
| go2 | floating | — | 110.05 | — | — | — | — | — |
| g1 | fixed | — | 239.08 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

## Integrators

### Integrator (x_{k+1})

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### Integrator_Gradient (∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

## Kinematics

### EE_POSE

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.39 | — | — | — | — | — |
| iiwa14 | floating | — | 15.65 | — | — | — | — | — |
| go2 | fixed | — | 1.21 | — | — | — | — | — |
| go2 | floating | — | 15.13 | — | — | — | — | — |
| g1 | fixed | — | 4.63 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.69 | — | — | — | — | — |
| iiwa14 | floating | — | 21.29 | — | — | — | — | — |
| go2 | fixed | — | 6.89 | — | — | — | — | — |
| go2 | floating | — | 21.03 | — | — | — | — | — |
| g1 | fixed | — | 10.40 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 6.97 | — | — | — | — | — |
| iiwa14 | floating | — | 37.12 | — | — | — | — | — |
| go2 | fixed | — | 7.23 | — | — | — | — | — |
| go2 | floating | — | 36.64 | — | — | — | — | — |
| g1 | fixed | — | 15.52 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 1.67 | — | — | — | — | — |
| iiwa14 | floating | — | 262.38 | — | — | — | — | — |
| go2 | fixed | — | 1.97 | — | — | — | — | — |
| go2 | floating | — | 263.04 | — | — | — | — | — |
| g1 | fixed | — | 13.64 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.19 | — | — | — | — | — |
| iiwa14 | floating | — | 269.50 | — | — | — | — | — |
| go2 | fixed | — | 7.77 | — | — | — | — | — |
| go2 | floating | — | 270.09 | — | — | — | — | — |
| g1 | fixed | — | 19.53 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 7.61 | — | — | — | — | — |
| iiwa14 | floating | — | 534.63 | — | — | — | — | — |
| go2 | fixed | — | 8.62 | — | — | — | — | — |
| go2 | floating | — | 536.00 | — | — | — | — | — |
| g1 | fixed | — | 33.94 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

## Second-Order

### IDSVA_SO (dispatched: body for fixed, world for floating)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.19 | — | — | — | — | — |
| iiwa14 | floating | — | 367.36 | — | — | — | — | — |
| go2 | fixed | — | 36.07 | — | — | — | — | — |
| go2 | floating | — | 548.52 | — | — | — | — | — |
| g1 | fixed | — | 1298.48 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.31 | — | — | — | — | — |
| iiwa14 | floating | — | 379.53 | — | — | — | — | — |
| go2 | fixed | — | 44.89 | — | — | — | — | — |
| go2 | floating | — | 564.60 | — | — | — | — | — |
| g1 | fixed | — | 1330.81 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 61.73 | — | — | — | — | — |
| iiwa14 | floating | — | 754.89 | — | — | — | — | — |
| go2 | fixed | — | 87.48 | — | — | — | — | — |
| go2 | floating | — | 1137.95 | — | — | — | — | — |
| g1 | fixed | — | 2997.48 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 26.54 | — | — | — | — | — |
| iiwa14 | floating | — | 2624.48 | — | — | — | — | — |
| go2 | fixed | — | 36.07 | — | — | — | — | — |
| go2 | floating | — | 3944.21 | — | — | — | — | — |
| g1 | fixed | — | 1293.33 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 33.33 | — | — | — | — | — |
| iiwa14 | floating | — | 2687.78 | — | — | — | — | — |
| go2 | fixed | — | 44.89 | — | — | — | — | — |
| go2 | floating | — | 3982.75 | — | — | — | — | — |
| g1 | fixed | — | 1330.94 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 64.39 | — | — | — | — | — |
| iiwa14 | floating | — | 5341.55 | — | — | — | — | — |
| go2 | fixed | — | 87.47 | — | — | — | — | — |
| go2 | floating | — | 7797.11 | — | — | — | — | — |
| g1 | fixed | — | 2997.90 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 236.96 | — | — | — | — | — |
| iiwa14 | floating | — | 367.21 | — | — | — | — | — |
| go2 | fixed | — | 376.05 | — | — | — | — | — |
| go2 | floating | — | 548.26 | — | — | — | — | — |
| g1 | fixed | — | 1207.32 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 243.87 | — | — | — | — | — |
| iiwa14 | floating | — | 379.61 | — | — | — | — | — |
| go2 | fixed | — | 388.67 | — | — | — | — | — |
| go2 | floating | — | 564.59 | — | — | — | — | — |
| g1 | fixed | — | 1255.28 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 480.33 | — | — | — | — | — |
| iiwa14 | floating | — | 754.90 | — | — | — | — | — |
| go2 | fixed | — | 771.27 | — | — | — | — | — |
| go2 | floating | — | 1138.01 | — | — | — | — | — |
| g1 | fixed | — | 2542.83 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

**single-call**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 49.53 | — | — | — | — | — |
| iiwa14 | floating | — | 527.65 | — | — | — | — | — |
| go2 | fixed | — | 65.17 | — | — | — | — | — |
| go2 | floating | — | 770.31 | — | — | — | — | — |
| g1 | fixed | — | 3221.13 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=16**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 55.61 | — | — | — | — | — |
| iiwa14 | floating | — | 526.39 | — | — | — | — | — |
| go2 | fixed | — | 82.23 | — | — | — | — | — |
| go2 | floating | — | 776.59 | — | — | — | — | — |
| g1 | fixed | — | 3680.24 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

**batch N=256**

| Robot | Base | pre_glass | glass | pin | mjx | frax_cpu | frax_gpu | glass/pre |
|-------|------|:---------:|:-----:|:---:|:---:|:--------:|:--------:|:---------:|
| iiwa14 | fixed | — | 105.04 | — | — | — | — | — |
| iiwa14 | floating | — | 1058.10 | — | — | — | — | — |
| go2 | fixed | — | 162.57 | — | — | — | — | — |
| go2 | floating | — | 1568.56 | — | — | — | — | — |
| g1 | fixed | — | 9225.77 | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — |

