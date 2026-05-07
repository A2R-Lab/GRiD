# GRiD Performance Benchmarks

**Machine**: plancher-asus25  
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (cc 12.0, CUDA 12.9)  
**CPU**: AMD Ryzen AI 9 HX 370 w/ Radeon 890M  
**Date**: 2026-05-07  
**Pinocchio**: ?

All times in **µs**.  GRiD *single*: kernel loop internal repeats, one GPU launch.  GRiD *N=256 compute*: compute-only (no cudaMemcpy).  Pinocchio *N=256*: multi-threaded CPU (codegen where available).  MJX *N=256*: vmapped JAX on GPU, compute-only.  GRiD/Pin and GRiD/MJX speedup = baseline N=256 / GRiD N=256 compute-only.

## Core Dynamics

### ID (Inverse Dynamics)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 13.03 | — | — | 243.67 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### Minv (M⁻¹)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 27.47 | — | — | 132.49 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### FD (Minv+RNEA)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 22.08 | — | — | 104.60 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### ABA (Articulated Body)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 92.51 | — | — | 94.87 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### CRBA

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 21.09 | — | — | 57.58 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

## Gradients

### ID_DU (∂ID/∂q,v)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 10.19 | — | — | 107.66 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### FD_DU (∂FD/∂q,v)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 1.23 | — | — | 3.74 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

## Kinematics

### EE_POSE

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 23.66 | — | — | 119.44 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### EE_POSE_GRADIENT (Jacobian)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | 11.20 | — | — | 82.37 | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

## Second-Order

> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU — expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). The large GRiD speedup here is expected.

> **Note (FDSVA_SO)**: No Pinocchio equivalent — GRiD numbers only.

### IDSVA_SO (2nd-order ID)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

### FDSVA_SO (2nd-order FD)

| Robot | Base | GRiD single | Pin single | MJX single | GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |
|-------|------|:-----------:|:----------:|:---------:|:----------:|:---------:|:---------:|:--------:|:--------:|
| iiwa14 | fixed | — | — | — | — | — | — | — | — |
| iiwa14 | floating | — | — | — | — | — | — | — | — |
| go2 | fixed | — | — | — | — | — | — | — | — |
| go2 | floating | — | — | — | — | — | — | — | — |
| g1 | fixed | — | — | — | — | — | — | — | — |
| g1 | floating | — | — | — | — | — | — | — | — |

## cuRobo Reference

cuRobo (arxiv 2603.05493) does not expose a standalone dynamics API — dynamics kernels are fused into the motion-planning optimization loop and are not independently benchmarkable. Numbers from the paper are shown below for context (Table 2 from the cuRobo paper; `compute-only` column, RTX 3090).

| Algorithm | cuRobo (batch 1024, µs) | Notes |
|-----------|:-----------------------:|-------|
| ID | ~2.3 | Fused forward pass |
| FD | ~4.1 | Fused forward pass |
| ID_DU | ~8.7 | Fused Jacobian pass |

> Numbers from cuRobo paper; methodology differs from GRiD/Pinocchio benchmarks above.

---

## Appendix A: Mid-Range Laptop

*Results pending — run `python test/benchmarks/run_benchmarks.py` on a laptop and commit the updated `benchmark.md`.*

## Appendix B: High-End Jetson (AGX Orin)

> **Note (Unified Memory)**: On Jetson platforms, `cudaMemcpy` is a no-op for unified memory. The **with-memory** and **compute-only** numbers will be similar; compare on **compute-only**.

*Results pending.*

## Appendix C: Embedded Jetson (Nano / Orin NX)

> **Note (Unified Memory)**: On Jetson platforms, `cudaMemcpy` is a no-op for unified memory. The **with-memory** and **compute-only** numbers will be similar; compare on **compute-only**.

*Results pending.*

