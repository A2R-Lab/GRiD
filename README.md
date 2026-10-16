# GRiD
[![CI](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/verify-gpu-proof.yml?branch=main&style=flat-square&label=CI)](https://github.com/A2R-Lab/GRiD/actions/workflows/verify-gpu-proof.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/gh-pages.yml?branch=main&style=flat-square&label=docs)](https://a2r-lab.org/GRiD/)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/pyproject.toml)
[![agent-ready](https://img.shields.io/badge/agent--ready-CLAUDE.md-8A2BE2?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/CLAUDE.md)
[![All Contributors](https://img.shields.io/github/all-contributors/A2R-Lab/GRiD?color=ee8449&style=flat-square)](#contributors)

GRiD is a GPU-accelerated library for robot dynamics, kinematics and collisions with analytical derivatives and Hessians.

![The GRiD package ecosystem: a user's URDF goes through URDFParser to the code generator (built on GLASS) and RBDReference, producing CUDA C++ with NumPy, JAX, and PyTorch wrappers; benchmarks and tests, backed by pytest-gpu-proof and external oracles, produce validated outputs and performance benchmarks.](https://raw.githubusercontent.com/A2R-Lab/GRiD/main/docs/imgs/GRiD.png)

GRiD generates optimized, robot-specific CUDA C++ from a URDF: rigid-body dynamics, kinematics, their analytical
first- and second-order derivatives, and a plant layer for trajectory optimization. The generated code is exposed as
a NumPy handle, a `jax.jit`-able FFI surface, `torch.autograd`-aware ops, or a single header-only `grid.cuh` for use
in your own kernels. Every algorithm runs as one CUDA block per problem and is batched across problems; results are
bit-deterministic and independent of the thread count. The same model and API rebuild for each target architecture,
with one artifact per `sm_XX`. GRiD builds on [URDFParser](https://github.com/A2R-Lab/URDFParser) for URDF parsing,
[RBDReference](https://github.com/A2R-Lab/RBDReference) for Pinocchio-validated reference dynamics and
[GLASS](https://github.com/A2R-Lab/GLASS) for GPU linear algebra.

The full documentation is the website, <https://a2r-lab.org/GRiD/>: installation, tutorials, the
[How do I…?](https://a2r-lab.org/GRiD/docs/how_do_i.html) index, concepts, API reference, benchmarks and validation.
This README points to it.

Version 0.5.1 is on [PyPI](https://pypi.org/project/grid-rbd/); 0.5.0 was the first packaged release. GRiD is alpha
software and APIs may change between minor versions (see the
[changelog](https://github.com/A2R-Lab/GRiD/blob/main/CHANGELOG.md)). The GPU test receipt and the release timing ran on
Linux x86-64 with CPython 3.12, CUDA toolkit 13.2 and one RTX 5090 (sm_120); CPython 3.10, 3.11 and 3.13 wheels are built and
import-checked on CPU in CI, and the CPU test lane runs on 3.10, 3.12 and 3.13. Other toolkits, GPUs and interpreters are untested; the
[compatibility page](https://a2r-lab.org/GRiD/docs/user_guide/getting_started/compatibility.html) lists the exact
environment. The [original ICRA 2022 paper](https://a2r-lab.org/publication/grid/)
describes the implementation preserved in the archival [robot-acceleration/GRiD](https://github.com/robot-acceleration/GRiD)
repository; it does not describe the feature set or the performance of this release.

## Two entry points
Both entry points use the same generated CUDA.

### Python wrappers
`load_robot(urdf)` generates, compiles and caches the robot's CUDA and exposes it as batched NumPy, JAX or PyTorch
calls. No CUDA code is required.
```shell
pip install grid-rbd            # code generator + NumPy backend + grid-generate CLI
pip install "grid-rbd[jax]"     # or "grid-rbd[torch]", "grid-rbd[all]"
```
```python
import numpy as np, grid_rbd
robot = grid_rbd.load_robot("grid_assets/robots/iiwa14.urdf",
                            algorithm_list=["forward_dynamics"], enable_mujoco_kernels=False)
q = np.zeros((8, robot.nq), dtype=np.float32)
qd, tau = (np.zeros((8, robot.nv), dtype=np.float32) for _ in range(2))
print(robot.forward_dynamics(q, qd, tau).shape)  # (8, 7)
```
See [Python wrappers](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html) and
[installation](https://a2r-lab.org/GRiD/docs/user_guide/getting_started/installation.html).
Deep tutorial: [gradient descent through GRiD with plain JAX](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/jax_ik_gradient_descent.html)
(`examples/jax/ik_gradient_descent.py`).

### Header-only CUDA
Generate one self-contained `grid.cuh` and `#include` it in your own kernels. This path needs only the generator and
`nvcc`: no per-robot `.so` and no Python at runtime.
```shell
pip install grid-rbd
grid-generate robot.urdf -o grid.cuh        # -f floating base, --algorithm-list to trim
nvcc -arch=sm_90 -std=c++17 my_kernel.cu    # the header vendors its GLASS linear algebra
```
See [Generate CUDA for a robot](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/codegen.html) and
[`examples/cuda/`](https://github.com/A2R-Lab/GRiD/tree/main/examples/cuda).
Deep tutorial: [fuse dynamics into your own kernel](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/fused_cuda_control_step.html)
(`examples/cuda/fused_control_step/`, checked and timed against the host-function path).

Installation needs no GPU or compiler. Registering a robot compiles its CUDA on a machine with a compatible NVIDIA
GPU, CUDA toolkit and C++17 host compiler; the first load takes minutes and later loads reuse the cache. To develop
GRiD, or to use the sample URDFs in `grid_assets/robots/`, work from a checkout: `git clone --recursive
https://github.com/A2R-Lab/GRiD.git && cd GRiD && bash install/base_install.sh && source .venv/bin/activate`.

A Docker image with grid-rbd, `jax[cuda12]`, a cu128 torch wheel and the generated iiwa14 header is published as
`ghcr.io/a2r-lab/grid:<version>` (`install/docker/Dockerfile`; run with `--gpus all`). A conda recipe
(`install/environment.yml`) and a uv recipe are on the
[installation](https://a2r-lab.org/GRiD/docs/user_guide/getting_started/installation.html) page; neither is tested in CI.

## Where things are documented

| Topic | Site page |
|-------|-----------|
| Everyday tasks, index of everything | [How do I…?](https://a2r-lab.org/GRiD/docs/how_do_i.html) |
| Python API (`register_robot`/`load_robot`, backends, env knobs, fp64 alpha) | [Python wrappers](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html) |
| `grid-generate` CLI, the generated header, debug/print helpers | [Generate CUDA for a robot](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/codegen.html) |
| Floating-base conventions (`pinocchio` / `legacy` orderings), URDF parsing | [URDF API](https://a2r-lab.org/GRiD/docs/api_reference/urdf.html) |
| Supported joints, algorithms and restrictions; per-backend inventory | [CUDA support status](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/cuda_support_status.html) · [backend coverage](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/backend_coverage.html) |
| C++ API layers (`*_device` / `*_kernel` / host wrapper) | [codegen architecture](https://a2r-lab.org/GRiD/docs/user_guide/concepts/codegen_architecture.html) |
| Tests, signed GPU receipts, fixing a red receipt CI job | [CUDA validation](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/cuda_validation.html) |
| Benchmark harness, bench-only troubleshooting (e.g. `--ptxas-opt-level`) | [benchmarks](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/benchmarks.html) |
| Release measurements and their method | [release measurements](https://a2r-lab.org/GRiD/docs/release_measurements.html) |
| Contributing, adding an algorithm | [contribution guidelines](https://a2r-lab.org/GRiD/docs/contribution_guidelines.html) · [adding an algorithm](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/adding_an_algorithm.html) · [CONTRIBUTING.md](https://github.com/A2R-Lab/GRiD/blob/main/CONTRIBUTING.md) |

## Repo map

| Directory | Owns |
|-----------|------|
| `grid_codegen/` | the code-generation engine: emits `grid.cuh` and the checked-in generated binding regions, driven by the `abi_specs.py` table |
| `bindings/` | the `grid-rbd` Python package (NumPy/JAX/torch handles over a cached per-robot `.so`); see [`bindings/README.md`](https://github.com/A2R-Lab/GRiD/blob/main/bindings/README.md) |
| `grid_assets/` | ten sample URDFs (`robots/`) + tuned per-GPU launch configs (`launch_configs/`) |
| `test/` | pytest suites + the split-suite/receipt machinery |
| `examples/` | `notebooks/` (Python tour), `codegen/`, `cuda/`; see [`examples/README.md`](https://github.com/A2R-Lab/GRiD/blob/main/examples/README.md) |
| `docs/` | the Sphinx site (`source/`), the cover page (`landing/`), `agent_debugging_guide.md`, `design_history/` |
| `external/`, `install/` | peer submodules (`GLASS`, `RBDReference`, `URDFParser`); install scripts and requirements files |

## Performance

![Core-operation speedups against seven baseline modes, with timing boundaries and fp64 exceptions labeled.](https://raw.githubusercontent.com/A2R-Lab/GRiD/main/docs/source/_static/release/speedup_core.png)

The release measurements (September and October 2026, one NVIDIA RTX 5090 and an Intel Core Ultra 9 285K) cover RNEA,
its analytical gradient and its Hessian on iiwa14, go2 and G1 at batch sizes 16 to 1024. Ratios are baseline time over
GRiD time, so values above 1x favor GRiD. The method, every timing boundary and the caveats are on the
[release measurements](https://a2r-lab.org/GRiD/docs/release_measurements.html) page, and the
[benchmark harness](https://github.com/A2R-Lab/GRiD/tree/main/test/benchmarks/) reproduces the collection.

## Citing GRiD
If you use GRiD in your research, please cite our paper ["GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients"](https://a2r-lab.org/publication/grid/):
```
@inproceedings{plancher2022grid,
  title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients}, 
  author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2022}, 
  month={May}
}
```

## License
MIT; see [LICENSE](https://github.com/A2R-Lab/GRiD/blob/main/LICENSE).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/plancherb1"><img src="https://avatars.githubusercontent.com/plancherb1?s=100" width="100px;" alt="Brian Plancher"/><br /><sub><b>Brian Plancher</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/Z4KH"><img src="https://avatars.githubusercontent.com/Z4KH?s=100" width="100px;" alt="Zachary Pestrikov"/><br /><sub><b>Zachary Pestrikov</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/kawotwi"><img src="https://avatars.githubusercontent.com/kawotwi?s=100" width="100px;" alt="Kwamena A"/><br /><sub><b>Kwamena A</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/A2R-Lab/GRiD/graphs/contributors"><img src="https://www.gravatar.com/avatar/b619b0ff13333ce2a22bb110eda8f7a9?d=identicon&s=100?s=100" width="100px;" alt="Danelle Tuchman"/><br /><sub><b>Danelle Tuchman</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/anncli"><img src="https://avatars.githubusercontent.com/anncli?s=100" width="100px;" alt="Ann Li"/><br /><sub><b>Ann Li</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/the-eater"><img src="https://avatars.githubusercontent.com/the-eater?s=100" width="100px;" alt="="/><br /><sub><b>=</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/caelyasutake"><img src="https://avatars.githubusercontent.com/caelyasutake?s=100" width="100px;" alt="Cael Yasutake"/><br /><sub><b>Cael Yasutake</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/naren-loganathan"><img src="https://avatars.githubusercontent.com/naren-loganathan?s=100" width="100px;" alt="Naren Loganathan"/><br /><sub><b>Naren Loganathan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/EmreAdabag"><img src="https://avatars.githubusercontent.com/EmreAdabag?s=100" width="100px;" alt="EmreAdabag"/><br /><sub><b>EmreAdabag</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/emilyburnett2003"><img src="https://avatars.githubusercontent.com/emilyburnett2003?s=100" width="100px;" alt="emilyburnett2003"/><br /><sub><b>emilyburnett2003</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pruyontrarakk"><img src="https://avatars.githubusercontent.com/pruyontrarakk?s=100" width="100px;" alt="pruyontrarakk"/><br /><sub><b>pruyontrarakk</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
