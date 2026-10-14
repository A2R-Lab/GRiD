# GRiD
[![CI](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/verify-gpu-proof.yml?branch=main&style=flat-square&label=CI)](https://github.com/A2R-Lab/GRiD/actions/workflows/verify-gpu-proof.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/A2R-Lab/GRiD/gh-pages.yml?branch=main&style=flat-square&label=docs)](https://a2r-lab.org/GRiD/)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/pyproject.toml)
[![agent-ready](https://img.shields.io/badge/agent--ready-CLAUDE.md-8A2BE2?style=flat-square)](https://github.com/A2R-Lab/GRiD/blob/main/CLAUDE.md)
[![All Contributors](https://img.shields.io/github/all-contributors/A2R-Lab/GRiD?color=ee8449&style=flat-square)](#contributors)

A GPU-accelerated library for robot dynamics, kinematics, and collisions, with analytical derivatives and Hessians for supported numerical operations.

![The GRiD package ecosystem: a user's URDF goes through URDFParser to the code generator (built on GLASS) and RBDReference, producing CUDA C++ with NumPy, JAX, and PyTorch wrappers; benchmarks and tests, backed by pytest-gpu-proof and external oracles, produce validated outputs and performance benchmarks.](https://raw.githubusercontent.com/A2R-Lab/GRiD/main/docs/imgs/GRiD.png)

GRiD turns a URDF into optimized, per-robot CUDA C++ for rigid-body dynamics, kinematics, their analytical
first- and second-order derivatives and a trajectory-optimization plant layer, then hands you that code as a
NumPy handle, a `jax.jit`-able FFI surface, `torch.autograd`-aware ops, or one header-only `grid.cuh` for your own
kernels. One CUDA block per problem, batched, bit-deterministic and thread-count invariant; the same model and API
rebuild for each target architecture (one artifact per `sm_XX`). GRiD builds on our
[URDFParser](https://github.com/A2R-Lab/URDFParser), [RBDReference](https://github.com/A2R-Lab/RBDReference) and
[GLASS](https://github.com/A2R-Lab/GLASS) packages (URDF parsing, Pinocchio-validated reference dynamics, GPU linear algebra).

**📖 The full documentation is the website: <https://a2r-lab.org/GRiD/>** (installation, tutorials, the
[How do I…?](https://a2r-lab.org/GRiD/docs/how_do_i.html) index, concepts, API reference, benchmarks, validation). This README is only a router.

**Status:** 0.5.1 on [PyPI](https://pypi.org/project/grid-rbd/) (first packaged release: 0.5.0) — alpha software, APIs may
change between minor versions ([CHANGELOG.md](https://github.com/A2R-Lab/GRiD/blob/main/CHANGELOG.md)). Linux x86-64,
CPython 3.10–3.12, CUDA 12+/13; validated on an RTX 5090. The [original ICRA 2022 paper](https://a2r-lab.org/publication/grid/)
describes the implementation preserved in the archival [robot-acceleration/GRiD](https://github.com/robot-acceleration/GRiD)
repository, not the feature set or performance of this release.

## Two entry points
Both reach the same generated CUDA; pick the one that matches how you want to use it.

### Easy: Python wrappers
`load_robot(urdf)` generates, compiles and caches the robot's CUDA, then exposes it as batched NumPy, JAX or PyTorch
calls. No CUDA code to write.
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
→ [Python wrappers](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html) ·
[installation](https://a2r-lab.org/GRiD/docs/user_guide/getting_started/installation.html)

### Power user: header-only CUDA
Generate one self-contained `grid.cuh` and `#include` it in your own kernels. Needs only the generator and `nvcc`;
no per-robot `.so`, no Python at runtime.
```shell
pip install grid-rbd
grid-generate robot.urdf -o grid.cuh        # -f floating base, --algorithm-list to trim
nvcc -arch=sm_90 -std=c++17 my_kernel.cu    # the header vendors its GLASS linear algebra
```
→ [Generate CUDA for a robot](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/codegen.html) ·
[`examples/cuda/`](https://github.com/A2R-Lab/GRiD/tree/main/examples/cuda)

Installing needs no GPU or compiler; registering a robot compiles its CUDA on a machine with a compatible NVIDIA
GPU, CUDA toolkit and C++17 host compiler (the first load can take minutes; later loads reuse the cache). To develop
GRiD, or to use the sample URDFs in `grid_assets/robots/`, work from a checkout instead: `git clone --recursive
https://github.com/A2R-Lab/GRiD.git && cd GRiD && bash install/base_install.sh && source .venv/bin/activate`.

## Where things are documented

| Topic | Site page |
|-------|-----------|
| Everyday tasks, index of everything | [How do I…?](https://a2r-lab.org/GRiD/docs/how_do_i.html) |
| Python API (`register_robot`/`load_robot`, backends, env knobs, fp64) | [Python wrappers](https://a2r-lab.org/GRiD/docs/user_guide/tutorials/python_wrappers.html) |
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
| `bindings/` | the `grid-rbd` Python package (NumPy/JAX/torch handles over a cached per-robot `.so`) — [`bindings/README.md`](https://github.com/A2R-Lab/GRiD/blob/main/bindings/README.md) |
| `grid_assets/` | ten sample URDFs (`robots/`) + tuned per-GPU launch configs (`launch_configs/`) |
| `test/` | pytest suites + the split-suite/receipt machinery |
| `examples/` | `notebooks/` (Python tour), `codegen/`, `cuda/` — [`examples/README.md`](https://github.com/A2R-Lab/GRiD/blob/main/examples/README.md) |
| `docs/` | the Sphinx site (`source/`), the cover page (`landing/`), `agent_debugging_guide.md`, `design_history/` |
| `external/` · `install/` | peer submodules (`GLASS`, `RBDReference`, `URDFParser`) · install scripts and requirements files |

## Performance

![Core-operation speedups against seven baseline modes, with timing boundaries and fp64 exceptions labeled.](https://raw.githubusercontent.com/A2R-Lab/GRiD/main/docs/source/_static/release/speedup_core.png)

Release measurements (September–October 2026, one NVIDIA RTX 5090 + Intel Core Ultra 9 285K): RNEA, its analytical gradient
and Hessian on iiwa14, go2 and G1 at batch sizes 16–1024; ratios are baseline time over GRiD time, above 1× favors GRiD.
Method, every timing boundary and the caveats: [release measurements](https://a2r-lab.org/GRiD/docs/release_measurements.html);
the [benchmark harness](https://github.com/A2R-Lab/GRiD/tree/main/test/benchmarks/) reproduces the collection.

## Citing GRiD
Please cite our paper ["GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients"](https://a2r-lab.org/publication/grid/):
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
MIT — see [LICENSE](https://github.com/A2R-Lab/GRiD/blob/main/LICENSE).

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
      <td align="center" valign="top" width="20%"><a href="https://github.com/EmreAdabag"><img src="https://avatars.githubusercontent.com/EmreAdabag?s=100" width="100px;" alt="EmreAdabag"/><br /><sub><b>EmreAdabag</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/caelyasutake"><img src="https://avatars.githubusercontent.com/caelyasutake?s=100" width="100px;" alt="Cael Yasutake"/><br /><sub><b>Cael Yasutake</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/naren-loganathan"><img src="https://avatars.githubusercontent.com/naren-loganathan?s=100" width="100px;" alt="Naren Loganathan"/><br /><sub><b>Naren Loganathan</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/emilyburnett2003"><img src="https://avatars.githubusercontent.com/emilyburnett2003?s=100" width="100px;" alt="emilyburnett2003"/><br /><sub><b>emilyburnett2003</b></sub></a><br /></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pruyontrarakk"><img src="https://avatars.githubusercontent.com/pruyontrarakk?s=100" width="100px;" alt="pruyontrarakk"/><br /><sub><b>pruyontrarakk</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/A2R-Lab/GRiD/graphs/contributors"><img src="https://www.gravatar.com/avatar/b6a2a74133224e85e9714e282a1576df?d=identicon&s=100?s=100" width="100px;" alt="Kimiya Shahamat"/><br /><sub><b>Kimiya Shahamat</b></sub></a><br /></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/A2R-Lab/GRiD/graphs/contributors"><img src="https://www.gravatar.com/avatar/b6a2a74133224e85e9714e282a1576df?d=identicon&s=100?s=100" width="100px;" alt="Kimiya Shahamat"/><br /><sub><b>Kimiya Shahamat</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
