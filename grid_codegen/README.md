# GRiD's code generator

The in-repository Python engine reads a model from `URDFParser` and emits
robot-specific CUDA plus the checked-in generated binding regions. It is part
of the root editable installation, not a separate submodule or package install.

```python
from grid_codegen import GRiDCodeGenerator

codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False)
codegen.gen_all_code(output_path="grid.cuh")
```

For command-line generation, use `grid-generate robot.urdf`. Subset builds
(`--algorithm-list`) and `--no-mujoco-kernels` reduce compilation cost.

## Where to change things

| Responsibility | Source |
|---|---|
| Algorithm emission and composition | `algorithms/`, `GRiDCodeGenerator.py` |
| Algorithm metadata and dependency profiles | `algo_registry.py`, `_algo_profiles.py` |
| Public binding shapes, arguments, differentiation | `abi_specs.py` |
| Shared/workspace arena sizing | `_constants_arena.py` |
| Generated C ABI, framework handlers and pybind bodies | `wrapper_body_gen.py`, `wrapper_plant_gen.py`, `core_body_gen.py` |
| Collision model preparation | `algorithms/_spherize.py`, `spherize.py` |

Regenerate binding regions with `make gen`; verify them with `make gen-check`.
Never hand-edit inside generated BEGIN/END markers. `make build` regenerates
before rebuilding the pybind extension.

## Canonical guides

- [Architecture](../docs/source/user_guide/concepts/codegen_architecture.rst):
  three external layers (device, kernel, host), plus internal compositional
  helpers. Scratch arguments vary by algorithm; consult generated signatures.
- [Adding an algorithm](../docs/source/user_guide/tutorials/adding_an_algorithm.rst):
  reference implementation, emitter, descriptors, ABI rows, tests and docs.
- [CUDA support](../docs/source/user_guide/tutorials/cuda_support_status.rst):
  operation-specific joint/model restrictions.
- [Collision workflow](../docs/source/user_guide/tutorials/collisions.rst):
  geometry preparation, fidelity reporting, and warp/block interfaces.

Byte-identical refactors should pass the byte gate. Numerical changes need
equivalence evidence; CPU drift checks alone do not establish CUDA correctness.
