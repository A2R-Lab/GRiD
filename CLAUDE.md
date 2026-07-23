# CLAUDE.md — agent & contributor onboarding

Orientation for an AI agent (or a new human) working in this repo. Read this first, then the
deeper docs it points at. This file is tracked; per-machine/session notes live in the (gitignored)
`docs/STARTUP_PROMPT.md`.

## What GRiD is

A GPU-accelerated rigid body dynamics library. GRiD reads a URDF and **generates** optimized,
per-robot CUDA C++ (`grid.cuh`) implementing forward/inverse dynamics, their analytical gradients,
second-order derivatives, kinematics, centroidal quantities, and a trajectory-optimization
`grid_plant` layer. Every algorithm exists on **two surfaces tested for numerical agreement**: a
numpy oracle in `RBDReference` (validated against Pinocchio) and the generated CUDA kernels
(validated against that oracle). Keep that invariant sacred.

## Layout

- `grid_codegen/` — the code-generation engine (the heart of GRiD); emits `grid.cuh`. Includes
  `grid_codegen/collision/` (collision-geometry SDF header + spherized assets for two-tier `config_free`).
- `external/` — the peer-product submodules `GLASS/`, `RBDReference/`, `URDFParser/` (GPU linear
  algebra, Pinocchio-validated reference dynamics, URDF parsing). **Four separate peer products** with
  GRiD, all under `A2R-Lab`; grouped here so the top level stays about GRiD itself.
- `bindings/` — the `grid-rbd` Python package: `register_robot(...)` → numpy / jax / torch handles.
  Agent-facing API tour: `bindings/examples/AGENT_INTEGRATION_GUIDE.md`.
- `examples/` — start here: `examples/notebooks/` (Python-wrapper walkthroughs, the "start here" track;
  see `examples/README.md`), `examples/codegen/` + `examples/cuda/` (generate + validate walkthroughs).
- `test/` — pytest suites (see markers below); `test/cuda_equivalents/` (CUDA equivalence runners),
  `test/python_wrappers/` (jax/torch), `test/benchmarks/` (timing harness + orchestration scripts).
- `docs/source/` — Sphinx docs (published to https://a2r-lab.github.io/GRiD/).
- `docs/agent_debugging_guide.md` — **the** debugging bible: enumerated bug classes (smem init,
  beta==0 output-reads, reduction nondeterminism, false-positive racecheck, …). Append learnings here.
- `docs/open-tasks/` — gitignored planning docs / SSOT ledgers (local-only, survive across sessions).

## Generate code

```bash
bash install/base_install.sh && source .venv/bin/activate      # single `pip install -e .` + `grid-generate` CLI
grid-generate path/to/robot.urdf [-f] [-t EE_JOINT] [-n NAMESPACE] [-c] [-d]
.venv/bin/python examples/codegen/generate_iiwa14.py             # fixed-base example
.venv/bin/python examples/codegen/generate_go2_floating.py       # floating-base example
```

Always use `.venv/bin/python` (never bare `python`). The editable install puts `grid_codegen`,
`URDFParser`, `RBDReference`, and `grid_rbd` on the venv path, so no `PYTHONPATH` is needed. Box CUDA
arch: `sm_120`/`compute_120` (per-machine notes belong in the gitignored `docs/STARTUP_PROMPT.md`).
Clear `grid_codegen/__pycache__` after codegen changes.

## Test

```bash
.venv/bin/python -m pytest -q                                   # full suite
.venv/bin/python -m pytest -m cuda_equivalence -q               # CUDA-vs-numpy equivalence (needs a GPU)
.venv/bin/python -m pytest -m python_wrappers -q                # jax/torch handles (needs GPU + grid-rbd)
.venv/bin/python -m pytest -m pinocchio_equivalence -q          # numpy oracle vs Pinocchio (CPU)
```

Markers: `pinocchio_equivalence`, `cuda_equivalence`, `python_wrappers`, `floating_base`,
`robot_{smoke,curated,nightly}`, `gpu_proof`, `notebooks`, `developer_only`. GPU test outcomes are captured in a signed
`gpu-proof.json` receipt (see `test/run_gpu_proof.sh`) that CPU-only CI verifies — so GPU correctness
can gate merges without paid GPU CI.

## Durable engineering conventions

- **Single-block per kernel/robot, always** — no multi-block / cooperative groups. Big-robot
  performance comes from in-block parallelism only.
- **Thread-count invariant** — a kernel's output must be identical at 1 / 32 / any thread count.
  Test it. Floating-base reductions must also be **run-to-run bit-deterministic** (fixed-order sums).
- **Byte-identical codegen discipline** — a refactor that shouldn't change emitted code must produce
  a byte-identical `grid.cuh` (regen before/after + `diff`). Never advance on a non-identical diff
  without a CUDA-equivalence sign-off.
- **Fix, don't guard** — no `xfail`/`skip`/defensive guards; fix the root cause.
- **Physics**: gravity `-9.81`; Pinocchio is authoritative. Prefer extending GLASS primitives over
  working around them (GLASS is first-party).
- **Verify yourself** before committing — re-run the sanitizers / equivalence / poison harness; don't
  trust a subagent's "done" (subagents can lose Bash mid-run).
- **Git**: short single-line commit messages, no Co-Authored-By footer; path-scoped `git add` (never
  `-A`/`.`); submodules committed/pushed before the parent pointer bump; push only when asked.

## Deeper reading

`docs/source/user_guide/concepts/{design_principles,codegen_architecture,resource_tier_system}.rst`
for the codegen architecture (the `*_inner`/`*_device`/`*_kernel`/host layer stack, the shared-memory
tier/spill system) and `docs/agent_debugging_guide.md` for the accumulated bug classes.
