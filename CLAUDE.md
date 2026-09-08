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

- `grid_codegen/` — the code-generation engine (the heart of GRiD); emits `grid.cuh` AND the three
  checked-in generated regions of `bindings/grid_rbd/wrapper_template.cu` (C-ABI bodies,
  kernel_max_threads table, mjx twins — driven by `abi_specs.py` via `wrapper_body_gen.py`;
  regenerate with `.venv/bin/python -m grid_codegen.wrapper_body_gen`, never hand-edit inside
  the BEGIN/END markers). Includes
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

Prefer the crash-isolated split driver for full GPU passes:
`test/run_split_suite.py` runs `python_wrappers` as per-module shards (compile-warm
phase + one pytest subprocess per module — one module's abort can't eat the rest;
`--changed-only` skips modules whose input fingerprint matches the last green run)
and, with `--domains wrappers,cuda`, partitions `cuda_equivalents` into granular
node-id shards bounded to ~2h each (bin-packed from rolling measured durations;
never `-k` — explicit node-id lists with a set-equality completeness gate). Runs
are pausable: `touch <out>/PAUSE` stops cleanly between shards, Ctrl-C/SIGTERM
stops within one, and `--resume <out>` continues without re-running completed
shards. `SPLIT=1 test/run_gpu_proof.sh` drives both domains and merges + re-signs
all shard receipts into the same repo-root `gpu-proof.json` the monolithic path
writes (`SPLIT_RESUME=<out>` to continue an interrupted pass). CPU-only gates for
the partition logic live in `test/test_split_partition.py`.
Compiles run through a RAM-aware parallel pool (`test/compile_sched.py`): Phase A
wrapper `.so` warms and cuda flagship header/exe pre-warms
(`test/prewarm_cuda_flagship.py` — imports the test module's own compile chain so
cache keys match by construction) execute as admission-controlled parallel
workers (predicted peak RSS from a rolling `/usr/bin/time -v` ledger at
`test/.split_suite/compile_rss.json`, conservative default + margin + MemAvailable
floor) and OVERLAP GPU shard execution — a shard only waits for its own compile
jobs, so the pool is the sole writer of a cache key until that shard starts (the
unlocked cache writers are never raced; a mimic robot's cells share one header and
fold into one serial job). `GRID_SPLIT_COMPILE_JOBS` sizes the pool (default 5;
`0` = legacy serial inline warm).
**Receipt policies (two-tier, user decision 2026-08-20).** The committed
`gpu-proof.json` goes stale — and CI's verify-receipt job goes RED — the
moment a push touches fingerprinted test files (`test/cuda_equivalents/`,
`test/python_wrappers/`). That red is by design, and the fix is a refresh:
- **Everyday** (`test/gpu-proof-policy.yaml`, `allow_carried: true`):
  `SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh` re-runs ONLY the shards
  whose narrow fingerprints changed vs the committed receipt and CARRIES the
  rest (`gpu-proof merge --carry-from`); commit the refreshed receipt and CI
  goes green — minutes-to-hours, not the full pass. A carried shard attests
  "these tests, whose files are unchanged, passed at an ancestor commit ≤30
  days old"; cross-cutting codegen/bindings changes are NOT re-proven by
  carry. `GRID_SPLIT_REFRESH_DRY=1` previews the stale/carried plan.
- **Release** (`test/gpu-proof-policy-release.yaml`, `allow_carried: false`):
  refuses carried shards outright, so a release receipt requires ONE fresh
  full `SPLIT=1` pass at the release tip — every shard executed at that exact
  commit. Verify with `--policy test/gpu-proof-policy-release.yaml`.
Internals a maintainer should know: stale shard NAMES are recycled into the
fresh partition automatically (the carry contract has no "superseded" state),
a deleted test module forces a full pass, and both compile caches are
CONTENT-keyed (header/source bytes), so byte-identical codegen edits cost
seconds of regeneration, never an nvcc rebuild.

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
- **Big floating-base builds are mjx-dominated.** On a floating, non-mimic robot GRiD also emits a
  MuJoCo-convention ("mjx") twin of each kernel; the second-order twins used to dwarf their pin
  kernels (`idsva_so_world_frame` twin was **28x** raw, cut to 5.5x by rolling, then to **2.42x** by
  block-parallelizing the epilogue; `fdsva_so` **1.41x**; first-order twins ~1.0x). That — not
  second-order kernel size — is why humanoid builds OOM. To shrink the mjx epilogue: it assembled each
  output slab in PER-THREAD register arrays (nv²-sized) that spilled to local memory; block-sharing
  them + spreading each op across the block removed the spill (both smaller AND faster) — see
  `docs/agent_debugging_guide.md` §1u. Build pin-only with `enable_mujoco_kernels=False`
  (`register_robot` / `gen_all_code`), or `GRID_ENABLE_MUJOCO_KERNELS=0` for a whole codegen
  session; an explicit argument always beats the env var. The CUDA equivalence suite defaults to
  pin-only (`test/cuda_equivalents/conftest.py`) — it exercises no mjx path, and that alone took a
  go2-floating second-order cell from 175 s to 20 s.
- **Verify yourself** before committing — re-run the sanitizers / equivalence / poison harness; don't
  trust a subagent's "done" (subagents can lose Bash mid-run).
- **Git**: short single-line commit messages, no Co-Authored-By footer; path-scoped `git add` (never
  `-A`/`.`); submodules committed/pushed before the parent pointer bump; push only when asked.

## Deeper reading

`docs/source/user_guide/concepts/{design_principles,codegen_architecture,resource_tier_system}.rst`
for the codegen architecture (the `*_inner`/`*_device`/`*_kernel`/host layer stack, the shared-memory
tier/spill system) and `docs/agent_debugging_guide.md` for the accumulated bug classes.
