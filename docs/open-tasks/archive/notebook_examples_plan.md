# Notebook examples plan — docs that double as smoke tests

**Status:** planning (read-only exploration done 2026-05-30).

**Goal.** A small, curated set of Jupyter notebooks that (a) serve as the
"getting started" + cookbook docs for the `grid_rbd` register-then-run UX, and
(b) are **executed in CI as smoke tests**, so the docs can never silently rot.
Each notebook is short, runs end-to-end on a single small robot where possible,
and demonstrates one cohesive slice of the API.

This complements — and in two places depends on — the D.3 torch plan
(`d3_pytorch_cudagraphs_plan.md`); cross-references below are to its sections.

---

## 0. What exists today (and the gaps)

- **No notebooks.** There is no `notebooks/` dir and no `*.ipynb` anywhere in the
  repo. `examples/` holds **four `.py` scripts** that are codegen-only (they call
  `URDFParser` + `GRiDCodeGenerator.gen_all_code()` to emit a `grid.cuh`, e.g.
  `examples/quickstart_iiwa14.py:38-43`) — they do **not** exercise the
  `grid_rbd` register-then-run package or run any kernel. So the notebooks are
  net-new and are the first artifacts to show the actual runtime API.
- **The package surfaces to demo already exist:** numpy `RobotHandle`
  (`python/grid_rbd/_handle.py`, 14 methods) and `JaxRobotHandle`
  (`python/grid_rbd/jax/__init__.py`, jittable). The torch handle + autograd +
  CUDA-Graphs surface is **D.3 (planned, not yet built)** — notebooks (b) and (g)
  below are gated on D.3 landing.
- **URDF assets are vendored:** `robot_assets/{iiwa14,go2,g1,h1_2,...}.urdf`
  (`robot_assets/`), so notebooks can register from a repo-local path without a
  network fetch. The existing `.py` examples instead pull from
  `robot_descriptions` (`examples/quickstart_iiwa14.py:16`); notebooks should
  prefer the **repo-local `robot_assets/` path** for hermetic CI.
- **Inline-URDF gap:** "define the URDF in a cell" requires `urdf_string=`, which
  is a **D.3 prerequisite** (see `d3_pytorch_cudagraphs_plan.md` §2b). Until it
  lands, the inline-URDF demo must instead point at a `robot_assets/*.urdf` path.
- **`grid_plant` has no Python surface yet.** The plant/cost/barrier layer is a
  CUDA-side `grid_plant::` namespace (`GRiDCodeGenerator/algorithms/_plant.py`)
  validated only by a CUDA executable test
  (`test/cuda_equivalents/cuda_plant_smoke_runner.cu`,
  `test/cuda_equivalents/test_cuda_plant_equivalence.py`). There is **no**
  `grid_rbd` handle method that calls `grid_plant::quadratic_state_cost` etc. —
  see notebook (e)'s prerequisite note.

---

## 1. Where they live & how they're structured

- **Directory:** new top-level `notebooks/` (sibling of `examples/`). Keep the
  codegen-only `.py` scripts in `examples/`; `notebooks/` is exclusively the
  runtime register-then-run cookbook. (Alternative considered: `examples/*.ipynb`
  alongside the scripts — rejected to keep "codegen scripts" and "runtime API
  notebooks" visually separate and to give CI a single glob to execute.)
- **Naming:** numeric prefix for reading order, e.g.
  `notebooks/01_quickstart_iiwa14.ipynb`.
- **Each notebook's first cell** is a standard preamble:
  - skip/guard banner: requires CUDA GPU + `nvcc` on PATH + `grid_rbd` installed
    (mirror the smoke-test skip conditions,
    `test/python_wrappers/test_iiwa14_smoke.py:33-45`);
  - a **compile-time expectations** markdown banner (iiwa14 = seconds; humanoids =
    minutes; recommend a pre-warmed cache — see `d3_pytorch_cudagraphs_plan.md`
    §2b "First-compile latency guidance");
  - registers its robot from a `robot_assets/*.urdf` path.
- **Determinism for CI:** seed numpy/torch RNG in-cell; assert against
  `RBDReference` or a closed-form check rather than printing free-form output, so
  `nbval` (or an executed-and-asserted run) actually validates numbers, not just
  "no exception."
- **Each notebook is a smoke test:** every notebook ends with one or more
  `assert` cells (e.g. `np.allclose(grid_result, reference, atol=5e-3)` using the
  same `_TOL=5e-3` the wrapper smoke test uses,
  `test/python_wrappers/test_iiwa14_smoke.py:51`). A green run == passing docs.

---

## 2. Proposed notebook set

Robots chosen for **fast compile** unless the notebook's point *is* the big
robot. iiwa14 = 7-DOF fixed-base (cheapest); go2 = quadruped floating-base
(moderate); g1 = humanoid floating-base (slow compile — gated).

### (a) `01_quickstart_iiwa14.ipynb` — the register-then-run loop
- **Demonstrates:** the core two-tier UX. Register iiwa14 in one cell; several
  cells later call `rnea` / `forward_dynamics` / `crba`; show the batch dimension
  (run `B=1` and `B=64` on the same handle); show the cache hit on a second
  `register_robot` call (second call returns instantly).
- **APIs:** `grid_rbd.register_robot(..., urdf_path=robot_assets/iiwa14.urdf)`
  (`__init__.py:54`), `handle.rnea` / `forward_dynamics` / `crba` / `minv` /
  `end_effector_pose` (`_handle.py:139, 169, 185, 153, 192`),
  `handle.num_joints`/`max_batch` metadata (`_handle.py:85, 101`).
- **Prereqs:** CUDA + nvcc + numpy. No torch/jax.
- **Robot:** iiwa14 (fixed-base). **Compile: seconds.**
- **Assert cell:** cross-check `rnea`/`crba` vs `RBDReference`
  (`test/python_wrappers/test_iiwa14_smoke.py` pattern), `atol=5e-3`.

### (b) `02_autograd_torch.ipynb` — gradients & gradient-based optimization
- **Demonstrates:** the torch backend's autograd-awareness (D.3 §2). Build a
  handle with `backend="torch"`; show `qdd = h.forward_dynamics(q,qd,u)` then
  `qdd.sum().backward()` populates `q.grad`. Then a **tiny IK / trajectory
  optimization**: minimize `‖end_effector_pose(q) − target‖²` over `q` with a few
  steps of `torch.optim.Adam`, plotting the loss curve. Optionally a
  finite-difference-vs-analytic gradient check (the float32 FD check from D.3 §5,
  not `torch.autograd.gradcheck`, since kernels are float32-only).
- **APIs:** `grid_rbd.register_robot(..., backend="torch")` (D.3 §2b), the
  autograd-aware `rnea`/`forward_dynamics`/`end_effector_pose[_gradient]` torch
  ops (D.3 §2 table), `torch.autograd`, `torch.optim`.
- **Prereqs:** **D.3 torch backend must be built.** CUDA + nvcc + torch.
- **Robot:** iiwa14 (fixed-base; IK target is its EE). **Compile: seconds.**
- **Assert cell:** analytic-vs-FD VJP within float32 tol (~1e-2 rel, per D.3 §5);
  IK loss decreases monotonically below a threshold.

### (c) `03_jax_vs_torch_parity.ipynb` — same kernels, two front-ends
- **Demonstrates:** that `grid_rbd.jax` and `grid_rbd.torch` are the **same
  compiled `.so`** (shared cache — `jax/__init__.py:18-21`) with two ABIs;
  run an algorithm on both, assert bit-for-similar agreement vs each other and vs
  numpy; show `jax.jit(handle.rnea)` and the torch eager call side by side; a
  short "when to pick which" markdown (JAX: `jax.jit`/XLA graph capture for free,
  functional; torch: explicit autograd `Function`s + `handle.capture` CUDA-Graphs,
  ecosystem). Note JAX surface is forward-only today; torch adds VJP (D.3 §4).
- **APIs:** `grid_rbd.jax.register_robot` (`jax/__init__.py:380`),
  `grid_rbd.torch.register_robot`, `jax.jit`, matching `rnea`/`forward_dynamics`.
- **Prereqs:** **D.3 torch backend built** + jax installed. CUDA + nvcc.
- **Robot:** iiwa14. **Compile: seconds** (one `.so`, two registrations = one
  compile + cache hit — good demo of the shared cache).
- **Assert cell:** `np.allclose(jax_out, torch_out.cpu(), atol=5e-3)` and both vs
  numpy `RobotHandle`.

### (d) `04_batched_mpc.ipynb` — batched / MPC-style rollout
- **Demonstrates:** the batch dimension as the workhorse. Roll out a batch of
  short trajectories with `integrator` (B parallel initial conditions), then a
  toy single-shooting MPC: per step, batch-evaluate `forward_dynamics` +
  `integrator` over candidate controls and pick the best. Show throughput scaling
  with `B` (time `B=1` vs `B=256`). Uses the device-resident path (JAX or torch).
- **APIs:** `handle.integrator(q,qd,u,dt,integrator_type=...)` (`_handle.py:282`,
  `_INTEGRATOR_CODES` at `_handle.py:28`), `forward_dynamics`, batch reshapes;
  `handle.max_batch` validation (`_handle.py:101`).
- **Prereqs:** CUDA + nvcc; jax **or** (post-D.3) torch. numpy works too but the
  point is on-device batching.
- **Robot:** iiwa14 or go2. **Compile: seconds–moderate.**
- **Assert cell:** integrator step vs `RBDReference` integrator; MPC cost
  non-increasing.

### (e) `05_plant_cost_control.ipynb` — plant + cost + barrier (T6 `grid_plant`)
- **Demonstrates:** the control-oriented `grid_plant` surface: `plant_step` /
  `plant_step_gradient` (the `[A|B]` linearization, a pass-through of
  `grid::integrator_gradient`), quadratic state/input cost value+grad+GN-Hessian,
  ee-position cost (`J_pᵀ W J_p`), and log-barriers for joint/vel/torque limits —
  i.e. everything a DDP/iLQR/SQP step needs. Walk one Gauss-Newton step on a toy
  trajopt problem.
- **APIs:** the `grid_plant::` primitives — `plant_step`/`plant_step_gradient`
  (`_plant.py:42, 69`), `quadratic_state_cost`/`quadratic_input_cost`
  (`_plant.py:247, 251`), `ee_pos_cost` (`_plant.py:259`), `plant_barriers`
  (`_plant.py:493`).
- **PREREQ / GAP:** `grid_plant` currently has **no `grid_rbd` Python handle
  surface** — it is only reachable from CUDA and validated by a CUDA executable
  test (`cuda_plant_smoke_runner.cu`,
  `test/cuda_equivalents/test_cuda_plant_equivalence.py:1-20`). **A prerequisite
  for this notebook is exposing `grid_plant` through the wrapper/handle** (new
  C-ABI + FFI/torch entry points in `wrapper_template.cu` + `_handle.py`, analogous
  to the dynamics methods). File this as a dependency task; the notebook is
  blocked until it exists. (Interim: the notebook could shell out to the CUDA smoke
  runner to *illustrate* the math, but that's not the package UX — prefer to wait
  for the handle surface.)
- **Robot:** iiwa14 (fixed-base; the CUDA plant test defaults to iiwa14-fixed,
  `test_cuda_plant_equivalence.py:`"Default robot is iiwa14-fixed"). **Compile:
  seconds.**
- **Assert cell:** mirror the CUDA test's NumPy recomputes (GN-diag Hessian vs
  `diag(Q)`; ee cost vs `J_pᵀ W`; barrier vs log-barrier; unbounded DOF ⇒ exactly
  zero) — these oracles already exist in `test_cuda_plant_equivalence.py`.

### (f) `06_floating_base_humanoid_g1.ipynb` — floating-base, with a compile caveat
- **Demonstrates:** floating-base registration and the floating-base-specific
  outputs: spatial EE Jacobian base block (`end_effector_pose_gradient`,
  `_handle.py:198-213`), world-frame `idsva_so` dispatch (`_handle.py:251`), the
  `NUM_POS != NUM_VEL` quaternion-vs-tangent distinction. Show
  `floating_base=True` registration and that `num_vel` ≠ `num_joints`.
- **APIs:** `register_robot(..., floating_base=True)` (`__init__.py:58`),
  `end_effector_pose_gradient`, `idsva_so`/`fdsva_so` (`_handle.py:251, 270`),
  `handle.floating_base`/`num_vel` (`_handle.py:97, 89`).
- **Prereqs:** CUDA + nvcc. **LOUD caveat banner:** g1/h1_2 nvcc compile is
  **minutes** (SO kernels; `_compile.py:118` always enables them) — recommend a
  pre-warmed cache (`d3_pytorch_cudagraphs_plan.md` §2b). For a faster live run,
  fall back to **go2** (smaller floating-base quadruped, `robot_assets/go2.urdf`).
- **Robot:** g1 (humanoid) — or go2 for a quick CI variant. **Compile: minutes
  (g1) / moderate (go2).**
- **Assert cell:** floating-base `rnea`/`crba` vs `RBDReference` floating-base
  reference; EE-Jacobian shape `(B, 6, NV)`.

### (g) `07_cudagraphs_speedup.ipynb` — CUDA-Graphs micro-demo
- **Demonstrates:** the D.3 `handle.capture(...)` CUDA-Graphs callable (D.3 §3).
  Capture `forward_dynamics` at a fixed batch, then time an MPC-style replay loop
  (`g.static_in[i].copy_(...)`; `g.replay()`) vs the eager per-call path; show the
  launch-overhead win. Note warmup is load-bearing (smem opt-in + init must run
  off-graph, D.3 §3) and that changing batch / threads-per-block requires
  recapture (D.3 §3 "Fixed-batch capture & recapture").
- **APIs:** `handle.capture("forward_dynamics", q0, qd0, u0)` → `GraphCallable`
  with `.static_in`/`.static_out`/`.replay()` (D.3 §3).
- **Prereqs:** **D.3 torch backend + CUDA-Graphs callable built.** CUDA + nvcc +
  torch.
- **Robot:** iiwa14. **Compile: seconds.**
- **Assert cell:** captured replay output == eager output bit-for-bit (same
  kernel/config, D.3 §5 point 3); replayed loop wall-time < eager loop wall-time.

**Summary table**

| # | Notebook | Robot | Backend | Compile | D.3-gated? |
|---|---|---|---|---|---|
| a | quickstart | iiwa14 | numpy | sec | no |
| b | autograd + IK | iiwa14 | torch | sec | **yes** |
| c | jax↔torch parity | iiwa14 | jax+torch | sec | **yes (torch half)** |
| d | batched / MPC | iiwa14/go2 | jax or torch | sec–mod | partly (torch half) |
| e | plant+cost+barrier | iiwa14 | numpy/torch | sec | **needs grid_plant handle** |
| f | floating-base humanoid | g1 (/go2) | numpy | **min** (g1) | no |
| g | CUDA-Graphs speedup | iiwa14 | torch | sec | **yes** |

**Ship order suggestion:** (a) and (f-with-go2) ship **now** (numpy surface
exists). (d) ships now on the JAX half. (b), (c-torch-half), (g) ship with D.3.
(e) ships when `grid_plant` gets a handle surface.

---

## 3. CI testing strategy

**Principle:** the notebooks are tests, run under the existing pytest harness so
they share markers, skip logic, and the local-GPU assumption. The repo has **no
GPU CI runner today** (the only GitHub workflows are docs/gh-pages and
contributors — `.github/workflows/`), so notebook execution is a **local /
self-hosted-runner pytest target**, exactly like the existing
`python_wrappers` and `cuda_equivalence` suites
(`pyproject.toml:45-53` markers; both are GPU-only and run locally).

- **Mechanism: `nbval`.** Add `nbval` to `requirements-dev.txt` (currently
  `pytest`, `pytest-xdist`, `pin`, `robot_descriptions`, `xacrodoc`, `pybind11` —
  no notebook dep yet). Run with `pytest --nbval-lax notebooks/` so cells must
  execute without error; the in-notebook `assert` cells do the numerical
  validation (don't rely on `nbval`'s stored-output diffing, which is brittle for
  float GPU output — use `--nbval-lax` + explicit asserts).
- **New pytest marker:** add `notebooks: Executable example notebooks; requires
  CUDA + grid-rbd installed.` to `pyproject.toml:45-53`, alongside
  `python_wrappers`. Run via `pytest -m notebooks --nbval-lax notebooks/`.
- **Skip guards (module/collection level):** skip the whole notebook suite if no
  CUDA GPU, no `nvcc` on PATH, or `grid_rbd` not importable — same triad as
  `test/python_wrappers/test_iiwa14_smoke.py:33-45`. Torch-gated notebooks (b, c,
  g) additionally `importorskip("torch")`; jax-gated (c, d) `importorskip("jax")`;
  the plant notebook (e) skips until the `grid_plant` handle exists.
- **Keep CI cheap:** the executed-in-CI set is the **small-robot** notebooks
  (iiwa14, go2). The g1 humanoid notebook (f) is marked `slow`/`robot_nightly`
  (reuse the existing `robot_nightly` marker, `pyproject.toml:51`) and runs only
  in a nightly/manual job, OR runs against a **pre-warmed cache** so its
  registration cell is a guaranteed cache hit (`__init__.py:131`) and the notebook
  itself takes seconds. Provide a `make warm-cache` / fixture that calls
  `register_robot` for each notebook's robot before the suite.
- **Headless execution:** notebooks must run with no display — use a non-GUI
  matplotlib backend (`matplotlib.use("Agg")`) in the preamble so plot cells don't
  block under `nbval`.
- **Lint hook (optional):** strip notebook outputs on commit (`nbstripout`) to keep
  diffs reviewable, since outputs are regenerated/validated in CI anyway.

---

## 4. Prerequisite gaps (consolidated)

1. **`urdf_string=` (D.3 §2b)** — required for *fully inline* "define the URDF in
   a cell" notebooks. Until it lands, notebooks register from
   `robot_assets/*.urdf`. Independent of the torch op work; can land first.
2. **D.3 torch backend** — notebooks (b), (g), and the torch half of (c)/(d) are
   blocked on the torch handle + autograd `Function`s + `handle.capture`
   (`d3_pytorch_cudagraphs_plan.md` §2, §3).
3. **`grid_plant` Python/handle surface** — notebook (e) is blocked: `grid_plant::`
   is CUDA-only today (`_plant.py`; CUDA test only). Needs C-ABI + handle methods
   mirroring the dynamics path before it's demoable from a notebook.
4. **`nbval` dependency + `notebooks` marker + GPU CI target** — no notebook
   tooling or runner exists yet; add `nbval` to `requirements-dev.txt`, a marker
   to `pyproject.toml`, and a local/self-hosted `pytest -m notebooks` target.
5. **Cache pre-warming for the humanoid notebook** — to keep (f) CI-fast, a
   warm-cache fixture/step is needed so g1's minutes-long nvcc isn't on the
   notebook's critical path.
