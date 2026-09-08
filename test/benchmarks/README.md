# GRiD Benchmarks

Performance comparison of GRiD vs. Pinocchio vs. MJX across 14 core algorithms,
for `iiwa14`, `baxter`, `go2`, `g1`, and `h2_plus` robots in fixed and
floating-base configurations.

The 14 rows are the first-order set (`inverse_dynamics`, `minv`, `forward_dynamics`,
`aba`, `crba`, `inverse_dynamics_gradient`, `forward_dynamics_gradient`,
`end_effector_pose`, `end_effector_pose_gradient`) plus the second-order set (`idsva_so`
— the dispatched winner, `idsva_so_body_frame`, `idsva_so_world_frame`,
`fdsva_so`). The two IDSVA-SO variants are mathematically equivalent and ship
side-by-side so the table shows the body-vs-world crossover; `idsva_so` itself
is the codegen-time dispatcher (body-frame for fixed-base, world-frame for
floating-base — see [docs/sweep-on-5090.md](../../docs/sweep-on-5090.md) for
the crossover numbers).

---

## Quick Start

All commands use the project virtualenv.  Create it once if it doesn't exist:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# For an apples-to-apples Pinocchio comparison, also install CppADCodeGen
# (enables the codegen-accelerated inverse_dynamics/minv/aba/forward_dynamics/crba/inverse_dynamics_gradient/forward_dynamics_gradient paths;
# without it those algorithms appear as null in the Pinocchio results):
.venv/bin/pip install cmeel-cppadcodegen
```

Then run benchmarks:

```bash
# Full suite (GRiD + Pinocchio, all robots, fixed + floating):
.venv/bin/python test/benchmarks/run_benchmarks.py

# Include MJX:
.venv/bin/python test/benchmarks/run_benchmarks.py --baselines grid pinocchio mjx

# Just GRiD, one robot (per-exe path: one TU/exe/process per algo, RAM-safe + crash-isolated):
.venv/bin/python test/benchmarks/per_algo_bench.py --robot iiwa14 --base fixed

# Just Pinocchio, one robot:
.venv/bin/python test/benchmarks/baselines/pinocchio/run.py --robot iiwa14 --base fixed

# Just MJX, one robot:
.venv/bin/python test/benchmarks/baselines/mjx/run.py --robot iiwa14 --base fixed

# Re-use cached binaries (skip recompile):
.venv/bin/python test/benchmarks/run_benchmarks.py --no-recompile
```

Results are saved to `test/benchmarks/results/` (gitignored) and
`test/benchmarks/benchmark.md` (committed snapshot).

---

## Prerequisites

### GRiD (CUDA)

CUDA Toolkit and `nvcc` must be on `PATH`.  These are already required to use GRiD itself.

```bash
nvcc --version   # should print CUDA release info
```

GRiD uses a vendored `glass` SIMT linalg helper subset that needs no extra
SDK. The cuBLASDx-backed `glass-nvidia` path was removed in v2.0; see
`docs/source/user_guide/concepts/cublasdx_removal_design.rst` for the
rationale and the `archive/last-cublasdx` git tag for the historical code
path.

### Pinocchio (CPU)

Pinocchio is installed as part of the dev dependencies (see Quick Start above).

Verify:

```bash
.venv/bin/python -c "import pinocchio; print(pinocchio.__version__)"
```

**CppADCodeGen (recommended for an apples-to-apples comparison):** The codegen-
accelerated algorithms (inverse_dynamics, minv, aba, forward_dynamics, crba,
inverse_dynamics_gradient, forward_dynamics_gradient) require CppADCodeGen
headers.  The benchmark runner detects availability automatically — if not found,
those algorithms are silently reported as null and the direct-API algorithms
(end_effector_pose, end_effector_pose_gradient, idsva_so_body_frame, idsva_so_world_frame) still run.
The dispatched `IDSVA_SO` row mirrors whichever variant the codegen picked.
FDSVA_SO has no Pinocchio equivalent and is GRiD-only.

Install the cmeel-packaged version into the same venv as Pinocchio:

```bash
.venv/bin/pip install cmeel-cppadcodegen
```

This drops `cppad/cg.hpp` into `.venv/lib/pythonX.Y/site-packages/cmeel.prefix/`
where the runner looks for it; no other configuration needed.  Bust the pinocchio
binary cache after installing so the rebuild picks up `-DHAVE_CPPADCG`:

```bash
rm -rf .pytest_cache/grid_cuda/pinocchio_benchmarks
```

If you'd rather use the system package or build from source, those still work too:

```bash
# Ubuntu/Debian (if available):
sudo apt-get install libcppadcg-dev
# Or build from source: https://github.com/joaoleal/CppADCodeGen
```

#### Pinocchio C++ binding for the equivalence suite (`pin_so_ext`)

The `RBDReference/` suite uses Pinocchio's C++ implementation
as the golden oracle for RBDReference + CUDA equivalence tests
(particularly for second-order derivatives, where Python Pinocchio's
SO API isn't directly comparable). This binding is built automatically
by `install/developer_install.sh` and on first import of the suite via
`setuptools`.

Requirements (on top of the standard dev install):

```bash
# Ubuntu/Debian — pkg-config-discoverable Pinocchio + a modern compiler:
sudo apt-get install pkg-config libpinocchio-dev g++

# Verify pinocchio's pkg-config is on the path:
pkg-config --modversion pinocchio

# pybind11 is in the dev dependencies but confirm:
.venv/bin/python -c "import pybind11; print(pybind11.__version__)"
```

If `pkg-config pinocchio` doesn't resolve (common when Pinocchio came
from cmeel rather than apt), the binding's `setup.py` falls back to
the cmeel prefix in the same venv.

**Verify the binding builds + runs:**

```bash
# Builds pin_so_ext on first run (~30s); subsequent runs are cache hits.
.venv/bin/pytest RBDReference/tests/test_second_order_pinocchio_equivalence.py -k iiwa14 -x
```

If you hit build errors, the most common fixes are:
- `pkg-config: command not found` → install pkg-config
- `Pinocchio.hpp: No such file` → `pkg-config --cflags pinocchio` is empty
  → install `libpinocchio-dev` or activate the cmeel venv
- `python.h: No such file` → install `python3-dev` (matching the venv's
  Python version)

### MJX (GPU via JAX)

MJX requires JAX with GPU support and MuJoCo:

```bash
.venv/bin/pip install mujoco mujoco-mjx
.venv/bin/pip install --upgrade "jax[cuda12]"   # adjust for your CUDA version
```

Verify:

```bash
.venv/bin/python -c "import mujoco.mjx; import jax; print(jax.devices())"
```

MJX is **not** run by default — pass `--baselines mjx` explicitly.  MJX is also skipped on
Jetson/unified-memory platforms since JAX/XLA is not optimized for that architecture.

### CPU Frequency Locking (Linux — optional but recommended for Pinocchio)

Locking the CPU to the performance governor removes frequency-scaling noise from
Pinocchio timings.  The benchmark runner attempts this automatically via `sudo`:

```bash
# One-time setup (passwordless sudo for cpupower):
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/cpupower" | sudo tee /etc/sudoers.d/cpupower

# Alternatively, use the repo-bundled script:
sudo bash test/benchmarks/setCPU.sh
```

If unavailable, the benchmark still runs — timing may be noisier.
On macOS / Windows, CPU freq locking is not supported and a warning is printed.

---

## Understanding the Results

### With-Memory vs. Compute-Only (GRiD)

| Label | Measures |
|-------|----------|
| **with-memory** | Full round-trip: `cudaMemcpy` host→device + kernel + `cudaMemcpy` device→host |
| **compute-only** | Kernel only (data already on GPU) |

On **Jetson** (unified memory), `cudaMemcpy` is a no-op: with-memory ≈ compute-only.
Compare on compute-only numbers in the Jetson appendices.

### Single vs. Batch

| Label | Measures |
|-------|----------|
| **single (1)** | One state, kernel loops internally for `TEST_ITERS` reps — minimizes launch overhead |
| **batch (N)** | Host loop launches N parallel states on the GPU |

### codegen vs. direct (Pinocchio)

| Label | Measures |
|-------|----------|
| **codegen** | CppAD-generated C code compiled to a shared library — fastest Pinocchio path |
| **direct** | Direct Pinocchio C++ API — used for algorithms without codegen support |

### MJX Algorithm Coverage

MJX exposes a subset of algorithms via `mujoco.mjx`:

| Algorithm | MJX Function | Notes |
|-----------|-------------|-------|
| **inverse_dynamics** | `mjx.inverse()` | RNEA |
| **forward_dynamics** | `mjx.forward()` | Full forward dynamics |
| **end_effector_pose** | `mjx.kinematics()` | Forward kinematics |
| **inverse_dynamics_gradient** | `jax.jacobian(mjx.inverse)` | AD through RNEA |
| minv, crba, aba, forward_dynamics_gradient, idsva_so_body_frame, idsva_so_world_frame, fdsva_so | — | Not available in MJX |

MJX uses `jax.vmap` for batching and `jax.block_until_ready()` to ensure GPU completion
before stopping the timer. The first two calls (JIT compilation + GPU warm-up) are discarded.

### ABA vs. FD (GRiD forward dynamics)

GRiD has two forward dynamics implementations:
- **forward_dynamics**: Minv + RNEA composition (`forward_dynamics`)
- **aba**: Articulated Body Algorithm (`aba`) — independent implementation

Both are benchmarked and shown separately.

---

## g1 EE Rows

The G1 humanoid has two distinct EE use cases:

| Label | Frame (GRiD) | Frame (Pinocchio/MJX) | Use case |
|-------|-------------|----------------------|----------|
| `g1` (arm) | `right_hand_palm_joint` | `right_rubber_hand` | Manipulation |
| `g1-foot` | — (no fixed ankle joint) | `right_ankle_roll_link` | Locomotion |

Both appear as separate rows in EE kinematics sections of `benchmark.md`.

---

## Reproducing the Multi-Version Comparison

Side-by-side benchmark of two GRiD versions vs three external GPU/CPU
references: **pre-GLASS** (git ref `d2c0d18`, the last commit before the GLASS
work), **glass** (HEAD with pure-SIMT GLASS), **pinocchio** (CPU codegen),
**mjx** (MuJoCo MJX on JAX-GPU), and **frax** (Frax on JAX-GPU,
https://github.com/danielpmorton/frax). The orchestrator manages a separate
git worktree for the pre-GLASS column. MJX exposes
inverse_dynamics/forward_dynamics/end_effector_pose/inverse_dynamics_gradient;
Frax exposes inverse_dynamics/forward_dynamics/crba/minv; the others render `—`.

**Prereqs on a fresh machine:**

```bash
# 1. Clone + check out the working branch + init submodules.
git clone <repo-url> GRiD-A2R
cd GRiD-A2R
git checkout <branch>
git submodule update --init --recursive

# 2. Python venv + dependencies (same as Quick Start above).
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install cmeel-cppadcodegen     # Pinocchio codegen-accelerated algos

# 3. CUDA toolkit + nvcc on PATH (required for all GRiD columns).
nvcc --version

# 4. MuJoCo MJX for the mjx column (skip with --columns if not wanted).
.venv/bin/pip install mujoco mujoco-mjx
.venv/bin/pip install --upgrade "jax[cuda12]"
.venv/bin/python -c "import mujoco.mjx; import jax; print(jax.devices())"

# 5. Frax for the frax column (skip with --columns if not wanted).
.venv/bin/pip install frax
.venv/bin/python -c "import frax; print('frax OK')"
```

**Run the sweep:**

```bash
# Where the d2c0d18 worktree gets created (defaults to ../GRiD-A2R-pre-glass).
export GRID_PRE_GLASS_WORKTREE=../GRiD-A2R-pre-glass

# Single robot/base (fastest, ~5 min on iiwa14_fixed):
.venv/bin/python test/benchmarks/run_multi_version.py \
    --robots iiwa14 --bases fixed

# Full sweep (~30+ min, dominated by g1 pinocchio cppadcg compile):
.venv/bin/python test/benchmarks/run_multi_version.py

# Skip a specific (robot, base) combo (e.g. if it hangs the compiler):
.venv/bin/python test/benchmarks/run_multi_version.py --skip iiwa14_floating

# Bump iter counts for more stable medians on fast/noisy hardware
# (default: 10000 inner reps for single-call, 100 outer reps for batch on GRiD/Pin,
#  500 reps on MJX/Frax). 5x bumps roughly 5x the run time:
.venv/bin/python test/benchmarks/run_multi_version.py \
    --single-call-iters 50000 --batch-iters 500
```

**Stability flags reference:**

| Flag | Default | When to bump |
|---|---|---|
| `--single-call-iters N` | 10000 (GRiD/Pin) | Single-call timings show high variance — bump to 50k+ for sub-µs algos |
| `--batch-iters N` | 100 (GRiD/Pin), 500 (MJX/Frax) | Batch medians noisy — bump 5–10× |
| `--pin-num-threads N` | auto (physical cores) | Sets Pinocchio's internal CPU_THREADS_GLOBAL. Auto-detect picks physical cores (NOT logical/SMT siblings — they hurt for batched-same-function workloads). Override if auto-detection is wrong (`PIN_PHYSICAL_CORES` env var also works). |
| `--no-rdc` | off | ptxas hangs on floating-base; first thing to try |
| `--no-licm-barrier` | off | ptxas still hangs after `--no-rdc`; strongest hammer |
| `--fixed-only` | off | Skip every floating-base combo (shortcut for `--bases fixed`). Run fixed first to get clean data, then revisit floating with the slow compile. |
| `--skip iiwa14_floating` | none | Exclude specific robot/base combos that are broken on your machine |

**Pinocchio parallelism details:** the Pinocchio runner uses two layers of parallelism — outer subprocess fan-out (one per algo, for parallel cppadcg JIT compile) and inner thread pool (`CPU_THREADS_GLOBAL` worker threads splitting the batch loop across timesteps). To avoid CPU oversubscription, the outer fan-out is `max(1, physical_cores / internal_threads)`. With the default `internal_threads = physical_cores`, that's 1 subprocess at a time (each gets all cores). Override via `PIN_MAX_WORKERS` env var if you want more parallelism (e.g. when JIT compile dominates and batch run is fast).

The orchestrator:
1. Creates a worktree at `$GRID_PRE_GLASS_WORKTREE` checked out to `d2c0d18`
   with the pinned submodule SHAs (idempotent — reuses if it already exists).
2. Runs each requested column for each robot/base (pre-glass is fixed-base only).
3. Writes per-column JSONs into `test/benchmarks/results/comparison/` with names
   like `iiwa14_fixed_grid_pre_glass.json`, `iiwa14_fixed_grid_glass.json`, etc.
4. Merges them into `benchmark_multi_version_<host>.json` and renders
   `test/benchmarks/benchmark_multi_version.md` with four columns + speedup
   ratios.

**Reading the report:**

- **glass/pre** column: N=256 compute-only ratio. `> 1.00×` = HEAD is faster
  than the pre-GLASS baseline; `< 1.00×` = HEAD regressed on that algo.
- Floating-base rows show `—` in the pre_glass column (harness doesn't support
  it at `d2c0d18`).

**Caches:** each version uses its own `.pytest_cache/grid_cuda/` directory
under its respective worktree, so codegen + binary caches don't collide.
Pass `--no-recompile` to reuse cached binaries on rerun.

**ccache (strongly recommended — large iteration speedup):** both the GRiD
nvcc compile and the Pinocchio g++ compile transparently route through
`ccache` if it's on `$PATH`. Install with `sudo apt install ccache` (or
`brew install ccache`).

The first compile populates the cache; subsequent compiles with the same
preprocessed source + flags are cache hits and skip the heavy ptxas /
Eigen-template work entirely — typically going from tens of seconds to
under a second per compile. Most useful when:

- Clearing `.pytest_cache/grid_benchmarks/` or `pinocchio_benchmarks/` but
  the underlying `.cu` / `.cpp` source hasn't changed.
- Iterating on the harness Python code without touching codegen output.

Disable per-binary with `GRID_NO_CCACHE=1` (GRiD) or `PIN_NO_CCACHE=1`
(Pinocchio). Inspect cache stats with `ccache -s`; clear with `ccache -C`.
Default cache size is 5 GB — bump if you're caching many builds:
`ccache -M 20G`.

---

## Adding Results from a New Machine

1. Run the full suite on your machine:
   ```bash
   python test/benchmarks/run_benchmarks.py
   ```
2. Review `test/benchmarks/benchmark.md` (auto-updated by the coordinator).
3. Commit the updated `benchmark.md`.
4. Optionally commit your raw JSON from `results/` as a named snapshot.

---

## File Structure

(Refreshed 2026-09-08 — tools added since the last sweep:)

- `autotune_tier_matrix.py` — per-tier forced-probe matrix sweep; writes the `matrix` block (E5)
- `bake_by_n_bucket.py` — E6 batch-switch bake (`ffi_bases_by_n`)
- `collect_kernel_limits.py` — kernel limit collection (min_smem join for the matrix tool)
- `gpu_resident_timing.py` — GPU-resident/no-transfer timing
- `analyze_competitive.py` / `plot_benchmarks.py` / `run_competitive_gpu_baselines.sh` — competitive pipeline
- `run_tier_sweep_phased.sh` — phased tier sweep driver
- `test_autotune_picker.py` / `test_per_algo_specs_bijection.py` — CPU gates over the pickers/specs
- `archive/` — superseded scripts + historical report snapshots


```
test/benchmarks/
├── README.md                    ← this file
├── run_benchmarks.py            ← main coordinator
├── per_algo_bench.py            ← per-exe GRiD orchestrator (one TU / exe / process per algo)
├── run_multi_version.py         ← multi-version comparison sweep (columns + baselines)
├── autotune_ffi.py              ← FFI-lane (bindings) launch-config autotuner
├── generate_report.py           ← JSON → benchmark.md
├── timing_parser.py             ← shared output parser
├── benchmark.md                 ← committed curated snapshot
├── baselines.json               ← committed curated baselines
├── setCPU.sh                    ← CPU frequency locking helper (Linux)
├── perf_regression_report.py    ← CI regression tool
├── results/                     ← GITIGNORED per-run JSON
├── .gitignore
└── baselines/
    ├── grid/
    │   ├── run.py               ← GRiD runner library (PER_ALGO_SPECS)
    │   ├── timeGRiD_bindings.py ← bindings-lane timing script
    │   └── timeGRiD_common.h    ← shared timing header for the per-algo TUs
    ├── pinocchio/
    │   ├── run.py               ← Pinocchio runner
    │   ├── timePinocchio.cpp    ← timing program
    │   └── ReusableThreads/     ← submodule (plancherb1/ReusableThreads)
    ├── mjx/
    │   ├── run.py               ← MJX runner
    │   └── timeMJX.py           ← JAX/MJX timing script
    ├── frax/
    │   ├── run.py               ← Frax runner
    │   └── timeFrax.py
    ├── bard/
    │   ├── run.py               ← BARD (PyTorch CPU+GPU) runner
    │   └── timeBARD.py
    ├── curobo/
    │   ├── run.py               ← cuRobo runner
    │   └── timeCurobo.py
    ├── mujoco_warp/
    │   ├── run.py               ← MuJoCo Warp runner
    │   └── timeMujocoWarp.py
    └── util/
        ├── experiment_helpers.h
        └── getters/
            ├── GetResRNEA.hpp
            └── GettersDerivatives.hpp
```
