# GRiD Benchmarks

Performance comparison of GRiD vs. Pinocchio vs. MJX across all 12 algorithms,
for `iiwa14`, `go2`, and `g1` robots in fixed and floating-base configurations.

---

## Quick Start

All commands use the project virtualenv.  Create it once if it doesn't exist:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# For an apples-to-apples Pinocchio comparison, also install CppADCodeGen
# (enables the codegen-accelerated id/minv/aba/fd/crba/id_du/fd_du paths;
# without it those algorithms appear as null in the Pinocchio results):
.venv/bin/pip install cmeel-cppadcodegen
```

Then run benchmarks:

```bash
# Full suite (GRiD + Pinocchio, all robots, fixed + floating):
.venv/bin/python test/benchmarks/run_benchmarks.py

# Include MJX:
.venv/bin/python test/benchmarks/run_benchmarks.py --baselines grid pinocchio mjx

# Just GRiD, one robot:
.venv/bin/python test/benchmarks/baselines/grid/run.py --robot iiwa14 --base fixed

# Just GRiD, explicitly force the default GLASS linear algebra backend:
.venv/bin/python test/benchmarks/baselines/grid/run.py \
  --robot iiwa14 --base fixed --linalg-backend glass

# Just GRiD, opt into experimental GLASS NVIDIA packed mode when MathDx is installed:
MATHDX_ROOT=/opt/nvidia/mathdx/25.12 \
.venv/bin/python test/benchmarks/baselines/grid/run.py \
  --robot g1 --base floating --linalg-backend glass-nvidia

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

cuBLASDx is optional. The generated GRiD headers default to a vendored `glass`
scalar/unrolled helper subset that needs no MathDx headers. The `glass-nvidia`
path is reserved for explicit packed-kernel experiments and requires C++17,
MathDx headers, and a `GRID_CUBLASDX_SM` target. The GRiD benchmark runner
handles those flags when requested:

```bash
MATHDX_ROOT=/opt/nvidia/mathdx/25.12 \
.venv/bin/python test/benchmarks/baselines/grid/run.py \
  --robot g1 --base floating --linalg-backend glass-nvidia
```

Use `--linalg-backend glass` for the default helper path.

### Pinocchio (CPU)

Pinocchio is installed as part of the dev dependencies (see Quick Start above).

Verify:

```bash
.venv/bin/python -c "import pinocchio; print(pinocchio.__version__)"
```

**CppADCodeGen (recommended for an apples-to-apples comparison):** The codegen-
accelerated algorithms (ID, Minv, ABA, FD, CRBA, ID_DU, FD_DU) require CppADCodeGen
headers.  The benchmark runner detects availability automatically — if not found,
those algorithms are silently reported as null and the direct-API algorithms
(EE_POSE, EE_POSE_GRADIENT, IDSVA_SO) still run.

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
| **ID** | `mjx.inverse()` | RNEA |
| **FD** | `mjx.forward()` | Full forward dynamics |
| **EE_POSE** | `mjx.kinematics()` | Forward kinematics |
| **ID_DU** | `jax.jacobian(mjx.inverse)` | AD through RNEA |
| Minv, CRBA, ABA, FD_DU, IDSVA_SO, FDSVA_SO | — | Not available in MJX |

MJX uses `jax.vmap` for batching and `jax.block_until_ready()` to ensure GPU completion
before stopping the timer. The first two calls (JIT compilation + GPU warm-up) are discarded.

### ABA vs. FD (GRiD forward dynamics)

GRiD has two forward dynamics implementations:
- **FD**: Minv + RNEA composition (`forward_dynamics`)
- **ABA**: Articulated Body Algorithm (`aba`) — independent implementation

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

Side-by-side benchmark of three GRiD versions vs two external GPU/CPU
references: **pre-GLASS** (git ref `d2c0d18`, the last commit before the GLASS
v2 integration), **glass** (HEAD with pure-SIMT GLASS v2), **glass-nvidia**
(HEAD with cuBLASDx), **pinocchio** (CPU codegen), and **mjx** (MuJoCo MJX on
JAX-GPU). The orchestrator manages a separate git worktree for the pre-GLASS
column. MJX exposes a subset of algorithms (id / fd / ee_pose / id_du);
others render `—`.

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

# 4. MathDx 25.12 for the glass-nvidia column (skip this column with --columns if
#    you don't have MathDx; the others still run).
ls /opt/nvidia/mathdx/25.12/include/cublasdx.hpp

# 5. MuJoCo MJX for the mjx column (skip with --columns if not wanted).
.venv/bin/pip install mujoco mujoco-mjx
.venv/bin/pip install --upgrade "jax[cuda12]"
.venv/bin/python -c "import mujoco.mjx; import jax; print(jax.devices())"
```

**Run the sweep:**

```bash
# Where the d2c0d18 worktree gets created (defaults to ../GRiD-A2R-pre-glass).
export GRID_PRE_GLASS_WORKTREE=../GRiD-A2R-pre-glass

# Single robot/base (fastest, ~5 min on iiwa14_fixed):
.venv/bin/python test/benchmarks/run_multi_version.py \
    --robots iiwa14 --bases fixed \
    --mathdx-root /opt/nvidia/mathdx/25.12

# Full sweep (~30+ min, dominated by g1 pinocchio cppadcg compile):
.venv/bin/python test/benchmarks/run_multi_version.py \
    --mathdx-root /opt/nvidia/mathdx/25.12

# Skip the glass-nvidia column if MathDx is not installed:
.venv/bin/python test/benchmarks/run_multi_version.py \
    --columns pre_glass glass pinocchio mjx

# Skip a specific (robot, base) combo (e.g. if it hangs the compiler):
.venv/bin/python test/benchmarks/run_multi_version.py --skip iiwa14_floating
```

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
- **glass_nv/glass** column: N=256 compute-only ratio. `> 1.00×` = cuBLASDx
  beats pure-SIMT for that shape; `< 1.00×` = SIMT wins (expected for small
  6×6×6 GEMMs on iiwa14; cuBLASDx is expected to win on larger shapes).
- Floating-base rows show `—` in the pre_glass column (harness doesn't support
  it at `d2c0d18`).

**Caches:** each version uses its own `.pytest_cache/grid_cuda/` directory
under its respective worktree, so codegen + binary caches don't collide.
Pass `--no-recompile` to reuse cached binaries on rerun.

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

```
test/benchmarks/
├── README.md                    ← this file
├── run_benchmarks.py            ← main coordinator
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
    │   ├── run.py               ← GRiD runner
    │   └── timeGRiD.cu          ← timing kernel (moved + extended)
    ├── pinocchio/
    │   ├── run.py               ← Pinocchio runner
    │   ├── timePinocchio.cpp    ← timing program (moved + extended)
    │   └── ReusableThreads/     ← submodule (plancherb1/ReusableThreads)
    ├── mjx/
    │   ├── run.py               ← MJX runner
    │   └── timeMJX.py           ← JAX/MJX timing script
    └── util/
        ├── experiment_helpers.h
        └── getters/
            ├── GetResRNEA.hpp
            └── GettersDerivatives.hpp
```
