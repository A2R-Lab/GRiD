# GRiD Benchmarks

Performance comparison of GRiD vs. Pinocchio (and MJX in PR 2) across all 12 algorithms,
for `iiwa14`, `go2`, and `g1` robots in fixed and floating-base configurations.

---

## Quick Start

```bash
# Full suite (GRiD + Pinocchio, all robots, fixed + floating):
python test/benchmarks/run_benchmarks.py

# Just GRiD, one robot:
python test/benchmarks/baselines/grid/run.py --robot iiwa14 --base fixed

# Just Pinocchio, one robot:
python test/benchmarks/baselines/pinocchio/run.py --robot iiwa14 --base fixed

# Re-use cached binaries (skip recompile):
python test/benchmarks/run_benchmarks.py --no-recompile
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

### Pinocchio (CPU)

Pinocchio with CppADCodeGen is installed as part of the dev dependencies:

```bash
pip install -e ".[dev]"   # from repo root
```

Verify:

```bash
python -c "import pinocchio; print(pinocchio.__version__)"
```

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

### ABA vs. FD (GRiD forward dynamics)

GRiD has two forward dynamics implementations:
- **FD**: Minv + RNEA composition (`forward_dynamics`)
- **ABA**: Articulated Body Algorithm (`aba`) — independent implementation

Both are benchmarked and shown separately.

---

## g1 EE Rows

The G1 humanoid has two distinct EE use cases:

| Label | Frame | Use case |
|-------|-------|----------|
| `g1` (arm) | `right_rubber_hand` | Manipulation |
| `g1-foot` | `right_ankle_roll_link` | Locomotion |

Both appear as separate rows in EE kinematics sections of `benchmark.md`.

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
    └── util/
        ├── experiment_helpers.h
        └── getters/
            ├── GetResRNEA.hpp
            └── GettersDerivatives.hpp
```
