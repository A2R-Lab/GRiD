# GRiD competitive benchmark — methodology (for credible, defensible numbers)

This documents how the GRiD-vs-competitors benchmark is run so the numbers survive scrutiny.
Established 2026-06-13 during the A1b competitive re-sweep.

## The three GRiD measurement layers (report all three; label each)
A single µs number hides where the cost goes. GRiD is reported at three layers:
1. **Raw / compute-only** — kernel only, inputs already GPU-resident. GRiD's design point
   (MPC rollouts / RL sampling). Source: the C++ harness `*_compute_only` path.
2. **C++ with-mem** — kernel + H2D/D2H transfer, NO python wrapper. Isolates transfer cost.
   Source: the C++ harness `with_mem` path (`batch_N_with_mem_us`).
3. **Through-bindings (wrapper-inclusive)** — the real `grid_rbd` python/jax/torch FFI
   end-to-end: pack + H2D + kernel + D2H + unpack + dispatch. The "what an adopter actually
   pays" number. Source: the grid_rbd binding.
   **SYMMETRIC JIT RULE — GRiD's jax/torch binding gets the SAME precompile-then-time treatment
   as the competitors.** The grid_rbd jax surface wraps the FFI call in `jax.jit`; timing it
   cold would inflate layer-3 with the first-call trace/compile (the exact mjx artifact). So
   warm the EXACT jitted FFI closure (`block_until_ready`, same shapes/dtypes, incl. the
   numpy→device path) in warmup, then time pure execution. Same for the torch surface (warm the
   graph). This is symmetric fairness — and it protects GRiD's own number from being unfairly worse.
Figures stack these (compute + transfer + wrapper), mirroring the classic compute + I/O-overhead bar.

## Fairness rules (apply to ALL baselines)
- **JIT precompile.** Every JIT/compiled baseline (mjx & frax via jax, mujoco_warp via warp,
  cuRobo via torch) must compile its kernels in WARMUP, never in the timed region. Warmup must
  call the EXACT timed closure (same shapes/dtypes, incl. the with-mem numpy→device path) and
  `block_until_ready`/`synchronize`. **Lesson:** mjx's first competitive number was ~41,500 µs
  (a fake 3624×) because the with-mem composite closure wasn't warmed end-to-end — the first
  timed iteration compiled the JIT and dominated the mean. After the fix it's ~1,317 µs (a real,
  credible ~45×). Don't strawman competitors. (Fix: commit b25f407.)
- **Fair competitor surface.** Competitors are timed through THEIR python APIs, which already
  include their wrapper + memory. So the honest library-vs-library bar is GRiD-through-bindings
  (layer 3) vs competitor-python-e2e. The raw compute-only bar (layer 1) is the legitimate
  "data already on GPU" comparison — labeled as such, not passed off as the headline.
- **Pinocchio = CODEGEN (cppADCodeGen), the fast path** — not pin-direct. (Fix: f621640 — a
  `needs_codegen()` token-mismatch left the codegen models uninitialized → SIGSEGV on
  id/fd/id_du/fd_du; now codegen JITs + captures on all robots incl. g1.)
- **Batch sweep:** N ∈ {16, 32, 64, 128, 256, 1024} — include small (32) AND large (1024) batch
  so the throughput story isn't cherry-picked at one size. (N=1024 plumbing: 34aa421; GRiD
  harness MAX_TIMESTEPS raised to 1024; binding needs `-DGRID_RBD_MAX_BATCH=1024` for layer 3.)
- **GRiD config = autotuned best** (per-algo tier×threads from `autotune_best_<host>.json`), so
  GRiD is shown at its real best, not a default. The autotune fixed the FFI thread pathology.
- **Build ≠ time.** Pre-compile all binaries (`--compile-only`, all tiers, RAM-safe
  `GRID_COMPILE_WORKERS=1`) BEFORE the timed run; the timed run is pure
  `--no-recompile` on a quiet GPU, one capture at a time (no concurrent heavy CPU/GPU work).

## Pipeline
1. Pre-compile the GRiD harness (all tiers, N=1024, `--compile-only`) + pin into cache.
2. Timed competitor capture (quiet GPU, serial): `run_competitive_gpu_baselines.sh <dir>` —
   pin-codegen + mjx + frax + mujoco_warp + cuRobo (cuRobo = g1/fixed only), each through its
   own run.py, all sweeping N∈{16,32,64,128,256,1024}.
3. GRiD layers 1+2: `run_multi_version.py --columns glass pinocchio --tiers shared lite minimal
   --no-recompile` → unified json (grid_glass compute_only + with_mem per N; feeds latency/compete
   plots and the analyze transfer-delta). Layer 1 best-tier also read from `autotune_best_<host>.json`.
4. GRiD layer 3: `baselines/grid/timeGRiD_bindings.py --robot {iiwa14,go2} --base {fixed,floating}`
   — wrapper-inclusive jax FFI e2e, SYMMETRIC JIT precompile (precompile bakes
   `-DGRID_RBD_MAX_BATCH=1024`). Emits a `grid_bindings` column.
5. `analyze_competitive.py` → tally; `plot_benchmarks.py` (latency / compete / summary) → figures.

## Known tails to close in the re-capture
- pinocchio BATCH-codegen for id/fd/id_du cells read "—" in the first g1 run (single-call works);
  verify the threaded-codegen path emits batch lines for the verbose algo names.
- grid_rbd binding kMaxBatch defaults to 256 — rebuild with `-DGRID_RBD_MAX_BATCH=1024` for N=1024.
