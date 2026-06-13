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
- **Build ≠ time.** Pre-compile all binaries (`build_all_for_recapture.sh`, `--compile-only`,
  all tiers, RAM-safe `GRID_COMPILE_WORKERS=1`) BEFORE the timed run; the timed run is pure
  `--no-recompile` on a quiet GPU, one capture at a time (no concurrent heavy CPU/GPU work).

## Pipeline
1. `build_all_for_recapture.sh` — pre-compile GRiD harness (all tiers, N=1024) + pin into cache.
2. Build grid_rbd bindings (iiwa14/go2, `-DGRID_RBD_MAX_BATCH=1024`) for layer 3.
3. Timed re-capture (quiet GPU, serial): GRiD (3 layers) + pin-codegen + mjx + frax + mujoco_warp
   + cuRobo, robots {iiwa14, go2, g1} × {fixed, floating} × N {32, 256, 1024}.
4. `analyze_competitive.py` → tally; `plot_benchmarks.py` (latency / compete / summary) → figures.

## Known tails to close in the re-capture
- pinocchio BATCH-codegen for id/fd/id_du cells read "—" in the first g1 run (single-call works);
  verify the threaded-codegen path emits batch lines for the verbose algo names.
- grid_rbd binding kMaxBatch defaults to 256 — rebuild with `-DGRID_RBD_MAX_BATCH=1024` for N=1024.
