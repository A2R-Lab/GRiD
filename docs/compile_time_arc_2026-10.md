# Compile-time arc (2026-10-09/10): how the per-robot `robot.so` is built, and why

Design note for contributors. User-facing knobs: `bindings/README.md`, Fast Robot Setup.
Bug-class learnings: `docs/agent_debugging_guide.md` §7.z43 / §7.z44. Ledger (measurements,
decisions D1–D4, gates): `docs/open-tasks/multi_tu_arc_2026-10-09.md` (gitignored, on the dev box).

## The problem
One `nvcc` translation unit per robot: cicc is single-threaded per TU and ptxas is superlinear
in kernel size, so a full g1 build was 48 min / 8 GB and iiwa14 11 min. Worse, the wrapper
called the header's aggregate `init_grid_kernel_attrs`, which address-takes EVERY kernel —
including the bench-only `_single_timing` twins and overloads the wrapper never launches — so
every build compiled everything.

## What was done (four decisions, all user-ratified)
- **D1** `kernel_attrs.py` emits library-safe per-algorithm registration twins
  `init_grid_kernel_attr_<algo>_checked` / `_mjx_checked` (no `_single_timing` entries). Additive
  change to every generated `grid.cuh` (byte gate: 0 lines removed).
- **D2** Bindings builds register per family through those twins; the twins and never-launched
  overloads (`abi_specs.NOT_LAUNCHED_BY_BINDINGS`) drop out: 3x on the old single unit by itself.
- **D3** Partition by MEASURED unit cost: `abi_specs.FAMILIES` (heavy-first), `FAMILY_OF[key]`
  for every ABI key; the singletons (`integrator_gradient`, `plant_step_gradient`, `plant_hessian`,
  `fdsva_so`, `idsva_so`) are the stragglers that bound the build.
- **D4** ONE build path (no monolith fallback); a jobs knob for small boxes.

## How it works
1. `wrapper_template.cu` is one fenced file. `// [[unit: <family>]]` … `// [[unit: end]]` marks a
   block for a family unit; `; surface: jax|torch` says the block lives inside that surface's
   `#if`; `// [[unit: prelude; surface: …]]` blocks are copied into every family unit using that
   surface. Fences never nest. Generated regions emit their own fences (`wrapper_body_gen.py`,
   `wrapper_plant_gen.py`); hand-written blocks carry hand fences.
2. `bindings/grid_rbd/_units.py` splits the template into `wrapper_core.cu` + `wrapper_<family>.cu`
   (`generate_sources` writes them beside `grid.cuh`); `check_partition` is the drift referee
   (every line in exactly one unit). Shared declarations live in `wrapper_common.cuh`.
3. Core address-takes NO kernel: it dispatches by name to per-family functions that live in the
   owning unit — `grid_rbd_family_<f>_init_attrs`, `grid_rbd_family_<f>_kernel_ceiling`,
   `grid_rbd_torch_def_/impl_<f>(torch::Library&)`; core owns the single `TORCH_LIBRARY` and is the
   only unit compiled with `-DGRID_RBD_TORCH_KEY=<store key>`.
4. `_compile._build_units` compiles every unit `-c` in a `ThreadPoolExecutor`
   (`GRID_RBD_BUILD_JOBS`, default `min(8, cores, MemAvailable/3 GB)`, heavy families first),
   adds `-Xfatbin -compress-all` (`-c` objects carry uncompressed fatbins), host-links with
   `nvcc --shared`. No `-rdc`: each unit's kernels are launched only from that unit.
5. Per-unit object cache `<cache_dir>/objects/<unit>-<key>.o`, key = sha256(unit text, grid.cuh,
   wrapper_common.cuh, flags minus the staging dir's `-I`, nvcc/host-cxx/GLASS/torch/jax tags).
   Hits link without nvcc; objects are kept forever (`GRID_RBD_OBJECT_CACHE_DAYS=N` opts in to
   pruning); `force_rebuild` recompiles every unit and refreshes the cache. Store GC:
   `python -m grid_rbd.gc_cache`.

## Numbers (RTX 5090 / 24-core, 8 units, idle box, 2026-10-10)
| build | one unit (09-24) | multi-unit | build RAM |
|---|---|---|---|
| iiwa14 fixed, all algorithms | 10 min 53 s | 101 s | 12 GB |
| go2 floating, all algorithms | 14 min 55 s | 128 s | 11 GB |
| g1 floating, all algorithms | 53 min 29 s | 366 s | 14 GB |
| g1, `algorithm_list` fd/id/fd-grad | 3 min | 78 s | 10 GB |
| iiwa14 after a wrapper comment edit | full rebuild | 22 s (12/14 units cached) | — |

`--split-compile 8` on the g1 `integrator_gradient` unit: 362 → 120 s, outputs bitwise identical,
kernel +1.3 % (noise), ~3x RAM, SASS differs → opt-in only (`GRID_RBD_SPLIT_COMPILE`), never on
test runners, never a default.

## Rules that fall out of it
- A new launch site, handler, registration or torch impl goes INSIDE its family's fence; the core
  referee (`test/test_wrapper_units.py`) fails otherwise.
- A new ABI key needs a `FAMILY_OF` entry (crosscheck test); a new family needs `FAMILIES` +
  `FAMILY_ATTR_SHORTS`.
- Measure before repartitioning: the straggler unit decides the wall time, not the count.
- Receipt: `run_validation.py` strips `GRID_*` env (set caps inside `validation_env`), needs a
  clean tree and an unmoved HEAD for the run; a grid.cuh byte change re-keys every CUDA
  equivalence header cache (one-time ~4.5 h of prewarms).

## Open (receipt-infra arc, `docs/open-tasks/receipt_infra_arc_2026-10-10.md`)
CUDA equivalence runner still registers via the aggregate (R1); `compile_sched` peak ledger sees
only the largest child (S1); heavy-first prewarm order (R3); two wrapper shards at a time (W1).
