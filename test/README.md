# test/ — what lives where

Four suites in subdirectories, plus a root full of CPU-only gates and the
receipt machinery. Commands + the receipt policy:
`docs/source/user_guide/tutorials/cuda_validation.rst` (and
`test/run_gpu_proof.sh --help`).

## Subdirectories

| dir | what | marker | GPU? |
|-----|------|--------|------|
| `cuda_equivalents/` | generated-CUDA vs numpy-oracle equivalence (per-algo runners + the split-suite harness) | `cuda_equivalence` | yes |
| `python_wrappers/` | the `grid_rbd` numpy/jax/torch handle suites | `python_wrappers` | yes |
| `benchmarks/` | the timing harness + orchestration scripts (never run during tests) | — | yes |
| `diagnostics/` | manual probes (tier instantiation smoke, tier baselines) — not collected by CI | — | yes |

⚠ `cuda_equivalents/` + `python_wrappers/` are the RECEIPT-FINGERPRINTED
trees: editing any file there stales the corresponding `gpu-proof.json`
shard(s) and turns CI's verify job red until a refresh
(`SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh`). Root files are NOT
fingerprinted.

## Root files, bucketed

**Receipt machinery** (the split driver stack):
`run_gpu_proof.sh` (entry point), `run_split_suite.py` (shard
partition/resume/carry), `compile_sched.py` (RAM-aware parallel compile
pool), `prewarm_cuda_flagship.py` (header/exe pre-warm),
`gpu-proof-policy.yaml` / `gpu-proof-policy-release.yaml` (everyday vs
release verification), `gpu-proof-expected-skips.txt`.

**Drift gates** (CPU-only; keep the generated regions + registry honest):
`test_wrapper_generated_block.py`, `test_core_generated_block.py`,
`test_abi_spec_crosscheck.py`, `test_core_pyside_crosscheck.py`,
`test_algo_descriptor_parity.py`, `test_algo_descriptor_arena_parity.py`,
`test_kernel_attr_manifest_consistency.py`, `test_feature_macro_coverage.py`,
`test_shared_arena_covers_carve.py`, `test_dynamics_fingerprint.py`.

**Policy/hygiene gates** (CPU-only): `test_marker_hygiene.py`,
`test_fast_compile_hygiene.py`, `test_plant_launch_hygiene.py`,
`test_bench_algo_arity.py`, `test_split_partition.py` (the shard-partition
logic itself), `test_collision_spherize.py`, `test_collision_flange_mapping.py`.

**Shared infra**: `conftest.py` (marker auto-application, CPU/GPU filters),
`helpers.py` (sample-input generation), `prepare_reference_models.py`.

## Artifacts (safe to delete; regenerated on demand)

- `test/.split_suite/` — receipt run dirs (`receipt_<stamp>/`), rolling shard
  durations, the compile-RSS ledger. Old `receipt_*` dirs are prunable.
- `.pytest_cache/` — ⚠ currently also holds the content-keyed nvcc build
  cache (several GB, warm = fast reruns); `pytest --cache-clear` nukes it.
  (Planned move to `.grid_build_cache/`.)
- `test/benchmarks/results/` — timing sweep outputs; prunable, but old
  sweeps feed the rolling launch-config bakes — check before bulk-deleting.
