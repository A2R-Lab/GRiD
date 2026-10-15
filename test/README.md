# test/ layout

Four suites in subdirectories, plus a root full of CPU-only gates and the
receipt machinery. Commands and the receipt policy are in
`docs/source/user_guide/tutorials/cuda_validation.rst`; the launcher is
`test/run_validation.py` (`--help`). [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)
explains how the equivalence tests are built and why; read it before adding a
kernel or a test.

## Subdirectories

| dir | what | marker | GPU? |
|-----|------|--------|------|
| `cuda_equivalents/` | generated-CUDA vs numpy-oracle equivalence (per-algo runners + the split-suite harness) | `cuda_equivalence` | yes |
| `python_wrappers/` | the `grid_rbd` numpy/jax/torch handle suites | `python_wrappers` | yes |
| `benchmarks/` | the timing harness + orchestration scripts; their CPU unit tests (`test/benchmarks/test_*.py`) run in the CPU lane, the timing itself never runs during tests | none | timing only |
| `diagnostics/` | manual probes (tier instantiation smoke, tier baselines), not collected by CI | none | yes |

Receipt correctness inputs are listed in `pyproject.toml` under
`tool.gpu_proof.fingerprint_paths`: generator, bindings, peer gitlinks,
configuration, GPU tests and shared harness/driver files. Changes require a
refresh (`test/run_validation.py --execute`, everyday; `--full` for a release
receipt with no carried shards); the driver determines which shards may carry.
Edits outside the fingerprinted paths (the Sphinx site, top-level docs) do not
stale evidence.
`test/gpu-proof-tests.txt` is the reviewed exact node-ID set for main (policy
`required_test_manifest`). After a reviewed full collection changes, regenerate it
with `test/receipt_integrity.py --receipt <run>/gpu-proof.json --write-manifest
test/gpu-proof-tests.txt` and commit it before recording; never replace it with a smoke set.

## Root files

### Receipt machinery

The split driver stack:
`run_validation.py` (the launcher; dry by default. `--execute` runs an
everyday refresh, `--full` a fresh release receipt, `--finalize <run>` verifies
a committed run; it strips `GRID_*` from its env and holds the timing lock),
`run_split_suite.py` (the driver the launcher runs: shard
partition/resume/carry), `run_gpu_proof.sh` (a thin wrapper over that driver
for ad-hoc runs; `SPLIT=1` selects it, `SCOPE=smoke|curated` narrows a local
diagnostic that is never a candidate receipt),
`receipt_integrity.py` (header-key verification + `--write-manifest` for
`gpu-proof-tests.txt`), `compile_sched.py` (RAM-aware parallel compile
pool), `prewarm_cuda_flagship.py` (flagship, integrator and SO smoke pre-warm;
read-only integrator/SO cache inventory with `--inventory --node-ids
test/gpu-proof-tests.txt --out inventory.json`),
`gpu-proof-policy.yaml` / `gpu-proof-policy-release.yaml` (everyday vs
release verification), `gpu-proof-tests.txt` (the exact node-ID manifest),
`gpu-proof-expected-skips.txt` (the skip baseline CI applies),
`codegen_neutrality.py` (refresh-soundness prover: covering matrix +
robot-asset gate), `header_key_replay.py` + `test_header_key_replay.py` +
`gpu-proof-header-keys.json` (per-shard byte-precise header replay, the
cuda-carry gate), `dynamics_fingerprint.json` + `test_dynamics_fingerprint.py`
(fleet numeric fingerprints).

### Receipt and driver gates

CPU-only referees for the machinery above:
`test_validation_launcher.py` (launcher acceptance matrix, driver calls
simulated), `test_receipt_integrity.py`, `test_gpu_proof_scope.py` (code
invalidates the receipt, website wording does not), `test_suite_floor.py`
(the conftest skip floor matches the harness's real skip strings),
`test_runner_registration.py` (CUDA runners register only the kernels they
launch), `test_prewarm_smoke.py`, `test_executable_cache.py` +
`test_cuda_cache_locking.py` (content-keyed build cache, one writer per key),
`test_cuda_sample_selection.py` + `test_integrator_sample_conditioning.py`
(the equivalence samples actually run and stay float32-meaningful).

### Codegen drift gates

CPU-only; they keep the generated regions and the registry consistent:
`test_wrapper_generated_block.py`, `test_core_generated_block.py`,
`test_abi_spec_crosscheck.py`, `test_core_pyside_crosscheck.py`,
`test_cabi_direct_mirror_sizes.py`, `test_algo_descriptor_parity.py`,
`test_algo_descriptor_arena_parity.py`, `test_algo_profiles_closure.py`,
`test_algorithm_completion_invariants.py`, `test_emission_plan.py`,
`test_generation_env_knobs.py` (one env-knob list, `grid_codegen/env_knobs.py`),
`test_kernel_attr_manifest_consistency.py`, `test_feature_macro_coverage.py`,
`test_shared_arena_covers_carve.py`, `test_helper_sincos_scratch_fits.py`,
`test_dynamics_fingerprint.py`, `test_header_fragments.py`,
`test_signature_uniformity.py`, `test_out_transform_equivalence.py`,
`test_vjp_common_equivalence.py`, `test_wrapper_plant_block.py`,
`test_integrator_contract_codegen.py`, `test_floating_aba_codegen.py`,
`test_mimic_floating_codegen.py`, `test_configuration_cotangent.py`
(quaternion / spherical-joint pullbacks), `test_bounded_spheres.py`,
`test_launch_config_gpu_select.py` (baked launch profile keyed on the device).

### Packaging

CPU-only: `test_package_resources.py` (installed resources +
pinned-source provenance).

### Wrappers

`grid_rbd`, CPU-only; the GPU handle suites are `python_wrappers/`:
`test_get_robot_jit.py` (compile-on-miss + the build
notice), `test_gc_cache.py`, `test_rbd_cache_identity.py`,
`test_unit_object_cache.py` + `test_wrapper_units.py` (multi-unit build:
fence grammar, per-unit object cache), `test_jax_backend_guard.py`,
`test_jax_handwritten_packing.py`, `test_operand_validation_table.py`.

### Policy and hygiene gates

CPU-only: `test_marker_hygiene.py`,
`test_fast_compile_hygiene.py`, `test_plant_launch_hygiene.py`,
`test_bench_algo_arity.py`, `test_split_partition.py` (the shard-partition
logic itself), `test_collision_spherize.py`, `test_collision_flange_mapping.py`.

### Shared infrastructure

`conftest.py` (marker auto-application, CPU/GPU filters),
`helpers.py` (sample-input generation), `prepare_reference_models.py`.

## Artifacts (safe to delete; regenerated on demand)

- `test/.split_suite/`: receipt run dirs (`receipt_<stamp>/`), rolling shard
  durations, the compile-RSS ledger. Old `receipt_*` dirs are prunable.
- `.grid_build_cache/cuda`: the content-keyed nvcc build cache (several GB;
  warm means fast reruns). Safe to delete; the next run rebuilds cold.
  `pytest --cache-clear` does not touch the build cache.
- `test/benchmarks/results/`: timing sweep outputs; prunable, but old
  sweeps feed the rolling launch-config bakes, so check before bulk-deleting.
