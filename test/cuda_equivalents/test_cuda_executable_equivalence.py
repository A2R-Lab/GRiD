"""The flagship CUDA-vs-numpy equivalence tests (Wave D harness split,
2026-09-09).

This module now holds ONLY the two flagship test functions; the ~1800-line
shared harness (codegen + nvcc compile chain with its content-keyed caches,
runner execution/parsing, sampling, tolerances, comparison) lives in
``cuda_harness.py``, which the 49 sibling ``test_cuda_*`` modules import.

Why the split: a harness edit affects EVERY cuda module, while an edit to
these two tests affects only the flagship shards — with the harness inside
this test module the receipt fingerprints could not tell those apart (the
same dishonesty class the wrapper shards had with bindings sources; both
fixed 2026-09-09). The split driver folds ``cuda_harness.py`` into every
cuda shard's fingerprint, so harness edits stale the whole domain HONESTLY
and flagship-test edits stale only the flagship shards.

Node ids are unchanged by construction (same module path, same test names,
same id scheme), so durations ledgers and receipt shard names carry over.

The (robot, base, cell, thread count) product is built by
``build_flagship_cell_params`` (2026-10-15) rather than by stacked parametrize
decorators, so a single launch point that cannot run on this hardware is
removed from the collection (``FLAGSHIP_DECOLLECTED_CELLS`` in the harness,
with the reason at the table) instead of being skipped at run time.
"""
import pytest

from test.cuda_equivalents.cuda_harness import (
    _run_flagship_split_cell,
    build_flagship_cell_params,
)


@pytest.mark.parametrize(("spec", "base_mode", "cell", "num_threads"), build_flagship_cell_params("fixed"))
def test_fixed_base_generated_cuda_matches_python_reference(spec, base_mode, cell, num_threads, tmp_path, request):
    _run_flagship_split_cell(spec, base_mode, cell, num_threads, tmp_path, request)


@pytest.mark.parametrize(("spec", "base_mode", "cell", "num_threads"), build_flagship_cell_params("floating"))
def test_floating_base_generated_cuda_matches_python_reference(spec, base_mode, cell, num_threads, tmp_path, request):
    _run_flagship_split_cell(spec, base_mode, cell, num_threads, tmp_path, request)
