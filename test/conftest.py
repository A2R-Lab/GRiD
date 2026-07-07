"""GRiD test-suite conftest.

Wires pytest-gpu-proof into the CUDA/wrapper suites without touching every test:
any item already carrying the ``cuda_equivalence`` or ``python_wrappers`` marker
is auto-tagged ``gpu_proof`` so its outcome lands in the signed receipt
(``[tool.gpu_proof] required_marker = "gpu_proof"``). See test/run_gpu_proof.sh.

The plugin now ships from PyPI (``pytest-gpu-proof`` in requirements-dev.txt); it
was previously a vendored ``test/pytest-gpu-proof`` submodule.
"""

import pytest

_GPU_PROOF_SOURCE_MARKERS = ("cuda_equivalence", "python_wrappers")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu_proof`` marker to every CUDA/wrapper equivalence item.

    Keeps the receipt's membership in lockstep with the existing marker taxonomy:
    add a new cuda_equivalence/python_wrappers test and it is covered automatically,
    with no per-test ``@pytest.mark.gpu_proof`` to remember.
    """
    for item in items:
        if any(item.get_closest_marker(name) for name in _GPU_PROOF_SOURCE_MARKERS):
            item.add_marker(pytest.mark.gpu_proof)
