"""GRiD test-suite conftest.

Wires pytest-gpu-proof into the CUDA/wrapper suites without touching every test:
any item already carrying the ``cuda_equivalence`` or ``python_wrappers`` marker
is auto-tagged ``gpu_proof`` so its outcome lands in the signed receipt
(``[tool.gpu_proof] required_marker = "gpu_proof"``). See test/run_gpu_proof.sh.

The plugin now ships from PyPI (``pytest-gpu-proof`` in requirements-dev.txt); it
was previously a vendored ``test/pytest-gpu-proof`` submodule.

★ THE SUITE FLOOR (2026-07-14). The GPU suite has 219 `pytest.skip` sites, ~61 of them a per-file
"nvcc not found" / "not in manifest" / "URDF parse failed" guard. Individually reasonable; collectively
they mean that if nvcc falls off PATH or the model manifest breaks, EVERY CUDA test skips, pytest exits
0, and CI is GREEN while nothing was actually tested. There was no floor. This adds one: when the box is
CAPABLE (nvcc on PATH AND a GPU visible), an environment-guard skip is treated as a HARD FAILURE — a
capable box skipping for "nvcc not found" is a real breakage, not an absent environment. On a genuinely
incapable box (no nvcc / no GPU) the guards skip freely, as before. Escape hatch for a deliberate partial
run: `GRID_TEST_NO_FLOOR=1`.
"""

import os
import re
import shutil
import subprocess

import pytest

_GPU_PROOF_SOURCE_MARKERS = ("cuda_equivalence", "python_wrappers")

# Skip reasons that CANNOT legitimately fire on a capable box — they mean the toolchain, the model
# manifest, or a VENDORED asset broke. (Hardware-limit skips like "shared-memory request > cap" or
# "too many resources requested for launch" are REAL even when capable, use different wording, and are
# deliberately NOT matched here.)
_ENV_GUARD_SKIP = re.compile(
    r"nvcc not found|nvcc not on PATH|CUDA runtime unavailable|"
    r"not in (the )?manifest|Could not resolve manifest|"
    r"URDF (parse failed|fixture not present)|\.urdf not (found|vendored)",
    re.IGNORECASE,
)


def _box_is_capable() -> bool:
    """nvcc on PATH AND at least one GPU visible. Cached on first call."""
    if getattr(_box_is_capable, "_cached", None) is None:
        capable = shutil.which("nvcc") is not None
        if capable:
            try:
                out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
                capable = out.returncode == 0 and "GPU" in out.stdout
            except Exception:
                capable = False
        _box_is_capable._cached = capable
    return _box_is_capable._cached


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu_proof`` marker to every CUDA/wrapper equivalence item.

    Keeps the receipt's membership in lockstep with the existing marker taxonomy:
    add a new cuda_equivalence/python_wrappers test and it is covered automatically,
    with no per-test ``@pytest.mark.gpu_proof`` to remember.
    """
    for item in items:
        if any(item.get_closest_marker(name) for name in _GPU_PROOF_SOURCE_MARKERS):
            item.add_marker(pytest.mark.gpu_proof)


def pytest_runtest_logreport(report):
    """Record environment-guard skips so the session-floor can fail on them (capable box only)."""
    if report.when != "setup" and report.when != "call":
        return
    if report.skipped and isinstance(report.longrepr, tuple):
        # longrepr for a skip is (path, lineno, "Skipped: <reason>")
        reason = report.longrepr[2]
        if _ENV_GUARD_SKIP.search(reason):
            _ENV_GUARD_SKIPS.append(f"{report.nodeid}: {reason}")


_ENV_GUARD_SKIPS: list[str] = []


def pytest_sessionfinish(session, exitstatus):
    """THE FLOOR: on a capable box, any environment-guard skip fails the session."""
    if os.environ.get("GRID_TEST_NO_FLOOR") == "1":
        return
    if not _ENV_GUARD_SKIPS or not _box_is_capable():
        return
    n = len(_ENV_GUARD_SKIPS)
    session.exitstatus = 1
    shown = "\n  ".join(_ENV_GUARD_SKIPS[:15])
    more = f"\n  ... and {n - 15} more" if n > 15 else ""
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "SUITE FLOOR VIOLATED", red=True, bold=True)
        reporter.write_line(
            f"{n} test(s) skipped for an ENVIRONMENT-GUARD reason on a CAPABLE box (nvcc + GPU present).\n"
            f"On a capable box these guards must not fire — they mean the toolchain, the model manifest, "
            f"or a vendored asset is broken, and a green run would be testing NOTHING. Fix the breakage "
            f"(or set GRID_TEST_NO_FLOOR=1 for a deliberate partial run):\n  {shown}{more}"
        )
