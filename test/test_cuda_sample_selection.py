"""CPU-only guard: the default floating sample selection must actually run its samples.

The floating flagship default named four samples but passed
include_corner_samples=False. Only "zero" is built without corners, so the other
names silently matched nothing and floating cells compared the zero state only.
Two generator bugs hid behind that for weeks: the floating x mimic ID/stride/ABA
bugs (agent_debugging_guide 7.z32) and the floating ABA NaN at an axis-permutation
base rotation (7.z33).
"""
from __future__ import annotations

import pytest

from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from test.cuda_equivalents import cuda_harness

_SAMPLE_ENV = ("GRID_CUDA_SAMPLE_NAMES", "GRID_CUDA_FLOATING_SAMPLE_NAMES", "GRID_CUDA_RANDOM_SAMPLES")


def _floating_spec(robot_id):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id} has no floating manifest entry")


@pytest.mark.parametrize("robot_id", ["iiwa14", "go2"])
def test_default_floating_selection_builds_every_named_sample(robot_id, monkeypatch):
    for key in _SAMPLE_ENV:
        monkeypatch.delenv(key, raising=False)
    selection = cuda_harness._sample_name_selection("floating")
    assert selection.names == set(cuda_harness.FLOATING_DEFAULT_SAMPLE_NAMES)
    assert not selection.explicit
    spec = _floating_spec(robot_id)
    model = build_project_adapter(spec, resolve_robot_spec(spec), base_mode="floating")
    samples = cuda_harness._build_cuda_samples(
        model, random_count=0, include_corner_samples=selection.include_corner_samples)
    matched = [s for s in samples if s.name in selection.names]
    assert {s.name for s in matched} == selection.names
    assert any(abs(s.qd).max() > 0 for s in matched)                    # velocity path exercised
    assert any(abs(s.q[3:6]).max() > 0 for s in matched if s.name.startswith("floating_quat"))


def test_fixed_default_selection_is_unfiltered(monkeypatch):
    for key in _SAMPLE_ENV:
        monkeypatch.delenv(key, raising=False)
    assert cuda_harness._sample_name_selection("fixed") == cuda_harness.SampleSelection(None, False, False)
