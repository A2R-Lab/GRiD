"""CPU-only tests of the checks that read a batched runner's replies.

They exercise the Python side with made-up runner output: the kinematics sweep
order, the poison run's nested verdict and the spill test's single reply.
"""
from __future__ import annotations

import pytest

pytest.importorskip("RBDReference")

from test.cuda_equivalents import cuda_harness
from test.cuda_equivalents import test_cuda_fd_du_output_spill as spill
from test.cuda_equivalents import test_cuda_kinematics_thread_invariance as kin
from test.cuda_equivalents import test_smem_poison as poison


@pytest.mark.parametrize("requested", [(1, 2, 16, 32, 64), (1, 2), (64, 32, 1), (32,)])
def test_the_baseline_runs_first_even_when_a_list_leaves_it_out(requested):
    counts = kin._baseline_first(requested)
    assert counts[0] == kin._INVARIANCE_BASELINE_THREADS
    assert set(counts) == set(requested) | {kin._INVARIANCE_BASELINE_THREADS}
    assert len(counts) == len(set(counts))


def test_the_default_sweep_compares_every_other_count_with_the_baseline(monkeypatch):
    monkeypatch.delenv("GRID_CUDA_KIN_THREADS", raising=False)
    counts = kin._thread_counts()
    assert counts[0] == kin._INVARIANCE_BASELINE_THREADS and len(counts) == len(kin._DEFAULT_THREAD_COUNTS)


def _junit(*cases):
    return "<testsuite>" + "".join(cases) + "</testsuite>"


_PASSES = [
    f'<testcase name="test_fixed_base_generated_cuda_matches_python_reference[iiwa14-fixed-{cell}-threads32]"/>'
    for cell in ("inverse_dynamics", "minv", "forward_dynamics", "aba", "crba", "end_effector_pose")
]
_UNSELECTED = ('<testcase name="gradient_cell"><skipped message="x: not in the fixed flagship '
               'algorithm selection"/></testcase>')
_OTHER_SKIP = '<testcase name="value_cell_2"><skipped message="CUDA runtime is unavailable"/></testcase>'
_FAIL = '<testcase name="value_cell_3"><failure message="boom"/></testcase>'


def test_a_poison_run_stands_on_passed_cells_and_unselected_skips():
    assert poison.nested_verdict(_junit(*_PASSES, _UNSELECTED), "iiwa14", "fixed") == []


@pytest.mark.parametrize("cases", [(_UNSELECTED,), (_UNSELECTED, _UNSELECTED), (*_PASSES, _OTHER_SKIP),
                                   (*_PASSES, _FAIL), _PASSES[:-1], ()])
def test_a_poison_run_that_skipped_its_cells_does_not_stand(cases):
    assert poison.nested_verdict(_junit(*cases), "iiwa14", "fixed")


@pytest.mark.parametrize("tokens", [None, {"RUN_FORWARD_DYNAMICS_GRADIENT"}])
def test_one_poison_launch_covers_both_gradient_outputs(tokens):
    blocks = {"forward_dynamics_gradient_q": [[0]], "forward_dynamics_gradient_qd": [[1]],
              "poison_launches": [[1]]}
    cuda_harness._assert_poison_launches(blocks, tokens)
    blocks["poison_launches"] = [[0]]
    with pytest.raises(AssertionError, match="poison launches"):
        cuda_harness._assert_poison_launches(blocks, tokens)


@pytest.mark.parametrize("replacement", [None, {}, {"result": [[2.000001]]}])
def test_sample_order_check_compares_each_named_block(monkeypatch, replacement):
    outputs = [{"result": [[1.0]]}, {"result": [[2.0]]}]

    def rerun(executable, compile_cmd, stdins, **kwargs):
        assert stdins == ["B", "A", "B"]
        return [outputs[1], outputs[0], outputs[1] if replacement is None else replacement]

    monkeypatch.setattr(cuda_harness, "_run_samples", rerun)
    if replacement is None:
        cuda_harness._assert_order_independent(None, [], ["A", "B"], outputs)
    else:
        with pytest.raises(AssertionError, match="sample 1 at position 2"):
            cuda_harness._assert_order_independent(None, [], ["A", "B"], outputs)


def _reply(q, qd):
    return (f"BEGIN forward_dynamics_gradient_q 1 2\n{q} {q}\nEND forward_dynamics_gradient_q\n"
            f"BEGIN forward_dynamics_gradient_qd 1 2\n{qd} {qd}\nEND forward_dynamics_gradient_qd\n")


def test_the_spill_test_reads_one_reply():
    dq, dqd = spill._one_reply(_reply(1.5, 2.5))
    assert dq.tolist() == [1.5, 1.5] and dqd.tolist() == [2.5, 2.5]


@pytest.mark.parametrize("stdout", [_reply(1, 2) + "NEXT_SAMPLE\n" + _reply(3, 4),
                                    _reply(1, 2) + _reply(3, 4), ""])
def test_the_spill_test_refuses_more_or_fewer_than_one_reply(stdout):
    with pytest.raises(AssertionError):
        spill._one_reply(stdout)
