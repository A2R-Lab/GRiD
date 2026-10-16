"""Standalone FLOATING-base VALUE equivalence — isolated from the gradient algos.

WHY THIS EXISTS (Bug A, 2026-06-17): go2-floating `forward_dynamics` VALUE was wrong
(root [angular,linear] vs public [linear,angular] convention) yet hid for a long time.
It hid because the ONLY place that validates floating FD value is the monolithic
`test_cuda_executable_equivalence` per-robot executable — which ALSO compiles
`forward_dynamics_gradient` (needs `crba_inner`). That build died on the missing
`crba_inner`, and the loud build error masked the latent VALUE bug behind it: FD/aba/
minv/ID/crba VALUES for go2/g1/h1_2-floating were never compared.

This test compiles ONLY the value algos (NO gradients) for floating robots, so a build
failure in a gradient algo can never again void validation of the values. It is the
defensive net for the whole floating-root convention bug class (any algo whose floating
qdd/tau/force root 6-vector is mis-ordered shows up here as a mismatch vs RBDReference).

Run (GPU box):
  GRID_CUDA_FLOATING_VALUE_ROBOTS=go2,g1 \
    .venv/bin/python -m pytest test/cuda_equivalents/test_cuda_floating_values_equivalence.py \
    -m cuda_equivalence -vv
Defaults to every floating manifest robot (widened 2026-10-15 from the two that
were masked, go2 and g1); narrow with the env var. The runner is compiled with
only the value RUN tokens, so the mimic robots (fr3 / h1_2), whose shared mimic
header also carries the gradients, build only the value kernels here.
"""

import os

import pytest

from test.cuda_equivalents.cuda_harness import (
    _run_cuda_equivalence_case,
    _sample_name_selection,
    build_floating_cuda_case_params,
)

# VALUE-only: deliberately EXCLUDES every *_gradient / second-order algo so this
# executable builds even when a gradient algo's codegen dependency (e.g. crba_inner)
# is broken. These are exactly the algos whose floating-root 6-vector output carries
# the [angular,linear]<->[linear,angular] convention and were masked by the build break.
FLOATING_VALUE_ALGORITHMS = (
    "inverse_dynamics",
    "minv",
    "forward_dynamics",
    "aba",
    "crba",
    "end_effector_pose",
)


# The runner RUN tokens for exactly the value algorithms above: with them the
# compile is gated to the value kernels on every robot (a mimic robot's shared
# header carries the gradients too, and without the tokens the runner would
# compile them).
FLOATING_VALUE_RUN_TOKENS = frozenset({
    "RUN_INVERSE_DYNAMICS", "RUN_MINV", "RUN_FORWARD_DYNAMICS", "RUN_ABA", "RUN_CRBA",
    "RUN_END_EFFECTOR_POSE",
})

_ALL_FLOATING_ROBOTS = "iiwa14,go2,g1,h1_2,fr3,rizon4,gen3,fetch,baxter"


def _selected_floating_value_params():
    """Floating cases, subset to GRID_CUDA_FLOATING_VALUE_ROBOTS (default: every
    floating manifest robot).

    Reuses ``build_floating_cuda_case_params`` so the cuda_equivalence/developer_only
    marks + manifest specs come along unchanged; just filters the robot set."""
    want = os.environ.get("GRID_CUDA_FLOATING_VALUE_ROBOTS", _ALL_FLOATING_ROBOTS)
    want_ids = {r.strip() for r in want.split(",") if r.strip()}
    params = []
    for param in build_floating_cuda_case_params():
        # ParameterSet: .values == (spec, base_mode); .marks carried through.
        spec = param.values[0]
        if getattr(spec, "robot_id", None) in want_ids:
            params.append(param)
    return params


@pytest.mark.parametrize(("spec", "base_mode"), _selected_floating_value_params())
def test_floating_value_only_matches_python_reference(spec, base_mode, tmp_path, request):
    """Floating VALUE algos (no gradients) must match RBDReference — catches the
    floating-root convention bug class (Bug A) without being maskable by a gradient
    build failure."""
    selection = _sample_name_selection(base_mode)
    random_count = 0 if selection.explicit and os.environ.get("GRID_CUDA_RANDOM_SAMPLES") is None else None
    _run_cuda_equivalence_case(
        spec,
        base_mode,
        tmp_path,
        FLOATING_VALUE_ALGORITHMS,
        sample_selection=selection,
        random_count=random_count,
        config=request.config,
        num_threads=0,
        # Emit a SUBSET header with ONLY these value algos — no *_gradient, so the
        # header has no crba_inner (or any gradient) dependency. This is what makes
        # the test un-maskable by a gradient codegen break (the Bug A masking mode).
        codegen_algorithm_list=FLOATING_VALUE_ALGORITHMS,
        # Gate the runner to the value kernels (mimic robots share one full header).
        run_tokens=FLOATING_VALUE_RUN_TOKENS,
    )
