"""Pin the per-timestep INPUT ABI of the generated header (`h_q_qd_u` / `d_q_qd_u`).

This is a PUBLISHED contract, not an internal detail: consumers that use `grid.cuh`
directly (GATO, MPCGPU, PDDP, any hand-written host code) pack this buffer
themselves, with no Python-side guard to catch a mistake.

The contract
------------
The per-timestep input block is **three NUM_POS-wide slots**::

    stride = Q_QD_U_STRIDE = 3 * NUM_POS          (NUM_JOINTS == NUM_POS)
    q   at  block + 0
    qd  at  block + NUM_POS
    u   at  block + 2*NUM_POS                     (or qdd, for the q|qd|qdd kernels)

On a FIXED base nq == nv, so "three nq-wide slots" and a tight ``q|qd|u`` packing are
the same bytes — which is exactly why a wrong tight assumption can sit unnoticed. On a
quaternion FLOATING base they diverge: nq = nv + 1, so a consumer that packs tightly
writes ``u`` at ``NUM_POS + NUM_VEL`` while every kernel reads it from ``2*NUM_POS``.
Nothing traps that — the read is in-bounds, so the result is silently wrong dynamics.
Floating base is new and about to be adopted downstream, so the ABI is pinned here.

On a floating base ``qd``/``u``/``qdd`` are still passed at the **nq** width: the nv
meaningful values occupy the LEADING slots of their block and the trailing slot is a
pad. (Matrix/gradient OUTPUTS are nv-wide — the asymmetry is the documented "Friction 3"
footgun that ``grid_rbd``'s ``_check_nq_width`` raises on for binding users.)

These checks are pure codegen text/constants (no GPU): they catch a silently MOVED
offset, which the end-to-end equivalence suite would also catch but only for algorithms
whose floating path is exercised with a non-degenerate ``u``.
"""
from __future__ import annotations

import re

import pytest

from test.cuda_equivalents.test_cuda_codegen_layout import _constants, _generate_header

# `T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[14]; T *s_u = &s_q_qd_u[28];` and the
# q|qd|qdd twins. Captured per-slice so a moved offset names itself in the failure.
_SLICE_RE = re.compile(
    r"T \*s_(?P<name>qd|u|qdd) = &s_(?P<buf>q_qd_u|q_qd_qdd)\[(?P<offset>\d+)\]"
)


def _slices(header: str):
    return [(m.group("name"), m.group("buf"), int(m.group("offset")))
            for m in _SLICE_RE.finditer(header)]


@pytest.mark.cuda_equivalence
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id, base_mode", [
    ("iiwa14", "floating"),
    ("go2", "floating"),
    ("iiwa14", "fixed"),
])
def test_input_block_is_three_num_pos_slots(tmp_path, robot_id, base_mode):
    header = _generate_header(tmp_path, robot_id, base_mode)
    c = _constants(header)
    num_pos, num_vel = c["NUM_POS"], c["NUM_VEL"]

    # NUM_JOINTS is the POSITION-space count; the device allocation and the host
    # malloc are written as `3*NUM_JOINTS*NUM_TIMESTEPS`, while kernels stride by
    # Q_QD_U_STRIDE. If these two ever diverge the buffer is under-allocated and
    # every timestep past the first runs off the end of it.
    assert c["NUM_JOINTS"] == num_pos, "NUM_JOINTS must equal NUM_POS (alloc uses 3*NUM_JOINTS)"
    assert c["Q_QD_U_STRIDE"] == 3 * num_pos, (
        f"input stride must be 3*NUM_POS ({3*num_pos}), got {c['Q_QD_U_STRIDE']} — this is a "
        "PUBLISHED ABI; changing it breaks every consumer packing h_q_qd_u")
    assert "3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)" in header, (
        "d_q_qd_u/h_q_qd_u allocation no longer sized 3*NUM_JOINTS per timestep")

    slices = _slices(header)
    assert slices, "no s_qd/s_u/s_qdd slice lines found — did the input plumbing change?"
    for name, buf, offset in slices:
        expected = num_pos if name == "qd" else 2 * num_pos
        assert offset == expected, (
            f"s_{name} is read from s_{buf}[{offset}] but the ABI puts it at {expected} "
            f"(NUM_POS={num_pos}, NUM_VEL={num_vel}). A tight packing would put it at "
            f"{num_pos + num_vel} — that is the silent-corruption bug this test exists for.")

    # The named offset constants are the ABI as consumers see it in grid.cuh —
    # they must agree with the offsets the kernels actually slice at.
    assert c["GRID_Q_OFFSET"] == 0
    assert c["GRID_QD_OFFSET"] == num_pos, "GRID_QD_OFFSET must equal NUM_POS (slot 1)"
    assert c["GRID_U_OFFSET"] == 2 * num_pos, "GRID_U_OFFSET must equal 2*NUM_POS (slot 2)"
    assert c["GRID_QDD_OFFSET"] == c["GRID_U_OFFSET"], (
        "qdd shares slot 2 with u (the q|qd|qdd kernels read the same offset)")


@pytest.mark.cuda_equivalence
@pytest.mark.robot_smoke
def test_floating_slot_padding_is_real_and_fixed_base_hides_it(tmp_path):
    """The floating ABI genuinely pads; the fixed-base one genuinely cannot.

    Guards the REASON this class of bug hides: on fixed base tight == slotted, so a
    fixed-base-only test can never distinguish them and proves nothing about floating.
    """
    floating = _constants(_generate_header(tmp_path, "iiwa14", "floating"))
    fixed = _constants(_generate_header(tmp_path, "iiwa14", "fixed"))

    assert floating["NUM_POS"] == floating["NUM_VEL"] + 1, (
        "quaternion floating base must carry exactly one more position than velocity")
    tight = floating["NUM_POS"] + 2 * floating["NUM_VEL"]
    assert floating["Q_QD_U_STRIDE"] > tight, (
        "floating input block must be SLOTTED (3*NUM_POS), strictly larger than a tight "
        f"q|qd|u packing ({tight}) — the difference is the per-block pad")
    assert floating["Q_QD_U_STRIDE"] - tight == 2 * (floating["NUM_POS"] - floating["NUM_VEL"]), (
        "pad must be exactly one trailing slot in each of the qd and u blocks")

    assert fixed["NUM_POS"] == fixed["NUM_VEL"], "fixed base must have nq == nv"
    assert fixed["Q_QD_U_STRIDE"] == fixed["NUM_POS"] + 2 * fixed["NUM_VEL"], (
        "on fixed base the slotted and tight layouts must coincide (nq == nv)")
