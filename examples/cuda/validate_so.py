#!/usr/bin/env python3
"""Validate idsva_so_host_example.cu against the RBDReference second-order oracle.

The example prints the flat 4*NV^3 tensor; this reconstructs the SAME fold the
CUDA equivalence harness uses (the four NVxNVxNV blocks back-to-back, C-order):
    [ d2tau_dq | d2tau_dqd | d2tau_dvdq | dM_dq ]
and diffs block by block against RBDReference.equivalents.

    PYTHONPATH=. .venv/bin/python examples/cuda/validate_so.py < so_output.txt
"""
from __future__ import annotations

import sys

import numpy as np
from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from RBDReference import RBDReference

GRAVITY = -9.81
BLOCK_NAMES = ("d2tau_dq", "d2tau_dqd", "d2tau_dvdq", "dM_dq")


def parse_block(text: str) -> np.ndarray:
    vals: list[float] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("BEGIN idsva_so_body_frame"):
            inside, vals = True, []
        elif line.startswith("END idsva_so_body_frame"):
            inside = False
        elif inside and line.strip():
            vals.extend(float(x) for x in line.split())
    return np.array(vals)


def main() -> int:
    robot = URDFParser().parse(iiwa14_description.URDF_PATH, floating_base=False)
    nv = robot.get_num_joints()
    n = nv

    q = np.array([0.1 * (i + 1) for i in range(n)])
    qd = np.array([0.01 * (i + 1) for i in range(n)])
    qdd = np.array([0.02 * (i + 1) for i in range(n)])

    ref_model = RBDReference(robot)
    # (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq), each NVxNVxNV
    blocks = ref_model.idsva_so_body_frame(q, qd, qdd, GRAVITY=GRAVITY)
    ref = np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks])

    got = parse_block(sys.stdin.read())
    if got.size == 0:
        print("ERROR: no idsva_so_body_frame block on stdin", file=sys.stderr)
        return 2
    if got.size != ref.size:
        print(f"ERROR: size mismatch CUDA={got.size} ref={ref.size}", file=sys.stderr)
        return 2

    bs = nv ** 3
    ok = True
    worst = 0.0
    for bi, name in enumerate(BLOCK_NAMES):
        sl = slice(bi * bs, (bi + 1) * bs)
        denom = max(np.linalg.norm(ref[sl]), 1e-9)
        rel = np.linalg.norm(got[sl] - ref[sl]) / denom
        worst = max(worst, rel)
        status = "OK" if rel < 1e-3 else "FAIL"
        if rel >= 1e-3:
            ok = False
        print(f"[{status}] {name:12s} rel_err={rel:.3e}")

    print(f"\nworst rel_err = {worst:.3e}  (float32 second-order, tol 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
