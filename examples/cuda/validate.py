#!/usr/bin/env python3
"""Validate the GRiD CUDA example output against the RBDReference numpy oracle.

Pipes the deterministic inputs through RBDReference.inverse_dynamics (the same
q/qd/qdd hard-coded in inverse_dynamics_kernel_example.cu) and diffs against the
labelled blocks the compiled example prints on stdin.

    PYTHONPATH=. .venv/bin/python examples/cuda/validate.py < example_output.txt
"""
from __future__ import annotations

import sys

import numpy as np
from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from RBDReference import RBDReference

GRAVITY = -9.81


def make_inputs(n: int):
    q = np.array([0.1 * (i + 1) for i in range(n)])
    qd = np.array([0.01 * (i + 1) for i in range(n)])
    qdd = np.array([0.02 * (i + 1) for i in range(n)])
    return q, qd, qdd


def parse_blocks(text: str) -> dict[str, np.ndarray]:
    blocks: dict[str, np.ndarray] = {}
    label = None
    vals: list[float] = []
    for line in text.splitlines():
        if line.startswith("BEGIN "):
            label, vals = line[6:].strip(), []
        elif line.startswith("END "):
            if label is not None:
                blocks[label] = np.array(vals)
            label = None
        elif label is not None and line.strip():
            vals.extend(float(x) for x in line.split())
    return blocks


def ref_id(ref: RBDReference, q, qd, qdd) -> np.ndarray:
    c = ref.inverse_dynamics(q, qd, qdd, GRAVITY=GRAVITY)
    if isinstance(c, tuple):
        c = c[0]
    return np.asarray(c).ravel()


def main() -> int:
    robot = URDFParser().parse(iiwa14_description.URDF_PATH, floating_base=False)
    ref = RBDReference(robot)
    n = robot.get_num_joints()
    q, qd, qdd = make_inputs(n)

    blocks = parse_blocks(sys.stdin.read())
    if not blocks:
        print("ERROR: no BEGIN/END blocks found on stdin", file=sys.stderr)
        return 2

    # Oracle for the single-state paths (_device, _inner) and each batch element.
    oracle = {"inverse_dynamics_device": ref_id(ref, q, qd, qdd),
              "inverse_dynamics_inner": ref_id(ref, q, qd, qdd)}
    B = 4
    for k in range(B):
        qk = q + 0.05 * k
        qdk = qd + 0.01 * k
        qddk = qdd + 0.02 * k
        oracle[f"inverse_dynamics_batch_{k}"] = ref_id(ref, qk, qdk, qddk)

    worst = 0.0
    ok = True
    for label, ref_vec in oracle.items():
        if label not in blocks:
            print(f"MISSING block: {label}", file=sys.stderr)
            ok = False
            continue
        got = blocks[label]
        denom = max(np.linalg.norm(ref_vec), 1e-9)
        rel = np.linalg.norm(got - ref_vec) / denom
        worst = max(worst, rel)
        status = "OK" if rel < 1e-4 else "FAIL"
        if rel >= 1e-4:
            ok = False
        print(f"[{status}] {label:32s} rel_err={rel:.3e}")

    print(f"\nworst rel_err = {worst:.3e}  (float32 RNEA, tol 1e-4)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
