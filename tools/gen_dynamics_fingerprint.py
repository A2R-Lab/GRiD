#!/usr/bin/env python
"""Generate test/dynamics_fingerprint.json — the model-alignment probe table.

Upstreamed from GATO (their tools/gen_dynamics_fingerprint.py, schema 1): the
fingerprint pins THE PLANT MODEL (URDF + dynamics semantics), not any solver.
For each fixed-base manifest robot, forward-dynamics accelerations qdd at a
small set of canonical (q, qd, u) probes, evaluated by the RBDReference numpy
oracle (pinocchio-validated, float64). Any consumer — GATO, MPCGPU, PDDP, a
MuJoCo/Isaac harness, a hardware pipeline — evaluates ITS model at the same
probes and compares per joint in seconds; a x2.6-3.2 per-joint ratio is an
effective-inertia mismatch (the PDDP cl_gato round-5 class), not noise.

Probe design (deliberately damping-immune where it matters):
  - "gravity":   qd = 0, u = 0 at the rest posture     -> gravity/bias vector
  - "inertia_j": qd = 0, u = 10 N m on joint j alone   -> ~10 * column j of
                 Minv + bias; qd = 0 keeps viscous damping OUT of the probe,
                 so a per-joint response ratio directly exposes effective-
                 inertia mismatch (armature, rotor, missing link mass)
  - "coriolis":  bent posture, qd != 0, u = 0          -> velocity terms
                 (includes joint damping IF a consumer models it; disagreement
                 HERE with agreement on inertia_j probes points at damping/
                 friction modeling, not inertia)

Semantics pinned in the metadata: FIXED base, gravity -9.81 z-down (world),
zero external wrench, NO joint damping/friction (the GRiD default — the
USE_JOINT_DYNAMICS/runtime toggle path is deliberately outside schema 1), URDF
identified by sha256. The iiwa14 probe states are copied VERBATIM from GATO's
table (f32-rounded decimals) so the two tables compare state-for-state; other
robots use rest = zeros and a fixed deterministic bent/qd pattern (no RNG
anywhere). Regenerate whenever a robot_assets URDF or the reference dynamics
semantics change — test/test_dynamics_fingerprint.py enforces staleness.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from RBDReference.tests import MANIFEST_PATH  # noqa: E402
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec  # noqa: E402
from RBDReference.equivalents.reference_backend import build_project_adapter  # noqa: E402

TORQUE = 10.0
OUT_PATH = os.path.join(REPO, "test", "dynamics_fingerprint.json")

# iiwa14 states copied verbatim from GATO's table (f32-rounded decimals) so the
# upstream and GATO tables are comparable state-for-state. GATO's "home" rest
# for iiwa14 is all-zeros; the bent posture is the PDDP t2 operating posture.
_IIWA14_BENT_Q = [-0.20280000567436218, 0.5335999727249146, -0.010999999940395355,
                  -1.628600001335144, 1.7575000524520874, -1.9313000440597534,
                  0.41530001163482666]
_IIWA14_BENT_QD = [0.5, -0.550000011920929, 0.6000000238418579, -0.6499999761581421,
                   0.699999988079071, -0.75, 0.800000011920929]


def _bent_q(nq):
    # fixed deterministic elbow-ish pattern; a fingerprint needs a REPRODUCIBLE
    # energetic state, not a feasible operating posture (limits irrelevant here)
    return np.array([0.4 * ((-1) ** j) * (1 + 0.1 * j) for j in range(nq)], dtype=np.float64)


def _bent_qd(nv):
    # GATO's qd pattern, kept identical
    return np.array([0.5 * ((-1) ** j) * (1 + 0.1 * j) for j in range(nv)], dtype=np.float64)


def probes_for(robot_id, nq, nv):
    rest = np.zeros(nq, dtype=np.float64)
    if robot_id == "iiwa14":
        bent_q = np.asarray(_IIWA14_BENT_Q, dtype=np.float64)
        bent_qd = np.asarray(_IIWA14_BENT_QD, dtype=np.float64)
    else:
        bent_q, bent_qd = _bent_q(nq), _bent_qd(nv)
    out = [("gravity", rest, np.zeros(nv), np.zeros(nv))]
    for j in range(nv):
        u = np.zeros(nv)
        u[j] = TORQUE
        out.append((f"inertia_{j}", rest, np.zeros(nv), u))
    out.append(("coriolis", bent_q, bent_qd, np.zeros(nv)))
    return out


def generate_table():
    table = {
        "schema": 1,
        "semantics": {
            "quantity": "forward-dynamics qdd(q, qd, u), fixed base, f_ext = 0",
            "gravity": -9.81,
            "joint_dynamics": "none (no viscous damping / dry friction — the GRiD default path)",
            "source": "GRiD RBDReference numpy oracle (pinocchio-validated), float64",
            "guidance": "per-joint |qdd_yours/qdd_ref| on the inertia_j probes: "
                        "1.00+-0.05 = aligned; >1.5 on any joint = effective-inertia "
                        "mismatch (check armature/rotor/link inertia). inertia_j "
                        "probes are qd=0 hence damping-immune; coriolis-only "
                        "disagreement points at damping/friction modeling.",
        },
        "plants": {},
    }
    failures = {}
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        rid = spec.robot_id
        try:
            resolved = resolve_robot_spec(spec)
            pm = build_project_adapter(spec, resolved, base_mode="fixed")
        except (RuntimeError, Exception) as exc:  # noqa: BLE001 — record + report, table stays honest
            failures[rid] = f"{type(exc).__name__}: {exc}"
            continue
        ref, nq, nv = pm.reference, pm.nq, pm.nv
        urdf = str(resolved.urdf_path)
        entries = []
        bad = None
        for name, q, qd, u in probes_for(rid, nq, nv):
            qdd = np.asarray(ref.forward_dynamics(q, qd, u), dtype=np.float64).ravel()
            if not np.all(np.isfinite(qdd)):
                bad = f"non-finite qdd on probe '{name}' (broken asset? see rizon4 backlog)"
                break
            entries.append({
                "name": name,
                "q": np.asarray(q, dtype=np.float64).tolist(),
                "qd": np.asarray(qd, dtype=np.float64).tolist(),
                "u": np.asarray(u, dtype=np.float64).tolist(),
                "qdd": qdd.tolist(),
            })
        if bad is not None:
            failures[rid] = bad
            continue
        table["plants"][rid] = {
            "nq": int(nq),
            "nv": int(nv),
            "urdf": os.path.relpath(urdf, REPO),
            "urdf_sha256": hashlib.sha256(open(urdf, "rb").read()).hexdigest(),
            "probes": entries,
        }
    return table, failures


def main():
    table, failures = generate_table()
    for rid, plant in table["plants"].items():
        g = plant["probes"][0]["qdd"]
        print(f"[{rid}] {len(plant['probes'])} probes (gravity qdd[0..2] = {g[:3]})")
    for rid, why in failures.items():
        print(f"[{rid}] EXCLUDED: {why}")
    with open(OUT_PATH, "w") as f:
        json.dump(table, f, indent=1)
        f.write("\n")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
