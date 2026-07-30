"""CUDA equivalence gate for the W3 Component E static geometry header
`collision/grid_collision_geometry.cuh` (grid_collision:: SDF primitives).

Pure geometry — NO grid.cuh, no robot model, no codegen. The runner bakes self-describing
collision configs (covering collision / free / edge cases: capsule t-clamps, cuboid
face/edge/corner/inside, rotated OBB), evaluates each SDF on device, and prints the full
geometry + GPU squared-gap. This gate:

  1. SDF ORACLE: the printed GPU squared-gap matches an independent NumPy re-derivation of
     each primitive (rtol 1e-12) for every config.
  2. SIGN / classification: collision (<0) vs free (>0) agrees with the oracle, incl. the
     fp32 lane (runner asserts fp64/fp32 signs agree -> FP32SIGN mismatches=0).
  3. HOST==DEVICE: the __host__ __device__ primitives are fp64 bit-exact (HOSTDEV maxdiff=0).
  4. COMPOSITION: environment reduction + baked-range self-collision behave (runner COMPO line).

SDF convention: squared_gap = d2 - r_sum^2 ; value < 0 <=> in collision.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_geometry_runner.cu")


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_collision_geometry_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O2",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-I", str(COLLISION_INCLUDE), "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "cuda_collision_geometry_runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


# ------------------------------------------------------------------ NumPy SDF oracles
def _ss_oracle(ax, ay, az, ar, bx, by, bz, br):
    d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
    return d2 - (ar + br) ** 2


def _sc_oracle(ax, ay, az, bx, by, bz, cr, px, py, pz, pr):
    a = np.array([ax, ay, az]); b = np.array([bx, by, bz]); p = np.array([px, py, pz])
    ab = b - a
    denom = float(ab @ ab)
    t = float((p - a) @ ab) / denom if denom > 0.0 else 0.0
    t = min(1.0, max(0.0, t))
    q = a + t * ab
    return float((p - q) @ (p - q)) - (pr + cr) ** 2


def _cb_oracle(cx, cy, cz, ux, uy, uz, hu, vx, vy, vz, hv, wx, wy, wz, hw, px, py, pz, pr):
    d = np.array([px - cx, py - cy, pz - cz])
    e = 0.0
    for (ax, ay, az, half) in ((ux, uy, uz, hu), (vx, vy, vz, hv), (wx, wy, wz, hw)):
        proj = abs(d @ np.array([ax, ay, az]))
        excess = max(proj - half, 0.0)
        e += excess * excess
    return e - pr ** 2


def _cc_oracle(a1x, a1y, a1z, b1x, b1y, b1z, r1, a2x, a2y, a2z, b2x, b2y, b2z, r2):
    """Exact min of |p1(s) - p2(t)|^2 over the unit square (convex QP): the minimum is
    at the unconstrained stationary point if it lies inside, else on an edge/corner —
    enumerate the 1-D clamped minimizers of all four edges plus the interior candidate.
    Independent of the header's Ericson-style branch structure."""
    a1 = np.array([a1x, a1y, a1z]); d1 = np.array([b1x, b1y, b1z]) - a1
    a2 = np.array([a2x, a2y, a2z]); d2 = np.array([b2x, b2y, b2z]) - a2
    r = a1 - a2

    def dist2(s, t):
        diff = r + s * d1 - t * d2
        return float(diff @ diff)

    def t_star(s):  # best t for fixed s
        e = float(d2 @ d2)
        return min(1.0, max(0.0, float(d2 @ (r + s * d1)) / e)) if e > 0 else 0.0

    def s_star(t):  # best s for fixed t
        a = float(d1 @ d1)
        return min(1.0, max(0.0, float(d1 @ (t * d2 - r)) / a)) if a > 0 else 0.0

    candidates = [(0.0, t_star(0.0)), (1.0, t_star(1.0)),
                  (s_star(0.0), 0.0), (s_star(1.0), 1.0)]
    A = np.array([[float(d1 @ d1), -float(d1 @ d2)], [-float(d1 @ d2), float(d2 @ d2)]])
    b = np.array([-float(d1 @ r), float(d2 @ r)])
    if np.linalg.det(A) > 0:
        st = np.linalg.solve(A, b)
        if 0.0 <= st[0] <= 1.0 and 0.0 <= st[1] <= 1.0:
            candidates.append((float(st[0]), float(st[1])))
    d2min = min(dist2(s, t) for (s, t) in candidates)
    return d2min - (r1 + r2) ** 2


def _cp_oracle(nx, ny, nz, d, ax, ay, az, bx, by, bz, r):
    s = min(nx * ax + ny * ay + nz * az - d, nx * bx + ny * by + nz * bz - d)
    e = max(s, 0.0)
    return e * e - r * r


def _cx_oracle(cx, cy, cz, ux, uy, uz, hu, vx, vy, vz, hv, wx, wy, wz, hw,
               ax, ay, az, bx, by, bz, cr):
    """Exact segment-vs-OBB core distance: in the box frame D2(t) is convex
    piecewise-quadratic; enumerate breakpoints where a coordinate crosses +-h and the
    clamped stationary point of each interval's active-set quadratic (independent
    numpy re-derivation of the same exact mathematical structure)."""
    R = np.array([[ux, uy, uz], [vx, vy, vz], [wx, wy, wz]])
    h = np.array([hu, hv, hw])
    pa = R @ (np.array([ax, ay, az]) - np.array([cx, cy, cz]))
    pb = R @ (np.array([bx, by, bz]) - np.array([cx, cy, cz]))
    dvec = pb - pa

    def D2(t):
        p = pa + t * dvec
        ex = np.maximum(np.abs(p) - h, 0.0)
        return float(ex @ ex)

    ts = {0.0, 1.0}
    for k in range(3):
        if dvec[k] != 0.0:
            for target in (h[k], -h[k]):
                t = (target - pa[k]) / dvec[k]
                if 0.0 < t < 1.0:
                    ts.add(float(t))
    ts = sorted(ts)
    best = min(D2(t) for t in ts)
    for lo, hi in zip(ts[:-1], ts[1:]):
        tm = 0.5 * (lo + hi)
        best = min(best, D2(tm))
        p = pa + tm * dvec
        active = np.abs(p) > h
        if active.any():
            sg = np.sign(p[active])
            c = sg * pa[active] - h[active]
            e = sg * dvec[active]
            see = float(e @ e)
            if see > 0.0:
                tstar = min(hi, max(lo, -float(c @ e) / see))
                best = min(best, D2(tstar))
    return best - cr * cr


_ORACLE = {
    "SS": (8, _ss_oracle),
    "SC": (11, _sc_oracle),
    "CB": (19, _cb_oracle),
    "CC": (14, _cc_oracle),
    "CP": (11, _cp_oracle),
    "CX": (22, _cx_oracle),
}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_geometry(tmp_path):
    build_dir = tmp_path / "collision_geometry"
    build_dir.mkdir()
    executable = _compile_runner(build_dir)

    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, (
        "collision_geometry runner FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    lines = result.stdout.splitlines()

    hostdev = next(l for l in lines if l.startswith("HOSTDEV"))
    assert float(hostdev.split("=")[1]) < 1e-12, f"host!=device beyond FMA noise: {hostdev}"
    fp32 = next(l for l in lines if l.startswith("FP32SIGN"))
    assert int(fp32.split("=")[1]) == 0, f"fp32/fp64 sign disagreement: {fp32}"
    compo = next(l for l in lines if l.startswith("COMPO"))
    assert compo == "COMPO env_hit=1 env_miss=0 self_hit=1 self_free=0", f"composition wrong: {compo}"

    n_checked = {"SS": 0, "SC": 0, "CB": 0, "CC": 0, "CP": 0, "CX": 0}
    for line in lines:
        tag = line[:2]
        if tag not in _ORACLE:
            continue
        arity, oracle = _ORACLE[tag]
        parts = line.split()
        assert parts[arity + 1] == "GAP", f"malformed {tag} line: {line}"
        cfg = [float(v) for v in parts[1:arity + 1]]
        gpu_gap = float(parts[arity + 2])
        expected = oracle(*cfg)
        np.testing.assert_allclose(
            gpu_gap, expected, rtol=1e-12, atol=1e-14,
            err_msg=f"{tag} SDF != NumPy oracle for config {cfg}")
        # collision/free classification must match away from the exact-touching boundary
        # (knife-edge gap~0 configs are covered by the numeric allclose above, not by sign)
        if abs(expected) > 1e-9:
            assert (gpu_gap < 0) == (expected < 0), f"{tag} collision-sign mismatch: {line}"
        n_checked[tag] += 1

    assert n_checked["SS"] >= 5 and n_checked["SC"] >= 6 and n_checked["CB"] >= 7, n_checked
    assert n_checked["CC"] >= 7 and n_checked["CP"] >= 4 and n_checked["CX"] >= 7, n_checked
    assert result.stdout.strip().endswith("RESULT: PASS"), result.stdout
    print(f"collision_geometry: {n_checked} SDF configs vs NumPy oracle OK")
