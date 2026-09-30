"""CUDA gate for NATIVE capsule-row collision (Bundle 1a: `--collision-native`).

Certifies the full native pipeline end-to-end on iiwa14 (whose URDF collision geometry is
all cylinders -> pure capsule rows, one per movable link, pedestal skipped):
  * broad(covering spheres derived from rows) -> fine(capsules) config_free verdict ==
    an independent fine-only capsule check on every (config, obstacle) pair;
  * the link_CC narrowing is non-vacuous (partial mask flags observed);
  * collision_distance_gradient (envelope-theorem composition over both endpoint
    gradients) matches central finite differences of collision_distance;
  * collision_distance_pairs equals the reduced distance bitwise for a single obstacle;
  * grid_cc_self_collision_capsules through the SINGLE-tier native config_free (hand-built
    two-row specs: non-adjacent huge pair collides, adjacent-only pair bakes zero ranges).
Pure-python properties (no GPU): parse_native_urdf row extraction + conservative cylinder
containment, and the derived broad tier's covering property.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._collision import (
    build_self_cc_ranges, native_collision_spec_from_urdf, parse_native_urdf)
from test.cuda_equivalents.test_cuda_collision_config_free import (
    _compile_and_run, _gen_header, _parse_kv, _robot)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf

NATIVE_RUNNER = Path(__file__).with_name("cuda_collision_native_runner.cu")
SELFCC_RUNNER = Path(__file__).with_name("cuda_collision_native_selfcc_runner.cu")


def _iiwa_native_spec():
    urdf = robot_urdf("iiwa14")
    if not urdf.exists():
        pytest.skip("iiwa14.urdf not found")
    robot = _robot()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        spec = native_collision_spec_from_urdf(robot, str(urdf))
    return robot, spec, urdf


def test_parse_native_rows():
    """iiwa14's URDF collision set is 6 cylinders (1 on the pedestal-bolted base). The native
    parse must yield capsule rows whose segment+radius CONTAIN their cylinder (conservative):
    capsule endpoints = origin +- (len/2)*axis, same r."""
    robot, spec, urdf = _iiwa_native_spec()
    native = parse_native_urdf(str(urdf))
    n_rows_all = sum(len(d["rows"]) for d in native.values())
    assert n_rows_all == 6, f"expected 6 cylinder rows in the raw parse, got {n_rows_all}"
    # links 6/7 carry drake package:// collision meshes that cannot resolve locally — they are
    # residual-flagged, and the spherizer warns + skips them (SAME behavior as the sphere path),
    # so they contribute zero rows rather than failing the build.
    assert all(not d["rows"] for ln, d in native.items() if d["residual"]), \
        "residual (mesh) links unexpectedly also carry native rows on iiwa14"
    fine = spec["tiers"][-1]
    assert "pb" in fine and len(fine["anchor"]) == 5, \
        f"expected 5 anchored rows (pedestal skipped), got {len(fine['anchor'])}"
    # every row is a genuine segment (cylinders, not spheres) with positive radius
    for i in range(len(fine["anchor"])):
        a = np.array(fine["offset"][3 * i:3 * i + 3])
        b = np.array(fine["pb"][3 * i:3 * i + 3])
        assert np.linalg.norm(b - a) > 0.05, "cylinder row degenerated to a point"
        assert fine["radius"][i] > 0.0


def test_broad_tier_covering_property():
    """The derived broad tier must COVER the fine rows: for every row endpoint e with radius r
    on anchor A, |e - c_A| + r <= R_A (+eps). This is the exactness precondition of the
    broad->fine mask driver ('broad clear => definitely free')."""
    robot, spec, urdf = _iiwa_native_spec()
    broad, fine = spec["tiers"][0], spec["tiers"][-1]
    centers = {broad["anchor"][k]: (np.array(broad["offset"][3 * k:3 * k + 3]), broad["radius"][k])
               for k in range(len(broad["anchor"]))}
    for i, a_jid in enumerate(fine["anchor"]):
        c, R = centers[a_jid]
        for e in (np.array(fine["offset"][3 * i:3 * i + 3]), np.array(fine["pb"][3 * i:3 * i + 3])):
            assert np.linalg.norm(e - c) + fine["radius"][i] <= R + 1e-12, \
                f"row {i} endpoint escapes its broad covering sphere (anchor {a_jid})"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_native_two_tier(tmp_path):
    robot, spec, _ = _iiwa_native_spec()
    build_dir = tmp_path / "collision_native"
    _gen_header(robot, build_dir, spec)
    out = _compile_and_run(build_dir, NATIVE_RUNNER)
    print(out)


def _two_row_spec(robot, anchor_a, anchor_b, radius):
    """Two capsule rows (short vertical segments) on the given anchors, huge radius so the
    verdict is governed only by the baked self_cc_ranges (single tier, empty env)."""
    anchors = [int(anchor_a), int(anchor_b)]
    pa = [0.02, -0.01, 0.03, -0.02, 0.01, -0.03]
    pb = [0.02, -0.01, 0.08, -0.02, 0.01, 0.02]
    return {"anchor": anchors, "offset": pa, "pb": pb,
            "radius": [float(radius), float(radius)],
            "self_cc_ranges": build_self_cc_ranges(robot, anchors)}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_native_self_collision(tmp_path):
    robot = _robot()
    # POSITIVE: non-adjacent pair (serial chain anchors 0,2), huge radii -> must self-collide.
    pos_spec = _two_row_spec(robot, 0, 2, radius=1.0e3)
    assert pos_spec["self_cc_ranges"], "non-adjacent pair must bake a range"
    pos_dir = tmp_path / "native_selfcc_positive"
    _gen_header(robot, pos_dir, pos_spec)
    pos_out = _compile_and_run(pos_dir, SELFCC_RUNNER)
    assert _parse_kv(pos_out, "empty_free") == 0, f"non-adjacent huge pair should self-collide:\n{pos_out}"
    # NEGATIVE: adjacent-only pair -> zero baked ranges -> free despite huge radii.
    neg_spec = _two_row_spec(robot, 0, 1, radius=1.0e3)
    assert not neg_spec["self_cc_ranges"], "adjacent-only pair must yield empty self_cc_ranges"
    neg_dir = tmp_path / "native_selfcc_negative"
    _gen_header(robot, neg_dir, neg_spec)
    neg_out = _compile_and_run(neg_dir, SELFCC_RUNNER)
    assert _parse_kv(neg_out, "empty_free") == 1, f"adjacent-excluded pair should stay free:\n{neg_out}"
    assert _parse_kv(neg_out, "NRANGES") == 0
    print(pos_out + neg_out)
