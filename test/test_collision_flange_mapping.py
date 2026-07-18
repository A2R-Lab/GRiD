"""Unit gate for the W3 Component D FLANGE mapping (`GRiDCodeGenerator/algorithms/_collision.py`).

The FLANGE mapping is the #1 silent-wrong-frame hazard in the collision pipeline: every foam
sphere must bind to ITS OWN GRiD `s_Xworld` frame slot. This test certifies that mapping WITHOUT
the (heavy, optional) foam toolchain, on:

  1. REAL iiwa14 (`config/robot_assets/iiwa14.urdf`): movable links `iiwa_link_1..7` -> joint ids 0..6
     (base-0 MONOTONE down the chain, T=I); the WELDED `iiwa_link_ee` folds onto its movable
     parent (`iiwa_joint_7` = jid 6) carrying the fixed transform.
  2. A synthetic 3-joint + fixed-flange robot: full `build_sphere_tiers` PRE-COMPOSES a welded
     sphere's offset through the fixed transform (offset != raw local), and `build_self_cc_ranges`
     emits exactly the non-adjacent (skip same/parent/child) pairs, brute-force cross-checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from GRiDCodeGenerator.algorithms._collision import (
    build_self_cc_ranges,
    build_sphere_tiers,
    parse_spherized_urdf,
    sphere_anchor_frames,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from config import robot_urdf
IIWA = robot_urdf("iiwa14")


def _parse(path):
    return URDFParser().parse(str(path))


# --------------------------------------------------------------------------- real iiwa14
def test_flange_mapping_iiwa_movable_monotone():
    robot = _parse(IIWA)
    links = [f"iiwa_link_{k}" for k in range(1, 8)]
    frames = sphere_anchor_frames(robot, links, str(IIWA))
    jids = [frames[l][0] for l in links]
    # iiwa_link_k anchors to the joint whose child it is: jid == k-1, strictly monotone.
    assert jids == list(range(7)), jids
    for l in links:
        assert np.allclose(frames[l][1], np.eye(4)), f"{l} movable anchor must have identity compose"


def test_flange_mapping_iiwa_welded_ee_folds_to_movable_parent():
    robot = _parse(IIWA)
    # Two welded frames off iiwa_link_7 (movable jid 6): iiwa_link_ee via tool0_joint,
    # iiwa_link_ee_kuka via iiwa_joint_ee. Each must anchor to jid 6 carrying ITS OWN
    # (name-bridged) fixed transform -- proving the foam-link -> URDF-joint -> GRiD-fixed-joint
    # bridge picks the correct joint even when several fixed frames share a movable parent.
    welded = {"iiwa_link_ee": "tool0_joint", "iiwa_link_ee_kuka": "iiwa_joint_ee"}
    frames = sphere_anchor_frames(robot, list(welded), str(IIWA))
    for link, jname in welded.items():
        anchor, T = frames[link]
        assert anchor == 6, (link, anchor)
        fj = robot.get_fixed_joint_by_name(jname)
        expected = np.asarray(fj.get_transformation_matrix_hom(), dtype=np.float64).reshape(4, 4)
        assert np.allclose(T, expected), f"{link}: welded compose must be {jname}'s fixed transform"
        assert not np.allclose(T, np.eye(4)), f"{link}: welded frame has a non-trivial offset"


def test_flange_mapping_root_link_is_skip_sentinel():
    robot = _parse(IIWA)
    frames = sphere_anchor_frames(robot, ["base"], str(IIWA))
    assert frames["base"][0] == -1, "root/world link must map to the -1 skip sentinel"


# --------------------------------------------------------------------------- synthetic robot
_MINI_URDF = """<?xml version="1.0"?>
<robot name="mini">
  <link name="base"/>
  <link name="l1"><inertial><mass value="1"/><origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><origin xyz="0 0 0.1"/><geometry><sphere radius="0.05"/></geometry></collision></link>
  <link name="l2"><inertial><mass value="1"/><origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="l3"><inertial><mass value="1"/><origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><origin xyz="0 0 0.12"/><geometry><sphere radius="0.05"/></geometry></collision></link>
  <link name="flange">
    <collision><origin xyz="0.05 0 0"/><geometry><sphere radius="0.03"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="jf" type="fixed"><parent link="l3"/><child link="flange"/>
    <origin xyz="0.1 0 0.2"/></joint>
</robot>
"""


@pytest.fixture
def mini(tmp_path):
    p = tmp_path / "mini_spherized.urdf"
    p.write_text(_MINI_URDF)
    return p


def test_parse_spherized_urdf(mini):
    parsed = parse_spherized_urdf(str(mini))
    assert parsed == {
        "l1": [(0.0, 0.0, 0.1, 0.05)],
        "l3": [(0.0, 0.0, 0.12, 0.05)],
        "flange": [(0.05, 0.0, 0.0, 0.03)],
    }


def test_build_sphere_tiers_composes_welded_offset(mini):
    robot = _parse(mini)
    tiers = build_sphere_tiers(robot, {"fine": str(mini)})
    d = tiers["fine"]
    # 3 spheres: l1(anchor 0), l3(anchor 2), flange(welded onto l3's joint j3 = anchor 2)
    assert d["n"] == 3
    assert d["anchor"] == [0, 2, 2], d["anchor"]
    off = np.array(d["offset"]).reshape(3, 3)
    np.testing.assert_allclose(off[0], [0.0, 0.0, 0.1])     # movable l1, raw
    np.testing.assert_allclose(off[1], [0.0, 0.0, 0.12])    # movable l3, raw
    # welded flange: local (0.05,0,0) pre-composed through jf origin (0.1,0,0.2) -> (0.15,0,0.2)
    np.testing.assert_allclose(off[2], [0.15, 0.0, 0.2], atol=1e-12)
    assert not np.allclose(off[2], [0.05, 0.0, 0.0]), "welded offset must be composed, not raw"
    assert d["radius"] == [0.05, 0.05, 0.03]


# ------------------------------------------------------- multi-hop welded chain (fixed->fixed->movable)
_MINI2_URDF = """<?xml version="1.0"?>
<robot name="mini2">
  <link name="base"/>
  <link name="l1"><inertial><mass value="1"/><origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="l2"><inertial><mass value="1"/><origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="mid"/>
  <link name="tip">
    <collision><origin xyz="0.02 0 0"/><geometry><sphere radius="0.03"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="jfA" type="fixed"><parent link="l2"/><child link="mid"/>
    <origin xyz="0.1 0 0.2" rpy="0 0 1.5707963267948966"/></joint>
  <joint name="jfB" type="fixed"><parent link="mid"/><child link="tip"/>
    <origin xyz="0.05 0 0"/></joint>
</robot>
"""


def _T_from_origin(xyz, rpy):
    """URDF origin -> 4x4 homogeneous (R = Rz(yaw) Ry(pitch) Rx(roll)). Independent oracle."""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = xyz
    return T


def test_multihop_welded_chain_composes_full_transform(tmp_path):
    """A sphere two fixed joints deep (tip <-fixed- mid <-fixed- l2 <-movable- j2) must anchor
    to the MOVABLE parent (j2 = jid 1) with its offset composed through BOTH fixed joints. GRiD
    pre-collapses fixed chains onto the nearest movable joint, so the one-hop lookup suffices --
    this locks that in against an independent two-link-chain oracle."""
    p = tmp_path / "mini2_spherized.urdf"
    p.write_text(_MINI2_URDF)
    robot = _parse(p)
    tiers = build_sphere_tiers(robot, {"fine": str(p)})
    d = tiers["fine"]
    assert d["n"] == 1 and d["anchor"] == [1], d          # NOT skipped, anchored to movable j2
    # independent oracle: compose the two fixed origins, apply to the sphere's local center
    T = _T_from_origin([0.1, 0.0, 0.2], [0, 0, np.pi / 2]) @ _T_from_origin([0.05, 0.0, 0.0], [0, 0, 0])
    expected = (T @ np.array([0.02, 0.0, 0.0, 1.0]))[:3]
    np.testing.assert_allclose(np.array(d["offset"]), expected, atol=1e-12)


def test_build_self_cc_ranges_matches_bruteforce(mini):
    robot = _parse(mini)
    # spheres on frames [l1, l1, l3, l3, l2] -> anchors [0,0,2,2,1]
    anchor = [0, 0, 2, 2, 1]
    ranges = build_self_cc_ranges(robot, anchor)

    # brute-force oracle: parent map {0:-1, 1:0, 2:1}; skip same/parent/child anchors
    parent = {0: -1, 1: 0, 2: 1}
    def adj(a, b):
        return a == b or parent[a] == b or parent[b] == a
    expected_pairs = {(i, j) for i in range(len(anchor)) for j in range(i + 1, len(anchor))
                      if not adj(anchor[i], anchor[j])}
    got_pairs = {(i, j) for (i, j0, j1) in ranges for j in range(j0, j1 + 1)}
    assert got_pairs == expected_pairs, (got_pairs, expected_pairs)
    # ranges must be well-formed (i < j0 <= j1)
    for (i, j0, j1) in ranges:
        assert i < j0 <= j1 < len(anchor)
