"""W3 Increment 1 — spherizer coverage gate (pure Python, no GPU).

The spherizer's contract is CONSERVATIVE COVERAGE: the union of emitted spheres must fully
contain the source collision geometry (no missed collisions). These tests sample each supported
primitive's surface (and a voxel-filled mesh) and assert every sample lies inside at least one
covering sphere, then check the URDF round-trip preserves the joint tree + emits the foam
`<collision><sphere/></collision>` interchange format that build_sphere_tiers consumes.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

from grid_codegen.algorithms._spherize import (
    _cover_sphere, _cover_cylinder, _cover_box, _cover_mesh, spherize_urdf,
)
from grid_codegen.algorithms._collision import parse_spherized_urdf, urdf_joint_tree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from grid_assets import robot_urdf


def _max_uncovered_gap(points, spheres):
    """max over points of (dist-to-nearest-sphere-center - that sphere's radius). <=0 => covered."""
    centers = np.array([[s[0], s[1], s[2]] for s in spheres])
    radii = np.array([s[3] for s in spheres])
    worst = -1e9
    for p in points:
        d = np.linalg.norm(centers - np.asarray(p), axis=1) - radii
        worst = max(worst, float(d.min()))
    return worst


def _cylinder_surface_samples(radius, length, n_ax=25, n_th=24):
    pts = []
    for z in np.linspace(-length / 2, length / 2, n_ax):
        for th in np.linspace(0, 2 * math.pi, n_th, endpoint=False):
            pts.append((radius * math.cos(th), radius * math.sin(th), z))
    for z in (-length / 2, length / 2):  # caps
        for rr in np.linspace(0, radius, 6):
            for th in np.linspace(0, 2 * math.pi, n_th, endpoint=False):
                pts.append((rr * math.cos(th), rr * math.sin(th), z))
    return pts


def _box_surface_samples(size, n=9):
    sx, sy, sz = size
    pts = []
    us = np.linspace(-0.5, 0.5, n)
    for a in us:
        for b in us:
            pts += [(a * sx, b * sy, sz / 2), (a * sx, b * sy, -sz / 2),
                    (a * sx, sy / 2, b * sz), (a * sx, -sy / 2, b * sz),
                    (sx / 2, a * sy, b * sz), (-sx / 2, a * sy, b * sz)]
    return pts


def test_cover_sphere_is_the_sphere():
    assert _cover_sphere(0.07) == [(0.0, 0.0, 0.0, 0.07)]


@pytest.mark.parametrize("radius,length,spacing", [(0.1, 0.29, 0.06), (0.05, 0.5, 0.05),
                                                   (0.12, 0.05, 0.06), (0.02, 1.0, 0.1)])
def test_cover_cylinder_covers_surface(radius, length, spacing):
    spheres = _cover_cylinder(radius, length, spacing)
    gap = _max_uncovered_gap(_cylinder_surface_samples(radius, length), spheres)
    assert gap <= 1e-6, f"cylinder surface not covered (worst gap {gap:.2e})"


@pytest.mark.parametrize("size,spacing", [((0.2, 0.1, 0.3), 0.05), ((0.15, 0.15, 0.15), 0.06),
                                          ((0.4, 0.05, 0.05), 0.05)])
def test_cover_box_covers_surface(size, spacing):
    spheres = _cover_box(size, spacing)
    gap = _max_uncovered_gap(_box_surface_samples(size), spheres)
    assert gap <= 1e-6, f"box surface not covered (worst gap {gap:.2e})"


def test_cover_mesh_covers_surface():
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=[0.2, 0.12, 0.3])
    with tempfile.TemporaryDirectory() as d:
        obj = os.path.join(d, "boxmesh.obj")
        mesh.export(obj)
        # minimal <mesh> element pointing at the exported OBJ (absolute path -> resolves directly)
        import xml.etree.ElementTree as ET
        elem = ET.fromstring(f'<mesh filename="{obj}"/>')
        spheres = _cover_mesh(elem, d, spacing=0.05, link_name="boxmesh")
    assert len(spheres) > 0
    samples = [tuple(p) for p in mesh.sample(2000)]
    gap = _max_uncovered_gap(samples, spheres)
    assert gap <= 1e-6, f"mesh surface not covered (worst gap {gap:.2e})"


def test_spherize_urdf_roundtrip_go2():
    """Spherize a real all-primitive robot and confirm the foam interchange round-trips: every
    link with source collision geometry emits sphere collisions, joints are preserved, radii>0."""
    urdf = str(robot_urdf("go2"))
    if not os.path.exists(urdf):
        pytest.skip("go2.urdf not found")
    with tempfile.TemporaryDirectory() as d:
        out = spherize_urdf(urdf, resolution=0.05, out_path=os.path.join(d, "go2_spherized.urdf"))
        parsed = parse_spherized_urdf(out)
        joints_out = urdf_joint_tree(out)
    assert parsed, "no links with collision spheres emitted"
    assert all(r > 0 for spheres in parsed.values() for (_, _, _, r) in spheres)
    # joint tree preserved (go2 is a quadruped with many joints)
    joints_in = urdf_joint_tree(urdf)
    assert set(joints_out) == set(joints_in), "spherize must preserve the URDF joint tree"


def test_strict_spherize_rejects_an_empty_model(tmp_path):
    """A URDF with no collision geometry is an error by default: an empty sphere model
    would bake zero clearance rows without a word. The opt-out keeps the warning."""
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="empty"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="0 collision spheres"):
        spherize_urdf(str(urdf), 0.05, out_path=str(tmp_path / "strict.urdf"))
    with pytest.warns(UserWarning, match="0 collision spheres"):
        out = spherize_urdf(str(urdf), 0.05, out_path=str(tmp_path / "lenient.urdf"), strict=False)
    assert parse_spherized_urdf(out) == {}


def test_strict_spherize_rejects_a_voxel_fill_fallback(tmp_path, monkeypatch):
    """A failed voxel fill is an error by default; strict=False warns and keeps the
    bounding-box cover, which still covers the mesh surface."""
    trimesh = pytest.importorskip("trimesh")
    import xml.etree.ElementTree as ET
    mesh = trimesh.creation.box(extents=[0.2, 0.12, 0.3])
    obj = tmp_path / "boxmesh.obj"
    mesh.export(str(obj))
    elem = ET.fromstring(f'<mesh filename="{obj}"/>')

    def broken(self, pitch):
        raise RuntimeError("probe: voxel fill unavailable")
    monkeypatch.setattr(trimesh.Trimesh, "voxelized", broken)
    with pytest.raises(ValueError, match="voxelization failed .*probe"):
        _cover_mesh(elem, str(tmp_path), spacing=0.05, link_name="boxmesh")
    with pytest.warns(UserWarning, match="bounding-box cover"):
        spheres = _cover_mesh(elem, str(tmp_path), spacing=0.05, link_name="boxmesh", strict=False)
    assert spheres
    gap = _max_uncovered_gap([tuple(p) for p in mesh.sample(2000)], spheres)
    assert gap <= 1e-6, f"bounding-box cover left the mesh surface uncovered (worst gap {gap:.2e})"
