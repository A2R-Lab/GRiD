"""W3 Increment 1 — automated URDF -> spherized-URDF (foam-compatible interchange format).

Turns a URDF's per-link `<collision>` geometry into a set of COVERING spheres and rewrites the
URDF with one `<collision><geometry><sphere/></geometry><origin xyz/></collision>` per sphere
(center in the LINK frame). This is the exact format `parse_spherized_urdf`/`build_sphere_tiers`
(the foam interchange in _collision.py) already consume, so a spherized URDF from here is drop-in
interchangeable with a foam-produced one.

Design choices (why custom, not foam-only):
  * PRIMITIVES (sphere/cylinder/box) are covered ANALYTICALLY -- exact, deterministic, and needing
    NO external mesh assets. Most GRiD collision geometry is primitive (iiwa14 = cylinders, go2 =
    boxes/cylinders/spheres), so the common case has zero heavy dependencies.
  * MESHES are voxel-filled via trimesh (interior fill -> one sphere per occupied voxel). If a mesh
    file cannot be resolved/loaded, that collision is SKIPPED with a loud warning (never silently
    dropped) rather than aborting the whole robot -- partial (primitive) coverage still generates a
    valid grid_collision.

`resolution` is the target sphere SPACING in meters: coarser -> fewer/larger spheres (broad tier),
finer -> more/smaller spheres (fine tier). Covering radii are chosen so the union of spheres fully
contains the source geometry (conservative: no missed collisions).
"""
import math
import os
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------- URDF origin math
def _rpy_to_matrix(rpy):
    """URDF rpy (roll, pitch, yaw) -> 3x3 rotation, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_of(elem):
    """(R 3x3, t 3) from an element's optional `<origin xyz rpy>` (identity if absent)."""
    origin = elem.find("origin") if elem is not None else None
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        if origin.get("xyz"):
            xyz = [float(v) for v in origin.get("xyz").split()]
        if origin.get("rpy"):
            rpy = [float(v) for v in origin.get("rpy").split()]
    return _rpy_to_matrix(rpy), np.asarray(xyz, dtype=float)


# --------------------------------------------------------------------------- analytic primitive covers
# Each returns a list of (x, y, z, radius) spheres in the geometry's OWN local frame (before the
# collision <origin> is applied). Covering condition: every surface point of the primitive lies
# within at least one returned sphere.
def _cover_sphere(radius):
    return [(0.0, 0.0, 0.0, float(radius))]


def _cover_cylinder(radius, length, spacing):
    """Cylinder (URDF: centered at origin, axis = +z) -> a line of spheres along z. Sphere radius
    sqrt(r^2 + half_gap^2) so the surface midway between two centers is exactly covered."""
    n = max(1, int(math.ceil(length / spacing)) + 1)
    if n == 1:
        zs = [0.0]
        half_gap = length / 2.0
    else:
        zs = list(np.linspace(-length / 2.0, length / 2.0, n))
        half_gap = (length / (n - 1)) / 2.0
    sr = math.sqrt(radius * radius + half_gap * half_gap)
    return [(0.0, 0.0, float(z), sr) for z in zs]


def _cover_box(size, spacing, center=(0.0, 0.0, 0.0)):
    """Box -> a voxel grid of spheres; each cell fully contained by a sphere of radius = half the
    cell space-diagonal. `size` = full extents (x,y,z); `center` = box center in the local frame."""
    ns = [max(1, int(math.ceil(s / spacing))) for s in size]
    cell = [size[i] / ns[i] for i in range(3)]
    sr = 0.5 * math.sqrt(cell[0] ** 2 + cell[1] ** 2 + cell[2] ** 2)
    out = []
    for i in range(ns[0]):
        for j in range(ns[1]):
            for k in range(ns[2]):
                idx = (i, j, k)
                c = [center[a] - size[a] / 2.0 + (idx[a] + 0.5) * cell[a] for a in range(3)]
                out.append((c[0], c[1], c[2], sr))
    return out


# --------------------------------------------------------------------------- mesh cover (trimesh)
def _resolve_mesh_path(filename, urdf_dir):
    """Resolve a URDF mesh filename to a local path, or None. Handles file://, package:// (tries
    the URDF dir with and without the leading package name), and relative/absolute paths."""
    if filename.startswith("file://"):
        p = filename[len("file://"):]
        return p if os.path.exists(p) else None
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        candidates = [os.path.join(urdf_dir, rest),
                      os.path.join(urdf_dir, rest.split("/", 1)[-1]),
                      os.path.join(urdf_dir, "..", rest)]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None
    p = filename if os.path.isabs(filename) else os.path.join(urdf_dir, filename)
    return p if os.path.exists(p) else None


def _cover_mesh(mesh_elem, urdf_dir, spacing, link_name):
    """Voxel-fill a collision mesh -> one sphere per occupied interior voxel (radius = half the
    voxel space-diagonal). Falls back to the mesh bounding box if fill fails; returns [] + warns
    if the mesh cannot be resolved/loaded (collision skipped, not aborted)."""
    import trimesh  # local import: primitive-only robots need no trimesh

    filename = mesh_elem.get("filename")
    path = _resolve_mesh_path(filename, urdf_dir)
    if path is None:
        warnings.warn(f"spherize: link '{link_name}' collision mesh '{filename}' could not be "
                      f"resolved; SKIPPING (that link is left without collision spheres).")
        return []
    scale = mesh_elem.get("scale")
    try:
        mesh = trimesh.load(path, force="mesh")
        if scale:
            mesh.apply_scale([float(v) for v in scale.split()])
        pitch = spacing
        vg = mesh.voxelized(pitch=pitch).fill()
        pts = np.asarray(vg.points, dtype=float)
        if len(pts) == 0:
            raise ValueError("empty voxelization")
        sr = pitch * math.sqrt(3.0) / 2.0
        return [(float(p[0]), float(p[1]), float(p[2]), sr) for p in pts]
    except Exception as exc:  # noqa: BLE001 -- degrade to a bounding-box cover, never abort a robot
        try:
            mesh = trimesh.load(path, force="mesh")
            if scale:
                mesh.apply_scale([float(v) for v in scale.split()])
            ext = np.asarray(mesh.bounding_box.extents, dtype=float)
            ctr = np.asarray(mesh.bounding_box.centroid, dtype=float)
            warnings.warn(f"spherize: link '{link_name}' mesh voxelization failed ({exc}); using a "
                          f"bounding-box cover (conservative, over-approximate).")
            return _cover_box(ext, spacing, center=ctr)
        except Exception as exc2:  # noqa: BLE001
            warnings.warn(f"spherize: link '{link_name}' mesh '{filename}' unusable ({exc2}); SKIPPING.")
            return []


# --------------------------------------------------------------------------- geometry dispatch
def _cover_collision(col_elem, spacing, mesh_spacing, urdf_dir, link_name):
    """All covering spheres for ONE <collision>, in the LINK frame (its <origin> applied)."""
    geom = col_elem.find("geometry")
    if geom is None or len(geom) == 0:
        return []
    prim = geom[0]
    tag = prim.tag
    if tag == "sphere":
        local = _cover_sphere(float(prim.get("radius")))
    elif tag == "cylinder":
        local = _cover_cylinder(float(prim.get("radius")), float(prim.get("length")), spacing)
    elif tag == "box":
        local = _cover_box([float(v) for v in prim.get("size").split()], spacing)
    elif tag == "mesh":
        local = _cover_mesh(prim, urdf_dir, mesh_spacing, link_name)
    else:
        warnings.warn(f"spherize: link '{link_name}' unsupported collision geometry <{tag}>; SKIPPING.")
        return []
    R, t = _origin_of(col_elem)
    out = []
    for (x, y, z, r) in local:
        c = R @ np.array([x, y, z]) + t
        out.append((float(c[0]), float(c[1]), float(c[2]), float(r)))
    return out


# --------------------------------------------------------------------------- public API
def spherize_urdf(urdf_path, resolution, out_path=None, mesh_resolution=None):
    """Rewrite `urdf_path` with covering-sphere collisions and return the output path. `resolution`
    = sphere spacing (m) for primitives; `mesh_resolution` = voxel pitch for meshes (defaults to
    `resolution`). Links, joints, visuals and inertials are preserved; only `<collision>` geometry
    is replaced. Output is a valid URDF in the foam spherized interchange format."""
    if mesh_resolution is None:
        mesh_resolution = resolution
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    total = 0
    for link in root.findall("link"):
        name = link.get("name")
        cols = link.findall("collision")
        spheres = []
        for col in cols:
            spheres.extend(_cover_collision(col, resolution, mesh_resolution, urdf_dir, name))
        for col in cols:
            link.remove(col)
        for (x, y, z, r) in spheres:
            col = ET.SubElement(link, "collision")
            o = ET.SubElement(col, "origin")
            o.set("xyz", "{:.9g} {:.9g} {:.9g}".format(x, y, z))
            o.set("rpy", "0 0 0")
            g = ET.SubElement(col, "geometry")
            s = ET.SubElement(g, "sphere")
            s.set("radius", "{:.9g}".format(r))
        total += len(spheres)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_spherized.urdf")
        os.close(fd)
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    if total == 0:
        warnings.warn(f"spherize: '{urdf_path}' produced 0 collision spheres (no supported "
                      f"collision geometry resolved).")
    return out_path
