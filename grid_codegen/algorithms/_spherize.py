"""W3 Increment 1 — automated URDF -> spherized-URDF (foam-compatible interchange format).

Turns a URDF's per-link `<collision>` geometry into a set of COVERING spheres and rewrites the
URDF with one `<collision><geometry><sphere/></geometry><origin xyz/></collision>` per sphere
(center in the LINK frame). This is the exact format `parse_spherized_urdf`/`build_sphere_tiers`
(the foam interchange in _collision.py) already consume, so a spherized URDF from here is drop-in
interchangeable with a foam-produced one.

Design choices (why custom, not foam-only):
  * PRIMITIVES (sphere/cylinder/box) are covered ANALYTICALLY -- exact, deterministic, and needing
    NO external mesh assets. go2 uses boxes/cylinders/spheres. The bundled
    iiwa14 model also references Drake meshes for links 6 and 7.
  * MESHES are voxel-filled via trimesh, or use the opt-in bounded-bulge fit.
    Missing/unusable collision geometry is an error, never a partial model.

`resolution` is the target sphere SPACING in meters: coarser -> fewer/larger spheres (broad tier),
finer -> more/smaller spheres (fine tier). Primitive covers are analytic;
mesh approximation and sampled fidelity do not certify continuous coverage.
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
    the URDF dir and explicit ROS_PACKAGE_PATH roots), and relative/absolute paths.
    Never downloads assets or searches unrelated caches implicitly.
    """
    if not filename:
        return None
    if filename.startswith("file://"):
        p = filename[len("file://"):]
        return p if os.path.exists(p) else None
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        candidates = [os.path.join(urdf_dir, rest),
                      os.path.join(urdf_dir, rest.split("/", 1)[-1]),
                      os.path.join(urdf_dir, "..", rest)]
        package, _, relative = rest.partition('/')
        for root in filter(None, os.environ.get('ROS_PACKAGE_PATH', '').split(os.pathsep)):
            candidates.append(os.path.join(root, rest))
            if os.path.basename(os.path.normpath(root)) == package:
                candidates.append(os.path.join(root, relative))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None
    p = filename if os.path.isabs(filename) else os.path.join(urdf_dir, filename)
    return p if os.path.exists(p) else None


def _load_mesh(mesh_elem, urdf_dir, link_name):
    import trimesh
    filename = mesh_elem.get('filename')
    path = _resolve_mesh_path(filename, urdf_dir)
    if path is None:
        raise ValueError(f"spherize: link '{link_name}' collision mesh '{filename}' could not be resolved; "
                         "provide the asset locally (ROS_PACKAGE_PATH for package:// URIs)")
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh_elem.get('scale'):
            mesh.apply_scale([float(v) for v in mesh_elem.get('scale').split()])
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
            raise ValueError('empty, nonfinite or degenerate mesh')
        return mesh
    except Exception as exc:
        raise ValueError(f"spherize: link '{link_name}' mesh '{filename}' unusable: {exc}") from exc


def _cover_mesh(mesh_elem, urdf_dir, spacing, link_name, mesh_mode='voxel'):
    """Voxel-fill a collision mesh -> one sphere per occupied interior voxel (radius = half the
    voxel space-diagonal). Falls back to a bounding-box cover if voxel fill
    fails. Unresolvable meshes fail before a model can be written."""
    try:
        import trimesh  # local import: primitive-only robots never reach this
    except ImportError as e:  # a declared base dependency, but name it for a hand-rolled env
        raise ImportError(
            f"spherize: link '{link_name}' has a mesh <collision> element, which needs the "
            "'trimesh' package (a grid-rbd base dependency: pip install 'trimesh>=4', or "
            "reinstall with pip install -e .)") from e

    mesh = _load_mesh(mesh_elem, urdf_dir, link_name)
    if mesh_mode == 'bounded-bulge':
        from ._bounded_spheres import bounded_spheres
        return bounded_spheres(mesh)
    if mesh_mode != 'voxel':
        raise ValueError(f'Unknown mesh mode: {mesh_mode}')
    try:
        pitch = spacing
        vg = mesh.voxelized(pitch=pitch).fill()
        pts = np.asarray(vg.points, dtype=float)
        if len(pts) == 0:
            raise ValueError("empty voxelization")
        sr = pitch * math.sqrt(3.0) / 2.0
        return [(float(p[0]), float(p[1]), float(p[2]), sr) for p in pts]
    except Exception as exc:
        ext = np.asarray(mesh.bounding_box.extents, dtype=float)
        ctr = np.asarray(mesh.bounding_box.centroid, dtype=float)
        warnings.warn(f"spherize: link '{link_name}' voxelization failed ({exc}); using a bounding-box cover")
        return _cover_box(ext, spacing, center=ctr)


# --------------------------------------------------------------------------- geometry dispatch
def _collision_geometry(col_elem, link_name):
    """One validated geometry element, shared by generation and fidelity reporting."""
    geom = col_elem.find("geometry")
    if geom is None or len(geom) == 0:
        raise ValueError(f"spherize: link '{link_name}' has empty collision geometry")
    prim = geom[0]
    tag = prim.tag
    if len(geom) != 1:
        raise ValueError(f"spherize: link '{link_name}' must have one shape per collision geometry")
    if tag in ('sphere', 'cylinder', 'box'):
        dims = ([float(v) for v in prim.get('size', '').split()] if tag == 'box'
                else [float(prim.get('radius', 'nan'))] +
                ([float(prim.get('length', 'nan'))] if tag == 'cylinder' else []))
        if (tag == 'box' and len(dims) != 3) or any(not math.isfinite(v) or v <= 0 for v in dims):
            raise ValueError(f"spherize: link '{link_name}' has invalid {tag} dimensions")
    return prim


def _cover_collision(col_elem, spacing, mesh_spacing, urdf_dir, link_name, mesh_mode='voxel'):
    """All covering spheres for ONE <collision>, in the LINK frame (its <origin> applied)."""
    prim = _collision_geometry(col_elem, link_name)
    tag = prim.tag
    if tag == "sphere":
        local = _cover_sphere(float(prim.get("radius")))
    elif tag == "cylinder":
        local = _cover_cylinder(float(prim.get("radius")), float(prim.get("length")), spacing)
    elif tag == "box":
        local = _cover_box([float(v) for v in prim.get("size").split()], spacing)
    elif tag == "mesh":
        local = _cover_mesh(prim, urdf_dir, mesh_spacing, link_name, mesh_mode)
    else:
        raise ValueError(f"spherize: link '{link_name}' unsupported collision geometry <{tag}>")
    R, t = _origin_of(col_elem)
    out = []
    for (x, y, z, r) in local:
        c = R @ np.array([x, y, z]) + t
        out.append((float(c[0]), float(c[1]), float(c[2]), float(r)))
    return out


# --------------------------------------------------------------------------- public API
def spherize_urdf(urdf_path, resolution, out_path=None, mesh_resolution=None, *, mesh_mode='voxel'):
    """Rewrite `urdf_path` with covering-sphere collisions and return the output path. `resolution`
    = sphere spacing (m) for primitives; `mesh_resolution` = voxel pitch for meshes (defaults to
    `resolution`). Links, joints, visuals and inertials are preserved; only `<collision>` geometry
    is replaced. Output is a valid URDF in the foam spherized interchange format."""
    if mesh_resolution is None:
        mesh_resolution = resolution
    if any(not math.isfinite(x) or x <= 0 for x in (resolution, mesh_resolution)):
        raise ValueError('Sphere resolution must be positive and finite')
    if mesh_mode not in ('voxel', 'bounded-bulge'):
        raise ValueError(f'Unknown mesh mode: {mesh_mode}')
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    total = 0
    for link in root.findall("link"):
        name = link.get("name")
        cols = link.findall("collision")
        spheres = []
        for col in cols:
            spheres.extend(_cover_collision(col, resolution, mesh_resolution, urdf_dir, name, mesh_mode))
        if any(not np.isfinite(s).all() or s[3] <= 0 for s in spheres):
            raise ValueError(f"spherize: link '{name}' has invalid collision spheres")
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
