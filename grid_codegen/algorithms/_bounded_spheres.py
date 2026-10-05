"""Deterministic, voxel-based bounded-bulge mesh approximation.

Adapted from HJCD-IK's make_bounded_bulge_spheres.py. One fixed policy:
2.5 mm voxels, 10 mm bulge budget, coarse candidates then full-lattice
fallback for thin parts. The budget is relative to the voxel approximation,
not a certified bound on the source mesh. No solver policy or Panda assets.
"""
import heapq

import numpy as np

PITCH = 0.0025
BULGE = 0.010
MAX_VOXELS = 4_000_000


def voxel_fields(mesh, pitch=PITCH):
    from scipy import ndimage
    # Bound memory before trimesh allocates the voxel lattice. This is an
    # explicit resource error, never a partial or empty collision model.
    shape = np.ceil(np.asarray(mesh.extents) / pitch) + 7
    if np.prod(shape) > MAX_VOXELS:
        raise ValueError('bounded-bulge mesh exceeds the voxel budget; split the mesh into links')
    vg = mesh.voxelized(pitch=pitch).fill()
    occ = np.pad(np.asarray(vg.matrix, dtype=bool), 2)
    if not occ.any():
        raise ValueError('mesh voxelization is empty')
    # Trimesh indices refer to voxel CENTRES, not voxel corners.
    origin = np.asarray(vg.transform)[:3, 3] - 2 * pitch
    inside = ndimage.distance_transform_edt(occ) * pitch
    surface = occ & ~ndimage.binary_erosion(
        occ, structure=ndimage.generate_binary_structure(3, 1))
    return occ, origin, inside, surface


def bounded_spheres(mesh):
    from scipy.spatial import cKDTree
    occ, origin, inside, surface = voxel_fields(mesh)
    xyz = np.argwhere(surface) * PITCH + origin
    tree = cKDTree(xyz)

    def fit(stride, min_radius):
        lattice = np.zeros_like(occ)
        lattice[::stride, ::stride, ::stride] = True
        idx = np.argwhere(occ & lattice & (inside >= min_radius))
        centers = idx * PITCH + origin
        radii = inside[tuple(idx.T)] + BULGE
        covered = np.zeros(len(xyz), dtype=bool)
        # Store gains, not every candidate's membership list: the latter can
        # dominate RAM for dense, thick meshes.
        margin = np.sqrt(3) * PITCH / 2
        counts = tree.query_ball_point(centers, radii - margin, return_length=True)
        heap = [(-int(n), i) for i, n in enumerate(counts) if n]
        heapq.heapify(heap)
        chosen = []
        while heap and not covered.all():
            _, i = heapq.heappop(heap)
            members = np.asarray(tree.query_ball_point(centers[i], radii[i] - margin), dtype=int)
            gain = int(np.count_nonzero(~covered[members]))
            if not gain:
                continue
            if heap and gain < -heap[0][0]:
                heapq.heappush(heap, (-gain, i))
                continue
            covered[members] = True
            chosen.append((*map(float, centers[i]), float(radii[i])))
        return chosen, bool(covered.all())

    spheres, complete = fit(2, 0.008)
    if not complete:
        spheres, complete = fit(1, 0.0)
    if not complete or not spheres:
        raise ValueError('bounded-bulge fit did not cover all surface voxels')
    return spheres


def sampled_fidelity(mesh, spheres, samples=2000):
    """Reproducible mesh-surface coverage and voxel-estimated sphere bulge.

    Uncovered fraction is sampled on triangles with zero gap tolerance.
    Bulge samples 400 points/sphere and measures distance to occupied voxel
    centres, with interior points set to zero. Neither is a geometric proof.
    """
    import trimesh
    from scipy.spatial import cKDTree
    spheres = np.asarray(spheres, dtype=float).reshape(-1, 4)
    if not len(spheres):
        raise ValueError('Cannot report fidelity for an empty sphere model')
    points, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    gaps = []
    for block in np.array_split(points, max(1, len(points) // 128)):
        gaps.extend(np.min(np.linalg.norm(block[:, None] - spheres[None, :, :3], axis=2)
                           - spheres[None, :, 3], axis=1))
    occ, origin, _, _ = voxel_fields(mesh)
    tree = cKDTree(np.argwhere(occ) * PITCH + origin)
    rng = np.random.default_rng(0)
    bulges = []
    for x, y, z, radius in spheres:
        directions = rng.normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = np.array([x, y, z]) + radius * directions
        idx = np.rint((points - origin) / PITCH).astype(int)
        valid = np.all((idx >= 0) & (idx < np.array(occ.shape)), axis=1)
        interior = np.zeros(len(idx), dtype=bool)
        interior[valid] = occ[tuple(idx[valid].T)]
        distances = tree.query(points)[0]
        distances[interior] = 0
        bulges.extend(distances)
    return dict(spheres=len(spheres), bulge_max_m=float(np.max(bulges)),
                bulge_p99_m=float(np.percentile(bulges, 99)),
                uncovered_surface_fraction=float(np.mean(np.asarray(gaps) > 0)),
                surface_samples=samples, voxel_pitch_m=PITCH)
