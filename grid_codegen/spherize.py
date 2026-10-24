"""CPU-only sphere-model generation and sampled fidelity reporting.

Run `python -m grid_codegen.spherize robot.urdf --out spheres.urdf
--mesh-mode bounded-bulge --report`, or pass an existing sphere URDF to
--report. Generation always reads the COLLISION elements. The report measures
against the collision geometry by default; `--geometry visual` measures against
the `<visual>` meshes instead, for comparison with fidelity tables published
against visual meshes (collision hulls of some robots sit well outside the
visible links). A visual mesh that does not resolve to a local file is reported
as such and its link skipped.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .algorithms._spherize import spherize_urdf, _load_mesh, _origin_of, _collision_geometry, _resolve_mesh_path
from .algorithms._collision import parse_spherized_urdf, validate_sphere_model


def fidelity_report(source, spheres, geometry='collision'):
    """Per-link sampled fidelity of the sphere model against the source's `collision` or `visual`
    geometry. A link whose visual mesh cannot be resolved yields a row with `unresolved` set to
    the mesh filename and no measurements; collision meshes must resolve (generation needs them)."""
    import trimesh
    from .algorithms._bounded_spheres import sampled_fidelity
    if geometry not in ('collision', 'visual'):
        raise ValueError("geometry must be 'collision' or 'visual'")
    validate_sphere_model(source, spheres)
    model = parse_spherized_urdf(spheres)
    rows = []
    for link in ET.parse(source).getroot().findall('link'):
        parts = []
        unresolved = None
        for col in link.findall(geometry):
            g = _collision_geometry(col, link.get('name'))
            if g.tag == 'mesh':
                if geometry == 'visual' and _resolve_mesh_path(g.get('filename'), str(Path(source).resolve().parent)) is None:
                    unresolved = g.get('filename')
                    break
                mesh = _load_mesh(g, str(Path(source).resolve().parent), link.get('name'))
            elif g.tag == 'box':
                mesh = trimesh.creation.box(extents=[float(v) for v in g.get('size').split()])
            elif g.tag == 'sphere':
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(g.get('radius')))
            elif g.tag == 'cylinder':
                mesh = trimesh.creation.cylinder(radius=float(g.get('radius')), height=float(g.get('length')), sections=64)
            else:
                raise ValueError(f'Unsupported geometry: {g.tag}')
            R, t = _origin_of(col)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            mesh.apply_transform(T)
            parts.append(mesh)
        if unresolved is not None:
            rows.append(dict(link=link.get('name'), unresolved=unresolved))
        elif parts and model.get(link.get('name')):
            mesh = trimesh.util.concatenate(parts)
            rows.append(dict(link=link.get('name'), **sampled_fidelity(mesh, model[link.get('name')])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdf', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resolution', type=float, default=0.05, help='Primitive/voxel spacing in metres')
    parser.add_argument('--mesh-mode', choices=('voxel', 'bounded-bulge'), default='voxel')
    parser.add_argument('--report', nargs='?', const='', metavar='SPHERE_URDF',
                        help='Report the generated model, or an existing sphere URDF')
    parser.add_argument('--geometry', choices=('collision', 'visual'), default='collision',
                        help='Geometry the report measures against (generation always uses collision)')
    args = parser.parse_args()
    if not args.out and not args.report:
        parser.error('Specify --out or --report SPHERE_URDF')
    if args.out:
        if args.out.resolve() == args.urdf.resolve():
            parser.error('Keep the original URDF; --out must be a different path')
        spherize_urdf(args.urdf, args.resolution, args.out, mesh_mode=args.mesh_mode)
    if args.report is not None:
        sphere_path = Path(args.report) if args.report else args.out
        print(f'Sampled fidelity against {args.geometry} geometry; not a continuous-coverage certificate.')
        print('Bulge is voxel-estimated (2.5 mm pitch); coverage samples 2000 mesh-surface points/link.')
        print('link spheres bulge_max_mm bulge_p99_mm uncovered_surface_percent')
        for row in fidelity_report(args.urdf, sphere_path, geometry=args.geometry):
            if 'unresolved' in row:
                print(f"{row['link']} visual mesh not resolved: {row['unresolved']}")
                continue
            print(f"{row['link']} {row['spheres']} {1000*row['bulge_max_m']:.3f} "
                  f"{1000*row['bulge_p99_m']:.3f} {100*row['uncovered_surface_fraction']:.3f}")


if __name__ == '__main__':
    main()
