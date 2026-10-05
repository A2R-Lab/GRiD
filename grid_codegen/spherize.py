"""CPU-only sphere-model generation and sampled fidelity reporting.

Run `python -m grid_codegen.spherize robot.urdf --out spheres.urdf
--mesh-mode bounded-bulge --report`, or pass an existing sphere URDF to
--report. Geometry is taken from COLLISION elements, not visual elements.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .algorithms._spherize import spherize_urdf, _load_mesh, _origin_of, _collision_geometry
from .algorithms._collision import parse_spherized_urdf, validate_sphere_model


def fidelity_report(source, spheres):
    import trimesh
    from .algorithms._bounded_spheres import sampled_fidelity
    validate_sphere_model(source, spheres)
    model = parse_spherized_urdf(spheres)
    rows = []
    for link in ET.parse(source).getroot().findall('link'):
        parts = []
        for col in link.findall('collision'):
            g = _collision_geometry(col, link.get('name'))
            if g.tag == 'mesh':
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
        if parts:
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
    args = parser.parse_args()
    if not args.out and not args.report:
        parser.error('Specify --out or --report SPHERE_URDF')
    if args.out:
        if args.out.resolve() == args.urdf.resolve():
            parser.error('Keep the original URDF; --out must be a different path')
        spherize_urdf(args.urdf, args.resolution, args.out, mesh_mode=args.mesh_mode)
    if args.report is not None:
        sphere_path = Path(args.report) if args.report else args.out
        print('Sampled fidelity against collision geometry; not a continuous-coverage certificate.')
        print('Bulge is voxel-estimated (2.5 mm pitch); coverage samples 2000 mesh-surface points/link.')
        print('link spheres bulge_max_mm bulge_p99_mm uncovered_surface_percent')
        for row in fidelity_report(args.urdf, sphere_path):
            print(f"{row['link']} {row['spheres']} {1000*row['bulge_max_m']:.3f} "
                  f"{1000*row['bulge_p99_m']:.3f} {100*row['uncovered_surface_fraction']:.3f}")


if __name__ == '__main__':
    main()
