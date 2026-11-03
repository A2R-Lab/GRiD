"""A four-joint chain whose second and fourth joints are polynomial followers
(GRiD's <mimic polycoef>), the smallest robot whose fold multipliers vary
with the leader coordinate. The same chain is the reference's own test
case (RBDReference/tests/test_embedding_projection.py); the reference
routes it through the projection of the full spanning tree, so it is the
oracle for the generated kernels here."""
from __future__ import annotations

_INERTIAL = ('<inertial><origin xyz="0.05 0.02 0.1"/><mass value="1.3"/>'
             '<inertia ixx="0.04" ixy="0.001" ixz="0.002" iyy="0.05" iyz="0.003" izz="0.03"/></inertial>')


def _joint(name, parent, child, axis, xyz, rpy, mimic_xml=""):
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis}"/><origin xyz="{xyz}" rpy="{rpy}"/>'
            '<limit lower="-3" upper="3" effort="10" velocity="10"/>' + mimic_xml + '</joint>')


def polynomial_chain_urdf():
    links = '<link name="base"/>' + "".join(f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, 5))
    joints = (_joint("j1", "base", "l1", "0 0 1", "0 0 0.1", "0 0 0")
              + _joint("j2", "l1", "l2", "0 1 0", "0.2 0 0.05", "0.1 0 0",
                       '<mimic joint="j1" polycoef="0.1 0.6 0.25 -0.1 0.04" q0="0.2"/>')
              + _joint("j3", "l2", "l3", "1 0 0", "0.15 0.05 0", "0 0.2 0")
              + _joint("j4", "l3", "l4", "0 1 0", "0.1 0 0.1", "0 0 0.3",
                       '<mimic joint="j3" polycoef="0 1.5 0 0.3"/>'))
    return f'<robot name="poly_chain">{links}{joints}</robot>'


def write_polynomial_chain(directory):
    path = directory / "poly_chain.urdf"
    path.write_text(polynomial_chain_urdf())
    return path
