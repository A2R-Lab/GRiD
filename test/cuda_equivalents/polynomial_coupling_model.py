"""A four-joint chain whose second and fourth joints are polynomial followers
(GRiD's <mimic polycoef>), the smallest robot whose fold multipliers vary
with the leader coordinate. The same chain is the reference's own test
case (RBDReference/tests/test_embedding_projection.py); the reference
routes it through the projection of the full spanning tree, so it is the
oracle for the generated kernels here."""
from __future__ import annotations

_INERTIAL = ('<inertial><origin xyz="0.05 0.02 0.1"/><mass value="1.3"/>'
             '<inertia ixx="0.04" ixy="0.001" ixz="0.002" iyy="0.05" iyz="0.003" izz="0.03"/></inertial>')


# per-joint (damping, friction) of the damped variant (joint_dynamics=True)
JOINT_DYNAMICS = {"j1": (0.1, 0.05), "j2": (0.4, 0.2), "j3": (0.25, 0.0), "j4": (0.3, 0.15)}
# the linear variant: the same chain with constant ratios 2 and -1.5 (fr3's
# ratio of one cannot tell alpha from alpha^2 in the joint-dynamics fold)
_LINEAR = {"j2": '<mimic joint="j1" multiplier="2.0" offset="0.1"/>',
           "j4": '<mimic joint="j3" multiplier="-1.5" offset="-0.2"/>'}
_POLY = {"j2": '<mimic joint="j1" polycoef="0.1 0.6 0.25 -0.1 0.04" q0="0.2"/>',
         "j4": '<mimic joint="j3" polycoef="0 1.5 0 0.3"/>'}


def _joint(name, parent, child, axis, xyz, rpy, mimic_xml="", joint_dynamics=False):
    dyn = ""
    if joint_dynamics:
        b, f = JOINT_DYNAMICS[name]
        dyn = f'<dynamics damping="{b}" friction="{f}"/>'
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis}"/><origin xyz="{xyz}" rpy="{rpy}"/>' + dyn +
            '<limit lower="-3" upper="3" effort="10" velocity="10"/>' + mimic_xml + '</joint>')


def polynomial_chain_urdf(kind="polynomial", joint_dynamics=False, branched=False):
    """kind: "polynomial" (the default chain) or "linear" (constant ratios);
    joint_dynamics adds viscous damping and Coulomb friction to every joint;
    branched puts j3/j4 on a second branch off the base, with j4 still the
    follower of j1 (a leader the second leaf's chain never reaches, the
    cross-finger coupling of a gripper), two leaves l2 and l4."""
    mimic = _POLY if kind == "polynomial" else _LINEAR
    links = '<link name="base"/>' + "".join(f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, 5))
    if branched:
        j4 = mimic["j4"].replace('joint="j3"', 'joint="j1"')
        joints = (_joint("j1", "base", "l1", "0 0 1", "0 0 0.1", "0 0 0", joint_dynamics=joint_dynamics)
                  + _joint("j2", "l1", "l2", "0 1 0", "0.2 0 0.05", "0.1 0 0", mimic["j2"], joint_dynamics)
                  + _joint("j3", "base", "l3", "1 0 0", "-0.15 0.05 0.1", "0 0.2 0", joint_dynamics=joint_dynamics)
                  + _joint("j4", "l3", "l4", "0 1 0", "0.1 0 0.1", "0 0 0.3", j4, joint_dynamics))
    else:
        joints = (_joint("j1", "base", "l1", "0 0 1", "0 0 0.1", "0 0 0", joint_dynamics=joint_dynamics)
                  + _joint("j2", "l1", "l2", "0 1 0", "0.2 0 0.05", "0.1 0 0", mimic["j2"], joint_dynamics)
                  + _joint("j3", "l2", "l3", "1 0 0", "0.15 0.05 0", "0 0.2 0", joint_dynamics=joint_dynamics)
                  + _joint("j4", "l3", "l4", "0 1 0", "0.1 0 0.1", "0 0 0.3", mimic["j4"], joint_dynamics))
    name = ("poly_chain" if kind == "polynomial" else "linear_chain") + ("_damped" if joint_dynamics else "") + ("_branched" if branched else "")
    return f'<robot name="{name}">{links}{joints}</robot>'


def write_polynomial_chain(directory, kind="polynomial", joint_dynamics=False, branched=False):
    name = ("poly_chain" if kind == "polynomial" else "linear_chain") + ("_damped" if joint_dynamics else "") + ("_branched" if branched else "")
    path = directory / f"{name}.urdf"
    path.write_text(polynomial_chain_urdf(kind, joint_dynamics, branched))
    return path
