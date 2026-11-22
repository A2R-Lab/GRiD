"""A small mobile manipulator on a planar root, shared by the planar-root tests.

The body is a base link carrying a continuous wheel and a prismatic torso; the
torso carries two limited revolute arm joints, a prismatic gripper finger, a
second finger that mimics the first, and a fixed tool frame. Every link has
simple collision geometry. The planar root comes either from a planar joint
named planar_base_joint declared in the URDF (variant "declared") or from the
planar_base parse option applied to the same body (variant "option").

With the default joint ordering the independent coordinates are
    q = [px, py, theta, torso_lift, shoulder, elbow, finger_left, wheel_spin]
and finger_right owns no coordinate.
"""
from pathlib import Path

ROBOT_NAME = "planar_root_manipulator"
TOOL_FRAME = "tool0"
JOINT_ORDER = ["planar_base_joint__sub0", "planar_base_joint__sub1", "planar_base_joint__sub2",
               "torso_lift", "shoulder", "elbow", "finger_left", "finger_right", "wheel_spin"]
COORDINATE_OWNERS = [name for name in JOINT_ORDER if name != "finger_right"]
NQ = NV = len(COORDINATE_OWNERS)
# (lower, upper) per coordinate; None is an unbounded side.
POSITION_LIMITS = [(None, None), (None, None), (None, None), (0.0, 0.35), (-1.2, 1.5),
                   (-2.0, 0.4), (0.0, 0.04), (None, None)]
FINGER_MULTIPLIER, FINGER_OFFSET = -1.0, 0.002

_WORLD = """
  <link name="world"/>
  <joint name="planar_base_joint" type="planar">
    <parent link="world"/><child link="base"/>
    <axis xyz="0 0 1"/>
  </joint>
"""

_FINGER_MIMIC = '<mimic joint="finger_left" multiplier="-1.0" offset="0.002"/>'

_BODY = """
  <link name="base">
    <inertial><origin xyz="0.02 -0.01 0.08" rpy="0 0 0"/><mass value="12.0"/>
      <inertia ixx="0.31" ixy="0.01" ixz="-0.02" iyy="0.36" iyz="0.015" izz="0.42"/></inertial>
    <collision><origin xyz="0 0 0.08"/><geometry><box size="0.5 0.4 0.16"/></geometry></collision>
  </link>
  <link name="wheel">
    <inertial><origin xyz="0 0.01 0" rpy="0 0 0"/><mass value="0.8"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.004"/></inertial>
    <collision><geometry><sphere radius="0.07"/></geometry></collision>
  </link>
  <link name="torso">
    <inertial><origin xyz="0.01 0.0 0.2" rpy="0 0 0"/><mass value="4.0"/>
      <inertia ixx="0.09" ixy="0.002" ixz="0.004" iyy="0.085" iyz="-0.003" izz="0.03"/></inertial>
    <collision><origin xyz="0 0 0.2"/><geometry><cylinder radius="0.08" length="0.4"/></geometry></collision>
  </link>
  <link name="upper_arm">
    <inertial><origin xyz="0.15 0.01 0.02" rpy="0 0 0"/><mass value="1.6"/>
      <inertia ixx="0.006" ixy="0.001" ixz="0.0005" iyy="0.02" iyz="0.0002" izz="0.021"/></inertial>
    <collision><origin xyz="0.15 0 0" rpy="0 1.5707963267948966 0"/>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="forearm">
    <inertial><origin xyz="0.12 -0.01 0.01" rpy="0 0 0"/><mass value="1.1"/>
      <inertia ixx="0.003" ixy="-0.0004" ixz="0.0003" iyy="0.011" iyz="0.0001" izz="0.012"/></inertial>
    <collision><origin xyz="0.12 0 0"/><geometry><box size="0.24 0.06 0.06"/></geometry></collision>
  </link>
  <link name="finger_left_link">
    <inertial><origin xyz="0.03 0.005 0" rpy="0 0 0"/><mass value="0.12"/>
      <inertia ixx="0.00003" ixy="0" ixz="0" iyy="0.00006" iyz="0" izz="0.00006"/></inertial>
    <collision><origin xyz="0.03 0 0"/><geometry><box size="0.06 0.012 0.02"/></geometry></collision>
  </link>
  <link name="finger_right_link">
    <inertial><origin xyz="0.03 -0.005 0" rpy="0 0 0"/><mass value="0.12"/>
      <inertia ixx="0.00003" ixy="0" ixz="0" iyy="0.00006" iyz="0" izz="0.00006"/></inertial>
    <collision><origin xyz="0.03 0 0"/><geometry><box size="0.06 0.012 0.02"/></geometry></collision>
  </link>
  <link name="tool"/>

  <joint name="wheel_spin" type="continuous">
    <parent link="base"/><child link="wheel"/>
    <origin xyz="-0.18 0.0 0.02" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="torso_lift" type="prismatic">
    <parent link="base"/><child link="torso"/>
    <origin xyz="0.05 0.0 0.16" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0.0" upper="0.35" effort="400" velocity="0.2"/>
  </joint>
  <joint name="shoulder" type="revolute">
    <parent link="torso"/><child link="upper_arm"/>
    <origin xyz="0.06 0.0 0.38" rpy="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-1.2" upper="1.5" effort="60" velocity="2.0"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper_arm"/><child link="forearm"/>
    <origin xyz="0.3 0.0 0.0" rpy="0.2 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.4" effort="40" velocity="2.5"/>
  </joint>
  <joint name="finger_left" type="prismatic">
    <parent link="forearm"/><child link="finger_left_link"/>
    <origin xyz="0.24 0.01 0.0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="0.0" upper="0.04" effort="20" velocity="0.1"/>
  </joint>
  <joint name="finger_right" type="prismatic">
    <parent link="forearm"/><child link="finger_right_link"/>
    <origin xyz="0.24 -0.01 0.0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-0.04" upper="0.002" effort="20" velocity="0.1"/>
    FINGER_MIMIC
  </joint>
  <joint name="tool0" type="fixed">
    <parent link="forearm"/><child link="tool"/>
    <origin xyz="0.3 0.0 0.01" rpy="0 0.1 0"/>
  </joint>
"""


def urdf_text(*, declared_planar_root, mimic=True):
    """URDF text of the fixture. declared_planar_root adds the world link and
    the planar_base_joint planar joint; mimic=False frees the second finger."""
    body = _BODY.replace("FINGER_MIMIC", _FINGER_MIMIC if mimic else "")
    return ('<?xml version="1.0"?>\n<robot name="' + ROBOT_NAME + '">'
            + (_WORLD if declared_planar_root else "") + body + "</robot>\n")


def write_planar_root_model(directory, variant, *, mimic=True):
    """Write one variant and return (path, planar_base option to parse it with).

    "declared": the URDF holds the planar joint, parse with planar_base=False.
    "option":   the URDF has no root joint, parse with planar_base=True.
    "fixed":    the URDF has no root joint, parse with planar_base=False.
    """
    if variant not in ("declared", "option", "fixed"):
        raise ValueError(f"unknown planar root variant: {variant}")
    path = Path(directory) / f"planar_root_{variant}{'' if mimic else '_free_finger'}.urdf"
    path.write_text(urdf_text(declared_planar_root=(variant == "declared"), mimic=mimic))
    return path, variant == "option"


def sample_states(count, seed, *, dtype=float):
    """States inside the joint limits whose base block covers small values,
    translations of tens of metres and yaw beyond one turn."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [37.5, -52.25, 4.0],
                     [-81.0, 64.5, -7.5], [12.0, 3.0, 3.3], [-0.05, 0.02, -3.6]])
    q = np.zeros((count, NQ))
    for row in range(count):
        q[row, :3] = base[row % len(base)] + (0 if row < len(base) else rng.uniform(-1, 1, 3))
        for col in range(3, NQ):
            lower, upper = POSITION_LIMITS[col]
            if lower is None:
                q[row, col] = rng.uniform(-9.0, 9.0)
            elif COORDINATE_OWNERS[col] == "shoulder":
                # Keeps the tool pitch away from +-pi/2, where rpy is singular.
                q[row, col] = rng.uniform(-1.0, 1.0)
            else:
                q[row, col] = rng.uniform(lower, upper)
    v = rng.uniform(-0.8, 0.8, (count, NV))
    u = rng.uniform(-3.0, 3.0, (count, NV))
    return q.astype(dtype), v.astype(dtype), u.astype(dtype)
