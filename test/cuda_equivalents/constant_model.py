"""The constant-joint fixture robot shared by the constant-joint and descriptor suites.

A helper module, not a test module, so the split driver's helper-import rule
fingerprints it into every shard that imports it.
"""
from URDFParser import URDFParser


def constant_model(directory, zero=False, locked=False, branched=False):
    inertial = ('<inertial><origin xyz=".03 -.02 .07"/><mass value="1"/>'
                '<inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial>')
    links = ''.join(f'<link name="{name}">{inertial}</link>'
                    for name in ('base', 'mount', 'arm', 'follower'))
    drive = ('<joint name="drive" type="fixed"><parent link="mount"/><child link="arm"/>'
             '<origin xyz=".1 0 .2"/></joint>' if zero else
             '<joint name="drive" type="revolute"><parent link="mount"/><child link="arm"/>'
             '<origin xyz=".1 0 .2"/><axis xyz="0 1 0"/>'
             '<limit lower="-3" upper="3" effort="10" velocity="10"/></joint>')
    follower = ('<joint name="constant" type="fixed"><parent link="arm"/><child link="follower"/>'
                '<origin xyz=".2 0 .3" rpy="0 0 .17"/></joint>' if locked else
                '<joint name="constant" type="revolute"><parent link="arm"/><child link="follower"/>'
                '<origin xyz=".2 0 .3"/><axis xyz="0 0 1"/>'
                '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
                '<mimic joint="fixed" multiplier="2" offset=".17"/></joint>')
    if branched:
        follower = follower.replace('<parent link="arm"/>', '<parent link="mount"/>')
    path = directory / f'model-{zero}-{locked}-{branched}.urdf'
    path.write_text(f'<robot name="constant">{links}'
                   '<joint name="fixed" type="fixed"><parent link="base"/><child link="mount"/></joint>'
                   f'{drive}{follower}</robot>')
    return URDFParser().parse(str(path))
