"""Descriptor-only forward kinematics: fields, elementary trig and q, no parser objects.

A helper module shared by the CPU descriptor tests and the GPU metadata and
joint-transform suites; the split driver fingerprints it into their shards.
"""
import numpy as np
from RBDReference import RBDReference


def descriptor_fk(d, q):
    """No parser objects or transform helpers: fields, elementary trig and q only."""
    world = []
    for j in d['joints']:
        local = q[j['q_start']:j['q_start']+j['local_nq']] if j['mimic'] is None else None
        if j['mimic'] is not None:
            m = j['mimic']
            value = m['offset']
            if m['leader_joint'] is not None:
                value += m['multiplier']*q[d['joints'][m['leader_joint']]['q_start']]
            local = np.array([value])
        motion = np.eye(4)
        if j['kind'] in ('floating', 'spherical'):
            x, y, z, w = local[-4:]
            motion[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                              [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                              [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
            if j['kind'] == 'floating':
                motion[:3, 3] = local[:3]
        else:
            axis = np.array(j['axis'])
            angle = local[0]
            if j['kind'] == 'prismatic':
                motion[:3, 3] = axis*angle
            else:
                x, y, z = axis
                cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                motion[:3, :3] = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
                if j['pitch'] is not None:
                    motion[:3, 3] = axis*angle*j['pitch']
        parent = np.eye(4) if j['parent_body'] == -1 else world[j['parent_body']]
        world.append(parent@np.array(j['T_parent_joint_zero'])@motion)
    return [(np.eye(4) if f['attached_body'] == -1 else world[f['attached_body']])
            @np.array(f['T_body_frame']) for f in d['frames']]


def assert_reference(d, robot, q):
    ref = RBDReference(robot)
    world, _ = ref._frame_world_placement_and_chain(q)
    for f, actual in zip(d['frames'], descriptor_fk(d, q)):
        if f['kind'] == 'root':
            expected = np.eye(4)
        else:
            jid, offset = ref._resolve_frame_joint(f['name'])
            expected = (np.eye(4) if jid == -1 else world[jid])@offset
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
