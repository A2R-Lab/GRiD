"""Normalized coordinate/frame metadata, independent of CUDA and artifact identity.

The schema describes geometry and ownership, not inertia, actuation or supported
algorithms. Canonical identity is shared by JSON and generated C++ consumers.
Only extraction imports numpy; validation/prepared loading needs no parser.
"""
import hashlib
import json
import math
import struct

SCHEMA_VERSION = 1
CONVENTION = "grid-xyzw-local-v1"
_WIDTHS = {"revolute": (1, 1), "continuous": (1, 1), "prismatic": (1, 1),
           "helical": (1, 1), "screw": (1, 1), "spherical": (4, 3), "floating": (7, 6)}
_TOP = {"schema_version", "model_name", "descriptor_id", "nq", "nv", "nb", "nj",
        "floating_base", "runtime_transform", "coordinate_convention", "joints",
        "bodies", "configuration_layout", "frames"}
_JOINT = {"id", "name", "kind", "parent_body", "child_body", "local_nq", "local_nv",
          "q_start", "v_start", "owns_coordinates", "axis", "pitch",
          "T_parent_joint_zero", "mimic", "limits"}
_FRAME = {"id", "name", "kind", "attached_body", "joint_id", "T_body_frame"}


def canonical_bytes(value):
    """Schema-1 typed canonical encoding; preserve names, normalize float -0."""
    def encode(item):
        if item is None:
            return ["null"]
        if type(item) is bool:
            return ["bool", item]
        if type(item) is int:
            return ["int", str(item)]
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("descriptor numbers must be finite")
            return ["f64", struct.pack('>d', 0.0 if item == 0 else item).hex()]
        if type(item) is str:
            item.encode('utf-8', errors='strict')
            return ["str", item]
        if type(item) is list:
            return ["list", [encode(v) for v in item]]
        if type(item) is dict and all(type(k) is str for k in item):
            for k in item:
                k.encode('utf-8', errors='strict')
            return ["map", [[k, encode(item[k])] for k in sorted(item)]]
        raise ValueError(f"unsupported descriptor value type: {type(item).__name__}")
    return json.dumps(encode(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def descriptor_id(descriptor):
    return hashlib.sha256(canonical_bytes({k: v for k, v in descriptor.items()
                                          if k != 'descriptor_id'})).hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError("invalid model descriptor: " + message)


def _keys(value, fields, name):
    _check(type(value) is dict and set(value) == fields, name + " fields")


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value):
    return type(value) is float and math.isfinite(value)


def _transform(value):
    _check(type(value) is list and len(value) == 4 and
           all(type(row) is list and len(row) == 4 and all(_number(x) for x in row)
               for row in value), "transform must be finite binary64 4x4")
    _check(value[3] == [0., 0., 0., 1.], "transform homogeneous last row")
    r = [row[:3] for row in value[:3]]
    _check(all(abs(sum(r[k][i]*r[k][j] for k in range(3)) - (i == j)) < 1e-10
               for i in range(3) for j in range(3)), "transform rotation orthogonality")
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    _check(abs(det-1) < 1e-10, "transform proper rotation")


def validate_descriptor(d):
    """Reject inconsistent geometry/ownership even if its hash was recomputed."""
    _keys(d, _TOP, "top-level")
    _check(type(d['schema_version']) is int and d['schema_version'] == SCHEMA_VERSION,
           "unsupported schema version")
    _check(d['coordinate_convention'] == CONVENTION, "coordinate convention")
    _check(type(d['model_name']) is str, "model name")
    _check(type(d['floating_base']) is bool and type(d['runtime_transform']) is bool,
           "model flags")
    _check(all(_integer(d[k]) for k in ('nq', 'nv', 'nb', 'nj')), "counts")
    for key in ('joints', 'bodies', 'frames', 'configuration_layout'):
        _check(type(d[key]) is list, key + " must be a list")
    joints = d['joints']
    _check(len(joints) == d['nj'] == d['nb'] == len(d['bodies']), "body/joint counts")
    _check(not d['floating_base'] or bool(joints), "missing floating root")
    # Validate records before any forward mimic-leader lookup.
    for j in joints:
        _keys(j, _JOINT, "joint")
    blocks, qend, vend, names = [], 0, 0, set()
    for i, j in enumerate(joints):
        _check(type(j['id']) is int and j['id'] == i and type(j['child_body']) is int
               and j['child_body'] == i, "joint/body ordering")
        _check(_integer(j['parent_body'], -1) and j['parent_body'] < i, "joint parent")
        _check(type(j['name']) is str and j['name'] not in names, "joint name")
        names.add(j['name'])
        kind = j['kind']
        _check(type(kind) is str and kind in _WIDTHS, "joint kind")
        nq, nv = _WIDTHS[kind]
        _check(type(j['local_nq']) is int and type(j['local_nv']) is int
               and (j['local_nq'], j['local_nv']) == (nq, nv), "local widths")
        _check((kind == 'floating') == (i == 0 and d['floating_base']), "floating root")
        axis = j['axis']
        if nq > 1:
            _check(axis is None, "multi-coordinate axis")
        else:
            _check(type(axis) is list and len(axis) == 3 and all(_number(x) for x in axis), "axis")
            _check(abs(sum(x*x for x in axis)-1.) < 1e-10, "unit axis")
        _check(_number(j['pitch']) if kind in ('helical', 'screw') else j['pitch'] is None,
               "pitch")
        _transform(j['T_parent_joint_zero'])
        m = j['mimic']
        _check(type(j['owns_coordinates']) is bool and j['owns_coordinates'] == (m is None),
               "coordinate ownership flag")
        if m is None:
            _check(type(j['q_start']) is int and type(j['v_start']) is int and
                   (j['q_start'], j['v_start']) == (qend, vend), "configuration layout")
            blocks.append(dict(kind=kind if nq > 1 else 'euclidean', q_start=qend,
                               v_start=vend, nq=nq, nv=nv))
            qend, vend = qend+nq, vend+nv
        else:
            _check(nq == nv == 1 and j['q_start'] is j['v_start'] is None, "dependent slices")
            _keys(m, {'leader_joint', 'multiplier', 'offset'}, "mimic")
            _check(_number(m['multiplier']) and _number(m['offset']), "mimic coefficients")
            leader = m['leader_joint']
            if leader is None:
                _check(m['multiplier'] == 0., "constant mimic multiplier")
            else:
                _check(_integer(leader) and leader < len(joints), "mimic leader")
                owner = joints[leader]
                _check(owner.get('mimic') is None and owner.get('local_nq') == 1,
                       "mimic leader must own one coordinate")
        _keys(j['limits'], {'position', 'velocity', 'effort'}, "limits")
        pos = j['limits']['position']
        _check(pos is None or (nq == 1 and type(pos) is list and len(pos) == 2
                              and all(x is None or _number(x) for x in pos)), "position limits")
        if pos is not None and all(x is not None for x in pos):
            _check(pos[0] <= pos[1], "ordered position limits")
        _check(kind != 'continuous' or pos is None, "continuous position limits")
        for key in ('velocity', 'effort'):
            _check(j['limits'][key] is None or _number(j['limits'][key]), "limit magnitude")
        b = d['bodies'][i]
        _keys(b, {'id', 'name', 'parent_body', 'joint_id'}, "body")
        _check(all(type(b[k]) is int for k in ('id', 'parent_body', 'joint_id')) and
               type(b['name']) is str and b == dict(id=i, name=b['name'],
               parent_body=j['parent_body'], joint_id=i), "body mapping")
    _check((qend, vend) == (d['nq'], d['nv']), "coordinate totals")
    # Type-aware comparison rejects bool/int/float aliases in redundant views.
    _check(canonical_bytes(d['configuration_layout']) == canonical_bytes(blocks),
           "configuration layout projection")
    frame_names = set()
    _check(len(d['frames']) >= 1+len(joints), "missing root/joint frames")
    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    for i, f in enumerate(d['frames']):
        _keys(f, _FRAME, "frame")
        _check(type(f['id']) is int and f['id'] == i, "frame ordering")
        _check(type(f['name']) is str and type(f['kind']) is str, "frame name/kind")
        _check((f['kind'], f['name']) not in frame_names, "duplicate typed frame name")
        frame_names.add((f['kind'], f['name']))
        _check(_integer(f['attached_body'], -1) and f['attached_body'] < len(joints), "frame attachment")
        _transform(f['T_body_frame'])
        if i == 0:
            _check(f['kind'] == 'root' and f['attached_body'] == -1 and
                   f['joint_id'] is None and f['T_body_frame'] == identity, "root frame")
        elif i <= len(joints):
            _check(f['kind'] == 'joint' and f['name'] == joints[i-1]['name'] and
                   type(f['joint_id']) is int and f['joint_id'] == f['attached_body'] == i-1
                   and f['T_body_frame'] == identity, "joint frame projection")
        else:
            _check(f['kind'] == 'fixed_joint' and f['joint_id'] is None, "fixed frame")
    _check(type(d['descriptor_id']) is str and d['descriptor_id'] == descriptor_id(d), "identity digest")
    return d


def descriptor_from_robot(robot, *, runtime_transform=False):
    """Extract effective placements and the configuration layout from one parsed model."""
    import numpy as np
    if not robot.using_quaternion:
        raise ValueError("schema 1 requires quaternion coordinates")
    if type(runtime_transform) is not bool:
        raise TypeError("runtime_transform must be bool")
    def finite_limit(value):
        if value is None or math.isinf(float(value)):
            return None
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("NaN model limit")
        return value
    joints, bodies, blocks = [], [], []
    for joint in robot.get_joints_ordered_by_id():
        kind = joint.jtype
        if kind not in _WIDTHS:
            raise ValueError(f"unsupported descriptor joint kind {kind!r}")
        nq, nv = _WIDTHS[kind]
        neutral = np.zeros(nq)
        if nq > 1:
            neutral[-1] = 1.
        zero = np.asarray(joint.get_transformation_matrix_hom_function()(
            neutral if nq > 1 else 0.), dtype=float).tolist()
        motion = np.asarray(joint.S, dtype=float)
        axis = None if nq > 1 else (motion[3:] if kind == 'prismatic' else motion[:3]).reshape(3).tolist()
        pitch = float(np.dot(motion[3:].reshape(3), axis)) if kind in ('helical', 'screw') else None
        mimic = None
        if joint.is_mimic:
            mimic = dict(leader_joint=None if joint.mimic_target_id in (None, -1)
                         else int(joint.mimic_target_id), multiplier=float(joint.get_mimic_multiplier()),
                         offset=float(joint.get_mimic_offset()))
        qi = None if mimic else int(np.asarray(robot.get_joint_index_q(joint.jid)).reshape(-1)[0])
        vi = None if mimic else int(np.asarray(robot.get_joint_index_v(joint.jid)).reshape(-1)[0])
        limits = robot.get_joint_limits_by_id(joint.jid) or []
        position = ([finite_limit(x) for x in limits] if nq == 1 and
                    kind != 'continuous' and len(limits) == 2 else None)
        parent = int(robot.get_parent_id(joint.jid))
        joints.append(dict(id=int(joint.jid), name=joint.name, kind=kind,
                           parent_body=parent, child_body=int(joint.jid), local_nq=nq,
                           local_nv=nv, q_start=qi, v_start=vi, owns_coordinates=mimic is None,
                           axis=axis, pitch=pitch, T_parent_joint_zero=zero, mimic=mimic,
                           limits=dict(position=position,
                               velocity=finite_limit(robot.get_velocity_limit_by_id(joint.jid)),
                               effort=finite_limit(robot.get_effort_limit_by_id(joint.jid)))))
        bodies.append(dict(id=int(joint.jid), name=joint.get_child(), parent_body=parent,
                           joint_id=int(joint.jid)))
        if mimic is None:
            blocks.append(dict(kind=kind if nq > 1 else 'euclidean', q_start=qi, v_start=vi,
                               nq=nq, nv=nv))
    identity = np.eye(4).tolist()
    frames = [dict(id=0, name='root', kind='root', attached_body=-1, joint_id=None,
                   T_body_frame=identity)]
    for j in joints:
        frames.append(dict(id=len(frames), name=j['name'], kind='joint', attached_body=j['id'],
                           joint_id=j['id'], T_body_frame=np.eye(4).tolist()))
    for f in robot.fixed_joints:
        parent = robot.get_joint_by_name(f.get_parent())
        frames.append(dict(id=len(frames), name=f.get_name(), kind='fixed_joint',
                           attached_body=-1 if parent is None else int(parent.jid), joint_id=None,
                           T_body_frame=np.asarray(f.get_transformation_matrix_hom(), dtype=float).tolist()))
    result = dict(schema_version=SCHEMA_VERSION, model_name=robot.get_name(), descriptor_id='',
                  nq=int(robot.get_num_pos()), nv=int(robot.get_num_vel()), nb=len(bodies),
                  nj=len(joints), floating_base=bool(robot.floating_base),
                  runtime_transform=runtime_transform, coordinate_convention=CONVENTION,
                  joints=joints, bodies=bodies, configuration_layout=sorted(blocks, key=lambda b:b['q_start']),
                  frames=frames)
    result['descriptor_id'] = descriptor_id(result)
    return validate_descriptor(result)
