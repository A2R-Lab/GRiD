"""C++ coordinate metadata emitted only from the validated model descriptor."""
import json
import re

from .model_descriptor import validate_descriptor

JOINT_KINDS = ('revolute', 'continuous', 'prismatic', 'helical', 'screw', 'spherical', 'floating')
FRAME_KINDS = ('root', 'joint', 'fixed_joint')


def emit_model_metadata(descriptor, namespace='grid'):
    """Host constexpr and runtime-indexed device values; no host-array device reads."""
    d = validate_descriptor(descriptor)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', namespace):
        raise ValueError('model metadata namespace must be a C++ identifier')
    def string(value):
        # UTF-8 bytes with fixed-width octal escapes avoid C++ hex escape greed
        # and universal-character restrictions for control characters.
        return '"' + ''.join(chr(b) if 32 <= b < 127 and b not in (34, 92)
                             else '\\%03o' % b for b in value.encode('utf-8')) + '"'
    def number(x):
        return repr(0.0 if x == 0 else x)
    def array(values):
        return '{' + ','.join(number(x) for x in values) + '}'
    def flat(matrix):
        return [matrix[r][c] for c in range(4) for r in range(4)]
    def optional(value):
        return str(-1 if value is None else value)
    lines = ['// Model-coordinate metadata; not a complete physical-model identity.',
             f'namespace {namespace} {{ namespace model {{',
             'enum class joint_kind { ' + ', '.join(JOINT_KINDS) + ' };',
             'enum class frame_kind { ' + ', '.join(FRAME_KINDS) + ' };',
             f'inline constexpr int schema_version = {d["schema_version"]};',
             f'inline constexpr int nq = {d["nq"]}, nv = {d["nv"]}, nb = {d["nb"]}, nj = {d["nj"]};',
             f'inline constexpr int frame_count = {len(d["frames"])};',
             f'inline constexpr bool runtime_transform = {str(d["runtime_transform"]).lower()};',
             f'inline constexpr char descriptor_id[] = {string(d["descriptor_id"])};',
             f'inline constexpr char descriptor_json[] = {string(json.dumps(d, ensure_ascii=False, allow_nan=False, separators=(",", ":")))};',
             'struct joint_record {',
             '  int id, parent_body, child_body, q_start, v_start, local_nq, local_nv;',
             '  joint_kind kind; bool owns_coordinates, has_axis, has_pitch, has_mimic;',
             '  int leader_joint; double multiplier, offset, axis[3], pitch, T_parent_joint_zero[16];',
             '  bool has_position_limit[2], has_velocity_limit, has_effort_limit;',
             '  double position_limit[2], velocity_limit, effort_limit;',
             '};',
             'struct frame_record {',
             '  int id, attached_body, joint_id; frame_kind kind; double T_body_frame[16];',
             '};',
             '__host__ __device__ constexpr bool joint(int id, joint_record &out) {',
             '  switch(id) {']
    for j in d['joints']:
        mimic = j['mimic']
        values = [str(j[k]) for k in ('id', 'parent_body', 'child_body')]
        values += [optional(j[k]) for k in ('q_start', 'v_start')]
        values += [str(j[k]) for k in ('local_nq', 'local_nv')]
        values += ['joint_kind::'+j['kind'], str(j['owns_coordinates']).lower(),
                   str(j['axis'] is not None).lower(), str(j['pitch'] is not None).lower(),
                   str(mimic is not None).lower(), optional(None if mimic is None else mimic['leader_joint']),
                   number(1.0 if mimic is None else (0.0 if mimic['multiplier'] is None else mimic['multiplier'])),
                   number(0.0 if mimic is None else (0.0 if mimic['offset'] is None else mimic['offset'])), array(j['axis'] or [0., 0., 0.]),
                   number(j['pitch'] or 0.0), array(flat(j['T_parent_joint_zero']))]
        pos = j['limits']['position'] or [None, None]
        vel, effort = j['limits']['velocity'], j['limits']['effort']
        values += ['{' + ','.join(str(x is not None).lower() for x in pos) + '}',
                   str(vel is not None).lower(), str(effort is not None).lower(),
                   array([x if x is not None else 0.0 for x in pos]),
                   number(vel if vel is not None else 0.0), number(effort if effort is not None else 0.0)]
        lines.append(f'  case {j["id"]}: out = joint_record{{'+','.join(values)+'}; return true;')
    lines += ['  default: return false;', '  }', '}',
              '__host__ __device__ constexpr bool frame(int id, frame_record &out) {', '  switch(id) {']
    for f in d['frames']:
        values = [str(f['id']), str(f['attached_body']), optional(f['joint_id']),
                   'frame_kind::'+f['kind'], array(flat(f['T_body_frame']))]
        lines.append(f'  case {f["id"]}: out = frame_record{{'+','.join(values)+'}; return true;')
    lines += ['  default: return false;', '  }', '}']
    for kind, records in [('joint', d['joints']), ('frame', d['frames'])]:
        lines += [f'__host__ constexpr const char *{kind}_name(int id) {{', '  switch(id) {']
        for r in records:
            lines.append(f'  case {r["id"]}: return {string(r["name"])};')
        lines += ['  default: return nullptr;', '  }', '}']
    lines += ['// Thread-scope composition; distinct 16-element input/output, no implicit fences.',
              '// Failure leaves output untouched. Root attachment expects the root world pose.',
              'template<class T> __host__ __device__ inline bool compose_frame_from_body(',
              '    int frame_id, int body_id, const T *world_body, T *world_frame) {',
              '  frame_record f{};',
              '  if (runtime_transform || !world_body || !world_frame || world_body == world_frame',
              '      || !frame(frame_id, f) || f.attached_body != body_id) return false;',
              '  for (int c=0;c<4;++c) for (int r=0;r<4;++r) {',
              '    T value = T(0);',
              '    for (int k=0;k<4;++k) value += world_body[r+4*k]*T(f.T_body_frame[k+4*c]);',
              '    world_frame[r+4*c] = value;',
              '  }',
              '  return true;',
              '}',
              '} } // namespace model', '']
    return '\n'.join(lines)
