"""Thread-local model operations; symbolic transforms and descriptor ownership.

No traversal, arena, solver, mutable table or independently parsed model.
"""
import sympy as sp

from .algorithms._eepose_gradient_hessian import (
    _ee_pose_inner_angle_expr, _ee_pose_root_fb_subst)
from .helpers._topology_helpers import _xi_spherical_quat_subst
from .helpers._expr_printer import expr_T


def emit_joint_transforms(gen):
    """joint_transform reads s_q; joint_transform_scalar takes one coordinate value.
    Both emit the parser's homogeneous joint matrices."""
    lines = [f'namespace {gen.file_namespace} {{ namespace model {{',
             '// Thread scope, all 16 column-major cells written; disjoint output.',
             '// Invalid IDs/modes leave output untouched. Unit xyzw inputs required.']
    mats = gen.robot.get_Xmats_hom_ordered_by_id(include_fixed_joints=False)
    for scalar in (False, True):
        name = 'joint_transform_scalar' if scalar else 'joint_transform'
        arg = 'T coordinate' if scalar else 'const T *s_q'
        lines += ['template<class T> __device__ __forceinline__ bool '+name+'(',
                  f'    int joint_id, {arg}, T *out) {{',
                  '  if (runtime_transform) return false;', '  switch(joint_id) {']
        for j in gen.model_descriptor['joints']:
            jid = j['id']
            if scalar and (not j['owns_coordinates'] or j['kind'] not in
                           ('revolute', 'continuous', 'prismatic')):
                continue
            lines.append(f'  case {jid}: {{')
            if j['local_nq'] == 1:
                angle = 'coordinate' if scalar else _ee_pose_inner_angle_expr(gen, jid)
                lines += [f'    const T theta={angle};',
                          '    const T s=sin(theta), c=cos(theta);', '    (void)s; (void)c;']
            for col in range(4):
                for row in range(4):
                    expression = expr_T(mats[jid][row, col])
                    if j['kind'] == 'floating':
                        expression = _ee_pose_root_fb_subst(expression)
                    elif j['kind'] == 'spherical':
                        expression = _xi_spherical_quat_subst(gen, expression, jid)
                    else:
                        expression = expression.replace('sin(theta)', 's').replace('cos(theta)', 'c')
                    lines.append(f'    out[{row+4*col}]=static_cast<T>({expression});')
            lines += ['    break;', '  }']
        lines += ['  default: return false;', '  }', '  return true;', '}']
    lines += ['} } // namespace model', '']
    return '\n'.join(lines)


def emit_configuration_primitives(descriptor, namespace):
    """Compact/dense forms share GLASS Lie algebra and one coordinate partition.

    Everything is computed in T.
    Matrix temporaries are at most 6x6, independent of model dimensions.
    """
    blocks = descriptor['configuration_layout']
    lines = [f'namespace {namespace} {{', 'namespace model {',
             'enum class coordinate_kind { euclidean, spherical, floating };',
             'struct coordinate_block_record { coordinate_kind kind; int q_start, v_start, nq, nv; };',
             f'inline constexpr int coordinate_block_count={len(blocks)};',
             '__host__ __device__ constexpr bool coordinate_block(int id, coordinate_block_record &out) {',
             '  switch(id) {']
    for index, b in enumerate(blocks):
        fields = ','.join(str(b[k]) for k in ('q_start', 'v_start', 'nq', 'nv'))
        lines.append(f'  case {index}: out=coordinate_block_record{{coordinate_kind::{b["kind"]},{fields}}}; return true;')
    lines += ['  default: return false;', '  }', '}', '} // namespace model']
    lines += r'''
namespace configuration_detail {
// Reference-first differences; independent quaternion sign representatives.
template<class T> __device__ inline void block_difference(
    const model::coordinate_block_record &b, const T *from, const T *to, T *e) {
    if(b.kind==model::coordinate_kind::euclidean) { e[0]=to[b.q_start]-from[b.q_start]; return; }
    const T *a=from+b.q_start, *z=to+b.q_start;
    if(b.kind==model::coordinate_kind::floating) {
        glass::thread::se3_difference<T>(a,z,e,e+3);
    } else {
        T conjugate[4],relative[4];
        glass::thread::quat_conj<T>(a,conjugate);
        glass::thread::quat_mul<T>(conjugate,z,relative);
        glass::thread::quat_log<T>(relative,e);
    }
}
// One nonlinear evaluation body prevents caller-dependent FMA contraction
// from making dense and compact representations disagree in their last bits.
template<class T> __device__ __noinline__ void nonlinear_block_jacobians(
    const model::coordinate_block_record &b, const T *from, const T *to, T *J1, T *J2) {
    T e[6]; block_difference(b,from,to,e);
    if(b.kind==model::coordinate_kind::spherical) {
        T A[9],B[9];
        glass::thread::so3_left_jacobian_inv<T>(e,A);
        glass::thread::so3_right_jacobian_inv<T>(e,B);
        for(int k=0;k<9;++k) { J1[k]=-A[k]; J2[k]=B[k]; }
        return;
    }
    // Invert the upper-triangular SE(3) Jr: [A,Q;0,A]^-1.
    T A[9],Q[9],AQ[9],C[9],inverse[36],base[36];
    glass::thread::so3_right_jacobian_inv<T>(e+3,A);
    glass::thread::se3_Q_block<T>(e,e+3,Q);
    glass::thread::gemm<T,3,3,3>(T(1),A,Q,T(0),AQ);
    glass::thread::gemm<T,3,3,3>(T(-1),AQ,A,T(0),C);
    for(int c=0;c<6;++c) for(int r=0;r<6;++r)
        inverse[c*6+r]=(r<3 && c<3)?A[c*3+r]:
            ((r>=3 && c>=3)?A[(c-3)*3+r-3]:((r<3 && c>=3)?C[(c-3)*3+r]:T(0)));
    glass::thread::se3_retract_jacobian_q<T>(e,e+3,base);
    for(int c=0;c<6;++c) for(int r=0;r<6;++r) {
        T v=T(0); for(int k=0;k<6;++k) v-=inverse[k*6+r]*base[c*6+k];
        J1[c*6+r]=v; J2[c*6+r]=inverse[c*6+r];
    }
}
template<class T> __device__ inline void block_jacobians(
    const model::coordinate_block_record &b, const T *from, const T *to, T *J1, T *J2) {
    if(b.kind==model::coordinate_kind::euclidean) { J1[0]=T(-1); J2[0]=T(1); return; }
    nonlinear_block_jacobians(b,from,to,J1,J2);
}
} // namespace configuration_detail
// Capacity is the number of T elements in EACH disjoint output, not bytes.
// False leaves both outputs untouched. Inputs may coincide; valid unit quaternions required.
template<class T> __device__ inline bool configuration_difference_jacobian_block(
    int block_id, const T *from, const T *to, T *J_from, T *J_to, size_t capacity) {
    model::coordinate_block_record b{};
    if(!model::coordinate_block(block_id,b) || capacity<size_t(b.nv*b.nv)) return false;
    configuration_detail::block_jacobians(b,from,to,J_from,J_to);
    return true;
}
// Thread scope; complete writes, no allocation, launch, mutation or collective.
template<class T> __device__ inline void configuration_difference(
    const T *from, const T *to, T *delta) {
    for(int k=0;k<model::coordinate_block_count;++k) {
        model::coordinate_block_record b{}; model::coordinate_block(k,b);
        T e[6]; configuration_detail::block_difference(b,from,to,e);
        for(int i=0;i<b.nv;++i) delta[b.v_start+i]=e[i];
    }
}
template<class T> __device__ inline void configuration_difference_jacobians(
    const T *from, const T *to, T *J_from, T *J_to) {
    for(int k=0;k<model::nv*model::nv;++k) { J_from[k]=T(0); J_to[k]=T(0); }
    for(int k=0;k<model::coordinate_block_count;++k) {
        model::coordinate_block_record b{}; model::coordinate_block(k,b);
        T A[36],B[36]; configuration_detail::block_jacobians(b,from,to,A,B);
        for(int c=0;c<b.nv;++c) for(int r=0;r<b.nv;++r) {
            int i=(b.v_start+c)*model::nv+b.v_start+r;
            J_from[i]=A[c*b.nv+r]; J_to[i]=B[c*b.nv+r];
        }
    }
}
'''.strip().splitlines()
    lines += [f'}} // namespace {namespace}', '']
    return '\n'.join(lines)
