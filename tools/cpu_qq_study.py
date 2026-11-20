"""CPU-only local-frame RNEA q-Hessian experiment; never a product kernel.

Spatial transforms/subspaces/inertias are rounded oracle inputs. Scalar joints
and constant mimic folds are supported. Floating root derivatives use the
existing ordered Lie-tangent convention, not a symmetric Euclidean Hessian.
All arithmetic in a candidate uses one dtype; float64 is an identity referee.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np


def cross(v):
    out = np.zeros(v.shape[:-1] + (6, 6), dtype=v.dtype)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        out[..., a, b] = -v[..., c]
        out[..., b, a] = v[..., c]
        out[..., a + 3, b + 3] = -v[..., c]
        out[..., b + 3, a + 3] = v[..., c]
        out[..., a + 3, b] = -v[..., c + 3]
        out[..., b + 3, a] = v[..., c + 3]
    return out


def add_terms(terms, order):
    terms = list(terms)
    if order == 'reverse':
        terms.reverse()
    elif order != 'forward':
        raise ValueError('unknown summation order')
    result = terms[0].copy()
    for term in terms[1:]:
        result += term
    return result


def local_qq(model, q, qd, qdd, dtype=np.float32, order='forward'):
    """Differentiate local RNEA twice with explicit first/second derivative jets."""
    robot, ref, n = model.robot, model.reference, model.nv
    if any(record['multiplier'] is None for record in robot.coupling_records()):
        raise ValueError('nonlinear coupling is outside this CPU prototype')
    q = np.array(q, dtype=np.float64)  # only the spatial-input oracle uses float64
    if robot.floating_base and robot.using_quaternion:
        q[3:7] = ref._normalize_xyzw_quaternion(q[3:7])
    qd, qdd = np.asarray(qd, dtype=dtype), np.asarray(qdd, dtype=dtype)

    def constant(value):
        value = np.asarray(value, dtype=dtype)
        return (value, np.zeros((n,) + value.shape, dtype=dtype),
                np.zeros((n, n) + value.shape, dtype=dtype))

    def plus(*jets):
        return tuple(add_terms((j[k] for j in jets), order) for k in range(3))

    def mv(matrix, vector):
        return np.matmul(matrix, vector[..., None])[..., 0]

    def apply(x, y):
        return (mv(x[0], y[0]),
                add_terms((mv(x[1], y[0]), mv(x[0], y[1])), order),
                add_terms((mv(x[2], y[0]), mv(x[1][:, None], y[1][None, :]),
                           mv(x[1][None, :], y[1][:, None]), mv(x[0], y[2])), order))

    def crm(jet, dual=False):
        values = tuple(cross(x) for x in jet)
        return tuple(-x.swapaxes(-1, -2) for x in values) if dual else values

    bodies = []
    for i in range(robot.get_num_bodies()):
        parent = robot.get_parent_id(i)
        vi = ref._as_index_list(robot.get_joint_index_v(i))
        s = np.asarray(robot.get_S_by_id(i), dtype=dtype).reshape(6, len(vi))
        alpha = dtype(ref._mimic_multiplier(i))
        x = constant(robot.get_Xmat_Func_by_id(i)(robot.q_for_joint(i, q)))
        inertia = constant(robot.get_Imat_by_id(i))
        vj = constant(s @ (alpha * qd[vi]))
        aj = constant(s @ (alpha * qdd[vi]))
        if parent < 0:
            if not robot.floating_base or len(vi) != 6:
                raise ValueError('this study requires a six-DoF floating root')
            gravity = np.zeros(6, dtype=dtype); gravity[5] = dtype(9.81)
            g = constant(np.linalg.solve(x[0], gravity))
            generators = [dtype(-1 if j < 3 else 1) * cross(s[:, j]) for j in range(6)]
            for j, vj_index in enumerate(vi):
                g[1][vj_index] = -generators[j] @ g[0]
                for k, vk_index in enumerate(vi):
                    g[2][vj_index, vk_index] = generators[j] @ generators[k] @ g[0]
            v, a = vj, plus(g, aj)
        else:
            if len(vi) != 1:
                raise ValueError('non-root multi-DoF joints need a separate derivative rule')
            generator = -alpha * cross(s[:, 0])
            x[1][vi[0]] = generator @ x[0]
            x[2][vi[0], vi[0]] = generator @ x[1][vi[0]]
            v = plus(apply(x, bodies[parent]['v']), vj)
            a = plus(apply(x, bodies[parent]['a']), aj, apply(crm(v), vj))
        f = plus(apply(inertia, a), apply(crm(v, dual=True), apply(inertia, v)))
        bodies.append(dict(parent=parent, vi=vi, s=s, alpha=alpha, x=x, v=v, a=a, f=f))

    result = np.zeros((n, n, n), dtype=dtype)
    # Project after all children have contributed. Reverse only the sibling
    # contribution order, never the dependency order of the tree.
    for i in reversed(range(len(bodies))):
        b = bodies[i]
        children = [j for j, d in enumerate(bodies) if d['parent'] == i]
        children.sort(reverse=(order == 'forward'))
        contributions = [b['f']]
        for child in children:
            d = bodies[child]
            contributions.append(apply(tuple(x.swapaxes(-1, -2) for x in d['x']), d['f']))
        b['f'] = plus(*contributions)
        projected = b['alpha'] * (b['f'][2] @ b['s']).transpose(2, 0, 1)
        for local, slot in enumerate(b['vi']):
            result[slot] += projected[local]
    assert result.dtype == dtype
    return result


class FloatNumpy:
    """Isolated adapter for reference-method CPU arithmetic, never global patching."""
    def __init__(self, dtype):
        self.dtype = dtype

    def __getattr__(self, name):
        original = getattr(np, name)
        if name not in {'array', 'asarray', 'zeros', 'ones', 'empty', 'eye'}:
            return original
        def call(*args, **kwargs):
            kwargs['dtype'] = self.dtype
            return original(*args, **kwargs)
        return call


def world_qq(model, q, qd, qdd, dtype=np.float32, reassociate=False):
    """Emulate the existing world algorithm on CPU, not CUDA instruction order.

    Uses the exact installed method source with dtype-controlled constructors
    and rounded robot spatial inputs. Checks all surviving local float arrays.
    """
    ref = model.reference
    # Match local_qq's spatial-input oracle exactly. Letting the isolated
    # reference normalize q in float32 would otherwise confound frame
    # arithmetic with transform-input precision in this comparison.
    spatial_q = np.array(q, dtype=np.float64)
    if model.robot.floating_base and model.robot.using_quaternion:
        spatial_q[3:7] = ref._normalize_xyzw_quaternion(spatial_q[3:7])
    spatial_x = [np.asarray(model.robot.get_Xmat_Func_by_id(i)(model.robot.q_for_joint(i, spatial_q)), dtype=dtype)
                 for i in range(model.robot.get_num_bodies())]
    fn = type(ref).idsva_so_world_frame
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    if reassociate:
        class Regroup(ast.NodeTransformer):
            def visit_BinOp(self, node):
                # Algebraically identical matrix-chain parenthesization and
                # reverse addition order; no precision change or kernel edit.
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    terms = []
                    def collect(expr, negative=False):
                        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
                            collect(expr.left, negative)
                            collect(expr.right, negative ^ isinstance(expr.op, ast.Sub))
                        else:
                            value = self.visit(expr)
                            terms.append(ast.UnaryOp(op=ast.USub(), operand=value) if negative else value)
                    collect(node)
                    return ast.Call(func=ast.Name(id='_reverse_sum', ctx=ast.Load()), args=terms, keywords=[])
                node = self.generic_visit(node)
                if isinstance(node.op, ast.MatMult) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.MatMult):
                    return ast.BinOp(left=node.left.left, op=ast.MatMult(),
                                     right=ast.BinOp(left=node.left.right, op=ast.MatMult(), right=node.right))
                return node
        tree = Regroup().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            node.value = ast.Call(func=ast.Name(id='_checked_return', ctx=ast.Load()),
                                  args=[node.value, ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])
    ast.fix_missing_locations(tree)
    def check(value, locals_):
        def arrays(v):
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                assert v.dtype == dtype, v.dtype
            elif isinstance(v, (tuple, list)):
                for item in v: arrays(item)
        for v in locals_.values(): arrays(v)
        arrays(value)
        return value
    def reverse_sum(*terms):
        result = terms[-1]
        for term in reversed(terms[:-1]): result = result + term
        return result
    scope = dict(fn.__globals__, np=FloatNumpy(dtype), _checked_return=check, _reverse_sum=reverse_sum)
    exec(compile(tree, '<isolated-world-float-study>', 'exec'), scope)

    class Robot:
        def __getattr__(self, name):
            original = getattr(model.robot, name)
            if name == 'get_Xmat_Func_by_id':
                return lambda i: lambda arg: spatial_x[i].copy()
            if name in {'get_Imat_by_id', 'get_S_by_id'}:
                return lambda i: np.asarray(original(i), dtype=dtype)
            return original

    class Reference:
        robot = Robot()
        def __getattr__(self, name): return getattr(ref, name)
        def cross_operator(self, v): return cross(v)
        def dual_cross_operator(self, v): return -cross(v).T
        def icrf(self, v): return np.asarray(ref.icrf(v), dtype=dtype)  # signs/rearrangement only
        def dot_matrix(self, inertia, v): return -cross(v).T @ inertia - inertia @ cross(v)

    values = scope[fn.__name__](Reference(), np.asarray(q, dtype=dtype),
                               np.asarray(qd, dtype=dtype), np.asarray(qdd, dtype=dtype))
    return values[0]
