"""Experimental CPU local-frame float32 dynamics chain, not a product emitter.

No reference dynamics values enter the candidate. Model constants and symbolic
transform formulas come from the pinned parser. Transform expressions, mimic
coordinates, RNEA jets, CRBA derivatives, factorization and contraction use the
selected dtype. Float64 is available only as a separate identity referee.
Scope: quaternion floating root, scalar joints, constant reduced mimic folds,
constant armature, no external forces. Ordered root Lie derivatives are retained.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from functools import lru_cache

import numpy as np
import scipy.linalg as sla

from tools.cpu_qq_study import cross, add_terms


def indices(value):
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]


@lru_cache(maxsize=512)
def typed_transform(function, dtype):
    """Recompile trusted lambdified parser expressions with typed constants.

    Each arithmetic result is checked, not merely rounded at the output. This
    evaluates the symbolic formula in FP32; it does not emulate CUDA FMA/libm.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
               ast.Return, ast.List, ast.Tuple, ast.Name, ast.Load, ast.Store,
               ast.Constant, ast.Call, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError('unsupported transform expression in CPU study')
    functions = {'array', 'sin', 'cos', 'sqrt', 'tan'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                          or node.func.id not in functions):
            raise ValueError('unsupported transform function in CPU study')
    class Typed(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                return ast.Call(func=ast.Name(id='_dtype', ctx=ast.Load()), args=[node], keywords=[])
            raise ValueError('non-numeric transform constant')
        def visit_BinOp(self, node):
            return ast.Call(func=ast.Name(id='_check', ctx=ast.Load()),
                            args=[self.generic_visit(node)], keywords=[])
        visit_UnaryOp = visit_BinOp
    def check(value):
        if np.asarray(value).dtype != dtype:
            raise TypeError(f'promoted transform intermediate: {np.asarray(value).dtype}')
        return value
    scope = {name: getattr(np, name) for name in functions}
    scope.update(_dtype=dtype, _check=check,
                 array=lambda values: np.array(values, dtype=dtype))
    tree = ast.fix_missing_locations(Typed().visit(tree))
    exec(compile(tree, '<typed-parser-transform>', 'exec'), scope)
    return scope[function.__name__]


def tree_data(model, q, dtype=np.float32):
    robot = model.robot
    if not robot.floating_base or not robot.using_quaternion:
        raise ValueError('study requires a quaternion floating root')
    if any(r['multiplier'] is None for r in robot.coupling_records()):
        raise ValueError('nonlinear coupling is outside the study')
    q = np.array(q, dtype=dtype)
    norm = np.sqrt(np.sum(q[3:7] * q[3:7], dtype=dtype))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('invalid root quaternion')
    q[3:7] /= norm
    data = []
    for i in range(robot.get_num_bodies()):
        joint = robot.get_joint_by_id(i)
        vi = indices(robot.get_joint_index_v(i))
        qi = indices(robot.get_joint_index_q(i))
        alpha, offset = dtype(1), dtype(0)
        if getattr(joint, 'is_mimic', False):
            alpha, offset = dtype(joint.get_mimic_multiplier()), dtype(joint.get_mimic_offset())
        local = alpha * q[qi] + offset
        if local.size == 1: local = local[0]
        x = typed_transform(robot.get_Xmat_Func_by_id(i), dtype)(local)
        parent = robot.get_parent_id(i)
        if (parent < 0 and len(vi) != 6) or (parent >= 0 and len(vi) != 1):
            raise ValueError('unsupported joint dimension')
        s = np.asarray(robot.get_S_by_id(i), dtype=dtype).reshape(6, len(vi))
        dx = np.zeros_like(x) if parent < 0 else (-alpha * cross(s[:, 0])) @ x
        ddx = np.zeros_like(x) if parent < 0 else (-alpha * cross(s[:, 0])) @ dx
        data.append(dict(parent=parent, vi=vi, x=x, dx=dx, ddx=ddx, s=s,
                         alpha=alpha, inertia=np.asarray(robot.get_Imat_by_id(i), dtype=dtype)))
    return data


def mass_derivative(data, armature, n, dtype=np.float32):
    """CRBA and its local tangent derivative, including reduced mimic folds."""
    ic = [d['inertia'].copy() for d in data]
    dic = [np.zeros((n, 6, 6), dtype=dtype) for d in data]
    dxs = []
    for d in data:
        dx = np.zeros((n, 6, 6), dtype=dtype)
        if d['parent'] >= 0: dx[d['vi'][0]] = d['dx']
        dxs.append(dx)
    for i in reversed(range(len(data))):
        d = data[i]; p = d['parent']
        if p < 0: continue
        x, dx = d['x'], dxs[i]
        dic[p] += dx.swapaxes(-1, -2) @ ic[i] @ x + x.T @ dic[i] @ x + x.T @ ic[i] @ dx
        ic[p] += (x.T @ ic[i]) @ x
    m = np.zeros((n, n), dtype=dtype)
    dm = np.zeros((n, n, n), dtype=dtype)
    for i, d in enumerate(data):
        vi, ai = d['vi'], d['alpha']
        force, df = ic[i] @ d['s'], dic[i] @ d['s']
        m[np.ix_(vi, vi)] += ai * ai * (d['s'].T @ force)
        dm[np.ix_(vi, vi, range(n))] += (ai * ai * (d['s'].T @ df)).transpose(1, 2, 0)
        j = i
        while data[j]['parent'] >= 0:
            dj = data[j]
            df = dxs[j].swapaxes(-1, -2) @ force + dj['x'].T @ df
            force = dj['x'].T @ force
            j = dj['parent']; ancestor = data[j]
            block = (ai * ancestor['alpha']) * (ancestor['s'].T @ force)
            db = (ai * ancestor['alpha']) * (ancestor['s'].T @ df)
            m[np.ix_(ancestor['vi'], vi)] += block
            m[np.ix_(vi, ancestor['vi'])] += block.T
            dm[np.ix_(ancestor['vi'], vi, range(n))] += db.transpose(1, 2, 0)
            dm[np.ix_(vi, ancestor['vi'], range(n))] += db.transpose(2, 1, 0)
    m[np.diag_indices(n)] += np.asarray(armature, dtype=dtype)
    return m, dm


def rnea_jets(data, qd, qdd, armature, dtype=np.float32, order='forward'):
    """RNEA value, q/v gradient and q/v Hessian at the supplied acceleration.

    Transform jets are sparse: do not allocate a dense 6x6 Hessian per body.
    Vector jets remain a CPU correctness model, not an approved scratch layout.
    """
    qd, qdd = np.asarray(qd, dtype=dtype), np.asarray(qdd, dtype=dtype)
    n = len(qd); width = 2*n
    def constant(v):
        v = np.asarray(v, dtype=dtype)
        return [v.copy(), np.zeros((width, 6), dtype=dtype),
                np.zeros((width, width, 6), dtype=dtype)]
    def plus(*jets):
        return [add_terms([j[k] for j in jets], order) for k in range(3)]
    def mv(m, v): return (m @ v[..., None])[..., 0]
    def transform(d, y, transpose=False):
        x, dx, ddx = [d[key].T if transpose else d[key] for key in ('x', 'dx', 'ddx')]
        out = [mv(x, v) for v in y]
        k = d['vi'][0]
        out[1][k] += dx @ y[0]
        out[2][k, :] += mv(dx, y[1])
        out[2][:, k] += mv(dx, y[1])
        out[2][k, k] += ddx @ y[0]
        return out
    def product(a, b, dual=False):
        def op(v, w):
            mat = cross(v)
            if dual: mat = -mat.swapaxes(-1, -2)
            return mv(mat, w)
        return [op(a[0], b[0]),
                add_terms([op(a[1], b[0]), op(a[0], b[1])], order),
                add_terms([op(a[2], b[0]), op(a[1][:, None], b[1][None, :]),
                           op(a[1][None, :], b[1][:, None]), op(a[0], b[2])], order)]
    bodies = []
    for i, d in enumerate(data):
        vi, s, alpha = d['vi'], d['s'], d['alpha']
        vj = constant(s @ (alpha * qd[vi]))
        vj[1][n + np.asarray(vi)] = alpha * s.T
        aj = constant(s @ (alpha * qdd[vi]))
        p = d['parent']
        if p < 0:
            gravity = np.zeros(6, dtype=dtype); gravity[5] = dtype(9.81)
            g = constant(np.linalg.solve(d['x'], gravity))
            generators = [dtype(-1 if j < 3 else 1) * cross(s[:, j]) for j in range(6)]
            for j, slot in enumerate(vi):
                g[1][slot] = -generators[j] @ g[0]
                for k, slot2 in enumerate(vi):
                    g[2][slot, slot2] = generators[j] @ generators[k] @ g[0]
            v, a = vj, plus(g, aj)
        else:
            v = plus(transform(d, bodies[p]['v']), vj)
            a = plus(transform(d, bodies[p]['a']), aj, product(v, vj))
        inertia = d['inertia']
        f = plus([mv(inertia, part) for part in a],
                 product(v, [mv(inertia, part) for part in v], dual=True))
        bodies.append(dict(v=v, a=a, f=f))
    tau = np.zeros(n, dtype=dtype)
    grad = np.zeros((n, width), dtype=dtype)
    hess = np.zeros((n, width, width), dtype=dtype)
    for i in reversed(range(len(data))):
        children = [j for j, d in enumerate(data) if d['parent'] == i]
        children.sort(reverse=order == 'forward')
        f = plus(bodies[i]['f'], *[transform(data[j], bodies[j]['f'], True) for j in children])
        bodies[i]['f'] = f
        d = data[i]
        projected = [d['alpha'] * (part @ d['s']) for part in f]
        for j, slot in enumerate(d['vi']):
            tau[slot] += projected[0][j]
            grad[slot] += projected[1][:, j]
            hess[slot] += projected[2][:, :, j]
    tau += np.asarray(armature, dtype=dtype) * qdd
    assert all(v.dtype == dtype for v in (tau, grad, hess))
    return tau, grad, (hess[:, :n, :n], hess[:, n:, n:], hess[:, n:, :n])


def factor(matrix):
    dtype = matrix.dtype.type
    scale = dtype(1) / np.sqrt(np.diag(matrix))
    c = sla.cho_factor((matrix * scale[:, None]) * scale[None, :], lower=True)
    def solve(rhs):
        out = scale[:, None] * sla.cho_solve(c, scale[:, None] * rhs)
        assert out.dtype == dtype
        return out
    return solve


def contract(solve, grad, tensors, dtype=np.float32):
    qq, vv, vq, dm = tensors
    n = len(grad); dq, dv = grad[:, :n], grad[:, n:]
    cols = solve(np.eye(n, dtype=dtype))
    term = np.einsum('ilk,lj->ijk', dm, dq)
    def apply(v): return -solve(v.reshape(n, -1)).reshape(v.shape)
    return (apply(qq + term + term.transpose(0, 2, 1)),
            apply(vq + np.einsum('ilk,lj->ijk', dm, dv)), apply(vv),
            apply(np.einsum('ilk,lj->ijk', dm, cols)))


def chain(model, q, qd, u, dtype=np.float32, order='forward'):
    """Entire candidate path; reference used only for constant armature data."""
    if getattr(model.reference, 'use_joint_dynamics', False):
        raise ValueError('joint-dynamics bias is outside this CPU study')
    data = tree_data(model, q, dtype)
    n = model.nv
    armature = np.asarray(model.reference.armature, dtype=dtype)
    matrix, dm = mass_derivative(data, armature, n, dtype)
    solve = factor(matrix)
    bias = rnea_jets(data, qd, np.zeros(n, dtype=dtype), armature, dtype, order)[0]
    qdd = solve((np.asarray(u, dtype=dtype) - bias)[:, None])[:, 0]
    tau, dc, tensors = rnea_jets(data, qd, qdd, armature, dtype, order)
    grad = -solve(dc)
    out = contract(solve, grad, (*tensors, dm), dtype)
    assert all(v.dtype == dtype for v in (matrix, dm, bias, qdd, tau, dc, grad, *out))
    return dict(matrix=matrix, dm=dm, bias=bias, qdd=qdd, tau=tau, dc=dc,
                grad=grad, tensors=tensors, output=out)
