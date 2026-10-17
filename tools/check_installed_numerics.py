"""Installed-artifact numerical acceptance matrix, also run in the signed suite.

Use an installed wheel environment from outside the checkout:
    python -I tools/check_installed_numerics.py --work DIR --urdf-dir DIR
URDFs are supplied by the caller, never fetched. Failures are recorded, never
waived. Uses FP64 bundled RBDReference and existing float32 error bounds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import grid_rbd
import grid_codegen
from grid_codegen import resources
from URDFParser import URDFParser
from RBDReference import RBDReference


def run(work, urdf_dir, robot_cases=(('iiwa14',False),('go2',True)),
        *, require_installed=True, cache_dir=None):
    WORK = Path(work).resolve()
    WORK.mkdir(parents=True,exist_ok=True)
    urdf_dir = Path(urdf_dir).resolve()
    if require_installed:
        assert resources.checkout_root() is None, 'Requires an installed artifact'
        assert Path(grid_rbd.__file__).resolve().parent.parent == Path(grid_codegen.__file__).resolve().parent.parent
    checks = []

    def compare(label, actual, expected, tol=5e-4):
        a, e = np.asarray(actual, float), np.asarray(expected, float)
        assert a.shape == e.shape, (label, a.shape, e.shape)
        assert np.isfinite(a).all() and np.isfinite(e).all(), label
        err = float(np.max(np.abs(a-e)) / max(1., np.max(np.abs(e))))
        passed = err < tol
        checks.append(dict(label=label, scaled_max_error=err, tolerance=tol, passed=passed))
        (WORK/'extended-results.json').write_text(json.dumps(checks,indent=2)+'\n')
        print('PASS' if passed else 'FAIL', label, 'scaled max error', err, 'limit',tol,flush=True)

    for robot_name, floating in robot_cases:
        urdf = str(urdf_dir / (robot_name + '.urdf'))
        algorithms = ['inverse_dynamics', 'forward_dynamics', 'aba', 'crba', 'minv',
                      'inverse_dynamics_gradient', 'forward_dynamics_gradient', 'end_effector_pose']
        if not floating:
            algorithms += ['idsva_so_body_frame', 'fdsva_so', 'integrator',
                           'end_effector_pose_gradient', 'end_effector_pose_hessian']
        options = dict(urdf_path=urdf, floating_base=floating, max_batch_size=32,
                       algorithm_list=algorithms, enable_mujoco_kernels=False,
                       cache_dir=str(cache_dir or WORK / 'extended-cache'))
        name = 'installed_' + robot_name
        print('REGISTER', name, algorithms, flush=True)
        h = grid_rbd.load_robot(name=name, **options)
        assert (h.nq, h.nv) == ((19,18) if floating else (7,7))
        ref = RBDReference(URDFParser().parse(urdf, floating_base=floating))
        rng = np.random.default_rng(1710)
        for batch in (1, 8, 32, 2):
            q = rng.uniform(-.4,.4,(batch,h.nq)).astype(np.float32)
            if floating:
                q[:,3:7] /= np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
            v = rng.uniform(-.3,.3,(batch,h.nv)).astype(np.float32)
            a = rng.uniform(-.5,.5,(batch,h.nv)).astype(np.float32)
            u = rng.uniform(-1,1,(batch,h.nv)).astype(np.float32)
            cases = {
                'id': (h.inverse_dynamics(q,v,a), lambda i: ref.inverse_dynamics(q[i].astype(float),v[i].astype(float),a[i].astype(float))[0]),
                'bias': (h.inverse_dynamics(q,v), lambda i: ref.inverse_dynamics(q[i].astype(float),v[i].astype(float))[0]),
                'fd': (h.forward_dynamics(q,v,u), lambda i: ref.forward_dynamics(q[i].astype(float),v[i].astype(float),u[i].astype(float))),
                'aba': (h.aba(q,v,u), lambda i: ref.aba(q[i].astype(float),v[i].astype(float),u[i].astype(float))),
                'crba': (h.crba(q), lambda i: ref.crba(q[i].astype(float))),
                'minv': (h.minv(q), lambda i: ref.minv(q[i].astype(float))),
                'idgrad': (h.inverse_dynamics_gradient(q,v,a), lambda i: ref.inverse_dynamics_gradient(q[i].astype(float),v[i].astype(float),a[i].astype(float))),
                'fdgrad': (h.forward_dynamics_gradient(q,v,u), lambda i: np.hstack(ref.forward_dynamics_gradient(q[i].astype(float),v[i].astype(float),u[i].astype(float)))),
            }
            for op,(actual,oracle) in cases.items():
                compare(f'{name}/B{batch}/{op}', actual, np.stack([oracle(i) for i in range(batch)]))
            compare(f'{name}/B{batch}/ID-FD roundtrip', h.forward_dynamics(q,v,cases['id'][0]), a)
            compare(f'{name}/B{batch}/zero acceleration default', cases['bias'][0], h.inverse_dynamics(q,v,np.zeros_like(v)), tol=1e-6)
            buf = h.pinned_empty((batch,2*h.nv*h.nv))
            assert h.is_pinned(buf)
            compare(f'{name}/B{batch}/pinned gradient out', h.inverse_dynamics_gradient(q,v,a,out=buf), cases['idgrad'][0], tol=1e-6)
        with h.context() as isolated:
            compare(f'{name}/isolated-context',isolated.inverse_dynamics(q,v,a), h.inverse_dynamics(q,v,a),tol=1e-6)
        fe=rng.uniform(-.1,.1,(len(q),6*h.nb)).astype(np.float32)
        force_actual=h.inverse_dynamics(q,v,a,f_ext=fe)
        force_expected=np.stack([
            ref.inverse_dynamics(q[i].astype(float),v[i].astype(float),a[i].astype(float),f_ext=fe[i].reshape(h.nb,6))[0]
            for i in range(len(q))])
        compare(f'{name}/external-forces-ID',force_actual,force_expected)
        compare(f'{name}/noncontiguous-cast-FD',h.forward_dynamics(np.asfortranarray(q),v.astype(float),u),h.forward_dynamics(q,v,u),tol=1e-6)
        # Native differentiable backends on real URDFs, same build/cache.
        import jax
        import jax.numpy as jnp
        import torch
        jh = grid_rbd.register_robot(name, backend='jax', **options)
        th = grid_rbd.register_robot(name, backend='torch', **options)
        # Every operand must vary independently across samples. Identical force
        # rows mask a missing batch offset even with arbitrary q/qd/u.
        for pattern, forces in [('distinct', fe), ('repeated', np.repeat(fe[:1], len(fe), axis=0)), ('zero', np.zeros_like(fe))]:
            for op in ('inverse_dynamics', 'forward_dynamics', 'aba', 'inverse_dynamics_gradient', 'forward_dynamics_gradient'):
                for bias in ((False, True) if op.startswith('inverse_dynamics') else (False,)):
                    third = None if bias else a
                    expected = []
                    for i in range(len(q)):
                        row = getattr(ref,op)(q[i].astype(float), v[i].astype(float),
                            None if bias else a[i].astype(float), f_ext=forces[i].reshape(h.nb,6))
                        if op == 'inverse_dynamics': row = row[0]
                        if op == 'forward_dynamics_gradient': row = np.hstack(row)
                        expected.append(row)
                    expected = np.stack(expected)
                    for backend, handle in [('numpy',h), ('jax',jh), ('torch',th)]:
                        convert = {'numpy': lambda x:x, 'jax': jnp.asarray,
                                   'torch': lambda x:torch.tensor(x,device='cuda')}[backend]
                        def call(order):
                            values = getattr(handle,op)(convert(q[order]),convert(v[order]),
                                None if bias else convert(third[order]),f_ext=convert(forces[order]))
                            return values.detach().cpu().numpy() if backend=='torch' else np.asarray(values)
                        label = f'{name}/force-{pattern}/{backend}/{op}/bias={bias}'
                        compare(label,call(slice(None)),expected)
                        if pattern=='distinct':
                            compare(label+'/permutation',call(np.array([1,0])),expected[[1,0]])
                            singles=np.concatenate([call(slice(i,i+1)) for i in range(len(q))])
                            compare(label+'/single-calls',singles,expected)
        for op,third in [('inverse_dynamics',a), ('forward_dynamics',u)]:
            expected = getattr(h,op)(q,v,third)
            jf = jax.jit(lambda x,y,z: getattr(jh,op)(x,y,z))
            compare(f'{name}/{op}/jax-jit', np.asarray(jf(jnp.asarray(q),jnp.asarray(v),jnp.asarray(third))), expected)
            tt = [torch.tensor(x,device='cuda',requires_grad=True) for x in (q,v,third)]
            tv = getattr(th,op)(*tt)
            compare(f'{name}/{op}/torch', tv.detach().cpu().numpy(), expected)
            tv.sum().backward()
            jg = jax.jit(jax.grad(lambda x,y,z: getattr(jh,op)(x,y,z).sum(), argnums=(0,1,2)))(jnp.asarray(q),jnp.asarray(v),jnp.asarray(third))
            differentiated = range(2 if op=='inverse_dynamics' else 3)
            if op=='inverse_dynamics':
                assert tt[2].grad is None  # documented: qdd is held constant
            for k in differentiated:
                compare(f'{name}/{op}/autograd-input{k}', np.asarray(jg[k]), tt[k].grad.cpu().numpy())
            # Independent central differences of the FP64 bundled oracle, including
            # quaternion coordinates for floating q (normalizing extension).
            def objective(x,y,z):
                out = getattr(ref,op)(x,y,z)
                return np.sum(out[0] if op=='inverse_dynamics' else out)
            base = [q[0].astype(float),v[0].astype(float),third[0].astype(float)]
            for k in differentiated:
                fd = np.zeros_like(base[k]); eps=1e-5
                for col in range(len(fd)):
                    plus=[x.copy() for x in base]; minus=[x.copy() for x in base]
                    plus[k][col]+=eps; minus[k][col]-=eps
                    fd[col]=(objective(*plus)-objective(*minus))/(2*eps)
                compare(f'{name}/{op}/autograd-oracle-input{k}', np.asarray(jg[k])[0], fd)
        # Backward paths must retain the matching force row too. Compare BOTH
        # samples independently, not merely the summed gradient or backend parity.
        for op, third in [('inverse_dynamics', a), ('forward_dynamics', u)]:
            differentiated = tuple(range(2 if op == 'inverse_dynamics' else 3))
            jf = jax.jit(jax.grad(
                lambda x, y, z: getattr(jh, op)(x, y, z, f_ext=jnp.asarray(fe)).sum(),
                argnums=differentiated))
            jg = jf(jnp.asarray(q), jnp.asarray(v), jnp.asarray(third))
            tensors = [torch.tensor(x, device='cuda', requires_grad=True) for x in (q, v, third)]
            getattr(th, op)(*tensors, f_ext=torch.tensor(fe, device='cuda')).sum().backward()
            for sample in range(len(q)):
                base = [q[sample].astype(float), v[sample].astype(float), third[sample].astype(float)]
                def force_objective(args):
                    value = getattr(ref, op)(*args, f_ext=fe[sample].reshape(h.nb, 6))
                    return np.sum(value[0] if op == 'inverse_dynamics' else value)
                for k in differentiated:
                    numeric = np.zeros_like(base[k])
                    for col in range(len(numeric)):
                        plus, minus = [x.copy() for x in base], [x.copy() for x in base]
                        plus[k][col] += 1e-5
                        minus[k][col] -= 1e-5
                        numeric[col] = (force_objective(plus) - force_objective(minus)) / 2e-5
                    for backend, gradient in [('jax', np.asarray(jg[k])),
                                              ('torch', tensors[k].grad.cpu().numpy())]:
                        compare(f'{name}/{op}/force-autograd/{backend}/row{sample}/input{k}',
                                gradient[sample], numeric)
        if not floating:
            for op in ('idsva_so','fdsva_so'):
                arg = a if op=='idsva_so' else u
                actual = getattr(h,op)(q,v,arg)
                expected = [getattr(ref,op)(q[i].astype(float),v[i].astype(float),arg[i].astype(float)) for i in range(len(q))]
                for block in range(4):
                    compare(f'{name}/{op}/block{block}', actual[block], np.stack([e[block] for e in expected]))
            for method in ('euler','semi_implicit_euler','constant_acceleration','midpoint','trapezoidal','rk4'):
                actual=h.integrator(q,v,u,.001,integrator_type=method)
                expected=np.stack([ref.integrator(q[i].astype(float),v[i].astype(float),u[i].astype(float),.001,integrator_type=method) for i in range(len(q))])
                compare(f'{name}/integrator/{method}',actual,expected)
            compare(f'{name}/pose', h.end_effector_pose(q),np.stack([ref.end_effector_pose(x.astype(float))[0].flatten() for x in q]))
            compare(f'{name}/pose-gradient',h.end_effector_pose_gradient(q), np.stack([ref.end_effector_pose_gradient(x.astype(float))[0] for x in q]))
            hess=h.end_effector_pose_hessian(q)
            assert hess.shape==(len(q),6*h.num_ees,h.nv,h.nv) and np.isfinite(hess).all()
            eps=1e-5
            numeric=np.empty_like(hess,dtype=float)
            for i,x in enumerate(q):
                for k in range(h.nv):
                    dx=np.zeros(h.nq);dx[k]=eps
                    numeric[i,:,:,k]=(ref.end_effector_pose_gradient(x.astype(float)+dx)[0]-ref.end_effector_pose_gradient(x.astype(float)-dx)[0])/(2*eps)
            compare(f'{name}/pose-hessian-oracle',hess,numeric)
        try:
            h.inverse_dynamics(q,v[:,:-1],a)
        except ValueError:
            print('PASS invalid velocity width rejected',name,flush=True)
        else:
            raise AssertionError('Bad velocity shape accepted')
        try:
            h.pinned_empty((1,h.nv),dtype=np.float64)
        except ValueError:
            print('PASS incompatible pinned dtype rejected',name,flush=True)
        else:
            raise AssertionError('Wrong pinned dtype accepted')
        (WORK/'extended-results.json').write_text(json.dumps(checks,indent=2)+'\n')
    failed=[x for x in checks if not x['passed']]
    print('Extended installed tests:',len(checks),'numerical comparisons;',len(failed),'FAILURES',flush=True)
    assert not failed, f"{len(failed)} numerical comparisons failed: {failed}"
    return checks

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--urdf-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.work,args.urdf_dir)
