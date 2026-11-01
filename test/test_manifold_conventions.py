"""Independent value/derivative checks for the native manifold convention.

Tolerances are fixed before implementation: double precision value FD uses a
fourth-order stencil at 2e-4; analytic retract derivatives use a 2e-8 bound.
These are CPU correctness tests, not a provider or solver implementation.
"""
from pathlib import Path

import numpy as np
import pytest
from RBDReference import RBDReference
from URDFParser import URDFParser

from test.cuda_equivalents.manifold_oracle import FIX, derivative, reference, state


@pytest.mark.parametrize('angle', [0., 1e-9, 1e-8, 3e-8, 1e-7, 1e-6, 1e-4, 1e-3])
def test_small_angle_so3_against_matrix_exponential(angle):
    # Independent convergent power series of the Lie algebra, not the
    # cancellation-prone trigonometric coefficients used by the implementation.
    phi = np.array([1., -2., 3.]) * (angle / np.sqrt(14.))
    S = RBDReference._so3_skew(phi)
    power = np.eye(3)
    exp = power.copy(); left = power.copy(); right = power.copy()
    factorial = 1.
    for k in range(1, 14):
        power = power @ S; factorial *= k
        exp += power / factorial
        left += power / (factorial*(k+1))
        right += (-1)**k * power / (factorial*(k+1))
    np.testing.assert_allclose(RBDReference._so3_exp(phi), exp, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(RBDReference._so3_V_matrix(phi), left, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(RBDReference._so3_right_jacobian(phi), right, atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize('name,floating', [('spherical_arm', False),
    ('mixed_spherical_arm', False), ('mixed_spherical_arm', True),
    ('rotated_inertial_arm', False), ('rotated_inertial_arm', True)])
@pytest.mark.parametrize('arg', ['q', 'v'])
def test_d2integrate_is_derivative_of_admitted_retract(name, floating, arg):
    ref = reference(name, floating)
    q, w = state(ref)
    expected = derivative(lambda e: ref.dIntegrate(q, w+e, arg), len(w))
    actual = ref.d2Integrate(q, w, arg, 'v')
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    np.testing.assert_array_equal(ref.d2Integrate(q, w, arg, 'q'), np.zeros_like(actual))


@pytest.mark.parametrize('name,floating', [('mixed_spherical_arm', False),
    ('mixed_spherical_arm', True), ('rotated_inertial_arm', False),
    ('rotated_inertial_arm', True)])
def test_difference_both_arguments_and_sign_controls(name, floating):
    ref = reference(name, floating)
    a, w = state(ref)
    b = ref.integrate(a, w)
    nv = len(w)
    for arg in ('from', 'to'):
        expected = derivative(lambda e: ref.difference(ref.integrate(a, e), b)
            if arg == 'from' else ref.difference(a, ref.integrate(b, e)), nv)
        actual = ref.dDifference(a, b, arg)
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
        assert np.max(np.abs(actual + expected)) > 1
        if ref._joint_retract_specs():
            assert np.max(np.abs(actual - actual.T)) > .01
            assert np.max(np.abs(actual - (1 if arg == 'to' else -1)*np.eye(nv))) > .01
    flipped = b.copy()
    for _, iq, _ in ref._joint_retract_specs():
        flipped[iq[-4:]] *= -1
    np.testing.assert_allclose(ref.difference(a, flipped), w, atol=2e-14)


@pytest.mark.parametrize('name', ['spherical_arm', 'mixed_spherical_arm'])
@pytest.mark.parametrize('scheme', ['euler', 'semi_implicit_euler'])
def test_spherical_plant_hessian_matches_moving_gradient(name, scheme):
    ref = reference(name)
    q, v = state(ref)
    u = np.linspace(-.4, .7, len(v))
    nv = len(v)
    def gradient(z):
        return ref.plant_step_gradient(ref.integrate(q, z[:nv]), v+z[nv:2*nv],
                                      u+z[2*nv:], .02, integrator_type=scheme)
    expected = derivative(gradient, 3*nv).transpose(0, 2, 1)
    actual = ref.plant_step_hessian(q, v, u, .02, integrator_type=scheme)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


def test_moving_hessian_is_not_fixed_chart_curvature():
    ref=reference('spherical_arm');q,v=state(ref);nv=len(v)
    u=np.linspace(-.2,.3,nv);dt=.02
    y=ref.plant_step(q,v,u,dt,'euler');nq=len(q)
    weight=np.linspace(.2,1.,2*nv)
    def scalar(z):
        other=ref.plant_step(ref.integrate(q,z[:nv]),v+z[nv:2*nv],u+z[2*nv:],dt,'euler')
        return weight@np.r_[ref.difference(y[:nq],other[:nq]),other[nq:]-y[nq:]]
    H=np.einsum('o,oab->ab',weight,ref.plant_step_hessian(q,v,u,dt,'euler'))
    # Two different spherical axes: moving q/v blocks are not symmetric.
    a,b=0,nv+1;h=3e-4;da=np.eye(3*nv)[a]*h;db=np.eye(3*nv)[b]*h
    fixed=(scalar(da+db)-scalar(da-db)-scalar(-da+db)+scalar(-da-db))/(4*h*h)
    assert abs(H[a,b]-H[b,a])>1e-3
    assert max(abs(fixed-H[a,b]),abs(fixed-H[b,a]))>5e-4
    # Swapping the perturbations in a fixed-chart scalar second difference is
    # symmetric; this does not license symmetrizing the moving-G API tensor.
    reversed=(scalar(db+da)-scalar(db-da)-scalar(-db+da)+scalar(-db-da))/(4*h*h)
    np.testing.assert_allclose(fixed,reversed,atol=2e-12,rtol=2e-12)


def test_unequal_dimension_analytic_fixture_axes_and_contractions():
    z = np.array([.3, -.4, .7, .2, -.6])  # x in R3, u in R2
    def value(d):
        a,b,c,p,q = z+d
        return np.array([a+b*p+q*q, b+c*q+p*p, c+np.sin(a)+p*q])
    a,b,c,p,q = z
    G = np.array([[1,p,0,b,2*q], [0,1,q,2*p,c], [np.cos(a),0,1,q,p]])
    H = np.zeros((3,5,5))
    H[0,1,3] = H[0,3,1] = 1
    H[0,4,4] = H[1,3,3] = 2
    H[1,2,4] = H[1,4,2] = H[2,3,4] = H[2,4,3] = 1
    H[2,0,0] = -np.sin(a)
    np.testing.assert_allclose(G, derivative(value, 5), atol=1e-11, rtol=1e-11)
    h = 1e-3
    Hfd = derivative(lambda d: derivative(lambda e: value(d+e), 5, h), 5, h)
    np.testing.assert_allclose(H, Hfd, atol=2e-9, rtol=2e-9)
    weights = np.array([.2,-.7,1.3])
    np.testing.assert_allclose(np.einsum('o,oab->ab', weights, H),
        derivative(lambda d: derivative(lambda e: weights@value(d+e), 5, h), 5, h),
        atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('name', ['spherical_arm', 'mixed_spherical_arm'])
def test_spherical_pinocchio_anchor(name):
    import pinocchio as pin
    from RBDReference.tests.test_spherical_joint_equivalence import _build_pin_spherical, _build_pin_mixed
    model = (_build_pin_spherical if name == 'spherical_arm' else _build_pin_mixed)()
    data = model.createData()
    ref = reference(name)
    q, v = state(ref)
    nv = len(v)
    u = np.linspace(-.4, .7, nv)
    b = pin.integrate(model, q, v)
    for arg, flag in [('from', pin.ARG0), ('to', pin.ARG1)]:
        np.testing.assert_allclose(ref.dDifference(q, b, arg), pin.dDifference(model, q, b, flag),
                                   atol=2e-11, rtol=2e-11)
    for arg, flag in [('q', pin.ARG0), ('v', pin.ARG1)]:
        fd = derivative(lambda e: np.array(pin.dIntegrate(model, q, v+e, flag)), nv)
        np.testing.assert_allclose(ref.d2Integrate(q, v, arg, 'v'), fd, atol=2e-8, rtol=2e-8)
    dt = .02
    for scheme in ('euler', 'semi_implicit_euler'):
        def pin_gradient(z):
            qp = pin.integrate(model, q, z[:nv]); vp=v+z[nv:2*nv]; up=u+z[2*nv:]
            acc = np.array(pin.aba(model, data, qp, vp, up), copy=True)
            dq,dv,du = [np.array(x, copy=True) for x in pin.computeABADerivatives(model, data, qp, vp, up)]
            V = np.hstack([dt*dq, np.eye(nv)+dt*dv, dt*du])
            w = dt*(vp+dt*acc if scheme=='semi_implicit_euler' else vp)
            A,B = pin.dIntegrate(model, qp, w)
            Q = np.hstack([A,np.zeros((nv,2*nv))])
            Q += B@(dt*V if scheme=='semi_implicit_euler' else np.hstack([np.zeros((nv,nv)),dt*np.eye(nv),np.zeros((nv,nv))]))
            return np.vstack([Q,V])
        np.testing.assert_allclose(ref.plant_step_gradient(q,v,u,dt,scheme), pin_gradient(np.zeros(3*nv)),atol=2e-10,rtol=2e-10)
        expected=derivative(pin_gradient,3*nv).transpose(0,2,1)
        np.testing.assert_allclose(ref.plant_step_hessian(q,v,u,dt,scheme),expected,atol=2e-7,rtol=2e-7)


@pytest.mark.parametrize('reverse', [False, True])
def test_floating_nonzero_defect_value_fd_and_row_normalization(reverse):
    # Fixture has positive inertials, unlike the kinematics-only rotated arm.
    ref=reference('mixed_spherical_arm',True)
    q,v=state(ref);nv=len(v);u=np.linspace(-.4,.7,nv);dt=.02
    def step(z):
        return ref.plant_step(ref.integrate(q,z[:nv]),v+z[nv:2*nv],u+z[2*nv:],dt,'euler')
    nq=len(q);y=step(np.zeros(3*nv));delta=np.linspace(-.35,.4,nv)
    s=np.r_[ref.integrate(y[:nq],delta),y[nq:]+.2]
    def difference(a,b): return np.r_[ref.difference(a[:nq],b[:nq]),b[nq:]-a[nq:]]
    def jacobian(a,b,arg):
        J=np.zeros((2*nv,2*nv));J[:nv,:nv]=ref.dDifference(a[:nq],b[:nq],arg)
        J[nv:,nv:]=(1 if arg=='to' else -1)*np.eye(nv)
        return J
    a,b=(s,y) if reverse else (y,s)
    Jpred=jacobian(a,b,'to' if reverse else 'from')
    Jsucc=jacobian(a,b,'from' if reverse else 'to')
    G=ref.plant_step_gradient(q,v,u,dt,'euler')
    pred_fd=derivative(lambda z: difference(s,step(z)) if reverse else difference(step(z),s),3*nv)
    succ_fd=derivative(lambda z: difference(np.r_[ref.integrate(s[:nq],z[:nv]),s[nq:]+z[nv:]],y)
        if reverse else difference(y,np.r_[ref.integrate(s[:nq],z[:nv]),s[nq:]+z[nv:]]),2*nv)
    np.testing.assert_allclose(Jpred@G,pred_fd,atol=3e-7,rtol=3e-7)
    np.testing.assert_allclose(Jsucc,succ_fd,atol=3e-8,rtol=3e-8)
    assert np.max(abs(Jpred@G-(1 if reverse else -1)*G))>.01
    rng=np.random.default_rng(722)
    dz=rng.normal(size=3*nv);ds=rng.normal(size=2*nv);mu=rng.normal(size=2*nv)
    c=difference(a,b);raw=c+Jpred@G@dz+Jsucc@ds
    normalized=np.linalg.solve(Jsucc,c)+np.linalg.solve(Jsucc,Jpred@G)@dz+ds
    np.testing.assert_allclose(Jsucc@normalized,raw,atol=2e-13,rtol=2e-13)
    original_covector=np.linalg.solve(Jsucc.T,mu)
    np.testing.assert_allclose(mu@normalized,original_covector@raw,atol=2e-13,rtol=2e-13)
    assert abs(mu@raw-mu@normalized)>.001  # omit covector map: negative control
