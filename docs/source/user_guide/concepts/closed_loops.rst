Closed loops by constraint embedding
====================================

GRiD models a closed kinematic loop by embedding it: the bodies of the loop
stay in the spanning tree, the dependent joint coordinates become explicit
functions of the independent ones, and the tree recursions run on the
reduced coordinates. A follower joint whose coordinate is a function of a
leader's is a one-body cluster with the motion subspace
:math:`S(y) = X\,\mathrm{diag}(S_i)\,G(y)` of the constraint-embedding
literature (Jain; Chignoli and Wensing); the URDF mimic joint is the case
of a constant :math:`G`, and GRiD's mimic fold has always been that
embedding. Multipliers, proximal iterations and soft equality constraints
are not used: the tree recursions, the branch parallelism, the per-body
threads and the derivative column parallelism are unchanged, and no
constraint force appears.

The projection
--------------

With :math:`q = F(y)` the full-tree coordinates of the independent ones,
:math:`G = \partial F / \partial y` and
:math:`\dot q = G \dot y`, :math:`\ddot q = G \ddot y + \dot G \dot y`,
every reduced quantity is the projection of the full tree's:

.. math::

   \tau_y = G^{\top} \tau(F(y), G\dot y, G\ddot y + \dot G \dot y), \qquad
   M_y = G^{\top} M G, \qquad
   c_y = G^{\top}\,\mathrm{rnea}(F(y), G\dot y, \dot G \dot y),

and the derivatives follow by the chain rule (:math:`\partial G / \partial y`
carries the second derivative of :math:`F`). The numpy reference
implements exactly this projection (``RBDReference/_embedding.py``) on the
full spanning tree, which ``URDFParser.parse(..., reduce_couplings=False)``
returns; on a mimic robot it agrees with the reduced recursions to
round-off, and it is the oracle the generated kernels are checked against.

Explicit coupling kinds
-----------------------

* **Linear** (the URDF ``<mimic joint multiplier offset>``): a constant
  multiplier. Every algorithm supports it; the fold is baked into the
  generated code and the headers of such robots are unchanged by the
  machinery below.
* **Polynomial** (``<mimic joint="l" polycoef="c0 c1 c2 c3 c4" q0="..."/>``,
  the MuJoCo joint equality): :math:`q = \sum_i c_i (q_l - q_0)^i` of degree
  at most four. The fold multiplier :math:`\alpha(q_l) = dq/dq_l` and its
  derivatives are evaluated on the device from the leader's coordinate;
  the follower accelerates by :math:`\alpha' \dot q_l^{\,2}` on top of
  :math:`\alpha \ddot q_l`, and the gradients carry the
  :math:`\alpha'`, :math:`\alpha''` and :math:`(\partial G/\partial q)^{\top} f`
  terms. See :doc:`/api_reference/urdf` for the element.

Planar four-bar closures and loops without a closed form (the ``<loop>``
element of URDF+, the ``connect`` and ``weld`` equalities of MJCF) are the
next cluster kinds; they close the loop numerically inside the kernel and
differentiate it by the implicit function theorem.

What a polynomial robot can generate today
------------------------------------------

The generator emits, for a robot with a polynomial follower: the inverse
dynamics, CRBA, :math:`M^{-1}`, forward dynamics and ABA and both first
order gradients; the end-effector pose and its gradient, also with runtime
targets; the frame Jacobian family (``J``, ``J̇``, the task inertia, the
frame velocity and acceleration and their derivatives); the six
integrators with their ``[A | B]`` gradients; the joint-torque regressor,
the energy regressors and the forward-dynamics parameter gradient; the
generalized gravity, the bias, the CoM, the centroidal momentum matrix and
the energies. Each is checked on the GPU against the projection on a
fixed and a floating-base chain (``test/cuda_equivalents/test_cuda_polynomial_coupling.py``).

Every other algorithm still bakes a constant multiplier into its fold and
``gen_all_code`` refuses it for such a robot with a message that names it:
the Coriolis matrix (whose reduced form carries :math:`G^{\top} M \dot G`),
the momentum-matrix derivatives, the external-force gradients, the pose
Hessian and the multi-target kernels, the regressor gradient, the
second-order derivatives, the plant layer and the joint damping and
friction folds. ``register_robot`` therefore takes an explicit
``algorithm_list`` for a polynomial robot
(``test/python_wrappers/test_polynomial_coupling_handle.py``).
