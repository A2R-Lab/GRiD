Integration, derivatives and model primitives
=============================================

This page specifies the native GRiD maps used when composing generated code.
The coordinate ownership and frame definitions in :doc:`model_coordinates`
remain authoritative. A configuration has ``nq`` stored entries and ``nv``
tangent entries; these widths need not agree.

Configuration and state maps
----------------------------

Write :math:`I(q,d)` for integration and :math:`D(a,b)` for difference.
The first argument of difference is the reference, so locally
:math:`I(a,D(a,b))=b`. Scalar independent coordinates use addition and
subtraction, including continuous joints; this is not a wrapped-angle metric.

A floating pose uses a right SE(3) increment, with local linear coordinates
before angular coordinates. For pose :math:`(p,R)` and increment
:math:`(\rho,\phi)`, translation becomes
:math:`p+R J_l(\phi)\rho`. A spherical joint uses a right SO(3) increment.
Quaternions are unit ``xyzw``. Difference uses the canonical rotation logarithm
and is invariant to quaternion sign. The rotation cut at pi is not smooth;
nonunit or nonfinite inputs are outside the numerical primitive contract.

For state :math:`x=(q,v)`, the native state retraction and difference are

.. math::

   R_x(\delta) &= (I(q,\delta_q), v+\delta_v),\\
   D_x(a,b) &= (D(a_q,b_q), b_v-a_v).

The velocity subtraction is a component convention, not physical transport
between body frames. Stored state width is ``nq+nv``; tangent state width is
``2*nv``. A different residual requires its own derivatives.

First derivatives and defect composition
-----------------------------------------------

Let :math:`y=F(x,u)` be a discrete step. Its native gradient is the derivative
at zero of :math:`D_x(y,F(R_x(\delta_x),u+\delta_u))`, holding nominal y
fixed. Logical rows are ``[dq_next, dv_next]`` and columns ``[dq, dv, du]``.
The torque plant has ``nu=nv`` and a ``2*nv`` by ``3*nv`` gradient; that control
width is not a universal restriction on consumer actuation.

Both argument Jacobians of configuration difference perturb the corresponding
input using I and differentiate the returned components. With :math:`e=D(a,b)`,

.. math::

   D_2(a,b) &= (\partial_d I(a,e))^{-1},\\
   D_1(a,b) &= -D_2(a,b)\partial_q I(a,e).

They reduce to +I and -I on scalar coordinates, but generally not at nonzero
rotational defects. The state versions append velocity blocks +I and -I.
For defect ``D_x(s,F(x,u))``, compose ``D_2(s,y)*G`` and use ``D_1(s,y)`` for
the successor. For ``D_x(F(x,u),s)``, use ``D_1(y,s)*G`` and ``D_2(y,s)``.
Do not replace these maps with a sign change at a nonzero manifold defect.
Actuation and contact-chain derivatives belong in G before this composition.

An invertible successor block permits row normalization of a linear constraint.
It does not redefine the nonlinear residual, its merit function or its dual
coordinates. With ``c=D_x(y,s)``, row-normalized blocks are
``A=-solve(D_2,D_1*Gx)``, ``B=-solve(D_2,D_1*Gu)`` and ``c_new=solve(D_2,c)``.
Hold D_2 fixed within that linearization. If the normalized covector is mu,
the original-residual covector is ``solve(D_2.T,mu)``. A solver owns this choice,
including warm starts, conditioning and higher-order terms.

What plant_step_hessian means
------------------------------------

The existing name denotes **moving-gradient sensitivity**:

.. math::

   H[o,a,b] = \left.\frac{d}{d\epsilon}
       G(R_x(\epsilon e_a),u)[o,b]\right|_{\epsilon=0}.

For a control direction, perturb u instead. Each evaluated G uses its own input
tangent basis and predicted-output anchor. The logical axes are output,
perturbation, gradient column. The native flat index is
``o*nz*nz + a*nz + b``, ``nz=3*nv``; native G is column-major.

On scalar Euclidean states this is an ordinary component Hessian. On a manifold
it can be asymmetric. It is **not** the symmetric Hessian in one fixed
input/output chart, nor the Hessian of a nonzero shooting defect. Symmetrization
does not derive the missing transport terms. Fixed-chart weighted curvature,
successor/mixed defect blocks and general second-order transport are not provided
by this API.

The native second-order step supports Euler and semi-implicit Euler on admitted
scalar, floating and fixed-spherical models. Spherical blocks use SO(3), even
with a fixed base. Multistage step Hessians remain unsupported. The NumPy/CUDA
second-order surface does not imply a JAX/Torch Hessian handle method. Existing
floating-plus-spherical and resource-tier restrictions remain separate; metadata
coverage alone does not establish plant support. The workspace binder in
:doc:`workspace_composition` has a narrower scope than the lower-level plant.

Thread-local generated primitives
---------------------------------

For namespace ``grid`` the configuration entry points are::

   configuration_difference<T>(from, to, delta);
   configuration_difference_jacobians<T>(from, to, J_from, J_to);
   bool configuration_difference_jacobian_block<T>(
       block_id, from, to, J_from, J_to, capacity);

Inputs are full independent configurations. Dense outputs contain ``nv`` and
two ``nv*nv`` values, with matrices column-major and off-block zeros written.
The compact form writes two width-by-width matrices. ``capacity`` counts T
elements in **each** output, not bytes. An invalid block or insufficient capacity
returns false with both outputs untouched. Empty coordinate outputs are not read
or written. ``model::coordinate_block_count`` and
``model::coordinate_block(id, record)`` expose kind, ``q_start``, ``v_start``,
``nq`` and ``nv`` from the descriptor's independent configuration partition.
Block IDs are not joint IDs; mimics add no independent block. Tangent widths
are 1, 3 and 6 for scalar, spherical and floating blocks.

These functions execute on one calling thread. They dynamically allocate no storage, launch
no kernels and impose no barrier. Small automatic blocks have dimensions at
most six, independent of nv. Caller-owned contiguous outputs must be disjoint
from inputs and each other; inputs may coincide. Complete writes do not publish
data to other threads: the caller supplies fences and workspace lifetime.
Dense convenience does not imply dense storage fits a consumer kernel. Compact
storage and actual enclosing-kernel resource admission must be checked together.
The helpers are emitted with integrator/plant derivative algorithm surfaces;
a lightweight kinematics header does not acquire plant code through them.

Local transforms use the descriptor's effective joint IDs::

   bool model::joint_transform<T>(joint_id, full_q, out16);
   bool model::joint_transform_scalar<T>(joint_id, coordinate, out16);

Both write all 16 column-major parent-from-child homogeneous entries from the
same resolved symbolic placement. The scalar input is an **absolute native
coordinate**, not an increment or a full configuration. It admits independent
revolute, continuous and prismatic joints only. Dependent/mimic, constant,
helical and multicoordinate joints use the general route. General transforms
read only the selected joint's necessary coordinates; a constant transform
requires no q read. Runtime IDs may differ across lanes. No constant-cell input,
FK traversal, arena, cached pose, mutable table or collective is involved.

Invalid IDs and unsupported scalar routes return false without output writes.
Both new routes refuse ``runtime_transform`` builds, whose live spatial dynamics
tables do not define the baked homogeneous placement. Existing transform APIs
are unchanged. Keep output disjoint from input, and retain caller publication
and scratch-reuse fences when replacing a legacy helper.

The scalar writer currently retains a device call boundary: the composed
divergent suffix regression exposed incorrect FP32 results with an inlined
writer on the tested toolchain. Standalone evaluation and memory sanitizers
did not reveal the failure. Account for compiled call-frame/local storage in
the enclosing caller; additive source APIs do not imply runtime cost parity.

Parameters, identity and mutation
----------------------------------------

Derivatives hold non-state/control parameters fixed. Mutability does not imply
derivatives with respect to a parameter.

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Category
     - Examples
     - Ownership and lifetime
   * - Structural model
     - Topology, joint kinds/axes, mimic, baked placements and physical data
     - Regenerate/rebuild. Descriptor identity excludes inertia and is not
       complete artifact identity.
   * - Structural generation
     - Algorithms, dtype, namespace, targets, table flags, backends/toolchain
     - New compatible artifact and prepared association. Respect module-revision
       process isolation.
   * - Per-call inputs
     - q/v/u, dt, gravity, exposed external wrenches, cost targets/weights
     - Operation-specific signatures govern exposure; hold buffers through use.
       A consumer may compile dt/gravity into its artifact.
   * - Enabled mutable tables
     - Inertia, joint dynamics, spatial transforms
     - Loaded-module state shared by backend views. Quiesce and order readers
       before mutation; separate handles do not isolate tables.
   * - Execution configuration
     - Stream, threads, launch resources, caller workspace
     - Caller owns admission, ordering, publication and lifetime.

``set_transform_params`` changes spatial dynamics transforms, not baked
homogeneous EE kinematics. Joint-dynamics setter defaults can restore baked
values rather than retain the previous mutation; sign friction is nonsmooth at
zero. Prepared loading does not snapshot live parameter state: reapply supported
configuration before evaluation. Physics changes also require review of cached
derivatives, costates and warm starts. No universal runtime parameter buffer,
actuation provider or heterogeneous per-sample model is implied.
