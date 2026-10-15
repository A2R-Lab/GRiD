Python backend interfaces
=========================

This selected-method inventory is checked against the handle class definitions
by ``docs/test_doc_examples.py``. Yes means the Python method exists, not
that every robot, dtype, convention or derivative order is supported. It is
not a GPU validation matrix or a promise of arbitrary-order framework autodiff.

.. csv-table:: Selected public methods
   :header: "Method", "NumPy", "JAX", "PyTorch"
   :widths: 55, 15, 15, 15

   inverse_dynamics, Yes, Yes, Yes
   inverse_dynamics_gradient, Yes, Yes, Yes
   idsva_so, Yes, Yes, Yes
   minv, Yes, Yes, Yes
   crba, Yes, Yes, Yes
   aba, Yes, Yes, Yes
   forward_dynamics, Yes, Yes, Yes
   forward_dynamics_gradient, Yes, Yes, Yes
   fdsva_so, Yes, Yes, Yes
   inverse_dynamics_regressor, Yes, Yes, Yes
   inverse_dynamics_wrt_params, No, Yes, Yes
   forward_dynamics_wrt_params, No, Yes, Yes
   forward_dynamics_parameter_gradient, No, Yes, Yes
   end_effector_pose, Yes, Yes, Yes
   end_effector_pose_gradient, Yes, Yes, Yes
   end_effector_pose_hessian, Yes, Yes, Yes
   end_effector_pose_runtime, Yes, Yes, Yes
   end_effector_pose_gradient_runtime, Yes, Yes, Yes
   fk_batched, Yes, No, No
   frame_jacobian, Yes, Yes, Yes
   frame_jacobian_dot, Yes, Yes, Yes
   osc_inertia, Yes, Yes, Yes
   com, Yes, Yes, Yes
   ccrba, Yes, Yes, Yes
   dccrba, Yes, Yes, Yes
   cmm_time_variation, Yes, Yes, Yes
   integrator, Yes, Yes, Yes
   integrator_gradient, Yes, Yes, Yes
   plant_step, Yes, Yes, Yes
   plant_step_gradient, Yes, Yes, Yes
   plant_step_hessian, Yes, No, No
   quadratic_state_cost, Yes, Yes, Yes
   com_cost, Yes, Yes, Yes
   momentum_cost, Yes, Yes, Yes
   capture, No, No, Yes

How to interpret availability
-----------------------------

* An absent interface is a ``No`` above. For example, framework handles do not
  expose ``plant_step_hessian``. NumPy exposes a regressor, not the separate
  framework ``inverse_dynamics_wrt_params`` method.
* A method may exist but not be built: its operation was not selected for this
  robot artifact. Check the artifact's available operations as described in
  :doc:`python_wrappers`; a different method call cannot add generated code.
* An unsupported combination is one where a generator or binding rejects the requested
  joint type, integration scheme or convention. See :doc:`cuda_support_status`
  and the individual algorithm pages. An untested combination is not evidence
  of support or of a failure.
* A resource limit arises when generated code exceeds the target GPU's launch or
  memory limits for a particular robot or batch. Interface presence does not
  remove those limits.

Use the :doc:`release measurements <../../release_measurements>` and linked
receipts for the configurations actually measured or validated.

Joint and interface restrictions
--------------------------------

This is the one place these restrictions are listed; the other pages link
here. It is a coverage summary, not a guarantee for every combination of joint
type, algorithm, backend and GPU resource budget.

* Joint types: planar and translation joints are decomposed into
  scalar-joint chains by the parser; mimic joints retain dependent bodies but
  reduce coordinates. Mimic and spherical ``minv`` use a dense inverse of the
  reduced CRBA matrix rather than the scalar-joint direct-inverse recursion.
* NumPy-only methods: ``fk_batched`` is a NumPy-only, restricted
  single-leaf pose helper; it is not emitted for spherical models or models
  with more than 32 joints. ``plant_step_hessian`` is available on the NumPy
  handle and the CUDA interface, not the JAX/PyTorch handles; it supports
  Euler and semi-implicit Euler on fixed and floating bases, not multi-stage
  RK, and the artifact must include its ``fdsva_so`` dependency.
* Integrators: multi-stage integrator gradients are not supported for
  spherical joints or MuJoCo-output twins; MuJoCo-output integration supports
  Euler/semi-implicit Euler only.
* Costs: momentum costs use the full ``2*NV`` tangent-state gradient and
  Gauss–Newton Hessian, requiring ``dccrba``.
* Differentiation scope: first-order autograd (torch ``backward``,
  ``jax.grad``/``vjp``) runs the analytic gradient kernels via ``custom_vjp``
  / ``autograd.Function``; higher-order differentiation through them is not
  provided. Second-order tensors (``idsva_so``, ``fdsva_so``, the explicit
  Hessians) are values, not autograd-composable operators. ``*_wrt_params``
  and ``forward_dynamics_parameter_gradient`` are JAX/PyTorch interfaces (the
  NumPy handle exposes the regressor) and are local sensitivities of the
  compiled model, not parameterized functions. ``capture`` is PyTorch-only.

See :doc:`../concepts/algorithms/kinematics` and
:doc:`../concepts/algorithms/integrators_and_plant` for shapes and method
scope before selecting a method; :doc:`cuda_support_status` is the catalog of
generated CUDA paths.
