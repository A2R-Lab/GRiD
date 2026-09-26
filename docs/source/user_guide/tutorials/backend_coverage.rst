Python backend interfaces
=========================

This selected-method inventory is checked against the handle class definitions
by ``docs/test_doc_examples.py``. **Yes means the Python method exists**, not
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

* **Absent interface:** a ``No`` above. For example, framework handles do not
  expose ``plant_step_hessian``. NumPy exposes a regressor, not the separate
  framework ``inverse_dynamics_wrt_params`` method.
* **Not built:** a method exists, but its operation was not selected for this
  robot artifact. Check the artifact's available operations as described in
  :doc:`python_wrappers`; a different method call cannot add generated code.
* **Unsupported combination:** a generator or binding rejects the requested
  joint type, integration scheme or convention. See :doc:`cuda_support_status`
  and the individual algorithm pages. An untested combination is not evidence
  of support or of a failure.
* **Resource limited:** generated code may exceed the target GPU's launch or
  memory limits for a particular robot or batch. Interface presence does not
  remove those limits.

Important examples: ``fk_batched`` is a restricted first-leaf pose helper;
plant step Hessians support Euler/semi-implicit Euler, not multi-stage RK;
and momentum-cost derivatives omit configuration blocks. See
:doc:`../concepts/algorithms/kinematics` and
:doc:`../concepts/algorithms/integrators_and_plant` before selecting a method.
Use the :doc:`release measurements <../../release_measurements>` and linked
receipts for the configurations actually measured or validated.
