Library Overview
=================

Each submodule of GRiD is an essential component to getting the most out of GRiD as a whole. Here we will discuss how each module relates to each other. 

Table of Contents
-----------------
I. `RBDReference`
II. `URDFParser`
III. `GRiDCodeGenerator`

I. RBDReference 
----------------

RBDReference is composed of ``RBDReference.py`` plus a set of topic mixins
(``_energy.py``, ``_centroidal.py``, ``_regressor.py``, ``_plant.py``) which
together host a class of functions responsible for the easy-to-read rigid body
dynamics algorithms in Python.

It currently supports the following algorithmic functions which can be viewed from the function glossary:

* ``apply_external_forces`` and an ``f_ext=`` kwarg on ``inverse_dynamics`` / ``inverse_dynamics_fpass`` / ``aba`` — opt-in per-body external forces (body-local frame, subtracted from the per-body force; an empty/``None`` value is a no-op)
* ``inverse_dynamics`` (RNEA / Recursive Newton-Euler Algorithm)
* ``inverse_dynamics_gradient``
* ``minv``
* ``aba``
* ``crba``
* ``forward_dynamics_gradient``

In addition, the mixins provide numpy reference oracles validated against
Pinocchio:

* Energy / forces (``_energy.py``): ``generalized_gravity``, ``nonlinear_effects``, ``kinetic_energy``, ``potential_energy``, ``mechanical_energy``, ``coriolis_matrix``, ``kinetic_energy_regressor``, ``potential_energy_regressor``
* Centroidal (``_centroidal.py``): ``com``, ``jacobian_com``, ``ccrba``, ``centroidal_momentum``, ``dccrba`` (analytic ∂A/∂q tensor — replaces the prior finite-difference oracle), ``cmm_time_variation`` (Ȧ)
* Regressor (``_regressor.py``): ``inverse_dynamics_regressor``
* Plant / costs / barriers (``_plant.py``): ``plant_step`` (+ ``plant_step_gradient`` / ``plant_step_hessian``), ``quadratic_state_cost``, ``quadratic_input_cost``, ``ee_pos_cost``, ``com_cost``, ``momentum_cost``, and the joint position/velocity/torque log-barriers — the reference for the generated ``grid_plant`` CUDA layer

Each of these functions and more included within the file call upon getters from URDFParser which initializes a convenient ``robotObj``.
Here is a list of relevant and helpful getters which can also be viewed from the function glossary for ``URDFParser``:

* ``self.robot.get_num_bodies()``
* ``self.robot.get_parent_id()``
* ``self.robot.get_joint_index_q()``
* ``self.robot.get_Xmat_Func_by_id()()`` 
* ``self.robot.get_Imat_by_id()``
* ``self.robot.get_subtree_by_id()``
* ``self.robot.get_num_vel()``
* ``self.robot.get_S_by_id()``

This is just a short list, please look to the function glossary for ``URDFParser`` for more detailed usage instructions and guidelines.

II. URDFParser 
---------------


III. GRiDCodeGenerator
-----------------------

GRiDCodeGenerator emits the per-robot ``grid.cuh``. Beyond the core dynamics
and kinematics algorithms (and their analytical gradients), recent additions
include:

* **External forces (``f_ext``):** an optional ``T *d_f_ext`` argument on RNEA,
  forward dynamics, ABA, and the inverse-/forward-dynamics gradients. It is
  GLOBAL, body-major (``6*NUM_BODIES``), body-local-frame, and subtracted from
  the per-body force; passing ``nullptr`` (the default) reproduces the
  no-force path byte-for-byte.
* **``grid_plant`` layer:** a sibling ``namespace grid_plant { ... }`` emitted
  after the ``grid`` namespace, providing ``plant_step`` (+ gradient and the
  F1 fixed-base ``plant_step_hessian``), quadratic state/input costs,
  end-effector-position / CoM / centroidal-momentum costs (Gauss-Newton
  Hessian), and joint position/velocity/torque log-barriers.
* **Resource tiers:** every emitted kernel and inline-CUDA ``_device`` /
  ``_inner`` surface takes a ``RESOURCE_TIER`` template parameter defaulting to
  ``TIER_SHARED`` (the old ``TIER_PERF`` alias has been removed). See
  :doc:`../concepts/resource_tier_system`.
* **Mimic joints:** non-gradient algorithms support mimic robots, and **every**
  gradient codegen now folds correctly to the reduced coordinates on both bases —
  ``inverse_dynamics_gradient`` / ``forward_dynamics_gradient``,
  ``end_effector_pose_gradient`` / ``end_effector_pose_hessian``, the second-order
  ``idsva_so`` / ``fdsva_so``, the external-force gradients (``f_ext_gradient``),
  and the integrator gradients. No mimic gradient raises ``NotImplementedError``
  anymore. The centroidal kinematics family (``com`` / ``ccrba`` / ``energy``)
  and the centroidal derivatives (``dccrba`` / ``cmm_time_variation``) are now
  mimic-reduced too: the per-body world Jacobian and per-unit motion columns
  carry the mimic multiplier (α), validated against the mimic-aware
  RBDReference oracle on fixed-base mimic robots. ``dccrba`` /
  ``cmm_time_variation`` additionally emit on big floating-base robots
  (e.g. ``g1`` / ``h1_2``-floating) via the sweep-pool spill path, so
  per-robot gating of the centroidal family is now essentially eliminated.
* **Coriolis matrix + energy regressors:** ``coriolis_matrix`` ``C(q,q̇)``
  (with ``C·q̇ + g = nonlinear_effects``) and the kinetic / potential
  inertial-parameter energy regressors (each length ``10·NB``), all
  CUDA-validated against the RBDReference oracle.
* **Runtime arbitrary multi-EE:** ``end_effector_pose_runtime`` and
  ``end_effector_pose_gradient_runtime`` take the target joint id and a
  per-target offset as RUNTIME arguments (rather than codegen-baked), so one
  compiled robot serves any leaf/target frame.
* **Runtime-mutable inertial parameters (flag-gated):** an opt-in
  ``d_inertia_params`` table plus a ``set_inertia_params`` device entry let
  sysID / domain-randomization mutate the per-link inertial parameters with no
  recompile; the baked default path is byte-identical.
* **Joint types (stage 1):** an arbitrary/skew ``<axis>`` (non-cardinal) is
  supported via a dense 6-vector motion subspace ``S`` — currently for
  ``inverse_dynamics`` and ``crba`` only (cardinal-axis robots stay
  byte-identical). Other algorithms and the helical / planar / spherical joint
  types are later stages.

