Algorithms
==========

GRiD implements several key rigid body dynamics algorithms

.. toctree::
    :maxdepth: 2

    inverse_dynamics
    aba
    crba
    minv
    frame_jacobian
    idsva
    fdsva_so

Algorithm Overview
------------------

Here's a quick overview of the main algorithms:

* **inverse_dynamics**: Recursive Newton-Euler Algorithm (RNEA).
* **crba**: Composite Rigid Body Algorithm (joint-space mass matrix).
* **aba**: Articulated Body Algorithm (forward dynamics).
* **minv**: Direct Inverse Mass Matrix.
* **Frame Jacobian**: general-frame geometric Jacobian :math:`J` for an
  arbitrary target frame in any of the three Pinocchio reference frames
  (``LOCAL`` / ``WORLD`` / ``LOCAL_WORLD_ALIGNED``), plus the
  reference-only Jacobian time-variation :math:`\dot J` and the
  operational-space (OSC) inertia
  :math:`\Lambda = (J M^{-1} J^{\top})^{-1}`.
* **IDSVA-SO**: Second-order Inverse Dynamics Spatial Vector Algorithm,
  with body-frame and world-frame variants and a codegen-time
  dispatcher (body-frame for fixed-base, world-frame for floating-base).
* **FDSVA-SO**: Second-order Forward Dynamics, layered on top of IDSVA-SO
  with a four-tier shared-memory selector for large floating-base
  robots.