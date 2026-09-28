Library Overview
=================

GRiD combines its own robot-specific CUDA generator and Python bindings with
three peer libraries. URDFParser, RBDReference and GLASS are Git submodules
under ``external/``. GRiD's code generator is in ``grid_codegen/`` in this
repository, not a separate submodule; the bindings are in ``bindings/``.

.. contents::
   :local:
   :depth: 1

.. _id1:

I. RBDReference
--------------------

RBDReference supplies NumPy reference algorithms for dynamics, kinematics,
derivatives, state integration and optimization costs. Its equivalence tests
compare against Pinocchio and other numerical checks. Generated CUDA uses
these CPU implementations as validation references, not runtime dependencies.

See the :doc:`RBDReference API <../../api_reference/rbd>` for a runnable
example, method families and state conventions.

.. _id2:

II. URDFParser
--------------

URDFParser builds the robot model consumed by the reference and generator:
joint ordering, motion subspaces, spatial inertias, transforms and limits.

* The default ``pinocchio_order`` uses depth-first ordering with Pinocchio's
  sibling sorting.
* ``floating_base=True`` adds a free-flyer root. Its default configuration is
  ``[x, y, z, qx, qy, qz, qw]`` and tangent ordering is ``[linear; angular]``.
* ``strict_inertial=True`` rejects missing or degenerate inertials on real
  moving bodies, with exemptions for root/base and dummy links.
* Planar and translation joints are decomposed into scalar joints; spherical
  joints retain quaternion configurations; mimic joints reduce independent
  coordinates while retaining their bodies. Closed kinematic loops are unsupported.

See the :doc:`parser tutorial <../tutorials/urdf_parser>` and
:doc:`parser API <../../api_reference/urdf>` for joint support, dimensions,
getters and errors.

.. _id3:

III. GRiD's code generator and bindings
------------------------------------------

The generator emits ``grid.cuh`` and derives wrapper entry points from a
shared ABI specification. It specializes algorithms to a robot's topology
and provides resource tiers for shared-memory and global-workspace use.

``grid_rbd`` exposes the generated computations through NumPy, JAX and
PyTorch. Robot registration selects algorithms, compiles an architecture-specific
artifact and caches it. Runtime contexts hold model parameters, buffers and
streams; supported model updates do not require regenerating the robot.

* :doc:`Generate CUDA <../tutorials/codegen>` or
  :doc:`call GRiD from Python <../tutorials/python_wrappers>`.
* :doc:`Explore algorithms <../concepts/algorithms/index>` for dynamics,
  kinematics, centroidal quantities and plant costs.
* :doc:`Check backend coverage <../tutorials/backend_coverage>` and
  :doc:`compatibility` before choosing a joint/model/operation combination.

.. _id4:

IV. GLASS
---------

`GLASS <https://a2r-lab.org/GLASS/>`_ supplies device-side linear and spatial
algebra, including dot products, matrix-vector products and matrix-matrix
products. GRiD builds its robot-specific CUDA algorithms on these primitives.
Generated headers embed GLASS by default; CUDA applications can instead use
an external ``glass.cuh`` via ``vendor_glass=False``.

.. _id5:

V. Validation tooling
---------------------

`pytest-GPU-proof <https://a2r-lab.org/pytest-gpu-proof/>`_ records signed
GPU-test results and source fingerprints for verification by CPU-only CI.
It is a development dependency, not a GRiD submodule or a runtime dependency
of generated kernels. See :doc:`../tutorials/cuda_validation` for the workflow.
