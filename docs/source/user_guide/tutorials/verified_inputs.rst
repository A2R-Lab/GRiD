CPU-checked input examples
==========================

These examples require an installed recursive source checkout and run without
CUDA compilation or GPU calls. From the repository root::

   python docs/examples/model_inputs.py
   python -m pytest docs/test_doc_examples.py -q

Fixed and floating input packing
--------------------------------------------------

The two fixtures are fixed-base iiwa14 (NQ=NV=7) and floating-base Go2
(NQ=19, NV=18). The example initializes a valid identity base quaternion and
builds the dynamics inputs and the plant state at their public widths. It uses
the default Pinocchio convention. Zero joint angles are
illustrative, not a guaranteed collision-free or joint-limit-safe posture.
This initializer is **not** general to spherical joints or other conventions;
use the model's coordinate maps for those cases.

.. literalinclude:: ../../../examples/model_inputs.py
   :language: python
   :pyobject: model_inputs

For a batch of two, Go2's ``q`` is ``(2, 19)``; ``qd`` and the control ``u``
are ``(2, 18)``, the tangent width, on every surface (NumPy, JAX, PyTorch and
the C ABI); the plant state ``x`` is ``(2, 37)``. Dynamics vector outputs such
as torques and accelerations come back ``(2, 18)`` as well. See
:doc:`../concepts/input_output_abi` and :doc:`python_wrappers` for GPU calls
and operation selection.

Constructing a regressor parameter vector
--------------------------------------------------

Read the parser's merged body inertias, not the original XML link order.
The example extracts mass, first moment and body-origin inertia into GRiD's
``[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`` order:

.. literalinclude:: ../../../examples/model_inputs.py
   :language: python
   :pyobject: inertial_parameters

The executable example checks ``Y @ pi == tau`` on a nonzero iiwa14 state
using the CPU reference. This verifies the parameter basis and example, not
an independent GPU comparison. Do not copy Pinocchio's dynamic-parameter
vector without converting its order. See
:doc:`../concepts/algorithms/centroidal_and_bias`.

An integrator diagnostic
--------------------------------------------------

Run ``python docs/examples/integrator_semantics.py`` to exercise the actual
CPU reference integrator with constant acceleration, initially zero position
and velocity, and final time 1. The exact final position is 0.5. Euler's
position errors for 10, 20, 40 and 80 steps are respectively
0.05, 0.025, 0.0125 and 0.00625. The updated CPU reference's full-state RK4,
midpoint, Heun (``trapezoidal``), and ``constant_acceleration`` schemes are
exact up to roundoff for this particular problem, not for arbitrary dynamics.

This diagnostic isolates the update rule; it is not a robot benchmark, a GPU
test or a general convergence certification. Independent oscillator and
quaternion-ODE tests in ``RBDReference/tests/test_integrator_contract.py``
check order two for midpoint/Heun and order four for RK4 in Euclidean
coordinates, and the second-order rotational limit of the base-point
retraction scheme. Reference availability does not establish GPU support;
see :doc:`../concepts/algorithms/integrators_and_plant` for the generated API.
