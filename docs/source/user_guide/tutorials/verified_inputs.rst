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
shows the difference between NumPy dynamics buffers and plant state/control
buffers. It uses the default Pinocchio convention. Zero joint angles are
illustrative, not a guaranteed collision-free or joint-limit-safe posture.
This initializer is **not** general to spherical joints or other conventions;
use the model's coordinate maps for those cases.

.. literalinclude:: ../../../examples/model_inputs.py
   :language: python
   :pyobject: model_inputs

For a batch of two, Go2's ``q``, padded ``qd`` and padded dynamics control
are ``(2, 19)``; plant ``x`` is ``(2, 37)`` and plant control is ``(2, 18)``.
Physical velocity/control entries occupy the first NV slots; padding is not
an additional degree of freedom. See :doc:`../concepts/input_output_abi`
and :doc:`python_wrappers` for GPU calls and operation selection.

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
and velocity, and final time 1. The exact final position is 0.5. The current
``rk4`` position errors for 10, 20, 40 and 80 steps are respectively
0.05, 0.025, 0.0125 and 0.00625. Halving the step halves the position error
in this case; the current name does not imply classical full-state RK4.
The current ``trapezoidal`` update is exact up to roundoff for this particular
constant-acceleration example, not for arbitrary dynamics.

This diagnostic isolates the update rule; it is not a robot benchmark, a GPU
test or a general convergence certification. Its characterization test must
be deliberately updated if the integration contract changes. See
:doc:`../concepts/algorithms/integrators_and_plant` for the present semantics.
