RBDReference
============

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. If your favorite rigid body
dynamics algorithm is not yet implemented please submit a PR with the
implementation. We'll then try to get a GPU, FPGA, and/or accelerator
implementation designed as soon as possible.

Usage and API:
--------------

This package relies on an already parsed ``robot`` object from our
`URDFParser <https://github.com/robot-acceleration/URDFParser>`__
package.

.. code:: python

   RBDReference = RBDReference(robot)
   outputs = RBDReference.ALGORITHM(inputs)

Currently implemented algorithms include the: + Inverse dynamics
(``inverse_dynamics``; the Recursive Newton-Euler Algorithm, RNEA):
``(c,v,a,f) = rbdReference.inverse_dynamics(q, qd, qdd = None, GRAVITY = -9.81)`` +
The gradient of inverse dynamics:
``dc_du = inverse_dynamics_gradient(q, qd, qdd = None, GRAVITY = -9.81)`` where
``dc_du = np.hstack((dc_dq,dc_dqd))`` + The Direct Inverse of the Mass
Matrix Algorithm: ``Minv = rbdReference.minv(q, output_dense = True)`` +
The Composite Rigid Body Algorithm: ``M = rbdReference.crba(q,qd)``

We also include functions that break these algorithms down into there
different passes and by their output types (dq vs dqd) to enable easier
testing of downstream GPU, FPGA, and accelerator implementations.

Testing Algorithms on URDFs
---------------------------

Run the following in terminal from forked repo here at base of the GRiD
working directory in the floating base branch. Run any URDF tests by
replacing the desired urdf with the terminal scripts below.

iiwa14.urdf testing ``python printReferenceValues.py iiwa.urdf -f``

This runs the print printReferenceValues.py script located
https://github.com/A2R-Lab/GRiD/blob/floating-base/printReferenceValues.py.
This will output values for ``RNEA``, ``Minv``, ``CRBA``, ``RNEA_grad``,
and more.

Once can test algorithm outputs using Matlab Spatial V2 using the
following sequence: 
1. Run ``python compare_rnea_fb_spatial.py INSERT_URDF.urdf -f`` which will
print the exact q, qd, qdd inputs for Spatial V2 floating base
algorithms. (Matlab expects floating base joint to input quaternion xyzw
and xyz location of Fb joint) 

2. After generating compatible inputs with
Matlab, create a floating base URDF.m file by running
``python generate_spatial_model.py INSERT_URDF.urdf -f``. 

3. Place the
newly generated ``URDF.m`` file in the working directory of
spatial_v2_extended. 

4. Run the script after ensuring spatial
directories are sourced correctly. 

5. Suggested testing includes
``RNEA (ID)``, ``HandC``, ``Hinverse``, and more.

Instalation Instructions::
--------------------------

The only external dependency is ``numpy`` which can be automatically
installed by running:

.. code:: shell

   pip3 install -r requirements.txt

This package also depends on our
`URDFParser <https://github.com/robot-acceleration/URDFParser>`__
package.
