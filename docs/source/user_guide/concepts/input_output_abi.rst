Input / Output ABI (``h_q_qd_u``)
=================================

**Read this before you pack a buffer by hand.** GRiD is built for power users who
call the generated kernels directly from their own CUDA (see
:doc:`design_principles`), which means *you* own the layout of the input buffer.
This page is the contract for that raw kernel buffer. Getting it wrong on a
floating base does not crash and does not warn: it silently returns wrong
dynamics.

The Python handles and the C ABI of the compiled ``grid-rbd`` wrapper do **not**
expose this buffer. They take ``q`` at the configuration width ``nq`` and every
velocity-like input (``qd``, ``qdd``, ``u``) at the tangent width ``nv``, and
return dynamics vectors at ``nv``, exactly as Pinocchio and MuJoCo do. The
padded layout below is staged inside the wrapper. See :doc:`../tutorials/python_wrappers`.

The per-timestep input block
----------------------------

Every algorithm that takes ``(q, qd, u)`` — forward dynamics, its gradients, the
second-order kernels, the integrators, the regressor — reads one contiguous block
per timestep out of ``h_q_qd_u`` / ``d_q_qd_u``. That block is **three
``NUM_POS``-wide slots**:

.. code-block:: text

   stride = Q_QD_U_STRIDE = 3 * NUM_POS          # NUM_JOINTS == NUM_POS
   timestep k occupies  [k*stride, (k+1)*stride)

       block + 0            q
       block + NUM_POS      qd
       block + 2*NUM_POS    u        (or qdd, for the q|qd|qdd kernels)

The generated kernels slice it exactly that way::

   T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[NUM_POS]; T *s_u = &s_q_qd_u[2*NUM_POS];

The header also emits the slot offsets as named constants, right next to the
contract stated as a comment block::

   grid::GRID_Q_OFFSET     // 0
   grid::GRID_QD_OFFSET    // NUM_POS
   grid::GRID_U_OFFSET     // 2*NUM_POS
   grid::GRID_QDD_OFFSET   // == GRID_U_OFFSET (qdd shares slot 2 with u)

Always derive your offsets from these emitted constants (or ``NUM_POS`` /
``Q_QD_U_STRIDE``) rather than hardcoding integers.

.. warning::

   **Do not pack ``q|qd|u`` tightly.** The slots are ``NUM_POS`` wide, *not*
   ``NUM_VEL`` wide. On a fixed base with independent scalar joints ``nq == nv``, so a tight packing happens to
   produce identical bytes and the mistake is invisible. On a quaternion floating
   base without spherical joints ``nq == nv + 1``; each independent spherical
   joint adds another quaternion offset. A tight packing writes ``u`` at ``NUM_POS + NUM_VEL``
   while every kernel reads it from ``2*NUM_POS``. That read is in bounds, so
   nothing traps: you get plausible, wrong numbers. Fixed-base code that is
   "known good" therefore proves nothing about your floating-base packing.

Floating base
-------------

For a floating base the root contributes **7 positions but only 6 velocities**:

.. code-block:: text

   q  (NUM_POS = 7 + n_joints)   [ translation(3) | quaternion xyzw(4) | joint positions ]
   qd (NUM_VEL = 6 + n_joints)   [ linear vel(3)  | angular vel(3)     | joint velocities ]

In the raw kernel buffer ``qd`` and ``u`` occupy **``NUM_POS``-wide slots**: their
``NUM_VEL`` meaningful values fill the *leading* entries of the slot and the
trailing entry is a pad (write zero). The kernels index velocities by tangent
index, so this "leading entries, then padding" rule holds for every model,
floating or spherical. The quaternion is ``xyzw``; identity is ``[0, 0, 0, 1]``.
The user-facing root velocity is ordered ``[linear; angular]``; GRiD permutes it
into the internal Featherstone ``[angular; linear]`` spatial ordering for you.

.. note::

   **Kernel buffer slots are nq-wide; every public width is physical.** Mass
   matrices, ``Minv``, the dynamics gradients and, on the wrapper surfaces, the
   dynamics vectors and the velocity-like inputs are all ``NUM_VEL`` wide. Only a
   hand-packed ``h_q_qd_u`` carries the padding.

Which surface protects you
--------------------------

* **Python bindings and the C ABI (**``grid-rbd``**)** — protected. You pass
  ``qd``/``qdd``/``u`` at ``nv`` and the wrapper stages the padded buffer itself;
  an ``nq``-wide (padded) array on a floating-base robot raises a precise
  ``ValueError`` naming the tangent width. Nothing is auto-padded or sliced.
* **Direct CUDA consumers** — unprotected by construction. You pack
  ``h_q_qd_u`` yourself, so this page is the only contract. Size the buffer as
  ``3 * NUM_JOINTS * NUM_TIMESTEPS`` (what ``init_gridData`` allocates) and take
  the offsets from the constants above.

Stability
---------

``Q_QD_U_STRIDE == 3 * NUM_POS`` and the three slot offsets are a **published
ABI**. They are pinned by ``test/cuda_equivalents/test_cuda_input_abi.py``, which
checks the emitted offsets themselves (not merely the stride constant) on both
floating and fixed robots, and asserts that the device allocation, the host
memcpy, and the kernel stride argument all still agree. A change here breaks every
consumer that packs the buffer, so it is a deliberate, announced change — not a
refactor.
