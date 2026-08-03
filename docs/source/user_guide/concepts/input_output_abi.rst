Input / Output ABI (``h_q_qd_u``)
=================================

**Read this before you pack a buffer by hand.** GRiD is built for power users who
call the generated kernels directly from their own CUDA (see
:doc:`design_principles`), which means *you* own the layout of the input buffer.
This page is the contract. Getting it wrong on a floating base does not crash and
does not warn — it silently returns wrong dynamics.

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

Always derive your offsets from the emitted ``NUM_POS`` / ``Q_QD_U_STRIDE``
constants rather than hardcoding integers.

.. warning::

   **Do not pack ``q|qd|u`` tightly.** The slots are ``NUM_POS`` wide, *not*
   ``NUM_VEL`` wide. On a fixed base ``nq == nv``, so a tight packing happens to
   produce identical bytes and the mistake is invisible. On a quaternion floating
   base ``nq == nv + 1``, so a tight packing writes ``u`` at ``NUM_POS + NUM_VEL``
   while every kernel reads it from ``2*NUM_POS``. That read is in bounds, so
   nothing traps: you get plausible, wrong numbers. Fixed-base code that is
   "known good" therefore proves nothing about your floating-base packing.

Floating base
-------------

For a floating base the root contributes **7 positions but only 6 velocities**:

.. code-block:: text

   q  (NUM_POS = 7 + n_joints)   [ translation(3) | quaternion xyzw(4) | joint positions ]
   qd (NUM_VEL = 6 + n_joints)   [ linear vel(3)  | angular vel(3)     | joint velocities ]

``qd`` and ``u`` are still passed at the **``NUM_POS`` width**: their ``NUM_VEL``
meaningful values occupy the *leading* slots of their block and the trailing slot
is a pad (write zero). The quaternion is ``xyzw`` — identity is
``[0, 0, 0, 1]``. The user-facing root velocity is ordered ``[linear; angular]``;
GRiD permutes it into the internal Featherstone ``[angular; linear]`` spatial
ordering for you.

.. note::

   **Inputs are nq-wide, outputs are nv-wide.** Mass matrices, ``Minv``, and the
   dynamics gradients all come back at ``NUM_VEL``. A caller porting from
   Pinocchio or MuJoCo naturally reaches for an ``nv``-wide ``qd``/``u`` to match
   those outputs — that is the single most common floating-base mistake.

Which surface protects you
--------------------------

* **Python bindings (**``grid-rbd``**)** — protected. The handle packs the buffer
  for you, and ``_check_nq_width`` raises a precise ``ValueError`` if you hand it
  an ``nv``-wide ``qd``/``qdd``/``u`` on a floating-base robot. It deliberately
  does *not* auto-pad: guessing the base-velocity layout would be worse than an
  explicit error.
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
