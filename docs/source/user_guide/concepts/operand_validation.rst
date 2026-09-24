Operand validation
==================

What each surface checks before a kernel runs, and where. The rules are per
**operand class**, not per method: every input of every algorithm belongs to
one of the classes below (a CPU test, ``test/test_operand_validation_table.py``,
asserts this against the ABI table so a new algorithm cannot add an unlisted
operand), and each surface applies the class rule at one place.

The rule set
------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 26 26

   * - Operand class
     - Rule
     - Members
     - Not checked
   * - ``configuration``
     - ``(B, num_joints)``; the handle dtype; ``1 <= B <= max_batch``;
       every operand of a call carries the same ``B`` as the leading one.
       A per-sample ``(num_joints,)`` input is accepted on numpy (batch of
       one, 1D result) and on JAX under ``vmap``
     - ``q``, ``qd``, ``qdd``, ``qdd_opt``, ``u``, ``var``, ``lower``,
       ``upper``, ``u_des``
     - finite values, joint limits, unit quaternion on a floating base
   * - ``force``
     - ``(B, 6 * num_bodies)`` body-major ``[angular; linear]`` local-frame
       wrenches; the native buffer must physically carry the state's ``B``
       (numpy and torch refuse a broadcast ``(1, 6NB)``; the JAX method
       materializes a supported broadcast before the FFI call, whose handler
       still refuses a mismatched physical batch)
     - ``f_ext``
     - frame consistency
   * - ``state``
     - ``(B, num_pos + num_vel)`` plant state; same ``B``
     - ``x``, ``x_des``
     - finite values
   * - ``plant operand``
     - ``(B, n)`` with ``n`` fixed per cost (``3`` for a point target, ``6``
       for a momentum target, ``num_joints`` / ``num_vel`` for the barriers);
       same ``B``
     - ``p_des``, ``W``, ``Q``, ``R``, ``h_des``
     - positive semi-definiteness of weights
   * - ``tool operand``
     - ``wrench`` ``(6,)`` and ``rc`` ``(3,)`` per call
     - ``wrench``, ``rc``
     - —
   * - ``runtime offset``
     - 16 numbers (a 4x4 column-major SE(3) transform) or none; the target
       joint index must be in ``[0, num_joints)``
     - ``offset``, ``target_jid``
     - orthonormality of the rotation block
   * - ``scalar``
     - Python floats / ints (``gravity``, ``dt``, ``it``, ``mu``); the
       integrator kind is validated by name on every surface
     - —
     - finiteness, ``dt > 0``

A wrong ``num_joints`` vs ``num_vel`` width on a **floating base** (an
``nv``-wide ``qd`` / ``u``, the mjx and Pinocchio habit) is refused by name on
the numpy and JAX surfaces (``num_vel`` footgun message) and by the generic
last-dimension rule on torch.

Where each surface enforces the rules
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 30 30 26

   * - Surface
     - Coercion
     - Shape / batch
     - Error type
   * - numpy
     - ``np.ascontiguousarray(x, dtype)`` in the handle, and the pybind
       ``py::array_t<c_style | forcecast>`` boundary (a non-contiguous or
       differently typed array is copied and cast, never rejected)
     - ``check_array_2d`` in the pybind core for **every** array: 2D, last
       dim, same batch, ``batch >= 1``; ``batch > max_batch`` is the C ABI's
       rc 2
     - ``ValueError`` (shape), ``RuntimeError`` (``max_batch``, rc codes)
   * - torch
     - none: a CPU tensor, a non-contiguous tensor or a wrong dtype is
       refused (the op never copies an input)
     - ``grid_torch_check`` per tensor (CUDA, contiguous, dtype, 2D, last
       dim); ``grid_torch_batch`` (``1 <= B <= max_batch``);
       ``grid_torch_check_rows`` for every operand after the leading one
       (inside the shared input pack, so no op can skip it); ``f_ext`` batch
     - ``RuntimeError`` (``TORCH_CHECK``)
   * - JAX
     - ``jnp.asarray(x, dtype)`` in ``_prep_2d`` / ``_prep_plant``; device
       placement is XLA's
     - Python: 2D or 1D-under-``vmap``, last dim, same batch, ``max_batch``.
       Native (every FFI handler, so a traced program or a direct handler
       call is covered too): ``GRID_RBD_FFI_VALIDATE_2D`` on the leading
       operand, ``GRID_RBD_FFI_VALIDATE_ROWS`` on every other one,
       ``1 <= B <= max_batch``
     - ``ValueError`` (Python), ``XlaRuntimeError`` (native)

The native checks are the contract; the Python checks exist to give the
friendlier message first. One asymmetry: an **empty batch** is refused on
numpy and torch (``batch must be >= 1``), but on JAX it returns an empty
result — XLA elides a zero-sized custom call, so no handler runs. The
malformed-input test module (``test/python_wrappers/test_operand_validation.py``)
drives the configuration and force classes through all three surfaces,
including the native JAX handler with the Python checks bypassed, plus the
plant, tool and runtime-offset classes on the surfaces that carry them; the
CPU table test proves the classification is complete, not that every class is
exercised on every surface.
