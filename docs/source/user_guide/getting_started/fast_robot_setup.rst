Fast Robot Setup
================

How to get a robot registered on your machine quickly and efficiently — and
why you only ever pay the compile cost **once**. The per-robot ``.so`` cache
is content-keyed and persistent, so a robot warmed today is an instant,
sub-second load in every later process. This page is the quick canonical
path; the full API tour lives at :doc:`../tutorials/python_wrappers`.

The three-line happy path
-------------------------

.. code-block:: python

   import grid_rbd
   handle = grid_rbd.load_robot("path/to/robot.urdf")
   qdd = handle.forward_dynamics(q, qd, u)

:py:func:`grid_rbd.load_robot` is the frictionless entry point — no name
ceremony, no two-call dance. It derives a stable, **content-addressed name**
from the URDF bytes (``{stem}_{fixed|floating}_{sha256(urdf)[:12]}``), so:

* the **first** call generates ``grid.cuh``, compiles the per-robot ``.so``
  with ``nvcc``, and caches it;
* **every later call — in any process, any day** — that loads the same URDF
  bytes with the same options is a sub-second cache hit. No recompile, ever.

Pass ``name="my_arm"`` if you want a human-friendly handle instead, and
``floating_base=True`` for free-base robots (the fixed and floating loads of
one URDF get distinct, stable names and distinct cache entries). Any
``register_robot`` keyword (``ee_joint_names``, ``max_batch_size``,
``algorithm_list``, ``dtype``, …) passes straight through.

The three entry points
----------------------

All three drive the same persistent cache; pick by ceremony level.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Call
     - Does
     - Use when
   * - ``load_robot(urdf_path, ...)``
     - build-if-missing **and** return a handle; the name is auto-derived
       from the URDF bytes
     - the frictionless default — "just give me this robot"
   * - ``register_robot(name, urdf_path, ...)``
     - build-if-missing **and** return a handle, under a name you choose;
       exposes the full option surface (``dtype``, ``algorithm_list``,
       ``runtime_inertia``, ``enable_mujoco_kernels``, …)
     - you want a stable human-friendly name, or non-default build options
   * - ``precompile(name, urdf_path, tiers=..., backends=...)``
     - build + populate the cache only (returns manifest entries, no
       handle); can prebuild several ``tiers`` (option variants) and warm
       several ``backends`` in one call
     - warm the cache **out-of-band** (CI / Docker / a setup script) so
       interactive use never compiles

Once a named robot is in the cache, ``get_robot(name)`` looks it up without
touching the URDF (raises ``RobotNotRegisteredError`` if absent), and
``list_registered()`` shows everything in the manifest.

All of these are **idempotent**: a robot already in the cache is an instant
no-op — re-registering never re-runs ``nvcc``.

Cache anatomy
-------------

The cache root is ``grid_rbd.default_cache_dir()`` — ``~/.cache/grid-rbd/``
by default, overridden by ``$GRID_RBD_CACHE_DIR`` or ``cache_dir=`` on any
entry point. Inside it, a ``manifest.json`` maps names to keys and each build
lives under ``store/<cache_key>/`` (the generated ``grid.cuh``, the compiled
``robot.so``, build metadata and log) — see the *Cache layout* section of
:doc:`../tutorials/python_wrappers` for the full tree.

The store is **content-keyed**: the cache key is a SHA-256 over the URDF
bytes + the canonicalized option dict + the ``grid-rbd`` version + the CUDA
arch. Consequences worth knowing:

* Re-registering with identical inputs is a pure lookup — no codegen, no
  ``nvcc``, regardless of the name used.
* An inline ``urdf_string=`` and the equivalent file dedupe to the same
  ``.so`` (the key hashes the bytes).
* Different option choices (``ee_joint_names``, ``dtype``,
  ``algorithm_list``, ``floating_base``, …) land in separate entries that
  coexist — nothing is clobbered.
* CUDA arch in the key means a home directory shared between machines (e.g.
  NFS) safely keeps separate ``.so`` files per GPU.
* The cache dir is portable state: build offline once, then **ship or keep
  the cache dir**, and every later run on a matching machine starts in well
  under a second.

What to expect: cold vs. warm
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Robot
     - First (cold) build
     - Every later load (warm)
   * - Small arm (e.g. iiwa14, 7-DOF fixed-base)
     - ~30–60 s of ``nvcc``
     - well under a second
   * - Large humanoid (e.g. g1, h1_2), floating base
     - minutes, and heavy RAM (a full g1-floating pin-only build is ~33 min
       at ~11 GB peak)
     - well under a second

The warm path is the whole point: pay the cold cost once — ideally
out-of-band, see below — and interactive sessions, notebooks, tests, and
deployed processes all start instantly.

RAM-safe big-robot builds
-------------------------

Large floating-base robots are dominated by the MuJoCo-convention ("mjx")
kernel twins that GRiD also emits on floating, non-mimic robots — the
*Build cost on large floating-base robots* section of
:doc:`../tutorials/python_wrappers` has the full story. Three levers keep
the build tractable:

1. **Skip the mjx twins** if you do not need MuJoCo-convention outputs:

   .. code-block:: python

      grid_rbd.load_robot("g1.urdf", floating_base=True,
                          enable_mujoco_kernels=False)

   On g1-floating this is the difference between not building at all and a
   ~33 min build; fixed-base and mimic robots never get mjx twins, so the
   flag is a no-op there.

2. **Build only the algorithms you need** with ``algorithm_list=`` — the
   named algorithms plus their auto-expanded dependencies are compiled,
   dramatically cutting ``nvcc`` wall time, peak RAM, and ``.so`` size.
   Un-built methods raise a clear "add to ``algorithm_list`` and rebuild"
   error, never a segfault. Via ``precompile`` tiers:

   .. code-block:: python

      grid_rbd.precompile(
          "g1", "g1.urdf", floating_base=True,
          tiers=[{"algorithm_list": ["forward_dynamics",
                                     "forward_dynamics_gradient"],
                  "enable_mujoco_kernels": False}],
      )

3. **Warm out-of-band.** Don't pay a minutes-long compile inside a notebook
   or an interactive session — run ``precompile`` once from a script or CI
   step (see the next two sections), and open the notebook against a warm
   cache.

Warming the jax / torch backends
--------------------------------

One compiled ``.so`` is shared across the numpy, jax, and torch surfaces —
``backend=`` never forces a recompile. ``precompile`` can warm each
surface's artifacts on that shared ``.so`` ahead of time:

.. code-block:: python

   grid_rbd.precompile("iiwa14", "iiwa.urdf",
                       backends=("numpy", "jax", "torch"))

Warm the surfaces you will actually use (each backend requires its extra
installed: ``pip install -e ".[jax]"`` / ``".[torch]"`` / ``".[all]"``).
Note that the first ``import jax`` / ``import torch`` and the first
``jax.jit`` trace in a fresh process carry their own one-time framework
costs — those belong to the framework, not to GRiD's cache.

For agents and CI: the non-interactive warm
-------------------------------------------

The exact one-liner to warm a robot from a script, CI job, or agent shell
(idempotent — safe to run every time):

.. code-block:: shell

   .venv/bin/python -c "import grid_rbd; grid_rbd.precompile('go2', 'config/robot_assets/go2.urdf', floating_base=True, backends=('numpy', 'jax'))"

Run it once per robot before any interactive or latency-sensitive use; every
subsequent ``load_robot`` / ``get_robot`` / ``jax.jit`` in any process is
then an instant cache hit.

See also
--------

* :doc:`../tutorials/python_wrappers` — the full ``grid-rbd`` API tour:
  method surface, cache layout, jax/torch backends, ``grid_plant``.
* ``bindings/examples/AGENT_INTEGRATION_GUIDE.md`` — GPU-resident usage
  patterns for agents (stay on device, CUDA graphs, zero-copy interop).
* ``examples/notebooks/`` — the guided notebook track (start with
  ``01_quickstart_iiwa14``).
