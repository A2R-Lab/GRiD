Installation
============

GRiD is published on PyPI as ``grid-rbd``: one distribution containing the
code generator, the ``grid-generate`` / ``grid-spherize`` CLIs and the
``grid_rbd`` Python wrapper with its NumPy, JAX and PyTorch adapters. Install
it with pip; work from a source checkout only when you want to develop GRiD or
use the checkout's sample robots.

Install from PyPI
~~~~~~~~~~~~~~~~~

Release wheels and the source distribution are published on PyPI as
``grid-rbd`` (Linux x86-64, CPython 3.10–3.12)::

   python -m pip install grid-rbd

This does not compile robot CUDA code or access a GPU. On a platform without
a matching wheel pip builds the source distribution, whose small Python
extension requires a C++17 compiler. GRiD is alpha software: APIs may change
between minor versions (see the `changelog
<https://github.com/A2R-Lab/GRiD/blob/main/CHANGELOG.md>`_ for what each
release ships).

The base install requires NumPy; the ``jax``, ``torch`` and ``all`` extras
select optional framework dependencies (see the :doc:`../../index` quick-start
for the extras). Install a GPU-enabled framework build suitable for your
hardware **before registering a robot**. Neither these extras nor GRiD install
the system NVIDIA driver or CUDA Toolkit.

Registration is explicit and can take substantial time for large robots::

   import grid_rbd
   robot = grid_rbd.register_robot(
       "my_robot", urdf_path="robot.urdf",
       algorithm_list=["inverse_dynamics", "inverse_dynamics_gradient"],
       backend="numpy")

Use ``backend="jax"`` or ``backend="torch"`` for the other interfaces. The
adapters share the content-addressed registration cache. A subsequent
registration with unchanged inputs reuses its compiled library; changed model,
generator, toolkit or framework ABI inputs can require a new build. See
:doc:`fast_robot_setup` for cache loading and subset selection.

Header-only use (CUDA power users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second entry point skips the Python wrappers entirely. The generator
emits one self-contained ``grid.cuh`` per robot; you ``#include`` it in your
own CUDA translation unit and build with ``nvcc``. Requirements: the
``grid-generate`` CLI (from the wheel above, or a source checkout) to produce
the header, and the CUDA Toolkit's ``nvcc`` plus a host compiler to build your
kernel. No per-robot ``.so``, no NumPy/JAX/PyTorch and no Python at runtime.

.. code-block:: shell

   pip install grid-rbd
   grid-generate robot.urdf -o grid.cuh      # -f for a floating base
   nvcc -arch=sm_90 -std=c++17 my_kernel.cu  # GLASS linear algebra is vendored into the header

See :doc:`../tutorials/codegen` for a minimal kernel and the validated
example under ``examples/cuda/``.

Source installation (development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository vendors its peer products (``GLASS``, ``RBDReference``,
``URDFParser``) as git submodules under ``external/``. Clone with
``--recursive`` so those populate (if you already cloned without it, run
``git submodule update --init --recursive``), then run the install script — a
single ``pip install -e .`` installs the codegen toolkit and the ``grid_rbd``
wrapper together and registers the CLIs in a local ``.venv``:

.. code-block:: shell

    git clone --recursive https://github.com/A2R-Lab/GRiD.git
    cd GRiD
    bash install/base_install.sh
    source .venv/bin/activate

An **editable install from a Git checkout** uses that checkout's generator,
peer submodules, wrapper template and launch profiles, so keep it in place.
``main`` is the branch documented here. Distribution wheels instead contain
their own pinned resources and do not need Git or a retained checkout.

For developer installs (adds Pinocchio, robot-description fixtures,
documentation tooling, and the Pinocchio second-order pybind11 extension
used as the golden oracle in the equivalence tests):

.. code-block:: shell

   bash install/developer_install.sh

The developer script is the *everything* install (the base install stays
NumPy-only): on top of the base install it adds the GPU frameworks
(``[all]`` extras, ``jax[cuda12]`` and a torch CUDA wheel from
``TORCH_INDEX_URL``, default cu128), the Pinocchio oracle stack pinned at
``pin<4``, ``frax`` for the benchmark column, builds CppADCodeGen into
``/usr/local`` (``sudo make install``) and the Pinocchio second-order
extension, and prepares the reference models. Opt out of the pieces you do
not need with ``SKIP_GPU_FRAMEWORKS=1``, ``SKIP_FRAX_INSTALL=1`` and
``SKIP_CPPADCG_INSTALL=1``; ``PINOCCHIO_EQUIVALENCE_TIER`` selects the robot
tier (only ``smoke`` exists today). On Debian/Ubuntu it installs the system
build deps needed by the Pinocchio pybind11 extension via ``apt-get``:
``pkg-config``, ``g++``, ``cmake``, ``libeigen3-dev``, ``liburdfdom-headers-dev``,
``libcppad-dev``.
The ``pin`` wheel ships its own ``pinocchio.pc`` inside the venv via
``cmeel``, and ``install/developer_install.sh`` computes the right
``PKG_CONFIG_PATH`` automatically for the extension build — no manual
configuration is required.

You can also install manually, with the extras of your choice:

.. code-block:: shell

   pip install -e ".[all]"

Bundled provenance
~~~~~~~~~~~~~~~~~~

Wheels and source distributions bundle URDFParser and RBDReference Python
sources from GRiD's exact submodule commits, plus GLASS headers and launch
profiles. They do not resolve newer peer versions during installation.
Inspect the commit IDs, source hashes and resource hashes with::

   from grid_codegen.resources import bundled_provenance
   print(bundled_provenance())

Third-party licenses are included under ``grid_codegen/_data/licenses``.
User URDF files and referenced meshes remain user-supplied. The checkout's
example robot collection and optional foam checkout are not bundled.

Runtime requirements
~~~~~~~~~~~~~~~~~~~~

What each activity needs:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Activity
     - Requirements
   * - Import ``grid_codegen`` / generate ``grid.cuh``
     - Python ≥ 3.10; populated submodules for an editable install, or the
       bundled distribution resources. No GPU, no ``nvcc``.
   * - ``grid_rbd.register_robot`` / ``warm_robot`` (first call per robot)
     - the CUDA Toolkit's ``nvcc`` on ``PATH`` (the toolkit that matches your
       driver), a host C++ compiler, and an NVIDIA GPU present (the compute
       capability is read from ``nvidia-smi`` unless you pass ``cuda_arch=``).
   * - Warm loads and every numeric method
     - a GPU with the arch the ``.so`` was built for; no ``nvcc``.
   * - ``backend="jax"`` / ``backend="torch"``
     - the ``[jax]`` / ``[torch]`` extra **plus a CUDA build of that framework
       matching your GPU arch** (the extras pin the CPU packages only; the
       CUDA wheel is your choice, e.g. ``pip install "jax[cuda12]"`` or
       ``"jax[cuda13]"``, and a ``cu1xx`` torch wheel — see the
       `bindings README <https://github.com/A2R-Lab/GRiD/blob/main/bindings/README.md>`_).
       A missing framework, or a CPU-only jax, is reported at
       ``register_robot`` time, not deep inside a call.
   * - Equivalence tests / the Pinocchio oracle / docs
     - ``install/developer_install.sh`` (apt build deps, ``pin``,
       robot-description fixtures, the Pinocchio second-order extension).

Platform: Linux x86_64 with CUDA 12.x/13.x is what is built and tested
(the committed GPU-proof receipt names the exact GPU and toolkit). Windows
and macOS are not supported. The submodules
must be populated before the first **generation** (not before the editable
install itself): ``install/base_install.sh`` runs ``pip install -e .`` and
then ``git submodule update --init --recursive``; a bare editable install on
a non-recursive clone succeeds and then fails at first generation with a
"GLASS submodule is missing" error naming the fix.

Containers
~~~~~~~~~~

GRiD does not ship an official Docker image. The recipes below are starting
points that are **not tested in CI** — pick a CUDA ``-devel`` base matching
your toolkit and target compute capability (``nvcc`` and the CUDA headers must
be in the image; the GPU driver comes from the host). Robot CUDA is compiled
at ``register_robot`` time, so warm the cache in the image build with
``grid_rbd.precompile`` (see :doc:`fast_robot_setup`) if production must never
pay the one-time ``nvcc`` cost.

PyPI install (runtime + CLI only):

.. code-block:: dockerfile

   FROM nvidia/cuda:13.2.0-devel-ubuntu24.04
   RUN apt-get update && apt-get install -y --no-install-recommends \
           python3 python3-venv python3-pip g++ \
       && rm -rf /var/lib/apt/lists/*
   RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install grid-rbd
   ENV PATH="/opt/venv/bin:${PATH}"
   CMD ["bash"]

Source install (adds the checkout's sample robots and, optionally, the
developer tooling):

.. code-block:: dockerfile

   FROM nvidia/cuda:13.2.0-devel-ubuntu24.04
   RUN apt-get update && apt-get install -y --no-install-recommends \
           python3 python3-venv python3-pip git pkg-config g++ \
           libeigen3-dev liburdfdom-headers-dev \
       && rm -rf /var/lib/apt/lists/*
   WORKDIR /opt
   RUN git clone --recurse-submodules https://github.com/A2R-Lab/GRiD.git
   WORKDIR /opt/GRiD
   RUN bash install/base_install.sh
   # RUN bash install/developer_install.sh   # Pinocchio oracle + fixtures (several GB)
   ENV PATH="/opt/GRiD/.venv/bin:${PATH}"
   CMD ["bash"]

Run with the NVIDIA Container Toolkit to expose the host GPU:

.. code-block:: shell

   docker build -t grid:dev .
   docker run --rm -it --gpus all grid:dev

Install CUDA Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

::

   sudo apt-get update
   sudo apt-get -y install xorg xorg-dev linux-headers-$(uname -r) apt-transport-https

Download and Install CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~

Note: the commands below are for Ubuntu 24.04 (``ubuntu2404``) —
substitute your release in the repo URL, and see
https://developer.nvidia.com/cuda-downloads for other distros. NVIDIA's
repos now use the ``cuda-keyring`` package (the old ``apt-key`` method
was removed in Ubuntu 22.04+):

::

   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit

Add the following to ``~/.bashrc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   export PATH="/usr/local/cuda/bin:$PATH"
   export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
   export PATH="/opt/nvidia/nsight-compute/:$PATH"

.. note::

    GRiD requires a C++17-capable host compiler (e.g. ``g++ >= 7`` or
    ``clang++ >= 5``). The benchmark and codegen runtime compile with
    ``-std=c++17``, needed for inline variables in the bench common
    header. With the ``[torch]`` extra the per-robot ``.so`` is compiled
    with whatever standard the installed torch's ATen headers demand
    (``-std=c++20`` from torch 2.14 on, detected from the header guard), so
    a torch-enabled build needs an nvcc and host compiler that accept C++20
    (CUDA 12+, ``g++ >= 10``). ``GRID_RBD_CXX_STD`` forces the standard.
