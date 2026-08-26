Installation
============

GRiD is a single repository with its peer products (``GLASS``, ``RBDReference``,
``URDFParser``) vendored as git submodules under ``external/``. Clone with
``--recursive`` so those populate, then run the install script — a single
``pip install -e .`` installs the codegen toolkit and the ``grid_rbd`` Python
wrapper together (see the :doc:`../../../index` quick-start for the extras).

.. code-block:: shell

    git clone --recursive https://github.com/A2R-Lab/GRiD.git
    cd GRiD

If you already cloned without ``--recursive``, populate the submodules with
``git submodule update --init --recursive``.

Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest path is to use the provided install scripts, which create a
local ``.venv`` and register the ``grid-generate`` CLI.

For end-user installs (just the runtime + CLI):

.. code-block:: shell

   bash install/base_install.sh
   source .venv/bin/activate

For developer installs (adds Pinocchio, robot-description fixtures,
documentation tooling, and the Pinocchio second-order pybind11 extension
used as the golden oracle in the equivalence tests):

.. code-block:: shell

   bash install/developer_install.sh

The developer script will, on Debian/Ubuntu, install the system build
deps needed by the Pinocchio pybind11 extension via ``apt-get``:
``pkg-config``, ``g++``, ``libeigen3-dev``, ``liburdfdom-headers-dev``.
The ``pin`` wheel ships its own ``pinocchio.pc`` inside the venv via
``cmeel``, and ``install/developer_install.sh`` computes the right
``PKG_CONFIG_PATH`` automatically for the extension build — no manual
configuration is required.

You can also install manually with:

.. code-block:: shell

   pip3 install -r install/requirements.txt

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
   export PATH="opt/nvidia/nsight-compute/:$PATH"

.. note::

    GRiD requires a C++17-capable host compiler (e.g. ``g++ >= 7`` or
    ``clang++ >= 5``). The benchmark and codegen runtime compile with
    ``-std=c++17``, needed for inline variables in the bench common
    header.

