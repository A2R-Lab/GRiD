Installation
============

It is recommended when installing GRiD to use git clone on the main library `GRiD <https://github.com/robot-acceleration/GRiD>`.
In the case that the linked submodules :doc:`URDFParser <../tutorials/urdf_parser>`, :doc:`GRiDCodeGenerator <../tutorials/codegen>`, :doc:`RBDReference <../tutorials/python_algorithms>` and are empty folders,
one can individually popoulate each submodule using its git link as needed. 

Run the following script to install GRiD and its related submodules.
`URDFParser <https://github.com/robot-acceleration/URDFParser>`__,
`GRiDCodeGenerator <https://github.com/robot-acceleration/GRiDCodeGenerator>`__,
and
`RBDReference <https://github.com/robot-acceleration/RBDReference>`__

.. code:: shell

    # In the root of your desired project directory
    git clone https://github.com/A2R-Lab/GRiD.git
    cd RBDReference
    git clone https://github.com/A2R-Lab/RBDReference.git
    cd ..
    cd URDFParser
    git clone https://github.com/A2R-Lab/URDFParser.git
    cd ..
    cd GRiDCodeGenerator
    git clone https://github.com/A2R-Lab/GRiDCodeGenerator.git

.. note::
    
    Alternatively, can directly download the zips from these links: `URDFParser <https://github.com/robot-acceleration/URDFParser>`__, `GRiDCodeGenerator <https://github.com/robot-acceleration/GRiDCodeGenerator>`__, and `RBDReference <https://github.com/robot-acceleration/RBDReference>`__.
    Note that directory setup in this manner requires adjustment of python ``import`` statements such that ``from URDFParser import URDFParser`` becomes ``from URDFParser.URDFParser import URDFParser``. Thus, each import statement from submodules will require an additional call for correct directory linking. 


It is also recommended to create a virtual environment for each external dependency for ease of access. Run the following script to list and update the requirements tab if other dependencies are needed during the installation process.

.. code-block:: shell

    # Run in virtual enviornment
    pip3 list # list all pip modules
    pip freeze > requirements.txt

Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest path is to use the provided install scripts, which create a
local ``.venv`` and register the ``grid-generate`` CLI.

For end-user installs (just the runtime + CLI):

.. code-block:: shell

   bash base_install.sh
   source .venv/bin/activate

For developer installs (adds Pinocchio, robot-description fixtures,
documentation tooling, and the Pinocchio second-order pybind11 extension
used as the golden oracle in the equivalence tests):

.. code-block:: shell

   bash developer_install.sh

The developer script will, on Debian/Ubuntu, install the system build
deps needed by the Pinocchio pybind11 extension via ``apt-get``:
``pkg-config``, ``g++``, ``libeigen3-dev``, ``liburdfdom-headers-dev``.
The ``pin`` wheel ships its own ``pinocchio.pc`` inside the venv via
``cmeel``, and ``developer_install.sh`` computes the right
``PKG_CONFIG_PATH`` automatically for the extension build — no manual
configuration is required.

You can also install manually with:

.. code-block:: shell

   pip3 install -r requirements.txt

Install CUDA Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

::

   sudo apt-get update
   sudo apt-get -y install xorg xorg-dev linux-headers-$(uname -r) apt-transport-https

Download and Install CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~

Note: for Ubuntu 20.04 see https://developer.nvidia.com/cuda-downloads
for other distros

::

   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda

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

