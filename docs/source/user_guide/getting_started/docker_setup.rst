Docker
======

GRiD does not currently ship an official Docker image. The repo's two
install scripts (``base_install.sh`` and ``developer_install.sh``) are
the supported install path on bare-metal hosts.

If you want to run GRiD in a container, the recipe below is a starting
point that mirrors what the install scripts do. It is **not tested in
CI** — adapt to your CUDA version and host arch before relying on it.

Reference Dockerfile sketch
---------------------------

.. code-block:: dockerfile

   # Pick a CUDA base matching your target compute capability. The repo
   # currently targets sm_120 (RTX 5090) and CUDA 13.x; older sm_8x
   # GPUs work with CUDA 12.x. Use a -devel image so nvcc + cuda headers
   # are present.
   FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

   # System build deps used by developer_install.sh for the Pinocchio
   # pybind11 extension. Plus python and the standard build tools.
   RUN apt-get update && apt-get install -y --no-install-recommends \
           python3 python3-venv python3-pip \
           git pkg-config g++ \
           libeigen3-dev liburdfdom-headers-dev \
       && rm -rf /var/lib/apt/lists/*

   # Clone the repo (or COPY a local checkout in instead).
   WORKDIR /opt
   RUN git clone --recurse-submodules https://github.com/A2R-Lab/GRiD.git
   WORKDIR /opt/GRiD

   # End-user install (creates the .venv used by all scripts).
   RUN bash base_install.sh

   # Developer install (adds Pinocchio + robot_descriptions + Pinocchio
   # second-order pybind11 extension). Comment out if you only need the
   # codegen CLI.
   RUN bash developer_install.sh

   # Make the CLI available on PATH.
   ENV PATH="/opt/GRiD/.venv/bin:${PATH}"

   CMD ["bash"]

Running with GPU access
-----------------------

Use the NVIDIA Container Toolkit to expose the host GPU:

.. code-block:: shell

   docker build -t grid:dev .
   docker run --rm -it --gpus all grid:dev

For the optional ``glass_nvidia`` linear-algebra backend (cuBLASDx via
MathDx), mount or copy the MathDx tarball-extracted tree to
``/opt/nvidia/mathdx/<VERSION>`` inside the container and set
``MATHDX_ROOT`` to match. See :doc:`installation` for details.

Caveats
-------

* The image will be large (CUDA devel + Eigen + Pinocchio is several GB).
* The MathDx package is not redistributable, so it cannot be baked into
  a public image — mount it at runtime if you need the cuBLASDx backend.
* Per-host GLASS autotuning writes a per-host file
  (``GLASS/bench/tuning/<hostname>.cuh``); inside a container with a
  stable hostname this is fine, but with random container hostnames
  you'll regenerate it on every run.

This recipe is a known-incomplete starting point; an officially
supported Docker image is on the long-term wishlist.
