# GRiD runtime image: grid-rbd from PyPI on a CUDA devel base, with the JAX
# and PyTorch CUDA builds, and the generated iiwa14 header baked in.
#
# Build from the repository root (the build context supplies the sample URDF):
#   docker build -f install/docker/Dockerfile -t grid:0.5.1 .
# Run with the NVIDIA Container Toolkit:
#   docker run --rm -it --gpus all -v grid-cache:/var/cache/grid-rbd grid:0.5.1
#
# The image holds no compiled robot library. grid_rbd compiles a robot for the
# GPU it finds at the first load_robot or register_robot call, with nvcc from
# this base image, and caches the result under GRID_RBD_CACHE_DIR. Mount that
# directory as a volume to keep the cache across containers. The baked header
# /opt/grid/iiwa14/grid.cuh serves the header-only path directly:
#   nvcc -arch=sm_90 -std=c++17 -I /opt/grid/iiwa14 my_kernel.cu
#
# The base is the CUDA 13.2 toolkit the release was built and tested with
# (compatibility page); the cu128 torch wheel and the cuda12 JAX plugin carry
# their own runtimes, as on the validated box. The image itself is built in CI
# without a GPU and is not exercised by any receipt. .github/workflows/docker.yml
# builds this file and publishes ghcr.io/a2r-lab/grid.
ARG CUDA_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG GRID_RBD_VERSION=0.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:${PATH} \
    GRID_RBD_CACHE_DIR=/var/cache/grid-rbd

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --upgrade pip

# torch first from the CUDA index so the [torch] extra below is already
# satisfied and pip does not also fetch the PyPI torch build.
RUN pip install torch --index-url "${TORCH_INDEX_URL}" \
    && pip install "grid-rbd[jax,torch]==${GRID_RBD_VERSION}" "jax[cuda12]"

# The sample robot and its generated header. The wheel ships no URDF; this one
# comes from the build context (grid_assets/robots/iiwa14.urdf).
COPY grid_assets/robots/iiwa14.urdf /opt/grid/robots/iiwa14.urdf
RUN grid-generate /opt/grid/robots/iiwa14.urdf -o /opt/grid/iiwa14/grid.cuh \
    && python -c "import grid_rbd, grid_codegen, jax, torch; print(grid_rbd.__version__, jax.__version__, torch.__version__)"

VOLUME ["/var/cache/grid-rbd"]
WORKDIR /work
CMD ["bash"]
