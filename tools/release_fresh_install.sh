#!/usr/bin/env bash
# Release gate: build the release artifacts, install the repaired wheel into a
# throwaway virtual environment outside the checkout, and exercise both entry
# points the README advertises from a directory outside the checkout.
#
#   make release-check            (or: bash tools/release_fresh_install.sh)
#
# Steps, each fatal on error:
#   1. python -m build (sdist, then the wheel from that sdist), auditwheel show
#      and repair, twine check. Artifacts land under WORK/dist and WORK/wheelhouse.
#   2. A fresh venv under WORK (never an editable install). The repaired wheel
#      is installed with --only-binary=:all:, then jax[cuda12] and a torch CUDA
#      wheel from TORCH_INDEX_URL (default cu128). The checkout must not appear
#      on sys.path; the script asserts it.
#   3. tools/check_installed_package.py from WORK: CPU with the GPU hidden,
#      then --gpu, then --gpu --reload (RELEASING.md gates 1 to 3).
#   4. Python-wrapper path: load_robot on iiwa14 (the URDF is copied out of
#      grid_assets/robots; the wheel does not ship it), forward_dynamics on a
#      batch of 8 checked against the bundled RBDReference oracle.
#   5. Header-only path: grid-generate on that URDF, nvcc on a copy of
#      examples/cuda/inverse_dynamics_kernel_example.cu, and
#      examples/cuda/validate.py against RBDReference.
#
# The last line is "release fresh install: PASS" or "release fresh install: FAIL".
# A log of everything printed is kept at WORK/release_fresh_install.log.
# Runs on the lab box: needs nvcc, nvidia-smi, a GPU and network access for
# the framework wheels. GRID_RBD_BUILD_JOBS is passed through to the bindings
# (set it to 1 to keep one nvcc process at a time).
set -Eeuo pipefail
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_PY="${REPO_ROOT}/.venv/bin/python"
VENV_PY="${BUILD_PY}"
WORK=""
KEEP=0

usage() {
  cat <<EOF
Usage: bash tools/release_fresh_install.sh [--work DIR] [--keep] [--python PATH]

Build the sdist and wheel, install the repaired wheel into a throwaway venv
outside the checkout and exercise the Python-wrapper and header-only entry
points (RELEASING.md, "Installed-artifact gates"). Needs nvcc, nvidia-smi and
a GPU; downloads jax[cuda12] and a torch CUDA wheel into the venv.

  --work DIR     work directory (created; must not exist or must be empty).
                 Default: a fresh directory under \${TMPDIR:-/tmp}.
  --keep         keep the throwaway venv after a PASS (artifacts, logs and
                 the robot cache are always kept).
  --python PATH  interpreter used to create the throwaway venv.
                 Default: the checkout's .venv interpreter (its base
                 interpreter becomes the venv's).
  -h, --help     this text.

Environment: TORCH_INDEX_URL (default https://download.pytorch.org/whl/cu128),
GRID_RBD_BUILD_JOBS (bindings compile pool size), GRID_SM_ARCH (override the
nvcc -arch for the header-only example; default from nvidia-smi).
The build itself always uses ${REPO_ROOT}/.venv/bin/python, which needs the
build, auditwheel, patchelf and twine packages (RELEASING.md, "Build and inspect").
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --python) VENV_PY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${WORK}" ]; then
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/grid-release-check.XXXXXX")"
else
  if [ -e "${WORK}" ] && [ -n "$(ls -A "${WORK}" 2>/dev/null)" ]; then
    echo "ERROR: --work ${WORK} exists and is not empty; a stale cache would mask a compile" >&2
    exit 2
  fi
  mkdir -p "${WORK}"
fi
WORK="$(cd "${WORK}" && pwd)"
case "${WORK}" in
  "${REPO_ROOT}"/*|"${REPO_ROOT}") echo "ERROR: the work directory must be outside the checkout" >&2; exit 2 ;;
esac

LOG="${WORK}/release_fresh_install.log"
exec 3>&1 4>&2
exec > >(tee -a "${LOG}") 2>&1
TEE_PID=$!

STATUS=FAIL
finish() {
  local rc=$?
  if [ "${STATUS}" = PASS ] && [ "${KEEP}" -eq 0 ]; then
    rm -rf "${WORK}/venv"
  fi
  echo
  echo "work directory: ${WORK}"
  echo "log: ${LOG}"
  echo "release fresh install: ${STATUS}"
  exec 1>&3 2>&4
  wait "${TEE_PID}" 2>/dev/null || true
  exit "${rc}"
}
trap finish EXIT
trap 'echo "ERROR: step failed at line ${LINENO}" >&2; exit 1' ERR

step() { echo; echo "== $*"; }

[ -x "${BUILD_PY}" ] || { echo "ERROR: ${BUILD_PY} is missing; run install/base_install.sh first" >&2; exit 1; }
[ -x "${VENV_PY}" ] || { echo "ERROR: --python ${VENV_PY} is not executable" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc is not on PATH" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is not on PATH" >&2; exit 1; }
"${BUILD_PY}" -c "import build, auditwheel, twine" 2>/dev/null \
  || { echo "ERROR: install build auditwheel patchelf twine into ${REPO_ROOT}/.venv first" >&2; exit 1; }

COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d ' .')"
[ -n "${COMPUTE_CAP}" ] || { echo "ERROR: nvidia-smi reported no GPU" >&2; exit 1; }
ARCH="${GRID_SM_ARCH:-sm_${COMPUTE_CAP}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

step "versions"
echo "checkout: ${REPO_ROOT}"
echo "work:     ${WORK}"
echo "build python: ${BUILD_PY} ($("${BUILD_PY}" --version))"
echo "venv python:  ${VENV_PY} ($("${VENV_PY}" --version))"
nvcc --version | tail -1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo "nvcc -arch for the header-only example: ${ARCH}"

# ---------------------------------------------------------------------------
step "1/5 build sdist, wheel from the sdist, auditwheel, twine"
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
export PATH="$(dirname "${BUILD_PY}"):${PATH}"   # auditwheel needs patchelf
"${BUILD_PY}" -m build --outdir "${WORK}/dist"
"${BUILD_PY}" -m auditwheel show "${WORK}"/dist/*.whl
"${BUILD_PY}" -m auditwheel repair "${WORK}"/dist/*.whl --wheel-dir "${WORK}/wheelhouse"
"${BUILD_PY}" -m twine check "${WORK}"/dist/*.tar.gz "${WORK}"/wheelhouse/*.whl
WHEELS=("${WORK}"/wheelhouse/*.whl)
[ "${#WHEELS[@]}" -eq 1 ] || { echo "ERROR: expected exactly one repaired wheel, got ${WHEELS[*]}" >&2; exit 1; }
WHEEL="${WHEELS[0]}"
(cd "${WORK}" && sha256sum dist/*.tar.gz wheelhouse/*.whl | tee SHA256SUMS.txt)

# ---------------------------------------------------------------------------
step "2/5 fresh venv outside the checkout; install the repaired wheel"
# ---------------------------------------------------------------------------
VENV="${WORK}/venv"
"${VENV_PY}" -m venv "${VENV}"
VPY="${VENV}/bin/python"
"${VPY}" -m pip install --quiet --upgrade pip
"${VPY}" -m pip install --only-binary=:all: "${WHEEL}"
"${VPY}" -m pip install --quiet "jax[cuda12]"
"${VPY}" -m pip install --quiet torch --index-url "${TORCH_INDEX_URL}"
cd "${WORK}"
REPO_ROOT="${REPO_ROOT}" VENV="${VENV}" "${VPY}" -I - <<'PYCHECK'
import os, sys
repo, venv = os.environ["REPO_ROOT"], os.environ["VENV"]
bad = [p for p in sys.path if os.path.abspath(p or ".").startswith(repo)]
assert not bad, f"checkout on sys.path: {bad}"
import grid_rbd, grid_codegen, URDFParser, RBDReference
for m in (grid_rbd, grid_codegen, URDFParser, RBDReference):
    assert m.__file__.startswith(venv), m.__file__
import jax, torch
print(f"grid-rbd {grid_rbd.__version__} from {os.path.dirname(grid_rbd.__file__)}")
print(f"jax {jax.__version__}; torch {torch.__version__}; cuda build {torch.version.cuda}")
PYCHECK

# ---------------------------------------------------------------------------
step "3/5 tools/check_installed_package.py: CPU (GPU hidden), --gpu, --gpu --reload"
# ---------------------------------------------------------------------------
CHECKER="${REPO_ROOT}/tools/check_installed_package.py"
CUDA_VISIBLE_DEVICES='' "${VPY}" -I "${CHECKER}" --work "${WORK}/package-check-cpu"
"${VPY}" -I "${CHECKER}" --work "${WORK}/package-check" --gpu
"${VPY}" -I "${CHECKER}" --work "${WORK}/package-check" --gpu --reload

# ---------------------------------------------------------------------------
step "4/5 Python wrappers: load_robot(iiwa14) forward_dynamics vs RBDReference"
# ---------------------------------------------------------------------------
mkdir -p "${WORK}/grid_assets/robots" "${WORK}/examples/cuda" "${WORK}/cache"
cp "${REPO_ROOT}/grid_assets/robots/iiwa14.urdf" "${WORK}/grid_assets/robots/"
export GRID_RBD_CACHE_DIR="${WORK}/cache"
"${VPY}" -I - "${WORK}/grid_assets/robots/iiwa14.urdf" <<'PYLOAD'
import sys
import numpy as np
import grid_rbd
from URDFParser import URDFParser
from RBDReference import RBDReference

urdf = sys.argv[1]
robot = grid_rbd.load_robot(urdf, algorithm_list=["forward_dynamics"], enable_mujoco_kernels=False)
B = 8
rng = np.random.default_rng(0)
q = rng.uniform(-1.0, 1.0, (B, robot.nq)).astype(np.float32)
qd = rng.uniform(-1.0, 1.0, (B, robot.nv)).astype(np.float32)
tau = rng.uniform(-5.0, 5.0, (B, robot.nv)).astype(np.float32)
qdd = np.asarray(robot.forward_dynamics(q, qd, tau))
assert qdd.shape == (B, robot.nv), qdd.shape
ref = RBDReference(URDFParser().parse(urdf, floating_base=False))
worst = 0.0
for i in range(B):
    expected = np.asarray(ref.forward_dynamics(q[i].astype(float), qd[i].astype(float), tau[i].astype(float))).ravel()
    rel = np.linalg.norm(qdd[i] - expected) / max(np.linalg.norm(expected), 1e-9)
    worst = max(worst, rel)
    assert rel < 1e-4, f"forward_dynamics row {i}: rel_err={rel:.3e}"
print(f"PASS load_robot iiwa14 forward_dynamics: shape {qdd.shape}, worst rel_err {worst:.3e} (tol 1e-4)")
PYLOAD

# ---------------------------------------------------------------------------
step "5/5 header-only: grid-generate, nvcc, examples/cuda/validate.py vs RBDReference"
# ---------------------------------------------------------------------------
cp "${REPO_ROOT}/examples/cuda/inverse_dynamics_kernel_example.cu" \
   "${REPO_ROOT}/examples/cuda/validate.py" "${WORK}/examples/cuda/"
cd "${WORK}/examples/cuda"
"${VENV}/bin/grid-generate" "${WORK}/grid_assets/robots/iiwa14.urdf" \
    --algorithm-list inverse_dynamics -o "${WORK}/examples/cuda/grid.cuh" | tail -1
nvcc -arch="${ARCH}" -std=c++17 -I "${WORK}/examples/cuda" \
    "${WORK}/examples/cuda/inverse_dynamics_kernel_example.cu" -o "${WORK}/examples/cuda/id_example"
"${WORK}/examples/cuda/id_example" > "${WORK}/examples/cuda/id_example.out"
"${VPY}" -I "${WORK}/examples/cuda/validate.py" < "${WORK}/examples/cuda/id_example.out"

# ---------------------------------------------------------------------------
step "summary"
# ---------------------------------------------------------------------------
echo "wheel: $(basename "${WHEEL}")"
echo "sha256: $(sha256sum "${WHEEL}" | cut -d' ' -f1)"
echo "python: $("${VPY}" --version)"
nvcc --version | tail -1
"${VPY}" -I -c "import jax, torch; print(f'jax {jax.__version__}; torch {torch.__version__}')"
STATUS=PASS
