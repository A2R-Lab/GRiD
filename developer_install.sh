#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TIER="${PINOCCHIO_EQUIVALENCE_TIER:-smoke}"

# System build deps for the pinocchio pybind11 extension built below. The `pin`
# wheel ships pinocchio.pc inside the venv via cmeel, but that .pc file
# references eigen3 and urdfdom_headers which cmeel does not provide.
APT_PKGS=(pkg-config g++ libeigen3-dev liburdfdom-headers-dev)
if command -v apt-get >/dev/null 2>&1; then
  missing=()
  for pkg in "${APT_PKGS[@]}"; do
    if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
      missing+=("${pkg}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Installing system packages: ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
  fi
else
  echo "Warning: apt-get not found. Ensure these packages are installed: ${APT_PKGS[*]}" >&2
fi

"${SCRIPT_DIR}/base_install.sh"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-dev.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/docs/requirements.txt"

# Optional: install CppADCodeGen so Pinocchio's code-generated algos (FD,
# FD_DU, IDSVA_SO, FDSVA_SO) populate in the multi-version benchmark. Without
# it, the Pinocchio harness still works for direct algos (ID, ABA, CRBA,
# ee_pose, ee_pose_gradient, ID_DU) and the benchmark renders the codegen
# cells as `—`. Skip with `SKIP_CPPADCG_INSTALL=1`.
#
# CppADCodeGen is header-only but not in apt; we clone + cmake-install to
# /usr/local. Its parent (CppAD) IS available via apt as `libcppad-dev`.
if [[ -z "${SKIP_CPPADCG_INSTALL:-}" ]]; then
  if ! pkg-config --exists cppadcg 2>/dev/null; then
    echo "CppADCodeGen not found via pkg-config — installing for Pinocchio bench codegen support."
    if command -v apt-get >/dev/null 2>&1; then
      if ! dpkg -s libcppad-dev >/dev/null 2>&1; then
        echo "Installing CppAD parent (apt: libcppad-dev)..."
        sudo apt-get update
        sudo apt-get install -y libcppad-dev cmake
      fi
    else
      echo "Warning: apt-get not found. Install CppAD manually (the apt package on Debian/Ubuntu is libcppad-dev)." >&2
    fi
    CPPADCG_SRC="${SCRIPT_DIR}/.cache/CppADCodeGen"
    if [[ ! -d "${CPPADCG_SRC}" ]]; then
      mkdir -p "$(dirname "${CPPADCG_SRC}")"
      echo "Cloning CppADCodeGen → ${CPPADCG_SRC}"
      git clone --depth 1 https://github.com/joaoleal/CppADCodeGen.git "${CPPADCG_SRC}"
    fi
    mkdir -p "${CPPADCG_SRC}/build"
    (cd "${CPPADCG_SRC}/build" \
      && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DGOOGLETEST_GIT=OFF -DCREATE_DOCUMENTATION=OFF -DUSE_VALGRIND=OFF \
      && sudo make install)
  else
    echo "CppADCodeGen already installed (pkg-config --exists cppadcg) — skipping."
  fi
fi

# Optional: frax JAX reference (one of the bench columns in
# test/benchmarks/run_multi_version.py). Small pure-Python wheel; depends on
# jax which requirements-dev should have if you're running the bench. Skip
# with `SKIP_FRAX_INSTALL=1`.
if [[ -z "${SKIP_FRAX_INSTALL:-}" ]]; then
  if ! "${VENV_DIR}/bin/python" -c "import frax" >/dev/null 2>&1; then
    echo "Installing frax for bench multi-version frax column..."
    "${VENV_DIR}/bin/python" -m pip install frax
  fi
fi

# Build the Pinocchio second-order RNEA pybind11 extension used as the golden
# oracle in RBDReference/tests/test_second_order_pinocchio_equivalence.py
# Pinocchio 3.x ships the C++ implementation but does not expose it to Python by
# default. The extension wraps `pinocchio::ComputeRNEASecondOrderDerivatives` and
# is built in-place so the loader at RBDReference/equivalents/pin_so_ext/__init__.py
# can import it directly.
PIN_SO_EXT_DIR="${SCRIPT_DIR}/RBDReference/equivalents/pin_so_ext"
# The `pin` wheel installs pinocchio.pc under the venv's cmeel.prefix rather
# than on the system pkg-config path, so point pkg-config at it.
CMEEL_PC_DIR="$("${VENV_DIR}/bin/python" -c 'import sysconfig, pathlib, cmeel.config; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / cmeel.config.CMEEL_PREFIX / "lib" / "pkgconfig")')"
export PKG_CONFIG_PATH="${CMEEL_PC_DIR}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
(cd "${PIN_SO_EXT_DIR}" && "${VENV_DIR}/bin/python" setup.py build_ext --inplace)

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/test/run_tests.py" --prepare-models --tier "${TIER}"
