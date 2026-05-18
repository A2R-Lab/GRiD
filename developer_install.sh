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

# Build the Pinocchio second-order RNEA pybind11 extension used as the golden
# oracle in test/pinocchio_equivalents/tests/test_second_order_pinocchio_equivalence.py
# Pinocchio 3.x ships the C++ implementation but does not expose it to Python by
# default. The extension wraps `pinocchio::ComputeRNEASecondOrderDerivatives` and
# is built in-place so the loader at test/pinocchio_equivalents/pin_so_ext/__init__.py
# can import it directly.
PIN_SO_EXT_DIR="${SCRIPT_DIR}/test/pinocchio_equivalents/pin_so_ext"
# The `pin` wheel installs pinocchio.pc under the venv's cmeel.prefix rather
# than on the system pkg-config path, so point pkg-config at it.
CMEEL_PC_DIR="$("${VENV_DIR}/bin/python" -c 'import sysconfig, pathlib, cmeel.config; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / cmeel.config.CMEEL_PREFIX / "lib" / "pkgconfig")')"
export PKG_CONFIG_PATH="${CMEEL_PC_DIR}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
(cd "${PIN_SO_EXT_DIR}" && "${VENV_DIR}/bin/python" setup.py build_ext --inplace)

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/test/run_tests.py" --prepare-models --tier "${TIER}"
