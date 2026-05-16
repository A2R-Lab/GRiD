#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TIER="${PINOCCHIO_EQUIVALENCE_TIER:-smoke}"

"${SCRIPT_DIR}/base_install.sh"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-dev.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/docs/requirements.txt"

# Build the Pinocchio second-order RNEA pybind11 extension used as the golden
# oracle in test/pinocchio_equivalents/tests/test_second_order_pinocchio_equivalence.py
# Pinocchio 3.x ships the C++ implementation but does not expose it to Python by
# default. The extension wraps `pinocchio::ComputeRNEASecondOrderDerivatives` and
# is built in-place so the loader at test/pinocchio_equivalents/pin_so_ext/__init__.py
# can import it directly. Requires pkg-config + g++ + pinocchio development
# headers (provided by base_install.sh).
PIN_SO_EXT_DIR="${SCRIPT_DIR}/test/pinocchio_equivalents/pin_so_ext"
(cd "${PIN_SO_EXT_DIR}" && "${VENV_DIR}/bin/python" setup.py build_ext --inplace)

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/test/run_tests.py" --prepare-models --tier "${TIER}"
