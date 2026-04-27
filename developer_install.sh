#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TIER="${PINOCCHIO_EQUIVALENCE_TIER:-smoke}"

"${SCRIPT_DIR}/base_install.sh"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-dev-pinocchio-equivalence.txt"
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/scripts/fetch_test_models.py" --tier "${TIER}" --cache-only
