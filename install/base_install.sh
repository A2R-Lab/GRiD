#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
# Peers first: the editable finder maps the packages present at install time,
# so a non-recursive clone must populate external/ BEFORE `pip install -e`.
git -C "${REPO_ROOT}" submodule update --init --recursive
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}"
