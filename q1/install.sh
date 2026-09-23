#!/usr/bin/env bash
set -euo pipefail
q1_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv --system-site-packages "$q1_project_root/.venv-q1"
"$q1_project_root/.venv-q1/bin/python" -m pip install -r "$q1_project_root/q1/requirements.txt"
# The upstream package pins old NumPy/PyTorch-adjacent dependencies. The adapters
# are tested against the workspace environment; do not downgrade it transitively.
"$q1_project_root/.venv-q1/bin/python" -m pip install --no-deps openface-test==0.1.26
cd "$q1_project_root"
"$q1_project_root/.venv-q1/bin/python" -m q1 doctor
