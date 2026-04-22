#!/usr/bin/env bash
# Create conda environment "swE2" for MAPPO (Python 3.9 + requirements.txt).
#
# Usage (from anywhere):
#   bash /path/to/MAPPO/scripts/create_swE2_env.sh
#
# GPU (CUDA) PyTorch: after creation, reinstall torch per your CUDA version, e.g.:
#   conda activate swE2
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#
# requirements.txt includes Python-RVO2 (import rvo2). Build needs system tools:
#   sudo apt install cmake git build-essential
# Or: bash scripts/build_rvo2.sh   # default installs Python-RVO2 (import rvo2)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ="${REPO_ROOT}/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Load Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx 'swE2'; then
  echo "Conda env 'swE2' already exists. Remove with: conda env remove -n swE2" >&2
  exit 1
fi

echo "Creating conda env swE2 (python=3.9)..."
conda create -y -n swE2 python=3.9 pip

echo "Installing pip dependencies from ${REQ}..."
conda run -n swE2 pip install --upgrade pip
conda run -n swE2 pip install -r "${REQ}"

echo "Done. Activate with:  conda activate swE2"
echo "Repo root: ${REPO_ROOT}"
