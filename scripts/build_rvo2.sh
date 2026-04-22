#!/usr/bin/env bash
# RVO2 helpers for MAPPO (conda env swE2).
#
# Default: install Python-RVO2 (sybrenstuvel/Python-RVO2 → `import rvo2`). Same as
#   bash scripts/install_python_rvo2.sh
#
# Optional: build upstream C++ RVO2 (snape/RVO2) when you have a source tree — MAPPO
# does not require this for `import rvo2`; Python-RVO2 bundles its own C++ RVO2.
#
# Upstream C++ CMake needs >= 3.26 unless you pass --relax-cmake-min (patches clone to 3.22).
#
# Usage:
#   bash scripts/build_rvo2.sh                      # Python rvo2 only (recommended)
#   bash scripts/build_rvo2.sh --python-only      # same as default
#   bash scripts/build_rvo2.sh --with-cpp        # also build ${RVO2_ROOT} if present
#   bash scripts/build_rvo2.sh --with-cpp /path/to/RVO2
#   bash scripts/build_rvo2.sh --cpp-only /path/to/RVO2   # C++ only, no pip install
#   bash scripts/build_rvo2.sh --relax-cmake-min --with-cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RELAX_CMAKE_MIN=0
CLEAN_BUILD=1
RVO2_ROOT="${RVO2_ROOT:-${HOME}/RVO2}"
DO_PYTHON=1
DO_CPP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --relax-cmake-min) RELAX_CMAKE_MIN=1; shift ;;
    --no-clean) CLEAN_BUILD=0; shift ;;
    --clean) CLEAN_BUILD=1; shift ;;
    --python-only) DO_CPP=0; DO_PYTHON=1; shift ;;
    --cpp-only) DO_CPP=1; DO_PYTHON=0; shift ;;
    --with-cpp) DO_CPP=1; shift ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      RVO2_ROOT="$1"
      shift
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /opt/conda/bin/conda ]]; then
    export PATH="/opt/conda/bin:${PATH}"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate swE2

PY="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Expected python at ${PY} (env swE2 missing?)" >&2
  exit 1
fi

install_python_rvo2() {
  # Drop ~/.local/bin so pip's cmake shim does not break subprocess cmake (CMAKE_ROOT).
  PATH_CLEAN=""
  IFS=':' read -r -a _parts <<< "${PATH}"
  for x in "${_parts[@]}"; do
    [[ -z "${x}" ]] && continue
    case "${x}" in
      "${HOME}/.local/bin" | "${HOME}/.local/bin/") continue ;;
    esac
    PATH_CLEAN+="${PATH_CLEAN:+:}${x}"
  done
  if [[ -x /usr/bin/cmake ]]; then
    export PATH="/usr/bin:${CONDA_PREFIX}/bin:${PATH_CLEAN}"
  else
    export PATH="${CONDA_PREFIX}/bin:${PATH_CLEAN}"
  fi
  hash -r

  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found. Install: sudo apt install cmake  OR  conda install -y -n swE2 cmake" >&2
    exit 1
  fi
  if ! cmake --version >/dev/null 2>&1; then
    echo "'cmake --version' failed. Try: \"${PY}\" -m pip uninstall cmake" >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "git not found." >&2
    exit 1
  fi
  if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    echo "Need g++ (build-essential)." >&2
    exit 1
  fi

  echo "======== Python-RVO2 (import rvo2) ========"
  echo "Using:  $("${PY}" -V)"
  echo "Using:  $(command -v cmake) — $(cmake --version | head -n1)"
  # Default pip build isolation re-downloads setuptools from PyPI → timeouts look like
  # "No matching distribution for setuptools". Pre-install build deps and skip isolation.
  export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
  "${PY}" -m pip install -U pip "cython>=3.0" "setuptools>=65" wheel
  "${PY}" -m pip install --no-build-isolation "git+https://github.com/sybrenstuvel/Python-RVO2.git"
  "${PY}" -c "import rvo2; print('import rvo2 OK:', getattr(rvo2, '__file__', rvo2))"
}

build_cpp_rvo2() {
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r

  if [[ ! -f "${RVO2_ROOT}/CMakeLists.txt" ]]; then
    echo "C++ RVO2 source not found at: ${RVO2_ROOT} (missing CMakeLists.txt)" >&2
    echo "Clone: git clone https://github.com/snape/RVO2.git \"\${HOME}/RVO2\"" >&2
    exit 1
  fi

  cmake_version_line="$(cmake --version 2>/dev/null | head -n1 || true)"
  cmake_ver="$(echo "${cmake_version_line}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)"
  if [[ -z "${cmake_ver}" ]]; then
    echo "cmake not found after activating swE2. Install with:" >&2
    echo "  conda install -y -n swE2 'cmake>=3.26'" >&2
    exit 1
  fi

  cmake_major="${cmake_ver%%.*}"
  rest="${cmake_ver#*.}"
  cmake_minor="${rest%%.*}"

  if [[ "${RELAX_CMAKE_MIN}" -eq 1 ]]; then
    echo "Patching cmake_minimum_required 3.26 -> 3.22 in ${RVO2_ROOT}/CMakeLists.txt (backup .bak)"
    sed -i.bak 's/cmake_minimum_required(VERSION 3\.26)/cmake_minimum_required(VERSION 3.22)/' \
      "${RVO2_ROOT}/CMakeLists.txt"
  fi

  need_patch=0
  if [[ "${cmake_major}" -lt 3 ]] || [[ "${cmake_major}" -eq 3 && "${cmake_minor}" -lt 26 ]]; then
    need_patch=1
  fi

  if [[ "${need_patch}" -eq 1 && "${RELAX_CMAKE_MIN}" -eq 0 ]]; then
    echo "Your CMake is ${cmake_ver} but snape/RVO2 requires >= 3.26 (${cmake_version_line})." >&2
    echo "  conda install -y -n swE2 'cmake>=3.26'   OR   bash scripts/build_rvo2.sh --relax-cmake-min --with-cpp ..." >&2
    exit 1
  fi

  BUILD_DIR="${RVO2_ROOT}/_build"
  if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
    echo "Removing ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
  mkdir -p "${BUILD_DIR}"

  echo "======== C++ RVO2 (snape/RVO2) ========"
  echo "RVO2:   ${RVO2_ROOT}"
  echo "Build:  ${BUILD_DIR}"
  echo "CMake:  $(command -v cmake) (${cmake_ver})"
  echo "--------"

  cmake -S "${RVO2_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DOCUMENTATION:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DENABLE_HARDENING:BOOL=ON \
    -DENABLE_OPENMP:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}"

  cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"

  echo "--------"
  echo "Built under ${BUILD_DIR}. Optional: cmake --install \"${BUILD_DIR}\""
}

if [[ "${DO_CPP}" -eq 1 ]]; then
  build_cpp_rvo2
fi

if [[ "${DO_PYTHON}" -eq 1 ]]; then
  install_python_rvo2
fi

echo "Done."
