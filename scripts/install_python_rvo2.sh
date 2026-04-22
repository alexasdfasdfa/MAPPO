#!/usr/bin/env bash
# Thin wrapper: installs Python-RVO2 into swE2 (see build_rvo2.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_rvo2.sh" --python-only "$@"
