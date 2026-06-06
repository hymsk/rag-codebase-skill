#!/usr/bin/env bash
# Compatibility entry point: installs only this project's Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RAG_VENV:-$HOME/.rag/.venv}"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 is required"'
if [ ! -e "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
elif [ ! -f "$VENV/pyvenv.cfg" ] || [ ! -x "$VENV/bin/python" ]; then
  printf 'Refusing to use a non-venv path: %s\n' "$VENV" >&2
  exit 1
fi
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
printf 'Environment ready: %s\n' "$VENV"
printf 'Activate with: source "%s/bin/activate"\n' "$VENV"
printf '%s\n' 'No shell/profile/global package-manager configuration was modified.'
printf '%s\n' 'Model use is cache-only by default. --download-model explicitly permits a model download.'
