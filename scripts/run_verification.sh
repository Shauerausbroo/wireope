#!/usr/bin/env bash
# Run the two verification layers and rewrite the root artefacts. Run this last, after
# every other edit, so the integrity manifest digests the final tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Resolve the interpreter explicitly: many systems ship `python3` only, and a launcher that
# assumes `python` runs nothing at all. PYTHON overrides the search.
PY_BIN="${PYTHON:-}"
if [[ -z "${PY_BIN}" ]]; then
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PY_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PY_BIN}" ]]; then
  echo "run_verification: no Python interpreter found; set PYTHON to its path" >&2
  exit 1
fi

exec "${PY_BIN}" -m wireope.cli.verify --repo-root "${REPO_ROOT}" "$@"
