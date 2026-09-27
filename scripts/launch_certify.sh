#!/usr/bin/env bash
# Estimate every candidate on the fitted behaviour and outcome models, then certify the
# strategy library under the support constraint. Writes the plain-text report tables.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
EXPERIMENT="${EXPERIMENT:-main}"

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
  echo "launch_certify: no Python interpreter found; set PYTHON to its path" >&2
  exit 1
fi

"${PY_BIN}" -m wireope.cli.estimate --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
"${PY_BIN}" -m wireope.cli.certify --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
"${PY_BIN}" -m wireope.cli.studies --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
