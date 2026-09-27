#!/usr/bin/env bash
# Assemble the release cohort and the whole-site partitions.
# The clinical cohort is held under data-sharing agreements, so this script never
# touches a record; it writes the cohort the executable studies run on.
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
  echo "prepare_data: no Python interpreter found; set PYTHON to its path" >&2
  exit 1
fi

exec "${PY_BIN}" -m wireope.cli.reconstruct \
  --repo-root "${REPO_ROOT}" \
  --experiment main \
  --out "${REPO_ROOT}/artefacts/traces" \
  "$@"
