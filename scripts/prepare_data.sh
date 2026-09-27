#!/usr/bin/env bash
# Assemble the release cohort and the whole-site partitions.
# The clinical cohort is held under data-sharing agreements, so this script never
# touches a record; it writes the cohort the executable studies run on.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m wireope.cli.reconstruct \
  --repo-root "${REPO_ROOT}" \
  --experiment main \
  --out "${REPO_ROOT}/artefacts/traces" \
  "$@"
