#!/usr/bin/env bash
# Estimate every candidate on the fitted behaviour and outcome models, then certify the
# strategy library under the support constraint. Writes the plain-text report tables.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
EXPERIMENT="${EXPERIMENT:-main}"

python -m wireope.cli.estimate --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
python -m wireope.cli.certify --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
python -m wireope.cli.studies --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
