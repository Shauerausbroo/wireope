#!/usr/bin/env bash
# Run the two verification layers and rewrite the root artefacts. Run this last, after
# every other edit, so the integrity manifest digests the final tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m wireope.cli.verify --repo-root "${REPO_ROOT}" "$@"
