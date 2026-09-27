#!/usr/bin/env bash
# Fit the behaviour model, the rung-wise outcome model and the selective state-space
# hazard head. Single-process default; set WORLD_SIZE>1 to launch through torchrun.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
WORLD_SIZE="${WORLD_SIZE:-1}"
EXPERIMENT="${EXPERIMENT:-main}"

if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${WORLD_SIZE}" -m wireope.cli.fit \
    --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
fi

exec python -m wireope.cli.fit \
  --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
