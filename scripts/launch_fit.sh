#!/usr/bin/env bash
# Fit the behaviour model, the rung-wise outcome model and the selective state-space
# hazard head. Single-process default; set WORLD_SIZE>1 to launch through torchrun.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
WORLD_SIZE="${WORLD_SIZE:-1}"
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
  echo "launch_fit: no Python interpreter found; set PYTHON to its path" >&2
  exit 1
fi

if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    exec torchrun --nproc_per_node="${WORLD_SIZE}" -m wireope.cli.fit \
      --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
  fi
  exec "${PY_BIN}" -m torch.distributed.run --nproc_per_node="${WORLD_SIZE}" -m wireope.cli.fit \
    --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
fi

exec "${PY_BIN}" -m wireope.cli.fit \
  --repo-root "${REPO_ROOT}" --experiment "${EXPERIMENT}" "$@"
