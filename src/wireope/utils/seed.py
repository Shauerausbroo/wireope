"""Deterministic seeding for every process that allocates randomness."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch, and pin the hashing seed for the process.

    Called before any tensor allocation; a seed applied after initialisation does not
    constrain the draw it was meant to constrain.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
