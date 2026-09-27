"""Fixed non-negative ladder weights carried by the graded crossing value functional.

Ref: Sec. 3.1, Eq. (3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wireope.config import LadderWeightsConfig
from wireope.utils.linalg import normalize_weights


def geometric_weights(depth: int, value: float) -> NDArray[np.float64]:
    """Geometric decay put on the shallower rungs so the endpoint keeps the largest weight."""
    if depth < 1:
        raise ValueError("depth must be at least one")
    if not 0.0 < value <= 1.0:
        raise ValueError("geometric value must lie in (0, 1]")
    exponents = np.arange(depth, dtype=np.float64)
    raw = value**exponents
    return np.asarray(raw, dtype=np.float64)


def linear_weights(depth: int) -> NDArray[np.float64]:
    if depth < 1:
        raise ValueError("depth must be at least one")
    return np.asarray(np.arange(1, depth + 1, dtype=np.float64), dtype=np.float64)


def uniform_weights(depth: int) -> NDArray[np.float64]:
    if depth < 1:
        raise ValueError("depth must be at least one")
    return np.full(depth, 1.0 / depth, dtype=np.float64)


def build_weights(depth: int, config: LadderWeightsConfig) -> tuple[float, ...]:
    if config.scheme == "geometric":
        raw = geometric_weights(depth, config.value)
    elif config.scheme == "linear":
        raw = linear_weights(depth)
    elif config.scheme == "uniform":
        raw = uniform_weights(depth)
    else:
        raise ValueError(f"unsupported weight scheme '{config.scheme}'")
    if config.require_non_negative and np.any(raw < 0.0):
        raise ValueError("ladder weights must be non-negative")
    final = normalize_weights(raw) if config.normalise else raw
    return tuple(float(item) for item in final)


def weight_ratio(weights: NDArray[np.float64]) -> float:
    """Ratio between the largest and smallest rung weight; a large ratio concentrates the value."""
    positive = weights[weights > 0.0]
    if positive.size == 0:
        raise ValueError("no positive ladder weight")
    return float(np.max(positive) / np.min(positive))


def graded_value(indicators: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.asarray(indicators, dtype=np.float64) @ np.asarray(
        weights, dtype=np.float64
    )
    return result
