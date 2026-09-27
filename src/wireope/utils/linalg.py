"""Numerical primitives used across the estimator and bound modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EPS = 1e-12


def as_float_array(values: ArrayLike, *, ndim: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, received {array.ndim}")
    return array


def clip_probability(values: ArrayLike, low: float = EPS, high: float = 1.0 - EPS) -> NDArray[np.float64]:
    return np.clip(np.asarray(values, dtype=np.float64), low, high)


def logit(values: ArrayLike) -> NDArray[np.float64]:
    probability = clip_probability(values)
    return np.log(probability) - np.log1p(-probability)


def sigmoid(values: ArrayLike) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def stable_softmax(values: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    peak = np.max(array, axis=axis, keepdims=True)
    shifted = array - peak
    exponent = np.exp(shifted)
    total = np.sum(exponent, axis=axis, keepdims=True)
    result: NDArray[np.float64] = exponent / total
    return result


def stable_logsumexp(values: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    peak = np.max(array, axis=axis, keepdims=True)
    shifted = array - peak
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result: NDArray[np.float64] = np.squeeze(peak, axis=axis) + np.log(np.squeeze(total, axis=axis))
    return result


def trapezoid(values: ArrayLike, spacing: float = 1.0) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if array.size == 1:
        return float(array[0]) * spacing
    total: float = float(np.trapezoid(array, dx=spacing))
    return total


def weighted_average(values: ArrayLike, weights: ArrayLike) -> float:
    numer = np.asarray(values, dtype=np.float64)
    denom = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(denom))
    if total <= EPS:
        return 0.0
    return float(np.sum(numer * denom) / total)


def normalize_weights(weights: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(weights, dtype=np.float64)
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(np.sum(array))
    if total <= EPS:
        raise ValueError("weights sum to zero")
    normalised: NDArray[np.float64] = array / total
    return normalised


def coefficient_of_variation(weights: ArrayLike) -> float:
    array = np.asarray(weights, dtype=np.float64)
    mean = float(np.mean(array))
    if mean <= EPS:
        return float("inf")
    return float(np.std(array) / mean)


def safe_ratio(numerator: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    result: NDArray[np.float64] = num / np.maximum(den, EPS)
    return result
