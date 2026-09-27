"""Bootstrap intervals and paired contrasts for the safety and utility endpoints.

Intervals are built from a fixed number of resamples with a caller-supplied seed, so a rerun
reproduces every reported bound byte for byte.

Ref: Sec. 3.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.seed import numpy_generator


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    resamples: int
    level: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def bootstrap_interval(
    statistic: NDArray[np.float64],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile interval over the supplied per-resample statistic values."""
    values = np.asarray(statistic, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0, level)
    tail = 0.5 * (1.0 - level)
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return BootstrapInterval(
        point=float(np.mean(values)),
        lower=float(lower),
        upper=float(upper),
        resamples=int(values.size),
        level=level,
    )


def bootstrap_statistic(
    sample: NDArray[np.float64],
    statistic: str,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample one vector and return the percentile interval of the chosen statistic."""
    values = np.asarray(sample, dtype=np.float64)
    rng = numpy_generator(seed)
    if values.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0, level)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, values.size, size=values.size)
        draws[index] = _apply(statistic, values[pick])
    return bootstrap_interval(draws, level=level, seed=seed)


def paired_bootstrap_difference(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Interval on the mean paired difference, resampling the pairs together."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired bootstrap needs equally sized samples")
    rng = numpy_generator(seed)
    difference = left - right
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, difference.size, size=difference.size)
        draws[index] = float(np.mean(difference[pick]))
    interval = bootstrap_interval(draws, level=level, seed=seed)
    return BootstrapInterval(
        point=float(np.mean(difference)),
        lower=interval.lower,
        upper=interval.upper,
        resamples=interval.resamples,
        level=level,
    )


def _apply(statistic: str, values: NDArray[np.float64]) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "std":
        return float(np.std(values))
    if statistic == "rate":
        return float(np.mean(values > 0.0))
    raise ValueError(f"unsupported bootstrap statistic '{statistic}'")


def clustered_bootstrap_interval(
    values: NDArray[np.float64],
    clusters: NDArray[np.str_],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample whole clusters rather than rows, so within-site dependence is respected."""
    rng = numpy_generator(seed)
    names = np.asarray(sorted(set(clusters.tolist())))
    if names.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0, level)
    indices = [np.flatnonzero(clusters == name) for name in names]
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, len(indices), size=len(indices))
        pooled = np.concatenate([values[indices[position]] for position in pick])
        draws[index] = float(np.mean(pooled))
    return bootstrap_interval(draws, level=level, seed=seed)


def clustered_difference_interval(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    clusters: NDArray[np.str_],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Cluster-resampled interval on a difference of means that share cluster membership."""
    if first.shape != second.shape:
        raise ValueError("cluster-resampled contrast needs equally sized samples")
    rng = numpy_generator(seed)
    names = np.asarray(sorted(set(clusters.tolist())))
    indices = [np.flatnonzero(clusters == name) for name in names]
    difference = first - second
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, len(indices), size=len(indices))
        pooled = np.concatenate([difference[indices[position]] for position in pick])
        draws[index] = float(np.mean(pooled))
    interval = bootstrap_interval(draws, level=level, seed=seed)
    return BootstrapInterval(
        point=float(np.mean(difference)),
        lower=interval.lower,
        upper=interval.upper,
        resamples=interval.resamples,
        level=level,
    )
