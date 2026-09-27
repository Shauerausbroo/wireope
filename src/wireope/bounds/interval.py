"""Confidence intervals on bounded rung observations.

Both directions are needed: the safety constraint is written on an upper confidence limit
of the risk, and the selection rule on a lower confidence limit of the graded value. The
limits are built from an effective sample size rather than from the record count, because
adding records that do not raise the count of the rarest rung does not tighten the bound.

Ref: Sec. 3.4, Eq. (7); the high-confidence construction is the established one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    radius: float
    delta: float
    bound: str

    @property
    def width(self) -> float:
        return self.upper - self.lower


def weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    total = float(np.sum(weights))
    if total <= EPS:
        return 0.0
    return float(np.sum(values * weights) / total)


def weighted_variance(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    mean = weighted_mean(values, weights)
    total = float(np.sum(weights))
    if total <= EPS:
        return 0.0
    variance = float(np.sum(weights * (values - mean) ** 2) / total)
    return max(variance, 0.0)


def hoeffding_radius(
    effective_sample_size: float,
    delta: float,
    bounded_range: float,
    union_size: int = 1,
) -> float:
    if effective_sample_size <= 0.0:
        return float("inf")
    union = max(union_size, 1)
    argument = np.log(2.0 * union / max(delta, EPS))
    return float(bounded_range * np.sqrt(max(argument, 0.0) / (2.0 * effective_sample_size)))


def empirical_bernstein_radius(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    effective_sample_size: float,
    delta: float,
    bounded_range: float,
    union_size: int = 1,
) -> float:
    """Empirical-Bernstein radius with the variance actually observed at the rung."""
    if effective_sample_size <= 0.0:
        return float("inf")
    union = max(union_size, 1)
    logarithm = np.log(3.0 * union / max(delta, EPS))
    variance = weighted_variance(values, weights)
    leading = np.sqrt(max(2.0 * variance * logarithm / effective_sample_size, 0.0))
    residual = 3.0 * bounded_range * logarithm / effective_sample_size
    return float(leading + residual)


def interval_from_radius(point: float, radius: float, delta: float, bound: str) -> Interval:
    return Interval(
        point=point,
        lower=point - radius,
        upper=point + radius,
        radius=radius,
        delta=delta,
        bound=bound,
    )


def lower_confidence_bound(point: float, radius: float) -> float:
    return float(point - radius)


def upper_confidence_bound(point: float, radius: float) -> float:
    return float(point + radius)


def interval_for_rung(
    observations: NDArray[np.float64],
    weights: NDArray[np.float64],
    effective_sample_size: float,
    delta: float,
    bounded_range: float,
    bound: str,
    union_size: int,
) -> Interval:
    point = weighted_mean(observations, weights)
    if bound == "empirical_bernstein":
        radius = empirical_bernstein_radius(
            observations, weights, effective_sample_size, delta, bounded_range, union_size
        )
    elif bound == "hoeffding":
        radius = hoeffding_radius(effective_sample_size, delta, bounded_range, union_size)
    else:
        raise ValueError(f"unsupported bound '{bound}'")
    return interval_from_radius(point, radius, delta, bound)


def reduces_to_hoeffding(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    effective_sample_size: float,
    delta: float,
    bounded_range: float,
    union_size: int,
) -> float:
    """Ratio of the Bernstein radius to the Hoeffding radius at the maximum-variance extreme.

    At sigma = B/2 the empirical-Bernstein leading term equals the Hoeffding term, so the
    ratio approaches one from below as the residual correction vanishes.
    """
    bernstein = empirical_bernstein_radius(
        values, weights, effective_sample_size, delta, bounded_range, union_size
    )
    hoeffding = hoeffding_radius(effective_sample_size, delta, bounded_range, union_size)
    if not np.isfinite(hoeffding) or hoeffding <= 0.0:
        return float("nan")
    return float(bernstein / hoeffding)
