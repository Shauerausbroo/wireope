"""Empirical-Bernstein version of the deviation bound used to report estimates.

The reported form keeps the union bound over the K rungs, replaces the worst-case variance
by the residual actually observed at the binding rung, and adds the small-sample residual
correction. At the maximum-variance extreme sigma_k = B/2 the leading term coincides with
the Hoeffding term of Eq. (7), so the two forms agree exactly there and the Bernstein form
is strictly tighter wherever the observed residuals are better conformed.

Ref: Sec. 3.4 and Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.hoeffding import effective_sample_size_minimum
from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class EmpiricalBernsteinBound:
    radius: float
    leading_term: float
    residual_correction: float
    effective_sample_size: float
    variance_binding: float
    depth: int
    delta: float

    @property
    def total(self) -> float:
        return float(self.leading_term + self.residual_correction)


def per_rung_variance(
    observations: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weighted variance of each rung's observation under the reconstructed weights."""
    weights_array = np.asarray(weights, dtype=np.float64)
    total = max(float(np.sum(weights_array)), EPS)
    mean = np.sum(observations * weights_array[:, None], axis=0) / total
    squared = np.sum(((observations - mean) ** 2) * weights_array[:, None], axis=0) / total
    return np.asarray(np.maximum(squared, 0.0), dtype=np.float64)


def empirical_bernstein_bound(
    observations: NDArray[np.float64],
    weights: NDArray[np.float64],
    per_rung_counts: NDArray[np.float64],
    delta: float,
    bounded_range: float,
) -> EmpiricalBernsteinBound:
    depth = int(observations.shape[1])
    counts = np.asarray(per_rung_counts, dtype=np.float64)
    effective = effective_sample_size_minimum(counts)
    if effective <= 0.0 or depth <= 0:
        return EmpiricalBernsteinBound(
            radius=float("inf"),
            leading_term=float("inf"),
            residual_correction=0.0,
            effective_sample_size=effective,
            variance_binding=0.0,
            depth=depth,
            delta=delta,
        )
    variance = per_rung_variance(observations, weights)
    binding = int(np.argmax(variance / np.maximum(counts, EPS)))
    v_k = float(variance[binding])
    logarithm = np.log(2.0 * depth / max(delta, EPS))
    leading = float(np.sqrt(max(2.0 * v_k * logarithm / effective, 0.0)))
    residual = float(3.0 * bounded_range * logarithm / effective)
    return EmpiricalBernsteinBound(
        radius=float(leading + residual),
        leading_term=leading,
        residual_correction=residual,
        effective_sample_size=effective,
        variance_binding=v_k,
        depth=depth,
        delta=delta,
    )


def maximum_variance(bounded_range: float) -> float:
    return float((bounded_range / 2.0) ** 2)


def agrees_with_hoeffding_at_maximum_variance(
    per_rung_counts: NDArray[np.float64],
    delta: float,
    bounded_range: float,
) -> dict[str, float]:
    """At sigma_k = B/2 the leading terms coincide; only the residual correction differs.

    Returns the two leading terms and their ratio, which is one by construction, plus the
    residual correction that separates the two total radii.
    """
    depth = int(np.asarray(per_rung_counts).size)
    effective = effective_sample_size_minimum(per_rung_counts)
    logarithm = np.log(2.0 * depth / max(delta, EPS))
    leading = float(
        np.sqrt(max(2.0 * maximum_variance(bounded_range) * logarithm / max(effective, EPS), 0.0))
    )
    hoeffding = float(bounded_range * np.sqrt(max(logarithm, 0.0) / (2.0 * max(effective, EPS))))
    residual = float(3.0 * bounded_range * logarithm / max(effective, EPS))
    return {
        "bernstein_leading": leading,
        "hoeffding_radius": hoeffding,
        "leading_ratio": leading / max(hoeffding, EPS),
        "residual_correction": residual,
    }


def residual_conformed(
    observations: NDArray[np.float64],
    weights: NDArray[np.float64],
    bounded_range: float,
) -> bool:
    """True when the observed rung variance is strictly below the worst case B^2/4."""
    variance = per_rung_variance(observations, weights)
    return bool(np.all(variance <= maximum_variance(bounded_range) + 1e-12))
