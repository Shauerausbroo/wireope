"""Explicit comparator: the high-confidence bound applied directly to the binary endpoint.

The comparator is deliberately not a ladder: it treats the injury indicator as the only
observable and puts a bound on its importance-weighted mean. Keeping it in the bake-off is
what separates the effect of the nested decomposition from the effect of the bound.

Ref: Sec. 3.4 and Sec. 4.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.interval import empirical_bernstein_radius, weighted_mean
from wireope.estimators.base import EstimatorResult, LogBatch, clip_weights
from wireope.utils.linalg import EPS


class BinaryHighConfidence:
    key = "binary_hcb"
    family = "high_confidence_bound"

    def __init__(self, delta: float = 0.05, bounded_range: float = 1.0, clipping: float = 0.0) -> None:
        self.delta = delta
        self.bounded_range = bounded_range
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        endpoint = batch.rung_observations[:, -1]
        point = weighted_mean(endpoint, weights)
        effective = batch.effective_sample_size(weights)
        radius = empirical_bernstein_radius(
            endpoint, weights, effective, self.delta, self.bounded_range, union_size=1
        )
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=point,
            per_rung_value=np.asarray([point], dtype=np.float64),
            risk=float(min(point + radius, 1.0)),
            effective_sample_size=effective,
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={
                "lower_bound": float(max(point - radius, 0.0)),
                "radius": float(radius),
                "ladder_used": 0.0,
            },
        )


def effective_positive_events(positive_case_weights: NDArray[np.float64]) -> float:
    """Effective count of positive events under the importance weights that carry them.

    The effective count is a Kish size over the attempts that actually produced an event at
    the rung, so it is a count of events and not of records.
    """
    weights = np.asarray(positive_case_weights, dtype=np.float64)
    squared = float(np.sum(weights**2))
    if squared <= EPS:
        return 0.0
    return float(np.sum(weights) ** 2 / squared)


def width_under_negative_dilution(
    positive_case_weights: NDArray[np.float64],
    delta: float,
    bounded_range: float,
    negative_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Bound width as negative records are added without new events at the rarest rung.

    The width depends on the effective positive-event count alone, so it is invariant to the
    record count; this is the arithmetic behind the reliability knee.
    """
    effective = effective_positive_events(positive_case_weights)
    sample = np.asarray(positive_case_weights, dtype=np.float64)
    widths = np.zeros(negative_counts.shape[0], dtype=np.float64)
    for position, _ in enumerate(negative_counts):
        widths[position] = empirical_bernstein_radius(
            np.ones_like(sample), sample, effective, delta, bounded_range, union_size=1
        )
    return widths
