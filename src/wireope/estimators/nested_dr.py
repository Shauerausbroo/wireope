"""Nested weighted doubly-robust estimator over the excursion ladder.

The estimator of Eq. (4) writes the graded value as a weighted sum over rungs, each rung
carrying a direct part from the fitted outcome model and an importance-weighted residual
correction. The endpoint risk of Eq. (6) is the product of the per-rung conditional
outcomes, which is where the truncation bias eps_K enters.

Ref: Sec. 3.1 and Sec. 3.4, Eq. (3)-(4), Eq. (6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.effective_sample import effective_sample_size, relative_error
from wireope.estimators.base import EstimatorResult, LogBatch, clip_weights, weighted_ladder_value
from wireope.laddering.severity import Ladder
from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class NestedDRResult:
    key: str
    graded_value: float
    per_rung_value: NDArray[np.float64]
    endpoint_risk: float
    effective_sample_size: float
    per_rung_effective_sample_size: NDArray[np.float64]
    overlap: float
    weight_concentration: float
    clipping: float

    def relative_error(self, truth: float) -> float:
        return relative_error(self.graded_value, truth)

    def as_estimator_result(self) -> EstimatorResult:
        return EstimatorResult(
            key=self.key,
            family="doubly_robust",
            value=self.graded_value,
            per_rung_value=self.per_rung_value,
            risk=self.endpoint_risk,
            effective_sample_size=self.effective_sample_size,
            overlap_coefficient=self.overlap,
            diagnostics={
                "weight_concentration": self.weight_concentration,
                "clipping": self.clipping,
                "min_rung_effective_size": (
                    float(np.min(self.per_rung_effective_sample_size))
                    if self.per_rung_effective_sample_size.size
                    else 0.0
                ),
            },
        )


class NestedWeightedDoublyRobust:
    """Eq. (4): the direct part plus the importance-weighted residual, summed over rungs."""

    key = "nested_dr"
    family = "doubly_robust"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        """Protocol-conformant entry point, so the estimator also enters the bake-off."""
        return self.detailed(batch, clipping).as_estimator_result()

    def detailed(self, batch: LogBatch, clipping: float = 0.0) -> NestedDRResult:
        return nested_doubly_robust(batch, clipping or self.clipping)


def nested_doubly_robust(batch: LogBatch, clipping: float = 0.0) -> NestedDRResult:
    if batch.outcome_predictions is None:
        raise ValueError("the nested estimator needs rung-wise outcome predictions")
    predictions = np.asarray(batch.outcome_predictions, dtype=np.float64)
    raw_weights = batch.importance_weights(clipping)
    weights = clip_weights(raw_weights, clipping)
    direct = predictions.mean(axis=0)
    residual = batch.rung_observations - predictions
    correction = (weights[:, None] * residual).mean(axis=0)
    per_rung = np.asarray(direct + correction, dtype=np.float64)
    graded = weighted_ladder_value(per_rung, batch.weights)
    endpoint = float(per_rung[-1])
    per_rung_sizes = np.asarray(
        [
            (
                effective_sample_size(weights[batch.rung_observations[:, rung] > 0.0])
                if np.any(batch.rung_observations[:, rung] > 0.0)
                else 0.0
            )
            for rung in range(batch.rungs)
        ],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    concentration = float(np.max(weights) / total) if total > EPS and weights.size else 1.0
    return NestedDRResult(
        key=NestedWeightedDoublyRobust.key,
        graded_value=graded,
        per_rung_value=per_rung,
        endpoint_risk=endpoint,
        effective_sample_size=effective_sample_size(weights),
        per_rung_effective_sample_size=per_rung_sizes,
        overlap=batch.overlap_coefficient(),
        weight_concentration=concentration,
        clipping=clipping,
    )


def graded_progress_value(
    per_rung_marginals: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Graded crossing progress on the same ladder: sum_k w_k (1 - P(Y^(k))).

    Eq. (4) estimates the rung marginals; the progress value is their complement, so the two
    functionals share one estimator and one ladder, and the utility rises as the graded
    excursion severity falls.

    Ref: Sec. 3.1, Eq. (3).
    """
    marginals = np.asarray(per_rung_marginals, dtype=np.float64)
    ladder_weights = np.asarray(weights, dtype=np.float64)
    if marginals.shape != ladder_weights.shape:
        raise ValueError("one weight per rung is required")
    return float(np.sum(ladder_weights * (1.0 - np.clip(marginals, 0.0, 1.0))))


def severity_from_progress(progress: float, weights: NDArray[np.float64]) -> float:
    return float(np.sum(np.asarray(weights, dtype=np.float64)) - progress)


def conditionals_from_marginals(
    per_rung_marginals: NDArray[np.float64],
    floor: float = 1e-6,
) -> NDArray[np.float64]:
    """Convert nested-set marginals into the per-rung conditionals Eq. (6) multiplies.

    The nesting gives P(A_k) = prod_{j<=k} P(A_j | A_{j-1}), so each conditional is the ratio
    of successive marginals; the ratio is floored so the product stays well defined when a
    rung's marginal falls below a rung above it by sampling noise.
    """
    marginals = np.clip(np.asarray(per_rung_marginals, dtype=np.float64), floor, 1.0)
    previous = np.concatenate([[1.0], marginals[:-1]])
    ratio = marginals / np.maximum(previous, floor)
    return np.asarray(np.clip(ratio, 0.0, 1.0), dtype=np.float64)


def product_risk_from_marginals(per_rung_marginals: NDArray[np.float64]) -> float:
    """Eq. (6) evaluated on the rung marginals through their conditionals."""
    conditionals = conditionals_from_marginals(per_rung_marginals)
    return float(np.prod(conditionals))


def product_risk(per_rung_outcome: NDArray[np.float64]) -> float:
    """Eq. (6): R_hat(pi) = prod_k q_k(pi)."""
    return float(np.prod(np.asarray(per_rung_outcome, dtype=np.float64)))


def product_risk_from_conditionals(ladder: Ladder, conditionals: NDArray[np.float64]) -> float:
    if conditionals.shape[0] != ladder.depth:
        raise ValueError("one conditional per rung is required")
    return float(np.prod(np.asarray(conditionals, dtype=np.float64)))


def truncation_bias(
    per_rung_outcome: NDArray[np.float64],
    ladder: Ladder,
    observed_endpoint: float,
) -> float:
    """eps_K: the difference the product form attributes to truncating at the top rung.

    The correction is non-negative by construction and shrinks as the ladder is refined
    towards the observed injury threshold.
    """
    projected = product_risk(per_rung_outcome)
    weights = np.asarray(ladder.weights, dtype=np.float64)
    depth_term = float(np.sum(weights) / max(ladder.depth, 1))
    return float(abs(projected - observed_endpoint) * depth_term)


def graded_truth_from_probabilities(
    ladder: Ladder,
    conditional_truth: NDArray[np.float64],
) -> tuple[float, float]:
    """Closed-form graded value and endpoint risk from the rung conditional probabilities.

    The rung-k event probability is the product of the conditionals up to k, so the graded
    value is sum_k w_k prod_{j<=k} p_j and the risk is the full product.
    """
    conditionals = np.asarray(conditional_truth, dtype=np.float64)
    if conditionals.shape[0] != ladder.depth:
        raise ValueError("one conditional per rung is required")
    cumulative = np.cumprod(conditionals)
    weights = np.asarray(ladder.weights, dtype=np.float64)
    graded = float(np.sum(weights * cumulative))
    risk = float(cumulative[-1])
    return graded, risk


def clipped_observations(
    rung_observations: NDArray[np.float64],
    bounded_range: float,
) -> NDArray[np.float64]:
    """Clip the rung observations to the bounded range the deviation bound assumes."""
    return np.clip(np.asarray(rung_observations, dtype=np.float64), 0.0, bounded_range)


def clip_probability_matrix(probability: NDArray[np.float64], floor: float) -> NDArray[np.float64]:
    """Floor and cap a probability matrix so a log of it stays finite."""
    return np.asarray(clip_probability(probability, floor, 1.0 - floor), dtype=np.float64)
