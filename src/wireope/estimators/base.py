"""Shared estimator interface for off-policy evaluation over the excursion ladder.

Every candidate consumes the same logged batch, so the bake-off compares estimators rather
than preprocessing. The batch carries the action observable, the reconstructed behaviour
propensity for the target action, and the fitted rung-wise outcome model.

Ref: Sec. 2.1, Sec. 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class LogBatch:
    features: NDArray[np.float64]
    behaviour_strategy: NDArray[np.str_]
    target_strategy: NDArray[np.str_]
    rung_observations: NDArray[np.float64]
    propensities: NDArray[np.float64]
    weights: NDArray[np.float64]
    strata: NDArray[np.str_]
    outcome_predictions: NDArray[np.float64] | None = None
    target_branch: NDArray[np.str_] | None = None
    observed_branch: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        for name, array in (
            ("behaviour_strategy", self.behaviour_strategy),
            ("target_strategy", self.target_strategy),
            ("strata", self.strata),
        ):
            if array.shape[0] != count:
                raise ValueError(f"{name} must have one entry per logged attempt")
        if self.rung_observations.shape != (count, self.weights.shape[0]):
            raise ValueError("rung observations must be (attempts, rungs)")
        if self.propensities.shape[0] != count:
            raise ValueError("propensities must have one entry per logged attempt")

    @property
    def attempts(self) -> int:
        return int(self.features.shape[0])

    @property
    def rungs(self) -> int:
        return int(self.weights.shape[0])

    @property
    def assigned(self) -> NDArray[np.bool_]:
        return np.asarray(self.behaviour_strategy == self.target_strategy, dtype=np.bool_)

    def importance_weights(self, clipping: float = 0.0) -> NDArray[np.float64]:
        """1{d = pi(s)} / d_b(pi(s) | s), the weight every family consumes.

        The reconstructed rung index the manuscript writes on rho is inherited from the
        nested observable rather than from a rung-specific behaviour model, so the rho part
        is the indicator and the ratio is the reconstructed behaviour propensity.
        """
        denominator = (
            clip_probability(self.propensities)
            if clipping <= 0.0
            else clip_probability(self.propensities, clipping, 1.0)
        )
        ratio = self.assigned.astype(np.float64) / np.maximum(denominator, EPS)
        return np.asarray(ratio, dtype=np.float64)

    def effective_sample_size(self, weights: NDArray[np.float64]) -> float:
        squared = float(np.sum(weights**2))
        if squared <= EPS:
            return 0.0
        return float(np.sum(weights) ** 2 / squared)

    def overlap_coefficient(self) -> float:
        """Mean behaviour propensity of the target action over the evaluated attempts."""
        return float(np.mean(clip_probability(self.propensities)))

    def rung_effective_sizes(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(self.rungs, self.effective_sample_size(weights), dtype=np.float64)


@dataclass(frozen=True)
class EstimatorResult:
    key: str
    family: str
    value: float
    per_rung_value: NDArray[np.float64]
    risk: float
    effective_sample_size: float
    overlap_coefficient: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def relative_error(self, truth: float) -> float:
        if abs(truth) <= EPS:
            return float("inf")
        return float(abs(self.value - truth) / abs(truth))


class OffPolicyEstimator(Protocol):
    key: str
    family: str

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult: ...


def weighted_ladder_value(
    per_rung: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    return float(np.sum(per_rung * weights))


def clip_weights(weights: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Self-normalising weight control: cap each importance weight at the threshold."""
    if threshold <= 0.0:
        return weights
    return np.asarray(np.minimum(weights, 1.0 / threshold), dtype=np.float64)


def self_normalised(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(np.sum(weights))
    if total <= EPS:
        return np.zeros_like(weights)
    result: NDArray[np.float64] = weights / total
    return result
