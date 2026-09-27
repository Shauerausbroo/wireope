"""Doubly-robust family: DR, weighted DR and the switching hybrid.

The hybrid family was the aggregate winner on the published benchmarks and is the least
stable family here: a corpus whose behaviour policy is a blend across sites and operators,
and whose action channel is approximated rather than witnessed, is the regime in which the
hybrid correction adds variance where it is meant to remove bias.

Ref: Sec. 2.1, Sec. 2.2, Sec. 3.4 and Sec. 4.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wireope.estimators.base import (
    EstimatorResult,
    LogBatch,
    clip_weights,
    self_normalised,
    weighted_ladder_value,
)
from wireope.utils.linalg import EPS


def _requirements(batch: LogBatch) -> NDArray[np.float64]:
    if batch.outcome_predictions is None:
        raise ValueError("the doubly-robust family needs rung-wise outcome predictions")
    return np.asarray(batch.outcome_predictions, dtype=np.float64)


def doubly_robust_rungs(batch: LogBatch, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    predictions = _requirements(batch)
    direct = np.mean(predictions, axis=0)
    residual = batch.rung_observations - predictions
    correction = np.mean(weights[:, None] * residual, axis=0)
    return np.asarray(direct + correction, dtype=np.float64)


def weighted_doubly_robust_rungs(batch: LogBatch, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    predictions = _requirements(batch)
    scaled = self_normalised(weights) * float(weights.shape[0])
    direct = np.mean(scaled[:, None] * predictions, axis=0)
    residual = batch.rung_observations - predictions
    correction = np.mean(scaled[:, None] * residual, axis=0)
    return np.asarray(direct + correction, dtype=np.float64)


class DoublyRobust:
    key = "dr"
    family = "doubly_robust"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        per_rung = doubly_robust_rungs(batch, weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=weighted_ladder_value(per_rung, batch.weights),
            per_rung_value=per_rung,
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"max_weight": float(np.max(weights)) if weights.size else 0.0},
        )


class WeightedDoublyRobust:
    key = "wdr"
    family = "doubly_robust"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        per_rung = weighted_doubly_robust_rungs(batch, weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=weighted_ladder_value(per_rung, batch.weights),
            per_rung_value=per_rung,
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"weight_sum": float(np.sum(weights))},
        )


class SwitchingHybrid:
    key = "magic"
    family = "switching_hybrid"

    def __init__(self, switch_constant: float = 1.0, clipping: float = 0.0) -> None:
        self.switch_constant = switch_constant
        self.clipping = clipping

    def switching_weights(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Smooth switch toward the direct part as the importance weight grows."""
        constant = max(self.switch_constant, EPS)
        switch: NDArray[np.float64] = weights / (weights + constant)
        return switch

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        switch = self.switching_weights(weights)
        predictions = _requirements(batch)
        direct = np.mean(predictions, axis=0)
        residual = batch.rung_observations - predictions
        correction = np.mean((switch * weights)[:, None] * residual, axis=0)
        per_rung = np.asarray(direct + correction, dtype=np.float64)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=weighted_ladder_value(per_rung, batch.weights),
            per_rung_value=per_rung,
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={
                "mean_switch": float(np.mean(switch)) if switch.size else 0.0,
                "switch_constant": self.switch_constant,
            },
        )


def correction_magnitude(batch: LogBatch, weights: NDArray[np.float64]) -> float:
    """Absolute size of the importance-weighted correction relative to the direct part."""
    predictions = _requirements(batch)
    direct = float(np.mean(np.abs(np.mean(predictions, axis=0))))
    correction = float(
        np.mean(np.abs(np.mean(weights[:, None] * (batch.rung_observations - predictions), axis=0)))
    )
    return correction / max(direct, EPS)
