"""Importance-sampling family: IS, PDIS, WIS and PDWIS.

Plain and per-decision importance sampling reweight the logged rungs by the reconstructed
behaviour propensity; the weighted variants self-normalise, which is what keeps the
estimator finite when a few weights dominate.

Ref: Sec. 2.1 and Sec. 3.4.
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


class ImportanceSampling:
    key = "is"
    family = "importance_sampling"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        endpoint = batch.rung_observations[:, -1]
        per_rung = np.zeros(batch.rungs, dtype=np.float64)
        per_rung[-1] = float(np.mean(weights * endpoint))
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=per_rung,
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"max_weight": float(np.max(weights)) if weights.size else 0.0},
        )


class PerDecisionImportanceSampling:
    key = "pdis"
    family = "importance_sampling"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        per_rung = np.mean(weights[:, None] * batch.rung_observations, axis=0)
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=np.asarray(per_rung, dtype=np.float64),
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"max_weight": float(np.max(weights)) if weights.size else 0.0},
        )


class WeightedImportanceSampling:
    key = "wis"
    family = "importance_sampling"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        normalised = self_normalised(weights)
        endpoint = batch.rung_observations[:, -1]
        per_rung = np.zeros(batch.rungs, dtype=np.float64)
        per_rung[-1] = float(np.sum(normalised * endpoint))
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=per_rung,
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"weight_sum": float(np.sum(weights))},
        )


class PerDecisionWeightedImportanceSampling:
    key = "pdis_w"
    family = "importance_sampling"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        weights = clip_weights(batch.importance_weights(self.clipping), clipping)
        normalised = self_normalised(weights)
        per_rung = np.sum(normalised[:, None] * batch.rung_observations, axis=0)
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=np.asarray(per_rung, dtype=np.float64),
            risk=float(per_rung[-1]),
            effective_sample_size=batch.effective_sample_size(weights),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"weight_sum": float(np.sum(weights))},
        )


def rung_ratio_matrix(batch: LogBatch, clipping: float = 0.0) -> NDArray[np.float64]:
    """Per-rung importance ratio, kept in its own helper so the families share one source."""
    weights = clip_weights(batch.importance_weights(clipping), clipping)
    return np.repeat(weights[:, None], batch.rungs, axis=1)
