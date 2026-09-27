"""Kernel direct family: infinite-horizon kernel regression over the state space.

The kernel estimator is a direct method whose weights come from the state density rather
than from the behaviour propensity, so it does not consume an importance weight and is
unaffected by the action-channel overlap that destabilises the hybrid family.

Ref: Sec. 2.1, Sec. 3.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wireope.estimators.base import EstimatorResult, LogBatch, weighted_ladder_value
from wireope.utils.linalg import EPS


class InfiniteHorizonKernel:
    key = "kernel_ih"
    family = "kernel_direct"

    def __init__(self, bandwidth_multiplier: float = 1.0, support_clip: float = 1e-3) -> None:
        self.bandwidth_multiplier = bandwidth_multiplier
        self.support_clip = support_clip

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        _ = clipping
        per_rung = kernel_rung_values(batch, self.bandwidth_multiplier, self.support_clip)
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=np.asarray(per_rung, dtype=np.float64),
            risk=float(per_rung[-1]),
            effective_sample_size=float(batch.attempts),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"bandwidth": median_bandwidth(batch.features) * self.bandwidth_multiplier},
        )


def median_bandwidth(features: NDArray[np.float64]) -> float:
    """Median-heuristic bandwidth over pairwise state distances."""
    if features.shape[0] < 2:
        return 1.0
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    positive = distances[distances > 0.0]
    if positive.size == 0:
        return 1.0
    bandwidth: float = float(np.median(positive)) / max(float(np.sqrt(features.shape[1])), 1.0)
    return bandwidth


def gaussian_kernel_matrix(features: NDArray[np.float64], bandwidth: float) -> NDArray[np.float64]:
    if bandwidth <= 0.0:
        raise ValueError("kernel bandwidth must be positive")
    squared = np.sum(features**2, axis=1)
    distances = squared[:, None] + squared[None, :] - 2.0 * features @ features.T
    distances = np.maximum(distances, 0.0)
    kernel: NDArray[np.float64] = np.exp(-distances / (2.0 * bandwidth**2))
    return kernel


def state_density_weights(
    features: NDArray[np.float64],
    bandwidth: float,
    support_clip: float,
) -> NDArray[np.float64]:
    """Normalised kernel weights per target state, floored at a support clip.

    The ratio is a state-density quantity, which is exactly why the manuscript replaces a
    global effective-sample-size check with a per-stratum, event-wise overlap floor.
    """
    kernel = gaussian_kernel_matrix(features, bandwidth)
    row_sums = np.maximum(kernel.sum(axis=1, keepdims=True), EPS)
    ratios = kernel / row_sums
    return np.asarray(np.maximum(ratios, support_clip), dtype=np.float64)


def kernel_rung_values(
    batch: LogBatch,
    bandwidth_multiplier: float = 1.0,
    support_clip: float = 1e-3,
) -> NDArray[np.float64]:
    bandwidth = median_bandwidth(batch.features) * bandwidth_multiplier
    kernel = gaussian_kernel_matrix(batch.features, bandwidth)
    denominator = np.maximum(kernel.sum(axis=1, keepdims=True), EPS)
    weights = kernel / denominator
    if support_clip > 0.0:
        weights = np.maximum(weights, support_clip)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    # Each target state gets its own kernel regression; the estimate averages those states.
    per_state = weights @ batch.rung_observations
    values = np.mean(per_state, axis=0)
    return np.asarray(values, dtype=np.float64)


def kernel_support_fraction(weights: NDArray[np.float64], threshold: float) -> float:
    return float(np.mean(weights > threshold))
