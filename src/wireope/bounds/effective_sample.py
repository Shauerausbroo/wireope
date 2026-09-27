"""Effective sample size, overlap coefficient and the diagnostics that select an estimator.

An importance-weighted effective sample size is a state-density quantity: it says how much
of the state distribution the weights reach, and it cannot say whether the injury-producing
region of the action space was visited. That is the reason the selection rule replaces a
global effective-sample-size check with a per-stratum, event-wise overlap floor, and the
reason the estimator that produced an estimate is reported alongside it.

Ref: Sec. 1, Sec. 3.5 and Sec. 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    array = np.asarray(weights, dtype=np.float64)
    squared = float(np.sum(array**2))
    if squared <= EPS:
        return 0.0
    return float(np.sum(array) ** 2 / squared)


def per_rung_effective_sample_size(
    weights: NDArray[np.float64],
    rung_assignment: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Effective sample size per rung, on the weights of the attempts that reached it."""
    sizes = np.zeros(rung_assignment.shape[1], dtype=np.float64)
    for rung in range(rung_assignment.shape[1]):
        member = rung_assignment[:, rung] > 0.0
        sizes[rung] = effective_sample_size(weights[member]) if member.any() else 0.0
    return sizes


def binding_rung(per_rung_sizes: NDArray[np.float64]) -> int:
    sizes = np.asarray(per_rung_sizes, dtype=np.float64)
    if sizes.size == 0:
        return -1
    return int(np.argmin(sizes))


def overlap_coefficient(
    behaviour_probability: NDArray[np.float64], target_probability: NDArray[np.float64]
) -> float:
    """Mean of the element-wise minimum of the behaviour and target probabilities.

    The coefficient is bounded by one and is zero when the target action is never taken by
    the behaviour policy, which is the regime the selection rule guards against.
    """
    behaviour = np.asarray(behaviour_probability, dtype=np.float64)
    target = np.asarray(target_probability, dtype=np.float64)
    if behaviour.shape != target.shape:
        raise ValueError("behaviour and target probabilities must share a shape")
    return float(np.mean(np.minimum(behaviour, target)))


@dataclass(frozen=True)
class SelectionDiagnostics:
    relative_error: float
    effective_sample_size: float
    overlap_coefficient: float
    selection_accuracy: float
    binding_rung: int

    def chosen_reason(self) -> str:
        """Plain-language statement of why the estimator was selected."""
        return (
            f"binding rung {self.binding_rung} at effective size "
            f"{self.effective_sample_size:.1f} with overlap {self.overlap_coefficient:.3f}"
        )


def relative_error(estimate: float, truth: float) -> float:
    if abs(truth) <= EPS:
        return float("inf")
    return float(abs(estimate - truth) / abs(truth))


def weight_concentration(weights: NDArray[np.float64]) -> float:
    """Share of the total weight carried by the single largest weight."""
    array = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(array))
    if total <= EPS:
        return 1.0
    return float(np.max(array) / total)


def degenerate_overlap(overlap: float, tolerance: float = 1e-3) -> bool:
    """True when the overlap is so small that a per-stratum floor would exclude everything."""
    return bool(overlap <= tolerance)
