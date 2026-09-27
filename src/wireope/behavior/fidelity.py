"""Fidelity of the reconstructed behaviour policy, per site and per operator stratum.

Because the behaviour policy is estimated rather than observed, its fidelity is a reported
outcome. The discretisation-granularity sweep shows how much of the delivered action the
reconstruction leaves unexplained: the recreated action is a coarsened record of the
actual motion, so specific motion is explained and applied force is not.

Ref: Sec. 3.5 and Sec. 4.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class FidelityReport:
    axis: str
    groups: tuple[str, ...]
    log_loss: NDArray[np.float64]
    top1_agreement: NDArray[np.float64]
    expected_calibration_error: NDArray[np.float64]

    def worst_log_loss_group(self) -> str:
        if self.groups:
            return self.groups[int(np.argmax(self.log_loss))]
        return ""


def multiclass_log_loss(probability: NDArray[np.float64], target_index: NDArray[np.int64]) -> float:
    rows = np.arange(probability.shape[0])
    selected = clip_probability(probability[rows, target_index])
    return float(-np.mean(np.log(selected)))


def top1_agreement(probability: NDArray[np.float64], target_index: NDArray[np.int64]) -> float:
    predicted = np.argmax(probability, axis=1)
    return float(np.mean(predicted == target_index))


def multiclass_ece(
    probability: NDArray[np.float64],
    target_index: NDArray[np.int64],
    bins: int = 10,
) -> float:
    """Expected calibration error of the top-1 confidence, on the higher-is-better scale."""
    confidence = np.max(probability, axis=1)
    predicted = np.argmax(probability, axis=1)
    correct = (predicted == target_index).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        member = (confidence > low) & (confidence <= high)
        if not member.any():
            continue
        share = float(np.mean(member))
        total += share * abs(float(np.mean(correct[member])) - float(np.mean(confidence[member])))
    return float(total)


def fidelity_by_axis(
    axis: str,
    groups: NDArray[np.str_],
    probability: NDArray[np.float64],
    target_index: NDArray[np.int64],
) -> FidelityReport:
    names = tuple(sorted(set(groups.tolist())))
    log_loss = np.zeros(len(names), dtype=np.float64)
    agreement = np.zeros(len(names), dtype=np.float64)
    calibration = np.zeros(len(names), dtype=np.float64)
    for position, name in enumerate(names):
        member = groups == name
        if not member.any():
            log_loss[position] = float("nan")
            agreement[position] = float("nan")
            calibration[position] = float("nan")
            continue
        log_loss[position] = multiclass_log_loss(probability[member], target_index[member])
        agreement[position] = top1_agreement(probability[member], target_index[member])
        calibration[position] = multiclass_ece(probability[member], target_index[member])
    return FidelityReport(
        axis=axis,
        groups=names,
        log_loss=log_loss,
        top1_agreement=agreement,
        expected_calibration_error=calibration,
    )


def operator_stratum(attempts: NDArray[np.int64]) -> NDArray[np.str_]:
    """Split operators by their accrued attempt volume into low, mid and high strata."""
    if attempts.size == 0:
        return np.zeros(0, dtype="<U8")
    low, high = np.quantile(attempts, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.full(attempts.shape[0], "mid", dtype="<U8")
    labels[attempts <= low] = "low"
    labels[attempts >= high] = "high"
    return labels


@dataclass(frozen=True)
class GranularityResidual:
    granularity: str
    action_bins: int
    residual: float
    explained_share: float


def discretisation_sweep(
    realised_actions: NDArray[np.float64],
    reconstructed_actions: NDArray[np.float64],
    bins_per_granularity: tuple[int, ...],
    labels: tuple[str, ...],
) -> tuple[GranularityResidual, ...]:
    """Coarsen both action series at several granularities and measure the residual."""
    if realised_actions.shape != reconstructed_actions.shape:
        raise ValueError("realised and reconstructed actions must share a shape")
    if len(bins_per_granularity) != len(labels):
        raise ValueError("every granularity needs a label")
    span = float(np.max(np.abs(realised_actions))) if realised_actions.size else 1.0
    span = max(span, EPS)
    results: list[GranularityResidual] = []
    for bins, label in zip(bins_per_granularity, labels):
        quantised = np.round(reconstructed_actions / span * bins)
        realised_quantised = np.round(realised_actions / span * bins)
        residual = float(np.mean(np.abs(quantised - realised_quantised)) / max(bins, 1))
        explained = 1.0 - residual / max(float(np.mean(np.abs(realised_actions))) / span, EPS)
        results.append(
            GranularityResidual(
                granularity=label,
                action_bins=bins,
                residual=residual,
                explained_share=float(np.clip(explained, 0.0, 1.0)),
            )
        )
    return tuple(results)


def reconstruction_cost(
    realised_actions: NDArray[np.float64],
    reconstructed_actions: NDArray[np.float64],
) -> dict[str, float]:
    """Cost of estimating the behaviour policy rather than observing it directly."""
    difference = np.asarray(reconstructed_actions - realised_actions, dtype=np.float64)
    return {
        "mean_absolute_residual": float(np.mean(np.abs(difference))),
        "root_mean_square_residual": float(np.sqrt(np.mean(difference**2))),
        "unexplained_share": float(np.mean(np.abs(difference) > 0.5)),
    }
