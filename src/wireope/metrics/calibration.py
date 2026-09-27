"""Calibration: expected calibration error on the higher-is-better scale, slope and intercept.

Calibration is reported as one minus the expected calibration error so that higher is
better, together with the calibration slope and intercept; the Brier score is quoted as a
point of reference rather than as a comparison, because no injury-endpoint Brier value is
available to compare against.

Ref: Sec. 3.7 and Table 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS, clip_probability, logit


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    one_minus_ece: float
    slope: float
    intercept: float
    brier: float
    bins: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "one_minus_ece": self.one_minus_ece,
            "calibration_slope": self.slope,
            "calibration_intercept": self.intercept,
            "brier": self.brier,
        }


def expected_calibration_error(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    bins: int = 10,
) -> float:
    """Gap between predicted confidence and observed frequency, averaged over equal-width bins."""
    values = clip_probability(probability)
    target = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        member = (values > low) & (values <= high)
        if not member.any():
            continue
        weight = float(np.mean(member))
        total += weight * abs(float(np.mean(target[member])) - float(np.mean(values[member])))
    return float(total)


def calibration_slope_intercept(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[float, float]:
    """Coefficients of the logistic recalibration of the score on the logit scale.

    A perfectly calibrated score has slope one and intercept zero; the pair is what makes a
    shift in the calibration visible independently of the discrimination.
    """
    x_value = logit(probability)[:, None]
    design = np.hstack([np.ones((x_value.shape[0], 1), dtype=np.float64), x_value])
    target = np.asarray(labels, dtype=np.float64)
    coefficients = _logistic_newton(design, target, np.zeros(2, dtype=np.float64))
    return float(coefficients[1]), float(coefficients[0])


def _logistic_newton(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    initial: NDArray[np.float64],
    iterations: int = 60,
    tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    coefficients = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(iterations):
        linear = design @ coefficients
        fitted = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (target - fitted)
        weights = np.maximum(fitted * (1.0 - fitted), EPS)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(hessian.shape[0]), gradient)
        coefficients = coefficients + step
        if float(np.max(np.abs(step))) <= tolerance:
            break
    return coefficients


def brier_score(probability: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    values = clip_probability(probability)
    target = np.asarray(labels, dtype=np.float64)
    return float(np.mean((values - target) ** 2))


def calibration_report(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    bins: int = 10,
) -> CalibrationReport:
    ece = expected_calibration_error(probability, labels, bins)
    slope, intercept = calibration_slope_intercept(probability, labels)
    return CalibrationReport(
        expected_calibration_error=ece,
        one_minus_ece=float(1.0 - ece),
        slope=slope,
        intercept=intercept,
        brier=brier_score(probability, labels),
        bins=bins,
    )


def reliability_curve(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    values = clip_probability(probability)
    target = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    means: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        member = (values > low) & (values <= high)
        counts.append(int(np.count_nonzero(member)))
        means.append(float(np.mean(values[member])) if member.any() else float("nan"))
        observed.append(float(np.mean(target[member])) if member.any() else float("nan"))
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def one_minus_ece(probability: NDArray[np.float64], labels: NDArray[np.int64], bins: int = 10) -> float:
    return float(1.0 - expected_calibration_error(probability, labels, bins))
