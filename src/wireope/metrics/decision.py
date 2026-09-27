"""Decision-analytic metrics: net benefit over a threshold grid and net reclassification.

Net benefit is the quantity the selection rule maximises a lower confidence limit of, and
the threshold grid is pre-specified rather than scanned. The treat-all and treat-none
references are carried explicitly so a score that beats neither is visible.

Ref: Sec. 3.1, Sec. 3.7 and Sec. 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class NetBenefitCurve:
    thresholds: NDArray[np.float64]
    treat_all: NDArray[np.float64]
    treat_none: NDArray[np.float64]

    def at(self, threshold: float) -> dict[str, float]:
        index = int(np.argmin(np.abs(self.thresholds - threshold)))
        return {
            "threshold": float(self.thresholds[index]),
            "treat_all": float(self.treat_all[index]),
            "treat_none": float(self.treat_none[index]),
        }


def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    return float(prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold))


def treat_none_net_benefit() -> float:
    return 0.0


def net_benefit(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
) -> float:
    """Decision curve analysis: true-positive share minus the weighted false-positive share."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    values = clip_probability(probability)
    target = np.asarray(labels, dtype=np.float64)
    positive = values >= threshold
    n_total = target.shape[0]
    if n_total == 0:
        return 0.0
    true_positive = float(np.sum(target[positive])) / n_total
    false_positive = float(np.sum((1.0 - target)[positive])) / n_total
    return float(true_positive - false_positive * threshold / (1.0 - threshold))


def net_benefit_curve(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    thresholds: tuple[float, ...],
) -> NDArray[np.float64]:
    return np.asarray(
        [net_benefit(probability, labels, threshold) for threshold in thresholds], dtype=np.float64
    )


def reference_curves(
    labels: NDArray[np.int64],
    thresholds: tuple[float, ...],
) -> NetBenefitCurve:
    prevalence = float(np.mean(labels)) if labels.shape[0] else 0.0
    return NetBenefitCurve(
        thresholds=np.asarray(thresholds, dtype=np.float64),
        treat_all=np.asarray(
            [treat_all_net_benefit(prevalence, threshold) for threshold in thresholds],
            dtype=np.float64,
        ),
        treat_none=np.zeros(len(thresholds), dtype=np.float64),
    )


def standardised_net_benefit(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
) -> float:
    """Net benefit divided by the event rate, the form that is comparable across cohorts."""
    prevalence = float(np.mean(labels)) if labels.shape[0] else 0.0
    if prevalence <= EPS:
        return float("nan")
    return float(net_benefit(probability, labels, threshold) / prevalence)


def net_reclassification_improvement(
    probability_new: NDArray[np.float64],
    probability_old: NDArray[np.float64],
    labels: NDArray[np.int64],
    threshold: float,
) -> dict[str, float]:
    """Movement of attempts across the decision threshold, split by their true endpoint."""
    target = np.asarray(labels, dtype=np.int64)
    new_positive = clip_probability(probability_new) >= threshold
    old_positive = clip_probability(probability_old) >= threshold
    up = new_positive & ~old_positive
    down = old_positive & ~new_positive
    n_total = max(target.shape[0], 1)
    events = int(np.count_nonzero(target))
    non_events = int(target.shape[0] - events)
    return {
        "up_moved": float(np.count_nonzero(up)) / n_total,
        "down_moved": float(np.count_nonzero(down)) / n_total,
        "events_up": float(np.count_nonzero(up & (target == 1))) / max(events, 1),
        "non_events_up": float(np.count_nonzero(up & (target == 0))) / max(non_events, 1),
        "nri": float(
            np.count_nonzero(up & (target == 1)) / max(events, 1)
            - np.count_nonzero(down & (target == 1)) / max(events, 1)
            + np.count_nonzero(down & (target == 0)) / max(non_events, 1)
            - np.count_nonzero(up & (target == 0)) / max(non_events, 1)
        ),
    }


def threshold_inside_range(threshold: float, low: float, high: float) -> bool:
    return bool(low <= threshold <= high)


def best_threshold_by_net_benefit(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    thresholds: tuple[float, ...],
) -> float:
    curve = net_benefit_curve(probability, labels, thresholds)
    return float(np.asarray(thresholds, dtype=np.float64)[int(np.argmax(curve))])


def net_benefit_advantage(
    probability: NDArray[np.float64],
    labels: NDArray[np.int64],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    """Net benefit against both reference strategies over the whole threshold grid."""
    curve = net_benefit_curve(probability, labels, thresholds)
    references = reference_curves(labels, thresholds)
    over_all = curve - references.treat_all
    over_none = curve - references.treat_none
    return {
        "beats_treat_all_everywhere": float(np.all(over_all > 0.0)),
        "beats_treat_none_everywhere": float(np.all(over_none > 0.0)),
        "min_advantage_over_treat_all": float(np.min(over_all)),
        "min_advantage_over_treat_none": float(np.min(over_none)),
    }
