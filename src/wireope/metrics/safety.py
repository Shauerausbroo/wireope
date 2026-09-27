"""Safety-endpoint metrics: violation rate, abstention with coverage, selection accuracy and regret.

Abstention is reported jointly with its coverage so that a safety gain cannot be produced by
declining to answer, and the violation rate is reported as an upper confidence limit rather
than as a point estimate.

Ref: Sec. 3.5, Sec. 3.7 and Sec. 4.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class AbstentionReport:
    abstention_rate: float
    coverage: float
    abstentions: int
    decisions: int

    @property
    def complementary(self) -> bool:
        return bool(abs((self.abstention_rate + self.coverage) - 1.0) < 1e-9)


def abstention_report(abstained: NDArray[np.bool_]) -> AbstentionReport:
    total = int(abstained.shape[0])
    if total == 0:
        return AbstentionReport(abstention_rate=0.0, coverage=1.0, abstentions=0, decisions=0)
    count = int(np.count_nonzero(abstained))
    rate = float(count / total)
    return AbstentionReport(
        abstention_rate=rate,
        coverage=float(1.0 - rate),
        abstentions=count,
        decisions=total - count,
    )


def violation_rate(
    realised_injuries: NDArray[np.int64],
    abstained: NDArray[np.bool_],
) -> float:
    """Share of the answered decisions in which the injury endpoint was realised.

    A safety claim made at the strategy level is breached when an answered decision ends in an
    injury, so the rate is taken over the decisions the rule was willing to take.
    """
    answered = ~np.asarray(abstained, dtype=np.bool_)
    if not np.any(answered):
        return 0.0
    return float(np.mean(np.asarray(realised_injuries, dtype=np.float64)[answered]))


def violation_rate_upper_bound(
    realised_injuries: NDArray[np.int64],
    abstained: NDArray[np.bool_],
    confidence: float = 0.95,
) -> float:
    """One-sided upper limit of the violation rate by the Wilson score interval."""
    answered = ~np.asarray(abstained, dtype=np.bool_)
    total = int(np.count_nonzero(answered))
    if total == 0:
        return 0.0
    events = float(np.sum(np.asarray(realised_injuries, dtype=np.float64)[answered]))
    proportion = events / total
    z_value = float(stats.norm.ppf(confidence))
    denominator = 1.0 + z_value**2 / total
    centre = proportion + z_value**2 / (2.0 * total)
    half = z_value * float(np.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2)))
    return float((centre + half) / denominator)


def selection_accuracy(chosen: NDArray[np.str_], oracle: NDArray[np.str_]) -> float:
    if chosen.shape[0] == 0:
        return 0.0
    return float(np.mean(chosen == oracle))


def regret(values: NDArray[np.float64], oracle_values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape != oracle_values.shape:
        raise ValueError("values and oracle values must share a shape")
    return np.asarray(np.maximum(oracle_values - values, 0.0), dtype=np.float64)


def mean_regret(values: NDArray[np.float64], oracle_values: NDArray[np.float64]) -> float:
    return float(np.mean(regret(values, oracle_values)))


def abstention_accuracy(abstained: NDArray[np.bool_], truly_unsafe: NDArray[np.bool_]) -> float:
    """Among the abstentions, the share that really did exceed the safety level."""
    mask = np.asarray(abstained, dtype=np.bool_)
    unsafe = np.asarray(truly_unsafe, dtype=np.bool_)
    if not np.any(mask):
        return 0.0
    return float(np.mean(unsafe[mask]))


def abstention_degeneracy(abstention: AbstentionReport) -> bool:
    """True when the rule abstained on everything or on nothing, so coverage says nothing."""
    return bool(abstention.abstention_rate <= 0.0 or abstention.abstention_rate >= 1.0)


def absolute_risk_difference(
    exposed_events: int,
    exposed_total: int,
    reference_events: int,
    reference_total: int,
) -> dict[str, float]:
    """Risk difference for a rare endpoint, with its Wald interval."""
    if exposed_total == 0 or reference_total == 0:
        raise ValueError("both arms need a non-zero denominator")
    exposed = exposed_events / exposed_total
    reference = reference_events / reference_total
    difference = float(exposed - reference)
    variance = float(
        exposed * (1.0 - exposed) / exposed_total + reference * (1.0 - reference) / reference_total
    )
    half = 1.96 * float(np.sqrt(max(variance, 0.0)))
    return {
        "risk_exposed": float(exposed),
        "risk_reference": float(reference),
        "absolute_risk_difference": difference,
        "lower": difference - half,
        "upper": difference + half,
    }


def safety_gain_from_constraint(
    unconstrained_violation: float,
    constrained_violation: float,
) -> float:
    return float(unconstrained_violation - constrained_violation)


def coverage_weighted_safety(gain: float, coverage: float) -> float:
    """Safety gain attributable to the constraint rather than to abstention."""
    if coverage <= EPS:
        return 0.0
    return float(gain / coverage)


def mean_selected_risk(selected_risks: NDArray[np.float64], abstained: NDArray[np.bool_]) -> float:
    answered = ~np.asarray(abstained, dtype=np.bool_)
    if not np.any(answered):
        return float("nan")
    return float(np.mean(np.asarray(selected_risks, dtype=np.float64)[answered]))
