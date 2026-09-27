"""Support-constrained lower-confidence selection with abstention.

Among the strategies that clear the per-stratum overlap floor, the rule maximises a lower
confidence limit on the graded value subject to an upper confidence limit on the endpoint
risk being at most the pre-set threshold. When no eligible strategy satisfies the
constraint the rule abstains, and abstention is reported as an outcome with its rate and
its coverage so that a safety gain cannot be produced by declining to answer.

Ref: Sec. 3.1 and Sec. 3.5, Eq. (2) and Eq. (5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.effective_sample import relative_error
from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class StrategyEstimate:
    strategy: str
    graded_value: float
    risk: float
    graded_radius: float
    risk_radius: float
    effective_sample_size: float
    overlap: float
    eligible: bool = True

    @property
    def graded_lcb(self) -> float:
        return float(self.graded_value - self.graded_radius)

    @property
    def risk_lcb(self) -> float:
        """Lower confidence limit of the endpoint risk; the quantity Eq. (2) constrains."""
        return float(self.risk - self.risk_radius)

    @property
    def risk_ucb(self) -> float:
        return float(self.risk + self.risk_radius)


@dataclass(frozen=True)
class CertificationOutcome:
    selected: str
    abstained: bool
    constrained: bool
    selected_lcb: float
    selected_risk_lcb: float
    selected_risk_ucb: float
    eligible_count: int
    eligible_strategies: tuple[str, ...]
    reason: str = ""


@dataclass
class CertificationTrace:
    outcomes: list[CertificationOutcome] = field(default_factory=list)

    def abstention_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return float(np.mean([outcome.abstained for outcome in self.outcomes]))

    def coverage(self) -> float:
        return float(1.0 - self.abstention_rate())


def select_constrained(
    estimates: tuple[StrategyEstimate, ...],
    risk_margin: float,
    support_floor_applied: bool = True,
) -> CertificationOutcome:
    """Eq. (2): maximise the utility lower limit subject to a risk lower limit at most the margin.

    The manuscript writes this constraint as a lower confidence limit on the risk estimator
    compared against a margin, which is the direction implemented here; Eq. (5) writes an
    upper limit against a threshold for the same functional. The release follows Eq. (2) and
    records the conflict between the two printed forms as a finding on the manuscript.
    """
    eligible = tuple(estimate for estimate in estimates if estimate.eligible or not support_floor_applied)
    if not eligible:
        return CertificationOutcome(
            selected="",
            abstained=True,
            constrained=support_floor_applied,
            selected_lcb=float("nan"),
            selected_risk_lcb=float("nan"),
            selected_risk_ucb=float("nan"),
            eligible_count=0,
            eligible_strategies=(),
            reason="no strategy cleared the per-stratum overlap floor",
        )
    feasible = [estimate for estimate in eligible if estimate.risk_lcb <= risk_margin]
    if not feasible:
        return CertificationOutcome(
            selected="",
            abstained=True,
            constrained=support_floor_applied,
            selected_lcb=float("nan"),
            selected_risk_lcb=float("nan"),
            selected_risk_ucb=float("nan"),
            eligible_count=len(eligible),
            eligible_strategies=tuple(estimate.strategy for estimate in eligible),
            reason="no eligible strategy satisfied the risk margin",
        )
    best = max(feasible, key=lambda estimate: estimate.graded_lcb)
    return CertificationOutcome(
        selected=best.strategy,
        abstained=False,
        constrained=support_floor_applied,
        selected_lcb=best.graded_lcb,
        selected_risk_lcb=best.risk_lcb,
        selected_risk_ucb=best.risk_ucb,
        eligible_count=len(eligible),
        eligible_strategies=tuple(estimate.strategy for estimate in eligible),
        reason="highest utility lower limit among the strategies within the risk margin",
    )


def select_unconstrained(
    estimates: tuple[StrategyEstimate, ...],
) -> CertificationOutcome:
    """The comparison arm: argmax of the graded value with neither floor nor risk constraint."""
    if not estimates:
        return CertificationOutcome(
            selected="",
            abstained=True,
            constrained=False,
            selected_lcb=float("nan"),
            selected_risk_lcb=float("nan"),
            selected_risk_ucb=float("nan"),
            eligible_count=0,
            eligible_strategies=(),
            reason="empty strategy library",
        )
    best = max(estimates, key=lambda estimate: estimate.graded_value)
    return CertificationOutcome(
        selected=best.strategy,
        abstained=False,
        constrained=False,
        selected_lcb=best.graded_lcb,
        selected_risk_lcb=best.risk_lcb,
        selected_risk_ucb=best.risk_ucb,
        eligible_count=len(estimates),
        eligible_strategies=tuple(estimate.strategy for estimate in estimates),
        reason="argmax of the utility with no floor and no risk margin",
    )


def selection_accuracy(
    chosen: NDArray[np.str_],
    best_by_truth: NDArray[np.str_],
) -> float:
    """Share of decisions on which the certified strategy is the best strategy under truth."""
    if chosen.size == 0:
        return 0.0
    return float(np.mean(chosen == best_by_truth))


def regret(
    chosen_value: float,
    best_value: float,
) -> float:
    return float(max(best_value - chosen_value, 0.0))


def abstention_accuracy(
    abstained: NDArray[np.bool_],
    truly_unsafe: NDArray[np.bool_],
) -> float:
    """Share of the abstentions that were decisions actually exceeding the safety level."""
    if not np.any(abstained):
        return 0.0
    return float(np.mean(truly_unsafe[abstained]))


def parity_holds(
    constrained_selection_accuracy: float,
    unconstrained_selection_accuracy: float,
    margin_pp: float,
) -> bool:
    """Pre-specified parity clause: within the stated margin of the best unconstrained arm."""
    gap_pp = 100.0 * (unconstrained_selection_accuracy - constrained_selection_accuracy)
    return bool(gap_pp <= margin_pp + 1e-9)


def relative_utility_gain(constrained_value: float, comparator_value: float) -> float:
    if abs(comparator_value) <= EPS:
        return float("inf")
    return float((constrained_value - comparator_value) / abs(comparator_value))


def estimate_relative_error(estimate: StrategyEstimate, truth: float) -> float:
    return relative_error(estimate.graded_value, truth)


def net_benefit_gap(benefit: float, treat_all: float, treat_none: float) -> dict[str, float]:
    """Net benefit against the two reference strategies at one threshold."""
    return {
        "benefit": float(benefit),
        "over_treat_all": float(benefit - treat_all),
        "over_treat_none": float(benefit - treat_none),
        "beats_both": float(benefit > treat_all and benefit > treat_none),
    }
