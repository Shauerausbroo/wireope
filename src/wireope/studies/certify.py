"""Certification of the strategy library: estimates, bounds, the support floor and abstention.

Each strategy is estimated with the nested doubly-robust estimator of Eq. (4). The utility is
the graded crossing progress on the ladder, the risk is the endpoint risk in the product form
of Eq. (6), and both carry a confidence radius so the rule of Eq. (5) has a lower limit to
maximise and an upper limit to constrain.

Ref: Sec. 3.1, Sec. 3.4 and Sec. 3.5, Eq. (4)-(6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.empirical_bernstein import empirical_bernstein_bound
from wireope.certification.eligibility import EligibilityReport, evaluate_floor
from wireope.certification.rule import (
    CertificationOutcome,
    StrategyEstimate,
    select_constrained,
    select_unconstrained,
)
from wireope.estimators.base import LogBatch
from wireope.estimators.nested_dr import (
    graded_progress_value,
    nested_doubly_robust,
    product_risk_from_marginals,
)
from wireope.studies.pipeline import AnalysisContext, ArmSwitches, batch_for_strategy


@dataclass(frozen=True)
class CertifiedLibrary:
    strategy: str
    estimates: tuple[StrategyEstimate, ...]
    floors: dict[str, EligibilityReport]
    constrained: CertificationOutcome
    unconstrained: CertificationOutcome
    support_floor_applied: bool


def ladder_bounds(
    batch: LogBatch,
    delta: float,
    bounded_range: float,
) -> tuple[float, float]:
    """Radius of the graded severity and of the endpoint risk for one batch.

    The severity radius comes from the empirical-Bernstein bound of the per-rung observations
    under the reconstructed weights, which is the form the manuscript reports with; the risk
    radius is the same construction applied to the endpoint rung.
    """
    weights = batch.importance_weights()
    counts = np.asarray(
        [np.count_nonzero(batch.rung_observations[:, rung] > 0.0) for rung in range(batch.rungs)],
        dtype=np.float64,
    )
    severity = empirical_bernstein_bound(batch.rung_observations, weights, counts, delta, bounded_range)
    endpoint_batch = LogBatch(
        features=batch.features,
        behaviour_strategy=batch.behaviour_strategy,
        target_strategy=batch.target_strategy,
        rung_observations=batch.rung_observations[:, -1:],
        propensities=batch.propensities,
        weights=batch.weights[-1:],
        strata=batch.strata,
        outcome_predictions=(
            None if batch.outcome_predictions is None else batch.outcome_predictions[:, -1:]
        ),
    )
    endpoint_counts = counts[-1:]
    risk = empirical_bernstein_bound(
        endpoint_batch.rung_observations, weights, endpoint_counts, delta, bounded_range
    )
    return float(severity.total), float(risk.total)


def estimate_library(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> tuple[StrategyEstimate, ...]:
    """One bounded estimate per strategy, in the utility and risk coordinates of the rule."""
    arm = switches if switches is not None else context.switches
    delta = context.config.confidence.delta
    bounded_range = context.config.confidence.bounded_range_b
    estimates: list[StrategyEstimate] = []
    for strategy in context.strategies:
        batch = batch_for_strategy(context, strategy, member, arm)
        result = nested_doubly_robust(batch, clipping)
        utility = graded_progress_value(result.per_rung_value, batch.weights)
        risk = product_risk_from_marginals(result.per_rung_value)
        severity_radius, risk_radius = ladder_bounds(batch, delta, bounded_range)
        estimates.append(
            StrategyEstimate(
                strategy=strategy.name,
                graded_value=utility,
                risk=risk,
                graded_radius=severity_radius,
                risk_radius=risk_radius,
                effective_sample_size=result.effective_sample_size,
                overlap=result.overlap,
                eligible=True,
            )
        )
    return tuple(estimates)


def eligibility_reports(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    switches: ArmSwitches | None = None,
) -> dict[str, EligibilityReport]:
    """Per-stratum, per-rung behaviour mass for every strategy against the floor."""
    arm = switches if switches is not None else context.switches
    floor = context.config.support.floor_c
    reports: dict[str, EligibilityReport] = {}
    for strategy in context.strategies:
        batch = batch_for_strategy(context, strategy, member, arm)
        reports[strategy.name] = evaluate_floor(
            strategy=strategy.name,
            strata=batch.strata,
            rung_indicators=batch.rung_observations,
            behaviour_probability=batch.propensities,
            floor=floor,
        )
    return reports


def certify_library(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> CertifiedLibrary:
    arm = switches if switches is not None else context.switches
    estimates = estimate_library(context, member, clipping, arm)
    floors = eligibility_reports(context, member, arm)
    constrained_estimates = tuple(
        StrategyEstimate(
            strategy=item.strategy,
            graded_value=item.graded_value,
            risk=item.risk,
            graded_radius=item.graded_radius,
            risk_radius=item.risk_radius,
            effective_sample_size=item.effective_sample_size,
            overlap=item.overlap,
            eligible=floors[item.strategy].eligible,
        )
        for item in estimates
    )
    threshold = context.config.sweep.risk_margin_m
    if arm.support_constraint:
        constrained = select_constrained(constrained_estimates, threshold, support_floor_applied=True)
    else:
        constrained = select_unconstrained(constrained_estimates)
    unconstrained = select_unconstrained(constrained_estimates)
    return CertifiedLibrary(
        strategy=constrained.selected,
        estimates=constrained_estimates,
        floors=floors,
        constrained=constrained,
        unconstrained=unconstrained,
        support_floor_applied=arm.support_constraint,
    )
