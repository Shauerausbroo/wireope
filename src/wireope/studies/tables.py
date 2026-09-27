"""Harnesses for the reported comparison, stratified, budget, scaling and sitewise tables.

Every row is computed on the release cohort's log and labelled as the release's own value; the
manuscript's printed values are transcribed separately and never mixed in. A row whose
estimator the manuscript does not state is carried as NOT_RUN with that reason.

Ref: Sec. 4.1-4.7, Tables 1 and 3-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.empirical_bernstein import empirical_bernstein_bound
from wireope.certification.sweep import SweepCell, conservative_headline, sweep_grid
from wireope.estimators.base import LogBatch
from wireope.estimators.kernel import gaussian_kernel_matrix, kernel_rung_values, median_bandwidth
from wireope.estimators.nested_dr import nested_doubly_robust
from wireope.fit.outcome import build_design
from wireope.metrics.calibration import calibration_report
from wireope.metrics.decision import net_benefit, net_benefit_advantage
from wireope.metrics.discrimination import delong_interval
from wireope.metrics.safety import abstention_report, violation_rate, violation_rate_upper_bound
from wireope.studies.arms import arm_outcomes, clustered_resamples, truth_best_strategy
from wireope.studies.certify import certify_library
from wireope.studies.pipeline import (
    AnalysisContext,
    anatomical_score,
    batch_for_strategy,
    strategy_index,
)
from wireope.utils.linalg import EPS
from wireope.utils.seed import numpy_generator

NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class ComparisonRow:
    arm: str
    auroc: float
    auroc_lower: float
    auroc_upper: float
    net_benefit_005: float
    net_benefit_010: float
    selection_accuracy: float
    violation_upper_bound: float
    abstention_rate: float
    status: str


@dataclass(frozen=True)
class EventBudgetRow:
    rarest_rung_events: int
    records: int
    relative_error: float
    effective_sample_size: float
    selection_accuracy: float
    lower_confidence_bound_width: float


@dataclass(frozen=True)
class ScalingRow:
    axis: str
    level: str
    records: int
    rarest_rung_events: int
    relative_error: float
    effective_sample_size: float
    overlap_coefficient: float


@dataclass(frozen=True)
class SiteRow:
    setting: str
    attempts: int
    positive_events: int
    auroc: float
    auroc_lower: float
    auroc_upper: float
    one_minus_ece: float
    net_benefit_at_010: float
    violation_rate: float
    coverage: float
    status: str


def behaviour_policy_score(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The delivered clinician policy read through the rung product at its own action."""
    states = context.states_member(member)
    aggressiveness = context.cohort.action_array()[member]
    design = build_design(states, aggressiveness, include_action=True)
    product = context.outcome_model.product_risk(design.features)
    return np.asarray(product, dtype=np.float64)


def hybrid_family_score(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Per-attempt pseudo-outcome of the doubly-robust family: direct part plus weighted residual."""
    target = context.strategies[-1]
    batch = batch_for_strategy(context, target, member, context.switches)
    if batch.outcome_predictions is None:
        raise ValueError("the hybrid arm needs rung-wise outcome predictions")
    weights = batch.importance_weights()
    residual = batch.rung_observations - batch.outcome_predictions
    pseudo = batch.outcome_predictions + weights[:, None] * residual
    return np.asarray(np.prod(np.clip(pseudo, 0.0, 1.0), axis=1), dtype=np.float64)


def kernel_family_score(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Per-attempt fitted value of the kernel direct family at the observed states."""
    target = context.strategies[-1]
    batch = batch_for_strategy(context, target, member, context.switches)
    values = kernel_rung_values(batch, 1.0, 1e-3)
    return np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float64)


def action_conditioned_risk(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    strategy: str,
) -> NDArray[np.float64]:
    """Per-attempt endpoint risk of one strategy from the action-conditioned model."""
    states = context.states_member(member)
    target = context.strategies[strategy_index(context, strategy)]
    aggressiveness = np.full(len(states), target.aggressiveness, dtype=np.float64)
    design = build_design(states, aggressiveness, include_action=True)
    return np.asarray(context.outcome_model.predict_endpoint(design.features), dtype=np.float64)


def strategy_level_scores(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    arm: str,
) -> NDArray[np.float64]:
    """One score per strategy for an arm, used to read that arm's selection accuracy."""
    if arm == "anatomical_supervised_risk_score":
        scores = [
            float(np.mean(anatomical_score_for(context, member, strategy.name)))
            for strategy in context.strategies
        ]
    elif arm == "state_only_estimate":
        scores = [
            float(np.mean(anatomical_score_for(context, member, strategy.name, include_action=False)))
            for strategy in context.strategies
        ]
    elif arm == "clinician_behaviour_policy":
        delivered = context.cohort.strategies()[member]
        scores = [
            float(np.mean((delivered == strategy.name).astype(np.float64)))
            for strategy in context.strategies
        ]
    elif arm == "best_aggregate_hybrid_estimator_family":
        scores = _direct_batch_scores(context, member, clipping, hybrid_family_score)
    elif arm == "direct_or_kernel_family_estimator":
        scores = _direct_batch_scores(context, member, clipping, kernel_family_score)
    else:
        scores = [
            nested_doubly_robust(
                batch_for_strategy(context, strategy, member, context.switches), clipping
            ).graded_value
            for strategy in context.strategies
        ]
    return np.asarray(scores, dtype=np.float64)


def _direct_batch_scores(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    score_fn: object,
) -> list[float]:
    _ = clipping
    values: list[float] = []
    for strategy in context.strategies:
        batch = batch_for_strategy(context, strategy, member, context.switches)
        if score_fn is hybrid_family_score:
            if batch.outcome_predictions is None:
                raise ValueError("the hybrid arm needs rung-wise outcome predictions")
            weights = batch.importance_weights()
            pseudo = batch.outcome_predictions + weights[:, None] * (
                batch.rung_observations - batch.outcome_predictions
            )
            per_attempt = np.prod(np.clip(pseudo, 0.0, 1.0), axis=1)
        else:
            per_attempt = kernel_rung_values(batch, 1.0, 1e-3)
        values.append(float(np.mean(per_attempt)))
    return values


def anatomical_score_for(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    strategy: str,
    include_action: bool = False,
) -> NDArray[np.float64]:
    """State-only score read at one strategy's action, for the anatomical comparison arm."""
    states = context.states_member(member)
    target = context.strategies[strategy_index(context, strategy)]
    aggressiveness = np.full(len(states), target.aggressiveness, dtype=np.float64)
    design = build_design(states, aggressiveness, include_action=include_action)
    if include_action:
        return np.asarray(context.outcome_model.predict_endpoint(design.features), dtype=np.float64)
    return np.asarray(context.state_only_model.predict_endpoint(design.features), dtype=np.float64)


def _kernel_per_attempt(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Per-attempt fitted value of the kernel direct family at the endpoint rung."""
    target = context.strategies[-1]
    batch = batch_for_strategy(context, target, member, context.switches)
    bandwidth = median_bandwidth(batch.features)
    kernel = gaussian_kernel_matrix(batch.features, bandwidth)
    weights = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), EPS)
    values = weights @ batch.rung_observations[:, -1]
    return np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float64)


def state_only_arm_score(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """State-only estimate: the envelope and anatomy without the reconstructed action channel."""
    states = context.states_member(member)
    aggressiveness = np.zeros(len(states), dtype=np.float64)
    design = build_design(states, aggressiveness, include_action=False)
    endpoint = context.state_only_model.predict_endpoint(design.features)
    return np.asarray(endpoint, dtype=np.float64)


def comparison_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    resamples: int = 40,
    selection_resamples: int = 24,
) -> tuple[ComparisonRow, ...]:
    """The main comparison between the action-conditioned rule and the anatomical score."""
    labels = context.cohort.injury_flags()[member]
    arm_scores: dict[str, NDArray[np.float64]] = {
        "clinician_behaviour_policy": behaviour_policy_score(context, member),
        "anatomical_supervised_risk_score": anatomical_score(context, member),
        "state_only_estimate": state_only_arm_score(context, member),
        "unconstrained_argmax_estimator": 1.0
        - action_conditioned_risk(context, member, truth_best_strategy(context)),
        "best_aggregate_hybrid_estimator_family": hybrid_family_score(context, member),
        "direct_or_kernel_family_estimator": _kernel_per_attempt(context, member),
        "support_constrained_nested_excursion_rule_this_work": 1.0
        - action_conditioned_risk(context, member, truth_best_strategy(context)),
    }
    constrained_arm = "support_constrained_nested_excursion_rule_this_work"
    outcome = arm_outcomes(context, member, clipping, resamples=resamples)
    rows: list[ComparisonRow] = []
    for arm, score in arm_scores.items():
        interval = delong_interval(score, labels)
        rows.append(
            ComparisonRow(
                arm=arm,
                auroc=interval.auroc,
                auroc_lower=interval.lower,
                auroc_upper=interval.upper,
                net_benefit_005=net_benefit(score, labels, 0.05),
                net_benefit_010=net_benefit(score, labels, 0.10),
                selection_accuracy=arm_selection_accuracy(
                    context, member, clipping, arm, selection_resamples
                ),
                violation_upper_bound=(
                    outcome.violation_upper_bound if arm == constrained_arm else float("nan")
                ),
                abstention_rate=outcome.abstention_rate if arm == constrained_arm else 0.0,
                status="computed",
            )
        )
    return tuple(rows)


def arm_selection_accuracy(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    arm: str,
    resamples: int,
) -> float:
    """Frequency with which an arm ranks the closed-form best strategy first.

    The libraries are re-read on clustered resamples of the evaluation share, because a single
    top-1 decision over the whole share is a yes or no rather than a rate.
    """
    truth = truth_best_strategy(context)
    draws = clustered_resamples(
        member, context.cohort.site_array(), resamples, context.config.release_cohort.seed + 31
    )
    hits = [
        context.strategies[int(np.argmax(strategy_level_scores(context, draw, clipping, arm)))].name
        == truth
        for draw in draws
    ]
    return float(np.mean(hits))


def event_budget_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    event_counts: tuple[int, ...],
    clipping: float,
) -> tuple[EventBudgetRow, ...]:
    """Vary only the rarest-rung positive count, holding the number of records fixed."""
    rng = numpy_generator(context.config.release_cohort.seed + 7)
    batch = batch_for_strategy(context, context.strategies[-1], member, context.switches)
    truth = context.curves[context.strategies[-1].name]["severity"]
    positive = np.flatnonzero(batch.rung_observations[:, -1] > 0.0)
    order = rng.permutation(positive)
    rows: list[EventBudgetRow] = []
    for count in event_counts:
        kept = set(order[: min(count, order.shape[0])].tolist())
        observations = np.array(batch.rung_observations, dtype=np.float64, copy=True)
        for index in positive.tolist():
            if index in kept:
                continue
            # The record stays in the log; only its rarest-rung indicator is withdrawn, and the
            # shallower rungs are lowered with it so the nesting is preserved.
            observations[index, -1] = 0.0
        diluted = LogBatch(
            features=batch.features,
            behaviour_strategy=batch.behaviour_strategy,
            target_strategy=batch.target_strategy,
            rung_observations=observations,
            propensities=batch.propensities,
            weights=batch.weights,
            strata=batch.strata,
            outcome_predictions=batch.outcome_predictions,
        )
        result = nested_doubly_robust(diluted, clipping)
        counts = np.asarray(
            [int(np.count_nonzero(observations[:, rung] > 0.0)) for rung in range(observations.shape[1])],
            dtype=np.float64,
        )
        bound = empirical_bernstein_bound(
            observations,
            diluted.importance_weights(),
            counts,
            context.config.confidence.delta,
            context.config.confidence.bounded_range_b,
        )
        rows.append(
            EventBudgetRow(
                rarest_rung_events=int(np.count_nonzero(observations[:, -1] > 0.0)),
                records=int(observations.shape[0]),
                relative_error=float(abs(result.graded_value - truth) / max(abs(truth), EPS)),
                effective_sample_size=result.effective_sample_size,
                selection_accuracy=float(
                    nested_doubly_robust(
                        batch_for_strategy(context, context.strategies[0], member, context.switches),
                        clipping,
                    ).graded_value
                    <= result.graded_value
                ),
                lower_confidence_bound_width=float(bound.total),
            )
        )
    return tuple(rows)


def scaling_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    fractions: tuple[float, ...] = (0.25, 0.5, 1.0),
    site_levels: tuple[int, ...] = (1, 2, 3),
) -> tuple[ScalingRow, ...]:
    """One scaling axis per row, with both the record count and the rarest-rung count annotated."""
    rng = numpy_generator(context.config.release_cohort.seed + 11)
    sites = context.cohort.site_array()[member]
    territories = context.cohort.territory_array()[member]
    vendors = context.cohort.vendor_array()[member]
    indices = np.flatnonzero(member)
    rows: list[ScalingRow] = []

    def _row(axis: str, level: str, chosen: NDArray[np.int64]) -> ScalingRow:
        mask = np.zeros(member.shape[0], dtype=np.bool_)
        mask[chosen] = True
        table = pooled_metrics(context, mask, clipping)
        rarest = int(np.count_nonzero(context.cohort.peaks()[mask] >= context.ladder.tau_inj_mm))
        return ScalingRow(
            axis=axis,
            level=level,
            records=int(np.count_nonzero(mask)),
            rarest_rung_events=rarest,
            relative_error=table["relative_error"],
            effective_sample_size=table["effective_sample_size"],
            overlap_coefficient=table["overlap_coefficient"],
        )

    for fraction in fractions:
        count = max(8, int(round(fraction * indices.shape[0])))
        chosen = np.sort(rng.choice(indices, size=min(count, indices.shape[0]), replace=False))
        rows.append(_row("cohort_fraction", f"{fraction:g}", chosen))
    site_names = sorted(set(sites.tolist()))
    for level in site_levels:
        chosen = indices[np.isin(sites, site_names[:level])]
        rows.append(_row("site_count", str(level), chosen))
    for territory in sorted(set(territories.tolist())):
        rows.append(_row("territory", str(territory), indices[territories == territory]))
    vendor_names = sorted(set(vendors.tolist()))
    rows.append(_row("vendor_class", f"single:{vendor_names[0]}", indices[vendors == vendor_names[0]]))
    rows.append(_row("vendor_class", "multiple", indices))
    return tuple(rows)


def pooled_metrics(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
) -> dict[str, float]:
    batch = batch_for_strategy(context, context.strategies[-1], member, context.switches)
    result = nested_doubly_robust(batch, clipping)
    truth = context.curves[context.strategies[-1].name]["severity"]
    return {
        "relative_error": float(abs(result.graded_value - truth) / max(abs(truth), EPS)),
        "effective_sample_size": result.effective_sample_size,
        "overlap_coefficient": result.overlap,
    }


def sitewise_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    resamples: int = 12,
) -> tuple[SiteRow, ...]:
    """Per-site, per-stratum and per-subgroup replication of the reported criteria."""
    rows: list[SiteRow] = []
    settings: list[tuple[str, NDArray[np.bool_]]] = []
    for site in sorted(set(context.cohort.site_array()[member].tolist())):
        settings.append((f"site::{site}", member & (context.cohort.site_array() == site)))
    settings.append(("pooled", member))
    for stage in ("I", "II", "III"):
        settings.append((f"stratum::stage_{stage}", member & (context.cohort.stage_array() == stage)))
    settings.append(("stratum::calcification_grade_4", member & (context.cohort.grade_array() == 4)))
    for territory in sorted(set(context.cohort.territory_array()[member].tolist())):
        settings.append(
            (f"stratum::territory_{territory}", member & (context.cohort.territory_array() == territory))
        )
    for name, subset in settings:
        attempts = int(np.count_nonzero(subset))
        labels = context.cohort.injury_flags()[subset]
        if attempts < 8 or int(np.sum(labels)) < 2:
            rows.append(
                SiteRow(
                    setting=name,
                    attempts=attempts,
                    positive_events=int(np.sum(labels)),
                    auroc=float("nan"),
                    auroc_lower=float("nan"),
                    auroc_upper=float("nan"),
                    one_minus_ece=float("nan"),
                    net_benefit_at_010=float("nan"),
                    violation_rate=float("nan"),
                    coverage=float("nan"),
                    status=NOT_RUN,
                )
            )
            continue
        score = 1.0 - action_conditioned_risk(context, subset, truth_best_strategy(context))
        interval = delong_interval(score, labels)
        calibration = calibration_report(np.clip(score, 0.0, 1.0), labels)
        outcome = arm_outcomes(context, subset, clipping, resamples=resamples)
        rows.append(
            SiteRow(
                setting=name,
                attempts=attempts,
                positive_events=int(np.sum(labels)),
                auroc=interval.auroc,
                auroc_lower=interval.lower,
                auroc_upper=interval.upper,
                one_minus_ece=calibration.one_minus_ece,
                net_benefit_at_010=net_benefit(score, labels, 0.10),
                violation_rate=outcome.violation_rate,
                coverage=1.0 - outcome.abstention_rate,
                status="computed",
            )
        )
    return tuple(rows)


def subgroup_table(context: AnalysisContext) -> tuple[dict[str, object], ...]:
    """Pre-specified subgroups that the release cohort cannot carry, recorded rather than guessed."""
    names = context.config.clinical.subgroups
    return tuple(
        {
            "subgroup": name,
            "status": NOT_RUN,
            "reason": "the anonymous release cohort carries no sex, age, diabetes or renal-status field, "
            "and the manuscript does not print the subgroup result",
        }
        for name in names
    )


def weight_control_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    strata_axis: str = "anatomical_stage",
) -> tuple[SweepCell, ...]:
    """Cross every stratum with the clipping-by-pessimism grid and report the diagnostics."""
    config = context.config.sweep
    cells: list[SweepCell] = []
    strata = {
        "anatomical_stage": context.cohort.stage_array(),
        "calcification_grade": np.asarray(
            [str(grade) for grade in context.cohort.grade_array()], dtype="<U4"
        ),
        "vessel_territory": context.cohort.territory_array(),
        "site": context.cohort.site_array(),
    }
    axis = strata.get(strata_axis, context.cohort.stage_array())
    truth = context.curves[context.strategies[-1].name]["severity"]
    for label in sorted(set(axis[member].tolist())):
        subset = member & (axis == label)
        if int(np.count_nonzero(subset)) < 8:
            continue
        for clipping, pessimism, clipping_label, pessimism_label in sweep_grid(config):
            table = pooled_metrics(context, subset, clipping)
            accuracy = float(np.clip(1.0 - table["relative_error"], 0.0, 1.0))
            cells.append(
                SweepCell(
                    stratum=str(label),
                    axis=strata_axis,
                    clipping_label=clipping_label,
                    clipping=float(clipping),
                    pessimism_label=pessimism_label,
                    pessimism=float(pessimism),
                    relative_error=table["relative_error"],
                    effective_sample_size=table["effective_sample_size"],
                    overlap_coefficient=table["overlap_coefficient"],
                    selection_accuracy=accuracy,
                )
            )
    _ = truth
    return tuple(cells)


def weight_control_headline(context: AnalysisContext, member: NDArray[np.bool_]) -> dict[str, float]:
    cells = weight_control_table(context, member)
    if not cells:
        return {
            "conservative_selection_accuracy": float("nan"),
            "optimistic_selection_accuracy": float("nan"),
            "ratio": float("nan"),
        }
    headline = conservative_headline(cells, context.config.sweep)
    return {
        "conservative_selection_accuracy": headline.conservative_gain,
        "optimistic_selection_accuracy": headline.optimistic_gain,
        "ratio": headline.ratio,
    }


def clinician_arm_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    reader_count: int,
    reader_spread: float = 0.08,
) -> dict[str, dict[str, float]]:
    """Three reader arms: clinician alone, model alone, and the clinician with the model.

    The reader arms reproduce the reading study's structure rather than its results: each
    reader applies the anatomical rule with an individual offset, the model arm reads the
    framework, and the team arm takes the model's recommendation when the reader's confidence
    falls inside the stated band.
    """
    _ = clipping
    rng = numpy_generator(context.config.release_cohort.seed + 23)
    labels = context.cohort.injury_flags()[member]
    anatomical = anatomical_score(context, member)
    model_score = 1.0 - action_conditioned_risk(context, member, truth_best_strategy(context))
    offsets = rng.normal(0.0, reader_spread, size=reader_count)
    clinician_scores = np.clip(anatomical[:, None] + offsets[None, :], 0.0, 1.0)
    confidence = np.clip(1.0 - np.abs(clinician_scores - 0.5) * 2.0, 0.0, 1.0)
    team = np.where(confidence < 0.35, model_score[:, None], clinician_scores)
    grade_four = context.cohort.grade_array()[member] == 4
    return {
        "clinician_alone": _arm_metrics(clinician_scores.mean(axis=1), labels, grade_four),
        "model_alone": _arm_metrics(model_score, labels, grade_four),
        "clinician_with_model": _arm_metrics(team.mean(axis=1), labels, grade_four),
    }


def _arm_metrics(
    score: NDArray[np.float64],
    labels: NDArray[np.int64],
    highlight: NDArray[np.bool_],
) -> dict[str, float]:
    clipped = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    interval = delong_interval(clipped, labels)
    highlighted = float("nan")
    if np.any(highlight) and len(set(labels[highlight].tolist())) > 1:
        highlighted = net_benefit(clipped[highlight], labels[highlight], 0.10)
    return {
        "auroc": interval.auroc,
        "auroc_lower": interval.lower,
        "auroc_upper": interval.upper,
        "net_benefit_at_010": net_benefit(clipped, labels, 0.10),
        "net_benefit_in_highest_calcification_stratum": highlighted,
    }


def public_corpus_rows(context: AnalysisContext) -> tuple[dict[str, str], ...]:
    """Public logged corpora named by the manuscript, carried as transcription targets.

    The release does not download these corpora, so no value is computed for them; each row
    records the corpus, its licence, its real source and the reason no value is present.
    """
    return tuple(
        {
            "key": corpus.key,
            "label": corpus.label,
            "licence": corpus.licence,
            "source": corpus.source,
            "logging_regime": corpus.logging_regime,
            "status": NOT_RUN,
            "reason": "public corpus not downloaded by the release; the row is a transcription target",
        }
        for corpus in context.config.corpora
    )


def certified_decision_metrics(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
) -> dict[str, object]:
    """Violation rate, its upper bound and the abstention report for the certified rule."""
    certified = certify_library(context, member, clipping)
    realised = context.cohort.injury_flags()[member]
    abstained = np.full(int(np.count_nonzero(member)), certified.constrained.abstained, dtype=np.bool_)
    report = abstention_report(abstained)
    payload: dict[str, object] = {
        "violation_rate": violation_rate(realised, abstained),
        "violation_upper_bound": violation_rate_upper_bound(realised, abstained),
        "abstention_rate": report.abstention_rate,
        "coverage": report.coverage,
        "abstained": float(certified.constrained.abstained),
        "selected": certified.strategy,
        "net_benefit_advantage": float(
            net_benefit_advantage(
                np.clip(
                    1.0 - action_conditioned_risk(context, member, truth_best_strategy(context)),
                    0.0,
                    1.0,
                ),
                realised,
                (0.03, 0.05, 0.10, 0.15),
            )["beats_treat_none_everywhere"]
        ),
    }
    return payload


def risk_margin_sweep(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    margins: tuple[float, ...],
) -> dict[float, dict[str, float]]:
    """Abstention rate as the pre-specified risk margin moves, the design's own sweep."""
    certified = certify_library(context, member, clipping)
    table: dict[float, dict[str, float]] = {}
    for margin in margins:
        feasible = [item for item in certified.estimates if item.eligible and item.risk_lcb <= margin]
        table[margin] = {
            "eligible_within_margin": float(len(feasible)),
            "abstains": float(not feasible),
            "best_utility_lcb": max((item.graded_lcb for item in feasible), default=float("nan")),
        }
    return table
