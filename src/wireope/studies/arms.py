"""Per-arm outcomes for the reported comparison tables.

One decision over a whole share is a single yes/no, so the reported selection accuracy and
violation rate are read off clustered bootstrap resamples of the evaluation share: the arm is
re-certified on each resample, and the accuracy is the frequency with which its selection
matches the closed-form best strategy. Resampling whole sites keeps the within-site dependence
that the whole-site hold-out creates. The endpoint under a *selected* strategy was never
observed, so the realised outcome of each resample is drawn from the release cohort's closed-form
injury probability at that strategy.

Ref: Sec. 3.7 and Sec. 4.1, Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.certification.rule import CertificationOutcome
from wireope.metrics.safety import abstention_report, violation_rate
from wireope.studies.certify import certify_library
from wireope.studies.pipeline import AnalysisContext, strategy_index
from wireope.utils.seed import numpy_generator

DEFAULT_RESAMPLES = 120


@dataclass(frozen=True)
class ArmOutcome:
    arm: str
    selection_accuracy: float
    answered_fraction: float
    violation_rate: float
    violation_upper_bound: float
    abstention_rate: float
    effective_sample_size: float
    overlap_coefficient: float
    ladder_depth: int
    eligible_count: float
    abstained: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "selection_accuracy": self.selection_accuracy,
            "answered_fraction": self.answered_fraction,
            "violation_rate": self.violation_rate,
            "violation_upper_bound": self.violation_upper_bound,
            "abstention_rate": self.abstention_rate,
            "effective_sample_size": self.effective_sample_size,
            "overlap_coefficient": self.overlap_coefficient,
            "ladder_depth": float(self.ladder_depth),
            "eligible_count": self.eligible_count,
            "abstained": self.abstained,
        }


def truth_best_strategy(context: AnalysisContext, key: str = "graded_progress") -> str:
    return max(context.curves, key=lambda name: context.curves[name][key])


def clustered_resamples(
    member: NDArray[np.bool_],
    clusters: NDArray[np.str_],
    resamples: int,
    seed: int,
) -> list[NDArray[np.bool_]]:
    """Bootstrap the share by drawing whole clusters with replacement."""
    rng = numpy_generator(seed)
    indices = np.flatnonzero(member)
    names = sorted(set(clusters[indices].tolist()))
    grouped = [indices[clusters[indices] == name] for name in names]
    draws: list[NDArray[np.bool_]] = []
    for _ in range(resamples):
        pick = rng.integers(0, len(grouped), size=len(grouped))
        chosen = np.concatenate([grouped[position] for position in pick])
        mask = np.zeros(member.shape[0], dtype=np.bool_)
        mask[chosen] = True
        draws.append(mask)
    return draws


def _selected_outcome(
    context: AnalysisContext,
    draw: NDArray[np.bool_],
    outcome: CertificationOutcome,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Draw the endpoint under the selected strategy and carry the abstention flag."""
    count = int(np.count_nonzero(draw))
    abstained = np.full(count, outcome.abstained, dtype=np.bool_)
    if outcome.abstained:
        return np.zeros(count, dtype=np.int64), abstained
    column = strategy_index(context, outcome.selected)
    probability = context.bundle.state_risk[draw][:, column]
    realised = (rng.random(probability.shape[0]) < probability).astype(np.int64)
    return realised, abstained


def arm_outcomes(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    depth: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
    truth_key: str = "graded_progress",
) -> ArmOutcome:
    """Re-certify the library on clustered resamples and aggregate the reported quantities."""
    arm = context.switches
    run_seed = context.config.release_cohort.seed if seed is None else seed
    draws = clustered_resamples(member, context.cohort.site_array(), resamples, run_seed)
    truth = truth_best_strategy(context, truth_key)
    rng = numpy_generator(run_seed + 1)
    selections: list[str] = []
    abstained_flags: list[bool] = []
    outcomes: list[CertificationOutcome] = []
    effective: list[float] = []
    overlap: list[float] = []
    upper_bounds: list[float] = []
    realised: list[NDArray[np.int64]] = []
    flagged: list[NDArray[np.bool_]] = []
    for draw in draws:
        library = certify_library(context, draw, clipping)
        outcome = library.constrained
        outcomes.append(outcome)
        selections.append(outcome.selected)
        abstained_flags.append(outcome.abstained)
        upper_bounds.append(outcome.selected_risk_ucb)
        if outcome.selected:
            chosen = next(item for item in library.estimates if item.strategy == outcome.selected)
            effective.append(chosen.effective_sample_size)
            overlap.append(chosen.overlap)
        realised_draw, flagged_draw = _selected_outcome(context, draw, outcome, rng)
        realised.append(realised_draw)
        flagged.append(flagged_draw)
    answered = [not flag for flag in abstained_flags]
    correct = (
        float(np.mean([name == truth for name, keep in zip(selections, answered) if keep]))
        if any(answered)
        else 0.0
    )
    pooled_injuries = np.concatenate(realised)
    pooled_abstained = np.concatenate(flagged)
    report = abstention_report(pooled_abstained)
    return ArmOutcome(
        arm="support_constrained" if arm.support_constraint else "unconstrained",
        selection_accuracy=correct,
        answered_fraction=float(np.mean(answered)),
        violation_rate=violation_rate(pooled_injuries, pooled_abstained),
        violation_upper_bound=float(np.mean(upper_bounds)),
        abstention_rate=report.abstention_rate,
        effective_sample_size=float(np.mean(effective)) if effective else 0.0,
        overlap_coefficient=float(np.mean(overlap)) if overlap else 0.0,
        ladder_depth=context.ladder_for(arm).depth if depth is None else depth,
        eligible_count=float(np.mean([outcome.eligible_count for outcome in outcomes])),
        abstained=float(np.mean(abstained_flags)),
    )


def counterfactual_risk(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    strategy: str,
) -> float:
    """Closed-form endpoint risk of one strategy over the evaluated share."""
    column = strategy_index(context, strategy)
    return float(np.mean(context.bundle.state_risk[member][:, column]))


def selection_gap_to_best(
    context: AnalysisContext,
    selected: str,
    key: str = "graded_progress",
) -> float:
    best = truth_best_strategy(context, key)
    return float(context.curves[best][key] - context.curves[selected][key])
