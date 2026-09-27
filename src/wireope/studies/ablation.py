"""Component ablation and ladder-depth sensitivity, Table 2 panel A of the manuscript.

Each variant removes one component, or a pair, from the full framework. The interaction ratio
is the joint change divided by the sum of the separate changes; the manuscript computes it on
the safety-margin violation rate, so the same contrast is reported here on that axis and the
selection-accuracy contrast is reported beside it for inspection.

Ref: Sec. 3.5 and Sec. 4.2, Table 2 panel A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.laddering.severity import ladder_for_depth
from wireope.studies.arms import ArmOutcome, arm_outcomes
from wireope.studies.pipeline import AnalysisContext, ArmSwitches, fit_context


@dataclass(frozen=True)
class AblationRow:
    variant: str
    delta_selection_pp: float
    delta_violation_pp: float
    interaction_ratio: float
    effective_sample_size: float
    overlap_coefficient: float
    ladder_depth: int
    abstention_rate: float


VARIANT_SWITCHES: dict[str, ArmSwitches] = {
    "full_framework": ArmSwitches(),
    "without_envelope_excursion_reconstruction": ArmSwitches(envelope_excursion_reconstruction=False),
    "without_nested_excursion_ladder": ArmSwitches(nested_excursion_ladder=False),
    "without_support_constraint": ArmSwitches(support_constraint=False),
    "without_reconstruction_and_without_ladder": ArmSwitches(
        envelope_excursion_reconstruction=False, nested_excursion_ladder=False
    ),
    "without_ladder_and_without_support": ArmSwitches(
        nested_excursion_ladder=False, support_constraint=False
    ),
    "without_reconstruction_and_without_support": ArmSwitches(
        envelope_excursion_reconstruction=False, support_constraint=False
    ),
}


def ablation_table(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    include_depth_row: bool = True,
    resamples: int = 60,
) -> tuple[AblationRow, ...]:
    """Run every variant and express each change relative to the full framework."""
    baseline = arm_outcomes(context, member, clipping, resamples=resamples)
    rows: list[AblationRow] = [_row("full_framework", baseline, baseline)]
    for variant in VARIANT_SWITCHES:
        if variant == "full_framework":
            continue
        variant_context = fit_context(context.bundle, context.configuration, VARIANT_SWITCHES[variant])
        outcome = arm_outcomes(variant_context, member, clipping, resamples=resamples)
        rows.append(_row(variant, outcome, baseline))
    if include_depth_row:
        depth = max(context.config.ladder.additional_depths)
        outcome = arm_outcomes(context, member, clipping, depth=depth, resamples=resamples)
        rows.append(_row("depth_sensitivity_deepest_ladder_tested", outcome, baseline))
    ratios = {row.variant: row for row in rows}
    return tuple(_with_ratios(ratios, row) for row in rows)


def _row(variant: str, outcome: ArmOutcome, baseline: ArmOutcome) -> AblationRow:
    return AblationRow(
        variant=variant,
        delta_selection_pp=100.0 * (outcome.selection_accuracy - baseline.selection_accuracy),
        delta_violation_pp=100.0 * (outcome.violation_rate - baseline.violation_rate),
        interaction_ratio=1.0,
        effective_sample_size=outcome.effective_sample_size,
        overlap_coefficient=outcome.overlap_coefficient,
        ladder_depth=outcome.ladder_depth,
        abstention_rate=outcome.abstention_rate,
    )


def _with_ratios(by_variant: dict[str, AblationRow], row: AblationRow) -> AblationRow:
    """Fill the interaction ratio of a pair row from the two single removals it combines."""
    pairs = {
        "without_reconstruction_and_without_ladder": (
            "without_envelope_excursion_reconstruction",
            "without_nested_excursion_ladder",
        ),
        "without_ladder_and_without_support": (
            "without_nested_excursion_ladder",
            "without_support_constraint",
        ),
        "without_reconstruction_and_without_support": (
            "without_envelope_excursion_reconstruction",
            "without_support_constraint",
        ),
    }
    if row.variant not in pairs:
        return row
    first, second = pairs[row.variant]
    if first not in by_variant or second not in by_variant:
        return row
    additive = by_variant[first].delta_violation_pp + by_variant[second].delta_violation_pp
    if abs(additive) < 1e-9:
        return row
    return AblationRow(
        variant=row.variant,
        delta_selection_pp=row.delta_selection_pp,
        delta_violation_pp=row.delta_violation_pp,
        interaction_ratio=float(row.delta_violation_pp / additive),
        effective_sample_size=row.effective_sample_size,
        overlap_coefficient=row.overlap_coefficient,
        ladder_depth=row.ladder_depth,
        abstention_rate=row.abstention_rate,
    )


def interaction_ratio_from_deltas(
    joint: float,
    first: float,
    second: float,
) -> float:
    """Ratio of a joint change to the sum of the separate changes on one axis."""
    additive = first + second
    if abs(additive) < 1e-12:
        return float("nan")
    return float(joint / additive)


def removal_is_non_monotone(
    unconstrained_selection_accuracy: float,
    constrained_selection_accuracy: float,
    unconstrained_violation: float,
    constrained_violation: float,
) -> bool:
    """True when removing the constraint improves utility while worsening the safety axis."""
    return bool(
        unconstrained_selection_accuracy > constrained_selection_accuracy
        and unconstrained_violation > constrained_violation
    )


def ladder_depth_sensitivity(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    depths: tuple[int, ...],
    resamples: int = 40,
) -> dict[int, dict[str, float]]:
    """Utility, violation rate and abstention as the ladder deepens."""
    table: dict[int, dict[str, float]] = {}
    for depth in depths:
        if depth > len(context.config.ladder.tau_mm):
            continue
        outcome = arm_outcomes(context, member, clipping, depth=depth, resamples=resamples)
        ladder = ladder_for_depth(context.config.ladder, context.config.ladder_weights, depth)
        resolved = (
            sum(
                1
                for earlier, later in zip(ladder.tau_mm, ladder.tau_mm[1:])
                if later - earlier > context.config.ladder.resolution_band_mm
            )
            + 1
        )
        table[depth] = {
            "selection_accuracy": outcome.selection_accuracy,
            "violation_rate": outcome.violation_rate,
            "abstention_rate": outcome.abstention_rate,
            "effective_sample_size": outcome.effective_sample_size,
            "rung_resolution_depth": float(resolved),
        }
    return table


def abstention_summary(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    resamples: int = 60,
) -> dict[str, float]:
    outcome = arm_outcomes(context, member, clipping, resamples=resamples)
    return {
        "abstention_rate": outcome.abstention_rate,
        "coverage": 1.0 - outcome.abstention_rate,
        "eligible": outcome.eligible_count,
    }
