"""Estimator bake-off on identical logs, splits, budgets and preprocessing.

The manuscript's second contribution is that the aggregate benchmark winner is not the winner
here, and that the quantity separating the families is the overlap coefficient of the strata
that produce the result rather than the sum of the families' benchmark ranks. This study
recomputes every candidate on one batch and reports the diagnostic that selected it.

Ref: Sec. 3.4 and Sec. 4.2, Table 2 panel B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.effective_sample import relative_error
from wireope.estimators.base import LogBatch
from wireope.estimators.registry import (
    FAMILY_BY_KEY,
    build_estimator,
    family_instability,
    within_ten_percent_share,
)
from wireope.studies.pipeline import AnalysisContext, ArmSwitches, batch_for_strategy


@dataclass(frozen=True)
class BakeoffRow:
    key: str
    family: str
    relative_error: float
    effective_sample_size: float
    overlap_coefficient: float
    selection_accuracy: float
    within_ten_percent: float
    value: float
    diagnostic: str


def _family_batch(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    switches: ArmSwitches,
    clipping: float,
) -> LogBatch:
    """Use the most aggressive strategy as the target, the arm the bench compares on."""
    target = context.strategies[-1]
    return batch_for_strategy(context, target, member, switches)


def run_bakeoff(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> tuple[BakeoffRow, ...]:
    """Estimate the graded severity with every candidate on one target strategy."""
    arm = switches if switches is not None else context.switches
    batch = _family_batch(context, member, arm, clipping)
    truth = context.curves[context.strategies[-1].name]
    rows: list[BakeoffRow] = []
    for key in sorted(FAMILY_BY_KEY):
        estimator = build_estimator(key, delta=context.config.confidence.delta)
        result = estimator.estimate(batch, clipping)
        rows.append(
            BakeoffRow(
                key=key,
                family=result.family,
                relative_error=relative_error(result.value, truth["severity"]),
                effective_sample_size=result.effective_sample_size,
                overlap_coefficient=result.overlap_coefficient,
                selection_accuracy=float("nan"),
                within_ten_percent=float("nan"),
                value=result.value,
                diagnostic=_diagnostic(result.family, result.diagnostics),
            )
        )
    return tuple(rows)


def strategy_selection_accuracy(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> dict[str, float]:
    """Share of the library each candidate ranks first against the closed-form ordering.

    Every candidate is run on every strategy, so the accuracy is the frequency with which the
    estimator's argmax matches the argmax of the closed-form graded severity.
    """
    arm = switches if switches is not None else context.switches
    truth = np.asarray(
        [context.curves[strategy.name]["severity"] for strategy in context.strategies],
        dtype=np.float64,
    )
    ordered = {strategy.name: strategy for strategy in context.strategies}
    accuracy: dict[str, float] = {}
    for key in sorted(FAMILY_BY_KEY):
        estimator = build_estimator(key, delta=context.config.confidence.delta)
        values = np.asarray(
            [
                estimator.estimate(
                    batch_for_strategy(context, ordered[strategy.name], member, arm), clipping
                ).value
                for strategy in context.strategies
            ],
            dtype=np.float64,
        )
        accuracy[key] = float(np.mean(np.argmax(values) == np.argmax(truth)))
    return accuracy


def within_ten_percent_column(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> dict[str, float]:
    """Near-top-frequency column, on this corpus rather than on the published benchmarks."""
    arm = switches if switches is not None else context.switches
    truth = np.asarray(
        [context.curves[strategy.name]["severity"] for strategy in context.strategies],
        dtype=np.float64,
    )
    column: dict[str, float] = {}
    for key in sorted(FAMILY_BY_KEY):
        estimator = build_estimator(key, delta=context.config.confidence.delta)
        values = np.asarray(
            [
                estimator.estimate(batch_for_strategy(context, strategy, member, arm), clipping).value
                for strategy in context.strategies
            ],
            dtype=np.float64,
        )
        column[key] = within_ten_percent_share(values, truth)
    return column


def bakeoff_summary(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> dict[str, object]:
    rows = run_bakeoff(context, member, clipping, switches)
    accuracy = strategy_selection_accuracy(context, member, clipping, switches)
    column = within_ten_percent_column(context, member, clipping, switches)
    enriched = tuple(
        BakeoffRow(
            key=row.key,
            family=row.family,
            relative_error=row.relative_error,
            effective_sample_size=row.effective_sample_size,
            overlap_coefficient=row.overlap_coefficient,
            selection_accuracy=accuracy[row.key],
            within_ten_percent=column[row.key],
            value=row.value,
            diagnostic=row.diagnostic,
        )
        for row in rows
    )
    best = min(enriched, key=lambda row: row.relative_error)
    families = ("importance_sampling", "doubly_robust", "switching_hybrid", "direct", "kernel_direct")
    worst_family = max(families, key=lambda family: family_instability(enriched, family))
    return {
        "rows": enriched,
        "best_by_relative_error": best.key,
        "most_unstable_family": worst_family,
        "hybrid_instability": family_instability(enriched, "switching_hybrid"),
        "family_instability": {family: family_instability(enriched, family) for family in families},
    }


def _diagnostic(family: str, diagnostics: dict[str, float]) -> str:
    if family in {"importance_sampling", "doubly_robust"}:
        return f"max_weight={diagnostics.get('max_weight', float('nan')):.3f}"
    if family == "switching_hybrid":
        return f"mean_switch={diagnostics.get('mean_switch', float('nan')):.3f}"
    if family == "kernel_direct":
        return f"bandwidth={diagnostics.get('bandwidth', float('nan')):.4f}"
    if family == "high_confidence_bound":
        return f"radius={diagnostics.get('radius', float('nan')):.3f}"
    return "weightless"
