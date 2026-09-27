"""Registry of the candidate estimators and the bake-off that compares them.

Every candidate is run on identical logs, splits, budgets and preprocessing; the diagnostic
that selected an estimator is reported with its estimate, because the estimator's position
reflects the interaction regime and not its own quality.

Ref: Sec. 3.4 and Sec. 4.2, Table 2 panel B.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from wireope.bounds.effective_sample import relative_error
from wireope.config import EstimatorLibraryConfig
from wireope.estimators.base import EstimatorResult, LogBatch, OffPolicyEstimator
from wireope.estimators.direct import FittedQEvaluation
from wireope.estimators.doubly_robust import DoublyRobust, SwitchingHybrid, WeightedDoublyRobust
from wireope.estimators.high_confidence import BinaryHighConfidence
from wireope.estimators.importance import (
    ImportanceSampling,
    PerDecisionImportanceSampling,
    PerDecisionWeightedImportanceSampling,
    WeightedImportanceSampling,
)
from wireope.estimators.kernel import InfiniteHorizonKernel
from wireope.estimators.nested_dr import NestedWeightedDoublyRobust

FAMILY_BY_KEY: dict[str, str] = {
    "is": "importance_sampling",
    "pdis": "importance_sampling",
    "pdis_w": "importance_sampling",
    "wis": "importance_sampling",
    "dr": "doubly_robust",
    "wdr": "doubly_robust",
    "magic": "switching_hybrid",
    "fqe": "direct",
    "kernel_ih": "kernel_direct",
    "binary_hcb": "high_confidence_bound",
    "nested_dr": "doubly_robust",
}


def build_estimator(key: str, delta: float = 0.05, bounded_range: float = 1.0) -> OffPolicyEstimator:
    if key == "is":
        return ImportanceSampling()
    if key == "pdis":
        return PerDecisionImportanceSampling()
    if key == "pdis_w":
        return PerDecisionWeightedImportanceSampling()
    if key == "wis":
        return WeightedImportanceSampling()
    if key == "dr":
        return DoublyRobust()
    if key == "wdr":
        return WeightedDoublyRobust()
    if key == "magic":
        return SwitchingHybrid()
    if key == "fqe":
        return FittedQEvaluation()
    if key == "kernel_ih":
        return InfiniteHorizonKernel()
    if key == "binary_hcb":
        return BinaryHighConfidence(delta=delta, bounded_range=bounded_range)
    if key == "nested_dr":
        return NestedWeightedDoublyRobust()
    raise ValueError(f"unknown estimator key '{key}'")


def build_library(library: EstimatorLibraryConfig, delta: float = 0.05) -> list[OffPolicyEstimator]:
    keys = [candidate.key for candidate in library.candidates]
    keys.extend(key for key in (library.comparator.key, "nested_dr") if key not in keys)
    return [build_estimator(key, delta=delta) for key in keys]


class RankedRow(Protocol):
    """The attributes any bake-off row must expose to be ranked."""

    @property
    def key(self) -> str: ...

    @property
    def family(self) -> str: ...

    @property
    def relative_error(self) -> float: ...

    @property
    def within_ten_percent(self) -> float: ...


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


def run_bakeoff(
    batch: LogBatch,
    truth_graded: float,
    clipping: float,
    delta: float,
    within_ten_percent: dict[str, float],
) -> tuple[BakeoffRow, ...]:
    """Estimate the graded value with every candidate on one batch."""
    rows: list[BakeoffRow] = []
    for key in sorted(FAMILY_BY_KEY):
        estimator = build_estimator(key, delta=delta)
        result: EstimatorResult = estimator.estimate(batch, clipping)
        rows.append(
            BakeoffRow(
                key=key,
                family=result.family,
                relative_error=relative_error(result.value, truth_graded),
                effective_sample_size=result.effective_sample_size,
                overlap_coefficient=result.overlap_coefficient,
                selection_accuracy=float("nan"),
                within_ten_percent=within_ten_percent.get(key, float("nan")),
                value=result.value,
                diagnostic=_diagnostic(result),
            )
        )
    return tuple(rows)


def _diagnostic(result: EstimatorResult) -> str:
    if result.family == "importance_sampling":
        return f"max_weight={result.diagnostics.get('max_weight', float('nan')):.3f}"
    if result.family == "doubly_robust":
        return f"max_weight={result.diagnostics.get('max_weight', float('nan')):.3f}"
    if result.family == "switching_hybrid":
        return f"mean_switch={result.diagnostics.get('mean_switch', float('nan')):.3f}"
    if result.family == "kernel_direct":
        return f"bandwidth={result.diagnostics.get('bandwidth', float('nan')):.3f}"
    if result.family == "high_confidence_bound":
        return f"radius={result.diagnostics.get('radius', float('nan')):.3f}"
    return "weightless"


def stable_family_ranking(rows: Sequence[RankedRow]) -> list[str]:
    """Rank the families by relative error, then by the published comparability column."""
    ordered = sorted(
        rows,
        key=lambda row: (row.relative_error, -_finite(row.within_ten_percent)),
    )
    return [row.key for row in ordered]


def family_instability(rows: Sequence[RankedRow], family: str) -> float:
    """Spread of relative error inside a family: the hybrid family is expected to be worst."""
    members = [row.relative_error for row in rows if row.family == family]
    if len(members) < 2:
        return 0.0
    return float(np.max(members) - np.min(members))


def selection_accuracy_from_ranking(
    estimated: NDArray[np.float64],
    truth: NDArray[np.float64],
) -> float:
    """Share of decisions on which the estimator's argmax matches the true argmax."""
    if estimated.size == 0 or truth.size == 0:
        return 0.0
    return float(np.mean(np.argmax(estimated) == np.argmax(truth)))


def within_ten_percent_share(
    estimated: NDArray[np.float64],
    truth: NDArray[np.float64],
    tolerance: float = 0.10,
) -> float:
    """Published aggregate near-top-frequency metric, reported for comparability only."""
    if truth.size == 0:
        return 0.0
    denominator = np.maximum(np.abs(truth), 1e-12)
    relative = np.abs(estimated - truth) / denominator
    return float(np.mean(relative <= tolerance))


def _finite(value: float) -> float:
    return value if np.isfinite(value) else -1.0
