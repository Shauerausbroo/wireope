"""Direct family: fitted Q-evaluation of the graded rung values.

The direct estimator reads the fitted outcome model only. It is stable by construction and
carries no importance weight, which is why the kernel and direct families survive the
overlap regime in which the hybrid family degrades.

Ref: Sec. 2.1, Sec. 3.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wireope.estimators.base import EstimatorResult, LogBatch, weighted_ladder_value


class FittedQEvaluation:
    key = "fqe"
    family = "direct"

    def __init__(self, clipping: float = 0.0) -> None:
        self.clipping = clipping

    def estimate(self, batch: LogBatch, clipping: float = 0.0) -> EstimatorResult:
        predictions = _predictions(batch)
        per_rung = np.mean(predictions, axis=0)
        value = weighted_ladder_value(per_rung, batch.weights)
        return EstimatorResult(
            key=self.key,
            family=self.family,
            value=value,
            per_rung_value=np.asarray(per_rung, dtype=np.float64),
            risk=float(per_rung[-1]),
            effective_sample_size=float(batch.attempts),
            overlap_coefficient=batch.overlap_coefficient(),
            diagnostics={"weightless": 1.0},
        )


def _predictions(batch: LogBatch) -> NDArray[np.float64]:
    if batch.outcome_predictions is None:
        raise ValueError("the direct family needs rung-wise outcome predictions on the batch")
    return np.asarray(batch.outcome_predictions, dtype=np.float64)


def direct_rung_risk(batch: LogBatch) -> float:
    predictions = _predictions(batch)
    return float(np.mean(predictions[:, -1]))


def direct_graded_value(batch: LogBatch) -> float:
    return weighted_ladder_value(np.mean(_predictions(batch), axis=0), batch.weights)
