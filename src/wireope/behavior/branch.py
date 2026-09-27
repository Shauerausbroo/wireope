"""Behaviour-policy reconstruction, branch stage.

The behaviour policy is a two-part policy pi_b = ( pi_b(branch | s), pi_b(a | branch, s) ).
The branch stage selects the crossing branch the operator worked in; it is estimated from
the log rather than observed, which is why its fidelity is reported as an outcome of the
work and not as a data-collection step.

Ref: Sec. 3.1 and Sec. 4.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from wireope.config import PropensityConfig
from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class BranchDesign:
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]


class BranchModel:
    """Multinomial logistic model over the crossing branches."""

    def __init__(self, config: PropensityConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self._model: LogisticRegression | None = None
        self._classes: NDArray[np.str_] = np.zeros(0, dtype=np.str_)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> NDArray[np.str_]:
        return self._classes

    def fit(self, features: NDArray[np.float64], branches: NDArray[np.str_]) -> BranchModel:
        self._classes = np.unique(branches)
        if self._classes.size < 2:
            raise ValueError("branch model needs at least two observed branches")
        model = LogisticRegression(
            C=1.0 / max(self.config.l2, EPS),
            max_iter=self.config.max_iter,
            solver="lbfgs",
            random_state=self.seed,
        )
        model.fit(features, branches)
        self._model = model
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("branch model has not been fitted")
        probability: NDArray[np.float64] = np.asarray(self._model.predict_proba(features), dtype=np.float64)
        return clip_probability(probability)

    def probability_of(
        self, features: NDArray[np.float64], branches: NDArray[np.str_]
    ) -> NDArray[np.float64]:
        probability = self.predict_proba(features)
        index = np.searchsorted(self._classes, branches)
        valid = (index >= 0) & (index < self._classes.size)
        if not np.all(self._classes[np.clip(index, 0, self._classes.size - 1)] == branches):
            raise ValueError("requested branch was not seen during fitting")
        rows = np.arange(probability.shape[0])
        selected: NDArray[np.float64] = probability[rows, np.clip(index, 0, self._classes.size - 1)]
        _ = valid
        return selected

    def log_probability(
        self, features: NDArray[np.float64], branches: NDArray[np.str_]
    ) -> NDArray[np.float64]:
        return np.log(self.probability_of(features, branches))

    def design(self) -> BranchDesign:
        return BranchDesign(
            feature_names=self.config.branch_features, classes=tuple(self._classes.tolist())
        )


def clip_propensity(probability: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Clip a reconstructed propensity from below at the weight-control threshold."""
    if threshold <= 0.0:
        raise ValueError("clipping threshold must be positive")
    clipped = np.maximum(np.asarray(probability, dtype=np.float64), threshold)
    return np.asarray(np.minimum(clipped, 1.0), dtype=np.float64)


def behaviour_agreement(predicted: NDArray[np.str_], observed: NDArray[np.str_]) -> float:
    if observed.size == 0:
        return 0.0
    return float(np.mean(predicted == observed))


def top1_branch(model: BranchModel, features: NDArray[np.float64]) -> NDArray[np.str_]:
    probability = model.predict_proba(features)
    index = np.argmax(probability, axis=1)
    return np.asarray(model.classes[index], dtype=np.str_)
