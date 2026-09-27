"""Behaviour-policy reconstruction, action stage.

The action stage selects the strategy class the operator delivered, conditioned on the
branch. Because the robotic systems do not log force, torque, progression or rotation as
coded signals, the behaviour model is estimated and its residual is a reported property of
the estimates rather than a nuisance.

Ref: Sec. 3.1, Sec. 4.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from wireope.config import PropensityConfig
from wireope.utils.linalg import EPS, clip_probability


@dataclass(frozen=True)
class ActionDesign:
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]


class ActionModel:
    """Conditional multinomial logistic model over strategies, given the branch."""

    def __init__(self, config: PropensityConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self._model: LogisticRegression | None = None
        self._classes: NDArray[np.str_] = np.zeros(0, dtype=np.str_)
        self._branch_classes: NDArray[np.str_] = np.zeros(0, dtype=np.str_)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> NDArray[np.str_]:
        return self._classes

    @property
    def branch_classes(self) -> NDArray[np.str_]:
        return self._branch_classes

    def fit(
        self,
        features: NDArray[np.float64],
        branches: NDArray[np.str_],
        strategies: NDArray[np.str_],
    ) -> ActionModel:
        self._classes = np.unique(strategies)
        self._branch_classes = np.unique(branches)
        if self._classes.size < 2:
            raise ValueError("action model needs at least two observed strategies")
        design = np.hstack([features, branch_one_hot(branches, self._branch_classes)])
        model = LogisticRegression(
            C=1.0 / max(self.config.l2, EPS),
            max_iter=self.config.max_iter,
            solver="lbfgs",
            random_state=self.seed,
        )
        model.fit(design, strategies)
        self._model = model
        return self

    def predict_proba(
        self, features: NDArray[np.float64], branches: NDArray[np.str_]
    ) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("action model has not been fitted")
        design = np.hstack([features, branch_one_hot(branches, self._branch_classes)])
        probability: NDArray[np.float64] = np.asarray(self._model.predict_proba(design), dtype=np.float64)
        return clip_probability(probability)

    def probability_of(
        self,
        features: NDArray[np.float64],
        branches: NDArray[np.str_],
        strategies: NDArray[np.str_],
    ) -> NDArray[np.float64]:
        probability = self.predict_proba(features, branches)
        index = _index_of(self._classes, strategies)
        rows = np.arange(probability.shape[0])
        selected: NDArray[np.float64] = probability[rows, index]
        return selected

    def log_probability(
        self,
        features: NDArray[np.float64],
        branches: NDArray[np.str_],
        strategies: NDArray[np.str_],
    ) -> NDArray[np.float64]:
        return np.log(self.probability_of(features, branches, strategies))

    def target_probability(
        self,
        features: NDArray[np.float64],
        branches: NDArray[np.str_],
        target: str,
    ) -> NDArray[np.float64]:
        """pi_b(a = target | branch, s) for a single target strategy, used by Eq. (4)."""
        strategies = np.full(features.shape[0], target, dtype="<U64")
        return self.probability_of(features, branches, strategies)

    def design(self) -> ActionDesign:
        return ActionDesign(
            feature_names=self.config.action_features, classes=tuple(self._classes.tolist())
        )


def branch_one_hot(branches: NDArray[np.str_], classes: NDArray[np.str_]) -> NDArray[np.float64]:
    index = np.searchsorted(classes, branches)
    one_hot = np.zeros((branches.shape[0], classes.size), dtype=np.float64)
    one_hot[np.arange(branches.shape[0]), index] = 1.0
    return one_hot


def _index_of(classes: NDArray[np.str_], values: NDArray[np.str_]) -> NDArray[np.int64]:
    index = np.searchsorted(classes, values)
    clamped = np.clip(index, 0, classes.size - 1)
    if not np.all(classes[clamped] == values):
        raise ValueError("requested strategy was not seen during fitting")
    return np.asarray(index, dtype=np.int64)


def one_hot_descriptors(stages: NDArray[np.str_], grades: NDArray[np.int64]) -> NDArray[np.float64]:
    """Small design matrix over the anatomical axes the propensity model conditions on."""
    stages_ordered = np.asarray(["I", "II", "III"], dtype="<U8")
    stage_block = branch_one_hot(stages, stages_ordered)
    grade_block = np.zeros((grades.shape[0], 5), dtype=np.float64)
    clipped = np.clip(grades, 0, 4)
    grade_block[np.arange(grades.shape[0]), clipped] = 1.0
    return np.hstack([stage_block, grade_block])
