"""Rung-wise outcome model entering the direct part of the nested estimator.

The outcome model is fitted per rung on the anatomical identifiers and the delivered action,
which is the pair the state and the strategy supply. It deliberately does not consume the
reconstructed excursion: that quantity is the action observable the rung indicator is built
from, so using it as a predictor would let the direct part read the outcome rather than the
state.

Ref: Sec. 3.1 and Sec. 3.4, Eq. (4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from wireope.cohort.release_cohort import AttemptState
from wireope.config import OutcomeConfig
from wireope.utils.linalg import EPS, clip_probability

STAGE_POSITION: dict[str, int] = {"I": 0, "II": 1, "III": 2}
SITE_POSITION: dict[str, int] = {"Site_A": 0, "Site_B": 1, "Site_C": 2}
TERRITORY_POSITION: dict[str, int] = {"femoropopliteal": 0, "tibial": 1, "other": 2}

ANATOMY_WIDTH = 16


@dataclass(frozen=True)
class OutcomeDesign:
    features: NDArray[np.float64]
    feature_names: tuple[str, ...]


def build_design(
    states: tuple[AttemptState, ...],
    aggressiveness: NDArray[np.float64],
    include_action: bool = True,
) -> OutcomeDesign:
    """Design matrix over the anatomical context and, when present, the action term."""
    width = ANATOMY_WIDTH + (1 if include_action else 0)
    rows = np.zeros((len(states), width), dtype=np.float64)
    for position, state in enumerate(states):
        rows[position, 0] = float(state.calcification_grade)
        rows[position, 1] = state.lesion_length_mm / 60.0
        rows[position, 2] = state.lesion_overlap_fraction
        rows[position, 3] = state.calcification_arc_deg / 360.0
        rows[position, 4 + STAGE_POSITION[state.stage]] = 1.0
        rows[position, 7 + SITE_POSITION.get(state.site, 0)] = 1.0
        rows[position, 10 + TERRITORY_POSITION.get(state.territory, 2)] = 1.0
        rows[position, 13] = 1.0
        rows[position, 14] = float(state.calcification_grade) ** 2 / 16.0
        rows[position, 15] = state.lesion_overlap_fraction * float(state.calcification_grade) / 4.0
        if include_action:
            rows[position, ANATOMY_WIDTH] = aggressiveness[position]
    names: tuple[str, ...] = (
        "calcification_grade",
        "length_fraction",
        "lesion_overlap_fraction",
        "calcification_arc_fraction",
        "stage_I",
        "stage_II",
        "stage_III",
        "site_A",
        "site_B",
        "site_C",
        "territory_femoropopliteal",
        "territory_tibial",
        "territory_other",
        "intercept",
        "grade_squared",
        "overlap_times_grade",
    )
    if include_action:
        names = (*names, "aggressiveness")
    return OutcomeDesign(features=rows, feature_names=names)


class RungOutcomeModel:
    """One logistic model per rung, sharing a design matrix and a regularisation strength."""

    def __init__(self, config: OutcomeConfig, rungs: int, seed: int = 0) -> None:
        self.config = config
        self.rungs = rungs
        self.seed = seed
        self.models: list[object] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.models)

    def fit(self, design: NDArray[np.float64], indicators: NDArray[np.float64]) -> RungOutcomeModel:
        if indicators.shape[1] != self.rungs:
            raise ValueError("one indicator column per rung is required")
        models: list[object] = []
        for rung in range(self.rungs):
            target = indicators[:, rung]
            if np.all(target == target[0]):
                model: object = _ConstantModel(float(target[0]))
            else:
                learner = LogisticRegression(
                    C=1.0 / max(self.config.l2, EPS),
                    max_iter=self.config.max_iter,
                    solver="lbfgs",
                    random_state=self.seed,
                )
                learner.fit(design, target.astype(np.int64))
                model = learner
            models.append(model)
        self.models = models
        return self

    def predict(self, design: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("outcome model has not been fitted")
        columns = [clip_probability(_probability(self.models[rung], design)) for rung in range(self.rungs)]
        return np.asarray(np.column_stack(columns), dtype=np.float64)

    def predict_endpoint(self, design: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = self.predict(design)
        return np.asarray(prediction[:, -1], dtype=np.float64)

    def product_risk(self, design: NDArray[np.float64]) -> NDArray[np.float64]:
        """Product form of Eq. (6) applied to the fitted per-rung conditionals."""
        prediction = self.predict(design)
        return np.asarray(np.prod(prediction, axis=1), dtype=np.float64)


class _ConstantModel:
    """Degenerate fit for a rung no attempt reached, which keeps the rung in the product."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_proba(self, design: NDArray[np.float64]) -> NDArray[np.float64]:
        column = np.zeros((design.shape[0], 2), dtype=np.float64)
        column[:, 0] = 1.0 - self.value
        column[:, 1] = self.value
        return column


def _probability(model: object, design: NDArray[np.float64]) -> NDArray[np.float64]:
    probability = model.predict_proba(design)  # type: ignore[attr-defined]
    array: NDArray[np.float64] = np.asarray(probability, dtype=np.float64)
    if array.ndim == 1:
        return array
    selected: NDArray[np.float64] = array[:, -1]
    return selected


def fit_rung_outcome(
    config: OutcomeConfig,
    design: NDArray[np.float64],
    indicators: NDArray[np.float64],
    seed: int = 0,
) -> RungOutcomeModel:
    return RungOutcomeModel(config, indicators.shape[1], seed).fit(design, indicators)


def outcome_log_loss(prediction: NDArray[np.float64], indicators: NDArray[np.float64]) -> float:
    """Mean binary log loss of the fitted rungs, reported as the direct part's residual."""
    clipped = clip_probability(prediction)
    target = np.asarray(indicators, dtype=np.float64)
    per_rung = -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    return float(np.mean(per_rung))
