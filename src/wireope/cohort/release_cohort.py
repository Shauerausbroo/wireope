"""The cohort this release assembles, with a closed-form injury truth.

The primary cohort is held under data-sharing agreements and is never bundled, so the
mechanisms are executed on a cohort the release assembles that follows the same data
dictionary. It
is built so that every quantity the estimators target has a closed form: the injury
probability of an attempt is a logistic function of its anatomy and its delivered action,
and the rung indicators follow from a binomial level model whose tail is analytic. That
makes the estimator layer falsifiable without a record of any patient.

The top rung is the observed injury event by construction, so Y^(K) = Y and the rung
indicators nest downward, as the ladder of Sec. 3.4 requires.

Ref: Sec. 3.1, Sec. 3.4, Sec. 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from wireope.config import ReleaseCohortConfig
from wireope.laddering.severity import Ladder
from wireope.schema import AnatomyRecord, AttemptFeatures, DeviceRecord, EventRecord
from wireope.traces.excursion import rung_indicators
from wireope.utils.linalg import EPS, sigmoid
from wireope.utils.seed import numpy_generator

BRANCHES: tuple[str, ...] = ("antegrade", "retrograde", "subintimal")
STAGE_ORDER: tuple[str, ...] = ("I", "II", "III")
TERRITORY_ORDER: tuple[str, ...] = ("femoropopliteal", "tibial", "other")
SITE_ORDER: tuple[str, ...] = ("Site_A", "Site_B", "Site_C")
SITE_VENDORS: dict[str, tuple[str, ...]] = {
    "Site_A": ("vendor_a", "vendor_b"),
    "Site_B": ("vendor_c", "vendor_d"),
    "Site_C": ("vendor_e", "vendor_f"),
}
DEVICE_CLASSES: tuple[str, ...] = ("polymer_jacketed", "hydrophilic", "stiff_support", "microcatheter")
CAP_CLASSES: tuple[str, ...] = ("calcified_nodule", "ambiguous", "tapered", "blunt")


@dataclass(frozen=True)
class Strategy:
    name: str
    branch: str
    aggressiveness: float


@dataclass(frozen=True)
class AttemptState:
    attempt_id: int
    procedure_id: int
    lesion_id: int
    patient_id: int
    site: str
    region: str
    operator: str
    vendor_class: str
    stage: str
    calcification_grade: int
    calcification_arc_deg: float
    cap_morphology: str
    lesion_length_mm: float
    territory: str
    lesion_overlap_fraction: float
    delivered_branch: str
    delivered_strategy: str
    behaviour_branch_probability: float
    behaviour_action_probability: float


@dataclass(frozen=True)
class ReleaseCohort:
    states: tuple[AttemptState, ...]
    features: tuple[AttemptFeatures, ...]
    events: tuple[tuple[EventRecord, ...], ...]
    sites: tuple[str, ...]
    calibration_intercepts: dict[str, float]

    @property
    def attempts(self) -> int:
        return len(self.states)

    def injury_flags(self) -> NDArray[np.int64]:
        return np.asarray([item.injury for item in self.features], dtype=np.int64)

    def peaks(self) -> NDArray[np.float64]:
        return np.asarray([item.max_excursion_mm for item in self.features], dtype=np.float64)

    def strategies(self) -> NDArray[np.str_]:
        return np.asarray([item.strategy for item in self.features], dtype="<U64")

    def site_array(self) -> NDArray[np.str_]:
        return np.asarray([item.site for item in self.states], dtype="<U64")

    def region_array(self) -> NDArray[np.str_]:
        return np.asarray([item.region for item in self.states], dtype="<U64")

    def operator_array(self) -> NDArray[np.str_]:
        return np.asarray([item.operator for item in self.states], dtype="<U64")

    def vendor_array(self) -> NDArray[np.str_]:
        return np.asarray([item.vendor_class for item in self.states], dtype="<U64")

    def stage_array(self) -> NDArray[np.str_]:
        return np.asarray([item.stage for item in self.states], dtype="<U64")

    def territory_array(self) -> NDArray[np.str_]:
        return np.asarray([item.territory for item in self.states], dtype="<U64")

    def grade_array(self) -> NDArray[np.int64]:
        return np.asarray([item.calcification_grade for item in self.states], dtype=np.int64)

    def lesion_overlap_array(self) -> NDArray[np.float64]:
        return np.asarray([item.lesion_overlap_fraction for item in self.states], dtype=np.float64)

    def action_array(self) -> NDArray[np.float64]:
        return np.asarray([item.action_norm for item in self.features], dtype=np.float64)

    def dwell_integral_array(self) -> NDArray[np.float64]:
        return np.asarray([item.dwell_integral_mm_s for item in self.features], dtype=np.float64)

    def branch_array(self) -> NDArray[np.str_]:
        return np.asarray([item.delivered_branch for item in self.states], dtype="<U64")

    def descriptor_matrix(self) -> NDArray[np.float64]:
        return np.vstack([item.descriptor_vector for item in self.features])

    def rung_matrix(self, ladder: Ladder) -> NDArray[np.float64]:
        return ladder.indicator_matrix(self.peaks())


def build_strategy_library(size: int) -> tuple[Strategy, ...]:
    if size < 2:
        raise ValueError("the strategy library needs at least two strategies")
    names = (
        "intimal_tracking_gentle",
        "intimal_tracking_standard",
        "intimal_tracking_firm",
        "subintimal_knuckle",
        "subintimal_reentry",
        "dual_lumen_exchange",
        "balloon_assisted_crossing",
        "wire_escalation_reentry",
    )
    chosen = names[:size]
    strategies: list[Strategy] = []
    for position, name in enumerate(chosen):
        aggressiveness = position / (len(chosen) - 1)
        branch = BRANCHES[min(position * len(BRANCHES) // len(chosen), len(BRANCHES) - 1)]
        strategies.append(Strategy(name=name, branch=branch, aggressiveness=float(aggressiveness)))
    return tuple(strategies)


def aggressiveness_of(strategies: tuple[Strategy, ...], name: str) -> float:
    for strategy in strategies:
        if strategy.name == name:
            return strategy.aggressiveness
    raise KeyError(f"unknown strategy '{name}'")


def branch_scores(stage_index: int) -> NDArray[np.float64]:
    return np.asarray(
        [1.2 - 0.5 * stage_index, 0.2 + 0.15 * stage_index, -0.4 + 0.6 * stage_index],
        dtype=np.float64,
    )


def preferred_aggressiveness(stage_index: int, grade: int) -> float:
    return float(np.clip(0.25 + 0.20 * stage_index + 0.05 * grade, 0.0, 1.0))


def action_scores(
    stage_index: int,
    grade: int,
    delivered_branch: str,
    strategies: tuple[Strategy, ...],
) -> NDArray[np.float64]:
    target = preferred_aggressiveness(stage_index, grade)
    scores = np.zeros(len(strategies), dtype=np.float64)
    for position, strategy in enumerate(strategies):
        affinity = 0.8 if strategy.branch == delivered_branch else 0.0
        scores[position] = -((strategy.aggressiveness - target) ** 2) + affinity
    return scores


def linear_predictor(
    coefficients: dict[str, float],
    stage_index: int,
    grade: int,
    overlap_fraction: float,
    aggressiveness: float,
) -> float:
    return float(
        coefficients["beta_stage"] * stage_index
        + coefficients["beta_calcification"] * grade
        + coefficients["beta_overlap"] * overlap_fraction
        + coefficients["beta_action"] * aggressiveness
    )


def injury_probability(intercept: float, linear: float) -> float:
    return float(sigmoid(intercept + linear))


def rung_probability(injury_probability_value: float, level_q: float, rungs: int, rung: int) -> float:
    """P(Y^(k) = 1) = p + (1 - p) P(Binomial(K-1, q) >= k), exact for the level model."""
    if rung < 1 or rung > rungs:
        raise ValueError("rung index out of range")
    tail = float(stats.binom.sf(rung - 1, rungs - 1, level_q))
    return float(injury_probability_value + (1.0 - injury_probability_value) * tail)


def level_parameter(config: ReleaseCohortConfig, aggressiveness: float) -> float:
    return float(np.clip(config.rung_base_q + config.rung_action_slope * aggressiveness, 0.0, 1.0))


def calibrate_site_intercept(
    linear_at_zero: NDArray[np.float64],
    target_events: int,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Bisect the site intercept so the expected event count meets the accrual target."""
    if linear_at_zero.shape[0] == 0:
        return 0.0
    low, high = -14.0, 8.0
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        expected = float(np.sum(sigmoid(linear_at_zero + middle)))
        if expected < target_events:
            low = middle
        else:
            high = middle
        if abs(high - low) <= tolerance:
            break
    return float(0.5 * (low + high))


def closed_form_curves(
    config: ReleaseCohortConfig,
    ladder: Ladder,
    strategies: tuple[Strategy, ...],
    states: tuple[AttemptState, ...],
    intercepts: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Analytic graded value and endpoint risk per strategy, averaged over the state pool.

    This reference uses the generating probabilities directly and never touches a rung
    observation, so it is independent of every estimator in the release.
    """
    coefficients = dict(config.coefficients)
    weights = np.asarray(ladder.weights, dtype=np.float64)
    curves: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        rung_curve = np.zeros((len(states), ladder.depth), dtype=np.float64)
        probability = np.zeros(len(states), dtype=np.float64)
        for position, state in enumerate(states):
            linear = linear_predictor(
                coefficients=coefficients,
                stage_index=STAGE_ORDER.index(state.stage),
                grade=state.calcification_grade,
                overlap_fraction=state.lesion_overlap_fraction,
                aggressiveness=strategy.aggressiveness,
            )
            p = injury_probability(intercepts[state.site], linear)
            probability[position] = p
            q = level_parameter(config, strategy.aggressiveness)
            for rung in range(1, ladder.depth + 1):
                rung_curve[position, rung - 1] = rung_probability(p, q, ladder.depth, rung)
        severity = np.mean(rung_curve @ weights)
        curves[strategy.name] = {
            "severity": float(severity),
            "graded_value": float(severity),
            "graded_progress": float(np.sum(weights) - severity),
            "risk": float(np.mean(rung_curve[:, -1])),
            "mean_rung_prevalence": float(np.mean(rung_curve[:, 0])),
            "mean_injury_probability": float(np.mean(probability)),
        }
    return curves


def closed_form_state_probabilities(
    config: ReleaseCohortConfig,
    ladder: Ladder,
    strategies: tuple[Strategy, ...],
    states: tuple[AttemptState, ...],
    intercepts: dict[str, float],
) -> NDArray[np.float64]:
    """Per-state injury probability under each strategy, the reference for per-attempt checks."""
    coefficients = dict(config.coefficients)
    result = np.zeros((len(states), len(strategies)), dtype=np.float64)
    for column, strategy in enumerate(strategies):
        for row, state in enumerate(states):
            linear = linear_predictor(
                coefficients=coefficients,
                stage_index=STAGE_ORDER.index(state.stage),
                grade=state.calcification_grade,
                overlap_fraction=state.lesion_overlap_fraction,
                aggressiveness=strategy.aggressiveness,
            )
            result[row, column] = injury_probability(intercepts[state.site], linear)
    return result


def assemble_release_cohort(
    config: ReleaseCohortConfig, ladder: Ladder, seed: int | None = None
) -> ReleaseCohort:
    """Assemble the cohort log: anatomy, delivered action, excursion and adjudicated injury."""
    rng = numpy_generator(config.seed if seed is None else seed)
    strategies = build_strategy_library(int(config.behaviour["library_size"]))
    coefficients = dict(config.coefficients)
    stage_levels = np.asarray(list(config.descriptor_mix["stage"]), dtype="<U8")
    stage_shares = _normalised(list(config.descriptor_mix["stage"].values()))
    grade_levels = np.asarray(
        [int(key.split("-")[0]) for key in config.descriptor_mix["calcification_grade"]],
        dtype=np.int64,
    )
    grade_shares = _normalised(list(config.descriptor_mix["calcification_grade"].values()))
    territory_levels = np.asarray(list(config.descriptor_mix["vessel_territory"]), dtype="<U8")
    territory_shares = _normalised(list(config.descriptor_mix["vessel_territory"].values()))

    states: list[AttemptState] = []
    features: list[AttemptFeatures] = []
    event_logs: list[tuple[EventRecord, ...]] = []
    intercepts: dict[str, float] = {}
    offset = 0
    for site_spec in config.site_targets:
        site = site_spec.site
        count = site_spec.attempts
        shift = config.site_shift.get(site, {})
        stage_draw = rng.choice(
            stage_levels, size=count, p=_tilt(stage_shares, shift.get("stage_shift", 0.0))
        )
        grade_draw = rng.choice(
            grade_levels, size=count, p=_tilt(grade_shares, shift.get("calcification_shift", 0.0))
        )
        territory_draw = rng.choice(territory_levels, size=count, p=territory_shares)
        operator_draw = rng.integers(0, int(config.behaviour["operator_count"]), size=count)
        vendor_draw = rng.choice(np.asarray(SITE_VENDORS.get(site, ("vendor_a",)), dtype="<U8"), size=count)
        overlap_draw = rng.uniform(
            config.lesion_overlap_range[0], config.lesion_overlap_range[1], size=count
        )
        cap_draw = rng.choice(np.asarray(CAP_CLASSES, dtype="<U8"), size=count)
        arc_draw = rng.uniform(30.0, 300.0, size=count)
        length_draw = rng.uniform(8.0, 55.0, size=count)
        stage_index = np.asarray([STAGE_ORDER.index(str(item)) for item in stage_draw], dtype=np.int64)

        local_states: list[AttemptState] = []
        linear_zero = np.zeros(count, dtype=np.float64)
        strategy_positions = np.zeros(count, dtype=np.int64)
        branch_positions = np.zeros(count, dtype=np.int64)
        branch_probabilities = np.zeros(count, dtype=np.float64)
        action_probabilities = np.zeros(count, dtype=np.float64)
        for position in range(count):
            branch_probability = _softmax(
                branch_scores(int(stage_index[position])) / max(config.behaviour["branch_temperature"], EPS)
            )
            branch_position = int(rng.choice(len(BRANCHES), p=branch_probability))
            scores = action_scores(
                int(stage_index[position]),
                int(grade_draw[position]),
                BRANCHES[branch_position],
                strategies,
            )
            action_probability = _softmax(scores / max(config.behaviour["action_temperature"], EPS))
            strategy_position = int(rng.choice(len(strategies), p=action_probability))
            branch_positions[position] = branch_position
            strategy_positions[position] = strategy_position
            branch_probabilities[position] = branch_probability[branch_position]
            action_probabilities[position] = action_probability[strategy_position]
            linear_zero[position] = linear_predictor(
                coefficients=coefficients,
                stage_index=int(stage_index[position]),
                grade=int(grade_draw[position]),
                overlap_fraction=float(overlap_draw[position]),
                aggressiveness=strategies[strategy_position].aggressiveness,
            )

        intercept = calibrate_site_intercept(
            linear_at_zero=linear_zero,
            target_events=site_spec.injury_events,
            tolerance=config.calibration.tolerance,
            max_iterations=config.calibration.max_iterations,
        )
        intercepts[site] = intercept
        probabilities = sigmoid(linear_zero + intercept)
        latent = probabilities + rng.normal(0.0, config.calibration.latent_noise_scale, size=count)
        order = np.argsort(-latent)
        injury = np.zeros(count, dtype=np.int64)
        injury[order[: site_spec.injury_events]] = 1

        for position in range(count):
            strategy = strategies[int(strategy_positions[position])]
            local_states.append(
                AttemptState(
                    attempt_id=offset + position,
                    procedure_id=(offset + position) // 2,
                    lesion_id=(offset + position) // 3,
                    patient_id=(offset + position) // 4,
                    site=site,
                    region=site_spec.region,
                    operator=f"{site}_op_{int(operator_draw[position]):02d}",
                    vendor_class=str(vendor_draw[position]),
                    stage=str(stage_draw[position]),
                    calcification_grade=int(grade_draw[position]),
                    calcification_arc_deg=float(arc_draw[position]),
                    cap_morphology=str(cap_draw[position]),
                    lesion_length_mm=float(length_draw[position]),
                    territory=str(territory_draw[position]),
                    lesion_overlap_fraction=float(overlap_draw[position]),
                    delivered_branch=BRANCHES[int(branch_positions[position])],
                    delivered_strategy=strategy.name,
                    behaviour_branch_probability=float(branch_probabilities[position]),
                    behaviour_action_probability=float(action_probabilities[position]),
                )
            )
        for position, state in enumerate(local_states):
            label = int(injury[position])
            event_logs.append(build_event_stream(state=state, strategies=strategies, rng=rng, injury=label))
            peak = sample_peak_excursion(
                rng=rng,
                ladder=ladder,
                level_q=level_parameter(config, aggressiveness_of(strategies, state.delivered_strategy)),
                injury=label,
            )
            features.append(
                AttemptFeatures(
                    attempt_id=state.attempt_id,
                    max_excursion_mm=float(peak),
                    dwell_integral_mm_s=float(peak * rng.uniform(0.5, 4.0)),
                    action_norm=float(aggressiveness_of(strategies, state.delivered_strategy)),
                    rung_indicators=rung_indicators(peak, ladder.tau_mm),
                    injury=label,
                    strategy=state.delivered_strategy,
                    site=state.site,
                    region=state.region,
                    operator=state.operator,
                    vendor_class=state.vendor_class,
                    stage=state.stage,
                    calcification_grade=state.calcification_grade,
                    territory=state.territory,
                    descriptor_vector=descriptor_vector(state),
                )
            )
        states.extend(local_states)
        offset += count
    return ReleaseCohort(
        states=tuple(states),
        features=tuple(features),
        events=tuple(event_logs),
        sites=tuple(spec.site for spec in config.site_targets),
        calibration_intercepts=intercepts,
    )


def descriptor_vector(state: AttemptState) -> NDArray[np.float64]:
    """Design vector over the anatomical axes the propensity and outcome models consume."""
    stage_one_hot = np.zeros(3, dtype=np.float64)
    stage_one_hot[STAGE_ORDER.index(state.stage)] = 1.0
    grade_one_hot = np.zeros(5, dtype=np.float64)
    grade_one_hot[int(np.clip(state.calcification_grade, 0, 4))] = 1.0
    territory_one_hot = np.zeros(3, dtype=np.float64)
    territory_index = TERRITORY_ORDER.index(state.territory) if state.territory in TERRITORY_ORDER else 2
    territory_one_hot[territory_index] = 1.0
    site_one_hot = np.zeros(3, dtype=np.float64)
    site_one_hot[SITE_ORDER.index(state.site) if state.site in SITE_ORDER else 0] = 1.0
    return np.concatenate(
        [
            stage_one_hot,
            grade_one_hot,
            territory_one_hot,
            site_one_hot,
            np.asarray(
                [state.lesion_length_mm / 60.0, state.calcification_arc_deg / 360.0], dtype=np.float64
            ),
        ]
    )


def build_event_stream(
    state: AttemptState,
    strategies: tuple[Strategy, ...],
    rng: np.random.Generator,
    injury: int,
) -> tuple[EventRecord, ...]:
    """Device-advancement and rotation stream whose length follows the lesion and the action."""
    aggressiveness = aggressiveness_of(strategies, state.delivered_strategy)
    base_steps = 6.0 + state.lesion_length_mm / 3.0
    steps = int(np.clip(round(base_steps) + int(rng.integers(-3, 4)), 4, 90))
    events: list[EventRecord] = []
    for step in range(steps):
        draw = float(rng.random())
        if draw < 0.12 + 0.25 * aggressiveness:
            events.append(
                EventRecord(step=step, kind="rotate_counterclockwise", magnitude=float(rng.integers(1, 3)))
            )
        elif draw < 0.22 + 0.25 * aggressiveness:
            events.append(EventRecord(step=step, kind="dwell", magnitude=float(rng.integers(1, 4))))
        elif draw < 0.27 + 0.25 * aggressiveness:
            events.append(EventRecord(step=step, kind="retract", magnitude=1.0))
        else:
            events.append(EventRecord(step=step, kind="advance", magnitude=float(rng.integers(1, 4))))
    if injury == 1:
        events.append(EventRecord(step=len(events), kind="advance", magnitude=2.0))
    return tuple(events)


def sample_peak_excursion(
    rng: np.random.Generator,
    ladder: Ladder,
    level_q: float,
    injury: int,
) -> float:
    """Peak excursion consistent with the level model, so the top rung is the injury event."""
    if injury == 1:
        return float(ladder.tau_inj_mm * (1.0 + 0.15 * rng.random()))
    level = int(stats.binom.rvs(ladder.depth - 1, level_q, random_state=int(rng.integers(0, 2**31 - 1))))
    level = int(np.clip(level, 0, ladder.depth - 1))
    low = 0.0 if level == 0 else ladder.tau_mm[level - 1]
    high = min(
        ladder.tau_mm[level] if level < ladder.depth else ladder.tau_inj_mm, ladder.tau_inj_mm * 0.999
    )
    return float(low + (high - low) * rng.random())


def injury_from_peak(peak: float, ladder: Ladder) -> int:
    return int(float(peak) >= ladder.tau_inj_mm)


def observed_action_probability(cohort: ReleaseCohort) -> NDArray[np.float64]:
    """Product of the two reconstructed behaviour stages evaluated at the delivered action."""
    return np.asarray(
        [
            state.behaviour_branch_probability * state.behaviour_action_probability
            for state in cohort.states
        ],
        dtype=np.float64,
    )


def anatomy_record(state: AttemptState) -> AnatomyRecord:
    return AnatomyRecord(
        stage=state.stage,
        calcification_grade=state.calcification_grade,
        calcification_arc_deg=state.calcification_arc_deg,
        cap_morphology=state.cap_morphology,
        lesion_length_mm=state.lesion_length_mm,
        territory=state.territory,
        site=state.site,
        operator=state.operator,
        vendor_class=state.vendor_class,
    )


def device_record(index: int) -> DeviceRecord:
    return DeviceRecord(
        device_class=DEVICE_CLASSES[index % len(DEVICE_CLASSES)],
        attempt_index=int(index % 3) + 1,
        device_to_lesion_relation="wire_in_lesion",
    )


def _normalised(values: list[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(array / float(np.sum(array)), dtype=np.float64)


def _softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - float(np.max(values))
    exponent = np.exp(shifted)
    return np.asarray(exponent / float(np.sum(exponent)), dtype=np.float64)


def _tilt(shares: NDArray[np.float64], shift: float) -> NDArray[np.float64]:
    """Tilt a categorical share vector towards its upper levels by the site shift."""
    if abs(shift) <= EPS:
        return np.asarray(shares / np.sum(shares), dtype=np.float64)
    tilt = np.exp(shift * np.arange(shares.shape[0], dtype=np.float64))
    adjusted = shares * tilt
    return np.asarray(adjusted / np.sum(adjusted), dtype=np.float64)
