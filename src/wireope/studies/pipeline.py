"""Assembly of the analysis: cohort, partitions, fitted behaviour and outcome models, batches.

The cohort is assembled once per experiment; the models are then fitted inside one held-out
configuration, so the development share and the evaluation share never share a patient, an
operator or a vendor class. Every study consumes the context built here, which is what makes
them run on identical logs, splits, budgets and preprocessing.

Ref: Sec. 3.1-3.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wireope.behavior.action import ActionModel
from wireope.behavior.branch import BranchModel
from wireope.bounds.effective_sample import relative_error
from wireope.cohort.partitions import (
    LeakageReport,
    PartitionIndex,
    PartitionSet,
    audit_leakage,
    build_partitions,
)
from wireope.cohort.release_cohort import (
    AttemptState,
    ReleaseCohort,
    Strategy,
    assemble_release_cohort,
    build_strategy_library,
    closed_form_curves,
    closed_form_state_probabilities,
)
from wireope.config import ProjectConfig, load_experiment
from wireope.estimators.base import LogBatch
from wireope.estimators.nested_dr import NestedDRResult, nested_doubly_robust
from wireope.fit.outcome import RungOutcomeModel, build_design, fit_rung_outcome
from wireope.laddering.severity import Ladder, build_ladder, ladder_for_depth

PRIMARY_CONFIGURATION = "primary_external"
CROSS_REGION_CONFIGURATION = "cross_region"
SITEWISE_CONFIGURATION = "sitewise"
HELD_OUT_ROLES: tuple[str, ...] = (
    "primary_held_out_site_and_region",
    "cross_region_holdout",
)


@dataclass(frozen=True)
class ArmSwitches:
    """The three components a Table 2 panel A row removes, one at a time or in pairs."""

    envelope_excursion_reconstruction: bool = True
    nested_excursion_ladder: bool = True
    support_constraint: bool = True

    @property
    def uses_action_term(self) -> bool:
        return self.envelope_excursion_reconstruction


@dataclass(frozen=True)
class CohortBundle:
    """Everything that depends only on the experiment, not on the split."""

    config: ProjectConfig
    ladder: Ladder
    cohort: ReleaseCohort
    strategies: tuple[Strategy, ...]
    curves: dict[str, dict[str, float]]
    partitions: PartitionSet
    leakage: LeakageReport
    descriptors: NDArray[np.float64]
    indicators: NDArray[np.float64]
    state_risk: NDArray[np.float64]


@dataclass(frozen=True)
class AnalysisContext:
    bundle: CohortBundle
    configuration: str
    switches: ArmSwitches
    branch_model: BranchModel
    action_model: ActionModel
    outcome_model: RungOutcomeModel
    state_only_model: RungOutcomeModel
    development_member: NDArray[np.bool_]
    calibration_member: NDArray[np.bool_]
    evaluation_member: NDArray[np.bool_]

    @property
    def config(self) -> ProjectConfig:
        return self.bundle.config

    @property
    def ladder(self) -> Ladder:
        return self.bundle.ladder

    @property
    def cohort(self) -> ReleaseCohort:
        return self.bundle.cohort

    @property
    def strategies(self) -> tuple[Strategy, ...]:
        return self.bundle.strategies

    @property
    def curves(self) -> dict[str, dict[str, float]]:
        return self.bundle.curves

    @property
    def leakage(self) -> LeakageReport:
        return self.bundle.leakage

    def ladder_for(self, switches: ArmSwitches | None = None) -> Ladder:
        return ladder_for_switches(
            self.config, self.bundle.ladder, switches if switches is not None else self.switches
        )

    def states_member(self, member: NDArray[np.bool_]) -> tuple[AttemptState, ...]:
        return tuple(state for state, keep in zip(self.cohort.states, member) if keep)

    def partition(self, name: str) -> PartitionIndex:
        return self.bundle.partitions.by_name(name)


def ladder_for_switches(
    config: ProjectConfig,
    base: Ladder,
    switches: ArmSwitches,
) -> Ladder:
    """The ladder an arm uses: the full nested ladder, or the single endpoint rung."""
    if switches.nested_excursion_ladder:
        return base
    return ladder_for_depth(config.ladder, config.ladder_weights, 1)


def prepare_bundle(repo_root: Path, experiment: str, seed: int | None = None) -> CohortBundle:
    """Assemble the cohort, the ladder, the strategy library and the closed-form curves."""
    config = load_experiment(repo_root, experiment)
    ladder = build_ladder(config.ladder, config.ladder_weights)
    cohort = assemble_release_cohort(config.release_cohort, ladder)
    strategies = build_strategy_library(int(config.release_cohort.behaviour["library_size"]))
    curves = closed_form_curves(
        config.release_cohort, ladder, strategies, cohort.states, cohort.calibration_intercepts
    )
    run_seed = config.release_cohort.seed if seed is None else seed
    patient_ids = np.asarray([state.patient_id for state in cohort.states], dtype=np.int64)
    partitions = build_partitions(
        config.partitions,
        cohort.site_array(),
        cohort.region_array(),
        patient_ids=patient_ids,
        calibration_fraction=0.25,
        seed=run_seed,
    )
    leakage = audit_leakage(
        partitions,
        patient_ids=patient_ids,
        operators=cohort.operator_array(),
        vendor_classes=cohort.vendor_array(),
        guards=config.leakage,
    )
    state_risk = closed_form_state_probabilities(
        config.release_cohort, ladder, strategies, cohort.states, cohort.calibration_intercepts
    )
    return CohortBundle(
        config=config,
        ladder=ladder,
        cohort=cohort,
        strategies=strategies,
        curves=curves,
        partitions=partitions,
        leakage=leakage,
        descriptors=cohort.descriptor_matrix(),
        indicators=ladder.indicator_matrix(cohort.peaks()),
        state_risk=state_risk,
    )


def prepare_context(
    repo_root: Path,
    experiment: str,
    configuration: str = PRIMARY_CONFIGURATION,
    switches: ArmSwitches | None = None,
    seed: int | None = None,
    bundle: CohortBundle | None = None,
) -> AnalysisContext:
    """Load the experiment and fit one held-out configuration."""
    prepared = bundle if bundle is not None else prepare_bundle(repo_root, experiment, seed)
    return fit_context(prepared, configuration, switches if switches is not None else ArmSwitches(), seed)


def fit_context(
    bundle: CohortBundle,
    configuration: str = PRIMARY_CONFIGURATION,
    switches: ArmSwitches | None = None,
    seed: int | None = None,
) -> AnalysisContext:
    """Fit the reconstructed behaviour and outcome models inside one held-out configuration."""
    prepared = bundle
    arm = switches if switches is not None else ArmSwitches()
    config = prepared.config
    run_seed = config.release_cohort.seed if seed is None else seed
    members = prepared.partitions.in_configuration(configuration)
    if not members:
        raise KeyError(f"configuration '{configuration}' has no partitions")
    development = _first_role(members, "development")
    if development is None:
        raise KeyError(f"configuration '{configuration}' has no development partition")
    calibration = _first_role(members, "disjoint_weight_and_threshold_fitting")
    evaluation = _first_held_out(members)
    if evaluation is None:
        raise KeyError(f"configuration '{configuration}' has no held-out partition")
    calibration_member = (
        calibration.member
        if calibration is not None
        else np.zeros(prepared.cohort.attempts, dtype=np.bool_)
    )
    branch_model = BranchModel(config.propensity, seed=run_seed).fit(
        prepared.descriptors, prepared.cohort.branch_array()
    )
    action_model = ActionModel(config.propensity, seed=run_seed).fit(
        prepared.descriptors,
        prepared.cohort.branch_array(),
        prepared.cohort.strategies(),
    )
    aggressiveness = prepared.cohort.action_array()
    # The outcome model is fitted on the ladder the arm under test actually uses, so a
    # single-rung arm carries a single outcome column and the batch widths agree.
    arm_ladder = (
        ladder_for_depth(config.ladder, config.ladder_weights, 1)
        if (not arm.nested_excursion_ladder)
        else prepared.ladder
    )
    arm_indicators = arm_ladder.indicator_matrix(prepared.cohort.peaks())
    design = build_design(prepared.cohort.states, aggressiveness, include_action=arm.uses_action_term)
    outcome_model = fit_rung_outcome(config.outcome, design.features, arm_indicators, seed=run_seed)
    state_design = build_design(prepared.cohort.states, aggressiveness, include_action=False)
    state_only_model = fit_rung_outcome(
        config.outcome, state_design.features, arm_indicators, seed=run_seed
    )
    return AnalysisContext(
        bundle=prepared,
        configuration=configuration,
        switches=arm,
        branch_model=branch_model,
        action_model=action_model,
        outcome_model=outcome_model,
        state_only_model=state_only_model,
        development_member=development.member,
        calibration_member=calibration_member,
        evaluation_member=evaluation.member,
    )


def _first_role(members: tuple[PartitionIndex, ...], role: str) -> PartitionIndex | None:
    for item in members:
        if item.role == role:
            return item
    return None


def _first_held_out(members: tuple[PartitionIndex, ...]) -> PartitionIndex | None:
    for role in HELD_OUT_ROLES:
        found = _first_role(members, role)
        if found is not None:
            return found
    return None


def behaviour_propensity(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    target: Strategy,
) -> NDArray[np.float64]:
    """Marginal d_b(pi(s) | s), summed over the branch stage of the two-part behaviour policy."""
    descriptors = context.bundle.descriptors[member]
    branch_probability = context.branch_model.predict_proba(descriptors)
    branches = context.branch_model.classes
    if branches.size == 0:
        raise RuntimeError("the branch model carries no classes")
    total = np.zeros(descriptors.shape[0], dtype=np.float64)
    for position, branch in enumerate(branches.tolist()):
        repeated = np.full(descriptors.shape[0], branch, dtype="<U64")
        action_probability = context.action_model.target_probability(descriptors, repeated, target.name)
        total += branch_probability[:, position] * action_probability
    return np.asarray(np.clip(total, 1e-6, 1.0), dtype=np.float64)


def batch_for_strategy(
    context: AnalysisContext,
    target: Strategy,
    member: NDArray[np.bool_],
    switches: ArmSwitches | None = None,
) -> LogBatch:
    """Assemble the logged batch the estimators consume for one target strategy."""
    arm = switches if switches is not None else context.switches
    ladder = context.ladder_for(arm)
    states = context.states_member(member)
    indicators = ladder.indicator_matrix(context.cohort.peaks()[member])
    target_names = np.full(indicators.shape[0], target.name, dtype="<U64")
    aggressiveness = np.full(indicators.shape[0], target.aggressiveness, dtype=np.float64)
    if arm.uses_action_term:
        propensity = behaviour_propensity(context, member, target)
        delivered = context.cohort.strategies()[member]
    else:
        # The falsification arm removes the reconstructed action channel, so no propensity for
        # the target action can be formed and the estimator degenerates to the state alone.
        propensity = np.ones(indicators.shape[0], dtype=np.float64)
        delivered = target_names
    design = build_design(states, aggressiveness, include_action=arm.uses_action_term)
    model = context.outcome_model if arm.uses_action_term else context.state_only_model
    predictions = model.predict(design.features)
    return LogBatch(
        features=context.bundle.descriptors[member],
        behaviour_strategy=delivered,
        target_strategy=target_names,
        rung_observations=indicators,
        propensities=propensity,
        weights=np.asarray(ladder.weights, dtype=np.float64),
        strata=context.cohort.stage_array()[member],
        outcome_predictions=predictions,
    )


def estimate_strategy(
    context: AnalysisContext,
    target: Strategy,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> tuple[LogBatch, NestedDRResult]:
    batch = batch_for_strategy(context, target, member, switches)
    return batch, nested_doubly_robust(batch, clipping)


def strategy_estimates(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    clipping: float,
    switches: ArmSwitches | None = None,
) -> dict[str, dict[str, float]]:
    """One nested doubly-robust estimate per strategy, with the closed-form truth beside it."""
    table: dict[str, dict[str, float]] = {}
    for strategy in context.strategies:
        _, result = estimate_strategy(context, strategy, member, clipping, switches)
        truth = context.curves[strategy.name]
        table[strategy.name] = {
            "graded_value": result.graded_value,
            "risk": result.endpoint_risk,
            "effective_sample_size": result.effective_sample_size,
            "overlap_coefficient": result.overlap,
            "relative_error": relative_error(result.graded_value, truth["graded_value"]),
            "truth_graded_value": truth["graded_value"],
            "truth_risk": truth["risk"],
            "weight_concentration": result.weight_concentration,
        }
    return table


def strategy_ranking(table: dict[str, dict[str, float]], key: str) -> tuple[str, ...]:
    return tuple(sorted(table, key=lambda name: -table[name][key]))


def strategy_index(context: AnalysisContext, strategy: str) -> int:
    """Position of a strategy in the library, used to address the closed-form risk matrix."""
    for position, item in enumerate(context.strategies):
        if item.name == strategy:
            return position
    raise KeyError(f"unknown strategy '{strategy}'")


def best_strategy_by_truth(context: AnalysisContext, key: str = "graded_value") -> str:
    return max(context.curves, key=lambda name: context.curves[name][key])


def selected_strategy(table: dict[str, dict[str, float]]) -> str:
    return max(table, key=lambda name: table[name]["graded_value"])


def anatomical_score(context: AnalysisContext, member: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Anatomical supervised risk score: the state-only model read at a fixed strategy."""
    states = context.states_member(member)
    target = max(context.strategies, key=lambda item: context.curves[item.name]["risk"])
    aggressiveness = np.full(len(states), target.aggressiveness, dtype=np.float64)
    design = build_design(states, aggressiveness, include_action=False)
    endpoint = context.state_only_model.predict_endpoint(design.features)
    return np.asarray(endpoint, dtype=np.float64)


def action_conditioned_score(
    context: AnalysisContext,
    member: NDArray[np.bool_],
    target: Strategy,
) -> NDArray[np.float64]:
    """Action-conditioned score for one strategy: the rung product of the fitted outcome model."""
    states = context.states_member(member)
    aggressiveness = np.full(len(states), target.aggressiveness, dtype=np.float64)
    design = build_design(states, aggressiveness, include_action=True)
    product = context.outcome_model.product_risk(design.features)
    return np.asarray(product, dtype=np.float64)
