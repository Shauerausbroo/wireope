"""Estimator library, the nested doubly-robust estimator and the bake-off."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wireope.bounds.effective_sample import (
    binding_rung,
    degenerate_overlap,
    effective_sample_size,
    overlap_coefficient,
    per_rung_effective_sample_size,
    relative_error,
    weight_concentration,
)
from wireope.estimators.base import LogBatch, clip_weights, self_normalised, weighted_ladder_value
from wireope.estimators.direct import FittedQEvaluation, direct_graded_value, direct_rung_risk
from wireope.estimators.doubly_robust import (
    DoublyRobust,
    SwitchingHybrid,
    WeightedDoublyRobust,
    correction_magnitude,
    doubly_robust_rungs,
    weighted_doubly_robust_rungs,
)
from wireope.estimators.high_confidence import (
    BinaryHighConfidence,
    effective_positive_events,
    width_under_negative_dilution,
)
from wireope.estimators.importance import (
    ImportanceSampling,
    PerDecisionImportanceSampling,
    PerDecisionWeightedImportanceSampling,
    WeightedImportanceSampling,
    rung_ratio_matrix,
)
from wireope.estimators.kernel import (
    InfiniteHorizonKernel,
    gaussian_kernel_matrix,
    kernel_rung_values,
    kernel_support_fraction,
    median_bandwidth,
    state_density_weights,
)
from wireope.estimators.nested_dr import (
    NestedWeightedDoublyRobust,
    clip_probability_matrix,
    clipped_observations,
    conditionals_from_marginals,
    graded_progress_value,
    graded_truth_from_probabilities,
    nested_doubly_robust,
    product_risk,
    product_risk_from_conditionals,
    product_risk_from_marginals,
    severity_from_progress,
    truncation_bias,
)
from wireope.estimators.registry import (
    FAMILY_BY_KEY,
    build_estimator,
    build_library,
    family_instability,
    run_bakeoff,
    selection_accuracy_from_ranking,
    stable_family_ranking,
    within_ten_percent_share,
)
from wireope.studies.pipeline import batch_for_strategy


def make_batch(
    attempts: int = 2000,
    rungs: int = 3,
    propensity: float = 0.4,
    share: float = 0.4,
    seed: int = 4,
) -> LogBatch:
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(0.6, 0.1, rungs)
    observations = np.zeros((attempts, rungs), dtype=np.float64)
    for rung in range(rungs):
        observations[:, rung] = (rng.random(attempts) < probabilities[rung]).astype(np.float64)
    for rung in range(rungs - 1, 0, -1):
        observations[:, rung - 1] = np.maximum(observations[:, rung - 1], observations[:, rung])
    delivered = np.where(rng.random(attempts) < share, "target", "other")
    return LogBatch(
        features=rng.normal(size=(attempts, 3)),
        behaviour_strategy=delivered,
        target_strategy=np.full(attempts, "target", dtype="<U64"),
        rung_observations=observations,
        propensities=np.full(attempts, propensity, dtype=np.float64),
        weights=np.full(rungs, 1.0 / rungs, dtype=np.float64),
        strata=np.full(attempts, "I", dtype="<U8"),
        outcome_predictions=np.tile(probabilities, (attempts, 1)),
    )


def test_batch_validates_its_arrays() -> None:
    batch = make_batch(attempts=20)
    assert batch.attempts == 20
    assert batch.rungs == 3
    assert batch.assigned.sum() > 0
    with pytest.raises(ValueError):
        LogBatch(
            features=batch.features,
            behaviour_strategy=batch.behaviour_strategy[:5],
            target_strategy=batch.target_strategy,
            rung_observations=batch.rung_observations,
            propensities=batch.propensities,
            weights=batch.weights,
            strata=batch.strata,
        )


def test_importance_weights_and_ess() -> None:
    batch = make_batch(attempts=400, propensity=0.25, share=0.25)
    weights = batch.importance_weights()
    expected = batch.assigned.astype(np.float64) / 0.25
    assert np.allclose(weights, expected)
    assert effective_sample_size(np.ones(4)) == pytest.approx(4.0)
    assert effective_sample_size(np.asarray([3.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert effective_sample_size(np.zeros(3)) == 0.0
    assert batch.effective_sample_size(weights) > 0.0
    assert 0.0 <= batch.overlap_coefficient() <= 1.0


def test_clipping_and_self_normalisation() -> None:
    weights = np.asarray([0.1, 2.0, 6.0])
    clipped = clip_weights(weights, 0.25)
    assert clipped.max() <= 4.0 + 1e-12
    assert clip_weights(weights, 0.0).tolist() == weights.tolist()
    normalised = self_normalised(weights)
    assert normalised.sum() == pytest.approx(1.0)
    assert self_normalised(np.zeros(3)).sum() == 0.0
    assert weighted_ladder_value(np.asarray([1.0, 2.0]), np.asarray([0.5, 0.5])) == pytest.approx(1.5)


def test_importance_family_runs() -> None:
    batch = make_batch()
    for estimator in (
        ImportanceSampling(),
        PerDecisionImportanceSampling(),
        WeightedImportanceSampling(),
        PerDecisionWeightedImportanceSampling(),
    ):
        result = estimator.estimate(batch, 0.1)
        assert math.isfinite(result.value)
        assert result.per_rung_value.shape[0] == batch.rungs
        assert math.isfinite(result.risk)
    assert rung_ratio_matrix(batch, 0.1).shape == (batch.attempts, batch.rungs)


def test_direct_family_ignores_weights() -> None:
    batch = make_batch()
    result = FittedQEvaluation().estimate(batch, 0.1)
    assert result.effective_sample_size == float(batch.attempts)
    assert result.diagnostics["weightless"] == 1.0
    assert direct_graded_value(batch) == pytest.approx(result.value)
    assert direct_rung_risk(batch) == pytest.approx(result.risk)


def test_doubly_robust_family_runs() -> None:
    batch = make_batch()
    for estimator in (DoublyRobust(), WeightedDoublyRobust(), SwitchingHybrid()):
        result = estimator.estimate(batch, 0.1)
        assert math.isfinite(result.value)
    weights = batch.importance_weights()
    assert doubly_robust_rungs(batch, weights).shape == (batch.rungs,)
    assert weighted_doubly_robust_rungs(batch, weights).shape == (batch.rungs,)
    assert correction_magnitude(batch, weights) >= 0.0
    switch = SwitchingHybrid(switch_constant=1.0).switching_weights(weights)
    assert np.all((switch >= 0.0) & (switch <= 1.0))


def test_kernel_family() -> None:
    batch = make_batch(attempts=300, seed=8)
    result = InfiniteHorizonKernel().estimate(batch, 0.1)
    assert math.isfinite(result.value)
    assert result.diagnostics["bandwidth"] > 0.0
    weights = state_density_weights(batch.features, 1.0, 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert kernel_rung_values(batch).shape == (batch.rungs,)
    assert 0.0 <= kernel_support_fraction(weights, 1e-3) <= 1.0
    assert median_bandwidth(batch.features) > 0.0
    with pytest.raises(ValueError):
        gaussian_kernel_matrix(batch.features, 0.0)
    assert median_bandwidth(np.zeros((1, 3))) == 1.0


def test_high_confidence_comparator() -> None:
    batch = make_batch()
    result = BinaryHighConfidence().estimate(batch, 0.1)
    assert math.isfinite(result.value)
    assert result.risk >= result.value - 1e-9
    assert result.diagnostics["ladder_used"] == 0.0
    positive = effective_positive_events(np.asarray([1.0, 1.0, 1.0, 1.0]))
    assert positive == pytest.approx(4.0)
    widths = width_under_negative_dilution(np.ones(12), 0.05, 1.0, np.asarray([100, 5000], dtype=np.int64))
    assert widths[0] == pytest.approx(widths[1])


def test_nested_estimator_recovers_known_functionals() -> None:
    batch = make_batch(attempts=20000, rungs=2, propensity=0.4, share=0.4, seed=12)
    wrong_direct = np.full((batch.attempts, 2), 0.05, dtype=np.float64)
    batch = LogBatch(
        features=batch.features,
        behaviour_strategy=batch.behaviour_strategy,
        target_strategy=batch.target_strategy,
        rung_observations=batch.rung_observations,
        propensities=batch.propensities,
        weights=np.asarray([0.5, 0.5]),
        strata=batch.strata,
        outcome_predictions=wrong_direct,
    )
    result = nested_doubly_robust(batch, 0.0)
    truth_marginals = batch.rung_observations.mean(axis=0)
    truth_graded = float(np.sum(batch.weights * truth_marginals))
    assert result.graded_value == pytest.approx(truth_graded, abs=0.02)
    assert result.endpoint_risk == pytest.approx(truth_marginals[-1], abs=0.02)
    assert result.per_rung_value.shape == (2,)
    assert result.weight_concentration > 0.0


def test_nested_estimator_class_and_helpers() -> None:
    rung_probability = np.asarray([0.5, 0.15])
    batch = make_batch(attempts=2000, rungs=2, seed=13)
    batch = LogBatch(
        features=batch.features,
        behaviour_strategy=batch.behaviour_strategy,
        target_strategy=batch.target_strategy,
        rung_observations=batch.rung_observations,
        propensities=batch.propensities,
        weights=np.asarray([0.5, 0.5]),
        strata=batch.strata,
        outcome_predictions=np.tile(rung_probability, (batch.attempts, 1)),
    )
    estimator = NestedWeightedDoublyRobust(clipping=0.1)
    as_result = estimator.estimate(batch, 0.1)
    assert as_result.family == "doubly_robust"
    detail = estimator.detailed(batch, 0.1)
    assert detail.relative_error(0.3) >= 0.0
    marginals = np.asarray([0.6, 0.3, 0.1, 0.02])
    conditionals = conditionals_from_marginals(marginals)
    assert conditionals[0] == pytest.approx(0.6)
    assert product_risk_from_marginals(marginals) == pytest.approx(0.02)
    assert graded_progress_value(marginals, np.full(4, 0.25)) == pytest.approx(
        0.25 * float(np.sum(1.0 - marginals))
    )
    assert severity_from_progress(0.75, np.full(4, 0.25)) == pytest.approx(0.25)
    assert product_risk(np.asarray([0.5, 0.4])) == pytest.approx(0.2)
    assert clipped_observations(np.asarray([-1.0, 2.0]), 1.0).tolist() == [0.0, 1.0]
    assert clip_probability_matrix(np.asarray([[0.0, 1.0]]), 0.01).min() >= 0.01


def test_product_risk_from_conditionals_and_truncation(bundle: object) -> None:
    ladder = bundle.ladder
    conditionals = np.full(ladder.depth, 0.5)
    assert product_risk_from_conditionals(ladder, conditionals) == pytest.approx(0.5**ladder.depth)
    with pytest.raises(ValueError):
        product_risk_from_conditionals(ladder, np.zeros(1))
    graded, risk = graded_truth_from_probabilities(ladder, conditionals)
    weights = np.asarray(ladder.weights)
    assert graded == pytest.approx(float(np.sum(weights * np.cumprod(conditionals))))
    assert risk == pytest.approx(0.5**ladder.depth)
    with pytest.raises(ValueError):
        graded_truth_from_probabilities(ladder, np.zeros(1))
    bias = truncation_bias(np.full(ladder.depth, 0.5), ladder, 0.2)
    assert bias >= 0.0


def test_registry_builds_every_candidate() -> None:
    for key in FAMILY_BY_KEY:
        estimator = build_estimator(key)
        assert estimator.key == key
    with pytest.raises(ValueError):
        build_estimator("not_an_estimator")


def test_bakeoff_and_ranking() -> None:
    batch = make_batch(attempts=1500, seed=21)
    truth = 0.3
    comparison = dict.fromkeys(FAMILY_BY_KEY, 6.3)
    rows = run_bakeoff(batch, truth, 0.1, 0.05, comparison)
    assert len(rows) == len(FAMILY_BY_KEY)
    assert all(row.relative_error >= 0.0 for row in rows)
    ranking = stable_family_ranking(rows)
    assert set(ranking) == set(FAMILY_BY_KEY)
    assert family_instability(rows, "importance_sampling") >= 0.0
    assert family_instability(rows, "not_a_family") == 0.0
    assert selection_accuracy_from_ranking(np.asarray([3.0, 1.0]), np.asarray([2.0, 1.0])) == 1.0
    assert within_ten_percent_share(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])) == 1.0
    assert within_ten_percent_share(np.zeros(0), np.zeros(0)) == 0.0


def test_build_library_from_config(bundle: object) -> None:
    library = bundle.config.library
    estimators = build_library(library)
    keys = {estimator.key for estimator in estimators}
    assert "nested_dr" in keys
    assert library.comparator.key in keys


def test_effective_sample_diagnostics() -> None:
    weights = np.asarray([1.0, 1.0, 4.0])
    sizes = per_rung_effective_sample_size(weights, np.asarray([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]))
    assert sizes.shape == (2,)
    assert binding_rung(sizes) in {0, 1}
    assert binding_rung(np.zeros(0)) == -1
    assert weight_concentration(weights) == pytest.approx(4.0 / 6.0)
    assert weight_concentration(np.zeros(3)) == 1.0
    assert degenerate_overlap(0.0)
    assert not degenerate_overlap(0.5)
    assert overlap_coefficient(np.asarray([0.5, 0.2]), np.asarray([0.5, 0.9])) == pytest.approx(0.35)
    with pytest.raises(ValueError):
        overlap_coefficient(np.asarray([0.5]), np.asarray([0.5, 0.5]))
    assert relative_error(1.0, 2.0) == pytest.approx(0.5)
    assert math.isinf(relative_error(1.0, 0.0))


def test_batch_for_strategy_shapes(context: object) -> None:
    strategies = context.strategies
    member = context.evaluation_member
    batch = batch_for_strategy(context, strategies[0], member, context.switches)
    assert batch.attempts == int(member.sum())
    assert batch.rungs == context.ladder.depth
    assert float(batch.propensities.min()) > 0.0
