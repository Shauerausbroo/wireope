"""Two-part behaviour reconstruction, cohort assembly, adjudication and partitions."""

from __future__ import annotations

import numpy as np
import pytest

from wireope.behavior.action import branch_one_hot, one_hot_descriptors
from wireope.behavior.branch import behaviour_agreement, clip_propensity, top1_branch
from wireope.behavior.fidelity import (
    discretisation_sweep,
    fidelity_by_axis,
    multiclass_ece,
    multiclass_log_loss,
    operator_stratum,
    reconstruction_cost,
    top1_agreement,
)
from wireope.cohort.adjudication import (
    adjudicate,
    adjudication_agreement,
    cohen_kappa,
    endpoint_components,
    endpoint_rate_by_stratum,
    nested_unit_counts,
)
from wireope.cohort.partitions import (
    audit_leakage,
    build_partitions,
    disjoint_calibration_share,
    partition_summary,
    sitewise_slices,
)
from wireope.cohort.release_cohort import (
    STAGE_ORDER,
    aggressiveness_of,
    assemble_release_cohort,
    build_strategy_library,
    closed_form_curves,
    closed_form_state_probabilities,
    descriptor_vector,
    device_record,
    injury_from_peak,
    injury_probability,
    level_parameter,
    linear_predictor,
    observed_action_probability,
    rung_probability,
    sample_peak_excursion,
)


def test_strategy_library_spans_the_aggressiveness_range() -> None:
    library = build_strategy_library(8)
    assert len(library) == 8
    assert library[0].aggressiveness == 0.0
    assert library[-1].aggressiveness == 1.0
    assert aggressiveness_of(library, library[3].name) == library[3].aggressiveness
    with pytest.raises(KeyError):
        aggressiveness_of(library, "not_a_strategy")
    with pytest.raises(ValueError):
        build_strategy_library(1)


def test_injury_probability_is_monotone_in_the_action() -> None:
    coefficients = {
        "beta_stage": 0.42,
        "beta_calcification": 0.51,
        "beta_overlap": 0.63,
        "beta_action": 0.44,
    }
    low = linear_predictor(coefficients, 1, 2, 0.2, 0.0)
    high = linear_predictor(coefficients, 1, 2, 0.2, 1.0)
    assert high > low
    assert injury_probability(-4.0, high) > injury_probability(-4.0, low)


def test_rung_probability_closed_form() -> None:
    assert rung_probability(0.05, 0.5, 4, 4) == pytest.approx(0.05)
    first = rung_probability(0.05, 0.5, 4, 1)
    second = rung_probability(0.05, 0.5, 4, 2)
    assert first > second > 0.05
    with pytest.raises(ValueError):
        rung_probability(0.05, 0.5, 4, 5)


def test_level_parameter_tracks_the_action(bundle: object) -> None:
    config = bundle.config.release_cohort
    assert level_parameter(config, 0.0) < level_parameter(config, 1.0)
    assert 0.0 <= level_parameter(config, 1.0) <= 1.0


def test_closed_form_curves_rank_the_gentle_strategy(bundle: object) -> None:
    ladder = bundle.ladder
    strategies = build_strategy_library(8)
    cohort = bundle.cohort
    curves = closed_form_curves(
        bundle.config.release_cohort,
        ladder,
        strategies,
        cohort.states,
        cohort.calibration_intercepts,
    )
    risks = [curves[strategy.name]["risk"] for strategy in strategies]
    assert risks[0] < risks[-1]
    progress = [curves[strategy.name]["graded_progress"] for strategy in strategies]
    assert progress[0] > progress[-1]
    for entry in curves.values():
        assert 0.0 <= entry["risk"] <= 1.0
        assert entry["graded_value"] + entry["graded_progress"] == pytest.approx(1.0)


def test_state_probability_matrix_shape(bundle: object) -> None:
    cohort = bundle.cohort
    matrix = closed_form_state_probabilities(
        bundle.config.release_cohort,
        bundle.ladder,
        build_strategy_library(8),
        cohort.states[:20],
        cohort.calibration_intercepts,
    )
    assert matrix.shape == (20, 8)
    assert float(matrix.min()) > 0.0
    assert float(matrix.max()) < 1.0


def test_release_cohort_nests_and_hits_the_targets(bundle: object) -> None:
    cohort = bundle.cohort
    ladder = bundle.ladder
    peaks = cohort.peaks()
    assert ladder.nesting_holds(peaks)
    assert np.array_equal(ladder.indicator_matrix(peaks)[:, -1], cohort.injury_flags().astype(np.float64))
    config = bundle.config.release_cohort
    for spec in config.site_targets:
        member = cohort.site_array() == spec.site
        assert int(np.count_nonzero(member)) == spec.attempts
        assert int(np.sum(cohort.injury_flags()[member])) == spec.injury_events


def test_peak_sampling_respects_the_top_rung(bundle: object) -> None:
    ladder = bundle.ladder
    rng = np.random.default_rng(11)
    for level_q in (0.1, 0.5, 0.9):
        for injured in (0, 1):
            peak = sample_peak_excursion(rng, ladder, level_q, injured)
            assert injury_from_peak(peak, ladder) == injured


def test_descriptor_vector_and_observed_probability(bundle: object) -> None:
    cohort = bundle.cohort
    vector = descriptor_vector(cohort.states[0])
    assert vector.shape[0] == 16
    assert vector[STAGE_ORDER.index(cohort.states[0].stage)] == 1.0
    probability = observed_action_probability(cohort)
    assert probability.shape[0] == cohort.attempts
    assert float(probability.min()) > 0.0
    assert float(probability.max()) <= 1.0


def test_device_record_cycles_the_classes() -> None:
    assert device_record(0)["device_class"] != device_record(1)["device_class"]
    assert device_record(0)["attempt_index"] == 1


def test_behaviour_models_fit_and_predict(context: object) -> None:
    descriptors = context.bundle.descriptors
    cohort = context.bundle.cohort
    branches = cohort.branch_array()
    strategies = cohort.strategies()
    branch_model = context.branch_model
    action_model = context.action_model
    assert branch_model.is_fitted and action_model.is_fitted
    probability = branch_model.predict_proba(descriptors[:20])
    assert probability.shape[1] == len(set(branches.tolist()))
    assert np.allclose(probability.sum(axis=1), 1.0)
    selected = branch_model.probability_of(descriptors[:20], branches[:20])
    assert np.all(selected > 0.0)
    agreement = behaviour_agreement(top1_branch(branch_model, descriptors[:200]), branches[:200])
    assert 0.0 <= agreement <= 1.0
    target = build_strategy_library(8)[0].name
    repeated = np.full(20, branches[0], dtype="<U64")
    action_probability = action_model.target_probability(descriptors[:20], repeated, target)
    assert np.all(action_probability > 0.0)
    assert action_model.log_probability(descriptors[:20], branches[:20], strategies[:20]).shape == (20,)
    assert isinstance(branch_model.design().feature_names, tuple)
    assert isinstance(action_model.design().classes, tuple)


def test_behaviour_models_reject_unknown_labels(context: object) -> None:
    descriptors = context.bundle.descriptors
    branch_model = context.branch_model
    unknown = np.full(3, "nowhere", dtype="<U64")
    with pytest.raises(ValueError):
        branch_model.probability_of(descriptors[:3], unknown)


def test_propensity_clipping_and_one_hot() -> None:
    clipped = clip_propensity(np.asarray([0.001, 0.5]), 0.02)
    assert clipped[0] == pytest.approx(0.02)
    classes = np.asarray(["I", "II", "III"], dtype="<U8")
    one_hot = branch_one_hot(np.asarray(["II", "III"], dtype="<U8"), classes)
    assert one_hot.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    descriptors = one_hot_descriptors(
        np.asarray(["I", "II"], dtype="<U8"), np.asarray([0, 4], dtype=np.int64)
    )
    assert descriptors.shape == (2, 8)


def test_fidelity_metrics() -> None:
    probability = np.asarray([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7]], dtype=np.float64)
    target = np.asarray([0, 1, 1], dtype=np.int64)
    assert multiclass_log_loss(probability, target) > 0.0
    assert top1_agreement(probability, target) == pytest.approx(1.0)
    assert 0.0 <= multiclass_ece(probability, target) <= 1.0
    report = fidelity_by_axis("site", np.asarray(["a", "a", "b"], dtype="<U8"), probability, target)
    assert report.groups == ("a", "b")
    assert report.worst_log_loss_group() in {"a", "b"}
    assert operator_stratum(np.asarray([1, 5, 9, 20], dtype=np.int64)).shape == (4,)


def test_granularity_sweep_and_cost() -> None:
    rng = np.random.default_rng(5)
    realised = rng.normal(size=200)
    reconstructed = realised + rng.normal(scale=0.1, size=200)
    residuals = discretisation_sweep(realised, reconstructed, (4, 16, 64), ("coarse", "medium", "fine"))
    assert len(residuals) == 3
    assert all(0.0 <= item.explained_share <= 1.0 for item in residuals)
    cost = reconstruction_cost(realised, reconstructed)
    assert cost["mean_absolute_residual"] > 0.0
    with pytest.raises(ValueError):
        discretisation_sweep(realised, reconstructed[:10], (4,), ("coarse",))


def test_adjudication_pairing_and_agreement() -> None:
    attempt_ids = np.arange(6, dtype=np.int64)
    calls = np.asarray([[1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int64)
    pairs = adjudicate(attempt_ids, calls, reviewers_min=2)
    assert len(pairs) == 6
    assert pairs[2].final == 1
    summary = adjudication_agreement(pairs)
    assert summary["agreement"] == pytest.approx(4 / 6)
    assert -1.0 <= cohen_kappa(calls[:, 0], calls[:, 1]) <= 1.0
    with pytest.raises(ValueError):
        adjudicate(attempt_ids, calls[:, :1], reviewers_min=2)


def test_endpoint_components_and_rates() -> None:
    perforation = np.asarray([1, 0, 0, 1], dtype=np.int64)
    dissection = np.asarray([0, 1, 1, 1], dtype=np.int64)
    components = endpoint_components(perforation, dissection)
    assert components["union"] == 4.0
    assert components["both"] == 1.0
    rates = endpoint_rate_by_stratum(
        np.asarray([1, 0, 1, 1], dtype=np.int64), np.asarray(["a", "a", "b", "b"], dtype="<U4")
    )
    assert rates["a"] == pytest.approx(0.5)
    counts = nested_unit_counts(
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    assert counts == {"procedures": 2, "lesions": 2, "attempts": 3}


def test_partition_construction_is_disjoint(bundle: object) -> None:
    cohort = bundle.cohort
    patient_ids = np.asarray([state.patient_id for state in cohort.states], dtype=np.int64)
    partitions = build_partitions(
        bundle.config.partitions,
        cohort.site_array(),
        cohort.region_array(),
        patient_ids=patient_ids,
        calibration_fraction=0.25,
        seed=1,
    )
    fit = partitions.by_name("fit_primary")
    calibration = partitions.by_name("calibration_primary")
    assert not np.any(fit.member & calibration.member)
    assert partitions.configurations() == ("cross_region", "primary_external", "sitewise")
    report = audit_leakage(
        partitions,
        patient_ids=patient_ids,
        operators=cohort.operator_array(),
        vendor_classes=cohort.vendor_array(),
        guards=bundle.config.leakage,
    )
    assert report.clean
    summary = partition_summary(fit, cohort.site_array())
    assert summary["size"] == float(np.count_nonzero(fit.member))
    assert set(sitewise_slices(cohort.site_array())) == {"Site_A", "Site_B", "Site_C"}


def test_disjoint_calibration_share_is_deterministic() -> None:
    member = np.ones(40, dtype=bool)
    patients = np.repeat(np.arange(10, dtype=np.int64), 4)
    first, second = disjoint_calibration_share(member, patients, 0.3, seed=7)
    again_first, again_second = disjoint_calibration_share(member, patients, 0.3, seed=7)
    assert np.array_equal(first, again_first)
    assert not np.any(first & second)
    with pytest.raises(ValueError):
        disjoint_calibration_share(member, patients, 1.5, seed=7)


def test_reassembled_cohort_is_identical(bundle: object) -> None:
    cohort = bundle.cohort
    config = bundle.config.release_cohort
    reassembled = assemble_release_cohort(config, bundle.ladder)
    assert np.array_equal(reassembled.injury_flags(), cohort.injury_flags())
    assert np.allclose(reassembled.peaks(), cohort.peaks())
