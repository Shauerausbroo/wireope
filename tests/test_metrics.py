"""Discrimination, calibration, decision-analytic, safety and bootstrap metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from wireope.metrics.bootstrap import (
    bootstrap_interval,
    bootstrap_statistic,
    clustered_bootstrap_interval,
    clustered_difference_interval,
    paired_bootstrap_difference,
)
from wireope.metrics.calibration import (
    brier_score,
    calibration_report,
    calibration_slope_intercept,
    expected_calibration_error,
    one_minus_ece,
    reliability_curve,
)
from wireope.metrics.decision import (
    best_threshold_by_net_benefit,
    net_benefit,
    net_benefit_advantage,
    net_benefit_curve,
    net_reclassification_improvement,
    reference_curves,
    standardised_net_benefit,
    threshold_inside_range,
    treat_all_net_benefit,
    treat_none_net_benefit,
)
from wireope.metrics.discrimination import (
    auroc,
    average_precision,
    average_precision_prevalence_axis,
    binormal_auroc_from_separation,
    delong_components,
    delong_interval,
    delong_test,
    discrimination_from_scores,
    midrank_matrix,
    parity_interval_overlap,
)
from wireope.metrics.safety import (
    absolute_risk_difference,
    abstention_accuracy,
    abstention_degeneracy,
    abstention_report,
    coverage_weighted_safety,
    mean_regret,
    mean_selected_risk,
    regret,
    safety_gain_from_constraint,
    selection_accuracy,
    violation_rate,
    violation_rate_upper_bound,
)


def drawn_scores(seed: int = 0, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.2 * scores))).astype(np.int64)
    return scores, labels


def test_auroc_matches_the_library() -> None:
    scores, labels = drawn_scores()
    assert auroc(scores, labels) == pytest.approx(float(roc_auc_score(labels, scores)), abs=1e-9)
    assert math.isnan(auroc(scores, np.zeros_like(labels)))


def test_auroc_pair_count_and_rank() -> None:
    scores = np.asarray([0.1, 0.5, 0.4, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.75)
    ranks = midrank_matrix(np.asarray([1.0, 2.0, 2.0, 3.0]))
    assert ranks[1] == ranks[2]
    v10, v01 = delong_components(scores, labels)
    assert v10.shape[0] == 2 and v01.shape[0] == 2


def test_delong_interval_and_test() -> None:
    scores, labels = drawn_scores(seed=1)
    interval = delong_interval(scores, labels)
    assert interval.lower <= interval.auroc <= interval.upper
    assert interval.positives + interval.negatives == scores.shape[0]
    other = scores + np.random.default_rng(3).normal(scale=0.3, size=scores.shape[0])
    z_value, p_value = delong_test(scores, other, labels)
    assert math.isfinite(z_value)
    assert 0.0 <= p_value <= 1.0
    narrow = delong_interval(np.asarray([0.1, 0.9]), np.asarray([0, 1]))
    assert not math.isfinite(narrow.lower) or math.isnan(narrow.lower)
    overlap = parity_interval_overlap(interval, interval)
    assert overlap["difference"] == pytest.approx(0.0)
    assert abs(overlap["share_of_narrower"] - 1.0) < 1e-9
    assert discrimination_from_scores(scores, labels).auroc == pytest.approx(interval.auroc)
    assert binormal_auroc_from_separation(1.0) == pytest.approx(0.7602, abs=1e-3)


def test_average_precision() -> None:
    scores, labels = drawn_scores(seed=5)
    assert average_precision(scores, labels) == pytest.approx(
        float(average_precision_score(labels, scores)), abs=1e-9
    )
    axis = average_precision_prevalence_axis(scores, labels, 0.038)
    assert axis["baseline_prevalence"] == pytest.approx(0.038)
    assert math.isnan(average_precision(scores, np.zeros_like(labels)))


def test_calibration_metrics() -> None:
    probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    ece = expected_calibration_error(probability, labels, bins=10)
    assert ece == pytest.approx((0.1 + 0.2 + 0.2 + 0.1) / 4.0, abs=1e-9)
    assert one_minus_ece(probability, labels) == pytest.approx(1.0 - ece)
    slope, intercept = calibration_slope_intercept(probability, labels)
    assert math.isfinite(slope) and math.isfinite(intercept)
    assert brier_score(probability, labels) == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4.0)
    report = calibration_report(probability, labels)
    assert report.one_minus_ece == pytest.approx(1.0 - report.expected_calibration_error)
    means, observed, counts = reliability_curve(probability, labels, bins=5)
    assert means.shape == observed.shape == counts.shape


def test_net_benefit_metrics() -> None:
    probability = np.asarray([0.9, 0.2, 0.6, 0.1])
    labels = np.asarray([1, 0, 1, 0])
    measured = net_benefit(probability, labels, 0.3)
    assert measured == pytest.approx(0.5)
    assert treat_none_net_benefit() == 0.0
    assert treat_all_net_benefit(0.25, 0.5) == pytest.approx(-0.5)
    assert treat_all_net_benefit(0.5, 0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        net_benefit(probability, labels, 0.0)
    with pytest.raises(ValueError):
        treat_all_net_benefit(0.25, 1.0)
    curve = net_benefit_curve(probability, labels, (0.1, 0.3))
    assert curve.shape == (2,)
    references = reference_curves(labels, (0.1, 0.3))
    assert references.at(0.3)["threshold"] == pytest.approx(0.3)
    assert standardised_net_benefit(probability, labels, 0.3) == pytest.approx(measured / 0.5)
    assert math.isnan(standardised_net_benefit(probability, np.zeros(4), 0.3))
    improvement = net_reclassification_improvement(
        np.asarray([0.9, 0.1]), np.asarray([0.1, 0.9]), np.asarray([1, 0]), 0.5
    )
    assert improvement["nri"] > 0.0
    assert best_threshold_by_net_benefit(probability, labels, (0.1, 0.3, 0.5)) in {0.1, 0.3, 0.5}
    advantage = net_benefit_advantage(probability, labels, (0.1, 0.3))
    assert advantage["beats_treat_none_everywhere"] in {0.0, 1.0}
    assert threshold_inside_range(0.05, 0.03, 0.15)


def test_safety_metrics() -> None:
    abstained = np.asarray([False, False, True, True])
    report = abstention_report(abstained)
    assert report.abstention_rate == pytest.approx(0.5)
    assert report.complementary
    assert not abstention_degeneracy(report)
    assert abstention_degeneracy(abstention_report(np.ones(4, dtype=bool)))
    realised = np.asarray([1, 0, 1, 0])
    assert violation_rate(realised, abstained) == pytest.approx(0.5)
    assert violation_rate_upper_bound(realised, abstained) >= 0.5
    assert violation_rate(np.asarray([1]), np.asarray([True])) == 0.0
    assert violation_rate_upper_bound(np.asarray([1]), np.asarray([True])) == 0.0
    assert selection_accuracy(np.asarray(["a"]), np.asarray(["a"])) == 1.0
    assert regret(np.asarray([0.1, 0.5]), np.asarray([0.3, 0.4])).tolist() == pytest.approx([0.2, 0.0])
    assert mean_regret(np.asarray([0.1]), np.asarray([0.3])) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        regret(np.asarray([0.1]), np.asarray([0.1, 0.2]))
    assert abstention_accuracy(np.asarray([True, True]), np.asarray([True, False])) == 0.5
    difference = absolute_risk_difference(10, 100, 20, 100)
    assert difference["absolute_risk_difference"] == pytest.approx(-0.1)
    assert difference["lower"] < difference["upper"]
    with pytest.raises(ValueError):
        absolute_risk_difference(1, 0, 1, 1)
    assert safety_gain_from_constraint(0.04, 0.01) == pytest.approx(0.03)
    assert coverage_weighted_safety(0.03, 0.5) == pytest.approx(0.06)
    assert coverage_weighted_safety(0.03, 0.0) == 0.0
    assert mean_selected_risk(np.asarray([0.1, 0.2]), np.asarray([False, True])) == pytest.approx(0.1)
    assert math.isnan(mean_selected_risk(np.asarray([0.1]), np.asarray([True])))


def test_bootstrap_intervals() -> None:
    rng = np.random.default_rng(9)
    sample = rng.normal(loc=1.0, size=200)
    interval = bootstrap_statistic(sample, "mean", resamples=200, seed=1)
    assert interval.lower < interval.point < interval.upper
    assert interval.width > 0.0
    assert bootstrap_statistic(np.zeros(0), "mean").resamples == 0
    with pytest.raises(ValueError):
        bootstrap_statistic(sample, "skew")
    paired = paired_bootstrap_difference(sample, sample - 0.5, resamples=150, seed=2)
    assert paired.point == pytest.approx(0.5)
    with pytest.raises(ValueError):
        paired_bootstrap_difference(sample, sample[:-1])
    clusters = np.repeat(np.asarray(["a", "b", "c", "d"]), 25)
    clustered = clustered_bootstrap_interval(sample, clusters, resamples=100, seed=3)
    assert clustered.resamples > 0
    contrast = clustered_difference_interval(sample, sample - 0.2, clusters, resamples=80, seed=4)
    assert contrast.point == pytest.approx(0.2)
    assert clustered_bootstrap_interval(np.zeros(0), np.zeros(0, dtype="<U4")).resamples == 0
    with pytest.raises(ValueError):
        clustered_difference_interval(sample, sample[:-1], clusters)
    assert bootstrap_interval(np.zeros(0)).resamples == 0
