"""Confidence bounds, the support floor, the selection rule and the weight-control sweep."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wireope.bounds.effective_sample import (
    effective_sample_size,
    overlap_coefficient,
    per_rung_effective_sample_size,
    relative_error,
)
from wireope.bounds.empirical_bernstein import (
    agrees_with_hoeffding_at_maximum_variance,
    empirical_bernstein_bound,
    maximum_variance,
    per_rung_variance,
    residual_conformed,
)
from wireope.bounds.hoeffding import (
    apparent_gain_collapse,
    design_event_requirement,
    effective_sample_size_minimum,
    hoeffding_bound,
    monotone_in_rarest_rung,
    radius_increases_with_depth,
)
from wireope.bounds.interval import (
    empirical_bernstein_radius,
    hoeffding_radius,
    interval_for_rung,
    interval_from_radius,
    lower_confidence_bound,
    reduces_to_hoeffding,
    upper_confidence_bound,
    weighted_mean,
    weighted_variance,
)
from wireope.certification.eligibility import (
    eligible_strategies,
    evaluate_floor,
    floor_effect_on_safety,
    stratum_rung_mass,
    support_covered_strata,
    worst_margin,
)
from wireope.certification.rule import (
    StrategyEstimate,
    abstention_accuracy,
    net_benefit_gap,
    parity_holds,
    regret,
    relative_utility_gain,
    select_constrained,
    select_unconstrained,
    selection_accuracy,
)
from wireope.certification.sweep import (
    SweepCell,
    cell_array,
    clipping_effect,
    conservative_headline,
    pessimism_radius_scale,
    stratum_labels,
    sweep_grid,
    sweep_monotone_in_clipping,
    sweep_monotone_in_pessimism,
)


def counts() -> np.ndarray:
    return np.asarray([400.0, 160.0, 60.0, 20.0])


def test_hoeffding_matches_the_closed_form() -> None:
    bound = hoeffding_bound(counts(), 0.05, 1.0, truncation_term=0.01)
    manual = math.sqrt(math.log(2.0 * 4 / 0.05) / (2.0 * 20.0))
    assert bound.radius == pytest.approx(manual)
    assert bound.total == pytest.approx(manual + 0.01)
    assert bound.effective_sample_size == pytest.approx(20.0)


def test_degenerate_bounds() -> None:
    empty = hoeffding_bound(np.zeros(0), 0.05, 1.0, 0.0)
    assert math.isinf(empty.radius)
    assert effective_sample_size_minimum(np.zeros(3)) == 0.0
    assert effective_sample_size_minimum(counts()) == 20.0
    zero_depth = hoeffding_bound(np.zeros(0), 0.05, 1.0, 0.0)
    assert zero_depth.total == float("inf")


def test_radius_falls_with_the_rarest_rung() -> None:
    base = counts()
    radii = [
        hoeffding_bound(np.concatenate([base[:-1], [value]]), 0.05, 1.0, 0.0).radius
        for value in (20.0, 40.0, 80.0)
    ]
    assert radii[0] > radii[1] > radii[2]
    assert monotone_in_rarest_rung(base, 0.05, 1.0)
    assert radius_increases_with_depth(base, 0.05, 1.0)


def test_design_event_requirement_inverts_the_bound() -> None:
    needed = design_event_requirement(0.05, 0.05, 1.0, 4, 0.0)
    assert needed == pytest.approx(1.0 * math.log(8.0 / 0.05) / (2.0 * 0.0025))
    assert math.isinf(design_event_requirement(0.01, 0.05, 1.0, 4, 0.05))
    collapse = apparent_gain_collapse(32.0, 3.0, 0.001, 0.5)
    assert collapse["ratio"] == pytest.approx(3.0 / 32.0)
    assert math.isnan(apparent_gain_collapse(0.0, 3.0, 0.001, 0.5)["ratio"])


def test_empirical_bernstein_and_its_limit() -> None:
    rng = np.random.default_rng(2)
    observations = (rng.random((400, 3)) < 0.25).astype(np.float64)
    weights = np.ones(400)
    rung_counts = np.asarray([100.0, 40.0, 12.0])
    bound = empirical_bernstein_bound(observations, weights, rung_counts, 0.05, 1.0)
    assert bound.total >= bound.residual_correction > 0.0
    assert per_rung_variance(observations, weights).shape == (3,)
    table = agrees_with_hoeffding_at_maximum_variance(rung_counts, 0.05, 1.0)
    assert table["leading_ratio"] == pytest.approx(1.0)
    assert maximum_variance(1.0) == pytest.approx(0.25)
    assert residual_conformed(observations, weights, 1.0)
    assert not residual_conformed(np.asarray([[0.0], [1.0], [0.0], [1.0]]), np.ones(4), 0.5)


def test_intervals() -> None:
    values = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
    weights = np.ones(5)
    assert weighted_mean(values, weights) == pytest.approx(0.6)
    assert weighted_variance(values, weights) == pytest.approx(0.24)
    assert weighted_mean(values, np.zeros(5)) == 0.0
    interval = interval_from_radius(0.5, 0.1, 0.05, "hoeffding")
    assert interval.width == pytest.approx(0.2)
    assert lower_confidence_bound(0.5, 0.1) == pytest.approx(0.4)
    assert upper_confidence_bound(0.5, 0.1) == pytest.approx(0.6)
    assert hoeffding_radius(0.0, 0.05, 1.0) == float("inf")
    assert empirical_bernstein_radius(values, weights, 0.0, 0.05, 1.0) == float("inf")
    rung = interval_for_rung(values, weights, 5.0, 0.05, 1.0, "hoeffding", 1)
    assert rung.lower <= rung.point <= rung.upper
    with pytest.raises(ValueError):
        interval_for_rung(values, weights, 5.0, 0.05, 1.0, "nope", 1)
    ratio = reduces_to_hoeffding(values, weights, 5.0, 0.05, 1.0, 1)
    assert ratio > 0.0


def test_support_floor_and_eligibility() -> None:
    strata = np.asarray(["I"] * 40 + ["II"] * 40, dtype="<U4")
    rng = np.random.default_rng(4)
    indicators = np.zeros((80, 2), dtype=np.float64)
    indicators[:, 0] = (rng.random(80) < 0.7).astype(np.float64)
    indicators[:, 1] = (rng.random(80) < 0.2).astype(np.float64)
    indicators[:, 0] = np.maximum(indicators[:, 0], indicators[:, 1])
    propensity = rng.uniform(0.05, 0.9, size=80)
    mass = stratum_rung_mass(strata, indicators, propensity)
    assert set(mass) == {("I", 0), ("I", 1), ("II", 0), ("II", 1)}
    report = evaluate_floor("s", strata, indicators, propensity, 0.02)
    assert report.eligible == (report.failing_count() == 0)
    assert len(report.entries) == 4
    assert math.isfinite(worst_margin(report))
    assert support_covered_strata(report)
    assert eligible_strategies(np.asarray(["s"], dtype="<U8"), {"s": report}).shape[0] == int(
        report.eligible
    )
    assert floor_effect_on_safety(0.05, 0.02) == pytest.approx(0.03)


def test_selection_rule_paths() -> None:
    estimates = (
        StrategyEstimate("a", 0.9, 0.06, 0.2, 0.02, 90.0, 0.4),
        StrategyEstimate("b", 0.7, 0.03, 0.1, 0.01, 120.0, 0.5),
        StrategyEstimate("c", 0.95, 0.02, 0.4, 0.01, 80.0, 0.3, eligible=False),
    )
    relaxed = select_constrained(estimates, risk_margin=0.05)
    assert relaxed.selected in {"a", "b"}
    tight = select_constrained(estimates, risk_margin=-1.0)
    assert tight.abstained
    assert "risk margin" in tight.reason
    all_ineligible = (StrategyEstimate("a", 0.9, 0.06, 0.2, 0.02, 90.0, 0.4, eligible=False),)
    floored = select_constrained(all_ineligible, risk_margin=0.05)
    assert floored.abstained and "floor" in floored.reason
    unconstrained = select_unconstrained(estimates)
    assert unconstrained.selected == "c" and not unconstrained.constrained
    empty = select_unconstrained(())
    assert empty.abstained


def test_rule_level_helpers() -> None:
    assert selection_accuracy(np.asarray(["a", "b"]), np.asarray(["a", "a"])) == pytest.approx(0.5)
    assert selection_accuracy(np.zeros(0), np.zeros(0)) == 0.0
    assert regret(0.4, 0.6) == pytest.approx(0.2)
    assert regret(0.8, 0.6) == 0.0
    assert abstention_accuracy(
        np.asarray([True, False, True]), np.asarray([True, False, False])
    ) == pytest.approx(0.5)
    assert abstention_accuracy(np.zeros(2, dtype=bool), np.zeros(2, dtype=bool)) == 0.0
    assert parity_holds(0.552, 0.556, 1.0)
    assert not parity_holds(0.500, 0.556, 1.0)
    assert relative_utility_gain(0.6, 0.5) == pytest.approx(0.2)
    gap = net_benefit_gap(0.05, 0.03, 0.0)
    assert gap["over_treat_all"] == pytest.approx(0.02)
    assert gap["beats_both"] == 1.0


def test_sweep_grid_and_headline(context: object) -> None:
    config = context.config.sweep
    grid = sweep_grid(config)
    assert len(grid) == len(config.clipping_values) * len(config.pessimism_values)
    cells = tuple(
        SweepCell(
            stratum=f"s{index}",
            axis="anatomical_stage",
            clipping_label=clipping_label,
            clipping=clipping,
            pessimism_label=pessimism_label,
            pessimism=pessimism,
            relative_error=0.2 - 0.02 * index,
            effective_sample_size=100.0 + index,
            overlap_coefficient=0.4,
            selection_accuracy=0.5 + 0.01 * index,
        )
        for index, (clipping, pessimism, clipping_label, pessimism_label) in enumerate(grid)
    )
    headline = conservative_headline(cells, config)
    assert headline.conservative_gain > 0.0
    assert headline.ratio != 0.0
    assert stratum_labels(cells) == tuple(sorted({cell.stratum for cell in cells}))
    assert cell_array(cells, "clipping").shape[0] == len(cells)
    assert isinstance(sweep_monotone_in_clipping(cells), bool)
    assert isinstance(sweep_monotone_in_pessimism(cells), bool)
    with pytest.raises(ValueError):
        conservative_headline((), config)


def test_sweep_scaling_helpers() -> None:
    assert pessimism_radius_scale(0.0, 0.2) == pytest.approx(0.2)
    assert pessimism_radius_scale(0.5, 0.2) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        pessimism_radius_scale(2.0, 0.2)
    assert clipping_effect(0.0, 0.3) == pytest.approx(0.3)
    assert clipping_effect(0.25, 0.3) < 0.3


def test_effective_sample_helpers() -> None:
    sizes = per_rung_effective_sample_size(
        np.asarray([1.0, 1.0, 2.0]), np.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    )
    assert sizes.shape == (2,)
    assert effective_sample_size(np.asarray([2.0, 2.0])) == pytest.approx(2.0)
    assert overlap_coefficient(np.asarray([0.4]), np.asarray([0.6])) == pytest.approx(0.4)
    assert relative_error(0.9, 1.0) == pytest.approx(0.1)
