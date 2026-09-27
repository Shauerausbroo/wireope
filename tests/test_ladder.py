"""Nested severity ladder, its weights, its resolution criterion and the truncation term."""

from __future__ import annotations

import numpy as np
import pytest

from wireope.config import LadderConfig, LadderWeightsConfig
from wireope.laddering.resolution import (
    adjacent_separations,
    band_from_tolerances,
    check_resolution,
    interior_optimum_depth,
    max_resolvable_depth,
    truncation_term,
)
from wireope.laddering.severity import (
    build_ladder,
    conditional_within_band,
    ladder_for_depth,
    rungs_from_injury_flags,
)
from wireope.laddering.weights import (
    build_weights,
    geometric_weights,
    graded_value,
    linear_weights,
    uniform_weights,
    weight_ratio,
)


def ladder_config() -> LadderConfig:
    return LadderConfig(
        depth=4,
        additional_depths=(3, 5, 6),
        tau_mm=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
        tau_inj_mm=20.0,
        observed_injury_defines_top_rung=True,
        conditional_frequency_band=(0.15, 0.60),
        resolution_rule="adjacent_separation_exceeds_propagating_band",
        resolution_band_mm=4.4411,
    )


def weights_config() -> LadderWeightsConfig:
    return LadderWeightsConfig(scheme="geometric", value=0.6, normalise=True, require_non_negative=True)


def test_ladder_validates_its_grid() -> None:
    with pytest.raises(ValueError):
        build_ladder(
            LadderConfig(4, (), (5.0, 10.0, 5.0, 20.0), 20.0, True, (0.1, 0.5), "rule", 4.0),
            weights_config(),
        )
    with pytest.raises(ValueError):
        build_ladder(
            LadderConfig(4, (), (5.0, 10.0, 10.0, 18.0), 20.0, True, (0.1, 0.5), "rule", 4.0),
            weights_config(),
        )
    with pytest.raises(ValueError):
        ladder_for_depth(ladder_config(), weights_config(), 9)


def test_ladder_depth_and_top_rung() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    assert ladder.depth == 4
    assert ladder.tau_mm[-1] == 20.0
    assert ladder.nesting_holds(np.asarray([3.0, 12.0, 22.0]))


def test_indicators_match_a_manual_loop() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    peaks = np.asarray([1.0, 7.0, 13.0, 18.0, 25.0])
    matrix = ladder.indicator_matrix(peaks)
    for row, peak in enumerate(peaks):
        for column, threshold in enumerate(ladder.tau_mm):
            assert matrix[row, column] == float(peak >= threshold)


def test_graded_value_and_risk() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    peaks = np.asarray([0.0, 8.0, 12.0, 25.0])
    weights = np.asarray(ladder.weights)
    manual = ladder.indicator_matrix(peaks) @ weights
    assert np.allclose(ladder.graded_value(peaks), manual)
    assert np.allclose(ladder.risk(peaks), ladder.indicator_matrix(peaks)[:, -1])


def test_conditional_frequencies_and_rung_sizes() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    peaks = np.linspace(0.0, 26.0, 200)
    frequencies = ladder.conditional_frequencies(peaks)
    assert frequencies.shape[0] == ladder.depth
    assert np.all(frequencies >= 0.0)
    assert np.all(frequencies <= 1.0)
    sizes = ladder.rung_sizes(peaks)
    assert sizes.shape[0] == ladder.depth
    assert sizes[-1] <= sizes[0]
    assert ladder.rarest_rung_size(peaks) == int(sizes[-1])


def test_conditional_band_helper() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    inside = np.asarray([0.8, 0.4, 0.4, 0.4, 0.4])
    outside = np.asarray([0.8, 0.9, 0.9, 0.9, 0.9])
    assert conditional_within_band(ladder, inside, (0.15, 0.60))
    assert not conditional_within_band(ladder, outside, (0.15, 0.60))


def test_reconstructed_risk_is_the_conditional_product() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    conditionals = np.asarray([0.5, 0.4, 0.3, 0.2])
    assert ladder.reconstruct_risk(conditionals) == pytest.approx(0.012)


def test_rungs_from_injury_flags_returns_the_requested_ratio() -> None:
    ratios = rungs_from_injury_flags(np.ones(4), np.zeros(4), 0.55)
    assert ratios.shape == (4,)
    assert np.allclose(ratios, 0.55)


def test_weight_schemes() -> None:
    assert geometric_weights(3, 0.5).tolist() == [1.0, 0.5, 0.25]
    assert linear_weights(3).tolist() == [1.0, 2.0, 3.0]
    assert uniform_weights(4).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        geometric_weights(0, 0.5)
    with pytest.raises(ValueError):
        geometric_weights(3, 1.5)


def test_weight_building_normalises() -> None:
    weights = build_weights(4, weights_config())
    assert sum(weights) == pytest.approx(1.0)
    assert weight_ratio(np.asarray(weights)) > 1.0
    raw = build_weights(4, LadderWeightsConfig("linear", 1.0, False, True))
    assert sum(raw) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        build_weights(3, LadderWeightsConfig("nope", 1.0, True, True))


def test_graded_value_helper() -> None:
    indicators = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.asarray([0.5, 0.3, 0.2])
    assert graded_value(indicators, weights).tolist() == pytest.approx([0.8, 0.5])


def test_resolution_criterion() -> None:
    ladder = build_ladder(ladder_config(), weights_config())
    report = check_resolution(ladder, 4.4411)
    assert report.resolvable
    assert report.max_depth == ladder.depth
    assert max_resolvable_depth((5.0, 6.0, 20.0), 4.4411) == 1
    assert max_resolvable_depth((5.0, 10.0, 20.0), 4.4411) == 3
    assert adjacent_separations((5.0, 10.0, 20.0)).tolist() == [5.0, 10.0]
    assert band_from_tolerances(4.44, 0.10, "quadrature") == pytest.approx(np.hypot(4.44, 0.10))
    assert band_from_tolerances(4.44, 0.10, "additive") == pytest.approx(4.54)
    with pytest.raises(ValueError):
        band_from_tolerances(4.44, 0.10, "product")


def test_unresolvable_ladder_is_reported() -> None:
    ladder = build_ladder(
        LadderConfig(3, (), (5.0, 5.5, 20.0), 20.0, True, (0.1, 0.5), "rule", 4.4411),
        weights_config(),
    )
    report = check_resolution(ladder, 4.4411)
    assert not report.resolvable
    assert "below the propagating band" in report.reason


def test_truncation_term_shrinks_with_depth() -> None:
    shallow = truncation_term(2, 3.0, 20.0)
    deep = truncation_term(6, 3.0, 20.0)
    assert shallow > deep > 0.0
    with pytest.raises(ValueError):
        truncation_term(2, 3.0, 0.0)


def test_interior_optimum_depth() -> None:
    counts = np.asarray([900.0, 400.0, 150.0, 60.0, 22.0, 8.0], dtype=np.float64)
    depth = interior_optimum_depth(counts, 0.05, 1.0, 3.0)
    assert 1 <= depth <= counts.size
    assert interior_optimum_depth(np.asarray([10.0, 0.0]), 0.05, 1.0, 1.0) == 1
    with pytest.raises(ValueError):
        interior_optimum_depth(np.zeros(0, dtype=np.float64), 0.05, 1.0, 1.0)
