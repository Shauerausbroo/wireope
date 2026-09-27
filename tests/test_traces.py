"""Device event integration, registration and the excursion observable."""

from __future__ import annotations

import numpy as np
import pytest

from wireope.config import KinematicsConfig, RegistrationConfig, UncertaintyConfig
from wireope.geometry.envelope import GridSpec
from wireope.geometry.vessel import VesselGeometry, radius_profile, straight_centreline
from wireope.schema import EventRecord
from wireope.traces.events import (
    action_norm,
    compress_event_stream,
    event_histogram,
    integrate_event_stream,
)
from wireope.traces.excursion import (
    build_trace,
    cumulative_excursion,
    dwell_integral,
    dwell_weighted_peak,
    excursion_peak_from_trace,
    propagating_band,
    rung_indicators,
    surface_integral,
    surface_integral_at_step,
)
from wireope.traces.registration import (
    align_trajectory,
    arc_length_parameterisation,
    fit_similarity,
    nearest_centreline_points,
    transform_trajectory,
)


def kinematics() -> KinematicsConfig:
    return KinematicsConfig(
        events=("advance", "retract", "rotate_clockwise", "rotate_counterclockwise", "dwell"),
        timestep_s=0.25,
        advance_step_mm=1.0,
        rotation_deg_per_event=15.0,
        max_steps=100,
        curvature_regularisation=0.0,
        tip_mean_absolute_error_mm=4.44,
    )


def events_of(kind: str, count: int, magnitude: float = 1.0) -> list[EventRecord]:
    return [EventRecord(step=index, kind=kind, magnitude=magnitude) for index in range(count)]


def test_advance_only_travels_the_closed_form_distance() -> None:
    trajectory = integrate_event_stream(events_of("advance", 8, 2.0), kinematics())
    assert trajectory.path_length_mm == pytest.approx(14.0)
    assert trajectory.steps == 8


def test_retract_moves_backward() -> None:
    mixed: list[EventRecord] = [
        EventRecord(step=0, kind="advance", magnitude=3.0),
        EventRecord(step=1, kind="retract", magnitude=1.0),
    ]
    trajectory = integrate_event_stream(mixed, kinematics())
    assert trajectory.positions_mm[-1][0] == pytest.approx(2.0)


def test_rotation_changes_the_heading() -> None:
    mixed: list[EventRecord] = [
        EventRecord(step=0, kind="rotate_counterclockwise", magnitude=2.0),
        EventRecord(step=1, kind="rotate_clockwise", magnitude=1.0),
        EventRecord(step=2, kind="advance", magnitude=1.0),
    ]
    trajectory = integrate_event_stream(mixed, kinematics())
    assert abs(float(trajectory.headings_rad[0])) > 0.0
    assert trajectory.rotation_turns > 0.0


def test_unsupported_event_is_rejected() -> None:
    with pytest.raises(ValueError):
        integrate_event_stream([EventRecord(step=0, kind="teleport", magnitude=1.0)], kinematics())


def test_empty_stream_is_rejected() -> None:
    with pytest.raises(ValueError):
        integrate_event_stream([], kinematics())


def test_step_budget_and_histogram() -> None:
    stream = events_of("advance", 40)
    compressed = compress_event_stream(stream, 10)
    assert len(compressed) == 10
    assert compress_event_stream(stream, 100) == stream
    histogram = event_histogram(stream)
    assert histogram["advance"] == 40


def test_action_norm_scales_with_radius() -> None:
    trajectory = integrate_event_stream(events_of("advance", 5, 1.0), kinematics())
    assert action_norm(trajectory, 1.0) == pytest.approx(4.0)
    assert action_norm(trajectory, 2.0) == pytest.approx(2.0)


def test_similarity_fit_recovers_a_known_transform() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(40, 2))
    angle = 0.3
    expected = np.asarray(
        [
            [1.2 * np.cos(angle), -1.2 * np.sin(angle)],
            [1.2 * np.sin(angle), 1.2 * np.cos(angle)],
        ]
    )
    target = source @ expected.T + np.asarray([1.0, -2.0])
    fitted = fit_similarity(source, target)
    assert fitted.scale == pytest.approx(1.2, abs=1e-8)
    assert np.allclose(fitted.apply(source), target, atol=1e-8)


def test_similarity_fit_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fit_similarity(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        fit_similarity(np.zeros((1, 2)), np.zeros((1, 2)))


def test_alignment_reduces_the_residual() -> None:
    centreline = straight_centreline((0.0, 0.0, -2.0), 14.0, 29)
    shifted = centreline + np.asarray([0.7, -0.5, 0.0])
    before = float(np.mean(np.linalg.norm(shifted[:, :2] - centreline[:, :2], axis=1)))
    result = align_trajectory(shifted, centreline, RegistrationConfig("centreline", "rigid", 60, 1e-4))
    assert result.residual_mm < before
    assert result.iterations >= 1


def test_transform_and_parameterisation() -> None:
    centreline = straight_centreline((0.0, 0.0, 0.0), 10.0, 21)
    result = align_trajectory(centreline, centreline, RegistrationConfig("centreline", "rigid", 20, 1e-3))
    moved = transform_trajectory(centreline, result)
    assert moved.shape == centreline.shape
    lengths = arc_length_parameterisation(centreline)
    assert lengths.shape[0] == centreline.shape[0]
    assert lengths[-1] == pytest.approx(10.0)
    assert arc_length_parameterisation(np.zeros((1, 3))).shape == (1,)


def test_nearest_centreline_points() -> None:
    centreline = straight_centreline((0.0, 0.0, 0.0), 4.0, 5)
    query = centreline[:2] + np.asarray([0.01, 0.0, 0.0])
    matched = nearest_centreline_points(query, centreline)
    assert matched.shape == (2, 2)


def test_propagating_band_modes() -> None:
    quadrature = propagating_band(UncertaintyConfig(4.44, 0.10, "quadrature", 1.0, "trapezoid"))
    additive = propagating_band(UncertaintyConfig(4.44, 0.10, "additive", 1.0, "trapezoid"))
    assert quadrature.combined_mm == pytest.approx(np.hypot(4.44, 0.10))
    assert additive.combined_mm == pytest.approx(4.54)
    with pytest.raises(ValueError):
        propagating_band(UncertaintyConfig(4.44, 0.10, "geometric", 1.0, "trapezoid"))


def vessel_envelope() -> object:
    geometry = VesselGeometry(
        centreline_mm=straight_centreline((0.0, 0.0, -2.0), 14.0, 29),
        radius_mm=radius_profile(29, 1.6, 1.0, 0.5, 0.2),
    )
    grid = GridSpec(shape=(20, 20, 32), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-5.0, -5.0, -2.0))
    return geometry, geometry.as_envelope(grid)


def test_excursion_series_and_dwell() -> None:
    geometry, envelope = vessel_envelope()
    positions = geometry.centreline_mm
    dwell = np.full(positions.shape[0], 0.25)
    trace = build_trace(0, envelope, positions, dwell)
    assert trace.excursion_mm.shape[0] == positions.shape[0]
    assert trace.max_excursion_mm >= 0.0
    assert excursion_peak_from_trace(trace) == trace.max_excursion_mm
    assert dwell_weighted_peak(trace) >= 0.0
    assert cumulative_excursion(trace) >= 0.0
    assert trace.dwell_integral_mm_s >= 0.0


def test_dwell_integral_is_the_product_sum() -> None:
    excursion = np.asarray([1.0, 2.0, 3.0])
    dwell = np.asarray([0.5, 0.5, 1.0])
    assert dwell_integral(excursion, dwell) == pytest.approx(4.5)
    assert dwell_integral(np.zeros(0), np.zeros(0)) == 0.0


def test_surface_integrals() -> None:
    geometry, envelope = vessel_envelope()
    positions = geometry.centreline_mm
    assert surface_integral(envelope, positions, 0.5) > 0.0
    assert surface_integral(envelope, positions[:1], 0.5) == 0.0
    assert surface_integral_at_step(envelope, positions) >= 0.0


def test_rung_indicators_nest_downward() -> None:
    tau = (5.0, 10.0, 15.0)
    indicators = rung_indicators(12.0, tau)
    assert indicators.tolist() == [1.0, 1.0, 0.0]
    assert rung_indicators(20.0, tau).tolist() == [1.0, 1.0, 1.0]
    assert rung_indicators(1.0, tau).tolist() == [0.0, 0.0, 0.0]
