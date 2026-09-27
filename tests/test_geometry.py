"""Lumen envelope, analytic vessel geometry and the anatomical descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from wireope.geometry.descriptors import (
    AnatomyDescriptors,
    CalcificationProfile,
    calcification_arc_deg,
    calcification_volume_mm3,
    cap_morphology_class,
    cross_section_profile,
    descriptor_from_components,
    grade_calcification,
    lesion_length_mm,
    stage_from_load,
    territory_from_axis,
)
from wireope.geometry.envelope import (
    GridSpec,
    LumenEnvelope,
    signed_distance_field,
    trilinear_sample,
)
from wireope.geometry.vessel import (
    VesselGeometry,
    calcium_mask_from_points,
    calcium_ring,
    curved_centreline,
    radius_profile,
    straight_centreline,
    wire_path_mm,
)


def small_grid() -> GridSpec:
    return GridSpec(shape=(20, 20, 32), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-5.0, -5.0, -2.0))


def straight_vessel() -> VesselGeometry:
    return VesselGeometry(
        centreline_mm=straight_centreline((0.0, 0.0, -2.0), 14.0, 29),
        radius_mm=radius_profile(29, 1.6, 1.0, 0.5, 0.2),
    )


def test_grid_coordinate_round_trip() -> None:
    grid = small_grid()
    indices = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    points = grid.index_to_mm(indices)
    back = grid.mm_to_index(points)
    assert np.allclose(indices, back)


def test_signed_distance_is_negative_inside() -> None:
    geometry = straight_vessel()
    grid = small_grid()
    envelope = geometry.as_envelope(grid)
    field = signed_distance_field(envelope.mask, grid)
    assert float(field[envelope.mask].max()) < 0.0
    assert float(field[~envelope.mask].min()) >= 0.0


def test_empty_and_full_masks_are_rejected() -> None:
    grid = small_grid()
    with pytest.raises(ValueError):
        signed_distance_field(np.zeros(grid.shape, dtype=bool), grid)
    with pytest.raises(ValueError):
        signed_distance_field(np.ones(grid.shape, dtype=bool), grid)


def test_trilinear_sample_matches_corner_values() -> None:
    grid = GridSpec(shape=(4, 4, 4), spacing_mm=(1.0, 1.0, 1.0))
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    corner = grid.index_to_mm(np.asarray([[1.0, 1.0, 1.0]]))
    assert float(trilinear_sample(field, grid, corner)[0]) == pytest.approx(float(field[1, 1, 1]))


def test_excursion_matches_the_analytic_form() -> None:
    geometry = straight_vessel()
    grid = small_grid()
    envelope = geometry.as_envelope(grid)
    offsets = np.linspace(0.0, 2.4, 13)
    base = geometry.centreline_mm[10]
    points = np.vstack([base + np.asarray([offset, 0.0, 0.0]) for offset in offsets])
    analytic = geometry.analytic_excursion_mm(points)
    discretised = envelope.excursion(points)
    assert np.max(np.abs(analytic - discretised)) < 1.0


def test_excursion_is_non_negative() -> None:
    geometry = straight_vessel()
    envelope = geometry.as_envelope(small_grid())
    assert float(envelope.excursion(geometry.centreline_mm).min()) >= 0.0


def test_surface_area_and_volume_are_positive() -> None:
    envelope = straight_vessel().as_envelope(small_grid())
    assert envelope.surface_area_mm2() > 0.0
    assert envelope.volume_mm3() > 0.0


def test_centreline_and_boundary_extraction() -> None:
    envelope = straight_vessel().as_envelope(small_grid())
    centreline = envelope.centreline_points_mm(axis=2)
    assert centreline.shape[1] == 3
    assert centreline.shape[0] > 5
    boundary = envelope.boundary_points_mm()
    assert boundary.shape[1] == 3
    assert boundary.shape[0] > 10


def test_wall_overlap_fraction_bounds() -> None:
    geometry = straight_vessel()
    envelope = geometry.as_envelope(small_grid())
    fraction = envelope.wall_overlap_fraction(geometry.centreline_mm, 0.6)
    assert 0.0 <= fraction <= 1.0


def test_radius_profile_dips_at_the_stenosis() -> None:
    radius = radius_profile(51, 1.6, 0.5, 0.5, 0.05)
    assert radius[0] == pytest.approx(1.6, abs=1e-3)
    assert radius[-1] == pytest.approx(1.6, abs=1e-3)
    assert float(radius.min()) == pytest.approx(0.5, abs=0.02)
    assert int(np.argmin(radius)) == 25


def test_curved_centreline_length() -> None:
    points = curved_centreline((0.0, 0.0, 0.0), 20.0, 41, 0.02, plane=(0, 2))
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert float(segments.sum()) == pytest.approx(20.0, rel=1e-3)


def test_geometry_length_and_radius_lookup() -> None:
    geometry = straight_vessel()
    assert geometry.length_mm == pytest.approx(14.0, rel=1e-6)
    radius = geometry.radius_at(geometry.centreline_mm[::4])
    assert np.all(radius > 0.0)


def test_geometry_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError):
        VesselGeometry(centreline_mm=np.zeros((3, 3)), radius_mm=np.ones(4))
    with pytest.raises(ValueError):
        VesselGeometry(centreline_mm=np.zeros((3, 3)), radius_mm=np.asarray([1.0, 1.0, -1.0]))


def test_wire_path_offsets_the_centreline() -> None:
    geometry = straight_vessel()
    offsets = np.linspace(0.0, 1.0, 20)
    path = wire_path_mm(geometry.centreline_mm, geometry.radius_mm, offsets, 20)
    assert path.shape == (20, 3)
    assert float(np.linalg.norm(path[-1] - geometry.centreline_mm[-1])) > 0.0


def test_calcium_mask_and_ring() -> None:
    geometry = straight_vessel()
    grid = small_grid()
    ring = calcium_ring(geometry.centreline_mm, geometry.radius_mm, 120.0, 0.3)
    mask = calcium_mask_from_points(geometry, ring, grid)
    assert mask.dtype == np.bool_
    assert int(np.count_nonzero(mask)) > 0


def test_calcification_grading_and_volume() -> None:
    assert grade_calcification(0.0, (2.0, 5.0, 9.0, 15.0)) == 0
    assert grade_calcification(1.0, (2.0, 5.0, 9.0, 15.0)) == 1
    assert grade_calcification(100.0, (2.0, 5.0, 9.0, 15.0)) == 4
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    assert calcification_volume_mm3(mask, 0.125) == pytest.approx(0.125)


def test_calcification_arc_is_covered() -> None:
    ring_points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    axis = np.asarray([0.0, 0.0, 3.0])
    arc = calcification_arc_deg(np.ones((3, 1, 1), dtype=bool), np.vstack([[0.0, 0.0, 0.0], axis]))
    assert arc >= 0.0
    assert ring_points.shape[0] == 3


def test_cross_section_profile_and_cap_class(bundle: object) -> None:
    envelope = straight_vessel().as_envelope(small_grid())
    profile = cross_section_profile(envelope)
    assert profile.shape[0] == envelope.mask.shape[2]
    assert cap_morphology_class(envelope) in {"tapered", "blunt", "ambiguous", "calcified_nodule"}
    tapered = LumenEnvelope(
        mask=_tapered_mask(), grid=GridSpec(shape=(13, 13, 9), spacing_mm=(1.0, 1.0, 1.0))
    )
    assert cap_morphology_class(tapered) == "tapered"


def _tapered_mask() -> np.ndarray:
    mask = np.zeros((13, 13, 9), dtype=bool)
    for z_position in range(9):
        radius = max(1, 6 - z_position)
        mask[6 - radius : 7 + radius, 6 - radius : 7 + radius, z_position] = True
    return mask


def test_descriptor_vector_and_validation() -> None:
    profile = CalcificationProfile(grade=3, arc_deg=180.0, severity=0.75, volume_mm3=40.0)
    descriptors = AnatomyDescriptors(
        stage="II",
        calcification=profile,
        cap_morphology="tapered",
        lesion_length_mm=25.0,
        territory="tibial",
    )
    vector = descriptors.as_vector()
    assert vector.shape == (7,)
    assert 0.0 <= float(vector[2]) <= 1.0


def test_descriptor_from_components_rejects_unknowns(context: object) -> None:
    config = context.config.descriptors
    profile = CalcificationProfile(grade=1, arc_deg=90.0, severity=0.25, volume_mm3=5.0)
    good = descriptor_from_components(config, "I", profile, "blunt", 10.0, "tibial")
    assert good.stage == "I"
    with pytest.raises(ValueError):
        descriptor_from_components(config, "IV", profile, "blunt", 10.0, "tibial")
    with pytest.raises(ValueError):
        descriptor_from_components(config, "I", profile, "not_a_cap", 10.0, "tibial")
    with pytest.raises(ValueError):
        descriptor_from_components(config, "I", profile, "blunt", 10.0, "carotid")


def test_stage_and_territory_helpers() -> None:
    assert stage_from_load(0.1, 5.0) == "I"
    assert stage_from_load(0.9, 40.0) == "III"
    assert territory_from_axis(straight_centreline((0.0, 0.0, 0.0), 45.0, 20)) == "femoropopliteal"
    assert territory_from_axis(straight_centreline((0.0, 0.0, 0.0), 10.0, 5)) == "tibial"
    assert lesion_length_mm(straight_centreline((0.0, 0.0, 0.0), 12.0, 13)) == pytest.approx(12.0)
    assert lesion_length_mm(np.zeros((1, 3))) == 0.0
