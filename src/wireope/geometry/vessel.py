"""Analytic vessel geometry used to give the reconstructed excursion a known truth.

The centreline is a parametric curve and the lumen radius is a profile along it, so the
distance from any point to the lumen surface has a closed form. The discretised distance
field of the envelope module is checked against this form, which keeps the excursion
observable falsifiable without a cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.geometry.envelope import GridSpec, LumenEnvelope, signed_distance_field
from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class VesselGeometry:
    centreline_mm: NDArray[np.float64]
    radius_mm: NDArray[np.float64]
    calcium_mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.centreline_mm.ndim != 2 or self.centreline_mm.shape[1] != 3:
            raise ValueError("centreline must be an (n, 3) array of millimetre coordinates")
        if self.radius_mm.shape[0] != self.centreline_mm.shape[0]:
            raise ValueError("radius profile must have one value per centreline point")
        if np.any(self.radius_mm <= 0.0):
            raise ValueError("lumen radius must be positive along the whole centreline")

    @property
    def length_mm(self) -> float:
        if self.centreline_mm.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.centreline_mm, axis=0), axis=1)))

    def segment_projections(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Nearest distance to each centreline segment and the parameter at the foot."""
        query = np.atleast_2d(points)
        starts = self.centreline_mm[:-1]
        ends = self.centreline_mm[1:]
        directions = ends - starts
        lengths_sq = np.sum(directions * directions, axis=1)
        lengths_sq = np.maximum(lengths_sq, EPS)
        offsets = query[:, None, :] - starts[None, :, :]
        projection = np.sum(offsets * directions[None, :, :], axis=2) / lengths_sq[None, :]
        clipped = np.clip(projection, 0.0, 1.0)
        closest = starts[None, :, :] + clipped[:, :, None] * directions[None, :, :]
        distances = np.linalg.norm(query[:, None, :] - closest, axis=2)
        return distances, clipped

    def nearest_segment(
        self, points: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        distances, clipped = self.segment_projections(points)
        index = np.argmin(distances, axis=1)
        rows = np.arange(distances.shape[0])
        return index.astype(np.int64), clipped[rows, index]

    def radius_at(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Lumen radius at the projection of each query point onto the centreline."""
        index, fraction = self.nearest_segment(points)
        local = (1.0 - fraction) * self.radius_mm[index] + fraction * self.radius_mm[index + 1]
        return np.asarray(local, dtype=np.float64)

    def signed_distance_mm(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Exact signed distance to the lumen surface, positive outside.

        The tube is the union of spheres along the centreline, so the distance to its
        surface is the distance to the centreline minus the local radius. A point beyond
        the end caps is measured to the cap sphere, which keeps the formula exact.
        """
        query = np.atleast_2d(points)
        distances, _ = self.segment_projections(query)
        nearest = np.min(distances, axis=1)
        cap_start = np.linalg.norm(query - self.centreline_mm[0], axis=1) - self.radius_mm[0]
        cap_end = np.linalg.norm(query - self.centreline_mm[-1], axis=1) - self.radius_mm[-1]
        lateral = nearest - self.radius_at(query)
        beyond = np.minimum(lateral, np.minimum(cap_start, cap_end))
        extended = bool(
            np.max(np.linalg.norm(np.diff(self.centreline_mm, axis=0), axis=1))
            > 2.0 * float(np.min(self.radius_mm))
        )
        if extended:
            return np.asarray(lateral, dtype=np.float64)
        return np.asarray(beyond, dtype=np.float64)

    def analytic_excursion_mm(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(self.signed_distance_mm(points), 0.0, None)

    def rasterise(self, grid: GridSpec) -> NDArray[np.bool_]:
        indices = np.indices(grid.shape, dtype=np.float64)
        stacked = np.stack([indices[axis].ravel() for axis in range(3)], axis=1)
        points = grid.index_to_mm(stacked)
        distance = self.signed_distance_mm(points)
        mask = (distance <= 0.0).reshape(grid.shape)
        return np.asarray(mask, dtype=np.bool_)

    def as_envelope(self, grid: GridSpec) -> LumenEnvelope:
        mask = self.rasterise(grid)
        distance = signed_distance_field(mask, grid)
        return LumenEnvelope(mask=mask, grid=grid, _distance=distance)


def straight_centreline(
    start_mm: tuple[float, float, float], length_mm: float, steps: int
) -> NDArray[np.float64]:
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    offsets = np.linspace(0.0, length_mm, steps)
    base = np.asarray(start_mm, dtype=np.float64)
    return np.asarray(base[None, :] + offsets[:, None] * axis[None, :], dtype=np.float64)


def curved_centreline(
    start_mm: tuple[float, float, float],
    length_mm: float,
    steps: int,
    curvature_mm_inv: float,
    plane: tuple[int, int] = (0, 2),
) -> NDArray[np.float64]:
    """In-plane circular arc, the simplest curved centreline with a closed-form length."""
    radius = 1.0 / max(curvature_mm_inv, EPS)
    sweep = length_mm / radius
    angle = np.linspace(0.0, sweep, steps)
    first = np.zeros(steps, dtype=np.float64)
    second = np.zeros(steps, dtype=np.float64)
    first[:] = radius * np.sin(angle)
    second[:] = radius * (1.0 - np.cos(angle))
    points = np.zeros((steps, 3), dtype=np.float64)
    points[:, plane[0]] = first
    points[:, plane[1]] = second
    points += np.asarray(start_mm, dtype=np.float64)[None, :]
    return points


def radius_profile(
    steps: int,
    nominal_mm: float,
    stenosis_mm: float,
    stenosis_centre: float,
    stenosis_width: float,
) -> NDArray[np.float64]:
    """Nominal radius with one smooth stenosis, the lesion the wire must cross."""
    position = np.linspace(0.0, 1.0, steps)
    dip = np.exp(-0.5 * ((position - stenosis_centre) / stenosis_width) ** 2)
    radius = nominal_mm - (nominal_mm - stenosis_mm) * dip
    return np.asarray(radius, dtype=np.float64)


def calcium_ring(
    centreline: NDArray[np.float64],
    radius_mm: NDArray[np.float64],
    arc_deg: float,
    thickness_mm: float,
    start_fraction: float = 0.2,
    end_fraction: float = 0.8,
) -> NDArray[np.float64]:
    """Centreline-relative shell segments approximating a calcified plaque arc."""
    count = centreline.shape[0]
    low = int(start_fraction * count)
    high = max(low + 1, int(end_fraction * count))
    centres = centreline[low:high]
    radii = radius_mm[low:high]
    samples = int(max(8, round(arc_deg / 10.0)))
    angles = np.linspace(0.0, np.deg2rad(arc_deg), samples)
    points: list[NDArray[np.float64]] = []
    for index in range(centres.shape[0]):
        axis = _local_axis(centreline, low + index)
        first, second = _orthonormal_basis(axis)
        for angle in angles:
            direction = np.cos(angle) * first + np.sin(angle) * second
            points.append(centres[index] + (radii[index] + 0.5 * thickness_mm) * direction)
    return np.vstack(points)


def calcium_mask_from_points(
    geometry: VesselGeometry, points_mm: NDArray[np.float64], grid: GridSpec
) -> NDArray[np.bool_]:
    index = np.rint(grid.mm_to_index(points_mm)).astype(np.int64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    inside = np.all((index >= 0) & (index < shape[None, :]), axis=1)
    mask = np.zeros(grid.shape, dtype=bool)
    if inside.any():
        kept = index[inside]
        mask[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    return mask


def _local_axis(centreline: NDArray[np.float64], index: int) -> NDArray[np.float64]:
    low = max(index - 1, 0)
    high = min(index + 1, centreline.shape[0] - 1)
    direction = np.asarray(centreline[high] - centreline[low], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    result: NDArray[np.float64] = direction / norm
    return result


def _orthonormal_basis(axis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    first = reference - float(np.dot(reference, axis)) * axis
    first = first / max(float(np.linalg.norm(first)), EPS)
    second = np.cross(axis, first)
    return first, second


def wire_path_mm(
    centreline: NDArray[np.float64],
    radii: NDArray[np.float64],
    lateral_offsets: NDArray[np.float64],
    steps: int,
) -> NDArray[np.float64]:
    """Reconstructed wire positions: the centreline displaced laterally out of the lumen.

    The displacement profile is provided by the caller, so a crossing that stays inside
    the envelope and one that breaches it differ only in that profile.
    """
    positions = np.linspace(0.0, centreline.shape[0] - 1, steps)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, centreline.shape[0] - 1)
    fraction = positions - low
    base = (1.0 - fraction)[:, None] * centreline[low] + fraction[:, None] * centreline[high]
    local_radius = (1.0 - fraction) * radii[low] + fraction * radii[high]
    axes = np.vstack([_local_axis(centreline, int(index)) for index in low])
    lateral = lateral_offsets[:steps][:, None] * _orthonormal_basis(axes[0])[1][None, :]
    _ = local_radius
    result: NDArray[np.float64] = base + lateral
    return result
