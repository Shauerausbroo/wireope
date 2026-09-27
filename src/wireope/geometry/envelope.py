"""Lumen envelope, its boundary distance field and its surface integral.

The envelope is the measurable lumen limit of the concerned cohort. The excursion
observable is the distance of the reconstructed wire body beyond that boundary, so the
boundary distance field is the object the whole framework reads.

Ref: Sec. 3.2-3.3, Eq. (1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def spacing_array(self) -> NDArray[np.float64]:
        return np.asarray(self.spacing_mm, dtype=np.float64)

    @property
    def origin_array(self) -> NDArray[np.float64]:
        return np.asarray(self.origin_mm, dtype=np.float64)

    def index_to_mm(self, indices: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = indices * self.spacing_array
        result: NDArray[np.float64] = scaled + self.origin_array
        return result

    def mm_to_index(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = points - self.origin_array
        result: NDArray[np.float64] = shifted / self.spacing_array
        return result

    def voxel_face_area_mm2(self) -> float:
        spacing = self.spacing_array
        face_areas = (
            spacing[1] * spacing[2],
            spacing[0] * spacing[2],
            spacing[0] * spacing[1],
        )
        return float(np.mean(np.asarray(face_areas, dtype=np.float64)))

    def surface_voxel_area_mm2(self) -> NDArray[np.float64]:
        spacing = self.spacing_array
        return np.asarray(
            [spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1]],
            dtype=np.float64,
        )


def _exposed_faces(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    padded = np.zeros(tuple(size + 2 for size in mask.shape), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = mask
    counts = np.zeros(3, dtype=np.int64)
    for axis in range(3):
        forward = np.diff(padded, axis=axis) == -1
        backward = np.diff(padded, axis=axis) == 1
        counts[axis] = int(np.count_nonzero(forward) + np.count_nonzero(backward))
    return counts


def signed_distance_field(mask: NDArray[np.bool_], grid: GridSpec) -> NDArray[np.float64]:
    """Signed distance to the envelope boundary, positive outside the lumen.

    The field is the distance to the nearest boundary voxel, negated inside the lumen.
    Outside the lumen the distance the wire has travelled beyond the envelope is exactly
    this field, which is why the excursion observable inherits the tracking and
    segmentation tolerances of the two stages that produce the mask.
    """
    binary = np.asarray(mask) > 0.5 if mask.dtype != np.bool_ else mask
    if not binary.any():
        raise ValueError("envelope mask is empty; a distance field is undefined")
    if binary.all():
        raise ValueError("envelope mask fills the grid; a boundary does not exist")
    inside = ndimage.distance_transform_edt(binary, sampling=grid.spacing_mm)
    outside = ndimage.distance_transform_edt(~binary, sampling=grid.spacing_mm)
    signed = np.where(binary, -inside, outside)
    return np.asarray(signed, dtype=np.float64)


def trilinear_sample(
    field: NDArray[np.float64], grid: GridSpec, points_mm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sample a scalar field at millimetre coordinates with trilinear interpolation."""
    index = grid.mm_to_index(np.atleast_2d(points_mm))
    shape = np.asarray(field.shape, dtype=np.float64)
    clamped = np.clip(index, 0.0, shape - 1.0 - EPS)
    lower = np.floor(clamped).astype(np.int64)
    frac = clamped - lower
    result = np.zeros(index.shape[0], dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (frac[:, 0] if dx else 1.0 - frac[:, 0])
                    * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                    * (frac[:, 2] if dz else 1.0 - frac[:, 2])
                )
                corner = field[lower[:, 0] + dx, lower[:, 1] + dy, lower[:, 2] + dz]
                result = result + weight * corner
    selected: NDArray[np.float64] = np.asarray(result, dtype=np.float64)
    return selected


@dataclass(frozen=True)
class LumenEnvelope:
    mask: NDArray[np.bool_]
    grid: GridSpec
    _distance: NDArray[np.float64] | None = None

    @property
    def distance(self) -> NDArray[np.float64]:
        if self._distance is None:
            return signed_distance_field(self.mask, self.grid)
        return self._distance

    def with_distance(self, distance: NDArray[np.float64]) -> LumenEnvelope:
        return LumenEnvelope(mask=self.mask, grid=self.grid, _distance=distance)

    def excursion(self, points_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Excursion observable h_t = rho(x_wire, dOmega), clipped at zero.

        The observable is non-negative by construction: a wire inside the envelope
        carries no excursion.
        """
        sampled = trilinear_sample(self.distance, self.grid, points_mm)
        return np.clip(sampled, 0.0, self.grid_mm_extent())

    def grid_mm_extent(self) -> float:
        spacing = self.grid.spacing_array
        shape = np.asarray(self.grid.shape, dtype=np.float64)
        return float(np.max(spacing * shape))

    def surface_area_mm2(self) -> float:
        """Voxel-face estimate of the envelope surface area."""
        counts = _exposed_faces(self.mask)
        areas = self.grid.surface_voxel_area_mm2()
        return float(np.sum(counts.astype(np.float64) * areas))

    def boundary_points_mm(self) -> NDArray[np.float64]:
        """Centres of the voxels that carry at least one exposed face."""
        padded = np.zeros(tuple(size + 2 for size in self.mask.shape), dtype=bool)
        padded[1:-1, 1:-1, 1:-1] = self.mask
        interior = padded[1:-1, 1:-1, 1:-1]
        exposed = np.zeros_like(self.mask)
        for axis in range(3):
            for shift in (-1, 1):
                neighbour = np.roll(padded, shift, axis=axis)[1:-1, 1:-1, 1:-1]
                exposed |= interior & ~neighbour
        indices = np.argwhere(exposed).astype(np.float64)
        return self.grid.index_to_mm(indices)

    def centreline_points_mm(self, axis: int = 2) -> NDArray[np.float64]:
        """Centroid of the mask per slice along the vessel axis of travel."""
        points: list[NDArray[np.float64]] = []
        for position in range(self.mask.shape[axis]):
            selector: list[Any] = [slice(None)] * 3
            selector[axis] = position
            slab = self.mask[tuple(selector)]
            if not slab.any():
                continue
            indices = np.argwhere(slab).astype(np.float64)
            centroid = indices.mean(axis=0)
            full = np.zeros(3, dtype=np.float64)
            remaining = [index for index in range(3) if index != axis]
            full[remaining[0]] = centroid[0]
            full[remaining[1]] = centroid[1]
            full[axis] = float(position)
            points.append(self.grid.index_to_mm(full))
        if not points:
            raise ValueError("envelope mask has no occupied slice along the vessel axis")
        return np.vstack(points)

    def volume_mm3(self) -> float:
        spacing = self.grid.spacing_array
        voxel_volume = float(np.prod(spacing))
        return float(np.count_nonzero(self.mask)) * voxel_volume

    def wall_overlap_fraction(self, points_mm: NDArray[np.float64], tolerance_mm: float) -> float:
        """Share of reconstructed wire samples that sit within one tolerance of the wall."""
        excursions = self.excursion(points_mm)
        near_wall = np.abs(excursions) <= tolerance_mm
        return float(np.mean(near_wall)) if near_wall.size else 0.0
