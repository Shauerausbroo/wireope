"""Registration of the reconstructed fluoroscopic frame onto the CTA envelope frame.

The two frames are aligned by the arc length of the vessel rather than by intensity, so
the transform is a two-dimensional similarity (rotation, isotropic scale, translation)
fitted in closed form and refined against the envelope centreline.

Ref: Sec. 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import RegistrationConfig
from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class SimilarityTransform2D:
    rotation_rad: float
    scale: float
    translation_mm: NDArray[np.float64]

    def as_matrix(self) -> NDArray[np.float64]:
        cosine = self.scale * np.cos(self.rotation_rad)
        sine = self.scale * np.sin(self.rotation_rad)
        return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    def apply(self, points_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        planar = np.atleast_2d(points_mm)[:, :2]
        rotated = planar @ self.as_matrix().T
        result: NDArray[np.float64] = rotated + self.translation_mm[None, :]
        return result


@dataclass(frozen=True)
class RegistrationResult:
    transform: SimilarityTransform2D
    iterations: int
    residual_mm: float
    converged: bool


def fit_similarity(source_mm: NDArray[np.float64], target_mm: NDArray[np.float64]) -> SimilarityTransform2D:
    """Closed-form least-squares similarity fit (Umeyama) in two dimensions."""
    source = np.atleast_2d(source_mm)[:, :2]
    target = np.atleast_2d(target_mm)[:, :2]
    if source.shape[0] != target.shape[0] or source.shape[0] < 2:
        raise ValueError("similarity fit needs equally sized point sets of at least two points")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centred_source = source - source_mean
    centred_target = target - target_mean
    covariance = centred_target.T @ centred_source / source.shape[0]
    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    sign = np.ones(2, dtype=np.float64)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0.0:
        sign[-1] = -1.0
    rotation = u_matrix @ np.diag(sign) @ vt_matrix
    variance = float(np.mean(np.sum(centred_source**2, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, EPS))
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform2D(
        rotation_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        scale=scale,
        translation_mm=np.asarray(translation, dtype=np.float64),
    )


def nearest_centreline_points(
    query_mm: NDArray[np.float64],
    centreline_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    query = np.atleast_2d(query_mm)[:, :2]
    reference = np.atleast_2d(centreline_mm)[:, :2]
    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    index = np.argmin(distances, axis=1)
    nearest: NDArray[np.float64] = np.asarray(reference[index], dtype=np.float64)
    return nearest


def align_trajectory(
    trajectory_mm: NDArray[np.float64],
    centreline_mm: NDArray[np.float64],
    config: RegistrationConfig,
) -> RegistrationResult:
    """Alternate nearest-point assignment and a closed-form similarity update."""
    transform = fit_similarity(_resample(trajectory_mm, centreline_mm.shape[0]), centreline_mm)
    residual = float("inf")
    converged = False
    iterations = 0
    for iteration in range(config.iterations):
        transformed = transform.apply(trajectory_mm)
        matched = nearest_centreline_points(transformed, centreline_mm)
        resampled_source = _resample(trajectory_mm, matched.shape[0])
        transform = fit_similarity(resampled_source, matched)
        new_residual = float(np.mean(np.linalg.norm(transform.apply(resampled_source) - matched, axis=1)))
        iterations = iteration + 1
        if abs(residual - new_residual) <= config.tolerance_mm:
            residual = new_residual
            converged = True
            break
        residual = new_residual
    return RegistrationResult(
        transform=transform,
        iterations=iterations,
        residual_mm=residual,
        converged=converged,
    )


def transform_trajectory(
    trajectory_mm: NDArray[np.float64],
    result: RegistrationResult,
    spatial_axis: int = 2,
) -> NDArray[np.float64]:
    """Apply the planar transform and carry the out-of-plane coordinate through unchanged."""
    planar = result.transform.apply(trajectory_mm)
    carried = np.array(trajectory_mm, dtype=np.float64, copy=True)
    carried[:, :2] = planar
    _ = spatial_axis
    return carried


def _resample(points: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    array = np.atleast_2d(points)
    if array.shape[0] == count:
        return array
    if array.shape[0] < 2:
        return np.repeat(array, count, axis=0)
    positions = np.linspace(0.0, array.shape[0] - 1, count)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, array.shape[0] - 1)
    fraction = (positions - low)[:, None]
    result: NDArray[np.float64] = (1.0 - fraction) * array[low] + fraction * array[high]
    return result


def arc_length_parameterisation(points_mm: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.atleast_2d(points_mm)
    if array.shape[0] < 2:
        return np.zeros(array.shape[0], dtype=np.float64)
    segments = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    result: NDArray[np.float64] = np.asarray(cumulative, dtype=np.float64)
    return result
