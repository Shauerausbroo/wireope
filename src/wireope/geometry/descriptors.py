"""Anatomical descriptors carried by the state beside the lumen envelope.

Calcification amount and distribution follow the Peripheral Arterial Calcium Scoring
System, which the manuscript treats as a proxy for plaque geometry; cap morphology,
lesion length and vessel territory complete the anatomical identifiers.

Ref: Sec. 3.1 and Sec. 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wireope.config import DescriptorConfig
from wireope.geometry.envelope import LumenEnvelope
from wireope.utils.linalg import EPS

STAGE_INDEX: dict[str, int] = {"I": 0, "II": 1, "III": 2}
CALCIFICATION_SEVERITY: dict[int, float] = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
TERRITORY_INDEX: dict[str, int] = {"femoropopliteal": 0, "tibial": 1, "other": 2}
CAP_INDEX: dict[str, int] = {"calcified_nodule": 0, "ambiguous": 1, "tapered": 2, "blunt": 3}


@dataclass(frozen=True)
class CalcificationProfile:
    grade: int
    arc_deg: float
    severity: float
    volume_mm3: float


@dataclass(frozen=True)
class AnatomyDescriptors:
    stage: str
    calcification: CalcificationProfile
    cap_morphology: str
    lesion_length_mm: float
    territory: str

    def as_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                float(STAGE_INDEX.get(self.stage, 0)),
                float(self.calcification.grade),
                self.calcification.severity,
                self.calcification.arc_deg / 360.0,
                self.lesion_length_mm / 60.0,
                float(TERRITORY_INDEX.get(self.territory, 2)),
                float(CAP_INDEX.get(self.cap_morphology, 1)),
            ],
            dtype=np.float64,
        )


def grade_calcification(load_mm3: float, bins_mm3: tuple[float, float, float, float]) -> int:
    """Map a calcified-plaque volume onto the five-point grading scale."""
    if load_mm3 <= 0.0:
        return 0
    for grade, threshold in enumerate(bins_mm3, start=1):
        if load_mm3 <= threshold:
            return grade
    return 4


def calcification_arc_deg(calcium_mask: NDArray[np.bool_], centreline: NDArray[np.float64]) -> float:
    """Angular extent of calcium around the vessel axis, in degrees.

    Each calcium voxel is projected onto the plane normal to the local centreline; the
    covered angular bins, not their count, give the arc, so a single dense plaque and a
    long thin one are separated.
    """
    if not calcium_mask.any():
        return 0.0
    axis = np.asarray(centreline[-1] - centreline[0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= EPS:
        return 0.0
    axis = axis / norm
    centroid = np.asarray(centreline.mean(axis=0), dtype=np.float64)
    indices = np.argwhere(calcium_mask).astype(np.float64)
    offsets = indices - centroid
    projection = offsets - np.outer(offsets @ axis, axis)
    radii = np.linalg.norm(projection, axis=1)
    valid = radii > EPS
    if not valid.any():
        return 0.0
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    first = reference - np.dot(reference, axis) * axis
    first = first / max(float(np.linalg.norm(first)), EPS)
    second = np.cross(axis, first)
    angles = np.arctan2(projection[valid] @ second, projection[valid] @ first)
    bins = np.floor((angles + np.pi) / (2.0 * np.pi) * 72.0).astype(np.int64)
    occupied = np.unique(np.clip(bins, 0, 71))
    return float(occupied.size * (360.0 / 72.0))


def cap_morphology_class(envelope: LumenEnvelope) -> str:
    """Classify the lesion cap from the occluded cross-section's profile.

    A blunt cap keeps its maximal cross-section until it stops; a tapered cap narrows
    monotonically; an ambiguous cap narrows and then re-expands once.
    """
    profile = cross_section_profile(envelope)
    if profile.size < 3:
        return "ambiguous"
    width = profile / max(float(np.max(profile)), EPS)
    diffs = np.diff(width)
    narrowing = float(np.mean(diffs < -0.02))
    monotone = bool(np.all(diffs <= 0.02))
    trailing = float(np.mean(diffs[int(len(diffs) * 0.75) :] < -0.05))
    if monotone and narrowing > 0.5:
        return "tapered"
    if trailing > 0.5 and narrowing > 0.4:
        return "calcified_nodule"
    if narrowing < 0.25:
        return "blunt"
    return "ambiguous"


def cross_section_profile(envelope: LumenEnvelope, axis: int = 2) -> NDArray[np.float64]:
    profile: list[float] = []
    for position in range(envelope.mask.shape[axis]):
        selector: list[Any] = [slice(None)] * 3
        selector[axis] = position
        slab = envelope.mask[tuple(selector)]
        profile.append(float(np.count_nonzero(slab)))
    return np.asarray(profile, dtype=np.float64)


def lesion_length_mm(centreline: NDArray[np.float64]) -> float:
    if centreline.shape[0] < 2:
        return 0.0
    segments = np.linalg.norm(np.diff(centreline, axis=0), axis=1)
    return float(np.sum(segments))


def territory_from_axis(centreline: NDArray[np.float64], tibial_cut_mm: float = 30.0) -> str:
    """Assign the vessel territory from the length of the reconstructed segment."""
    length = lesion_length_mm(centreline)
    if length <= 0.0:
        return "other"
    return "femoropopliteal" if length >= tibial_cut_mm else "tibial"


def stage_from_load(calcification_severity: float, lesion_length: float) -> str:
    """Anatomical stage from the two anatomy axes the staging system combines."""
    score = calcification_severity + min(lesion_length / 60.0, 1.0)
    if score < 0.45:
        return "I"
    if score < 1.05:
        return "II"
    return "III"


def descriptor_from_components(
    config: DescriptorConfig,
    stage: str,
    calcification: CalcificationProfile,
    cap: str,
    length_mm: float,
    territory: str,
) -> AnatomyDescriptors:
    if stage not in config.stages:
        raise ValueError(f"unknown anatomical stage '{stage}'")
    if territory not in config.territory_classes:
        raise ValueError(f"unknown vessel territory '{territory}'")
    if cap not in config.cap_classes:
        raise ValueError(f"unknown cap morphology '{cap}'")
    return AnatomyDescriptors(
        stage=stage,
        calcification=calcification,
        cap_morphology=cap,
        lesion_length_mm=length_mm,
        territory=territory,
    )


def calcification_volume_mm3(calcium_mask: NDArray[np.bool_], voxel_volume_mm3: float) -> float:
    return float(np.count_nonzero(calcium_mask)) * voxel_volume_mm3
