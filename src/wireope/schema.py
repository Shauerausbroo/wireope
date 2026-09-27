"""Shared record types for logs, reconstructed traces and study results.

The unit of analysis is the crossing attempt, nested within the lesion, nested within
the procedure record; every record type below carries that nesting explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class Territory(StrEnum):
    FEMOROPOPLITEAL = "femoropopliteal"
    TIBIAL = "tibial"
    OTHER = "other"


class CapMorphology(StrEnum):
    CALCIFIED_NODULE = "calcified_nodule"
    AMBIGUOUS = "ambiguous"
    TAPERED = "tapered"
    BLUNT = "blunt"


class EstimatorFamily(StrEnum):
    IMPORTANCE_SAMPLING = "importance_sampling"
    DIRECT = "direct"
    DOUBLY_ROBUST = "doubly_robust"
    SWITCHING_HYBRID = "switching_hybrid"
    KERNEL_DIRECT = "kernel_direct"
    HIGH_CONFIDENCE_BOUND = "high_confidence_bound"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


class AnatomyRecord(TypedDict):
    stage: str
    calcification_grade: int
    calcification_arc_deg: float
    cap_morphology: str
    lesion_length_mm: float
    territory: str
    site: str
    operator: str
    vendor_class: str


class DeviceRecord(TypedDict):
    device_class: str
    attempt_index: int
    device_to_lesion_relation: str


class EventRecord(TypedDict):
    step: int
    kind: str
    magnitude: float


class AttemptRecord(TypedDict):
    attempt_id: int
    procedure_id: int
    lesion_id: int
    patient_id: int
    site: str
    region: str
    operator: str
    vendor_class: str
    anatomy: AnatomyRecord
    device: DeviceRecord
    events: list[EventRecord]
    strategy: str
    injury: int


@dataclass(frozen=True)
class LadderSpec:
    depth: int
    tau_mm: tuple[float, ...]
    tau_inj_mm: float
    weights: tuple[float, ...]

    @property
    def rung_weights(self) -> NDArray[np.float64]:
        return np.asarray(self.weights, dtype=np.float64)


@dataclass(frozen=True)
class ExcursionTrace:
    attempt_id: int
    positions_mm: NDArray[np.float64]
    excursion_mm: NDArray[np.float64]
    dwell_s: NDArray[np.float64]
    surface_integral_mm2: float

    @property
    def max_excursion_mm(self) -> float:
        if self.excursion_mm.size == 0:
            return 0.0
        return float(np.max(self.excursion_mm))

    @property
    def dwell_integral_mm_s(self) -> float:
        if self.excursion_mm.size == 0:
            return 0.0
        return float(np.sum(self.excursion_mm * self.dwell_s))


@dataclass(frozen=True)
class AttemptFeatures:
    attempt_id: int
    max_excursion_mm: float
    dwell_integral_mm_s: float
    action_norm: float
    rung_indicators: NDArray[np.float64]
    injury: int
    strategy: str
    site: str
    region: str
    operator: str
    vendor_class: str
    stage: str
    calcification_grade: int
    territory: str
    descriptor_vector: NDArray[np.float64] = field(default_factory=lambda: np.zeros(1))


@dataclass(frozen=True)
class SiteLog:
    name: str
    region: str
    features: tuple[AttemptFeatures, ...]

    @property
    def count(self) -> int:
        return len(self.features)

    @property
    def injury_events(self) -> int:
        return int(sum(item.injury for item in self.features))

    def stack(self, key: str) -> NDArray[np.float64]:
        values = [getattr(item, key) for item in self.features]
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ProgressUpdate:
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float


@dataclass
class FitHistory:
    updates: list[ProgressUpdate] = field(default_factory=list)

    def append(self, update: ProgressUpdate) -> None:
        self.updates.append(update)

    @property
    def first_loss(self) -> float:
        return self.updates[0].loss if self.updates else float("nan")

    @property
    def last_loss(self) -> float:
        return self.updates[-1].loss if self.updates else float("nan")
