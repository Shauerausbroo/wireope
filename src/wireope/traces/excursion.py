"""Envelope-excursion observable, dwell integral and surface integral.

The excursion h_t = rho(x_wire_t, dOmega) is the distance of the reconstructed wire body
beyond the boundary of the lumen envelope, and the dwell integral accumulates it over the
advancement steps. Both inherit the tip-tracking tolerance and the envelope segmentation
tolerance, which is why the propagating band bounds how close two rungs may be.

Ref: Sec. 3.3, Eq. (1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import UncertaintyConfig
from wireope.geometry.envelope import LumenEnvelope
from wireope.schema import ExcursionTrace
from wireope.utils.linalg import trapezoid


@dataclass(frozen=True)
class PropagatingBand:
    tip_tolerance_mm: float
    segmentation_tolerance_mm: float
    combined_mm: float
    combination: str


def propagating_band(config: UncertaintyConfig) -> PropagatingBand:
    """Combine the two tolerances that bound the excursion observable."""
    tip = float(config.tip_tolerance_mm)
    segmentation = float(config.segmentation_tolerance_mm)
    if config.combination == "quadrature":
        combined = float(np.hypot(tip, segmentation))
    elif config.combination == "additive":
        combined = tip + segmentation
    else:
        raise ValueError(f"unsupported tolerance combination '{config.combination}'")
    return PropagatingBand(
        tip_tolerance_mm=tip,
        segmentation_tolerance_mm=segmentation,
        combined_mm=combined * float(config.confidence_multiplier),
        combination=str(config.combination),
    )


def excursion_series(
    envelope: LumenEnvelope,
    positions_mm: NDArray[np.float64],
) -> NDArray[np.float64]:
    if positions_mm.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return envelope.excursion(positions_mm)


def dwell_integral(excursion_mm: NDArray[np.float64], dwell_s: NDArray[np.float64]) -> float:
    if excursion_mm.size == 0:
        return 0.0
    return float(np.sum(excursion_mm * dwell_s))


def surface_integral(
    envelope: LumenEnvelope,
    positions_mm: NDArray[np.float64],
    exposed_fraction: float,
) -> float:
    """Wire surface that sits beyond the envelope boundary, in square millimetres.

    The reconstructed body is treated as a cylinder of unit circumference per millimetre
    of path; only the share of the body outside the boundary contributes.
    """
    if positions_mm.shape[0] < 2:
        return 0.0
    segments = np.linalg.norm(np.diff(positions_mm, axis=0), axis=1)
    outside = exposed_fraction
    return float(np.sum(segments) * outside)


def surface_integral_at_step(
    envelope: LumenEnvelope,
    positions_mm: NDArray[np.float64],
    radius_mm: float = 0.18,
) -> float:
    """Closed-form cylinder-surface estimate: 2 pi r times the beyond-boundary path length."""
    excursions = excursion_series(envelope, positions_mm)
    if excursions.size < 2:
        return 0.0
    segments = np.linalg.norm(np.diff(positions_mm, axis=0), axis=1)
    outside = excursions[1:] > 0.0
    return float(2.0 * np.pi * radius_mm * np.sum(segments[outside]))


def build_trace(
    attempt_id: int,
    envelope: LumenEnvelope,
    positions_mm: NDArray[np.float64],
    dwell_s: NDArray[np.float64],
) -> ExcursionTrace:
    excursions = excursion_series(envelope, positions_mm)
    if dwell_s.shape[0] != excursions.shape[0]:
        dwell_s = np.resize(dwell_s, excursions.shape[0])
    outside = float(np.mean(excursions > 0.0)) if excursions.size else 0.0
    return ExcursionTrace(
        attempt_id=attempt_id,
        positions_mm=np.asarray(positions_mm, dtype=np.float64),
        excursion_mm=np.asarray(excursions, dtype=np.float64),
        dwell_s=np.asarray(dwell_s, dtype=np.float64),
        surface_integral_mm2=surface_integral(envelope, positions_mm, outside),
    )


def rung_indicators(peak_excursion_mm: float, tau_mm: tuple[float, ...]) -> NDArray[np.float64]:
    """Nested-set indicators Y^(k) = 1{ max_t h_t >= tau_k } for every rung.

    tau is increasing, so the indicators are non-increasing in k: every rung k+1 event is
    also a rung k event, which is the nesting the ladder relies on.
    """
    threshold = np.asarray(tau_mm, dtype=np.float64)
    values: NDArray[np.float64] = (float(peak_excursion_mm) >= threshold).astype(np.float64)
    return values


def excursion_peak_from_trace(trace: ExcursionTrace) -> float:
    return trace.max_excursion_mm


def dwell_weighted_peak(trace: ExcursionTrace) -> float:
    if trace.excursion_mm.size == 0:
        return 0.0
    weights = trace.dwell_s / max(float(np.sum(trace.dwell_s)), 1e-12)
    return float(np.sum(trace.excursion_mm * weights))


def cumulative_excursion(trace: ExcursionTrace) -> float:
    return trapezoid(trace.excursion_mm, 1.0)
