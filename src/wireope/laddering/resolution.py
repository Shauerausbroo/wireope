"""Rung-resolution criterion.

Adjacent excursion levels must be separated by more than the band propagated from the
tip-tracking tolerance and the envelope segmentation tolerance, otherwise two rungs are
not distinguishable as events. The criterion fixes the depth range over which a ladder is
interpretable, and the truncation term grows with depth while the per-event rarity falls,
which is what puts an interior optimum on the depth.

Ref: Sec. 3.3 and Sec. 3.4, Eq. (7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.laddering.severity import Ladder


@dataclass(frozen=True)
class ResolutionReport:
    band_mm: float
    separations_mm: tuple[float, ...]
    resolvable: bool
    max_depth: int
    reason: str


def adjacent_separations(tau_mm: tuple[float, ...]) -> NDArray[np.float64]:
    return np.asarray(np.diff(np.asarray(tau_mm, dtype=np.float64)), dtype=np.float64)


def max_resolvable_depth(tau_mm: tuple[float, ...], band_mm: float) -> int:
    """Largest prefix of the threshold grid whose adjacent separations all exceed the band."""
    separations = adjacent_separations(tau_mm)
    depth = 1
    for separation in separations:
        if separation > band_mm:
            depth += 1
        else:
            break
    return depth


def check_resolution(ladder: Ladder, band_mm: float) -> ResolutionReport:
    separations = adjacent_separations(ladder.tau_mm)
    resolvable = bool(np.all(separations > band_mm))
    max_depth = max_resolvable_depth(ladder.tau_mm, band_mm)
    if resolvable:
        reason = "every adjacent rung separation exceeds the propagating band"
    else:
        too_close = int(np.sum(separations <= band_mm))
        reason = f"{too_close} adjacent separation(s) fall at or below the propagating band"
    return ResolutionReport(
        band_mm=float(band_mm),
        separations_mm=tuple(float(item) for item in separations),
        resolvable=resolvable,
        max_depth=max_depth,
        reason=reason,
    )


def truncation_term(depth: int, rung_truncation_mm: float, tau_inj_mm: float) -> float:
    """Bounded output of the deepest rung relative to the injury threshold, in ladder units.

    The term is non-negative and decreases as the ladder is refined, which is the
    behaviour the deviation bound of Eq. (7) assumes.
    """
    if tau_inj_mm <= 0.0:
        raise ValueError("injury threshold must be positive")
    ratio = min(1.0, max(0.0, rung_truncation_mm / tau_inj_mm))
    return float(ratio / max(depth, 1))


def interior_optimum_depth(
    rung_counts: NDArray[np.int64],
    delta: float,
    bounded_range: float,
    truncation_scale_mm: float,
) -> int:
    """Depth minimising the Eq. (7) bound: log-depth inflation against rarer events.

    The union bound contributes ln(2K/delta), which grows with depth, while the effective
    event count at the binding rung falls; the sum of the two terms is scanned over depth
    and the argmin is returned.
    """
    counts = np.asarray(rung_counts, dtype=np.float64)
    if counts.size == 0:
        raise ValueError("rung counts are empty")
    best_depth = 1
    best_bound = float("inf")
    for depth in range(1, counts.size + 1):
        effective = float(np.min(counts[:depth]))
        if effective <= 0.0:
            continue
        union = float(np.log(2.0 * depth / max(delta, 1e-12)))
        statistical = bounded_range * float(np.sqrt(union / (2.0 * effective)))
        truncation = truncation_term(depth, truncation_scale_mm, truncation_scale_mm * depth)
        total = statistical + truncation
        if total < best_bound:
            best_bound = total
            best_depth = depth
    return best_depth


def band_from_tolerances(tip_tolerance_mm: float, segmentation_tolerance_mm: float, mode: str) -> float:
    if mode == "quadrature":
        return float(np.hypot(tip_tolerance_mm, segmentation_tolerance_mm))
    if mode == "additive":
        return float(tip_tolerance_mm + segmentation_tolerance_mm)
    raise ValueError(f"unsupported tolerance combination '{mode}'")
