"""Hoeffding form of the deviation bound for the product risk estimator.

For any delta in (0, 1), with probability at least 1 - delta,

    |R_hat(pi) - R(pi)| <= B sqrt( ln(2K/delta) / (2 E_eff) ) + eps_K,  E_eff := min_k E_k,

so the binding resource is the effective count of positive events at the rarest rung rather
than the number of logged attempts. Adding attempts that do not raise the count at the
rarest rung cannot tighten this bound.

Ref: Sec. 3.4, Eq. (7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class HoeffdingBound:
    radius: float
    effective_sample_size: float
    depth: int
    delta: float
    bounded_range: float
    truncation_term: float

    @property
    def total(self) -> float:
        return float(self.radius + self.truncation_term)


def effective_sample_size_minimum(per_rung_counts: NDArray[np.float64]) -> float:
    """E_eff := min_k E_k over the rungs."""
    counts = np.asarray(per_rung_counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    positive = counts[counts > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive))


def hoeffding_bound(
    per_rung_counts: NDArray[np.float64],
    delta: float,
    bounded_range: float,
    truncation_term: float,
) -> HoeffdingBound:
    depth = int(np.asarray(per_rung_counts).size)
    effective = effective_sample_size_minimum(per_rung_counts)
    if effective <= 0.0 or depth <= 0:
        return HoeffdingBound(
            radius=float("inf"),
            effective_sample_size=effective,
            depth=depth,
            delta=delta,
            bounded_range=bounded_range,
            truncation_term=truncation_term,
        )
    argument = np.log(2.0 * depth / max(delta, EPS))
    radius = float(bounded_range * np.sqrt(max(argument, 0.0) / (2.0 * effective)))
    return HoeffdingBound(
        radius=radius,
        effective_sample_size=effective,
        depth=depth,
        delta=delta,
        bounded_range=bounded_range,
        truncation_term=float(truncation_term),
    )


def design_event_requirement(
    radius_target: float,
    delta: float,
    bounded_range: float,
    depth: int,
    truncation_term: float,
    tolerance: float = 1e-12,
) -> float:
    """Effective event count needed to bring the Eq. (7) radius to a stated safety margin.

    Inverting the bound gives E_eff >= B^2 ln(2K/delta) / (2 (r - eps_K)^2); the count is
    what a registry has to be sized on, not the number of procedures or sites.
    """
    budget = radius_target - truncation_term
    if budget <= tolerance:
        return float("inf")
    argument = np.log(2.0 * depth / max(delta, EPS))
    return float(bounded_range**2 * argument / (2.0 * budget**2))


def radius_increases_with_depth(
    per_rung_counts: NDArray[np.float64],
    delta: float,
    bounded_range: float,
) -> bool:
    """The statistical radius is non-decreasing in depth: the union bound only grows."""
    counts = np.asarray(per_rung_counts, dtype=np.float64)
    radii = [
        hoeffding_bound(counts[: depth + 1], delta, bounded_range, 0.0).radius
        for depth in range(1, counts.size + 1)
    ]
    return bool(all(later >= earlier - 1e-12 for earlier, later in zip(radii, radii[1:])))


def monotone_in_rarest_rung(
    per_rung_counts: NDArray[np.float64], delta: float, bounded_range: float
) -> bool:
    """Raising only the count at the rarest rung cannot increase the radius."""
    counts = np.asarray(per_rung_counts, dtype=np.float64)
    if counts.size == 0:
        return True
    base = hoeffding_bound(counts, delta, bounded_range, 0.0).radius
    augmented = counts.copy()
    augmented[-1] = augmented[-1] + 1.0
    return bool(hoeffding_bound(augmented, delta, bounded_range, 0.0).radius <= base + 1e-12)


def apparent_gain_collapse(
    optimistic_gain: float,
    pessimistic_gain: float,
    alpha_low: float,
    alpha_high: float,
) -> dict[str, float]:
    """Reproduce the collapse pattern the manuscript cites for the pessimism sweep.

    The published sweep moved an apparent gain from +32% to +3% as the pessimism
    coefficient moved from 0.001 to 0.5; the same ratio is reported here for the sweep that
    this release runs.
    """
    if abs(optimistic_gain) <= EPS:
        return {"ratio": float("nan"), "alpha_low": alpha_low, "alpha_high": alpha_high}
    return {
        "ratio": float(pessimistic_gain / optimistic_gain),
        "alpha_low": float(alpha_low),
        "alpha_high": float(alpha_high),
    }
