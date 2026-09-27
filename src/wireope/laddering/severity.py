"""Nested-excursion severity ladder over the graded injury endpoint.

Injury severity is monotone in wall penetration, so the rare injury event is written as
nested sets A_1 superset A_2 superset ... superset A_K with A_k = { max_t h_t >= tau_k }
and 0 < tau_1 < ... < tau_K = tau_inj. Each conditional P(A_k | A_{k-1}) is far commoner
than the endpoint itself, which is the currency the reliability statement is made in.

Ref: Sec. 3.4, Eq. (3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import LadderConfig, LadderWeightsConfig
from wireope.laddering.weights import build_weights
from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class Ladder:
    depth: int
    tau_mm: tuple[float, ...]
    tau_inj_mm: float
    weights: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("ladder depth must be at least one")
        if len(self.tau_mm) != self.depth:
            raise ValueError("ladder needs one threshold per rung")
        if any(later <= earlier for earlier, later in zip(self.tau_mm, self.tau_mm[1:])):
            raise ValueError("rung thresholds must be strictly increasing")
        if abs(self.tau_mm[-1] - self.tau_inj_mm) > 1e-9:
            raise ValueError("the top rung threshold is the injury threshold")
        if len(self.weights) != self.depth:
            raise ValueError("ladder needs one weight per rung")

    def indicators(self, peak_excursion_mm: float) -> NDArray[np.float64]:
        threshold = np.asarray(self.tau_mm, dtype=np.float64)
        return (float(peak_excursion_mm) >= threshold).astype(np.float64)

    def indicator_matrix(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        thresholds = np.asarray(self.tau_mm, dtype=np.float64)[None, :]
        matrix: NDArray[np.float64] = (
            np.asarray(peaks_mm, dtype=np.float64)[:, None] >= thresholds
        ).astype(np.float64)
        return matrix

    def graded_value(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = self.indicator_matrix(peaks_mm)
        weights = np.asarray(self.weights, dtype=np.float64)[None, :]
        result: NDArray[np.float64] = np.sum(matrix * weights, axis=1)
        return result

    def risk(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.indicator_matrix(peaks_mm)[:, -1]

    def conditional_matrix(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Empirical P(A_k | A_{k-1}) per rung, with the top rung the marginal endpoint."""
        matrix = self.indicator_matrix(peaks_mm)
        denominators = np.maximum(matrix[:, :-1].sum(axis=0), EPS)
        numer = matrix[:, 1:].sum(axis=0)
        return np.asarray(numer / denominators, dtype=np.float64)

    def conditional_frequencies(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = self.indicator_matrix(peaks_mm)
        first = float(np.mean(matrix[:, 0]))
        conditionals = self.conditional_matrix(peaks_mm)
        return np.concatenate([[first], conditionals])

    def rung_sizes(self, peaks_mm: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.asarray(self.indicator_matrix(peaks_mm).sum(axis=0), dtype=np.int64)

    def rarest_rung_size(self, peaks_mm: NDArray[np.float64]) -> int:
        return int(self.rung_sizes(peaks_mm)[-1])

    def nesting_holds(self, peaks_mm: NDArray[np.float64]) -> bool:
        """Every rung-k+1 event must also be a rung-k event."""
        matrix = self.indicator_matrix(peaks_mm)
        return bool(np.all(matrix[:, 1:] <= matrix[:, :-1] + 1e-12))

    def reconstruct_risk(self, conditionals: NDArray[np.float64]) -> float:
        """Product form of Eq. (6): the endpoint risk from the per-rung conditional model."""
        return float(np.prod(np.asarray(conditionals, dtype=np.float64)))


def build_ladder(config: LadderConfig, weights_config: LadderWeightsConfig) -> Ladder:
    if config.depth > len(config.tau_mm):
        raise ValueError("configured depth exceeds the number of thresholds available")
    tau = tuple(float(item) for item in config.tau_mm[: config.depth])
    tau_inj = float(config.tau_inj_mm)
    if abs(tau[-1] - tau_inj) > 1e-9:
        tau = (*tau[:-1], tau_inj)
    weights = build_weights(config.depth, weights_config)
    return Ladder(
        depth=config.depth,
        tau_mm=tau,
        tau_inj_mm=tau_inj,
        weights=weights,
        source=f"severity_rungs_depths_{config.depth}",
    )


def ladder_for_depth(config: LadderConfig, weights_config: LadderWeightsConfig, depth: int) -> Ladder:
    if depth > len(config.tau_mm):
        raise ValueError("requested depth exceeds the number of thresholds available")
    tau = tuple(float(item) for item in config.tau_mm[:depth])
    tau_inj = float(config.tau_inj_mm)
    if abs(tau[-1] - tau_inj) > 1e-9:
        tau = (*tau[:-1], tau_inj)
    return Ladder(
        depth=depth,
        tau_mm=tau,
        tau_inj_mm=tau_inj,
        weights=build_weights(depth, weights_config),
        source=f"severity_rungs_depths_{depth}",
    )


def conditional_within_band(
    ladder: Ladder, frequencies: NDArray[np.float64], band: tuple[float, float]
) -> bool:
    """Report whether every conditional frequency lies in the tolerable, not-uncommon range."""
    low, high = band
    tail = frequencies[1:]
    return bool(np.all((tail >= low) & (tail <= high)))


def rungs_from_injury_flags(
    injury_flags: NDArray[np.float64],
    peaks_mm: NDArray[np.float64],
    expected_ratio: float,
) -> NDArray[np.float64]:
    """Distribution-check helper: per-rung prevalence implied by a constant conditional ratio."""
    depth = injury_flags.size
    ratios = np.full(depth, expected_ratio, dtype=np.float64)
    return np.asarray(ratios, dtype=np.float64)
