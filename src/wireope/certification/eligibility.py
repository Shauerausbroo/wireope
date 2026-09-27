"""Support-constrained eligibility: the per-stratum, event-wise overlap floor.

For every anatomical stratum g and every rung k the behaviour policy must satisfy
P_pib(A_k | g) >= c. A strategy that fails the floor is ineligible rather than silently
extrapolated, because a global effective-sample-size check is a state-density quantity and
cannot see whether the injury-producing region of the action space was visited.

Ref: Sec. 1, Sec. 3.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.utils.linalg import EPS


@dataclass(frozen=True)
class StratumRungSupport:
    stratum: str
    rung: int
    behaviour_mass: float
    passes_floor: bool


@dataclass(frozen=True)
class EligibilityReport:
    floor: float
    strategy: str
    entries: tuple[StratumRungSupport, ...]
    eligible: bool
    failing: tuple[tuple[str, int], ...]

    def failing_count(self) -> int:
        return len(self.failing)


def stratum_rung_mass(
    strata: NDArray[np.str_],
    rung_indicators: NDArray[np.float64],
    behaviour_probability: NDArray[np.float64],
) -> dict[tuple[str, int], float]:
    """P_pib(A_k | g): behaviour-probability mass of the attempts in stratum g at rung k."""
    mass: dict[tuple[str, int], float] = {}
    for name in sorted(set(strata.tolist())):
        member = strata == name
        if not member.any():
            continue
        local = behaviour_probability[member]
        total = max(float(np.sum(local)), EPS)
        indicators = rung_indicators[member]
        for rung in range(indicators.shape[1]):
            reached = indicators[:, rung] > 0.0
            share = float(np.sum(local[reached]) / total)
            mass[(str(name), rung)] = share
    return mass


def evaluate_floor(
    strategy: str,
    strata: NDArray[np.str_],
    rung_indicators: NDArray[np.float64],
    behaviour_probability: NDArray[np.float64],
    floor: float,
) -> EligibilityReport:
    mass = stratum_rung_mass(strata, rung_indicators, behaviour_probability)
    entries = tuple(
        StratumRungSupport(
            stratum=stratum,
            rung=rung,
            behaviour_mass=value,
            passes_floor=bool(value >= floor),
        )
        for (stratum, rung), value in sorted(mass.items())
    )
    failing = tuple((entry.stratum, entry.rung) for entry in entries if not entry.passes_floor)
    return EligibilityReport(
        floor=float(floor),
        strategy=strategy,
        entries=entries,
        eligible=not failing,
        failing=failing,
    )


def eligible_strategies(
    strategy_ids: NDArray[np.str_],
    floors: dict[str, EligibilityReport],
) -> NDArray[np.str_]:
    """Pi_rho: the strategies that clear the floor on every stratum and every rung."""
    keep = [
        name for name in strategy_ids.tolist() if floors.get(name) is not None and floors[name].eligible
    ]
    return np.asarray(keep, dtype=np.str_)


def worst_margin(report: EligibilityReport) -> float:
    """Smallest gap between an observed stratum-rung mass and the floor."""
    if not report.entries:
        return float("-inf")
    return float(min(entry.behaviour_mass - report.floor for entry in report.entries))


def floor_effect_on_safety(
    ineligible_violation_rate: float,
    eligible_violation_rate: float,
) -> float:
    """Reduction in the violation rate attributable to excluding unsupported strategies."""
    return float(ineligible_violation_rate - eligible_violation_rate)


def support_covered_strata(
    report: EligibilityReport,
) -> tuple[str, ...]:
    return tuple(sorted({entry.stratum for entry in report.entries if entry.passes_floor}))
