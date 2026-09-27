"""Outcome adjudication and the nesting of the analysis unit.

The vessel-injury endpoint is adjudicated per crossing attempt by at least two reviewers
blinded to the reconstructed trajectory, and the attempt is nested inside the lesion, which
is nested inside the procedure record. Agreement between reviewers is reported so that the
endpoint is not a single reader's opinion.

Ref: Sec. 3.6 and Sec. 3.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AdjudicationPair:
    attempt_id: int
    reviewer_a: int
    reviewer_b: int
    call_a: int
    call_b: int

    @property
    def agreed(self) -> bool:
        return self.call_a == self.call_b

    @property
    def final(self) -> int:
        """The adjudicated call: injury requires either reviewer to have called it."""
        return int(max(self.call_a, self.call_b))


def adjudicate(
    attempt_ids: NDArray[np.int64],
    reviewer_calls: NDArray[np.int64],
    reviewers_min: int,
) -> tuple[AdjudicationPair, ...]:
    """Pair the reviewers of every attempt and record their calls."""
    if reviewer_calls.shape[1] < reviewers_min:
        raise ValueError("fewer reviewer calls than the minimum the protocol requires")
    pairs: list[AdjudicationPair] = []
    for position, attempt in enumerate(attempt_ids):
        calls = reviewer_calls[position]
        pairs.append(
            AdjudicationPair(
                attempt_id=int(attempt),
                reviewer_a=0,
                reviewer_b=1,
                call_a=int(calls[0]),
                call_b=int(calls[1]),
            )
        )
    return tuple(pairs)


def cohen_kappa(call_a: NDArray[np.int64], call_b: NDArray[np.int64]) -> float:
    """Chance-corrected agreement between the two review channels."""
    if call_a.shape[0] != call_b.shape[0]:
        raise ValueError("both review channels must cover the same attempts")
    if call_a.shape[0] == 0:
        return 0.0
    observed = float(np.mean(call_a == call_b))
    expected = float(np.mean(call_a) * np.mean(call_b) + (1.0 - np.mean(call_a)) * (1.0 - np.mean(call_b)))
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return float((observed - expected) / (1.0 - expected))


def endpoint_rate_by_stratum(
    labels: NDArray[np.int64],
    strata: NDArray[np.str_],
) -> dict[str, float]:
    """Share of the endpoint events per stratum, so a shift between strata stays visible."""
    result: dict[str, float] = {}
    for name in sorted(set(strata.tolist())):
        member = strata == name
        total = int(np.count_nonzero(member))
        if total == 0:
            result[str(name)] = 0.0
            continue
        result[str(name)] = float(np.count_nonzero(labels[member]) / total)
    return result


def nested_unit_counts(
    procedure_ids: NDArray[np.int64],
    lesion_ids: NDArray[np.int64],
    attempt_ids: NDArray[np.int64],
) -> dict[str, int]:
    """Attempts, lesions and procedures, to show the nesting the unit of analysis implies."""
    return {
        "procedures": int(len(set(procedure_ids.tolist()))),
        "lesions": int(len(set(lesion_ids.tolist()))),
        "attempts": int(len(set(attempt_ids.tolist()))),
    }


def adjudication_agreement(pairs: tuple[AdjudicationPair, ...]) -> dict[str, float]:
    if not pairs:
        return {"agreement": 0.0, "kappa": 0.0, "events": 0.0}
    call_a = np.asarray([pair.call_a for pair in pairs], dtype=np.int64)
    call_b = np.asarray([pair.call_b for pair in pairs], dtype=np.int64)
    return {
        "agreement": float(np.mean(call_a == call_b)),
        "kappa": cohen_kappa(call_a, call_b),
        "events": float(np.sum([pair.final for pair in pairs])),
    }


def endpoint_components(perforation: NDArray[np.int64], dissection: NDArray[np.int64]) -> dict[str, float]:
    """Perforation and dissection counts, with their union as the adjudicated endpoint."""
    if perforation.shape != dissection.shape:
        raise ValueError("the two components must cover the same attempts")
    union = np.asarray((perforation + dissection) > 0, dtype=np.int64)
    return {
        "perforation": float(np.sum(perforation)),
        "dissection": float(np.sum(dissection)),
        "union": float(np.sum(union)),
        "both": float(np.sum((perforation + dissection) == 2)),
    }
