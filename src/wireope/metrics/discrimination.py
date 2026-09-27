"""Discrimination: AUROC with a DeLong interval, and the precision-recall area.

The anatomical and action-conditioned scores are compared on parity rather than superiority,
because anatomical models already carry most of the discrimination available in this
endpoint; the interval is the object the parity clause is written on.

Ref: Sec. 3.7 and Sec. 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class Discrimination:
    auroc: float
    lower: float
    upper: float
    positives: int
    negatives: int

    @property
    def width(self) -> float:
        return self.upper - self.lower


def auroc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Rank-based AUROC with ties counted at one half, computed without a library call."""
    positive = np.asarray(labels, dtype=np.int64) == 1
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(positive.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(np.asarray(scores, dtype=np.float64))
    rank_sum = float(np.sum(ranks[positive]))
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def midrank_matrix(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Structural components of the DeLong covariance: the midrank of every score."""
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    ranks = np.empty(sorted_scores.shape[0], dtype=np.float64)
    position = 0
    while position < sorted_scores.shape[0]:
        end = position
        while end + 1 < sorted_scores.shape[0] and sorted_scores[end + 1] == sorted_scores[position]:
            end += 1
        average_rank = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = average_rank
        position = end + 1
    return ranks


def delong_components(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-subject placement values for the positive and negative groups."""
    positive = np.asarray(labels, dtype=np.int64) == 1
    ranks = midrank_matrix(scores)
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(positive.shape[0] - n_pos)
    positive_ranks = ranks[positive]
    negative_ranks = ranks[~positive]
    v10 = (positive_ranks - (positive_ranks + 1.0) / 2.0) / max(n_neg, 1)
    v01 = 1.0 - (negative_ranks - (negative_ranks + 1.0) / 2.0) / max(n_pos, 1)
    return np.asarray(v10, dtype=np.float64), np.asarray(v01, dtype=np.float64)


def delong_interval(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    confidence: float = 0.95,
) -> Discrimination:
    """Normal-approximation interval from the DeLong variance of the AUROC."""
    positive = np.asarray(labels, dtype=np.int64) == 1
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(positive.shape[0] - n_pos)
    estimate = auroc(scores, labels)
    if n_pos < 2 or n_neg < 2 or not np.isfinite(estimate):
        return Discrimination(
            auroc=estimate, lower=float("nan"), upper=float("nan"), positives=n_pos, negatives=n_neg
        )
    v10, v01 = delong_components(scores, labels)
    variance = float(np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg)
    z_value = float(stats.norm.ppf(0.5 + confidence / 2.0))
    half = z_value * float(np.sqrt(max(variance, 1e-18)))
    return Discrimination(
        auroc=estimate,
        lower=float(max(0.0, estimate - half)),
        upper=float(min(1.0, estimate + half)),
        positives=n_pos,
        negatives=n_neg,
    )


def delong_test(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[float, float]:
    """Two-sided DeLong p-value for the difference of two correlated AUROCs."""
    v10_a, v01_a = delong_components(scores_a, labels)
    v10_b, v01_b = delong_components(scores_b, labels)
    n_pos = v10_a.shape[0]
    n_neg = v01_a.shape[0]
    if n_pos < 2 or n_neg < 2:
        return float("nan"), float("nan")
    difference = auroc(scores_a, labels) - auroc(scores_b, labels)
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    variance = float((s10[0, 0] - 2.0 * s10[0, 1] + s10[1, 1]) / n_pos)
    variance += float((s01[0, 0] - 2.0 * s01[0, 1] + s01[1, 1]) / n_neg)
    if variance <= 1e-18:
        return 0.0, 1.0
    z_value = difference / float(np.sqrt(variance))
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    return float(z_value), p_value


def average_precision(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Area under the precision-recall curve, computed on the ranked score order."""
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if np.count_nonzero(target) == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    sorted_target = target[order]
    cumulative = np.cumsum(sorted_target) / np.arange(1, sorted_target.shape[0] + 1)
    return float(np.sum(cumulative * sorted_target) / np.sum(sorted_target))


def average_precision_prevalence_axis(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    prevalence: float,
) -> dict[str, float]:
    """PR area reported against a stated baseline prevalence, with the no-skill level."""
    area = average_precision(scores, labels)
    observed = float(np.mean(labels)) if labels.shape[0] else 0.0
    return {
        "average_precision": area,
        "baseline_prevalence": float(prevalence),
        "observed_prevalence": observed,
        "no_skill": float(prevalence),
        "lift_over_no_skill": float(area - prevalence),
    }


def binormal_auroc_from_separation(separation: float) -> float:
    """AUROC implied by a standardised separation between the two score distributions."""
    return float(stats.norm.cdf(separation / np.sqrt(2.0)))


def parity_interval_overlap(
    first: Discrimination,
    second: Discrimination,
) -> dict[str, float]:
    """Overlap of two AUROC intervals, the quantity the parity clause is written on."""
    low = max(first.lower, second.lower)
    high = min(first.upper, second.upper)
    overlap = max(0.0, high - low)
    narrower = min(first.width, second.width)
    return {
        "overlap": overlap,
        "share_of_narrower": float(overlap / narrower) if narrower > 0.0 else 0.0,
        "difference": float(first.auroc - second.auroc),
    }


def discrimination_from_scores(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    confidence: float = 0.95,
) -> Discrimination:
    return delong_interval(scores, labels, confidence)
