"""Weight-control and pessimism sweep, with the conservative end as the headline.

The sweep is part of the design rather than a sensitivity analysis added afterwards. It
reports the whole grid, adopts the conservative end as the headline, and reports the
optimistic end as a bound, mirroring the published sweep in which an apparent gain
collapsed from +32% to +3% as the pessimism coefficient moved from 0.001 to 0.5.

Ref: Sec. 3.5, Sec. 4.3 and Table 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import SweepConfig


@dataclass(frozen=True)
class SweepCell:
    stratum: str
    axis: str
    clipping_label: str
    clipping: float
    pessimism_label: str
    pessimism: float
    relative_error: float
    effective_sample_size: float
    overlap_coefficient: float
    selection_accuracy: float


@dataclass(frozen=True)
class SweepHeadline:
    conservative_gain: float
    optimistic_gain: float
    ratio: float
    clipping: float
    pessimism: float
    label: str


def sweep_grid(config: SweepConfig) -> tuple[tuple[float, float, str, str], ...]:
    """The clipping-by-pessimism grid, clipped values paired across the pessimism axis."""
    cells: list[tuple[float, float, str, str]] = []
    for clipping, clipping_label in zip(config.clipping_values, config.clipping_labels):
        for pessimism, pessimism_label in zip(config.pessimism_values, config.pessimism_labels):
            cells.append((clipping, pessimism, clipping_label, pessimism_label))
    return tuple(cells)


def pessimism_radius_scale(pessimism: float, base_radius: float) -> float:
    """Inflate a confidence radius by the pessimism coefficient.

    A coefficient near zero leaves the radius nearly untouched; a coefficient near one half
    widens it, which is the direction in which the cited published gain collapsed.
    """
    if not 0.0 <= pessimism <= 1.0:
        raise ValueError("pessimism coefficient must lie in [0, 1]")
    return float(base_radius * (1.0 + 2.0 * pessimism))


def clipping_effect(clipping: float, unscaled_relative_error: float) -> float:
    """Effect of aggressive weight clipping on the relative error.

    Clipping removes the largest importance weights, so the variance of the reweighted mean
    falls while the bias from truncated weights grows; the sweep reports the balance rather
    than assuming a direction.
    """
    if clipping <= 0.0:
        return float(unscaled_relative_error)
    return float(unscaled_relative_error * np.sqrt(clipping))


def conservative_headline(
    cells: tuple[SweepCell, ...],
    config: SweepConfig,
) -> SweepHeadline:
    """Select the conservative end of the grid as the headline and the other end as a bound."""
    if not cells:
        raise ValueError("the sweep produced no cells")
    headline_cells = [
        cell
        for cell in cells
        if cell.clipping_label == config.clipping_headline
        and cell.pessimism_label == config.pessimism_headline
    ]
    bound_cells = [
        cell
        for cell in cells
        if cell.clipping_label != config.clipping_headline
        or cell.pessimism_label != config.pessimism_headline
    ]
    conservative = headline_cells[0] if headline_cells else cells[0]
    optimistic = max(bound_cells, key=lambda cell: cell.selection_accuracy) if bound_cells else conservative
    conservative_gain = conservative.selection_accuracy
    optimistic_gain = optimistic.selection_accuracy
    ratio = float(optimistic_gain / conservative_gain) if conservative_gain > 0.0 else float("nan")
    return SweepHeadline(
        conservative_gain=float(conservative_gain),
        optimistic_gain=float(optimistic_gain),
        ratio=ratio,
        clipping=float(conservative.clipping),
        pessimism=float(conservative.pessimism),
        label=f"clipping={conservative.clipping_label}, pessimism={conservative.pessimism_label}",
    )


def sweep_monotone_in_clipping(cells: tuple[SweepCell, ...]) -> bool:
    """More aggressive clipping must not increase the relative error inside a pessimism level."""
    labels = sorted({cell.clipping_label for cell in cells})
    for pessimism in sorted({cell.pessimism_label for cell in cells}):
        series = [
            np.mean(
                [
                    cell.relative_error
                    for cell in cells
                    if cell.clipping_label == label and cell.pessimism_label == pessimism
                ]
            )
            for label in labels
        ]
        if any(later > earlier + 1e-9 for earlier, later in zip(series, series[1:])):
            return False
    return True


def sweep_monotone_in_pessimism(cells: tuple[SweepCell, ...]) -> bool:
    """More pessimism must not improve the selection accuracy, inside a clipping level."""
    labels = sorted({cell.pessimism_label for cell in cells})
    for clipping in sorted({cell.clipping_label for cell in cells}):
        series = [
            np.mean(
                [
                    cell.selection_accuracy
                    for cell in cells
                    if cell.pessimism_label == label and cell.clipping_label == clipping
                ]
            )
            for label in labels
        ]
        if any(later > earlier + 1e-9 for earlier, later in zip(series, series[1:])):
            return False
    return True


def stratum_labels(cells: tuple[SweepCell, ...]) -> tuple[str, ...]:
    return tuple(sorted({cell.stratum for cell in cells}))


def cell_array(cells: tuple[SweepCell, ...], attribute: str) -> NDArray[np.float64]:
    return np.asarray([float(getattr(cell, attribute)) for cell in cells], dtype=np.float64)
