"""Kinematic reconstruction from the intra-procedural device event stream.

The event stream carries advancement, rotation, dwell and retraction events. Force,
torque, progression and rotation as continuous signals are not logged by the robotic
system, so this integration is an interpretation of the recorded motion and confines
every claim made about the delivered action.

Ref: Sec. 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import KinematicsConfig
from wireope.schema import EventRecord
from wireope.utils.linalg import EPS

ADVANCE = "advance"
RETRACT = "retract"
ROTATE_CW = "rotate_clockwise"
ROTATE_CCW = "rotate_counterclockwise"
DWELL = "dwell"


@dataclass(frozen=True)
class TipTrajectory:
    positions_mm: NDArray[np.float64]
    headings_rad: NDArray[np.float64]
    dwell_s: NDArray[np.float64]
    step_index: NDArray[np.int64]
    events_consumed: int

    @property
    def steps(self) -> int:
        return int(self.positions_mm.shape[0])

    @property
    def path_length_mm(self) -> float:
        if self.steps < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.positions_mm, axis=0), axis=1)))

    @property
    def rotation_turns(self) -> float:
        if self.headings_rad.size == 0:
            return 0.0
        return float(np.abs(np.diff(self.headings_rad)).sum() / (2.0 * np.pi))

    def curvature(self) -> NDArray[np.float64]:
        if self.steps < 3:
            return np.zeros(0, dtype=np.float64)
        deltas = np.diff(self.positions_mm, axis=0)
        angles = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        turning = np.abs(np.diff(angles))
        return np.asarray(turning, dtype=np.float64)


def integrate_event_stream(events: list[EventRecord], config: KinematicsConfig) -> TipTrajectory:
    """Integrate the discrete event stream into tip positions, headings and dwell times."""
    positions: list[NDArray[np.float64]] = []
    headings: list[float] = []
    dwell: list[float] = []
    steps: list[int] = []
    position = np.zeros(3, dtype=np.float64)
    heading = 0.0
    for event in events[: config.max_steps]:
        kind = str(event["kind"])
        magnitude = float(event["magnitude"])
        if kind == ADVANCE:
            step_mm = config.advance_step_mm * max(magnitude, 0.0)
            position = position + step_mm * np.asarray([np.cos(heading), np.sin(heading), 0.0])
            dwell.append(config.timestep_s)
        elif kind == RETRACT:
            step_mm = config.advance_step_mm * max(magnitude, 0.0)
            position = position - step_mm * np.asarray([np.cos(heading), np.sin(heading), 0.0])
            dwell.append(config.timestep_s)
        elif kind == ROTATE_CW:
            heading, position = _rotate_tip(heading, position, magnitude, config, direction=-1.0)
            dwell.append(0.0)
        elif kind == ROTATE_CCW:
            heading, position = _rotate_tip(heading, position, magnitude, config, direction=1.0)
            dwell.append(0.0)
        elif kind == DWELL:
            dwell.append(config.timestep_s * max(magnitude, 1.0))
            positions.append(position.copy())
            headings.append(heading)
            steps.append(int(event["step"]))
            continue
        else:
            raise ValueError(f"unsupported device event '{kind}'")
        positions.append(position.copy())
        headings.append(heading)
        steps.append(int(event["step"]))
    if not positions:
        raise ValueError("event stream produced no tip samples")
    return TipTrajectory(
        positions_mm=np.vstack(positions),
        headings_rad=np.asarray(headings, dtype=np.float64),
        dwell_s=np.asarray(dwell, dtype=np.float64),
        step_index=np.asarray(steps, dtype=np.int64),
        events_consumed=min(len(events), config.max_steps),
    )


def _rotate_tip(
    heading: float,
    position: NDArray[np.float64],
    magnitude: float,
    config: KinematicsConfig,
    direction: float,
) -> tuple[float, NDArray[np.float64]]:
    """Rotate the tip about its current contact point and apply a curvature regulariser."""
    angle = direction * np.deg2rad(config.rotation_deg_per_event * max(magnitude, 0.0))
    new_heading = heading + angle
    lever = np.asarray([np.cos(heading), np.sin(heading), 0.0])
    swept = lever * np.sin(abs(angle)) * config.curvature_regularisation
    return new_heading, np.asarray(position + swept, dtype=np.float64)


def compress_event_stream(events: list[EventRecord], max_steps: int) -> list[EventRecord]:
    """Keep the event stream within the step budget, preserving the event order."""
    if len(events) <= max_steps:
        return list(events)
    index = np.linspace(0, len(events) - 1, max_steps).round().astype(np.int64)
    return [events[int(item)] for item in index]


def event_histogram(events: list[EventRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def action_norm(trajectory: TipTrajectory, reference_radius_mm: float) -> float:
    """Scaled action magnitude: rotation turns plus advancement in units of the lumen radius."""
    advancement = trajectory.path_length_mm / max(reference_radius_mm, EPS)
    return float(advancement + trajectory.rotation_turns)
