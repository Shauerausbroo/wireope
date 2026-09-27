"""Torch views over the reconstructed excursions, for the hazard head.

Each item is one crossing attempt: the sequence of excursion values along the advancement
steps, the anatomical context, and the rung indicators the head is fitted against. Sequences
are held at a fixed length so a batch is a dense tensor, and the peak of the held sequence is
the attempt's excursion peak.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from wireope.cohort.release_cohort import ReleaseCohort
from wireope.config import KinematicsConfig
from wireope.fit.hazard import DEFAULT_CONTEXT_DIM, context_tensor
from wireope.laddering.severity import Ladder

SEQUENCE_LENGTH = 64
RISE_EXPONENT = 0.8


def excursion_sequence(peak_mm: float, steps: int) -> NDArray[np.float64]:
    """Monotone rise from zero to the excursion peak over the recorded advancement steps.

    The sequence shape is an engineering default: the manuscript records the peak and the
    event stream, not the per-step excursion, so the rise is reconstructed rather than read.
    """
    if steps < 1:
        raise ValueError("a sequence needs at least one step")
    position = (np.arange(steps, dtype=np.float64) + 1.0) / steps
    return np.asarray(peak_mm * np.power(position, RISE_EXPONENT), dtype=np.float64)


def pad_or_truncate(values: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    if values.shape[0] == length:
        return values
    if values.shape[0] > length:
        return np.asarray(values[:length], dtype=np.float64)
    padded = np.zeros(length, dtype=np.float64)
    padded[: values.shape[0]] = values
    return padded


class ExcursionDataset(Dataset[dict[str, Tensor]]):
    """Crossing attempts as fixed-length excursion sequences with anatomy context."""

    def __init__(
        self,
        cohort: ReleaseCohort,
        ladder: Ladder,
        kinematics: KinematicsConfig,
        sequence_length: int = SEQUENCE_LENGTH,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        member: NDArray[np.bool_] | None = None,
    ) -> None:
        self.ladder = ladder
        self.kinematics = kinematics
        self.sequence_length = sequence_length
        self.context_dim = context_dim
        self.indices = (
            np.arange(len(cohort.features), dtype=np.int64)
            if member is None
            else np.flatnonzero(np.asarray(member, dtype=np.bool_))
        )
        self.sequences = np.zeros((self.indices.shape[0], sequence_length), dtype=np.float32)
        self.contexts = np.zeros((self.indices.shape[0], context_dim), dtype=np.float32)
        self.indicators = np.zeros((self.indices.shape[0], ladder.depth), dtype=np.float32)
        peaks = cohort.peaks()
        descriptors = cohort.descriptor_matrix()
        for row, index in enumerate(self.indices):
            events = cohort.events[int(index)]
            steps = min(len(events), self.kinematics.max_steps, sequence_length)
            self.sequences[row] = pad_or_truncate(
                excursion_sequence(float(peaks[index]), max(steps, 1)), sequence_length
            ).astype(np.float32)
            context = descriptor_tensor(descriptors[index], context_dim)
            self.contexts[row] = context.numpy()
            self.indicators[row] = ladder.indicators(float(peaks[index])).astype(np.float32)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        return {
            "sequence": torch.from_numpy(self.sequences[item]),
            "context": torch.from_numpy(self.contexts[item]),
            "indicators": torch.from_numpy(self.indicators[item]),
        }


def descriptor_tensor(descriptor: NDArray[np.float64], context_dim: int) -> Tensor:
    row = torch.from_numpy(np.asarray(descriptor, dtype=np.float32)).unsqueeze(0)
    projected: Tensor = context_tensor(row, context_dim).squeeze(0)
    return projected


def rung_weight_tensor(ladder: Ladder) -> Tensor:
    return torch.tensor(np.asarray(ladder.weights, dtype=np.float32))
