"""Exponential moving average of the model parameters."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
from torch import nn


class ExponentialMovingAverage:
    """Keeps a shadow copy of the parameters and updates it after every step."""

    def __init__(self, module: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = decay
        self.shadow = copy.deepcopy(module)
        self.shadow.eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        self.updates += 1
        bias = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for target, source in zip(self.shadow.parameters(), module.parameters()):
            target.mul_(bias).add_(source.detach(), alpha=1.0 - bias)
        for target_buffer, source_buffer in zip(self.shadow.buffers(), module.buffers()):
            target_buffer.copy_(source_buffer)

    def decay_bias(self) -> float:
        """Effective decay after the warmup correction."""
        return float(min(self.decay, (1.0 + self.updates) / (10.0 + self.updates)))

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.shadow.parameters()


def ema_divergence(first: nn.Module, second: nn.Module) -> float:
    """Mean absolute gap between two parameter sets, used to report how far the EMA trails."""
    total = 0.0
    count = 0
    for left, right in zip(first.parameters(), second.parameters()):
        total += float(torch.sum(torch.abs(left.detach() - right.detach())))
        count += int(left.numel())
    if count == 0:
        return 0.0
    return float(total / count)
