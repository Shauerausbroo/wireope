"""Learning-rate schedule: linear warmup followed by cosine decay."""

from __future__ import annotations

import math

import torch

from wireope.config import ScheduleConfig


class WarmupCosineSchedule:
    """Step-indexed schedule with a linear warmup and a cosine tail."""

    def __init__(self, config: ScheduleConfig, steps_per_epoch: int) -> None:
        if steps_per_epoch < 1:
            raise ValueError("a schedule needs at least one step per epoch")
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = max(1, config.epochs * steps_per_epoch)
        self.warmup_steps = int(round(config.warmup_fraction * self.total_steps))

    def get_lr(self, step: int) -> float:
        """Learning-rate multiplier at a step index, in [min_lr_fraction, 1]."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(max(step + 1, 1) / self.warmup_steps)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.config.min_lr_fraction
        return float(floor + (1.0 - floor) * cosine)


def apply_schedule(
    optimizer: torch.optim.Optimizer,
    schedule: WarmupCosineSchedule,
    base_lr: float,
    step: int,
) -> float:
    """Write the scheduled learning rate into every parameter group and return it."""
    multiplier = schedule.get_lr(step)
    learning_rate = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def cosine_weights(epochs: int, min_fraction: float) -> list[float]:
    """The multiplier sequence the schedule produces, for reporting and for tests."""
    schedule = ScheduleConfig(
        name="cosine",
        warmup_fraction=0.0,
        min_lr_fraction=min_fraction,
        epochs=epochs,
        batch_size=1,
        grad_accum=1,
        seeds=(0,),
        eval_every_epochs=1,
    )
    helper = WarmupCosineSchedule(schedule, 1)
    return [helper.get_lr(step) for step in range(helper.total_steps)]
