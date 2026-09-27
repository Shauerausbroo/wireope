"""Training loop for the hazard head, with checkpointing and an EMA copy.

The loop is the execution path the release verifies end to end: sequences are read, the head
runs forward, the rung loss is computed, gradients are taken, parameters are updated, and a
checkpoint is written and read back. The same loop runs on the smoke configuration used by
the unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from wireope.config import HazardConfig, OptimizerConfig, ScheduleConfig
from wireope.fit.checkpoint import load_checkpoint, save_checkpoint
from wireope.fit.dataset import ExcursionDataset
from wireope.fit.ema import ExponentialMovingAverage, ema_divergence
from wireope.fit.hazard import HazardModel, hazard_loss
from wireope.fit.optim import build_optimizer, clip_gradients
from wireope.fit.schedule import WarmupCosineSchedule, apply_schedule
from wireope.schema import FitHistory, ProgressUpdate
from wireope.utils.logging_setup import get_logger
from wireope.utils.seed import set_seed

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FitResult:
    history: FitHistory
    epochs_run: int
    steps_run: int
    final_loss: float
    first_loss: float
    gradient_norm: float
    checkpoint_digest: str
    ema_divergence: float
    seconds: float
    seed: int

    @property
    def loss_decreased(self) -> bool:
        return bool(
            np.isfinite(self.first_loss)
            and np.isfinite(self.final_loss)
            and self.final_loss < self.first_loss
        )


def build_dataloader(
    dataset: ExcursionDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )


def train_hazard_head(
    dataset: ExcursionDataset,
    rungs: int,
    hazard: HazardConfig,
    optimizer_config: OptimizerConfig,
    schedule_config: ScheduleConfig,
    seed: int,
    rung_weights: Tensor,
    checkpoint_path: Path | None = None,
    context_dim: int = 16,
    max_epochs: int | None = None,
) -> tuple[HazardModel, FitResult]:
    """Fit the hazard head and report the loss trajectory it actually produced."""
    set_seed(seed)
    model = HazardModel(hazard, rungs=rungs, context_dim=context_dim)
    loader = build_dataloader(dataset, schedule_config.batch_size, True, seed)
    optimizer = build_optimizer(model.parameters(), optimizer_config)
    steps_per_epoch = max(1, len(loader))
    schedule = WarmupCosineSchedule(schedule_config, steps_per_epoch)
    ema = (
        ExponentialMovingAverage(model, optimizer_config.ema_decay)
        if optimizer_config.ema_enabled
        else None
    )
    weights = rung_weights
    epochs = max_epochs if max_epochs is not None else schedule_config.epochs
    history = FitHistory()
    step = 0
    started = time.time()
    last_gradient_norm = 0.0
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            learning_rate = apply_schedule(optimizer, schedule, optimizer_config.lr, step)
            output = model(batch["sequence"], batch["context"])
            # The decision point is the deepest step of the reconstructed crossing.
            loss = hazard_loss(output.logits[:, -1, :], batch["indicators"], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            last_gradient_norm = clip_gradients(model.parameters(), optimizer_config.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            history.append(
                ProgressUpdate(
                    epoch=epoch,
                    step=step,
                    loss=float(loss.detach().item()),
                    learning_rate=learning_rate,
                    gradient_norm=last_gradient_norm,
                )
            )
            step += 1
        LOGGER.info("epoch %d finished, last loss %.6f", epoch, history.last_loss)
    digest = ""
    if checkpoint_path is not None:
        digest = save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=epochs - 1,
            step=step,
            seed=seed,
            metrics={"final_loss": history.last_loss, "first_loss": history.first_loss},
        )
        restored = load_checkpoint(checkpoint_path, model, optimizer)
        if restored.seed != seed:
            raise RuntimeError("the checkpoint did not carry the run seed")
    divergence = 0.0
    if ema is not None:
        divergence = ema_divergence(model, ema.shadow)
    return model, FitResult(
        history=history,
        epochs_run=epochs,
        steps_run=step,
        final_loss=history.last_loss,
        first_loss=history.first_loss,
        gradient_norm=last_gradient_norm,
        checkpoint_digest=digest,
        ema_divergence=divergence,
        seconds=float(time.time() - started),
        seed=seed,
    )


def gradient_norm_after_step(model: nn.Module) -> float:
    """Norm of the gradient currently held by the model, used by the verification pass."""
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() ** 2))
    return float(total**0.5)


def parameter_fingerprint(model: nn.Module) -> float:
    """Cheap scalar summary of the parameters, used to detect that an update happened."""
    total = 0.0
    for parameter in model.parameters():
        total += float(torch.sum(parameter.detach()))
    return float(total)
