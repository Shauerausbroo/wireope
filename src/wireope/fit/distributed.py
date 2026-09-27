"""Distributed helpers.

The framework is single-process by default; when a world size above one is requested the
training loop is wrapped so the same code path runs under torchrun without further changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    enabled: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def effective_batch_multiplier(self) -> int:
        return max(self.world_size, 1)


def detect_context(requested_world_size: int) -> DistributedContext:
    """Read the torchrun environment, falling back to a single process."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(max(requested_world_size, 1))))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        enabled=world_size > 1,
    )


def initialise_process_group(context: DistributedContext, backend: str = "gloo") -> bool:
    """Initialise the process group when the launch has more than one process."""
    if not context.enabled:
        return False
    if not torch.distributed.is_available():
        raise RuntimeError("a world size above one was requested but torch.distributed is unavailable")
    if torch.distributed.is_initialized():
        return True
    torch.distributed.init_process_group(backend=backend, rank=context.rank, world_size=context.world_size)
    return True


def wrap_model(model: nn.Module, context: DistributedContext, device: int = 0) -> nn.Module:
    """Wrap in DDP when the process group is live, otherwise return the model unchanged."""
    if not context.enabled or not torch.distributed.is_initialized():
        return model
    wrapped: nn.Module = nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if torch.cuda.is_available() else None
    )
    return wrapped


def reduce_sum(value: float, context: DistributedContext) -> float:
    """Sum a scalar across ranks, used for the effective batch arithmetic in the report."""
    if not context.enabled or not torch.distributed.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64)
    torch.distributed.all_reduce(tensor)
    return float(tensor.item())


def shutdown(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
