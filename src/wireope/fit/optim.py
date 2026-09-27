"""Optimiser construction and parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from wireope.config import OptimizerConfig


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build the configured optimiser; the weight-decay group excludes norms and biases."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if config.name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            amsgrad=config.amsgrad,
        )
    if config.name == "adam":
        return torch.optim.Adam(
            groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            amsgrad=config.amsgrad,
        )
    if config.name == "sgd":
        return torch.optim.SGD(groups, lr=config.lr, momentum=0.9, nesterov=True)
    raise ValueError(f"unsupported optimiser '{config.name}'")


def global_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Norm of the gradient over the trainable parameters, before clipping."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared += float(torch.sum(parameter.grad.detach() ** 2))
    return float(squared**0.5)


def clip_gradients(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Clip gradients in place and return the norm that was measured before clipping."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    norm = global_gradient_norm(trainable)
    if max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    return norm
