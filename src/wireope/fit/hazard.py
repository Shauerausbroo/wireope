"""Selective state-space hazard head over the excursion history.

The rung conditionals are modelled over the sequence of excursion values h_1:t conditioned on
the anatomical context, using a selective state-space backbone: the step size, the input
matrix and the output matrix are all functions of the current input, so the model can hold
the excursion history differently in a long crossing than in a short one. A diagonal
state transition keeps the scan exact without an approximation.

Ref: Sec. 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from wireope.config import HazardConfig

DEFAULT_CONTEXT_DIM = 16


@dataclass(frozen=True)
class HazardOutput:
    logits: Tensor
    conditionals: Tensor
    states: Tensor
    final_state: Tensor


def _softplus(values: Tensor) -> Tensor:
    result: Tensor = torch.nn.functional.softplus(values)
    return result


class SelectiveScan(nn.Module):
    """Diagonal selective state-space layer with an explicit scan over the sequence."""

    def __init__(self, config: HazardConfig, input_dim: int) -> None:
        super().__init__()
        self.config = config
        inner = config.expand * config.d_model
        self.inner = inner
        self.input_projection = nn.Linear(input_dim, inner)
        self.x_projection = nn.Linear(inner, 2 * config.d_state + config.dt_rank)
        self.dt_projection = nn.Linear(config.dt_rank, inner)
        self.out_projection = nn.Linear(inner, config.d_model)
        self.conv = nn.Conv1d(
            inner, inner, kernel_size=config.d_conv, padding=config.d_conv - 1, groups=inner
        )
        self.a_log = nn.Parameter(torch.zeros(inner, config.d_state))
        self.d_parameter = nn.Parameter(torch.ones(inner))
        self.dt_bias = nn.Parameter(torch.zeros(inner))
        nn.init.uniform_(self.a_log, -1.0, 1.0)

    def _select(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projected: Tensor = self.x_projection(values)
        dt_rank, d_state = self.config.dt_rank, self.config.d_state
        delta_raw = projected[..., :dt_rank]
        b_term = projected[..., dt_rank : dt_rank + d_state]
        c_term = projected[..., dt_rank + d_state :]
        delta: Tensor = _softplus(self.dt_projection(delta_raw) + self.dt_bias)
        return delta, b_term, c_term

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        projected: Tensor = self.input_projection(inputs)
        convolved = self.conv(projected.transpose(1, 2))[:, :, : inputs.shape[1]]
        activated: Tensor = torch.nn.functional.silu(convolved).transpose(1, 2)
        delta, b_term, c_term = self._select(activated)
        a_term = -torch.exp(self.a_log)
        batch, steps, inner = activated.shape
        state = torch.zeros(batch, inner, self.config.d_state, dtype=activated.dtype)
        states = torch.zeros(batch, steps, inner, self.config.d_state, dtype=activated.dtype)
        for step in range(steps):
            step_delta = delta[:, step, :].unsqueeze(-1)
            a_bar = torch.exp(step_delta * a_term.unsqueeze(0))
            b_bar = step_delta * b_term[:, step, :].unsqueeze(1)
            state = a_bar * state + b_bar * activated[:, step, :].unsqueeze(-1)
            states[:, step] = state
        # states carry an inner axis and a state axis; the readout contracts the state axis.
        readout = torch.einsum("btis,bts->bti", states, c_term)
        output: Tensor = self.out_projection(readout + self.d_parameter * activated)
        return output, state


class HazardModel(nn.Module):
    """Per-rung hazard head: a state-space backbone followed by a rung-wise logit head."""

    def __init__(
        self,
        config: HazardConfig,
        rungs: int,
        context_dim: int = DEFAULT_CONTEXT_DIM,
    ) -> None:
        super().__init__()
        self.config = config
        self.rungs = rungs
        self.context_dim = context_dim
        self.input_projection = nn.Linear(1 + self.context_dim, config.d_model)
        self.layers = nn.ModuleList(SelectiveScan(config, config.d_model) for _ in range(config.n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in range(config.n_layers))
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, rungs)
        self.rung_bias = nn.Parameter(torch.zeros(rungs))

    def forward(self, sequence: Tensor, context: Tensor) -> HazardOutput:
        if sequence.dim() != 2:
            raise ValueError("the excursion sequence must be (batch, steps)")
        expanded = context.unsqueeze(1).expand(-1, sequence.shape[1], -1)
        hidden: Tensor = self.input_projection(torch.cat([sequence.unsqueeze(-1), expanded], dim=-1))
        final_state = torch.zeros(
            hidden.shape[0],
            self.config.expand * self.config.d_model,
            self.config.d_state,
            dtype=hidden.dtype,
        )
        states = hidden
        for layer, norm in zip(self.layers, self.norms):
            scanned, final_state = layer(hidden)
            hidden = norm(self.dropout(scanned) + hidden)
            states = hidden
        logits: Tensor = self.head(hidden) + self.rung_bias
        conditionals: Tensor = torch.sigmoid(logits)
        return HazardOutput(
            logits=logits,
            conditionals=conditionals,
            states=states,
            final_state=final_state,
        )

    def pooled_logits(self, sequence: Tensor, context: Tensor) -> Tensor:
        """Rung logits at the deepest available step, which is the attempt-level decision point."""
        output = self.forward(sequence, context)
        pooled: Tensor = output.logits[:, -1, :]
        return pooled


def hazard_loss(
    logits: Tensor,
    target_indicators: Tensor,
    rung_weights: Tensor,
) -> Tensor:
    """Weighted binary cross-entropy over the rung indicators.

    The loss is taken on the conditional rung probabilities, so a rung that the crossing only
    rarely reaches still contributes.
    """
    per_rung = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target_indicators, reduction="none"
    )
    weights = rung_weights.unsqueeze(0).expand_as(per_rung)
    total: Tensor = torch.sum(per_rung * weights) / torch.clamp(torch.sum(weights), min=1.0)
    return total


def survival_from_conditionals(conditionals: Tensor) -> Tensor:
    """Product form of Eq. (6) over the rung conditionals at the deepest sequence position."""
    per_position: Tensor = torch.prod(conditionals, dim=-1)
    final: Tensor = per_position[:, -1]
    return final


def context_tensor(
    features: Tensor,
    context_dim: int = DEFAULT_CONTEXT_DIM,
) -> Tensor:
    """Project a descriptor block onto the fixed context width the backbone expects."""
    width = context_dim
    if features.shape[1] == width:
        return features
    if features.shape[1] > width:
        reduced: Tensor = features[:, :width]
        return reduced
    padding = torch.zeros(features.shape[0], width - features.shape[1], dtype=features.dtype)
    padded: Tensor = torch.cat([features, padding], dim=1)
    return padded
