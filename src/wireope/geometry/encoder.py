"""Frozen 3D encoder and its trainable envelope adapter.

The encoder establishes the measurable lumen limit of the cohort and is frozen; the
adapter that maps its features onto the envelope and centreline heads is the only part
fitted on the training subset.

Ref: Sec. 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from wireope.config import EncoderConfig
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def _norm(channels: int, groups: int) -> nn.Module:
    # GroupNorm keeps a valid norm when a small volume downsamples to a spatial extent of
    # one, which an affine instance norm rejects.
    return nn.GroupNorm(min(groups, channels), channels)


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unsupported activation '{name}'")


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int, activation: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _norm(channels, groups)
        self.act1 = _activation(activation)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(channels, groups)
        self.act2 = _activation(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden: Tensor = self.act1(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        out: Tensor = self.act2(hidden + inputs)
        return out


class DownStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        groups: int,
        activation: str,
        stride: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm = _norm(out_channels, groups)
        self.act = _activation(activation)
        self.blocks = nn.ModuleList(
            ResidualBlock3D(out_channels, groups, activation) for _ in range(max(1, blocks))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden: Tensor = self.act(self.norm(self.projection(inputs)))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class VoxelTrunk3D(nn.Module):
    """Stem, downsampling stages and a bottleneck over a single-channel CT volume."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, config.stem_channels, kernel_size=3, padding=1, bias=False),
            _norm(config.stem_channels, config.norm_groups),
            _activation(config.activation),
        )
        stages: list[nn.Module] = []
        in_channels = config.stem_channels
        for index, channels in enumerate(config.stage_channels):
            blocks = config.stage_blocks[index] if index < len(config.stage_blocks) else 1
            stride = 1 if index == 0 else 2
            stages.append(
                DownStage3D(
                    in_channels,
                    channels,
                    blocks,
                    config.norm_groups,
                    config.activation,
                    stride=stride,
                )
            )
            in_channels = channels
        self.stages = nn.ModuleList(stages)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, config.bottleneck_channels, kernel_size=3, padding=1, bias=False),
            _norm(config.bottleneck_channels, config.norm_groups),
            _activation(config.activation),
        )
        self.out_channels = config.bottleneck_channels

    def forward(self, volumes: Tensor) -> Tensor:
        hidden: Tensor = self.stem(volumes)
        for stage in self.stages:
            hidden = stage(hidden)
        bottleneck: Tensor = self.bottleneck(hidden)
        return bottleneck


class EnvelopeAdapter(nn.Module):
    """Maps trunk features back to the full resolution envelope and centreline heads."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        channels = config.bottleneck_channels
        self.lateral1 = nn.Conv3d(channels, config.stage_channels[-1], kernel_size=1)
        self.lateral2 = nn.Conv3d(config.stage_channels[-1], config.stem_channels, kernel_size=1)
        self.norm1 = _norm(config.stage_channels[-1], config.norm_groups)
        self.norm2 = _norm(config.stem_channels, config.norm_groups)
        self.act = _activation(config.activation)
        self.heads = nn.ModuleDict(
            {
                "envelope_logits": nn.Conv3d(config.stem_channels, 1, kernel_size=1),
                "centreline_heatmap": nn.Conv3d(config.stem_channels, 1, kernel_size=1),
            }
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        hidden: Tensor = self.act(self.norm1(self.lateral1(features)))
        upsampled_stage: Tensor = nn.functional.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.act(self.norm2(self.lateral2(upsampled_stage)))
        upsampled_full: Tensor = nn.functional.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = upsampled_full
        outputs: dict[str, Tensor] = {}
        for head in self.heads:
            projected: Tensor = self.heads[head](hidden)
            outputs[head] = projected
        return outputs


@dataclass
class EncoderOutput:
    envelope_logits: Tensor
    centreline_heatmap: Tensor
    trunk_frozen: bool


class EnvelopeEncoder(nn.Module):
    """Frozen trunk plus adapter, with the trainable parameter set exposed for the optimiser."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = VoxelTrunk3D(config)
        self.adapter = EnvelopeAdapter(config)
        if config.frozen:
            self.freeze_trunk()

    def freeze_trunk(self) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        self.trunk.eval()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.adapter.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True) -> EnvelopeEncoder:
        super().train(mode)
        if self.config.frozen:
            # The trunk stays in evaluation mode whatever the adapter's mode is, so the
            # frozen normalisation statistics the envelope was established with are kept.
            self.trunk.eval()
        return self

    def forward(self, volumes: Tensor) -> EncoderOutput:
        with torch.no_grad():
            features = self.trunk(volumes)
        outputs = self.adapter(features)
        return EncoderOutput(
            envelope_logits=outputs["envelope_logits"],
            centreline_heatmap=outputs["centreline_heatmap"],
            trunk_frozen=self.config.frozen,
        )

    def envelope_probability(self, volumes: Tensor) -> Tensor:
        logits = self.forward(volumes).envelope_logits
        probability: Tensor = torch.sigmoid(logits)
        return probability


def dice_loss(predicted: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    predicted_flat = predicted.reshape(predicted.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = torch.sum(predicted_flat * target_flat, dim=1)
    denominator = torch.sum(predicted_flat, dim=1) + torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def adapter_parameter_count(config: EncoderConfig) -> int:
    encoder = EnvelopeEncoder(config)
    return int(sum(parameter.numel() for parameter in encoder.trainable_parameters()))
