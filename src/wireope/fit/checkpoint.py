"""Atomic checkpoint writer that carries the run seed and the configuration digest."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointPayload:
    epoch: int
    step: int
    seed: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    metrics: dict[str, float]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int,
    seed: int,
    metrics: dict[str, float],
) -> str:
    """Write a checkpoint through a staged temporary file and one replace.

    The seed travels in the payload so a resumed run restores the sampler state it was
    fitted with rather than drawing a new one.
    """
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "seed": int(seed),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": {str(key): float(value) for key, value in metrics.items()},
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> CheckpointPayload:
    """Restore a checkpoint and report the seed it was written with."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return CheckpointPayload(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        seed=int(payload["seed"]),
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
    )


def payload_digest(payload: CheckpointPayload) -> str:
    """Digest of the tensor payload rather than of the container bytes.

    A container digest changes on every write because the archive carries metadata; a
    payload digest is what makes two runs comparable.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps({"epoch": payload.epoch, "step": payload.step, "seed": payload.seed}).encode())
    for key in sorted(payload.model_state):
        tensor = payload.model_state[key]
        if torch.is_tensor(tensor):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prune_checkpoints(directory: Path, keep_last: int) -> list[str]:
    """Remove the oldest checkpoints beyond the retention count and report what was kept."""
    if keep_last < 1:
        raise ValueError("retention must keep at least one checkpoint")
    files = sorted(directory.glob("*.pt"), key=lambda item: item.stat().st_mtime)
    removed: list[str] = []
    for path in files[:-keep_last]:
        removed.append(path.name)
        path.unlink(missing_ok=True)
    return removed
