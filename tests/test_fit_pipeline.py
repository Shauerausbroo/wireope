"""The fitting pipeline: encoder, hazard head, trainer, schedule, EMA and checkpointing."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wireope.config import OptimizerConfig
from wireope.fit.checkpoint import (
    load_checkpoint,
    payload_digest,
    prune_checkpoints,
    save_checkpoint,
)
from wireope.fit.dataset import (
    SEQUENCE_LENGTH,
    ExcursionDataset,
    descriptor_tensor,
    excursion_sequence,
    pad_or_truncate,
    rung_weight_tensor,
)
from wireope.fit.distributed import (
    detect_context,
    initialise_process_group,
    reduce_sum,
    shutdown,
    wrap_model,
)
from wireope.fit.ema import ExponentialMovingAverage, ema_divergence
from wireope.fit.hazard import (
    DEFAULT_CONTEXT_DIM,
    HazardModel,
    context_tensor,
    hazard_loss,
    survival_from_conditionals,
)
from wireope.fit.loop import (
    build_dataloader,
    gradient_norm_after_step,
    parameter_fingerprint,
    train_hazard_head,
)
from wireope.fit.optim import build_optimizer, clip_gradients, global_gradient_norm
from wireope.fit.outcome import (
    RungOutcomeModel,
    build_design,
    fit_rung_outcome,
    outcome_log_loss,
)
from wireope.fit.schedule import WarmupCosineSchedule, apply_schedule, cosine_weights
from wireope.geometry.encoder import (
    EnvelopeAdapter,
    EnvelopeEncoder,
    adapter_parameter_count,
    dice_loss,
)
from wireope.utils.seed import set_seed


def tiny_hazard_config(context: object) -> object:
    return context.config.hazard


def test_excursion_sequence_and_padding() -> None:
    sequence = excursion_sequence(8.0, 20)
    assert sequence.shape == (20,)
    assert float(sequence.max()) == pytest.approx(8.0)
    assert bool(np.all(np.diff(sequence) >= 0.0))
    with pytest.raises(ValueError):
        excursion_sequence(1.0, 0)
    assert pad_or_truncate(sequence, 10).shape == (10,)
    padded = pad_or_truncate(sequence[:3], 10)
    assert padded.shape == (10,) and padded[-1] == 0.0
    assert pad_or_truncate(sequence, 20).shape == (20,)


def test_dataset_shapes(context: object) -> None:
    dataset = ExcursionDataset(
        context.bundle.cohort,
        context.ladder,
        context.config.kinematics,
    )
    assert len(dataset) == context.bundle.cohort.attempts
    item = dataset[0]
    assert tuple(item["sequence"].shape) == (SEQUENCE_LENGTH,)
    assert tuple(item["context"].shape) == (DEFAULT_CONTEXT_DIM,)
    assert tuple(item["indicators"].shape) == (context.ladder.depth,)
    assert float(item["sequence"].max()) == pytest.approx(float(context.bundle.cohort.peaks()[0]), rel=0.05)
    weights = rung_weight_tensor(context.ladder)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_descriptor_and_context_tensors() -> None:
    descriptor = np.zeros(16, dtype=np.float64)
    tensor = descriptor_tensor(descriptor, 8)
    assert tuple(tensor.shape) == (8,)
    padded = context_tensor(torch.ones(3, 4), 8)
    assert tuple(padded.shape) == (3, 8)
    reduced = context_tensor(torch.ones(3, 12), 8)
    assert tuple(reduced.shape) == (3, 8)


def test_hazard_forward_and_loss(context: object) -> None:
    set_seed(3)
    config = tiny_hazard_config(context)
    model = HazardModel(config, rungs=3, context_dim=8)
    output = model(torch.zeros(4, 16), torch.zeros(4, 8))
    assert tuple(output.logits.shape) == (4, 16, 3)
    assert tuple(output.conditionals.shape) == (4, 16, 3)
    assert float(output.conditionals.min()) > 0.0
    assert float(output.conditionals.max()) < 1.0
    indicators = (torch.rand(4, 3) < 0.3).float()
    loss = hazard_loss(
        model.pooled_logits(torch.zeros(4, 16), torch.zeros(4, 8)), indicators, torch.full((3,), 1.0 / 3.0)
    )
    assert math.isfinite(float(loss.item()))
    survival = survival_from_conditionals(output.conditionals)
    assert tuple(survival.shape) == (4,)
    with pytest.raises(ValueError):
        model(torch.zeros(4, 16, 2), torch.zeros(4, 8))


def test_encoder_freezes_the_trunk(context: object) -> None:
    set_seed(5)
    config = context.config.envelope.encoder
    encoder = EnvelopeEncoder(config)
    before = [parameter.detach().clone() for parameter in encoder.trunk.parameters()]
    output = encoder(torch.rand(2, 1, 16, 16, 16))
    assert tuple(output.envelope_logits.shape)[0] == 2
    assert tuple(output.centreline_heatmap.shape)[0] == 2
    output.envelope_logits.mean().backward()  # type: ignore[no-untyped-call]
    assert all(parameter.grad is None for parameter in encoder.trunk.parameters())
    assert all(torch.equal(old, now) for old, now in zip(before, encoder.trunk.parameters()))
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in encoder.adapter.parameters()
    )
    assert adapter_parameter_count(config) > 0
    assert float(dice_loss(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2))) == pytest.approx(0.0)
    mismatch = torch.zeros(1, 1, 2, 2, 2)
    mismatch[0, 0, 0, 0, 0] = 1.0
    assert float(dice_loss(mismatch, torch.zeros(1, 1, 2, 2, 2))) > 0.0


def test_encoder_probability_and_adapter_module(context: object) -> None:
    config = context.config.envelope.encoder
    encoder = EnvelopeEncoder(config)
    probability = encoder.envelope_probability(torch.rand(1, 1, 16, 16, 16))
    with torch.no_grad():
        assert float(probability.min()) >= 0.0
        assert float(probability.max()) <= 1.0
    assert isinstance(encoder.adapter, EnvelopeAdapter)


def test_optimizer_and_schedule(context: object) -> None:
    set_seed(7)
    model = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    optimizer = build_optimizer(model.parameters(), context.config.optimizer)
    schedule_config = context.config.schedule
    schedule = WarmupCosineSchedule(schedule_config, steps_per_epoch=4)
    rate = apply_schedule(optimizer, schedule, 0.01, step=0)
    assert rate > 0.0
    assert schedule.get_lr(0) <= schedule.get_lr(schedule.total_steps - 1) + 1.0
    weights = cosine_weights(6, 0.1)
    assert len(weights) == 6
    assert weights[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(schedule_config, steps_per_epoch=0)
    loss = model.pooled_logits(torch.zeros(2, 8), torch.zeros(2, 8)).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    norm = clip_gradients(model.parameters(), 0.5)
    assert norm >= 0.0
    assert global_gradient_norm(model.parameters()) >= 0.0
    assert gradient_norm_after_step(model) >= 0.0
    assert math.isfinite(parameter_fingerprint(model))


def test_optimizer_rejects_unknown(context: object) -> None:
    config = context.config.optimizer
    broken = OptimizerConfig(
        name="nope",
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        grad_clip=config.grad_clip,
        ema_enabled=config.ema_enabled,
        ema_decay=config.ema_decay,
    )
    model = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    with pytest.raises(ValueError):
        build_optimizer(model.parameters(), broken)


def test_ema_tracks_parameters(context: object) -> None:
    set_seed(9)
    model = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    ema = ExponentialMovingAverage(model, 0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        loss = model.pooled_logits(torch.zeros(2, 8), torch.zeros(2, 8)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        ema.update(model)
    assert 0.0 < ema.decay_bias() < 1.0
    assert ema_divergence(model, ema.shadow) >= 0.0
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, 1.5)


def test_checkpoint_round_trip_and_pruning(tmp_path: Path, context: object) -> None:
    set_seed(11)
    model = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    optimizer = build_optimizer(model.parameters(), context.config.optimizer)
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, model, optimizer, epoch=2, step=9, seed=5, metrics={"loss": 1.0})
    assert len(digest) == 64
    reloaded = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    payload = load_checkpoint(path, reloaded, None)
    assert payload.seed == 5
    assert payload.epoch == 2
    assert all(
        torch.equal(left.detach(), right.detach())
        for left, right in zip(model.parameters(), reloaded.parameters())
    )
    assert len(payload_digest(payload)) == 64
    for index in range(3):
        save_checkpoint(
            tmp_path / f"extra_{index}.pt", model, None, epoch=index, step=index, seed=1, metrics={}
        )
    removed = prune_checkpoints(tmp_path, keep_last=2)
    assert len(removed) == 2
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, keep_last=0)


def test_distributed_context_helpers(context: object) -> None:
    single = detect_context(1)
    assert not single.enabled
    assert single.is_main
    assert single.effective_batch_multiplier == 1
    assert not initialise_process_group(single)
    model = HazardModel(tiny_hazard_config(context), rungs=2, context_dim=8)
    assert wrap_model(model, single) is model
    assert reduce_sum(2.0, single) == pytest.approx(2.0)
    shutdown(single)


def test_outcome_model_design_and_fit(context: object) -> None:
    bundle = context.bundle
    states = bundle.cohort.states
    aggressiveness = bundle.cohort.action_array()
    design = build_design(states, aggressiveness, include_action=True)
    assert design.features.shape[0] == len(states)
    assert design.features.shape[1] == 17
    state_only = build_design(states, aggressiveness, include_action=False)
    assert state_only.features.shape[1] == 16
    model = fit_rung_outcome(context.config.outcome, design.features, bundle.indicators)
    assert model.is_fitted
    prediction = model.predict(design.features[:10])
    assert prediction.shape == (10, bundle.ladder.depth)
    assert float(prediction.min()) > 0.0
    assert float(prediction.max()) <= 1.0
    assert model.predict_endpoint(design.features[:10]).shape == (10,)
    assert model.product_risk(design.features[:10]).shape == (10,)
    assert outcome_log_loss(prediction, bundle.indicators[:10]) >= 0.0
    unfitted = RungOutcomeModel(context.config.outcome, 3)
    with pytest.raises(RuntimeError):
        unfitted.predict(design.features[:2])
    constant = fit_rung_outcome(context.config.outcome, design.features, np.zeros((len(states), 2)))
    assert constant.predict(design.features[:5]).shape == (5, 2)
    with pytest.raises(ValueError):
        RungOutcomeModel(context.config.outcome, 3).fit(design.features, bundle.indicators[:, :1])


def test_dataloader_batches(context: object) -> None:
    dataset = ExcursionDataset(
        context.bundle.cohort,
        context.ladder,
        context.config.kinematics,
    )
    loader = build_dataloader(dataset, batch_size=32, shuffle=True, seed=1)
    batch = next(iter(loader))
    assert tuple(batch["sequence"].shape)[1] == SEQUENCE_LENGTH
    assert tuple(batch["indicators"].shape)[1] == context.ladder.depth


def test_training_loop_writes_and_reads_a_checkpoint(tmp_path: Path, context: object) -> None:
    dataset = ExcursionDataset(
        context.bundle.cohort,
        context.ladder,
        context.config.kinematics,
    )
    _, result = train_hazard_head(
        dataset=dataset,
        rungs=context.ladder.depth,
        hazard=context.config.hazard,
        optimizer_config=context.config.optimizer,
        schedule_config=context.config.schedule,
        seed=0,
        rung_weights=rung_weight_tensor(context.ladder),
        checkpoint_path=tmp_path / "loop.pt",
        max_epochs=2,
    )
    assert result.steps_run > 0
    assert result.epochs_run == 2
    assert len(result.checkpoint_digest) == 64
    assert result.gradient_norm >= 0.0
    assert math.isfinite(result.final_loss)
