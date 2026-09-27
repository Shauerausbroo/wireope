"""End-to-end smoke: two epochs on the smoke configuration, with a loss that falls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wireope.config import load_experiment
from wireope.fit.dataset import ExcursionDataset, rung_weight_tensor
from wireope.fit.loop import train_hazard_head
from wireope.studies.pipeline import prepare_bundle


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory: pytest.TempPathFactory, repo_root: Path) -> dict[str, float]:
    directory = tmp_path_factory.mktemp("smoke")
    config = load_experiment(repo_root, "_smoke")
    bundle = prepare_bundle(repo_root, "_smoke")
    dataset = ExcursionDataset(bundle.cohort, bundle.ladder, config.kinematics)
    _, result = train_hazard_head(
        dataset=dataset,
        rungs=bundle.ladder.depth,
        hazard=config.hazard,
        optimizer_config=config.optimizer,
        schedule_config=config.schedule,
        seed=0,
        rung_weights=rung_weight_tensor(bundle.ladder),
        checkpoint_path=directory / "smoke.pt",
    )
    return {
        "first_loss": result.first_loss,
        "final_loss": result.final_loss,
        "steps": float(result.steps_run),
        "epochs": float(result.epochs_run),
        "gradient_norm": result.gradient_norm,
        "digest_length": float(len(result.checkpoint_digest)),
        "seconds": result.seconds,
    }


def test_smoke_runs_two_epochs(smoke_result: dict[str, float], repo_root: Path) -> None:
    config = load_experiment(repo_root, "_smoke")
    assert smoke_result["epochs"] == float(config.schedule.epochs)
    assert smoke_result["steps"] > 0.0
    assert smoke_result["digest_length"] == 64.0
    assert smoke_result["gradient_norm"] >= 0.0


def test_smoke_loss_decreases(smoke_result: dict[str, float]) -> None:
    assert smoke_result["final_loss"] < smoke_result["first_loss"]


def test_smoke_record_is_written(tmp_path: Path, repo_root: Path) -> None:
    from wireope.cli.fit import main

    exit_code = main(
        [
            "--repo-root",
            str(repo_root),
            "--experiment",
            "_smoke",
            "--out",
            str(tmp_path),
            "--checkpoint-dir",
            str(tmp_path),
            "--max-epochs",
            "1",
        ]
    )
    assert exit_code == 0
    payload = json.loads((tmp_path / "fit_summary.json").read_text())
    assert payload["experiment"] == "_smoke"
    assert payload["hazard"]["steps"] > 0
    assert payload["hazard"]["loss_decreased"] in {True, False}
