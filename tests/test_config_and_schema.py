"""Configuration loading, inheritance, and the data dictionary."""

from __future__ import annotations

from pathlib import Path

import pytest

from wireope.cohort.schema import DATA_DICTIONARY, cohort_totals, describe, site_table
from wireope.config import ConfigError, experiment_names, load_experiment
from wireope.studies.pipeline import ArmSwitches


def test_every_experiment_loads(repo_root: Path) -> None:
    names = experiment_names(repo_root)
    assert names, "the release must ship experiment configurations"
    for name in names:
        config = load_experiment(repo_root, name)
        assert config.experiment.name == name
        assert config.ladder.depth >= 1


def test_inheritance_threads_both_ways(repo_root: Path) -> None:
    parent = load_experiment(repo_root, "main")
    child = load_experiment(repo_root, "ablation_ladder_depth_six")
    assert parent.ladder.depth == 4
    assert child.ladder.depth == 6
    assert child.experiment.components == parent.experiment.components


def test_smoke_configuration_is_smaller(repo_root: Path) -> None:
    main = load_experiment(repo_root, "main")
    smoke = load_experiment(repo_root, "_smoke")
    assert smoke.release_cohort.attempts < main.release_cohort.attempts
    assert smoke.schedule.epochs < main.schedule.epochs
    assert smoke.hazard.d_model == main.hazard.d_model


def test_effective_batch_matches_the_product(repo_root: Path) -> None:
    config = load_experiment(repo_root, "main")
    schedule = config.schedule
    runtime = config.runtime
    assert schedule.effective_batch_size == schedule.batch_size * schedule.grad_accum
    assert runtime.world_size >= 1


def test_unknown_component_is_rejected(repo_root: Path, tmp_path: Path) -> None:
    config_root = tmp_path / "configs" / "experiment"
    config_root.mkdir(parents=True)
    (config_root / "broken.yaml").write_text("name: broken\ncomponents:\n  - cohort/nowhere\n")
    with pytest.raises(ConfigError):
        load_experiment(tmp_path, "broken")


def test_compute_target_is_not_invented(repo_root: Path) -> None:
    config = load_experiment(repo_root, "main")
    assert config.runtime.compute_target_reported == "single_processor_graphics_node"
    assert config.runtime.accelerator_model == "COMPUTE_NOT_REPORTED"


def test_cohort_dictionary_and_totals(bundle: object) -> None:
    cohort = bundle.cohort
    tables = site_table(
        cohort.site_array(),
        cohort.region_array(),
        cohort.injury_flags(),
        cohort.operator_array(),
        cohort.vendor_array(),
    )
    totals = cohort_totals(tables)
    assert totals["attempts"] == float(cohort.attempts)
    assert totals["injury_events"] == float(cohort.injury_flags().sum())
    assert describe(("attempt_id", "injury")).keys() == {"attempt_id", "injury"}


def test_dictionary_is_complete() -> None:
    for key in ("patient_id", "procedure_id", "lesion_id", "events", "strategy"):
        assert key in DATA_DICTIONARY
    with pytest.raises(KeyError):
        describe(("not_a_field",))


def test_arm_switches_uses_action_term() -> None:
    assert ArmSwitches().uses_action_term
    assert not ArmSwitches(envelope_excursion_reconstruction=False).uses_action_term
