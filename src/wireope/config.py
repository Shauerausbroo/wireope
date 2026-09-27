"""Typed configuration for every stage of the certification pipeline.

An experiment file names the configuration sections it composes and the switches it
flips; the loader merges the section files, applies the experiment's overrides, and
returns one typed object. Unknown keys are rejected so a silent typo cannot change a
reported number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing a required field."""


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return data[key]


def _reject_unknown(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")


@dataclass(frozen=True)
class SiteSpec:
    site: str
    region: str
    attempts: int
    injury_events: int


@dataclass(frozen=True)
class ClinicalCohortConfig:
    name: str
    access: str
    record_level_available: bool
    regions: tuple[str, ...]
    sites: tuple[SiteSpec, ...]
    accrual: str
    unit_of_analysis: str
    nesting: tuple[str, ...]
    outcome_components: tuple[str, ...]
    reviewers_min: int
    blinded_to_reconstruction: bool
    secondary_endpoints: tuple[str, ...]
    strata: Mapping[str, tuple[str, ...]]
    subgroups: tuple[str, ...]
    literature_perforation_prevalence: float

    @property
    def attempts(self) -> int:
        return sum(site.attempts for site in self.sites)

    @property
    def injury_events(self) -> int:
        return sum(site.injury_events for site in self.sites)


@dataclass(frozen=True)
class CalibrationSpec:
    parameter: str
    method: str
    tolerance: float
    max_iterations: int
    latent_noise_scale: float


@dataclass(frozen=True)
class ReleaseCohortConfig:
    name: str
    version: int
    seed: int
    volume_shape: tuple[int, int, int]
    voxel_mm: float
    attempts: int
    site_targets: tuple[SiteSpec, ...]
    prevalence_target: float
    calibration: CalibrationSpec
    coefficients: Mapping[str, float]
    rung_base_q: float
    rung_action_slope: float
    lesion_overlap_range: tuple[float, float]
    nominal_lumen_radius_mm: float
    descriptor_mix: Mapping[str, Mapping[str, float]]
    site_shift: Mapping[str, Mapping[str, float]]
    behaviour: Mapping[str, float]


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    sites: tuple[str, ...]
    role: str
    configuration: str = ""
    regions: tuple[str, ...] = ()
    disjoint_from: str = ""


@dataclass(frozen=True)
class LeakageGuardConfig:
    patient_disjoint: bool
    operator_disjoint_across_partitions: bool
    vendor_disjoint_across_partitions: bool
    calibration_share_disjoint: bool
    fail_on_overlap: bool


@dataclass(frozen=True)
class PublicCorpusSpec:
    key: str
    label: str
    logging_regime: str
    licence: str
    source: str
    role: str
    rows_millions: int | None = None


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    architecture: str
    in_channels: int
    stem_channels: int
    stage_channels: tuple[int, ...]
    stage_blocks: tuple[int, ...]
    bottleneck_channels: int
    norm: str
    norm_groups: int
    activation: str
    frozen: bool
    output_heads: tuple[str, ...]


@dataclass(frozen=True)
class EnvelopeConfig:
    encoder: EncoderConfig
    surface_method: str
    surface_level: float
    spacing_mm: float
    distance_clip_mm: float
    signed_distance: bool
    centreline_resample_step_mm: float


@dataclass(frozen=True)
class DescriptorConfig:
    calcification_grades: tuple[int, ...]
    calcification_arc_bins_deg: tuple[int, ...]
    calcification_severity_weights: Mapping[int, float]
    cap_classes: tuple[str, ...]
    lesion_length_bins_mm: tuple[int, ...]
    territory_classes: tuple[str, ...]
    stages: tuple[str, ...]


@dataclass(frozen=True)
class KinematicsConfig:
    events: tuple[str, ...]
    timestep_s: float
    advance_step_mm: float
    rotation_deg_per_event: float
    max_steps: int
    curvature_regularisation: float
    tip_mean_absolute_error_mm: float


@dataclass(frozen=True)
class RegistrationConfig:
    method: str
    transform: str
    iterations: int
    tolerance_mm: float


@dataclass(frozen=True)
class UncertaintyConfig:
    tip_tolerance_mm: float
    segmentation_tolerance_mm: float
    combination: str
    confidence_multiplier: float
    dwell_weighting: str


@dataclass(frozen=True)
class LadderConfig:
    depth: int
    additional_depths: tuple[int, ...]
    tau_mm: tuple[float, ...]
    tau_inj_mm: float
    observed_injury_defines_top_rung: bool
    conditional_frequency_band: tuple[float, float]
    resolution_rule: str
    resolution_band_mm: float


@dataclass(frozen=True)
class LadderWeightsConfig:
    scheme: str
    value: float
    normalise: bool
    require_non_negative: bool


@dataclass(frozen=True)
class PropensityConfig:
    branch_features: tuple[str, ...]
    action_features: tuple[str, ...]
    l2: float
    max_iter: int
    clipping_thresholds: tuple[float, ...]
    clipping_labels: tuple[str, ...]
    default_clipping: float
    fidelity_metrics: tuple[str, ...]
    fidelity_axes: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEstimator:
    key: str
    label: str
    family: str


@dataclass(frozen=True)
class EstimatorLibraryConfig:
    candidates: tuple[CandidateEstimator, ...]
    comparator: CandidateEstimator
    selection_criterion: str
    reported_diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class HazardConfig:
    architecture: str
    d_model: int
    d_state: int
    expand: int
    n_layers: int
    d_conv: int
    dt_rank: int
    dropout: float
    head: str
    sequence: str
    conditioning: str
    hazard_weight: float = 1.0
    direct_weight: float = 1.0


@dataclass(frozen=True)
class OutcomeConfig:
    form: str
    rung_features: tuple[str, ...]
    l2: float
    max_iter: int
    clip_observations: bool


@dataclass(frozen=True)
class ConfidenceConfig:
    delta: float
    bounded_range_b: float
    bound_type: str
    fallback_bound_type: str
    report_both: bool
    ess_definition: str
    overlap_coefficient: str
    clip_denominator: float
    union_bound_per_rung: bool


@dataclass(frozen=True)
class SupportConfig:
    floor_c: float
    axis: tuple[str, ...]
    event_wise: bool
    unconstrained_fallback: bool
    report_abstention: bool


@dataclass(frozen=True)
class SweepConfig:
    risk_margin_m: float
    pessimism_values: tuple[float, ...]
    pessimism_labels: tuple[str, ...]
    pessimism_headline: str
    clipping_values: tuple[float, ...]
    clipping_labels: tuple[str, ...]
    clipping_headline: str
    net_benefit_thresholds: tuple[float, ...]
    primary_utility: str
    parity_margin_pp: float


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    grad_clip: float
    ema_enabled: bool
    ema_decay: float


@dataclass(frozen=True)
class ScheduleConfig:
    name: str
    warmup_fraction: float
    min_lr_fraction: float
    epochs: int
    batch_size: int
    grad_accum: int
    seeds: tuple[int, ...]
    eval_every_epochs: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


@dataclass(frozen=True)
class RuntimeConfig:
    world_size: int
    device: str
    precision: str
    amp: bool
    tf32: bool
    num_workers: int
    pin_memory: bool
    deterministic: bool
    checkpoint_dir: str
    keep_last: int
    volume_shape: tuple[int, int, int]
    compute_target_reported: str
    accelerator_model: str
    vram_gb: str
    wall_clock: str
    storage_gb: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    components: tuple[str, ...]
    switches: Mapping[str, Any]
    report: Mapping[str, Any]
    sampling: Mapping[str, Any] = field(default_factory=dict)
    settings: tuple[str, ...] = ()
    grid: Mapping[str, Any] = field(default_factory=dict)
    axes: tuple[Mapping[str, Any], ...] = ()
    corpora: tuple[str, ...] = ()
    arms: tuple[str, ...] = ()
    readers: Mapping[str, Any] = field(default_factory=dict)
    overrides: Mapping[str, Any] = field(default_factory=dict)
    path: str = ""


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentSpec
    clinical: ClinicalCohortConfig
    release_cohort: ReleaseCohortConfig
    partitions: tuple[PartitionSpec, ...]
    leakage: LeakageGuardConfig
    corpora: tuple[PublicCorpusSpec, ...]
    envelope: EnvelopeConfig
    descriptors: DescriptorConfig
    kinematics: KinematicsConfig
    registration: RegistrationConfig
    uncertainty: UncertaintyConfig
    ladder: LadderConfig
    ladder_weights: LadderWeightsConfig
    propensity: PropensityConfig
    library: EstimatorLibraryConfig
    hazard: HazardConfig
    outcome: OutcomeConfig
    confidence: ConfidenceConfig
    support: SupportConfig
    sweep: SweepConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    runtime: RuntimeConfig
    raw: Mapping[str, Any]


SECTION_FILES: Mapping[str, str] = {
    "cohort/clinical": "cohort",
    "cohort/release": "release_cohort",
    "cohort/partitions": "partitions",
    "cohort/public_corpora": "corpora",
    "geometry/envelope": "envelope",
    "geometry/descriptors": "descriptors",
    "traces/kinematics": "kinematics",
    "traces/uncertainty": "uncertainty",
    "laddering/severity": "ladder",
    "laddering/weights": "ladder_weights",
    "behavior/propensity": "propensity",
    "estimators/library": "library",
    "estimators/hazard": "hazard",
    "estimators/outcome": "outcome",
    "bounds/confidence": "confidence",
    "certification/support": "support",
    "certification/sweep": "sweep",
    "fit/optimizer": "optimizer",
    "fit/schedule": "schedule",
    "fit/runtime": "runtime",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path.name}")
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return {str(key): value for key, value in loaded.items()}


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_section(config_root: Path, component: str) -> dict[str, Any]:
    if component not in SECTION_FILES:
        raise ConfigError(f"unknown component '{component}'")
    path = config_root / f"{component}.yaml"
    if not path.is_file():
        path = config_root / f"{component}.yml"
    return _read_yaml(path)


def load_experiment(repo_root: Path, name: str) -> ProjectConfig:
    config_root = Path(repo_root) / "configs"
    path = config_root / "experiment" / f"{name}.yaml"
    current = _read_yaml(path)
    depth = 0
    while "inherits" in current:
        parent_name = str(current.pop("inherits"))
        depth += 1
        if depth > 8:
            raise ConfigError(f"{name}: experiment inheritance chain is too deep")
        parent_raw = _read_yaml(config_root / "experiment" / f"{parent_name}.yaml")
        current = _deep_merge(parent_raw, current)
    components = tuple(str(item) for item in _require(current, "components", name))
    merged: dict[str, Any] = {}
    for component in components:
        section = _load_section(config_root, component)
        merged = _deep_merge(merged, section)
    overrides = current.get("overrides", {}) or {}
    merged = _deep_merge(merged, overrides)
    spec = ExperimentSpec(
        name=str(_require(current, "name", name)),
        components=components,
        switches=current.get("switches", {}) or {},
        report=current.get("report", {}) or {},
        sampling=current.get("sampling", {}) or {},
        settings=tuple(str(item) for item in current.get("settings", []) or []),
        grid=current.get("grid", {}) or {},
        axes=tuple(current.get("axes", []) or ()),
        corpora=tuple(str(item) for item in current.get("corpora", []) or []),
        arms=tuple(str(item) for item in current.get("arms", []) or ()),
        readers=current.get("readers", {}) or {},
        overrides=overrides,
        path=str(Path("configs") / "experiment" / f"{name}.yaml"),
    )
    return build_project_config(spec, merged)


def build_project_config(spec: ExperimentSpec, merged: Mapping[str, Any]) -> ProjectConfig:
    cohort = _require(merged, "cohort", "config")
    release_block = _require(merged, "release_cohort", "config")
    envelope = _require(merged, "envelope", "config")
    descriptors = _require(merged, "descriptors", "config")
    kinematics = _require(merged, "kinematics", "config")
    uncertainty = _require(merged, "uncertainty", "config")
    ladder = _require(merged, "ladder", "config")
    ladder_weights = _require(merged, "ladder_weights", "config")
    propensity = _require(merged, "propensity", "config")
    library = _require(merged, "library", "config")
    hazard = _require(merged, "hazard", "config")
    outcome = _require(merged, "outcome", "config")
    confidence = _require(merged, "confidence", "config")
    support = _require(merged, "support", "config")
    sweep = _require(merged, "sweep", "config")
    optimizer = _require(merged, "optimizer", "config")
    schedule = _require(merged, "schedule", "config")
    runtime = _require(merged, "runtime", "config")

    clinical = ClinicalCohortConfig(
        name=str(cohort["name"]),
        access=str(cohort["access"]),
        record_level_available=bool(cohort["record_level_available"]),
        regions=tuple(str(item) for item in cohort["regions"]),
        sites=tuple(_site(item) for item in cohort["sites"]),
        accrual=str(cohort["accrual"]),
        unit_of_analysis=str(cohort["unit_of_analysis"]),
        nesting=tuple(str(item) for item in cohort["nesting"]),
        outcome_components=tuple(str(item) for item in cohort["outcome_adjudication"]["components"]),
        reviewers_min=int(cohort["outcome_adjudication"]["reviewers_min"]),
        blinded_to_reconstruction=bool(cohort["outcome_adjudication"]["blinded_to_reconstruction"]),
        secondary_endpoints=tuple(str(item) for item in cohort["secondary_endpoints"]),
        strata={str(key): tuple(str(v) for v in value) for key, value in cohort["strata"].items()},
        subgroups=tuple(str(item) for item in cohort["subgroups"]),
        literature_perforation_prevalence=float(cohort["literature_perforation_prevalence"]),
    )

    calibration = release_block["calibration"]
    release_config = ReleaseCohortConfig(
        name=str(release_block["name"]),
        version=int(release_block["version"]),
        seed=int(release_block["seed"]),
        volume_shape=(
            int(release_block["volume_shape"][0]),
            int(release_block["volume_shape"][1]),
            int(release_block["volume_shape"][2]),
        ),
        voxel_mm=float(release_block["voxel_mm"]),
        attempts=int(release_block["attempts"]),
        site_targets=tuple(_site(item) for item in release_block["site_targets"]),
        prevalence_target=float(release_block["prevalence_target"]),
        calibration=CalibrationSpec(
            parameter=str(calibration["parameter"]),
            method=str(calibration["method"]),
            tolerance=float(calibration["tolerance"]),
            max_iterations=int(calibration["max_iterations"]),
            latent_noise_scale=float(calibration["latent_noise_scale"]),
        ),
        coefficients={str(k): float(v) for k, v in release_block["closed_form"]["coefficients"].items()},
        rung_base_q=float(release_block["closed_form"]["rung_base_q"]),
        rung_action_slope=float(release_block["closed_form"]["rung_action_slope"]),
        lesion_overlap_range=(
            float(release_block["closed_form"]["lesion_overlap_range"][0]),
            float(release_block["closed_form"]["lesion_overlap_range"][1]),
        ),
        nominal_lumen_radius_mm=float(release_block["closed_form"]["nominal_lumen_radius_mm"]),
        descriptor_mix={
            str(axis): {str(level): float(share) for level, share in levels.items()}
            for axis, levels in release_block["descriptor_mix"].items()
        },
        site_shift={
            str(site): {str(key): float(value) for key, value in shifts.items()}
            for site, shifts in release_block["site_shift"].items()
        },
        behaviour={str(k): float(v) for k, v in release_block["behaviour"].items()},
    )

    partition_data = merged["partitions"]
    leakage = merged["leakage_guards"]
    envelope_config = EnvelopeConfig(
        encoder=EncoderConfig(
            name=str(envelope["encoder"]["name"]),
            architecture=str(envelope["encoder"]["architecture"]),
            in_channels=int(envelope["encoder"]["in_channels"]),
            stem_channels=int(envelope["encoder"]["stem_channels"]),
            stage_channels=tuple(int(item) for item in envelope["encoder"]["stage_channels"]),
            stage_blocks=tuple(int(item) for item in envelope["encoder"]["stage_blocks"]),
            bottleneck_channels=int(envelope["encoder"]["bottleneck_channels"]),
            norm=str(envelope["encoder"]["norm"]),
            norm_groups=int(envelope["encoder"]["norm_groups"]),
            activation=str(envelope["encoder"]["activation"]),
            frozen=bool(envelope["encoder"]["frozen"]),
            output_heads=tuple(str(item) for item in envelope["encoder"]["output_heads"]),
        ),
        surface_method=str(envelope["surface"]["method"]),
        surface_level=float(envelope["surface"]["level"]),
        spacing_mm=float(envelope["surface"]["spacing_mm"]),
        distance_clip_mm=float(envelope["boundary"]["distance_clip_mm"]),
        signed_distance=bool(envelope["boundary"]["signed_distance"]),
        centreline_resample_step_mm=float(envelope["centreline"]["resample_step_mm"]),
    )
    descriptor_config = DescriptorConfig(
        calcification_grades=tuple(int(item) for item in descriptors["calcification"]["grades"]),
        calcification_arc_bins_deg=tuple(
            int(item) for item in descriptors["calcification"]["arc_bins_deg"]
        ),
        calcification_severity_weights={
            int(key): float(value)
            for key, value in descriptors["calcification"]["severity_weights"].items()
        },
        cap_classes=tuple(str(item) for item in descriptors["cap_morphology"]["classes"]),
        lesion_length_bins_mm=tuple(int(item) for item in descriptors["lesion_length"]["bins_mm"]),
        territory_classes=tuple(str(item) for item in descriptors["vessel_territory"]["classes"]),
        stages=tuple(str(item) for item in descriptors["anatomical_stage"]["stages"]),
    )
    kinematics_config = KinematicsConfig(
        events=tuple(str(item) for item in kinematics["events"]),
        timestep_s=float(kinematics["timestep_s"]),
        advance_step_mm=float(kinematics["advance_step_mm"]),
        rotation_deg_per_event=float(kinematics["rotation_deg_per_event"]),
        max_steps=int(kinematics["max_steps"]),
        curvature_regularisation=float(kinematics["curvature_regularisation"]),
        tip_mean_absolute_error_mm=float(kinematics["tip_tracking"]["mean_absolute_error_mm"]),
    )
    registration_config = RegistrationConfig(
        method=str(kinematics["registration"]["method"]),
        transform=str(kinematics["registration"]["transform"]),
        iterations=int(kinematics["registration"]["iterations"]),
        tolerance_mm=float(kinematics["registration"]["tolerance_mm"]),
    )
    uncertainty_config = UncertaintyConfig(
        tip_tolerance_mm=float(uncertainty["tip_tolerance_mm"]),
        segmentation_tolerance_mm=float(uncertainty["segmentation_tolerance_mm"]),
        combination=str(uncertainty["combination"]),
        confidence_multiplier=float(uncertainty["confidence_multiplier"]),
        dwell_weighting=str(uncertainty["surface_integral"]["dwell_weighting"]),
    )
    ladder_config = LadderConfig(
        depth=int(ladder["depth"]),
        additional_depths=tuple(int(item) for item in ladder["additional_depths"]),
        tau_mm=tuple(float(item) for item in ladder["tau_mm"]),
        tau_inj_mm=float(ladder["tau_inj_mm"]),
        observed_injury_defines_top_rung=bool(ladder["observed_injury_defines_top_rung"]),
        conditional_frequency_band=(
            float(ladder["conditional_frequency_band"][0]),
            float(ladder["conditional_frequency_band"][1]),
        ),
        resolution_rule=str(ladder["resolution"]["rule"]),
        resolution_band_mm=float(ladder["resolution"]["band_mm"]),
    )
    weights_config = LadderWeightsConfig(
        scheme=str(ladder_weights["scheme"]),
        value=float(ladder_weights["value"]),
        normalise=bool(ladder_weights["normalise"]),
        require_non_negative=bool(ladder_weights["require_non_negative"]),
    )
    propensity_config = PropensityConfig(
        branch_features=tuple(str(item) for item in propensity["branch_model"]["features"]),
        action_features=tuple(str(item) for item in propensity["action_model"]["features"]),
        l2=float(propensity["branch_model"]["l2"]),
        max_iter=int(propensity["branch_model"]["max_iter"]),
        clipping_thresholds=tuple(float(item) for item in propensity["clipping"]["thresholds"]),
        clipping_labels=tuple(str(item) for item in propensity["clipping"]["labels"]),
        default_clipping=float(propensity["clipping"]["default"]),
        fidelity_metrics=tuple(str(item) for item in propensity["fidelity"]["metrics"]),
        fidelity_axes=tuple(str(item) for item in propensity["fidelity"]["report_axes"]),
    )
    library_config = EstimatorLibraryConfig(
        candidates=tuple(_candidate(item) for item in library["candidates"]),
        comparator=_candidate(library["comparator"]),
        selection_criterion=str(library["selection"]["criterion"]),
        reported_diagnostics=tuple(str(item) for item in library["selection"]["reported_diagnostics"]),
    )
    hazard_config = HazardConfig(
        architecture=str(hazard["architecture"]),
        d_model=int(hazard["d_model"]),
        d_state=int(hazard["d_state"]),
        expand=int(hazard["expand"]),
        n_layers=int(hazard["n_layers"]),
        d_conv=int(hazard["d_conv"]),
        dt_rank=int(hazard["dt_rank"]),
        dropout=float(hazard["dropout"]),
        head=str(hazard["head"]),
        sequence=str(hazard["sequence"]),
        conditioning=str(hazard["conditioning"]),
        hazard_weight=float(hazard.get("hazard_weight", 1.0)),
        direct_weight=float(hazard.get("direct_weight", 1.0)),
    )
    outcome_config = OutcomeConfig(
        form=str(outcome["form"]),
        rung_features=tuple(str(item) for item in outcome["rung_features"]),
        l2=float(outcome["l2"]),
        max_iter=int(outcome["max_iter"]),
        clip_observations=bool(outcome["clip_observations"]),
    )
    confidence_config = ConfidenceConfig(
        delta=float(confidence["delta"]),
        bounded_range_b=float(confidence["bounded_range_B"]),
        bound_type=str(confidence["bound_type"]),
        fallback_bound_type=str(confidence["fallback_bound_type"]),
        report_both=bool(confidence["report_both"]),
        ess_definition=str(confidence["effective_sample_size"]["definition"]),
        overlap_coefficient=str(confidence["effective_sample_size"]["overlap_coefficient"]),
        clip_denominator=float(confidence["effective_sample_size"]["clip_denominator"]),
        union_bound_per_rung=bool(confidence["union_bound"]["per_rung"]),
    )
    support_config = SupportConfig(
        floor_c=float(support["floor_c"]),
        axis=tuple(str(item) for item in support["axis"]),
        event_wise=bool(support["event_wise"]),
        unconstrained_fallback=bool(support["unconstrained_fallback"]),
        report_abstention=bool(support["abstain_outcome"]["reported"]),
    )
    sweep_config = SweepConfig(
        risk_margin_m=float(sweep["risk_margin_m"]),
        pessimism_values=tuple(float(item) for item in sweep["pessimism"]["values"]),
        pessimism_labels=tuple(str(item) for item in sweep["pessimism"]["labels"]),
        pessimism_headline=str(sweep["pessimism"]["headline"]),
        clipping_values=tuple(float(item) for item in sweep["clipping"]["values"]),
        clipping_labels=tuple(str(item) for item in sweep["clipping"]["labels"]),
        clipping_headline=str(sweep["clipping"]["headline"]),
        net_benefit_thresholds=tuple(float(item) for item in sweep["net_benefit_thresholds"]),
        primary_utility=str(sweep["primary_utility"]),
        parity_margin_pp=float(sweep["parity_margin_pp"]),
    )
    optimizer_config = OptimizerConfig(
        name=str(optimizer["name"]),
        lr=float(optimizer["lr"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        weight_decay=float(optimizer["weight_decay"]),
        amsgrad=bool(optimizer["amsgrad"]),
        grad_clip=float(optimizer["gradient_clipping"]["max_norm"]),
        ema_enabled=bool(optimizer["ema"]["enabled"]),
        ema_decay=float(optimizer["ema"]["decay"]),
    )
    schedule_config = ScheduleConfig(
        name=str(schedule["name"]),
        warmup_fraction=float(schedule["warmup_fraction"]),
        min_lr_fraction=float(schedule["min_lr_fraction"]),
        epochs=int(schedule["epochs"]),
        batch_size=int(schedule["batch_size"]),
        grad_accum=int(schedule["grad_accum"]),
        seeds=tuple(int(item) for item in schedule["seeds"]),
        eval_every_epochs=int(schedule["eval_every_epochs"]),
    )
    runtime_config = RuntimeConfig(
        world_size=int(runtime["world_size"]),
        device=str(runtime["device"]),
        precision=str(runtime["precision"]),
        amp=bool(runtime["amp"]),
        tf32=bool(runtime["tf32"]),
        num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        deterministic=bool(runtime["deterministic"]),
        checkpoint_dir=str(runtime["checkpoint"]["dir"]),
        keep_last=int(runtime["checkpoint"]["keep_last"]),
        volume_shape=(
            int(runtime["volume_shape"][0]),
            int(runtime["volume_shape"][1]),
            int(runtime["volume_shape"][2]),
        ),
        compute_target_reported=str(runtime["compute_target"]["reported"]),
        accelerator_model=str(runtime["compute_target"]["accelerator_model"]),
        vram_gb=str(runtime["compute_target"]["vram_gb"]),
        wall_clock=str(runtime["compute_target"]["wall_clock"]),
        storage_gb=str(runtime["compute_target"]["storage_gb"]),
    )
    return ProjectConfig(
        experiment=spec,
        clinical=clinical,
        release_cohort=release_config,
        partitions=tuple(
            PartitionSpec(
                name=str(item["name"]),
                sites=tuple(str(s) for s in item["sites"]),
                role=str(item["role"]),
                configuration=str(item.get("configuration", "")),
                regions=tuple(str(r) for r in item.get("regions", [])),
                disjoint_from=str(item.get("disjoint_from", "")),
            )
            for item in partition_data
        ),
        leakage=LeakageGuardConfig(
            patient_disjoint=bool(leakage["patient_disjoint"]),
            operator_disjoint_across_partitions=bool(leakage["operator_disjoint_across_partitions"]),
            vendor_disjoint_across_partitions=bool(leakage["vendor_disjoint_across_partitions"]),
            calibration_share_disjoint=bool(leakage["calibration_share_disjoint"]),
            fail_on_overlap=bool(leakage["fail_on_overlap"]),
        ),
        corpora=tuple(
            PublicCorpusSpec(
                key=str(item["key"]),
                label=str(item["label"]),
                logging_regime=str(item["logging_regime"]),
                licence=str(item["licence"]),
                source=str(item["source"]),
                role=str(item["role"]),
                rows_millions=int(item["rows_millions"]) if "rows_millions" in item else None,
            )
            for item in merged["corpora"]
        ),
        envelope=envelope_config,
        descriptors=descriptor_config,
        kinematics=kinematics_config,
        registration=registration_config,
        uncertainty=uncertainty_config,
        ladder=ladder_config,
        ladder_weights=weights_config,
        propensity=propensity_config,
        library=library_config,
        hazard=hazard_config,
        outcome=outcome_config,
        confidence=confidence_config,
        support=support_config,
        sweep=sweep_config,
        optimizer=optimizer_config,
        schedule=schedule_config,
        runtime=runtime_config,
        raw=dict(merged),
    )


def _site(item: Mapping[str, Any]) -> SiteSpec:
    _reject_unknown(item, ("site", "region", "attempts", "injury_events"), "site entry")
    return SiteSpec(
        site=str(item["site"]),
        region=str(item["region"]),
        attempts=int(item["attempts"]),
        injury_events=int(item["injury_events"]),
    )


def _candidate(item: Mapping[str, Any]) -> CandidateEstimator:
    return CandidateEstimator(key=str(item["key"]), label=str(item["label"]), family=str(item["family"]))


def experiment_names(repo_root: Path) -> list[str]:
    directory = Path(repo_root) / "configs" / "experiment"
    return sorted(path.stem for path in directory.glob("*.yaml"))
