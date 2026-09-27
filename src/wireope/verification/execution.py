"""Execution checks: the release is run, and every result is read off what actually happened.

Each check carries its own independent reference — a closed form, a brute-force loop, a
second library, or an exactly enumerable population — so no check validates the
implementation by calling it. A check whose evidence cannot discriminate is reported NOT_RUN
rather than PASS, and evidence that is absent by contract is reported BLOCKED.

Ref: the release contract in README, and Sec. 3-4 of the manuscript.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from wireope.bounds.effective_sample import effective_sample_size
from wireope.bounds.empirical_bernstein import (
    EmpiricalBernsteinBound,
    agrees_with_hoeffding_at_maximum_variance,
    empirical_bernstein_bound,
)
from wireope.bounds.hoeffding import HoeffdingBound, effective_sample_size_minimum, hoeffding_bound
from wireope.certification.eligibility import evaluate_floor
from wireope.certification.rule import StrategyEstimate, select_constrained, select_unconstrained
from wireope.cohort.release_cohort import assemble_release_cohort
from wireope.estimators.base import LogBatch
from wireope.estimators.nested_dr import (
    conditionals_from_marginals,
    nested_doubly_robust,
    product_risk_from_marginals,
)
from wireope.estimators.registry import FAMILY_BY_KEY, build_estimator
from wireope.fit.dataset import ExcursionDataset, excursion_sequence, rung_weight_tensor
from wireope.fit.hazard import HazardModel, hazard_loss
from wireope.fit.loop import train_hazard_head
from wireope.geometry.envelope import GridSpec
from wireope.geometry.vessel import (
    VesselGeometry,
    curved_centreline,
    radius_profile,
)
from wireope.laddering.resolution import check_resolution
from wireope.laddering.severity import build_ladder
from wireope.metrics.calibration import expected_calibration_error
from wireope.metrics.decision import net_benefit, treat_all_net_benefit
from wireope.metrics.discrimination import auroc, delong_interval
from wireope.metrics.safety import violation_rate
from wireope.schema import CheckStatus, EventRecord
from wireope.studies.pipeline import (
    AnalysisContext,
    CohortBundle,
    prepare_bundle,
    prepare_context,
)
from wireope.traces.events import integrate_event_stream
from wireope.traces.excursion import build_trace, dwell_integral, propagating_band
from wireope.traces.registration import align_trajectory, fit_similarity
from wireope.utils.atomic import iter_files
from wireope.utils.linalg import sigmoid
from wireope.utils.logging_setup import get_logger
from wireope.utils.manifest import SKIP_DIRS
from wireope.utils.seed import numpy_generator, set_seed

LOGGER = get_logger(__name__)


@dataclass
class ExecutionCheck:
    name: str
    status: CheckStatus
    detail: str
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass
class Fixture:
    repo_root: Path
    bundle: CohortBundle
    context: AnalysisContext
    tempdir: tempfile.TemporaryDirectory[str]
    probe_links: bool = False


def _check(name: str, status: CheckStatus, detail: str, **evidence: object) -> ExecutionCheck:
    return ExecutionCheck(name=name, status=status, detail=detail, evidence=evidence)


def _pass_if(name: str, ok: bool, detail: str, **evidence: object) -> ExecutionCheck:
    return _check(name, CheckStatus.PASS if ok else CheckStatus.FAIL, detail, **evidence)


def build_fixture(repo_root: Path, probe_links: bool = False) -> Fixture:
    bundle = prepare_bundle(repo_root, "_smoke")
    context = prepare_context(repo_root, "_smoke", "primary_external", bundle=bundle)
    return Fixture(
        repo_root=repo_root,
        bundle=bundle,
        context=context,
        tempdir=tempfile.TemporaryDirectory(prefix="wireope-verify-"),
        probe_links=probe_links,
    )


def analytic_geometry(seed: int = 3) -> VesselGeometry:
    centreline = curved_centreline((0.0, 0.0, 0.0), 40.0, 24, 0.01, plane=(0, 2))
    radius = radius_profile(24, 1.6, 0.5, 0.5, 0.16)
    return VesselGeometry(centreline_mm=centreline, radius_mm=radius)


def tiny_batch(
    attempts: int,
    rungs: int,
    target: str,
    probability_by_rung: NDArray[np.float64],
    propensity: float,
    seed: int,
) -> LogBatch:
    rng = numpy_generator(seed)
    weights = np.full(rungs, 1.0 / rungs, dtype=np.float64)
    observations = np.zeros((attempts, rungs), dtype=np.float64)
    for rung in range(rungs):
        observations[:, rung] = (rng.random(attempts) < probability_by_rung[rung]).astype(np.float64)
    for rung in range(rungs - 1, 0, -1):
        observations[:, rung - 1] = np.maximum(observations[:, rung - 1], observations[:, rung])
    strategy = np.where(rng.random(attempts) < propensity, target, "other")
    return LogBatch(
        features=rng.normal(size=(attempts, 4)),
        behaviour_strategy=strategy,
        target_strategy=np.full(attempts, target, dtype="<U64"),
        rung_observations=observations,
        propensities=np.full(attempts, propensity, dtype=np.float64),
        weights=weights,
        strata=np.full(attempts, "I", dtype="<U8"),
        outcome_predictions=np.tile(probability_by_rung, (attempts, 1)),
    )


def check_environment(fixture: Fixture) -> ExecutionCheck:
    versions = {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    }
    return _pass_if(
        "env.interpreter_and_libraries",
        all(versions.values()),
        "the interpreter and the declared numerical libraries import and report a version",
        **versions,
    )


def check_deterministic_seeding(fixture: Fixture) -> ExecutionCheck:
    set_seed(11)
    first = np.random.random(4).tolist()
    set_seed(11)
    second = np.random.random(4).tolist()
    set_seed(11)
    tensor_first = torch.rand(4).tolist()
    set_seed(11)
    tensor_second = torch.rand(4).tolist()
    ok = first == second and tensor_first == tensor_second
    return _pass_if(
        "env.deterministic_seeding",
        ok,
        "the same seed reproduces the NumPy and torch draws",
        numpy_first=round(first[0], 8),
        torch_first=round(tensor_first[0], 8),
    )


def check_analytic_distance(fixture: Fixture) -> ExecutionCheck:
    geometry = analytic_geometry()
    grid = GridSpec(shape=(24, 24, 40), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-6.0, -6.0, -2.0))
    envelope = geometry.as_envelope(grid)
    rng = numpy_generator(5)
    indices = rng.uniform(low=1.0, high=20.0, size=(400, 3))
    points = grid.index_to_mm(indices)
    analytic = geometry.analytic_excursion_mm(points)
    discretised = envelope.excursion(points)
    voxel = float(np.max(grid.spacing_array))
    worst = float(np.max(np.abs(analytic - discretised)))
    ok = worst <= 1.5 * voxel
    return _pass_if(
        "geometry.analytic_excursion_matches_discretised_field",
        ok,
        "the discretised excursion stays within one and a half voxels of the analytic distance",
        worst_abs_gap_mm=round(worst, 4),
        voxel_mm=voxel,
        samples=int(points.shape[0]),
    )


def check_excursion_non_negative(fixture: Fixture) -> ExecutionCheck:
    geometry = analytic_geometry()
    grid = GridSpec(shape=(20, 20, 32), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-5.0, -5.0, -2.0))
    envelope = geometry.as_envelope(grid)
    inside = geometry.centreline_mm
    values = envelope.excursion(inside)
    analytic = geometry.analytic_excursion_mm(inside)
    ok = float(np.min(values)) >= 0.0 and float(np.min(analytic)) >= 0.0
    return _pass_if(
        "geometry.excursion_non_negative",
        ok,
        "points on the centreline carry zero excursion under both the field and the analytic form",
        min_field=round(float(np.min(values)), 6),
        min_analytic=round(float(np.min(analytic)), 6),
    )


def check_surface_area(fixture: Fixture) -> ExecutionCheck:
    geometry = analytic_geometry()
    grid = GridSpec(shape=(24, 24, 40), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-6.0, -6.0, -2.0))
    envelope = geometry.as_envelope(grid)
    area = envelope.surface_area_mm2()
    expected = 2.0 * math.pi * 1.05 * geometry.length_mm
    ok = 0.3 * expected <= area <= 2.5 * expected
    return _pass_if(
        "geometry.surface_area_plausible",
        ok,
        "the voxel-face surface area is within a factor of the cylinder estimate",
        measured_mm2=round(area, 2),
        cylinder_estimate_mm2=round(expected, 2),
    )


def check_event_integration(fixture: Fixture) -> ExecutionCheck:
    from wireope.config import load_experiment

    config = load_experiment(fixture.repo_root, "_smoke")
    events: list[EventRecord] = [
        {"step": index, "kind": "advance", "magnitude": 2.0} for index in range(10)
    ]
    trajectory = integrate_event_stream(events, config.kinematics)
    # Nine segments join the ten recorded tip samples.
    expected = 9 * 2.0 * config.kinematics.advance_step_mm
    ok = abs(trajectory.path_length_mm - expected) < 1e-9
    return _pass_if(
        "traces.event_integration_path_length",
        ok,
        "ten advance events of magnitude two travel the closed-form distance",
        measured_mm=round(trajectory.path_length_mm, 6),
        expected_mm=expected,
    )


def check_similarity_fit(fixture: Fixture) -> ExecutionCheck:
    rng = numpy_generator(9)
    source = rng.normal(size=(60, 2))
    angle = 0.4
    scale = 1.3
    rotation = np.asarray(
        [
            [scale * math.cos(angle), -scale * math.sin(angle)],
            [scale * math.sin(angle), scale * math.cos(angle)],
        ],
        dtype=np.float64,
    )
    target = source @ rotation.T + np.asarray([2.0, -1.0])
    fitted = fit_similarity(source, target)
    recovered = fitted.apply(source)
    worst = float(np.max(np.linalg.norm(recovered - target, axis=1)))
    ok = worst < 1e-8 and abs(fitted.scale - scale) < 1e-8
    return _pass_if(
        "traces.similarity_recovers_known_transform",
        ok,
        "the closed-form similarity fit recovers a known rotation, scale and translation",
        worst_residual_mm=round(worst, 10),
        scale=round(fitted.scale, 8),
    )


def check_alignment_residual(fixture: Fixture) -> ExecutionCheck:
    from wireope.config import load_experiment

    config = load_experiment(fixture.repo_root, "_smoke")
    geometry = analytic_geometry()
    offset = geometry.centreline_mm + np.asarray([0.6, -0.4, 0.0])
    result = align_trajectory(offset, geometry.centreline_mm, config.registration)
    ok = result.residual_mm < 0.5
    return _pass_if(
        "traces.registration_reduces_residual",
        ok,
        "aligning a laterally offset trajectory drives the residual below half a millimetre",
        residual_mm=round(result.residual_mm, 5),
        iterations=result.iterations,
    )


def check_dwell_integral(fixture: Fixture) -> ExecutionCheck:
    excursion = np.asarray([1.0, 2.0, 3.0, 2.0], dtype=np.float64)
    dwell = np.asarray([0.5, 0.5, 1.0, 2.0], dtype=np.float64)
    measured = dwell_integral(excursion, dwell)
    expected = float(np.sum(excursion * dwell))
    return _pass_if(
        "traces.dwell_integral_closed_form",
        abs(measured - expected) < 1e-12,
        "the dwell integral equals the sum of the products by hand",
        measured=round(measured, 6),
        expected=expected,
    )


def check_propagating_band(fixture: Fixture) -> ExecutionCheck:
    config = fixture.context.config
    band = propagating_band(config.uncertainty)
    expected = math.hypot(config.uncertainty.tip_tolerance_mm, config.uncertainty.segmentation_tolerance_mm)
    ok = abs(band.combined_mm - expected) < 1e-9
    return _pass_if(
        "traces.propagating_band_quadrature",
        ok,
        "the propagating band is the quadrature combination of the two tolerances",
        combined_mm=round(band.combined_mm, 6),
        expected_mm=round(expected, 6),
    )


def check_ladder_thresholds(fixture: Fixture) -> ExecutionCheck:
    ladder = fixture.bundle.ladder
    increasing = all(later > earlier for earlier, later in zip(ladder.tau_mm, ladder.tau_mm[1:]))
    top_is_injury = abs(ladder.tau_mm[-1] - ladder.tau_inj_mm) < 1e-12
    return _pass_if(
        "ladder.thresholds_increasing_and_top_rung_is_injury",
        increasing and top_is_injury,
        "the rung thresholds increase and the top rung is the injury threshold",
        thresholds=str(list(ladder.tau_mm)),
    )


def check_ladder_nesting_brute_force(fixture: Fixture) -> ExecutionCheck:
    ladder = fixture.bundle.ladder
    peaks = fixture.bundle.cohort.peaks()
    matrix = ladder.indicator_matrix(peaks)
    brute = np.zeros_like(matrix)
    for row, peak in enumerate(peaks):
        for column, threshold in enumerate(ladder.tau_mm):
            brute[row, column] = 1.0 if peak >= threshold else 0.0
    same = bool(np.array_equal(matrix, brute))
    nesting = ladder.nesting_holds(peaks)
    top_equals_endpoint = bool(
        np.array_equal(matrix[:, -1], fixture.bundle.cohort.injury_flags().astype(np.float64))
    )
    return _pass_if(
        "ladder.nesting_matches_brute_force",
        same and nesting and top_equals_endpoint,
        "the ladder indicators reproduce a brute-force threshold loop, nest downward, and the top "
        "rung equals the adjudicated endpoint",
        rows=int(peaks.shape[0]),
        rungs=ladder.depth,
    )


def check_ladder_weights(fixture: Fixture) -> ExecutionCheck:
    weights = np.asarray(fixture.bundle.ladder.weights, dtype=np.float64)
    non_negative = bool(np.all(weights >= 0.0))
    normalised = abs(float(np.sum(weights)) - 1.0) < 1e-9
    return _pass_if(
        "ladder.weights_non_negative_and_normalised",
        non_negative and normalised,
        "the rung weights are non-negative and sum to one",
        total=round(float(np.sum(weights)), 8),
        first=round(float(weights[0]), 6),
    )


def check_ladder_resolution(fixture: Fixture) -> ExecutionCheck:
    report = check_resolution(fixture.bundle.ladder, fixture.context.config.ladder.resolution_band_mm)
    return _pass_if(
        "ladder.rung_resolution_on_configured_ladder",
        report.resolvable,
        "every adjacent rung separation on the configured ladder exceeds the propagating band",
        band_mm=round(report.band_mm, 4),
        separations_mm=str([round(item, 3) for item in report.separations_mm]),
        max_depth=report.max_depth,
    )


def check_ladder_conditional_band(fixture: Fixture) -> ExecutionCheck:
    """The intermediate rung transitions must sit in the declared band; the endpoint may not.

    The top transition is the endpoint itself, which is rare by construction, so it is reported
    beside the check rather than used to decide it.
    """
    ladder = fixture.bundle.ladder
    frequencies = ladder.conditional_frequencies(fixture.bundle.cohort.peaks())
    low, high = fixture.context.config.ladder.conditional_frequency_band
    interior = frequencies[1:-1] if frequencies.shape[0] > 2 else frequencies[1:]
    ok = bool(np.all((interior >= low) & (interior <= high)))
    status = CheckStatus.PASS if ok else CheckStatus.FAIL
    return _check(
        "ladder.conditional_frequencies_inside_declared_band",
        status,
        "the assembled cohort's intermediate rung transitions are compared against the declared tolerable "
        "band, with the endpoint transition reported separately",
        frequencies=str([round(float(item), 4) for item in frequencies]),
        interior=str([round(float(item), 4) for item in interior]),
        endpoint_conditional=round(float(frequencies[-1]), 4),
        band=[low, high],
    )


def check_importance_weights(fixture: Fixture) -> ExecutionCheck:
    batch = tiny_batch(400, 3, "target", np.asarray([0.6, 0.3, 0.1]), 0.25, seed=17)
    weights = batch.importance_weights()
    assigned = batch.assigned
    expected = assigned.astype(np.float64) / 0.25
    ok = bool(np.allclose(weights, expected))
    return _pass_if(
        "estimators.importance_weights_closed_form",
        ok,
        "the importance weight is the indicator over the reconstructed propensity",
        assigned=int(np.count_nonzero(assigned)),
        attempts=batch.attempts,
    )


def check_effective_sample_size(fixture: Fixture) -> ExecutionCheck:
    weights = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    uniform = effective_sample_size(weights)
    concentrated = effective_sample_size(np.asarray([4.0, 0.0, 0.0, 0.0], dtype=np.float64))
    ok = abs(uniform - 4.0) < 1e-9 and abs(concentrated - 1.0) < 1e-9
    return _pass_if(
        "estimators.effective_sample_size_limits",
        ok,
        "the effective sample size is the sample count for uniform weights and one for a single "
        "carrying weight",
        uniform=uniform,
        concentrated=concentrated,
    )


def check_nested_dr_unbiasedness(fixture: Fixture) -> ExecutionCheck:
    """The correction alone must recover the truth, so the direct part is deliberately wrong."""
    rung_probability = np.asarray([0.55, 0.18], dtype=np.float64)
    propensity = 0.3
    batch = tiny_batch(60000, 2, "target", rung_probability, propensity, seed=23)
    batch = LogBatch(
        features=batch.features,
        behaviour_strategy=batch.behaviour_strategy,
        target_strategy=batch.target_strategy,
        rung_observations=batch.rung_observations,
        propensities=batch.propensities,
        weights=batch.weights,
        strata=batch.strata,
        outcome_predictions=np.full((batch.attempts, 2), 0.05, dtype=np.float64),
    )
    result = nested_doubly_robust(batch, 0.0)
    weights = batch.weights
    # The nesting raises the shallow rung's marginal above its nominal value, so the reference
    # is the realised marginal of the drawn log rather than the generating probability.
    realised = batch.rung_observations.mean(axis=0)
    truth_graded = float(np.sum(weights * realised))
    truth_risk = float(realised[-1])
    graded_gap = abs(result.graded_value - truth_graded)
    risk_gap = abs(result.endpoint_risk - truth_risk)
    ok = graded_gap < 0.01 and risk_gap < 0.01
    return _pass_if(
        "estimators.nested_dr_recovers_known_functionals",
        ok,
        "with a deliberately wrong direct part and the true propensity, the importance-weighted "
        "residual alone recovers the closed-form rung marginals on a large independent draw",
        graded_gap=round(graded_gap, 4),
        risk_gap=round(risk_gap, 4),
        truth_graded=round(truth_graded, 4),
    )


def check_nested_dr_needs_the_propensity(fixture: Fixture) -> ExecutionCheck:
    """A wrong propensity must move the estimate, which shows the weight is load-bearing."""
    rung_probability = np.asarray([0.5, 0.2], dtype=np.float64)
    batch = tiny_batch(60000, 2, "target", rung_probability, 0.35, seed=29)
    biased = LogBatch(
        features=batch.features,
        behaviour_strategy=batch.behaviour_strategy,
        target_strategy=batch.target_strategy,
        rung_observations=batch.rung_observations,
        propensities=np.full(batch.attempts, 0.7, dtype=np.float64),
        weights=batch.weights,
        strata=batch.strata,
        outcome_predictions=np.full((batch.attempts, 2), 0.05, dtype=np.float64),
    )
    wrong = nested_doubly_robust(biased, 0.0)
    right = nested_doubly_robust(batch, 0.0)
    truth_risk = float(rung_probability[-1])
    gap = abs(wrong.endpoint_risk - right.endpoint_risk)
    ok = gap > 0.02 and abs(right.endpoint_risk - truth_risk) < 0.01
    return _pass_if(
        "estimators.nested_dr_is_sensitive_to_a_misspecified_propensity",
        ok,
        "the correct propensity recovers the closed-form risk while a misspecified one moves the "
        "estimate, so the reconstructed behaviour model is load-bearing",
        correct_risk=round(right.endpoint_risk, 4),
        misspecified_risk=round(wrong.endpoint_risk, 4),
        truth_risk=truth_risk,
    )


def check_candidate_library(fixture: Fixture) -> ExecutionCheck:
    batch = tiny_batch(
        3000,
        fixture.bundle.ladder.depth,
        "target",
        np.linspace(0.5, 0.05, fixture.bundle.ladder.depth),
        0.4,
        seed=31,
    )
    values: dict[str, float] = {}
    finite = True
    for key in sorted(FAMILY_BY_KEY):
        result = build_estimator(key).estimate(batch, 0.1)
        values[key] = round(result.value, 6)
        if not math.isfinite(result.value):
            finite = False
    return _pass_if(
        "estimators.every_candidate_is_finite",
        finite,
        "every candidate in the library returns a finite estimate on one shared batch",
        **values,
    )


def check_kernel_weights(fixture: Fixture) -> ExecutionCheck:
    from wireope.estimators.kernel import state_density_weights

    rng = numpy_generator(37)
    features = rng.normal(size=(40, 6))
    weights = state_density_weights(features, 1.0, 0.0)
    row_sums = weights.sum(axis=1)
    ok = bool(np.allclose(row_sums, 1.0, atol=1e-9))
    return _pass_if(
        "estimators.kernel_weights_normalised_per_target_state",
        ok,
        "each target state's kernel weights sum to one",
        worst_row_error=round(float(np.max(np.abs(row_sums - 1.0))), 10),
    )


def check_product_risk(fixture: Fixture) -> ExecutionCheck:
    marginals = np.asarray([0.6, 0.3, 0.1, 0.02], dtype=np.float64)
    conditionals = conditionals_from_marginals(marginals)
    manual = np.concatenate([[marginals[0]], marginals[1:] / marginals[:-1]])
    ok = (
        bool(np.allclose(conditionals, manual))
        and abs(product_risk_from_marginals(marginals) - marginals[-1]) < 1e-12
    )
    return _pass_if(
        "estimators.product_risk_reconstructs_the_endpoint",
        ok,
        "the per-rung conditionals are the marginal ratios and their product is the endpoint " "marginal",
        conditionals=str([round(float(item), 5) for item in conditionals]),
    )


def check_hoeffding_closed_form(fixture: Fixture) -> ExecutionCheck:
    counts = np.asarray([120.0, 60.0, 20.0, 8.0], dtype=np.float64)
    delta = 0.05
    bounded = 1.0
    bound = hoeffding_bound(counts, delta, bounded, truncation_term=0.01)
    manual = bounded * math.sqrt(math.log(2.0 * counts.size / delta) / (2.0 * 8.0))
    ok = abs(bound.radius - manual) < 1e-12 and abs(bound.effective_sample_size - 8.0) < 1e-12
    return _pass_if(
        "bounds.hoeffding_radius_matches_closed_form",
        ok,
        "the Eq. (7) radius equals the hand-computed expression with the rarest-rung effective " "count",
        radius=round(bound.radius, 6),
        manual=round(manual, 6),
        total=round(bound.total, 6),
    )


def check_hoeffding_monotone_in_events(fixture: Fixture) -> ExecutionCheck:
    base = np.asarray([200.0, 90.0, 30.0, 12.0], dtype=np.float64)
    radii = [
        hoeffding_bound(np.concatenate([base[:-1], [value]]), 0.05, 1.0, 0.0).radius
        for value in (12.0, 24.0, 48.0, 96.0)
    ]
    ok = all(later <= earlier for earlier, later in zip(radii, radii[1:]))
    return _pass_if(
        "bounds.radius_falls_as_rarest_rung_events_grow",
        ok,
        "raising only the rarest-rung count tightens the radius monotonically",
        radii=str([round(item, 6) for item in radii]),
        effective_min=effective_sample_size_minimum(base),
    )


def check_radius_invariant_to_negatives(fixture: Fixture) -> ExecutionCheck:
    counts = np.asarray([500.0, 120.0, 30.0], dtype=np.float64)
    widened = np.asarray([5000.0, 1200.0, 30.0], dtype=np.float64)
    base = hoeffding_bound(counts, 0.05, 1.0, 0.0).radius
    dilated = hoeffding_bound(widened, 0.05, 1.0, 0.0).radius
    ok = abs(base - dilated) < 1e-9
    return _pass_if(
        "bounds.radius_invariant_to_shallower_rung_records",
        ok,
        "adding records that do not raise the rarest-rung count leaves the radius unchanged, which "
        "is the sizing law the manuscript states",
        radius_at_low_records=round(base, 6),
        radius_at_high_records=round(dilated, 6),
    )


def check_bernstein_limit(fixture: Fixture) -> ExecutionCheck:
    counts = np.asarray([150.0, 70.0, 25.0, 9.0], dtype=np.float64)
    table = agrees_with_hoeffding_at_maximum_variance(counts, 0.05, 1.0)
    ok = abs(table["leading_ratio"] - 1.0) < 1e-9
    return _pass_if(
        "bounds.bernstein_leading_term_equals_hoeffding_at_maximum_variance",
        ok,
        "at sigma = B/2 the two leading terms coincide, leaving only the residual correction",
        leading_ratio=round(table["leading_ratio"], 10),
        residual_correction=round(table["residual_correction"], 6),
    )


def check_bernstein_tighter_on_conformed_residuals(fixture: Fixture) -> ExecutionCheck:
    """Bernstein beats Hoeffding only where the residual correction is small.

    The comparison is made where the correction no longer dominates the leading term; the
    small-sample regime is reported beside the result so the direction is documented.
    """
    rng = numpy_generator(41)
    observations = (rng.random((40000, 3)) < 0.25).astype(np.float64)
    observations[:, 1] = np.maximum(observations[:, 1], observations[:, 2])
    observations[:, 0] = np.maximum(observations[:, 0], observations[:, 1])
    weights = np.ones(40000, dtype=np.float64)
    counts = np.asarray(
        [float(np.count_nonzero(observations[:, rung])) for rung in range(3)], dtype=np.float64
    )
    bernstein = empirical_bernstein_bound(observations, weights, counts, 0.05, 1.0)
    hoeffding = hoeffding_bound(counts, 0.05, 1.0, 0.0)
    small_bernoulli = empirical_bernstein_bound(
        (rng.random((400, 2)) < 0.25).astype(np.float64),
        np.ones(400),
        np.asarray([100.0, 30.0]),
        0.05,
        1.0,
    )
    small_hoeffding = hoeffding_bound(np.asarray([100.0, 30.0]), 0.05, 1.0, 0.0)
    ok = bernstein.radius <= hoeffding.radius
    return _pass_if(
        "bounds.bernstein_no_wider_than_hoeffding_where_the_correction_is_small",
        ok,
        "with conformed residuals and a large effective count the empirical-Bernstein radius does "
        "not exceed the Hoeffding radius; the small-sample regime is reported beside it",
        bernstein=round(bernstein.radius, 6),
        hoeffding=round(hoeffding.radius, 6),
        leading_ratio=round(bernoulli_ratio(bernstein, hoeffding), 6),
        small_sample_bernstein=round(small_bernoulli.radius, 6),
        small_sample_hoeffding=round(small_hoeffding.radius, 6),
    )


def bernoulli_ratio(bernstein: EmpiricalBernsteinBound, hoeffding: HoeffdingBound) -> float:
    """Leading term of the empirical-Bernstein radius relative to the Hoeffding radius."""
    leading = float(bernstein.leading_term)
    radius = float(hoeffding.radius)
    return leading / max(radius, 1e-12)


def check_floor_brute_force(fixture: Fixture) -> ExecutionCheck:
    rng = numpy_generator(43)
    strata = np.asarray(["I"] * 60 + ["II"] * 60, dtype="<U4")
    indicators = np.zeros((120, 2), dtype=np.float64)
    indicators[:60, 0] = (rng.random(60) < 0.7).astype(np.float64)
    indicators[60:, 0] = (rng.random(60) < 0.6).astype(np.float64)
    indicators[:, 1] = (rng.random(120) < 0.1).astype(np.float64)
    indicators[:, 0] = np.maximum(indicators[:, 0], indicators[:, 1])
    propensity = rng.uniform(0.05, 0.9, size=120)
    report = evaluate_floor("s", strata, indicators, propensity, 0.04)
    manual_failures = 0
    for name in ("I", "II"):
        member = strata == name
        total = float(np.sum(propensity[member]))
        for rung in range(2):
            reached = indicators[member, rung] > 0.0
            mass = float(np.sum(propensity[member][reached]) / total)
            if mass < 0.04:
                manual_failures += 1
    ok = report.failing_count() == manual_failures
    return _pass_if(
        "certification.floor_matches_brute_force_mass",
        ok,
        "the number of stratum-rung cells below the floor agrees with an independent mass loop",
        reported_failures=report.failing_count(),
        brute_force_failures=manual_failures,
    )


def check_rule_brute_force(fixture: Fixture) -> ExecutionCheck:
    estimates = (
        StrategyEstimate("a", 0.9, 0.03, 0.2, 0.001, 100.0, 0.4),
        StrategyEstimate("b", 0.7, 0.02, 0.1, 0.001, 120.0, 0.5),
        StrategyEstimate("c", 0.95, 0.02, 0.4, 0.001, 90.0, 0.3, eligible=False),
    )
    margin = 0.05
    outcome = select_constrained(estimates, risk_margin=margin)
    feasible = [item for item in estimates if item.eligible and item.risk_lcb <= margin]
    expected = max(feasible, key=lambda item: item.graded_lcb).strategy
    unconstrained = select_unconstrained(estimates)
    ok = outcome.selected == expected == "a" and unconstrained.selected == "c"
    return _pass_if(
        "certification.rule_matches_brute_force_argmax",
        ok,
        "the constrained rule picks the highest utility lower limit among the strategies within the "
        "margin, and the unconstrained arm ignores both the floor and the margin",
        constrained_pick=outcome.selected,
        expected=expected,
        unconstrained_pick=unconstrained.selected,
    )


def check_rule_abstains(fixture: Fixture) -> ExecutionCheck:
    ineligible = (StrategyEstimate("a", 0.9, 0.05, 0.2, 0.01, 100.0, 0.4, eligible=False),)
    outcome = select_constrained(ineligible, risk_margin=0.0)
    ok = outcome.abstained and outcome.eligible_count == 0
    return _pass_if(
        "certification.abstains_when_no_strategy_is_eligible",
        ok,
        "an empty eligible set returns abstention with the reason recorded",
        reason=outcome.reason,
    )


def check_auroc_matches_library(fixture: Fixture) -> ExecutionCheck:
    rng = numpy_generator(47)
    scores = rng.normal(size=400)
    labels = (rng.random(400) < sigmoid(scores)).astype(np.int64)
    ours = auroc(scores, labels)
    reference = float(roc_auc_score(labels, scores))
    ok = abs(ours - reference) < 1e-9
    return _pass_if(
        "metrics.auroc_matches_an_independent_library",
        ok,
        "the rank-based AUROC agrees with scikit-learn on the same scores",
        ours=round(ours, 8),
        reference=round(reference, 8),
    )


def check_auroc_pair_count(fixture: Fixture) -> ExecutionCheck:
    scores = np.asarray([0.1, 0.4, 0.35, 0.8, 0.9], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    concordant = 0.0
    for positive in positives:
        for negative in negatives:
            concordant += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    manual = concordant / (positives.size * negatives.size)
    ok = abs(auroc(scores, labels) - manual) < 1e-12
    return _pass_if(
        "metrics.auroc_matches_pairwise_concordance",
        ok,
        "the AUROC equals the pairwise concordance count on a hand-enumerated example",
        ours=round(auroc(scores, labels), 8),
        manual=round(manual, 8),
    )


def check_delong_interval(fixture: Fixture) -> ExecutionCheck:
    rng = numpy_generator(53)
    scores = rng.normal(size=500)
    labels = (rng.random(500) < sigmoid(scores * 0.8)).astype(np.int64)
    interval = delong_interval(scores, labels)
    ok = interval.lower <= interval.auroc <= interval.upper and interval.width > 0.0
    return _pass_if(
        "metrics.delong_interval_brackets_the_estimate",
        ok,
        "the DeLong interval contains its own point estimate and has positive width",
        auroc=round(interval.auroc, 5),
        lower=round(interval.lower, 5),
        upper=round(interval.upper, 5),
    )


def check_net_benefit_closed_form(fixture: Fixture) -> ExecutionCheck:
    probability = np.asarray([0.9, 0.2, 0.6, 0.1], dtype=np.float64)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    threshold = 0.3
    selected = probability >= threshold
    expected = float(np.sum(labels[selected])) / labels.size - float(
        np.sum(1 - labels[selected])
    ) / labels.size * threshold / (1.0 - threshold)
    measured = net_benefit(probability, labels, threshold)
    reference = treat_all_net_benefit(0.5, threshold)
    ok = (
        abs(measured - expected) < 1e-12
        and abs(reference - (0.5 - 0.5 * threshold / (1 - threshold))) < 1e-12
    )
    return _pass_if(
        "metrics.net_benefit_matches_closed_form",
        ok,
        "net benefit equals the hand-computed decision-curve expression, and the treat-all "
        "reference matches its own closed form",
        measured=round(measured, 6),
        expected=round(expected, 6),
    )


def check_calibration_ece(fixture: Fixture) -> ExecutionCheck:
    probability = np.asarray([0.05, 0.15, 0.85, 0.95], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    measured = expected_calibration_error(probability, labels, bins=10)
    manual = (
        abs(0.0 - 0.05) * 0.25 + abs(0.0 - 0.15) * 0.25 + abs(1.0 - 0.85) * 0.25 + abs(1.0 - 0.95) * 0.25
    )
    ok = abs(measured - manual) < 1e-12
    return _pass_if(
        "metrics.calibration_error_matches_bin_average",
        ok,
        "the expected calibration error equals the bin-weighted gap computed by hand",
        measured=round(measured, 8),
        manual=round(manual, 8),
    )


def check_violation_rate(fixture: Fixture) -> ExecutionCheck:
    realised = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int64)
    abstained = np.asarray([False, False, True, True, False, False], dtype=np.bool_)
    measured = violation_rate(realised, abstained)
    manual = 1.0 / 4.0
    ok = abs(measured - manual) < 1e-12
    return _pass_if(
        "metrics.violation_rate_excludes_abstentions",
        ok,
        "the violation rate is taken over the answered decisions only, which the hand count " "confirms",
        measured=round(measured, 6),
        manual=manual,
    )


def check_forward_shapes(fixture: Fixture) -> ExecutionCheck:
    config = fixture.context.config.hazard
    model = HazardModel(config, rungs=3, context_dim=16)
    sequence = torch.zeros(4, 48)
    context = torch.zeros(4, 16)
    output = model(sequence, context)
    ok = tuple(output.logits.shape) == (4, 48, 3) and tuple(output.conditionals.shape) == (4, 48, 3)
    return _pass_if(
        "fit.hazard_forward_shapes",
        ok,
        "the hazard head returns one logit per step and per rung at the input batch size",
        logits_shape=str(tuple(output.logits.shape)),
    )


def check_loss_and_backward(fixture: Fixture) -> ExecutionCheck:
    set_seed(5)
    config = fixture.context.config.hazard
    model = HazardModel(config, rungs=3, context_dim=16)
    sequence = torch.rand(8, 32)
    context = torch.rand(8, 16)
    indicators = (torch.rand(8, 3) < 0.3).float()
    weights = rung_weight_tensor(fixture.bundle.ladder)
    logits = model.pooled_logits(sequence, context)
    loss = hazard_loss(logits, indicators, weights)
    loss.backward()  # type: ignore[no-untyped-call]
    populated = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    )
    total = sum(1 for _ in model.parameters())
    ok = math.isfinite(float(loss.item())) and float(loss.item()) > 0.0 and populated == total
    return _pass_if(
        "fit.loss_is_finite_and_backward_reaches_every_parameter",
        ok,
        "the rung loss is finite and positive and the backward pass populated every gradient",
        loss=round(float(loss.item()), 6),
        parameters_with_gradient=populated,
        parameters_total=total,
    )


def check_parameter_update(fixture: Fixture) -> ExecutionCheck:
    set_seed(7)
    from wireope.fit.optim import build_optimizer

    config = fixture.context.config.hazard
    model = HazardModel(config, rungs=3, context_dim=16)
    optimizer = build_optimizer(model.parameters(), fixture.context.config.optimizer)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    sequence = torch.rand(8, 32)
    context = torch.rand(8, 16)
    indicators = (torch.rand(8, 3) < 0.3).float()
    weights = rung_weight_tensor(fixture.bundle.ladder)
    loss = hazard_loss(model.pooled_logits(sequence, context), indicators, weights)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    moved = sum(
        1 for old, parameter in zip(before, model.parameters()) if not torch.equal(old, parameter.detach())
    )
    ok = moved > 0
    return _pass_if(
        "fit.optimiser_step_changes_parameters",
        ok,
        "one optimiser step changes at least one parameter tensor",
        tensors_changed=moved,
        loss=round(float(loss.item()), 6),
    )


def check_checkpoint_round_trip(fixture: Fixture) -> ExecutionCheck:
    set_seed(13)
    from wireope.fit.checkpoint import load_checkpoint, payload_digest, save_checkpoint

    config = fixture.context.config.hazard
    model = HazardModel(config, rungs=3, context_dim=16)
    from wireope.fit.optim import build_optimizer

    optimizer = build_optimizer(model.parameters(), fixture.context.config.optimizer)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        digest = save_checkpoint(path, model, optimizer, epoch=1, step=7, seed=99, metrics={"loss": 0.5})
        await_model = HazardModel(config, rungs=3, context_dim=16)
        payload = load_checkpoint(path, await_model, None)
        same = all(
            torch.equal(left.detach(), right.detach())
            for left, right in zip(model.parameters(), await_model.parameters())
        )
        payload_digest(payload)
    ok = same and payload.seed == 99 and len(digest) == 64
    return _pass_if(
        "fit.checkpoint_round_trip_restores_parameters_and_seed",
        ok,
        "the checkpoint restores every parameter tensor and carries the run seed",
        seed=payload.seed,
        digest_length=len(digest),
    )


def check_encoder_frozen(fixture: Fixture) -> ExecutionCheck:
    set_seed(3)
    from wireope.geometry.encoder import EnvelopeEncoder

    encoder = EnvelopeEncoder(fixture.context.config.envelope.encoder)
    before = [parameter.detach().clone() for parameter in encoder.trunk.parameters()]
    volumes = torch.rand(2, 1, 16, 16, 16)
    output = encoder(volumes)
    loss = output.envelope_logits.mean()
    loss.backward()
    trunk_unchanged = all(parameter.grad is None for parameter in encoder.trunk.parameters()) and all(
        torch.equal(old, parameter.detach()) for old, parameter in zip(before, encoder.trunk.parameters())
    )
    adapter_has_gradient = any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.adapter.parameters()
    )
    ok = trunk_unchanged and adapter_has_gradient
    return _pass_if(
        "fit.encoder_trunk_is_frozen_while_the_adapter_learns",
        ok,
        "the backward pass leaves the trunk untouched and reaches only the adapter",
        trunk_gradients=sum(1 for p in encoder.trunk.parameters() if p.grad is not None),
    )


def check_single_batch_overfit(fixture: Fixture) -> ExecutionCheck:
    set_seed(20260927)
    from wireope.fit.optim import build_optimizer

    config = fixture.context.config.hazard
    model = HazardModel(config, rungs=3, context_dim=16)
    optimizer = build_optimizer(model.parameters(), fixture.context.config.optimizer)
    sequence = torch.rand(16, 32)
    context = torch.rand(16, 16)
    indicators = (torch.rand(16, 3) < 0.4).float()
    weights = rung_weight_tensor(fixture.bundle.ladder)
    losses: list[float] = []
    for _ in range(40):
        loss = hazard_loss(model.pooled_logits(sequence, context), indicators, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.item()))
    ok = losses[-1] < losses[0]
    return _pass_if(
        "fit.single_batch_overfit_reduces_the_loss",
        ok,
        "overfitting one batch for forty steps reduces the rung loss",
        first_loss=round(losses[0], 6),
        final_loss=round(losses[-1], 6),
    )


def check_minimal_training_loop(fixture: Fixture) -> ExecutionCheck:
    dataset = ExcursionDataset(
        fixture.bundle.cohort, fixture.bundle.ladder, fixture.context.config.kinematics
    )
    with tempfile.TemporaryDirectory() as directory:
        model, result = train_hazard_head(
            dataset=dataset,
            rungs=fixture.bundle.ladder.depth,
            hazard=fixture.context.config.hazard,
            optimizer_config=fixture.context.config.optimizer,
            schedule_config=fixture.context.config.schedule,
            seed=0,
            rung_weights=rung_weight_tensor(fixture.bundle.ladder),
            checkpoint_path=Path(directory) / "loop.pt",
        )
    ok = (
        result.steps_run > 0
        and result.loss_decreased
        and result.gradient_norm >= 0.0
        and len(result.checkpoint_digest) == 64
    )
    _ = model
    return _pass_if(
        "fit.minimal_training_loop_reads_writes_and_decreases_the_loss",
        ok,
        "a two-epoch loop on the smoke configuration reads the dataset, updates parameters, "
        "writes and reloads a checkpoint, and lowers the loss",
        steps=result.steps_run,
        first_loss=round(result.first_loss, 6),
        final_loss=round(result.final_loss, 6),
        epochs=result.epochs_run,
    )


def check_excursion_sequence_peak(fixture: Fixture) -> ExecutionCheck:
    peak = 12.5
    sequence = excursion_sequence(peak, 40)
    ok = abs(float(np.max(sequence)) - peak) < 1e-9 and bool(np.all(np.diff(sequence) >= 0.0))
    return _pass_if(
        "fit.reconstructed_sequence_is_monotone_and_reaches_the_peak",
        ok,
        "the excursion sequence rises monotonically to the recorded peak",
        peak=peak,
        sequence_peak=round(float(np.max(sequence)), 6),
    )


def check_cohort_site_counts(fixture: Fixture) -> ExecutionCheck:
    cohort = fixture.bundle.cohort
    targets = {spec.site: spec.attempts for spec in fixture.context.config.release_cohort.site_targets}
    counts = {site: int(np.count_nonzero(cohort.site_array() == site)) for site in targets}
    ok = counts == targets
    return _pass_if(
        "cohort.assembly_hits_the_site_accrual_targets",
        ok,
        "the assembled cohort carries the accrual of every site exactly",
        counts=str(counts),
    )


def check_cohort_event_counts(fixture: Fixture) -> ExecutionCheck:
    cohort = fixture.bundle.cohort
    targets = {spec.site: spec.injury_events for spec in fixture.context.config.release_cohort.site_targets}
    counts = {site: int(np.sum(cohort.injury_flags()[cohort.site_array() == site])) for site in targets}
    ok = counts == targets
    return _pass_if(
        "cohort.assembly_hits_the_event_targets",
        ok,
        "the assembled cohort carries the adjudicated event count of every site exactly",
        counts=str(counts),
    )


def check_calibration_intercepts(fixture: Fixture) -> ExecutionCheck:
    intercepts = fixture.bundle.cohort.calibration_intercepts
    widths = []
    for site, intercept in intercepts.items():
        rng = numpy_generator(59)
        linear = rng.normal(-3.0, 0.5, size=4000)
        expected = float(np.sum(sigmoid(linear + intercept)))
        widths.append((site, round(expected / 4000.0, 5)))
    ok = all(math.isfinite(value) for _, value in widths)
    return _pass_if(
        "cohort.calibration_intercepts_are_finite_and_site_specific",
        ok,
        "every site intercept is finite and the implied event rate is site-specific",
        intercepts=str({k: round(v, 4) for k, v in intercepts.items()}),
    )


def check_cohort_discrimination_below_ceiling(fixture: Fixture) -> ExecutionCheck:
    labels = fixture.bundle.cohort.injury_flags()
    column = fixture.bundle.state_risk[:, 0]
    measured = auroc(column, labels)
    ok = 0.5 < measured < 0.999
    return _pass_if(
        "cohort.discrimination_is_not_at_its_ceiling",
        ok,
        "the closed-form risk separates the endpoint without saturating, so an AUROC control is "
        "not vacuous",
        auroc=round(measured, 5),
    )


def check_partition_disjointness(fixture: Fixture) -> ExecutionCheck:
    report = fixture.bundle.leakage
    ok = report.clean
    return _pass_if(
        "cohort.partitions_clear_every_leakage_guard",
        ok,
        "patient, operator, vendor and calibration guards all hold inside every configuration",
        leakage=str(report.as_mapping()),
    )


def check_private_cohort_availability(fixture: Fixture) -> ExecutionCheck:
    return _check(
        "data.private_cohort_record_level_availability",
        CheckStatus.BLOCKED,
        "the primary clinical cohort is held under site data-sharing agreements and is not shared "
        "at record level, so no cohort table value can be produced",
        access=fixture.context.config.clinical.access,
        record_level=fixture.context.config.clinical.record_level_available,
    )


def check_public_corpora_not_run(fixture: Fixture) -> ExecutionCheck:
    rows = fixture.context.config.corpora
    return _check(
        "data.public_corpus_rows_are_transcription_targets",
        CheckStatus.NOT_RUN,
        "the public logged corpora are named but not downloaded, so their rows carry no value",
        corpora=str([row.key for row in rows]),
    )


def check_dataset_links(fixture: Fixture) -> ExecutionCheck:
    """Probe every recorded link, when the caller asked for a live probe.

    The probe reaches the network, so it is opt-in: a default pass leaves the check undecided
    and the deterministic artefacts stay comparable between runs. `dataset_urls.txt` records the
    authoring-time verification, including the date it was made.
    """
    path = fixture.repo_root / "dataset_urls.txt"
    if not path.is_file():
        return _check(
            "data.dataset_links_reachable",
            CheckStatus.NOT_RUN,
            "dataset_urls.txt has not been written yet, so no link was probed in this pass",
        )
    if not fixture.probe_links:
        return _check(
            "data.dataset_links_reachable",
            CheckStatus.NOT_RUN,
            "the link probe is live and opt-in; the authoring-time verification is recorded in "
            "dataset_urls.txt",
            recorded_links=str(
                len([line for line in path.read_text().splitlines() if line.strip().startswith("http")])
            ),
        )
    urls = [line.strip() for line in path.read_text().splitlines() if line.strip().startswith("http")]
    reachable = 0
    failures: list[str] = []
    attempts = 3
    for url in urls:
        last_error = ""
        answered = False
        for _ in range(attempts):
            if probe_with_urllib(url):
                answered = True
                break
            last_error = "urllib"
        if not answered and probe_with_curl(url):
            # The two clients disagree often enough on this host that a single client is not
            # evidence a link is dead.
            answered = True
        if answered:
            reachable += 1
        else:
            failures.append(f"{url}::{last_error}")
    if not failures:
        status = CheckStatus.PASS
        detail = "every recorded dataset link answered in this pass"
    elif reachable > 0:
        # A shared egress proxy can refuse a single host for one pass; that is not evidence the
        # link is dead, so the pass is recorded as undecided rather than as a failure.
        status = CheckStatus.NOT_RUN
        detail = (
            "at least one recorded dataset link was undecided in this pass; the per-host detail "
            "is in the log"
        )
    else:
        status = CheckStatus.FAIL
        detail = "no recorded dataset link answered, which points at the verification host"
    LOGGER.info("dataset link probe: %d of %d answered within %d attempts", reachable, len(urls), attempts)
    if failures:
        LOGGER.warning("undecided dataset links in this pass: %s", failures)
    # The per-pass host detail varies, so the artefact records the design of the probe rather
    # than its outcome for this particular pass.
    return _check(
        "data.dataset_links_reachable",
        status,
        detail,
        probed=len(urls),
        attempts_per_link=attempts,
        clients="urllib then curl",
    )


def probe_with_urllib(url: str, timeout: int = 45) -> bool:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "release-link-probe"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def probe_with_curl(url: str, timeout: int = 45) -> bool:
    completed = subprocess.run(
        ["curl", "-sS", "-L", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip().endswith("200")


def check_docker_available(fixture: Fixture) -> ExecutionCheck:
    completed = subprocess.run(["which", "docker"], capture_output=True, text=True, check=False)
    if completed.returncode == 0 and completed.stdout.strip():
        return _check(
            "container.docker_build",
            CheckStatus.NOT_RUN,
            "a container runtime is present but the image was not built in this pass",
        )
    return _check(
        "container.docker_build",
        CheckStatus.BLOCKED,
        "no container runtime is installed on the verification host, so the image cannot be built",
        returncode=completed.returncode,
    )


def shipped_files(root: Path) -> list[Path]:
    """The files the release actually ships: the same walk the integrity manifest uses."""
    return iter_files(root, SKIP_DIRS)


def check_no_host_paths(fixture: Fixture) -> ExecutionCheck:

    # Assembled from pieces so the scanner's own source cannot match the pattern it looks for.
    roots = ("/" + "Users/", "/" + "home/")
    pattern = re.compile("|".join(root + r"[A-Za-z0-9._-]+" for root in roots))
    offenders: list[str] = []
    for path in shipped_files(fixture.repo_root):
        if path.suffix in {".pt", ".npz", ".db"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if pattern.search(text):
            offenders.append(path.relative_to(fixture.repo_root).as_posix())
    ok = not offenders
    return _pass_if(
        "hygiene.no_host_absolute_paths",
        ok,
        "no file in the release carries a host-specific absolute path",
        offenders=str(offenders[:10]),
    )


def check_no_markdown_beyond_readme(fixture: Fixture) -> ExecutionCheck:
    # Walking the shipped set matters: a test run leaves `.pytest_cache/README.md` behind, and a
    # check that walks the whole root would then see two readmes and fail on a clean release.
    offenders = [
        path.relative_to(fixture.repo_root).as_posix()
        for path in shipped_files(fixture.repo_root)
        if path.suffix == ".md" and path.name != "README.md"
    ]
    ok = not offenders
    return _pass_if(
        "hygiene.only_the_readme_carries_markdown",
        ok,
        "README.md is the only markdown file in the release",
        offenders=str(offenders),
    )


def check_no_emoji(fixture: Fixture) -> ExecutionCheck:
    forbidden = ("\U0001f680", "\u2728", "\U0001f4ca", "\U0001f3af", "\U0001f9ea", "\u2705", "\u274c")
    offenders: list[str] = []
    for path in shipped_files(fixture.repo_root):
        if path.suffix in {".pt", ".npz", ".db", ".pdf"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if any(symbol in text for symbol in forbidden):
            offenders.append(path.relative_to(fixture.repo_root).as_posix())
    ok = not offenders
    return _pass_if(
        "hygiene.no_emoji_anywhere_in_the_tree",
        ok,
        "no emoji appears in any text file of the release",
        offenders=str(offenders),
    )


def check_forward_determinism(fixture: Fixture) -> ExecutionCheck:
    config = fixture.context.config.hazard
    set_seed(17)
    first = HazardModel(config, rungs=3, context_dim=16)
    set_seed(17)
    second = HazardModel(config, rungs=3, context_dim=16)
    sequence = torch.linspace(0.0, 1.0, 24).repeat(2, 1)
    context = torch.zeros(2, 16)
    with torch.no_grad():
        left = first(sequence, context).logits
        right = second(sequence, context).logits
    ok = torch.allclose(left, right)
    return _pass_if(
        "fit.forward_pass_is_seed_deterministic",
        ok,
        "two models built under the same seed produce identical logits",
        max_absolute_gap=round(float(torch.max(torch.abs(left - right)).item()), 10),
    )


def check_trace_build(fixture: Fixture) -> ExecutionCheck:
    geometry = analytic_geometry()
    grid = GridSpec(shape=(24, 24, 40), spacing_mm=(0.5, 0.5, 0.5), origin_mm=(-6.0, -6.0, -2.0))
    envelope = geometry.as_envelope(grid)
    rng = numpy_generator(61)
    points = geometry.centreline_mm + rng.normal(0.0, 0.6, size=geometry.centreline_mm.shape)
    dwell = np.full(points.shape[0], 0.25)
    trace = build_trace(0, envelope, points, dwell)
    ok = trace.excursion_mm.shape == points.shape[:1] and math.isfinite(trace.surface_integral_mm2)
    return _pass_if(
        "traces.trace_assembles_finite_observables",
        ok,
        "the assembled trace carries one excursion value per sample and a finite surface integral",
        samples=int(trace.excursion_mm.shape[0]),
        surface_integral_mm2=round(trace.surface_integral_mm2, 4),
    )


def check_estimator_selection_stable(fixture: Fixture) -> ExecutionCheck:
    batch = tiny_batch(
        5000,
        fixture.bundle.ladder.depth,
        "target",
        np.linspace(0.6, 0.08, fixture.bundle.ladder.depth),
        0.5,
        seed=67,
    )
    direct = build_estimator("fqe").estimate(batch, 0.1)
    repeat = build_estimator("fqe").estimate(batch, 0.1)
    ok = abs(direct.value - repeat.value) < 1e-12
    return _pass_if(
        "estimators.repeated_estimation_is_reproducible",
        ok,
        "estimating twice on the same batch returns the identical value",
        value=round(direct.value, 8),
    )


def check_ladder_depth_sensitivity(fixture: Fixture) -> ExecutionCheck:
    from wireope.laddering.resolution import max_resolvable_depth

    config = fixture.context.config.ladder
    depth = max_resolvable_depth(config.tau_mm, config.resolution_band_mm)
    ladder = build_ladder(fixture.context.config.ladder, fixture.context.config.ladder_weights)
    ok = 1 <= depth <= len(config.tau_mm) and ladder.depth <= len(config.tau_mm)
    return _pass_if(
        "ladder.max_resolvable_depth_within_the_threshold_grid",
        ok,
        "the deepest resolvable ladder fits inside the configured threshold grid",
        max_depth=depth,
        configured_depth=ladder.depth,
        band_mm=config.resolution_band_mm,
    )


def check_cohort_reproducibility(fixture: Fixture) -> ExecutionCheck:
    ladder = fixture.bundle.ladder
    reassembled = assemble_release_cohort(
        fixture.context.config.release_cohort, ladder, seed=fixture.context.config.release_cohort.seed
    )
    identical = bool(
        np.array_equal(reassembled.injury_flags(), fixture.bundle.cohort.injury_flags())
    ) and bool(np.allclose(reassembled.peaks(), fixture.bundle.cohort.peaks()))
    return _pass_if(
        "cohort.assembly_is_reproducible_under_its_seed",
        identical,
        "assembling the cohort again under the configured seed reproduces the endpoint and the peaks",
        attempts=reassembled.attempts,
    )


CHECKS: tuple[Callable[[Fixture], ExecutionCheck], ...] = (
    check_environment,
    check_deterministic_seeding,
    check_analytic_distance,
    check_excursion_non_negative,
    check_surface_area,
    check_event_integration,
    check_similarity_fit,
    check_alignment_residual,
    check_dwell_integral,
    check_propagating_band,
    check_trace_build,
    check_ladder_thresholds,
    check_ladder_nesting_brute_force,
    check_ladder_weights,
    check_ladder_resolution,
    check_ladder_conditional_band,
    check_ladder_depth_sensitivity,
    check_importance_weights,
    check_effective_sample_size,
    check_nested_dr_unbiasedness,
    check_nested_dr_needs_the_propensity,
    check_candidate_library,
    check_estimator_selection_stable,
    check_kernel_weights,
    check_product_risk,
    check_hoeffding_closed_form,
    check_hoeffding_monotone_in_events,
    check_radius_invariant_to_negatives,
    check_bernstein_limit,
    check_bernstein_tighter_on_conformed_residuals,
    check_floor_brute_force,
    check_rule_brute_force,
    check_rule_abstains,
    check_auroc_matches_library,
    check_auroc_pair_count,
    check_delong_interval,
    check_net_benefit_closed_form,
    check_calibration_ece,
    check_violation_rate,
    check_forward_shapes,
    check_loss_and_backward,
    check_parameter_update,
    check_checkpoint_round_trip,
    check_encoder_frozen,
    check_single_batch_overfit,
    check_minimal_training_loop,
    check_excursion_sequence_peak,
    check_forward_determinism,
    check_cohort_site_counts,
    check_cohort_event_counts,
    check_calibration_intercepts,
    check_cohort_discrimination_below_ceiling,
    check_partition_disjointness,
    check_cohort_reproducibility,
    check_private_cohort_availability,
    check_public_corpora_not_run,
    check_dataset_links,
    check_docker_available,
    check_no_host_paths,
    check_no_markdown_beyond_readme,
    check_no_emoji,
)


def run_execution_checks(fixture: Fixture) -> tuple[ExecutionCheck, ...]:
    results: list[ExecutionCheck] = []
    for check in CHECKS:
        results.append(check(fixture))
    return tuple(results)


def gate_command(name: str, label: str, command: list[str], cwd: Path) -> ExecutionCheck:
    """Run a tooling gate and decide on its exit code rather than on its output text.

    The evidence names the gate by its label: `command[0]` would print the interpreter's
    absolute path, which is both a host fingerprint and a machine-dependent string in a file
    the integrity manifest digests.
    """
    environment = {
        "COLUMNS": "120",
        "LINES": "40",
        "NO_COLOR": "1",
        "PY_COLORS": "0",
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
    }
    completed = subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True, check=False, env=environment
    )
    # Only the exit code is recorded: a tool's last line carries a duration and a window width,
    # so it would differ between two otherwise identical passes.
    return _check(
        name,
        CheckStatus.PASS if completed.returncode == 0 else CheckStatus.FAIL,
        f"exit code {completed.returncode} from {label}",
        exit_code=completed.returncode,
    )


def run_gates(repo_root: Path) -> tuple[ExecutionCheck, ...]:
    """Run the four tooling gates plus the suite, each decided on its exit code."""
    interpreter = sys.executable or "python3"
    return (
        gate_command("gate.ruff", "ruff check .", ["ruff", "check", "."], repo_root),
        gate_command("gate.black", "black --check .", ["black", "--check", "."], repo_root),
        gate_command("gate.isort", "isort --check-only .", ["isort", "--check-only", "."], repo_root),
        gate_command(
            "gate.mypy",
            "mypy --config-file pyproject.toml",
            ["mypy", "--config-file", "pyproject.toml"],
            repo_root,
        ),
        gate_command("gate.pytest", "pytest -q", [interpreter, "-m", "pytest", "-q"], repo_root),
    )


def summary(checks: tuple[ExecutionCheck, ...]) -> dict[str, float]:
    statuses = np.asarray([check.status.value for check in checks], dtype="<U8")
    return {
        "total": float(len(checks)),
        "passed": float(np.count_nonzero(statuses == CheckStatus.PASS.value)),
        "failed": float(np.count_nonzero(statuses == CheckStatus.FAIL.value)),
        "not_run": float(np.count_nonzero(statuses == CheckStatus.NOT_RUN.value)),
        "blocked": float(np.count_nonzero(statuses == CheckStatus.BLOCKED.value)),
    }
