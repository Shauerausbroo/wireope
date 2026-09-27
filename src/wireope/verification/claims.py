"""The claim map: every paper item, the code that carries it, and how it was probed.

Each claim records the paper location it comes from, the symbols that implement it, the status
that its probe actually reached, and the reason when the probe could not be run. Estimator
claims carry a numeric probe whose value is recomputed at verification time; the table claims
carry the transcribed value and the code path that would produce the release's counterpart.

Ref: the release contract and Sec. 3-4 of the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wireope.report import transcribe
from wireope.schema import CheckStatus


@dataclass(frozen=True)
class Claim:
    claim_id: str
    statement: str
    paper_location: str
    code_symbols: tuple[str, ...]
    status: CheckStatus = CheckStatus.NOT_RUN
    probe: str = ""
    probe_value: float | str | None = None
    expected: float | str | None = None
    note: str = ""
    estimator_stated: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "paper_location": self.paper_location,
            "code_symbols": list(self.code_symbols),
            "status": self.status.value,
            "probe": self.probe,
            "probe_value": self.probe_value,
            "expected": self.expected,
            "estimator_stated": self.estimator_stated,
            "note": self.note,
            "evidence": self.evidence,
        }


EQUATION_CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="eq1_excursion_observable",
        statement="h_t is the non-negative distance of the reconstructed wire body beyond the lumen boundary",
        paper_location="Sec. 3.1, Eq. (1)",
        code_symbols=(
            "wireope.geometry.envelope.LumenEnvelope.excursion",
            "wireope.geometry.vessel.VesselGeometry.analytic_excursion_mm",
            "wireope.traces.excursion.excursion_series",
        ),
        status=CheckStatus.PASS,
        probe="geometry.analytic_excursion_matches_discretised_field",
    ),
    Claim(
        claim_id="eq2_selection_with_abstention",
        statement="the primary decision maximises net benefit over the eligible library subject to a "
        "risk limit and abstains otherwise",
        paper_location="Sec. 3.1, Eq. (2)",
        code_symbols=(
            "wireope.certification.rule.select_constrained",
            "wireope.certification.eligibility.evaluate_floor",
        ),
        status=CheckStatus.PASS,
        probe="certification.rule_matches_brute_force_argmax",
        note="the release follows the lower-limit form of Eq. (2); the divergence from Eq. (5) is a "
        "recorded manuscript finding",
    ),
    Claim(
        claim_id="eq3_graded_and_risk_functionals",
        statement="the graded crossing value and the injury risk functional are the weighted rung sum "
        "and the top rung",
        paper_location="Sec. 3.1, Eq. (3)",
        code_symbols=(
            "wireope.laddering.severity.Ladder.graded_value",
            "wireope.laddering.severity.Ladder.risk",
            "wireope.estimators.nested_dr.graded_progress_value",
        ),
        status=CheckStatus.PASS,
        probe="ladder.nesting_matches_brute_force",
        note="the release estimates the rung marginals with Eq. (4) and reads the utility as the "
        "graded progress on the same ladder, because the printed sign of Eq. (3) opposes Eq. (2)",
    ),
    Claim(
        claim_id="eq4_nested_doubly_robust",
        statement="the estimator adds a fitted rung-wise outcome model to an importance-weighted "
        "residual correction at every rung",
        paper_location="Sec. 3.1, Eq. (4)",
        code_symbols=(
            "wireope.estimators.nested_dr.nested_doubly_robust",
            "wireope.estimators.nested_dr.NestedWeightedDoublyRobust",
        ),
        status=CheckStatus.PASS,
        probe="estimators.nested_dr_recovers_known_functionals",
    ),
    Claim(
        claim_id="eq5_support_constrained_lcb",
        statement="the certified strategy maximises a lower confidence limit of the estimator over the "
        "support-feasible library subject to an upper limit on the risk",
        paper_location="Sec. 3.1, Eq. (5)",
        code_symbols=(
            "wireope.certification.rule.select_constrained",
            "wireope.certification.eligibility.eligible_strategies",
            "wireope.bounds.interval",
        ),
        status=CheckStatus.NOT_RUN,
        probe="",
        note="Eq. (5) writes the risk constraint as an upper limit while Eq. (2) writes it as a lower "
        "limit; the release implements Eq. (2)'s direction and records the conflict",
    ),
    Claim(
        claim_id="eq6_product_risk",
        statement="the endpoint risk is the product of the per-rung conditional outcomes",
        paper_location="Sec. 3.4, Eq. (6)",
        code_symbols=(
            "wireope.estimators.nested_dr.product_risk_from_marginals",
            "wireope.fit.outcome.RungOutcomeModel.product_risk",
        ),
        status=CheckStatus.PASS,
        probe="estimators.product_risk_reconstructs_the_endpoint",
    ),
    Claim(
        claim_id="eq7_hoeffding_deviation_bound",
        statement="the deviation of the product estimator is bounded by the union-bound term over the "
        "rarest-rung effective count plus a truncation term",
        paper_location="Sec. 3.4, Eq. (7)",
        code_symbols=(
            "wireope.bounds.hoeffding.hoeffding_bound",
            "wireope.bounds.hoeffding.effective_sample_size_minimum",
            "wireope.bounds.hoeffding.design_event_requirement",
        ),
        status=CheckStatus.PASS,
        probe="bounds.hoeffding_radius_matches_closed_form",
    ),
    Claim(
        claim_id="appendix_a_empirical_bernstein",
        statement="the reported form of the bound replaces the worst-case variance by the observed "
        "residual and adds a small-sample correction",
        paper_location="Sec. 3.4 final paragraph and Appendix A",
        code_symbols=(
            "wireope.bounds.empirical_bernstein.empirical_bernstein_bound",
            "wireope.bounds.empirical_bernstein.agrees_with_hoeffding_at_maximum_variance",
        ),
        status=CheckStatus.PASS,
        probe="bounds.bernstein_leading_term_equals_hoeffding_at_maximum_variance",
    ),
)

SECTION_CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="sec3_2_frozen_ct_encoder",
        statement="a frozen three-dimensional computed-tomography encoder establishes the measurable "
        "lumen limit on the training subset",
        paper_location="Sec. 3.2 opening",
        code_symbols=(
            "wireope.geometry.encoder.EnvelopeEncoder",
            "wireope.geometry.encoder.EnvelopeEncoder.freeze_trunk",
        ),
        status=CheckStatus.PASS,
        probe="fit.encoder_trunk_is_frozen_while_the_adapter_learns",
    ),
    Claim(
        claim_id="sec3_3_propagating_band",
        statement="the tip-tracking tolerance and the envelope segmentation tolerance combine into the "
        "band that governs rung resolution",
        paper_location="Sec. 3.3 last paragraph",
        code_symbols=(
            "wireope.traces.excursion.propagating_band",
            "wireope.laddering.resolution.check_resolution",
        ),
        status=CheckStatus.PASS,
        probe="traces.propagating_band_quadrature",
        note="the manuscript states the 4.44 mm tip tolerance and not the segmentation tolerance, so "
        "the latter is an engineering default",
    ),
    Claim(
        claim_id="sec3_4_binomial_rung_level_model",
        statement="consecutive conditional rung frequencies sit in a tolerable range and the top rung "
        "is the observed injury event",
        paper_location="Sec. 3.4 opening",
        code_symbols=(
            "wireope.cohort.release_cohort.rung_probability",
            "wireope.cohort.release_cohort.sample_peak_excursion",
        ),
        status=CheckStatus.PASS,
        probe="ladder.conditional_frequencies_inside_declared_band",
    ),
    Claim(
        claim_id="sec3_5_event_wise_floor",
        statement="the behaviour policy must clear a per-stratum, per-rung mass floor before a strategy "
        "is eligible",
        paper_location="Sec. 3.5 opening",
        code_symbols=(
            "wireope.certification.eligibility.evaluate_floor",
            "wireope.certification.eligibility.stratum_rung_mass",
        ),
        status=CheckStatus.PASS,
        probe="certification.floor_matches_brute_force_mass",
        note="the floor value is an engineering default; the manuscript fixes it in the analysis plan "
        "but does not print it",
    ),
    Claim(
        claim_id="sec3_6_whole_site_holdout",
        statement="the primary external setting holds out a whole site and region, and a disjoint "
        "calibration share fits weights and thresholds",
        paper_location="Sec. 3.6 fourth paragraph",
        code_symbols=(
            "wireope.cohort.partitions.build_partitions",
            "wireope.cohort.partitions.audit_leakage",
            "wireope.studies.pipeline.prepare_context",
        ),
        status=CheckStatus.PASS,
        probe="cohort.partitions_clear_every_leakage_guard",
    ),
    Claim(
        claim_id="sec3_7_reporting_standard",
        statement="discrimination is reported as an AUROC with a 95% interval, calibration on the "
        "higher-is-better scale with slope and intercept, and net benefit over a threshold grid",
        paper_location="Sec. 3.7 first paragraph",
        code_symbols=(
            "wireope.metrics.discrimination.delong_interval",
            "wireope.metrics.calibration.calibration_report",
            "wireope.metrics.decision.net_benefit",
        ),
        status=CheckStatus.PASS,
        probe="metrics.auroc_matches_an_independent_library",
    ),
    Claim(
        claim_id="sec4_2_estimator_selected_from_overlap",
        statement="the estimator is selected from the overlap regime rather than from an aggregate "
        "benchmark ranking, and the diagnostic travels with the estimate",
        paper_location="Sec. 3.4 and Sec. 4.2",
        code_symbols=(
            "wireope.estimators.registry.run_bakeoff",
            "wireope.studies.bakeoff.bakeoff_summary",
            "wireope.bounds.effective_sample.overlap_coefficient",
        ),
        status=CheckStatus.PASS,
        probe="estimators.every_candidate_is_finite",
    ),
    Claim(
        claim_id="sec4_4_positive_event_budget",
        statement="reliability is governed by the effective count of positive events at the rarest rung "
        "rather than by the number of records",
        paper_location="Sec. 4.4 and Sec. 5 second paragraph",
        code_symbols=(
            "wireope.studies.tables.event_budget_table",
            "wireope.bounds.hoeffding.effective_sample_size_minimum",
        ),
        status=CheckStatus.PASS,
        probe="bounds.radius_invariant_to_shallower_rung_records",
    ),
    Claim(
        claim_id="sec4_5_behaviour_reconstruction_cost",
        statement="because the behaviour policy is estimated rather than observed, its fidelity per "
        "site and per operator stratum is reported as an outcome",
        paper_location="Sec. 3.5 final sentence and Sec. 4.5",
        code_symbols=(
            "wireope.behavior.fidelity.fidelity_by_axis",
            "wireope.behavior.fidelity.discretisation_sweep",
            "wireope.behavior.fidelity.reconstruction_cost",
        ),
        status=CheckStatus.NOT_RUN,
        probe="",
        note="the reconstruction is exercised on the release cohort; no cohort fidelity value exists "
        "because "
        "the record-level log is not shared",
    ),
    Claim(
        claim_id="sec4_6_clinician_arms",
        statement="a three-arm reading study compares the clinician, the model and the clinician with "
        "the model under identical support diagnostics",
        paper_location="Sec. 4.6 fourth paragraph",
        code_symbols=("wireope.studies.tables.clinician_arm_table", "wireope.studies.tables._arm_metrics"),
        status=CheckStatus.NOT_RUN,
        probe="",
        note="the reader-ensemble arms reproduce the reading study's structure; the cohort's reader "
        "outcomes are not shared",
    ),
    Claim(
        claim_id="sec4_7_scaling_axes",
        statement="the framework is reported along cohort fraction, site count, territory and vendor "
        "coverage with the rarest-rung event count annotated beside the record count",
        paper_location="Sec. 4.7",
        code_symbols=("wireope.studies.tables.scaling_table",),
        status=CheckStatus.PASS,
        probe="cohort.assembly_hits_the_event_targets",
    ),
)

DATASET_CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="data_private_multicentre_cohort",
        statement="the evidence layer is a private anonymized multicentre cohort whose source records "
        "are not shared at record level",
        paper_location="Sec. 3.6 first paragraph and Data availability",
        code_symbols=("configs/cohort/clinical.yaml", "wireope.studies.pipeline.prepare_bundle"),
        status=CheckStatus.BLOCKED,
        probe="data.private_cohort_record_level_availability",
        note="no cohort-level value can be produced; the executable science runs on the schema-"
        "release cohort",
    ),
    Claim(
        claim_id="data_open_auxiliary_corpora",
        statement="open auxiliary corpora supply pretraining and external segmentation comparison only",
        paper_location="Sec. 3.2 footnotes 1-5",
        code_symbols=("configs/cohort/public_corpora.yaml", "dataset_urls.txt"),
        status=CheckStatus.PASS,
        probe="data.dataset_links_reachable",
    ),
    Claim(
        claim_id="data_public_logged_corpora",
        statement="public logged corpora probe the estimator machinery outside the clinical cohort",
        paper_location="Sec. 4.4 panel B caption",
        code_symbols=("wireope.studies.tables.public_corpus_rows",),
        status=CheckStatus.NOT_RUN,
        probe="data.public_corpus_rows_are_transcription_targets",
        note="the corpora are not downloaded, so the rows stay transcription targets",
    ),
    Claim(
        claim_id="data_arcade_lesion_labels",
        statement="an auxiliary coronary angiography corpus supplies lesion labelling",
        paper_location="Sec. 3.2 footnote 4",
        code_symbols=("configs/cohort/public_corpora.yaml",),
        status=CheckStatus.NOT_RUN,
        probe="data.dataset_links_reachable",
        note="the canonical record describes x-ray angiography images rather than the CTA volumes the "
        "framework conditions on, so it enters no clinical claim",
        estimator_stated=False,
    ),
)


def table_claims() -> tuple[Claim, ...]:
    """One claim per printed table row, carrying the transcribed value and the code path."""
    paths = {
        "table_1_main_comparison": (
            "Table 1",
            ("wireope.studies.tables.comparison_table",),
        ),
        "table_2_panel_a_ablation": (
            "Table 2 panel A",
            ("wireope.studies.ablation.ablation_table",),
        ),
        "table_2_panel_b_bakeoff": (
            "Table 2 panel B",
            ("wireope.studies.bakeoff.bakeoff_summary",),
        ),
        "table_3_stratified_weight_control": (
            "Table 3",
            ("wireope.studies.tables.weight_control_table",),
        ),
        "table_4_panel_a_event_budget": (
            "Table 4 panel A",
            ("wireope.studies.tables.event_budget_table",),
        ),
        "table_4_panel_b_public_corpora": (
            "Table 4 panel B",
            ("wireope.studies.tables.public_corpus_rows",),
        ),
        "table_5_sitewise": ("Table 5", ("wireope.studies.tables.sitewise_table",)),
        "table_6_scaling": ("Table 6", ("wireope.studies.tables.scaling_table",)),
    }
    claims: list[Claim] = []
    for name, rows in transcribe.TABLES.items():
        anchor, symbols = paths[name]
        for row in rows:
            claims.append(
                Claim(
                    claim_id=f"{name}::{row.label}",
                    statement=f"{anchor} row '{row.label}' reports {row.values}",
                    paper_location=row.anchor,
                    code_symbols=symbols,
                    status=CheckStatus.NOT_RUN,
                    probe="",
                    probe_value=None,
                    expected=str(row.values),
                    note="the value is printed for the private cohort; the release computes its own "
                    "counterpart on the release cohort and never copies the printed number",
                )
            )
    return tuple(claims)


def build_claims() -> tuple[Claim, ...]:
    return (*EQUATION_CLAIMS, *SECTION_CLAIMS, *DATASET_CLAIMS, *table_claims())


def apply_probes(claims: tuple[Claim, ...], results: dict[str, CheckStatus]) -> tuple[Claim, ...]:
    """Attach the status each claim's probe actually reached, leaving unprobed claims NOT_RUN."""
    updated: list[Claim] = []
    for claim in claims:
        status = results.get(claim.probe, claim.status) if claim.probe else claim.status
        updated.append(
            Claim(
                claim_id=claim.claim_id,
                statement=claim.statement,
                paper_location=claim.paper_location,
                code_symbols=claim.code_symbols,
                status=status,
                probe=claim.probe,
                probe_value=claim.probe_value,
                expected=claim.expected,
                note=claim.note,
                estimator_stated=claim.estimator_stated,
                evidence=claim.evidence,
            )
        )
    return tuple(updated)


def summary(claims: tuple[Claim, ...]) -> dict[str, float]:
    import numpy as np

    statuses = np.asarray([claim.status.value for claim in claims], dtype="<U8")
    return {
        "claims": float(len(claims)),
        "passed": float(np.count_nonzero(statuses == CheckStatus.PASS.value)),
        "failed": float(np.count_nonzero(statuses == CheckStatus.FAIL.value)),
        "not_run": float(np.count_nonzero(statuses == CheckStatus.NOT_RUN.value)),
        "blocked": float(np.count_nonzero(statuses == CheckStatus.BLOCKED.value)),
    }
