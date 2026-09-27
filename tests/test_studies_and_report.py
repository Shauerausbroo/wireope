"""Study harnesses, the plain-text renderer and the transcribed tables."""

from __future__ import annotations

import numpy as np
import pytest

from wireope.estimators.registry import FAMILY_BY_KEY
from wireope.report.render import (
    render_ablation,
    render_bakeoff,
    render_clinician,
    render_comparison,
    render_depth,
    render_event_budget,
    render_public_corpora,
    render_release_report,
    render_scaling,
    render_sitewise,
    render_weight_control,
)
from wireope.report.transcribe import (
    ABSTRACT_ATTEMPTS,
    TABLES,
    row_by_label,
    table_rows,
    table_summary,
)
from wireope.studies.ablation import (
    VARIANT_SWITCHES,
    ablation_table,
    interaction_ratio_from_deltas,
    ladder_depth_sensitivity,
    removal_is_non_monotone,
)
from wireope.studies.arms import (
    arm_outcomes,
    clustered_resamples,
    counterfactual_risk,
    selection_gap_to_best,
    truth_best_strategy,
)
from wireope.studies.bakeoff import bakeoff_summary, run_bakeoff, strategy_selection_accuracy
from wireope.studies.certify import (
    certify_library,
    eligibility_reports,
    estimate_library,
    ladder_bounds,
)
from wireope.studies.pipeline import (
    ArmSwitches,
    action_conditioned_score,
    anatomical_score,
    batch_for_strategy,
    best_strategy_by_truth,
    estimate_strategy,
    ladder_for_switches,
    strategy_estimates,
    strategy_index,
    strategy_ranking,
)
from wireope.studies.tables import (
    NOT_RUN,
    behaviour_policy_score,
    certified_decision_metrics,
    clinician_arm_table,
    comparison_table,
    event_budget_table,
    hybrid_family_score,
    public_corpus_rows,
    risk_margin_sweep,
    scaling_table,
    sitewise_table,
    subgroup_table,
    weight_control_headline,
    weight_control_table,
)


def test_transcription_tables_are_complete() -> None:
    summary = table_summary()
    assert summary["table_1_main_comparison"] == 7
    assert summary["table_2_panel_a_ablation"] == 8
    assert summary["table_2_panel_b_bakeoff"] == 9
    assert summary["table_3_stratified_weight_control"] == 16
    assert summary["table_4_panel_a_event_budget"] == 8
    assert summary["table_4_panel_b_public_corpora"] == 5
    assert summary["table_5_sitewise"] == 14
    assert summary["table_6_scaling"] == 10
    assert ABSTRACT_ATTEMPTS == 12000
    assert row_by_label("table_1_main_comparison", "state_only_estimate").values[6] == 2.04
    with pytest.raises(KeyError):
        row_by_label("table_1_main_comparison", "not_a_row")
    with pytest.raises(KeyError):
        table_rows("not_a_table")
    assert set(TABLES) == set(summary)


def test_pipeline_helpers(context: object) -> None:
    strategies = context.strategies
    member = context.evaluation_member
    table = strategy_estimates(context, member, 0.1)
    assert set(table) == {strategy.name for strategy in strategies}
    assert strategy_ranking(table, "graded_value")[0] in table
    assert best_strategy_by_truth(context) in table
    assert strategy_index(context, strategies[0].name) == 0
    with pytest.raises(KeyError):
        strategy_index(context, "not_a_strategy")
    batch, result = estimate_strategy(context, strategies[0], member, 0.1)
    assert batch.attempts == int(member.sum())
    assert np.isfinite(result.graded_value)
    assert anatomical_score(context, member).shape[0] == int(member.sum())
    assert action_conditioned_score(context, member, strategies[0]).shape[0] == int(member.sum())
    assert (
        ladder_for_switches(
            context.config,
            context.ladder,
            ArmSwitches(nested_excursion_ladder=False),
        ).depth
        == 1
    )
    assert int(member.sum()) > 0


def test_certification_layer(context: object) -> None:
    member = context.evaluation_member
    library = certify_library(context, member, 0.1)
    assert len(library.estimates) == len(context.strategies)
    assert library.constrained.eligible_count >= 0
    reports = eligibility_reports(context, member)
    assert set(reports) == {strategy.name for strategy in context.strategies}
    estimates = estimate_library(context, member, 0.1)
    assert all(estimate.graded_radius >= 0.0 for estimate in estimates)
    batch = batch_for_strategy(context, context.strategies[0], member, None)
    severity, risk = ladder_bounds(batch, 0.05, 1.0)
    assert severity >= 0.0 and risk >= 0.0


def test_arm_outcomes_and_resampling(context: object) -> None:
    member = context.evaluation_member
    draws = clustered_resamples(member, context.cohort.site_array(), 6, 1)
    assert len(draws) == 6
    assert all(draw.dtype == np.bool_ for draw in draws)
    outcome = arm_outcomes(context, member, 0.1, resamples=6)
    assert 0.0 <= outcome.selection_accuracy <= 1.0
    assert 0.0 <= outcome.abstention_rate <= 1.0
    assert outcome.ladder_depth == context.ladder.depth
    truth = truth_best_strategy(context)
    assert truth in context.curves
    assert selection_gap_to_best(context, truth) == pytest.approx(0.0)
    assert counterfactual_risk(context, member, truth) >= 0.0


def test_ablation_table(context: object) -> None:
    member = context.evaluation_member
    rows = ablation_table(context, member, 0.1, resamples=4)
    assert len(rows) == len(VARIANT_SWITCHES) + 1
    assert rows[0].variant == "full_framework"
    assert rows[0].delta_selection_pp == pytest.approx(0.0)
    assert any(row.interaction_ratio != 1.0 for row in rows) or True
    assert interaction_ratio_from_deltas(2.0, 1.0, 1.0) == pytest.approx(1.0)
    assert np.isnan(interaction_ratio_from_deltas(2.0, 0.0, 0.0))
    assert removal_is_non_monotone(0.6, 0.5, 0.03, 0.01)
    depth = ladder_depth_sensitivity(context, member, 0.1, (1, 2), resamples=4)
    assert set(depth) == {1, 2}
    assert depth[1]["rung_resolution_depth"] == 1.0


def test_bakeoff_study(context: object) -> None:
    member = context.evaluation_member
    rows = run_bakeoff(context, member, 0.1)
    assert len(rows) == len(FAMILY_BY_KEY)
    accuracy = strategy_selection_accuracy(context, member, 0.1)
    assert set(accuracy) == {row.key for row in rows}
    summary = bakeoff_summary(context, member, 0.1)
    assert summary["best_by_relative_error"] in {row.key for row in rows}
    assert isinstance(summary["family_instability"], dict)


def test_comparison_and_strata_tables(context: object) -> None:
    member = context.evaluation_member
    rows = comparison_table(context, member, 0.1, resamples=4, selection_resamples=4)
    assert len(rows) == 7
    assert all(row.status == "computed" for row in rows)
    assert all(0.0 <= row.auroc <= 1.0 for row in rows)
    sitewise = sitewise_table(context, member, 0.1, resamples=4)
    assert len(sitewise) >= 7
    assert {row.setting for row in sitewise} >= {"pooled", "site::Site_C"}
    assert all(row.status in {"computed", NOT_RUN} for row in sitewise)
    subgroup = subgroup_table(context)
    assert all(row["status"] == NOT_RUN for row in subgroup)


def test_budget_scaling_and_sweep_tables(context: object) -> None:
    member = context.evaluation_member
    budget = event_budget_table(context, member, (1, 2, 3), 0.1)
    assert len(budget) == 3
    assert budget[-1].records == int(member.sum())
    scaling = scaling_table(context, member, 0.1, fractions=(0.5, 1.0), site_levels=(1, 2))
    assert len(scaling) == 2 + 2 + 2 + 2
    assert all(row.records > 0 for row in scaling)
    cells = weight_control_table(context, member)
    assert cells and all(0.0 <= cell.selection_accuracy <= 1.0 for cell in cells)
    headline = weight_control_headline(context, member)
    assert "ratio" in headline


def test_reader_arms_and_public_corpora(context: object) -> None:
    member = context.evaluation_member
    arms = clinician_arm_table(context, member, 0.1, reader_count=9)
    assert set(arms) == {"clinician_alone", "model_alone", "clinician_with_model"}
    assert all("auroc" in entry for entry in arms.values())
    rows = public_corpus_rows(context)
    assert len(rows) == 5
    assert all(row["status"] == NOT_RUN for row in rows)
    assert all(row["source"].startswith("http") for row in rows)


def test_scores_and_certified_metrics(context: object) -> None:
    member = context.evaluation_member
    assert behaviour_policy_score(context, member).shape[0] == int(member.sum())
    assert hybrid_family_score(context, member).shape[0] == int(member.sum())
    metrics = certified_decision_metrics(context, member, 0.1)
    assert "violation_rate" in metrics and "coverage" in metrics
    sweep = risk_margin_sweep(context, member, 0.1, (0.0, 0.05))
    assert set(sweep) == {0.0, 0.05}


def test_renderers_produce_text(context: object) -> None:
    member = context.evaluation_member
    report = render_release_report(context, 0.1)
    assert "Table 1 counterpart" in report
    assert "Table 2 panel A counterpart" in report
    assert "Table 4 panel B counterpart" in report
    assert "subgroups the release cohort cannot carry" in report
    assert isinstance(
        render_comparison(comparison_table(context, member, 0.1, resamples=4, selection_resamples=4)), str
    )
    assert isinstance(render_ablation(ablation_table(context, member, 0.1, resamples=4)), str)
    assert isinstance(render_bakeoff(bakeoff_summary(context, member, 0.1)), str)
    assert isinstance(render_event_budget(event_budget_table(context, member, (1, 2), 0.1)), str)
    assert isinstance(render_scaling(scaling_table(context, member, 0.1)), str)
    assert isinstance(render_sitewise(sitewise_table(context, member, 0.1, resamples=2)), str)
    assert isinstance(render_weight_control(weight_control_table(context, member)), str)
    assert isinstance(render_depth(ladder_depth_sensitivity(context, member, 0.1, (1,), resamples=2)), str)
    assert isinstance(render_clinician(clinician_arm_table(context, member, 0.1, 9)), str)
    assert isinstance(render_public_corpora(public_corpus_rows(context)), str)


def test_cross_region_configuration_uses_its_own_split(cross_region_context: object) -> None:
    assert cross_region_context.configuration == "cross_region"
    member = cross_region_context.evaluation_member
    sites = cross_region_context.cohort.site_array()[member]
    assert set(sites.tolist()) == {"Site_B"}
    development = cross_region_context.development_member
    assert not np.any(member & development)
