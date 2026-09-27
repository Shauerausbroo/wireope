"""Checks that recompute the manuscript's own tabulated arithmetic.

These checks read the transcribed values and compare them with each other, never with the
release's computed numbers, so a disagreement is a property of the manuscript. A failure here
is a finding about the paper, not a defect in the release, and is reported under its own
family with its own status.

Ref: Tables 1-6, the Abstract and Eq. (2)-(7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wireope.report import transcribe
from wireope.schema import CheckStatus

MARGIN = 1e-6


@dataclass(frozen=True)
class ManuscriptCheck:
    name: str
    status: CheckStatus
    detail: str
    anchor: str
    evidence: dict[str, float | str]


def _status(ok: bool) -> CheckStatus:
    return CheckStatus.PASS if ok else CheckStatus.FAIL


def site_partition_sums() -> ManuscriptCheck:
    rows = transcribe.table_rows("table_5_sitewise")
    sites = [row for row in rows if row.label.startswith("site_") and row.label != "site_C_held_out"]
    held_out = transcribe.row_by_label("table_5_sitewise", "site_C_held_out")
    attempts = sum(int(row.values[0]) for row in (*sites, held_out))
    events = sum(int(row.values[1]) for row in (*sites, held_out))
    pooled = transcribe.row_by_label("table_5_sitewise", "pooled")
    ok = attempts == int(pooled.values[0]) and events == int(pooled.values[1])
    return ManuscriptCheck(
        name="manuscript.site_partition_sums",
        status=_status(ok),
        detail=f"three sites sum to {attempts} attempts and {events} events against a pooled "
        f"{int(pooled.values[0])} and {int(pooled.values[1])}",
        anchor="Table 5 rows 1-5",
        evidence={
            "attempts_sum": attempts,
            "events_sum": events,
            "pooled_attempts": int(pooled.values[0]),
            "pooled_events": int(pooled.values[1]),
        },
    )


def territory_partition_events() -> ManuscriptCheck:
    tibial = transcribe.row_by_label("table_5_sitewise", "stratum_tibial")
    femoropopliteal = transcribe.row_by_label("table_5_sitewise", "stratum_femoropopliteal")
    pooled = transcribe.row_by_label("table_5_sitewise", "pooled")
    events = int(tibial.values[1]) + int(femoropopliteal.values[1])
    gap = int(pooled.values[1]) - events
    return ManuscriptCheck(
        name="manuscript.territory_partition_events",
        status=_status(gap == 0),
        detail=f"the two territory rows carry {events} events while the pooled row reports "
        f"{int(pooled.values[1])}, a gap of {gap}",
        anchor="Table 5 rows 5, 10, 11",
        evidence={"territory_events": events, "pooled_events": int(pooled.values[1]), "gap": gap},
    )


def ablation_interaction_violation() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_2_panel_a_ablation")}
    pairs = (
        (
            "without_reconstruction_and_without_ladder",
            "without_envelope_excursion_reconstruction",
            "without_nested_excursion_ladder",
        ),
        (
            "without_ladder_and_without_support",
            "without_nested_excursion_ladder",
            "without_support_constraint",
        ),
        (
            "without_reconstruction_and_without_support",
            "without_envelope_excursion_reconstruction",
            "without_support_constraint",
        ),
    )
    recomputed = [
        rows[joint].values[1] / (rows[first].values[1] + rows[second].values[1])
        for joint, first, second in pairs
    ]
    printed = [rows[joint].values[2] for joint, _, _ in pairs]
    ok = all(abs(round(value, 2) - expected) < 1e-9 for value, expected in zip(recomputed, printed))
    return ManuscriptCheck(
        name="manuscript.ablation_interaction_violation",
        status=_status(ok),
        detail=f"the violation-rate deltas recompute to {[round(v, 3) for v in recomputed]} against "
        f"the printed {printed}",
        anchor="Table 2 panel A rows 5-7",
        evidence={"recomputed": str([round(v, 4) for v in recomputed]), "printed": str(printed)},
    )


def ablation_interaction_selection() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_2_panel_a_ablation")}
    pairs = (
        (
            "without_reconstruction_and_without_ladder",
            "without_envelope_excursion_reconstruction",
            "without_nested_excursion_ladder",
        ),
        (
            "without_ladder_and_without_support",
            "without_nested_excursion_ladder",
            "without_support_constraint",
        ),
        (
            "without_reconstruction_and_without_support",
            "without_envelope_excursion_reconstruction",
            "without_support_constraint",
        ),
    )
    ratios = [
        abs(rows[joint].values[0]) / (abs(rows[first].values[0]) + abs(rows[second].values[0]))
        for joint, first, second in pairs
    ]
    printed = [rows[joint].values[2] for joint, _, _ in pairs]
    ok = all(abs(round(value, 2) - expected) < 1e-9 for value, expected in zip(ratios, printed))
    return ManuscriptCheck(
        name="manuscript.ablation_interaction_selection",
        status=_status(ok),
        detail="the caption states the interaction ratio is computed on the violation rate and on "
        f"selection accuracy; on selection accuracy the three pairs give "
        f"{[round(v, 2) for v in ratios]} against the printed {printed}",
        anchor="Table 2 panel A caption and column 4",
        evidence={"selection_ratios": str([round(v, 3) for v in ratios]), "printed": str(printed)},
    )


def state_only_violation_row() -> ManuscriptCheck:
    table_one = transcribe.row_by_label("table_1_main_comparison", "state_only_estimate")
    constrained = transcribe.row_by_label(
        "table_1_main_comparison", "support_constrained_nested_excursion_rule"
    )
    rows = {row.label: row for row in transcribe.table_rows("table_2_panel_a_ablation")}
    full = constrained.values[6]
    delta = rows["without_envelope_excursion_reconstruction"].values[1]
    implied = full + delta
    return ManuscriptCheck(
        name="manuscript.state_only_violation_row",
        status=_status(abs(implied - table_one.values[6]) < 0.005),
        detail=f"Table 1 prints {table_one.values[6]} for the state-only violation rate while the "
        f"full framework plus the Table 2 removal delta gives {implied:.2f}",
        anchor="Table 1 row 3 and Table 2 panel A row 2",
        evidence={
            "table_1_state_only": table_one.values[6],
            "implied": implied,
            "full_framework": full,
            "delta": delta,
        },
    )


def constraint_direction() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.constraint_direction",
        status=CheckStatus.FAIL,
        detail="Eq. (2) constrains a lower confidence limit of the risk estimator at or below a "
        "margin while Eq. (5) constrains an upper limit at or below a threshold on the same "
        "functional; the two printed forms disagree in direction",
        anchor="Eq. (2) and Eq. (5)",
        evidence={"eq2": "L(pi; delta) <= m", "eq5": "UCB_alpha(R_hat(pi)) <= kappa"},
    )


def utility_direction() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.utility_direction",
        status=CheckStatus.FAIL,
        detail="Eq. (2) maximises decision-analytic net benefit while Eq. (3) defines the "
        "functional that Eq. (5) maximises as a graded excursion value whose sign runs the other "
        "way, so the two printed objectives cannot both be the utility",
        anchor="Eq. (2), Eq. (3) and Eq. (5)",
        evidence={"eq2": "argmax NB(pi; pt)", "eq3": "V(pi) = E[sum_k w_k Y^(k)(pi)]"},
    )


def ladder_nesting_direction() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.ladder_nesting_direction",
        status=CheckStatus.FAIL,
        detail="Sec. 3.1 states the ladder implication as Y^(k) = 1 implying Y^(k+1) = 1, which "
        "makes the indicator non-decreasing in depth, while Sec. 3.4 nests the rungs as "
        "A_1 superset ... superset A_K with increasing thresholds, which makes it non-increasing",
        anchor="Sec. 3.1 formal statement and Sec. 3.4 opening",
        evidence={"sec_3_1": "Y^(k) = 1 => Y^(k+1) = 1", "sec_3_4": "A_1 superset A_2 ... superset A_K"},
    )


def abstention_coverage_complement() -> ManuscriptCheck:
    table_one = transcribe.row_by_label(
        "table_1_main_comparison", "support_constrained_nested_excursion_rule"
    )
    held_out = transcribe.row_by_label("table_5_sitewise", "site_C_held_out")
    pooled = transcribe.row_by_label("table_5_sitewise", "pooled")
    implied_held_out = 100.0 - held_out.values[8]
    implied_pooled = 100.0 - pooled.values[8]
    gap = abs(table_one.values[7] - implied_held_out)
    return ManuscriptCheck(
        name="manuscript.abstention_coverage_complement",
        status=_status(gap <= 0.2),
        detail=f"Table 1 reports {table_one.values[7]}% abstention while Table 5's coverage implies "
        f"{implied_held_out:.1f}% on the held-out site and {implied_pooled:.1f}% pooled",
        anchor="Table 1 row 7, column 8 and Table 5 column 7",
        evidence={
            "table_1_abstention": table_one.values[7],
            "implied_held_out": implied_held_out,
            "implied_pooled": implied_pooled,
        },
    )


def abstract_matches_table_one() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_1_main_comparison")}
    constrained = rows["support_constrained_nested_excursion_rule"].values
    unconstrained = rows["unconstrained_argmax_estimator"].values
    ok = (
        abs(constrained[0] - transcribe.ABSTRACT_AUROC_CONSTRAINED) < MARGIN
        and abs(unconstrained[0] - transcribe.ABSTRACT_AUROC_UNCONSTRAINED) < MARGIN
        and abs(constrained[4] - transcribe.ABSTRACT_NET_BENEFIT_CONSTRAINED) < MARGIN
        and abs(unconstrained[4] - transcribe.ABSTRACT_NET_BENEFIT_UNCONSTRAINED) < MARGIN
        and abs(constrained[6] - transcribe.ABSTRACT_VIOLATION_CONSTRAINED) < MARGIN
        and abs(unconstrained[6] - transcribe.ABSTRACT_VIOLATION_UNCONSTRAINED) < MARGIN
        and abs(constrained[7] - transcribe.ABSTRACT_ABSTENTION) < MARGIN
    )
    return ManuscriptCheck(
        name="manuscript.abstract_matches_table_one",
        status=_status(ok),
        detail="every abstract headline value is carried by the corresponding Table 1 row",
        anchor="Abstract and Table 1 rows 4 and 7",
        evidence={"constrained_auroc": constrained[0], "unconstrained_auroc": unconstrained[0]},
    )


def selection_accuracy_parity_clause() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_1_main_comparison")}
    gap_pp = 100.0 * (
        rows["unconstrained_argmax_estimator"].values[5]
        - rows["support_constrained_nested_excursion_rule"].values[5]
    )
    return ManuscriptCheck(
        name="manuscript.selection_accuracy_parity_clause",
        status=_status(abs(gap_pp) <= 1.0),
        detail=f"the parity clause allows one percentage point and the printed gap is "
        f"{abs(gap_pp):.1f} points",
        anchor="Table 1 caption and column 6",
        evidence={"gap_pp": abs(gap_pp)},
    )


def event_budget_knee() -> ManuscriptCheck:
    rows = transcribe.table_rows("table_4_panel_a_event_budget")
    errors = [row.values[0] for row in rows]
    accuracy = [row.values[2] for row in rows]
    thin = max(errors[4] - errors[5], errors[5] - errors[6], abs(errors[7] - errors[6]))
    thick = errors[0] - errors[3]
    return ManuscriptCheck(
        name="manuscript.event_budget_knee",
        status=_status(thick > 0.2 and thin < 0.02),
        detail=f"the relative error falls by {thick:.3f} over the first four steps and by no more "
        f"than {thin:.3f} past the sixth, placing the knee in the low tens",
        anchor="Table 4 panel A column 2",
        evidence={"early_drop": thick, "late_drop": thin, "accuracy_at_60": accuracy[6]},
    )


def published_within_ten_percent() -> ManuscriptCheck:
    quoted = transcribe.PUBLISHED_WITHIN_TEN_PERCENT
    rows = {row.label: row for row in transcribe.table_rows("table_2_panel_b_bakeoff")}
    ok = all(abs(rows[key].values[4] - value) < MARGIN for key, value in quoted.items())
    return ManuscriptCheck(
        name="manuscript.published_within_ten_percent",
        status=_status(ok),
        detail="the comparability column of the bake-off reproduces the published aggregate "
        "near-top-frequency values quoted in the Introduction",
        anchor="Table 2 panel B column 6, Sec. 1 and Sec. 2.1",
        evidence={"is": rows["is"].values[4], "kernel_ih": rows["kernel_ih"].values[4]},
    )


def rung_resolution_at_stated_tolerance() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.rung_resolution_at_stated_tolerance",
        status=CheckStatus.NOT_RUN,
        detail="the resolution criterion needs the propagating band and the rung spacing; the "
        "manuscript states the 4.44 mm tip tolerance but neither the segmentation tolerance nor "
        "the spacing, so the criterion cannot be evaluated on its own numbers",
        anchor="Sec. 3.3 last paragraph and Sec. 3.4 opening",
        evidence={
            "tip_tolerance_mm": 4.44,
            "segmentation_tolerance": "unreported",
            "rung_spacing": "unreported",
        },
    )


def unfilled_placeholder() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.unfilled_placeholder",
        status=CheckStatus.FAIL,
        detail="the Introduction carries the unfilled placeholder '[insert number]%' in the first "
        "sentence of the background",
        anchor="Sec. 1 first paragraph",
        evidence={"text": "[insert number]% 40 of patients"},
    )


def prevalence_axis() -> ManuscriptCheck:
    pooled = transcribe.row_by_label("table_5_sitewise", "pooled")
    prevalence = 100.0 * pooled.values[1] / pooled.values[0]
    return ManuscriptCheck(
        name="manuscript.prevalence_axis",
        status=_status(abs(prevalence - 3.8) < 0.2),
        detail=f"the precision-recall axis is reported against a 3.8% prevalence while the cohort's "
        f"own rate is {prevalence:.2f}%",
        anchor="Sec. 3.7 mid paragraph and Table 5 row 5",
        evidence={"cohort_prevalence_pct": prevalence, "axis_prevalence_pct": 3.8},
    )


def intro_perforation_fraction() -> ManuscriptCheck:
    fraction = 100.0 * 367.0 / 9618.0
    return ManuscriptCheck(
        name="manuscript.intro_perforation_fraction",
        status=_status(abs(fraction - 3.8) < 0.05),
        detail=f"367 of 9,618 is {fraction:.2f}%, consistent with the 3.8% quoted beside it",
        anchor="Sec. 1 first paragraph",
        evidence={"fraction_pct": fraction},
    )


def table_numbering() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.table_numbering",
        status=CheckStatus.FAIL,
        detail="Sec. 4.6 refers to the per-site table as 'Table 9.01' while the table carrying "
        "those numbers is Table 5",
        anchor="Sec. 4.6 first sentence and Table 5",
        evidence={"reference": "Table 9.01", "actual": "Table 5"},
    )


def weight_control_monotonicity() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_3_stratified_weight_control")}
    errors = [
        rows["clipping_least_aggressive"].values[0],
        rows["clipping_intermediate"].values[0],
        rows["clipping_most_aggressive"].values[0],
    ]
    sizes = [
        rows["clipping_least_aggressive"].values[1],
        rows["clipping_intermediate"].values[1],
        rows["clipping_most_aggressive"].values[1],
    ]
    pessimistic_error = rows["pessimism_alpha_0_5"].values[0]
    optimistic_error = rows["pessimism_alpha_0_001"].values[0]
    ok = (
        all(later <= earlier for earlier, later in zip(errors, errors[1:]))
        and all(later >= earlier for earlier, later in zip(sizes, sizes[1:]))
        and pessimistic_error > optimistic_error
    )
    return ManuscriptCheck(
        name="manuscript.weight_control_monotonicity",
        status=_status(ok),
        detail="more aggressive clipping lowers the relative error and raises the effective sample "
        "size monotonically, and the pessimistic end is worse than the optimistic end",
        anchor="Table 3 rows 12-16",
        evidence={
            "clipping_errors": str(errors),
            "clipping_sizes": str(sizes),
            "pessimistic_error": pessimistic_error,
            "optimistic_error": optimistic_error,
        },
    )


def table_six_sums() -> ManuscriptCheck:
    rows = {row.label: row for row in transcribe.table_rows("table_6_scaling")}
    two_sites = rows["two_sites"].values
    three_sites = rows["three_sites"].values
    single = rows["single_site"].values
    ok = (
        int(single[0]) + 3986 == int(two_sites[0])
        and int(two_sites[0]) + 3172 == int(three_sites[0])
        and int(single[1]) + 190 == int(two_sites[1])
        and int(two_sites[1]) + 142 == int(three_sites[1])
    )
    return ManuscriptCheck(
        name="manuscript.table_six_sums",
        status=_status(ok),
        detail="the site-count rows of the scaling table are the cumulative site partitions",
        anchor="Table 6 rows 4-6 and Table 5 rows 1-3",
        evidence={"single": single[0], "two": two_sites[0], "three": three_sites[0]},
    )


def table_six_prevalence() -> ManuscriptCheck:
    rows = transcribe.table_rows("table_6_scaling")[:3]
    prevalences = [100.0 * row.values[1] / row.values[0] for row in rows]
    spread = max(prevalences) - min(prevalences)
    return ManuscriptCheck(
        name="manuscript.table_six_prevalence",
        status=_status(spread < 0.05),
        detail=f"the cohort-fraction rows hold the prevalence at {prevalences[0]:.2f}% while both "
        f"the records and the event count scale",
        anchor="Table 6 rows 1-3",
        evidence={"prevalences_pct": str([round(item, 3) for item in prevalences])},
    )


def full_cohort_effective_sample_size() -> ManuscriptCheck:
    panel = {row.label: row for row in transcribe.table_rows("table_2_panel_a_ablation")}
    scaling = {row.label: row for row in transcribe.table_rows("table_6_scaling")}
    panel_size = panel["full_framework"].values[3]
    scaling_size = scaling["full_cohort"].values[3]
    panel_overlap = panel["full_framework"].values[4]
    scaling_overlap = scaling["full_cohort"].values[4]
    agree_on_overlap = abs(panel_overlap - scaling_overlap) < MARGIN
    agree_on_size = abs(panel_size - scaling_size) < 1.0
    return ManuscriptCheck(
        name="manuscript.full_cohort_effective_sample_size",
        status=_status(agree_on_size or not agree_on_overlap),
        detail=f"the full framework carries effective size {panel_size:g} in Table 2 and "
        f"{scaling_size:g} in Table 6 while both rows print the same overlap coefficient "
        f"{panel_overlap:g}, so the two entries cannot be the same quantity under one definition",
        anchor="Table 2 panel A row 1 and Table 6 row 3",
        evidence={"table_2_size": panel_size, "table_6_size": scaling_size, "overlap": panel_overlap},
    )


def equation_seven_parameters() -> ManuscriptCheck:
    return ManuscriptCheck(
        name="manuscript.equation_seven_parameters",
        status=CheckStatus.NOT_RUN,
        detail="the lower-confidence-bound width column of the event-budget table cannot be "
        "recomputed because the bounded range B, the depth K and the level delta are not printed "
        "for that table",
        anchor="Eq. (7) and Table 4 panel A column 5",
        evidence={"bounded_range": "unreported", "depth": "unreported", "delta": "unreported"},
    )


CHECKS: tuple[Callable[[], ManuscriptCheck], ...] = (
    site_partition_sums,
    territory_partition_events,
    ablation_interaction_violation,
    ablation_interaction_selection,
    state_only_violation_row,
    constraint_direction,
    utility_direction,
    ladder_nesting_direction,
    abstention_coverage_complement,
    abstract_matches_table_one,
    selection_accuracy_parity_clause,
    event_budget_knee,
    published_within_ten_percent,
    rung_resolution_at_stated_tolerance,
    unfilled_placeholder,
    prevalence_axis,
    intro_perforation_fraction,
    table_numbering,
    weight_control_monotonicity,
    table_six_sums,
    table_six_prevalence,
    full_cohort_effective_sample_size,
    equation_seven_parameters,
)


def run_manuscript_checks() -> tuple[ManuscriptCheck, ...]:
    return tuple(check() for check in CHECKS)


def summary(checks: tuple[ManuscriptCheck, ...]) -> dict[str, float]:
    statuses = np.asarray([check.status.value for check in checks], dtype="<U8")
    return {
        "total": float(checks.__len__()),
        "passed": float(np.count_nonzero(statuses == CheckStatus.PASS.value)),
        "failed": float(np.count_nonzero(statuses == CheckStatus.FAIL.value)),
        "not_run": float(np.count_nonzero(statuses == CheckStatus.NOT_RUN.value)),
    }
