"""Plain-text rendering of the release's own computed tables.

The manuscript's printed values are never mixed in here; each block reports what the release
computed on the release cohort, with the closed-form reference beside it where one exists. The output
is plain text because the release keeps README.md as its only markdown file.

Ref: the release contract in README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wireope.studies import tables
from wireope.studies.ablation import ablation_table, ladder_depth_sensitivity
from wireope.studies.bakeoff import bakeoff_summary
from wireope.studies.pipeline import AnalysisContext
from wireope.utils.atomic import atomic_write_text


def _row(cells: tuple[object, ...], widths: tuple[int, ...]) -> str:
    parts = []
    for cell, width in zip(cells, widths):
        if isinstance(cell, float):
            text = f"{cell:.4f}" if abs(cell) < 1000 else f"{cell:.1f}"
        else:
            text = str(cell)
        parts.append(text.ljust(width))
    return "  ".join(parts).rstrip()


def render_comparison(rows: tuple[tables.ComparisonRow, ...]) -> str:
    widths = (54, 8, 8, 8, 10, 10, 8, 8, 8)
    lines = [
        "Table 1 counterpart: main comparison on the held-out external site",
        _row(("arm", "auroc", "low", "high", "nb@0.05", "nb@0.10", "select", "95% ub", "abstain"), widths),
    ]
    for row in rows:
        lines.append(
            _row(
                (
                    row.arm,
                    row.auroc,
                    row.auroc_lower,
                    row.auroc_upper,
                    row.net_benefit_005,
                    row.net_benefit_010,
                    row.selection_accuracy,
                    row.violation_upper_bound,
                    row.abstention_rate,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_ablation(rows: tuple[Any, ...]) -> str:
    widths = (52, 10, 10, 10, 10, 10, 10)
    lines = [
        "Table 2 panel A counterpart: component ablation and ladder depth",
        _row(("variant", "d select pp", "d viol pp", "ratio", "ess", "overlap", "abstain"), widths),
    ]
    for row in rows:
        lines.append(
            _row(
                (
                    row.variant,
                    row.delta_selection_pp,
                    row.delta_violation_pp,
                    row.interaction_ratio,
                    row.effective_sample_size,
                    row.overlap_coefficient,
                    row.abstention_rate,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_bakeoff(summary: dict[str, Any]) -> str:
    widths = (14, 10, 10, 10, 10, 10, 40)
    lines = [
        "Table 2 panel B counterpart: estimator bake-off on identical logs and budget",
        _row(("estimator", "rel err", "ess", "overlap", "select", "within10", "diagnostic"), widths),
    ]
    for row in summary["rows"]:
        lines.append(
            _row(
                (
                    row.key,
                    row.relative_error,
                    row.effective_sample_size,
                    row.overlap_coefficient,
                    row.selection_accuracy,
                    row.within_ten_percent,
                    row.diagnostic,
                ),
                widths,
            )
        )
    lines.append("")
    lines.append(f"best by relative error : {summary['best_by_relative_error']}")
    lines.append(f"most unstable family   : {summary['most_unstable_family']}")
    for family, value in summary["family_instability"].items():
        lines.append(f"  instability {family:22s}: {value:.4f}")
    return "\n".join(lines)


def render_event_budget(rows: tuple[tables.EventBudgetRow, ...]) -> str:
    widths = (10, 10, 10, 10, 12, 12)
    lines = [
        "Table 4 panel A counterpart: rarest-rung positive-event subsampling at fixed records",
        _row(("events", "records", "rel err", "ess", "select", "lcb width"), widths),
    ]
    for row in rows:
        lines.append(
            _row(
                (
                    row.rarest_rung_events,
                    row.records,
                    row.relative_error,
                    row.effective_sample_size,
                    row.selection_accuracy,
                    row.lower_confidence_bound_width,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_scaling(rows: tuple[tables.ScalingRow, ...]) -> str:
    widths = (18, 24, 10, 12, 10, 10, 10)
    lines = [
        "Table 6 counterpart: scaling behaviour along one axis per row",
        _row(("axis", "level", "records", "rarest", "rel err", "ess", "overlap"), widths),
    ]
    for row in rows:
        lines.append(
            _row(
                (
                    row.axis,
                    row.level,
                    row.records,
                    row.rarest_rung_events,
                    row.relative_error,
                    row.effective_sample_size,
                    row.overlap_coefficient,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_sitewise(rows: tuple[tables.SiteRow, ...]) -> str:
    widths = (34, 10, 10, 8, 8, 8, 10, 10, 10, 10)
    lines = [
        "Table 5 counterpart: per-site, per-stratum and per-subgroup replication",
        _row(
            (
                "setting",
                "attempts",
                "events",
                "auroc",
                "low",
                "high",
                "1-ece",
                "nb@0.10",
                "viol",
                "coverage",
            ),
            widths,
        ),
    ]
    for row in rows:
        lines.append(
            _row(
                (
                    row.setting,
                    row.attempts,
                    row.positive_events,
                    row.auroc,
                    row.auroc_lower,
                    row.auroc_upper,
                    row.one_minus_ece,
                    row.net_benefit_at_010,
                    row.violation_rate,
                    row.coverage,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_weight_control(cells: tuple[Any, ...]) -> str:
    widths = (24, 10, 18, 12, 18, 12, 10, 10, 10)
    lines = [
        "Table 3 counterpart: stratum by clipping-by-pessimism grid",
        _row(
            ("stratum", "axis", "clipping", "value", "pessimism", "value", "rel err", "ess", "overlap"),
            widths,
        ),
    ]
    for cell in cells:
        lines.append(
            _row(
                (
                    cell.stratum,
                    cell.axis,
                    cell.clipping_label,
                    cell.clipping,
                    cell.pessimism_label,
                    cell.pessimism,
                    cell.relative_error,
                    cell.effective_sample_size,
                    cell.overlap_coefficient,
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_depth(depth_table: dict[int, dict[str, float]]) -> str:
    widths = (10, 10, 10, 10, 12, 12)
    lines = [
        "Ladder-depth sensitivity on the evaluation share",
        _row(("depth", "select", "violation", "abstain", "ess", "resolvable"), widths),
    ]
    for depth in sorted(depth_table):
        entry = depth_table[depth]
        lines.append(
            _row(
                (
                    depth,
                    entry["selection_accuracy"],
                    entry["violation_rate"],
                    entry["abstention_rate"],
                    entry["effective_sample_size"],
                    entry["rung_resolution_depth"],
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_clinician(arms: dict[str, dict[str, float]]) -> str:
    widths = (24, 10, 10, 10, 12, 16)
    lines = [
        "Sec. 4.6 counterpart: three reader-ensemble arms",
        _row(("arm", "auroc", "low", "high", "nb@0.10", "nb in highest calc"), widths),
    ]
    for name, entry in arms.items():
        lines.append(
            _row(
                (
                    name,
                    entry["auroc"],
                    entry["auroc_lower"],
                    entry["auroc_upper"],
                    entry["net_benefit_at_010"],
                    entry["net_benefit_in_highest_calcification_stratum"],
                ),
                widths,
            )
        )
    return "\n".join(lines)


def render_public_corpora(rows: tuple[dict[str, str], ...]) -> str:
    widths = (14, 34, 12, 12)
    lines = [
        "Table 4 panel B counterpart: public logged corpora, transcription targets only",
        _row(("key", "label", "licence", "status"), widths),
    ]
    for row in rows:
        lines.append(_row((row["key"], row["label"], row["licence"], row["status"]), widths))
        lines.append(f"    source: {row['source']}")
        lines.append(f"    reason: {row['reason']}")
    return "\n".join(lines)


def render_release_report(context: AnalysisContext, clipping: float) -> str:
    """Render every block the release computed, in one plain-text document."""
    member = context.evaluation_member
    blocks = [
        f"release : {context.bundle.config.experiment.name}",
        f"configuration : {context.configuration}",
        f"evaluation share : {int(member.sum())} attempts",
        f"closed-form best strategy by graded progress : "
        f"{max(context.curves, key=lambda name: context.curves[name]['graded_progress'])}",
        "",
        render_comparison(tables.comparison_table(context, member, clipping)),
        "",
        render_ablation(ablation_table(context, member, clipping, resamples=24)),
        "",
        render_bakeoff(bakeoff_summary(context, member, clipping)),
        "",
        render_event_budget(
            tables.event_budget_table(
                context,
                member,
                tuple(int(item) for item in context.config.experiment.sampling.get("event_counts", []))
                or (5, 10, 20, 40, 80),
                clipping,
            )
        ),
        "",
        render_scaling(tables.scaling_table(context, member, clipping)),
        "",
        render_sitewise(tables.sitewise_table(context, member, clipping)),
        "",
        render_weight_control(tables.weight_control_table(context, member)),
        "",
        render_depth(ladder_depth_sensitivity(context, member, clipping, (1, 2, 3, context.ladder.depth))),
        "",
        render_clinician(
            tables.clinician_arm_table(
                context, member, clipping, int(context.config.experiment.readers.get("total", 9))
            )
        ),
        "",
        render_public_corpora(tables.public_corpus_rows(context)),
        "",
        "subgroups the release cohort cannot carry",
        *[
            f"  {row['subgroup']}: {row['status']} ({row['reason']})"
            for row in tables.subgroup_table(context)
        ],
    ]
    return "\n".join(blocks) + "\n"


def write_release_report(path: Path, context: AnalysisContext, clipping: float) -> Path:
    atomic_write_text(path, render_release_report(context, clipping))
    return path
