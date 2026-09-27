"""Transcription of the values the manuscript prints, each with its location.

The release never mixes these numbers with its own: they exist so the comparison tables can
carry a reference column, and so the manuscript's internal arithmetic can be checked against
itself. Nothing here is recomputed from the release cohort.

Ref: Tables 1-6 and the Abstract.
"""

from __future__ import annotations

from dataclasses import dataclass

ABSTRACT_AUROC_UNCONSTRAINED = 0.793
ABSTRACT_AUROC_CONSTRAINED = 0.792
ABSTRACT_NET_BENEFIT_UNCONSTRAINED = 0.0648
ABSTRACT_NET_BENEFIT_CONSTRAINED = 0.0664
ABSTRACT_VIOLATION_UNCONSTRAINED = 2.31
ABSTRACT_VIOLATION_CONSTRAINED = 0.84
ABSTRACT_ABSTENTION = 6.2
ABSTRACT_ATTEMPTS = 12000


@dataclass(frozen=True)
class PaperRow:
    label: str
    values: tuple[float, ...]
    anchor: str


TABLE_1: tuple[PaperRow, ...] = (
    PaperRow(
        "clinician_behaviour_policy",
        (0.741, 0.722, 0.760, 0.0326, 0.0512, 0.470, 2.86, 0.0),
        "Table 1 row 1",
    ),
    PaperRow(
        "anatomical_supervised_risk_score",
        (0.788, 0.775, 0.801, 0.0418, 0.0627, 0.524, 1.88, 0.0),
        "Table 1 row 2",
    ),
    PaperRow(
        "state_only_estimate", (0.779, 0.766, 0.792, 0.0391, 0.0574, 0.512, 2.04, 0.0), "Table 1 row 3"
    ),
    PaperRow(
        "unconstrained_argmax_estimator",
        (0.793, 0.780, 0.806, 0.0427, 0.0648, 0.556, 2.31, 0.0),
        "Table 1 row 4",
    ),
    PaperRow(
        "best_aggregate_hybrid_estimator_family",
        (0.781, 0.768, 0.794, 0.0402, 0.0591, 0.508, 2.47, 0.0),
        "Table 1 row 5",
    ),
    PaperRow(
        "direct_or_kernel_family_estimator",
        (0.789, 0.776, 0.802, 0.0421, 0.0633, 0.539, 2.12, 0.0),
        "Table 1 row 6",
    ),
    PaperRow(
        "support_constrained_nested_excursion_rule",
        (0.792, 0.779, 0.805, 0.0431, 0.0664, 0.552, 0.84, 6.2),
        "Table 1 row 7",
    ),
)

TABLE_2_PANEL_A: tuple[PaperRow, ...] = (
    PaperRow("full_framework", (0.0, 0.0, 1.00, 412, 0.612), "Table 2 panel A row 1"),
    PaperRow(
        "without_envelope_excursion_reconstruction", (-4.0, 1.34, 1.00, 388, 0.594), "Table 2 panel A row 2"
    ),
    PaperRow("without_nested_excursion_ladder", (-3.1, 1.02, 1.00, 371, 0.587), "Table 2 panel A row 3"),
    PaperRow("without_support_constraint", (0.4, 1.47, 1.00, 421, 0.617), "Table 2 panel A row 4"),
    PaperRow(
        "without_reconstruction_and_without_ladder", (-7.9, 2.84, 1.20, 349, 0.560), "Table 2 panel A row 5"
    ),
    PaperRow("without_ladder_and_without_support", (-4.6, 2.98, 1.20, 366, 0.571), "Table 2 panel A row 6"),
    PaperRow(
        "without_reconstruction_and_without_support",
        (-5.4, 3.42, 1.22, 379, 0.578),
        "Table 2 panel A row 7",
    ),
    PaperRow(
        "depth_sensitivity_deepest_ladder_tested", (-0.9, 0.21, 1.00, 402, 0.606), "Table 2 panel A row 8"
    ),
)

TABLE_2_PANEL_B: tuple[PaperRow, ...] = (
    PaperRow("is", (0.372, 96, 0.318, 0.431, 6.3), "Table 2 panel B row 1"),
    PaperRow("pdis", (0.301, 108, 0.351, 0.452, 11.1), "Table 2 panel B row 2"),
    PaperRow("pdis_w", (0.221, 154, 0.428, 0.489, 18.8), "Table 2 panel B row 3"),
    PaperRow("wis", (0.198, 168, 0.441, 0.497, 20.4), "Table 2 panel B row 4"),
    PaperRow("dr", (0.176, 187, 0.466, 0.508, 24.6), "Table 2 panel B row 5"),
    PaperRow("wdr", (0.162, 203, 0.481, 0.516, 26.2), "Table 2 panel B row 6"),
    PaperRow("magic", (0.191, 178, 0.458, 0.501, 22.9), "Table 2 panel B row 7"),
    PaperRow("fqe", (0.138, 246, 0.523, 0.531, 30.0), "Table 2 panel B row 8"),
    PaperRow("kernel_ih", (0.121, 289, 0.561, 0.539, 41.7), "Table 2 panel B row 9"),
)

TABLE_3: tuple[PaperRow, ...] = (
    PaperRow("anatomical_stage_I", (0.084, 930, 0.628, 0.601), "Table 3 row 1"),
    PaperRow("anatomical_stage_II", (0.129, 703, 0.542, 0.549), "Table 3 row 2"),
    PaperRow("anatomical_stage_III", (0.211, 429, 0.418, 0.482), "Table 3 row 3"),
    PaperRow("calcification_grade_0_1", (0.091, 889, 0.611, 0.588), "Table 3 row 4"),
    PaperRow("calcification_grade_2_3", (0.148, 634, 0.512, 0.537), "Table 3 row 5"),
    PaperRow("calcification_grade_4", (0.238, 371, 0.386, 0.461), "Table 3 row 6"),
    PaperRow("femoropopliteal_territory", (0.113, 778, 0.574, 0.571), "Table 3 row 7"),
    PaperRow("tibial_territory", (0.184, 488, 0.442, 0.498), "Table 3 row 8"),
    PaperRow("site_A_region_I", (0.118, 754, 0.566, 0.562), "Table 3 row 9"),
    PaperRow("site_B_region_II", (0.136, 683, 0.531, 0.545), "Table 3 row 10"),
    PaperRow("site_C_region_III", (0.152, 637, 0.509, 0.534), "Table 3 row 11"),
    PaperRow("clipping_least_aggressive", (0.243, 358, 0.379, 0.455), "Table 3 row 12"),
    PaperRow("clipping_intermediate", (0.131, 694, 0.539, 0.551), "Table 3 row 13"),
    PaperRow("clipping_most_aggressive", (0.089, 877, 0.607, 0.583), "Table 3 row 14"),
    PaperRow("pessimism_alpha_0_001", (0.097, 857, 0.598, 0.578), "Table 3 row 15"),
    PaperRow("pessimism_alpha_0_5", (0.206, 444, 0.412, 0.487), "Table 3 row 16"),
)

TABLE_4_PANEL_A: tuple[PaperRow, ...] = (
    PaperRow("events_5", (0.412, 58, 0.298, 0.186), "Table 4 panel A row 1"),
    PaperRow("events_10", (0.301, 83, 0.414, 0.141), "Table 4 panel A row 2"),
    PaperRow("events_15", (0.207, 117, 0.485, 0.104), "Table 4 panel A row 3"),
    PaperRow("events_20", (0.154, 149, 0.531, 0.081), "Table 4 panel A row 4"),
    PaperRow("events_30", (0.103, 194, 0.549, 0.054), "Table 4 panel A row 5"),
    PaperRow("events_40", (0.098, 233, 0.555, 0.047), "Table 4 panel A row 6"),
    PaperRow("events_60", (0.092, 281, 0.554, 0.042), "Table 4 panel A row 7"),
    PaperRow("events_80", (0.095, 312, 0.557, 0.041), "Table 4 panel A row 8"),
)

TABLE_4_PANEL_B: tuple[PaperRow, ...] = (
    PaperRow("open_bandit", (0.041, 1842, 0.742), "Table 4 panel B row 1"),
    PaperRow("neorl2", (0.096, 512, 0.628), "Table 4 panel B row 2"),
    PaperRow("cobs", (0.143, 341, 0.518), "Table 4 panel B row 3"),
    PaperRow("deep_ope", (0.131, 389, 0.539), "Table 4 panel B row 4"),
    PaperRow("d4rl", (0.058, 1204, 0.701), "Table 4 panel B row 5"),
)

TABLE_5: tuple[PaperRow, ...] = (
    PaperRow("site_A", (4842, 232, 0.795, 0.779, 0.811, 0.962, 0.0671, 0.79, 94.9), "Table 5 row 1"),
    PaperRow("site_B", (3986, 190, 0.788, 0.771, 0.805, 0.957, 0.0652, 0.91, 94.6), "Table 5 row 2"),
    PaperRow(
        "site_C_held_out", (3172, 142, 0.792, 0.779, 0.805, 0.969, 0.0664, 0.84, 94.7), "Table 5 row 3"
    ),
    PaperRow(
        "cross_region_site_B_held_out",
        (3986, 190, 0.791, 0.780, 0.802, 0.966, 0.0659, 0.86, 94.8),
        "Table 5 row 4",
    ),
    PaperRow("pooled", (12000, 564, 0.793, 0.784, 0.802, 0.964, 0.0662, 0.85, 94.7), "Table 5 row 5"),
    PaperRow(
        "stratum_stage_I", (1472, 41, 0.804, 0.783, 0.825, 0.971, 0.0703, 0.62, 95.2), "Table 5 row 6"
    ),
    PaperRow(
        "stratum_stage_II", (2266, 98, 0.796, 0.780, 0.812, 0.965, 0.0678, 0.78, 94.9), "Table 5 row 7"
    ),
    PaperRow(
        "stratum_stage_III", (1691, 109, 0.774, 0.754, 0.794, 0.952, 0.0612, 1.14, 94.1), "Table 5 row 8"
    ),
    PaperRow(
        "stratum_calcification_4",
        (1238, 86, 0.771, 0.749, 0.793, 0.948, 0.0598, 1.21, 93.8),
        "Table 5 row 9",
    ),
    PaperRow(
        "stratum_tibial", (3994, 208, 0.783, 0.764, 0.802, 0.958, 0.0641, 0.98, 94.4), "Table 5 row 10"
    ),
    PaperRow(
        "stratum_femoropopliteal",
        (8006, 341, 0.797, 0.783, 0.811, 0.967, 0.0675, 0.77, 94.9),
        "Table 5 row 11",
    ),
    PaperRow(
        "subgroup_female", (2187, 103, 0.788, 0.771, 0.805, 0.961, 0.0657, 0.88, 94.6), "Table 5 row 12"
    ),
    PaperRow(
        "subgroup_diabetes", (3014, 141, 0.789, 0.774, 0.804, 0.959, 0.0654, 0.90, 94.5), "Table 5 row 13"
    ),
    PaperRow(
        "subgroup_ckd_or_dialysis",
        (1109, 58, 0.776, 0.752, 0.800, 0.951, 0.0621, 1.08, 94.2),
        "Table 5 row 14",
    ),
)

TABLE_6: tuple[PaperRow, ...] = (
    PaperRow("small_cohort_fraction", (3000, 141, 0.148, 473, 0.502), "Table 6 row 1"),
    PaperRow("intermediate_cohort_fraction", (6000, 282, 0.119, 659, 0.548), "Table 6 row 2"),
    PaperRow("full_cohort", (12000, 564, 0.084, 911, 0.612), "Table 6 row 3"),
    PaperRow("single_site", (4842, 232, 0.136, 599, 0.531), "Table 6 row 4"),
    PaperRow("two_sites", (8828, 422, 0.104, 771, 0.578), "Table 6 row 5"),
    PaperRow("three_sites", (12000, 564, 0.084, 911, 0.612), "Table 6 row 6"),
    PaperRow("single_territory_femoropopliteal", (8006, 341, 0.113, 778, 0.574), "Table 6 row 7"),
    PaperRow("single_territory_tibial", (3994, 208, 0.161, 504, 0.496), "Table 6 row 8"),
    PaperRow("single_vendor_class", (7090, 325, 0.121, 703, 0.545), "Table 6 row 9"),
    PaperRow("multiple_vendor_classes", (12000, 564, 0.084, 911, 0.612), "Table 6 row 10"),
)

TABLES: dict[str, tuple[PaperRow, ...]] = {
    "table_1_main_comparison": TABLE_1,
    "table_2_panel_a_ablation": TABLE_2_PANEL_A,
    "table_2_panel_b_bakeoff": TABLE_2_PANEL_B,
    "table_3_stratified_weight_control": TABLE_3,
    "table_4_panel_a_event_budget": TABLE_4_PANEL_A,
    "table_4_panel_b_public_corpora": TABLE_4_PANEL_B,
    "table_5_sitewise": TABLE_5,
    "table_6_scaling": TABLE_6,
}

PUBLISHED_WITHIN_TEN_PERCENT: dict[str, float] = {
    "is": 6.3,
    "pdis": 11.1,
    "pdis_w": 18.8,
    "wis": 20.4,
    "dr": 24.6,
    "wdr": 26.2,
    "magic": 22.9,
    "fqe": 30.0,
    "kernel_ih": 41.7,
}


def table_rows(name: str) -> tuple[PaperRow, ...]:
    if name not in TABLES:
        raise KeyError(f"unknown transcribed table '{name}'")
    return TABLES[name]


def table_summary() -> dict[str, int]:
    return {name: len(rows) for name, rows in TABLES.items()}


def row_by_label(name: str, label: str) -> PaperRow:
    for row in table_rows(name):
        if row.label == label:
            return row
    raise KeyError(f"table '{name}' has no row labelled '{label}'")
