"""Intentional departures from the manuscript, each with its location and its reason.

The manuscript fixes several quantities without printing them and prints two constraints that
disagree with one another. Everything below is either an engineering default for a quantity the
paper leaves open or a decision taken after a printed conflict, and each entry names the paper
location it departs from.

Ref: the release contract in README.
"""

from __future__ import annotations

from typing import Any

DEVIATIONS: list[dict[str, Any]] = [
    {
        "paper_location": "Sec. 3.1, Eq. (2) and Eq. (5)",
        "departure": "the risk constraint is implemented as a lower confidence limit compared "
        "against a margin, following Eq. (2), rather than as an upper limit against a threshold as "
        "Eq. (5) prints",
        "why": "the two printed forms disagree in direction; following Eq. (2) keeps a single "
        "definition of the constraint, and the conflict is reported as a manuscript finding",
    },
    {
        "paper_location": "Sec. 3.1, Eq. (2) and Eq. (3)",
        "departure": "the utility the rule maximises is the graded crossing progress on the ladder, "
        "the complement of the graded excursion severity of Eq. (3)",
        "why": "Eq. (2) maximises net benefit while Eq. (3) writes a graded excursion value whose sign "
        "runs the other way, so taking Eq. (3) literally would have the rule maximise severity; the "
        "estimator of Eq. (4) still estimates the rung marginals exactly as printed",
    },
    {
        "paper_location": "Sec. 3.4 opening",
        "departure": "the rung indicators nest downward, Y^(k+1) = 1 implying Y^(k) = 1, with the top "
        "rung equal to the adjudicated endpoint",
        "why": "Sec. 3.4 nests the rungs by increasing thresholds, which forces a downward nesting, "
        "while Sec. 3.1 prints the implication upward; the product form of Eq. (6) requires the "
        "downward reading",
    },
    {
        "paper_location": "Sec. 3.2 footnote 5 and encoder paragraph",
        "departure": "the frozen encoder is a compact three-dimensional convolutional trunk with a "
        "trainable envelope adapter rather than the cited foundation model itself",
        "why": "the cited weights are distributed outside this repository; the encoder interface, its "
        "frozen trunk and its two heads are kept so the adapter can be replaced by the cited model "
        "without touching the pipeline",
    },
    {
        "paper_location": "Sec. 3.3 tip tolerance",
        "departure": "the envelope segmentation tolerance is set to 0.10 mm",
        "why": "the manuscript states the 4.44 mm tip-tracking tolerance but not the segmentation "
        "tolerance; the band is therefore an engineering default exposed in "
        "configs/traces/uncertainty.yaml",
    },
    {
        "paper_location": "Sec. 3.4 opening",
        "departure": "the ladder depth defaults to four rungs and the deepest tested depth is six",
        "why": "the manuscript reports a depth-sensitivity row at K = 6 but does not print the default "
        "depth, the threshold grid or the weight scheme",
    },
    {
        "paper_location": "Sec. 3.4 opening",
        "departure": "the ladder weights use a geometric scheme with ratio 0.6, normalised to one",
        "why": "the manuscript states the weights are fixed and non-negative without printing them",
    },
    {
        "paper_location": "Sec. 3.5 opening",
        "departure": "the per-stratum overlap floor defaults to 0.02, about one sixth of the uniform "
        "behaviour share over the eight-strategy library",
        "why": "the manuscript fixes the floor in the analysis plan and does not print its value",
    },
    {
        "paper_location": "Sec. 3.1 decision rule and Sec. 3.4 bounds",
        "departure": "the analysis-plan risk margin defaults to 0.01, the level delta to 0.05 and the "
        "bounded rung range to 1.0",
        "why": "none of the three is printed, and the deviation bound of Eq. (7) cannot be evaluated "
        "without them",
    },
    {
        "paper_location": "Sec. 4.3 and Table 3",
        "departure": "the clipping grid is 0.02, 0.10 and 0.50 with the most aggressive end as the "
        "headline, and the pessimism grid is 0.001 and 0.5 with the pessimistic end as the headline",
        "why": "the manuscript states the sweep exists and prints its endpoints' labels, not the "
        "clipping values",
    },
    {
        "paper_location": "Sec. 3.7 final paragraph",
        "departure": "the reported fits use the optimiser, schedule and seeds in configs/fit with "
        "batch 256, 40 epochs and a cosine schedule",
        "why": "the manuscript reports no training hyperparameters and only states that the seeds are "
        "hard-wired and that a single processor graphics node is used",
    },
    {
        "paper_location": "Sec. 3.6 first paragraph and Data availability",
        "departure": "cohort-level values are produced on the cohort the release assembles for itself "
        "rather than on "
        "the private cohort",
        "why": "the record-level source data are not shared; every cohort table value is therefore "
        "absent and the assembled cohort carries the same data dictionary",
    },
    {
        "paper_location": "Sec. 3.3",
        "departure": "the excursion sequence over the advancement steps is reconstructed as a monotone "
        "rise to the recorded peak",
        "why": "the record carries the peak and the event stream rather than a per-step excursion, so "
        "the rise is an engineering default used only by the hazard head",
    },
    {
        "paper_location": "Sec. 4.6",
        "departure": "the three-arm reading study is executed over a reader ensemble with "
        "the structure the manuscript describes",
        "why": "the cohort's reader outcomes are not shared, so no reader-level result can be reported",
    },
    {
        "paper_location": "Sec. 4.4 panel B caption",
        "departure": "the public logged corpora are listed with their licences and sources but not "
        "downloaded, so their rows carry no value",
        "why": "the row values depend on corpora the release does not fetch, and no honest substitute "
        "for a downloaded benchmark exists",
    },
    {
        "paper_location": "Sec. 4.4 panel A and column 5",
        "departure": "the rarest-rung event budget is exercised by withdrawing top-rung indicators "
        "while keeping every record in the log",
        "why": "the manuscript describes holding the record count fixed while the rarest-rung count "
        "moves, which is what withdrawing the indicator reproduces",
    },
]


def as_payload() -> list[dict[str, Any]]:
    return [dict(entry) for entry in DEVIATIONS]


def count() -> int:
    return len(DEVIATIONS)


def locations() -> list[str]:
    return [str(entry["paper_location"]) for entry in DEVIATIONS]
