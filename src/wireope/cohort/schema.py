"""Cohort data dictionary, join keys and provenance fields.

Every site contributes through the same dictionary; the CTA that precedes the procedure is
attached to the patient record and to the outcome through patient-level identifiers present
only inside the site. The join key never crosses a site boundary, and no re-linking key,
coding scheme, time-shift offset or translation mechanism survives anonymisation.

Ref: Sec. 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import ClinicalCohortConfig

DATA_DICTIONARY: dict[str, str] = {
    "patient_id": "site-local pseudonymous patient key; destroyed after linkage",
    "procedure_id": "site-local procedure key",
    "lesion_id": "site-local lesion key nested inside the procedure",
    "attempt_id": "crossing attempt key nested inside the lesion",
    "site": "participating site pseudonym",
    "region": "region pseudonym",
    "operator": "operator stratum key, site-local",
    "vendor_class": "device vendor class, recorded as a class and never as a serial",
    "anatomy": "anatomical identifiers: stage, calcification grade and arc, cap morphology, "
    "lesion length, vessel territory",
    "device": "device class, attempt index and device-to-lesion relation",
    "events": "time-ordered device-advancement and rotation event stream",
    "strategy": "delivered strategy class as documented in the procedure record",
    "injury": "adjudicated vessel-injury endpoint for the attempt",
}


@dataclass(frozen=True)
class CohortProvenance:
    patient_selection: str
    outcome_ascertainment: str
    clinical_linkage: str
    treatment_exposure: str
    federation: str
    leakage_safeguards: str
    accrual_window: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "patient_selection": self.patient_selection,
            "outcome_ascertainment": self.outcome_ascertainment,
            "clinical_linkage": self.clinical_linkage,
            "treatment_exposure": self.treatment_exposure,
            "federation": self.federation,
            "leakage_safeguards": self.leakage_safeguards,
            "accrual_window": self.accrual_window,
        }


PROVENANCE = CohortProvenance(
    patient_selection=(
        "every eligible anonymized crossing record processed by the pre-specified "
        "inclusion and exclusion criteria before any modelling, with no post hoc removal "
        "of high-risk lesions"
    ),
    outcome_ascertainment=(
        "the vessel-injury endpoint adjudicated from procedure records and angiographic "
        "images by at least two reviewers blinded to the reconstructed trajectory"
    ),
    clinical_linkage=(
        "the pre-procedural CTA attached to the patient record and to the outcome through "
        "patient-level identifiers in the same dictionary at every site; the join key is "
        "present only inside the site"
    ),
    treatment_exposure=(
        "the delivered strategy class taken from the procedure record while the time-ordered "
        "action channel is reconstructed from the event stream, so exposure and action carry "
        "separate provenance"
    ),
    federation=(
        "each site serves a distinct patient and operator pool under a common dictionary and "
        "is the custodian of its own records only"
    ),
    leakage_safeguards=(
        "patient-disjoint partitions, whole-site and whole-region hold-outs rather than random "
        "subsets, and a disjoint calibration share used for weights and thresholds"
    ),
    accrual_window="consecutive accrual inside each site's window",
)


@dataclass(frozen=True)
class SiteTable:
    site: str
    region: str
    attempts: int
    injury_events: int
    operators: int
    vendor_classes: int

    @property
    def prevalence(self) -> float:
        if self.attempts == 0:
            return 0.0
        return float(self.injury_events / self.attempts)


def site_table(
    sites: NDArray[np.str_],
    regions: NDArray[np.str_],
    injury: NDArray[np.int64],
    operators: NDArray[np.str_],
    vendor_classes: NDArray[np.str_],
) -> tuple[SiteTable, ...]:
    tables: list[SiteTable] = []
    for name in sorted(set(sites.tolist())):
        member = sites == name
        tables.append(
            SiteTable(
                site=str(name),
                region=str(regions[member][0]),
                attempts=int(np.count_nonzero(member)),
                injury_events=int(np.sum(injury[member])),
                operators=int(len(set(operators[member].tolist()))),
                vendor_classes=int(len(set(vendor_classes[member].tolist()))),
            )
        )
    return tuple(tables)


def cohort_totals(tables: tuple[SiteTable, ...]) -> dict[str, float]:
    attempts = sum(table.attempts for table in tables)
    events = sum(table.injury_events for table in tables)
    return {
        "attempts": float(attempts),
        "injury_events": float(events),
        "prevalence": float(events / attempts) if attempts else 0.0,
    }


def expected_prevalence(config: ClinicalCohortConfig) -> float:
    if config.attempts == 0:
        return 0.0
    return float(config.injury_events / config.attempts)


def literature_gap(config: ClinicalCohortConfig) -> float:
    """Gap between the cohort's own prevalence and the literature value used for the PR axis."""
    return float(expected_prevalence(config) - config.literature_perforation_prevalence)


def dictionary_keys() -> tuple[str, ...]:
    return tuple(DATA_DICTIONARY)


def describe(keys: tuple[str, ...]) -> dict[str, str]:
    missing = [key for key in keys if key not in DATA_DICTIONARY]
    if missing:
        raise KeyError(f"undocumented dictionary keys: {missing}")
    return {key: DATA_DICTIONARY[key] for key in keys}
