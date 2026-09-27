"""Whole-site and whole-region partitions with leakage guards.

The split is by site, never by row: the primary external setting removes Site C and Region
III entirely, the cross-region setting removes Site B, and the cross-site consistency
setting reports all three sites separately. Every partition is patient-disjoint, and the
calibration share is disjoint from the fitting share because the weight control and the
thresholds are fitted there.

Ref: Sec. 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wireope.config import LeakageGuardConfig, PartitionSpec


@dataclass(frozen=True)
class PartitionIndex:
    name: str
    role: str
    sites: tuple[str, ...]
    regions: tuple[str, ...]
    member: NDArray[np.bool_]
    configuration: str = ""

    @property
    def size(self) -> int:
        return int(np.count_nonzero(self.member))


@dataclass(frozen=True)
class PartitionSet:
    partitions: tuple[PartitionIndex, ...]

    def by_name(self, name: str) -> PartitionIndex:
        for partition in self.partitions:
            if partition.name == name:
                return partition
        raise KeyError(f"unknown partition '{name}'")

    def names(self) -> tuple[str, ...]:
        return tuple(partition.name for partition in self.partitions)

    def configurations(self) -> tuple[str, ...]:
        return tuple(
            sorted({partition.configuration for partition in self.partitions if partition.configuration})
        )

    def in_configuration(self, name: str) -> tuple[PartitionIndex, ...]:
        return tuple(partition for partition in self.partitions if partition.configuration == name)


def build_partitions(
    specs: tuple[PartitionSpec, ...],
    sites: NDArray[np.str_],
    regions: NDArray[np.str_],
    patient_ids: NDArray[np.int64] | None = None,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> PartitionSet:
    """Materialise every partition, carving the calibration share out by patient.

    A spec that declares `disjoint_from` is built from that partition's patients rather
    than from the site list alone, so the calibration share is genuinely disjoint instead of
    merely declared so.
    """
    partitions: list[PartitionIndex] = []
    built: dict[str, PartitionIndex] = {}
    for spec in specs:
        site_member = np.isin(sites, np.asarray(spec.sites))
        if spec.regions:
            region_member = np.isin(regions, np.asarray(spec.regions))
            member = site_member if spec.name != "external_site" else (site_member & region_member)
        else:
            member = site_member
        if spec.disjoint_from and patient_ids is not None:
            base = built.get(spec.disjoint_from)
            if base is None:
                raise KeyError(f"partition '{spec.name}' refers to '{spec.disjoint_from}' before it exists")
            carved, remainder = disjoint_calibration_share(
                base.member, patient_ids, calibration_fraction, seed
            )
            member = carved
            # The development share gives up the calibration patients, so the two are disjoint
            # rather than merely declared to be.
            trimmed = PartitionIndex(
                name=base.name,
                role=base.role,
                sites=base.sites,
                regions=base.regions,
                member=remainder,
                configuration=base.configuration,
            )
            partitions[partitions.index(base)] = trimmed
            built[base.name] = trimmed
        index = PartitionIndex(
            name=spec.name,
            role=spec.role,
            sites=tuple(spec.sites),
            regions=tuple(spec.regions),
            member=np.asarray(member, dtype=np.bool_),
            configuration=spec.configuration,
        )
        partitions.append(index)
        built[spec.name] = index
    return PartitionSet(partitions=tuple(partitions))


def patient_disjointness(
    patient_ids: NDArray[np.int64],
    left: PartitionIndex,
    right: PartitionIndex,
    patient_site: dict[int, str],
) -> bool:
    """No patient may appear in two shares, and no patient may cross a site boundary."""
    left_patients = {int(item) for item in patient_ids[left.member]}
    right_patients = {int(item) for item in patient_ids[right.member]}
    if left_patients & right_patients:
        return False
    for patient in left_patients | right_patients:
        site = patient_site.get(patient)
        if site is None:
            continue
        permitted = set(left.sites) | set(right.sites)
        if site not in permitted:
            return False
    return True


def operator_disjointness(
    operators: NDArray[np.str_],
    left: PartitionIndex,
    right: PartitionIndex,
) -> bool:
    left_operators = set(operators[left.member].tolist())
    right_operators = set(operators[right.member].tolist())
    return not (left_operators & right_operators)


def vendor_disjointness(
    vendor_classes: NDArray[np.str_],
    left: PartitionIndex,
    right: PartitionIndex,
) -> bool:
    left_vendors = set(vendor_classes[left.member].tolist())
    right_vendors = set(vendor_classes[right.member].tolist())
    return not (left_vendors & right_vendors)


SPLIT_ROLES: tuple[str, ...] = (
    "development",
    "disjoint_weight_and_threshold_fitting",
    "primary_held_out_site_and_region",
    "cross_region_holdout",
)


@dataclass(frozen=True)
class LeakageReport:
    patient_disjoint: bool
    operator_disjoint: bool
    vendor_disjoint: bool
    calibration_disjoint: bool
    overlapping_partitions: tuple[tuple[str, str], ...]

    @property
    def clean(self) -> bool:
        return (
            self.patient_disjoint
            and self.operator_disjoint
            and self.vendor_disjoint
            and self.calibration_disjoint
            and not self.overlapping_partitions
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "patient_disjoint": self.patient_disjoint,
            "operator_disjoint": self.operator_disjoint,
            "vendor_disjoint": self.vendor_disjoint,
            "calibration_disjoint": self.calibration_disjoint,
            "overlapping_partitions": [list(item) for item in self.overlapping_partitions],
            "clean": self.clean,
        }


def audit_leakage(
    partitions: PartitionSet,
    patient_ids: NDArray[np.int64],
    operators: NDArray[np.str_],
    vendor_classes: NDArray[np.str_],
    guards: LeakageGuardConfig,
) -> LeakageReport:
    """Audit the guards the protocol promises, and name any pair that violates one."""
    overlapping: list[tuple[str, str]] = []
    names = tuple(name for name in partitions.names() if partitions.by_name(name).role in SPLIT_ROLES)
    patient_ok = True
    operator_ok = True
    vendor_ok = True
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            left = partitions.by_name(names[first])
            right = partitions.by_name(names[second])
            if left.configuration != right.configuration:
                # The three held-out settings are alternative configurations of the same
                # cohort, so a partition of one shares sites with the development share of
                # another by construction; only within-configuration pairs are audited.
                continue
            if "disjoint_weight_and_threshold_fitting" in (left.role, right.role):
                # The calibration share is carved out of the fitting site pool, so it shares
                # operators and vendors with it by design; only its patients must be disjoint.
                continue
            intersection = int(np.count_nonzero(left.member & right.member))
            if intersection > 0:
                overlapping.append((left.name, right.name))
            if guards.patient_disjoint:
                left_patients = set(patient_ids[left.member].tolist())
                right_patients = set(patient_ids[right.member].tolist())
                if left_patients & right_patients:
                    patient_ok = False
            if guards.operator_disjoint_across_partitions and not operator_disjointness(
                operators, left, right
            ):
                operator_ok = False
            if guards.vendor_disjoint_across_partitions and not vendor_disjointness(
                vendor_classes, left, right
            ):
                vendor_ok = False
    calibration_disjoint = True
    if guards.calibration_share_disjoint:
        for configuration in partitions.configurations():
            members = partitions.in_configuration(configuration)
            development = [item for item in members if item.role == "development"]
            calibration_share = [
                item for item in members if item.role == "disjoint_weight_and_threshold_fitting"
            ]
            if not development or not calibration_share:
                continue
            fit_patients = {int(item) for item in patient_ids[development[0].member]}
            calibration_patients = {int(item) for item in patient_ids[calibration_share[0].member]}
            if fit_patients & calibration_patients:
                calibration_disjoint = False
    return LeakageReport(
        patient_disjoint=patient_ok,
        operator_disjoint=operator_ok,
        vendor_disjoint=vendor_ok,
        calibration_disjoint=calibration_disjoint,
        overlapping_partitions=tuple(overlapping),
    )


def disjoint_calibration_share(
    member: NDArray[np.bool_],
    patient_ids: NDArray[np.int64],
    fraction: float,
    seed: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split a fitting share into weights/threshold fitting and model fitting by patient."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids[member])
    count = max(1, int(round(fraction * unique_patients.shape[0])))
    chosen = np.sort(rng.choice(unique_patients, size=count, replace=False))
    calibration = member & np.isin(patient_ids, chosen)
    fitting = member & ~np.isin(patient_ids, chosen)
    return np.asarray(calibration, dtype=np.bool_), np.asarray(fitting, dtype=np.bool_)


def partition_summary(partition: PartitionIndex, site_array: NDArray[np.str_]) -> dict[str, float]:
    summary: dict[str, float] = {"size": float(partition.size)}
    for site in partition.sites:
        summary[f"site::{site}"] = float(np.count_nonzero(site_array[partition.member] == site))
    return summary


def sitewise_slices(sites: NDArray[np.str_]) -> dict[str, NDArray[np.bool_]]:
    return {str(name): np.asarray(sites == name, dtype=np.bool_) for name in sorted(set(sites.tolist()))}
