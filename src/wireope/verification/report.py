"""Writers for the release artefacts.

The order matters: the tooling gates run first, then the claim map and the report are written,
then the plain-text summary, and the integrity manifest last so that it digests the finished
tree and excludes only itself. Every writer takes the repository root explicitly, so a test can
point it at a temporary directory without overwriting the shipped artefacts.

Ref: the release contract in README.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wireope.utils.atomic import atomic_write_json, atomic_write_text
from wireope.utils.manifest import write_integrity_manifest
from wireope.verification.claims import Claim, summary as claim_summary
from wireope.verification.execution import ExecutionCheck, summary as execution_summary
from wireope.verification.manuscript import ManuscriptCheck, summary as manuscript_summary
from wireope.version import RELEASE_SLUG

SUMMARY_NAME = "verification_summary.txt"
REPORT_NAME = "verification_report.json"
CLAIMS_NAME = "claim_to_code.json"
MANIFEST_NAME = "integrity_manifest.json"


def overall_status(checks: tuple[ExecutionCheck, ...]) -> str:
    """PARTIALLY_VERIFIED whenever anything is outstanding, UNVERIFIED when anything failed."""
    counts = execution_summary(checks)
    if counts["failed"] > 0:
        return "UNVERIFIED"
    if counts["not_run"] > 0 or counts["blocked"] > 0:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def code_status(checks: tuple[ExecutionCheck, ...]) -> str:
    """Verdict over the release's own code, excluding the manuscript family and the gates."""
    relevant = tuple(
        check
        for check in checks
        if not check.name.startswith("manuscript.") and not check.name.startswith("gate.")
    )
    counts = execution_summary(relevant)
    if counts["failed"] > 0:
        return "UNVERIFIED"
    if counts["not_run"] > 0 or counts["blocked"] > 0:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def build_report(
    execution: tuple[ExecutionCheck, ...],
    gates: tuple[ExecutionCheck, ...],
    manuscript: tuple[ManuscriptCheck, ...],
    claims: tuple[Claim, ...],
) -> dict[str, Any]:
    all_checks = (*execution, *gates)
    return {
        "root": RELEASE_SLUG,
        "overall_status": overall_status(all_checks),
        "code_status": code_status(all_checks),
        "gate_status": overall_status(gates) if gates else "NOT_RUN",
        "counts": execution_summary(all_checks),
        "claim_counts": claim_summary(claims),
        "manuscript_counts": manuscript_summary(manuscript),
        "families": {
            "execution": [asdict(check) | {"status": check.status.value} for check in execution],
            "gates": [asdict(check) | {"status": check.status.value} for check in gates],
            "manuscript": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "detail": check.detail,
                    "anchor": check.anchor,
                    "evidence": check.evidence,
                }
                for check in manuscript
            ],
        },
        "manuscript_discrepancies": [
            {"name": check.name, "detail": check.detail, "anchor": check.anchor}
            for check in manuscript
            if check.status.value == "FAIL"
        ],
        "interpretation": (
            "A failure in the manuscript family is a disagreement inside the paper's own tables and "
            "is not a defect in this release; code_status is computed over every other family. A "
            "NOT_RUN entry means the check could not discriminate on the available evidence, and a "
            "BLOCKED entry means the evidence is absent by contract, as the private cohort is."
        ),
    }


def render_summary(report: dict[str, Any], claims: tuple[Claim, ...]) -> str:
    counts = report["counts"]
    lines = [
        f"root               : {report['root']}",
        f"overall status     : {report['overall_status']}",
        f"code status        : {report['code_status']}",
        f"gate status        : {report['gate_status']}",
        f"checks             : {int(counts['total'])}",
        f"counts             : {counts}",
        f"claim map          : {int(report['claim_counts']['claims'])} claims, "
        f"{int(report['claim_counts']['passed'])} passed, "
        f"{int(report['claim_counts']['not_run'])} not run, "
        f"{int(report['claim_counts']['blocked'])} blocked",
        f"manuscript findings: {int(report['manuscript_counts']['failed'])}",
    ]
    failing = [entry["name"] for entry in report["families"]["execution"] if entry["status"] == "FAIL"]
    not_running = [
        entry["name"] for entry in report["families"]["execution"] if entry["status"] == "NOT_RUN"
    ]
    blocked = [entry["name"] for entry in report["families"]["execution"] if entry["status"] == "BLOCKED"]
    if failing:
        lines.append(f"failed             : {failing}")
    if not_running:
        lines.append(f"not run            : {not_running}")
    if blocked:
        lines.append(f"blocked            : {blocked}")
    lines.append("")
    lines.append("manuscript findings:")
    for entry in report["manuscript_discrepancies"]:
        lines.append(f"  - {entry['name']}: {entry['detail']} [{entry['anchor']}]")
    lines.append("")
    lines.append("operations whose estimator the manuscript does not state:")
    for claim in claims:
        if not claim.estimator_stated:
            lines.append(f"  - {claim.claim_id}: {claim.note}")
    lines.append("")
    lines.append(f"interpretation     : {report['interpretation']}")
    return "\n".join(lines) + "\n"


def write_claims(root: Path, claims: tuple[Claim, ...], deviations: list[dict[str, str]]) -> Path:
    payload = {
        "project": RELEASE_SLUG,
        "claim_counts": claim_summary(claims),
        "deviations": deviations,
        "claims": [claim.as_mapping() for claim in claims],
    }
    target = root / CLAIMS_NAME
    atomic_write_json(target, payload)
    return target


def write_report(root: Path, report: dict[str, Any]) -> Path:
    target = root / REPORT_NAME
    atomic_write_json(target, report)
    return target


def write_summary(root: Path, text: str) -> Path:
    target = root / SUMMARY_NAME
    atomic_write_text(target, text)
    return target


def write_manifest(root: Path) -> Path:
    target = root / MANIFEST_NAME
    write_integrity_manifest(root, target)
    return target


def load_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload
