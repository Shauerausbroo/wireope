"""The verification driver: run both layers, gate the tooling, and write the artefacts.

Run order inside one pass: the tooling gates first, then the execution and manuscript layers,
then the claim map, the report, the plain-text summary, and the integrity manifest last. Running
the driver twice with no edits in between must produce byte-identical artefacts.

Ref: the release contract in README.
"""

from __future__ import annotations

from pathlib import Path

from wireope.schema import CheckStatus
from wireope.utils.logging_setup import get_logger
from wireope.verification import deviations
from wireope.verification.claims import apply_probes, build_claims
from wireope.verification.execution import (
    Fixture,
    build_fixture,
    run_execution_checks,
    run_gates,
)
from wireope.verification.manuscript import run_manuscript_checks
from wireope.verification.report import (
    build_report,
    render_summary,
    write_claims,
    write_manifest,
    write_report,
    write_summary,
)

LOGGER = get_logger(__name__)


def run_verification(
    repo_root: Path,
    skip_tools: bool = False,
    skip_gates: bool = False,
    probe_links: bool = False,
) -> dict[str, object]:
    """Execute the two verification layers and rewrite every root artefact."""
    root = Path(repo_root).resolve()
    fixture: Fixture = build_fixture(root, probe_links=probe_links)
    try:
        execution = run_execution_checks(fixture)
        manuscript = run_manuscript_checks()
        gates = () if skip_gates else run_gates(root)
        statuses = {check.name: check.status for check in (*execution, *gates)}
        claims = apply_probes(build_claims(), statuses)
        report = build_report(execution, gates, manuscript, claims)
        summary_text = render_summary(report, claims)
        write_claims(root, claims, deviations.as_payload())
        write_report(root, report)
        write_summary(root, summary_text)
        write_manifest(root)
    finally:
        fixture.tempdir.cleanup()
    LOGGER.info("verification finished with status %s", report["overall_status"])
    return report


def report_path(repo_root: Path) -> Path:
    return Path(repo_root) / "verification_report.json"


def read_status(repo_root: Path) -> str:
    import json

    payload = json.loads(report_path(repo_root).read_text())
    return str(payload["overall_status"])


def failed_names(repo_root: Path) -> list[str]:
    import json

    payload = json.loads(report_path(repo_root).read_text())
    names: list[str] = []
    for family in ("execution", "gates"):
        for entry in payload["families"][family]:
            if entry["status"] == CheckStatus.FAIL.value:
                names.append(str(entry["name"]))
    return names
