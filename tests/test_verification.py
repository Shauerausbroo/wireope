"""The verification layers: claims, the execution checks and the manuscript arithmetic.

The artefact writers are exercised against a temporary root so a test run can never replace the
shipped reports, and one test asserts that the shipped report is the complete one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wireope.report import transcribe
from wireope.schema import CheckStatus
from wireope.utils.manifest import verify_manifest
from wireope.verification.claims import (
    DATASET_CLAIMS,
    EQUATION_CLAIMS,
    SECTION_CLAIMS,
    apply_probes,
    build_claims,
    summary as claim_summary,
    table_claims,
)
from wireope.verification.deviations import DEVIATIONS, as_payload, count, locations
from wireope.verification.execution import (
    build_fixture,
    run_execution_checks,
    summary as execution_summary,
)
from wireope.verification.manuscript import run_manuscript_checks, summary as manuscript_summary
from wireope.verification.report import (
    build_report,
    code_status,
    overall_status,
    render_summary,
    write_claims,
    write_manifest,
    write_report,
    write_summary,
)
from wireope.version import RELEASE_SLUG


def test_manuscript_checks_are_reproducible() -> None:
    first = run_manuscript_checks()
    second = run_manuscript_checks()
    assert [check.name for check in first] == [check.name for check in second]
    assert [check.status for check in first] == [check.status for check in second]
    summary = manuscript_summary(first)
    assert summary["total"] == float(len(first))
    assert summary["passed"] + summary["failed"] + summary["not_run"] == summary["total"]


def test_manuscript_findings_match_the_transcription() -> None:
    checks = {check.name: check for check in run_manuscript_checks()}
    territory = checks["manuscript.territory_partition_events"]
    assert territory.status is CheckStatus.FAIL
    assert territory.evidence["gap"] == 15
    direction = checks["manuscript.ladder_nesting_direction"]
    assert direction.status is CheckStatus.FAIL
    parity = checks["manuscript.selection_accuracy_parity_clause"]
    assert parity.status is CheckStatus.PASS
    assert parity.evidence["gap_pp"] == pytest.approx(0.4, abs=1e-9)
    resolution = checks["manuscript.rung_resolution_at_stated_tolerance"]
    assert resolution.status is CheckStatus.NOT_RUN


def test_every_transcribed_row_is_claimed() -> None:
    claims = table_claims()
    expected = sum(len(rows) for rows in transcribe.TABLES.values())
    assert len(claims) == expected
    identifiers = {claim.claim_id for claim in claims}
    assert len(identifiers) == expected
    assert all(claim.status is CheckStatus.NOT_RUN for claim in claims)
    assert all(not claim.estimator_stated or claim.expected for claim in claims)


def test_claim_map_and_probe_application() -> None:
    claims = build_claims()
    assert len(claims) == len(EQUATION_CLAIMS) + len(SECTION_CLAIMS) + len(DATASET_CLAIMS) + len(
        table_claims()
    )
    statuses = {"geometry.analytic_excursion_matches_discretised_field": CheckStatus.PASS}
    updated = apply_probes(claims, statuses)
    by_id = {claim.claim_id: claim for claim in updated}
    assert by_id["eq1_excursion_observable"].status is CheckStatus.PASS
    assert by_id["eq5_support_constrained_lcb"].status is CheckStatus.NOT_RUN
    summary = claim_summary(updated)
    assert summary["claims"] == float(len(claims))
    assert sum(summary[key] for key in ("passed", "failed", "not_run", "blocked")) == summary["claims"]


def test_deviations_are_anchored() -> None:
    assert count() == len(DEVIATIONS)
    assert all(entry["paper_location"] for entry in DEVIATIONS)
    assert all(entry["why"] for entry in DEVIATIONS)
    assert len(locations()) == count()
    assert as_payload() is not DEVIATIONS


def test_execution_checks_run_and_report(repo_root: Path) -> None:
    fixture = build_fixture(repo_root)
    try:
        checks = run_execution_checks(fixture)
    finally:
        fixture.tempdir.cleanup()
    names = [check.name for check in checks]
    assert len(names) == len(set(names)), "check names must be unique"
    summary = execution_summary(checks)
    assert summary["total"] == float(len(checks))
    by_name = {check.name: check for check in checks}
    assert by_name["geometry.analytic_excursion_matches_discretised_field"].status is CheckStatus.PASS
    assert by_name["estimators.nested_dr_recovers_known_functionals"].status is CheckStatus.PASS
    assert by_name["metrics.auroc_matches_an_independent_library"].status is CheckStatus.PASS
    assert by_name["data.private_cohort_record_level_availability"].status is CheckStatus.BLOCKED
    assert by_name["container.docker_build"].status in {CheckStatus.BLOCKED, CheckStatus.NOT_RUN}
    tolerated = {"data.dataset_links_reachable"}
    assert not any(check.status is CheckStatus.FAIL and check.name not in tolerated for check in checks)


def test_status_aggregation() -> None:
    from wireope.verification.execution import ExecutionCheck

    clean = (
        ExecutionCheck("a", CheckStatus.PASS, "ok"),
        ExecutionCheck("b", CheckStatus.NOT_RUN, "absent"),
    )
    assert overall_status(clean) == "PARTIALLY_VERIFIED"
    assert code_status(clean) == "PARTIALLY_VERIFIED"
    failed = (ExecutionCheck("a", CheckStatus.FAIL, "bad"),)
    assert overall_status(failed) == "UNVERIFIED"


def test_artefact_writers_use_the_given_root(tmp_path: Path) -> None:
    claims = build_claims()
    manuscript = run_manuscript_checks()
    report = build_report((), (), manuscript, claims)
    summary_text = render_summary(report, claims)
    claims_path = write_claims(tmp_path, claims, as_payload())
    report_path = write_report(tmp_path, report)
    summary_path = write_summary(tmp_path, summary_text)
    manifest_path = write_manifest(tmp_path)
    assert claims_path.parent == tmp_path
    assert report_path.exists() and summary_path.exists() and manifest_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["root"] == RELEASE_SLUG
    assert payload["counts"]["total"] == 0.0
    manifest = json.loads(manifest_path.read_text())
    recorded = {entry["path"] for entry in manifest["files"]}
    assert "verification_report.json" in recorded
    assert "verification_summary.txt" in recorded
    assert "integrity_manifest.json" not in recorded
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest_digest_matches(tmp_path, manifest_path)


def manifest_digest_matches(root: Path, manifest_path: Path) -> bool:
    report = verify_manifest(root, manifest_path)
    return bool(report["intact"])


def test_shipped_report_is_complete(repo_root: Path) -> None:
    path = repo_root / "verification_report.json"
    if not path.is_file():
        pytest.skip("the shipped report is written by the verification driver")
    payload = json.loads(path.read_text())
    assert payload["counts"]["total"] > 40.0
    assert payload["claim_counts"]["claims"] > 50.0
    assert "manuscript_discrepancies" in payload
    assert payload["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
