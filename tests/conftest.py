"""Shared fixtures: one release-cohort bundle and one fitted context per test session."""

from __future__ import annotations

from pathlib import Path

import pytest

from wireope.studies.pipeline import (
    AnalysisContext,
    CohortBundle,
    fit_context,
    prepare_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def bundle(repo_root: Path) -> CohortBundle:
    return prepare_bundle(repo_root, "_smoke")


@pytest.fixture(scope="session")
def context(bundle: CohortBundle) -> AnalysisContext:
    return fit_context(bundle, "primary_external")


@pytest.fixture(scope="session")
def cross_region_context(bundle: CohortBundle) -> AnalysisContext:
    return fit_context(bundle, "cross_region")
