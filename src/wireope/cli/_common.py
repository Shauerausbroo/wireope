"""Shared command-line helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from wireope.studies.pipeline import (
    AnalysisContext,
    ArmSwitches,
    CohortBundle,
    fit_context,
    prepare_bundle,
)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="release root")
    parser.add_argument("--experiment", default="main", help="experiment name under configs/experiment")
    parser.add_argument(
        "--configuration",
        default="primary_external",
        help="held-out configuration: primary_external, cross_region or sitewise",
    )
    parser.add_argument("--clipping", type=float, default=0.10, help="importance-weight clipping point")
    return parser


def add_arm_switches(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--without-reconstruction",
        action="store_true",
        help="remove the reconstructed action channel (falsification arm)",
    )
    parser.add_argument(
        "--without-ladder",
        action="store_true",
        help="replace the nested ladder with the binary endpoint comparator",
    )
    parser.add_argument(
        "--without-support",
        action="store_true",
        help="remove the per-stratum overlap floor and the risk constraint",
    )


def arm_from_args(args: argparse.Namespace) -> ArmSwitches:
    return ArmSwitches(
        envelope_excursion_reconstruction=not getattr(args, "without_reconstruction", False),
        nested_excursion_ladder=not getattr(args, "without_ladder", False),
        support_constraint=not getattr(args, "without_support", False),
    )


def display(target: Path, root: Path) -> str:
    """Path for a log line: relative to the release root when it sits inside, else its name."""
    try:
        return str(target.relative_to(root))
    except ValueError:
        return target.name


def load_bundle(args: argparse.Namespace) -> CohortBundle:
    return prepare_bundle(args.repo_root, args.experiment)


def load_context(args: argparse.Namespace, bundle: CohortBundle | None = None) -> AnalysisContext:
    prepared = bundle if bundle is not None else load_bundle(args)
    return fit_context(prepared, args.configuration, arm_from_args(args))
