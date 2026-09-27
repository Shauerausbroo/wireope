"""Assemble the release cohort, the partitions and the reconstructed observables."""

from __future__ import annotations

import sys
from dataclasses import asdict

import numpy as np

from wireope.cli._common import base_parser, display, load_bundle
from wireope.cohort.schema import cohort_totals, site_table
from wireope.utils.atomic import atomic_write_json
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("assemble the release cohort, its partitions and its observables")
    parser.add_argument("--out", default="artefacts/cohort", help="directory for the released summary")
    args = parser.parse_args(argv)
    bundle = load_bundle(args)
    tables = site_table(
        bundle.cohort.site_array(),
        bundle.cohort.region_array(),
        bundle.cohort.injury_flags(),
        bundle.cohort.operator_array(),
        bundle.cohort.vendor_array(),
    )
    payload = {
        "experiment": args.experiment,
        "attempts": bundle.cohort.attempts,
        "injury_events": int(bundle.cohort.injury_flags().sum()),
        "rung_prevalence": [
            float(np.mean(bundle.indicators[:, rung])) for rung in range(bundle.indicators.shape[1])
        ],
        "sites": [asdict(table) | {"prevalence": table.prevalence} for table in tables],
        "totals": cohort_totals(tables),
        "calibration_intercepts": bundle.cohort.calibration_intercepts,
        "leakage": bundle.leakage.as_mapping(),
        "ladder_depth": bundle.ladder.depth,
        "ladder_thresholds_mm": list(bundle.ladder.tau_mm),
    }
    target = args.repo_root / args.out / "cohort_summary.json"
    atomic_write_json(target, payload)
    LOGGER.info("wrote %s", display(target, args.repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
