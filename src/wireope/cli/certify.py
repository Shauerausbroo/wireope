"""Certify the strategy library under the support floor and the risk margin."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from wireope.cli._common import add_arm_switches, base_parser, load_context
from wireope.studies.certify import certify_library
from wireope.utils.atomic import atomic_write_json
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("certify the strategy library")
    add_arm_switches(parser)
    parser.add_argument("--out", default="artefacts/certify", help="directory for the decision record")
    args = parser.parse_args(argv)
    context = load_context(args)
    library = certify_library(context, context.evaluation_member, args.clipping)
    payload = {
        "experiment": args.experiment,
        "configuration": args.configuration,
        "clipping": args.clipping,
        "support_floor": context.config.support.floor_c,
        "risk_margin": context.config.sweep.risk_margin_m,
        "decision": asdict(library.constrained) | {"abstained": library.constrained.abstained},
        "unconstrained_decision": asdict(library.unconstrained),
        "strategies": [
            asdict(estimate) | {"graded_lcb": estimate.graded_lcb, "risk_lcb": estimate.risk_lcb}
            for estimate in library.estimates
        ],
        "floor_failures": {
            name: [list(cell) for cell in report.failing] for name, report in library.floors.items()
        },
    }
    target = Path(args.repo_root) / args.out / "certification.json"
    atomic_write_json(target, payload)
    LOGGER.info(
        "selected %s, abstained %s, eligible %d",
        library.strategy or "(none)",
        library.constrained.abstained,
        library.constrained.eligible_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
