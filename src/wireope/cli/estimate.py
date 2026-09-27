"""Estimate the strategy library and write the per-strategy estimate table."""

from __future__ import annotations

import sys
from pathlib import Path

from wireope.cli._common import add_arm_switches, base_parser, display, load_context
from wireope.studies.pipeline import strategy_estimates
from wireope.utils.atomic import atomic_write_json
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("estimate every strategy in the library")
    add_arm_switches(parser)
    parser.add_argument("--out", default="artefacts/estimate", help="directory for the estimate table")
    args = parser.parse_args(argv)
    context = load_context(args)
    table = strategy_estimates(context, context.evaluation_member, args.clipping)
    ordered = sorted(table.items(), key=lambda item: -item[1]["graded_value"])
    payload = {
        "experiment": args.experiment,
        "configuration": args.configuration,
        "clipping": args.clipping,
        "attempts": int(context.evaluation_member.sum()),
        "estimator": "nested_weighted_doubly_robust",
        "strategies": [
            {"strategy": name, **{key: round(value, 6) for key, value in values.items()}}
            for name, values in ordered
        ],
    }
    target = Path(args.repo_root) / args.out / "strategy_estimates.json"
    atomic_write_json(target, payload)
    LOGGER.info("wrote %s", display(target, args.repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
