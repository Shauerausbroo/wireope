"""Run every reported table harness and write the release's own plain-text report."""

from __future__ import annotations

import sys
from pathlib import Path

from wireope.cli._common import base_parser, display, load_context
from wireope.report.render import write_release_report
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("run the reported table harnesses and write the release report")
    parser.add_argument("--out", default="artefacts/report.txt", help="plain-text report path")
    args = parser.parse_args(argv)
    context = load_context(args)
    target = Path(args.repo_root) / args.out
    write_release_report(target, context, args.clipping)
    LOGGER.info("wrote %s", display(target, args.repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
