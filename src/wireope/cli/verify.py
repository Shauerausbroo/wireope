"""Run the two verification layers and rewrite every root artefact."""

from __future__ import annotations

import sys
from pathlib import Path

from wireope.utils.logging_setup import get_logger
from wireope.verification.driver import run_verification

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="run the release verification suite")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="release root")
    parser.add_argument("--skip-gates", action="store_true", help="skip the tooling gates")
    parser.add_argument(
        "--skip-tools",
        action="store_true",
        help="alias for --skip-gates, kept for symmetric invocation",
    )
    parser.add_argument(
        "--probe-links",
        action="store_true",
        help="reach the network to re-probe every recorded dataset link",
    )
    args = parser.parse_args(argv)
    report = run_verification(
        args.repo_root,
        skip_tools=args.skip_tools,
        skip_gates=args.skip_gates or args.skip_tools,
        probe_links=args.probe_links,
    )
    LOGGER.info("overall status %s", report["overall_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
