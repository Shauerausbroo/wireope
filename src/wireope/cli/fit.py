"""Fit the two-part behaviour model, the rung-wise outcome model and the hazard head."""

from __future__ import annotations

import sys
from pathlib import Path

from wireope.cli._common import add_arm_switches, arm_from_args, base_parser, display, load_bundle
from wireope.fit.dataset import ExcursionDataset, rung_weight_tensor
from wireope.fit.loop import train_hazard_head
from wireope.fit.outcome import build_design
from wireope.utils.atomic import atomic_write_json
from wireope.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fit the reconstructed models and the hazard head")
    add_arm_switches(parser)
    parser.add_argument("--out", default="artefacts/fit", help="directory for the fit record")
    parser.add_argument("--seed", type=int, default=None, help="override the configured seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="cap the epochs (smoke only)")
    parser.add_argument("--skip-hazard", action="store_true", help="fit only the tabular models")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="directory for the checkpoint; defaults to the configured runtime path",
    )
    args = parser.parse_args(argv)
    bundle = load_bundle(args)
    arm = arm_from_args(args)
    config = bundle.config
    run_seed = config.release_cohort.seed if args.seed is None else args.seed
    aggressiveness = bundle.cohort.action_array()
    design = build_design(bundle.cohort.states, aggressiveness, include_action=arm.uses_action_term)
    payload: dict[str, object] = {
        "experiment": args.experiment,
        "seed": run_seed,
        "outcome_design": list(design.feature_names),
        "attempts": bundle.cohort.attempts,
    }
    if not args.skip_hazard:
        dataset = ExcursionDataset(bundle.cohort, bundle.ladder, config.kinematics)
        checkpoint_root = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir
            else Path(args.repo_root) / config.runtime.checkpoint_dir
        )
        checkpoint = checkpoint_root / f"{args.experiment}.pt"
        _, result = train_hazard_head(
            dataset=dataset,
            rungs=bundle.ladder.depth,
            hazard=config.hazard,
            optimizer_config=config.optimizer,
            schedule_config=config.schedule,
            seed=run_seed,
            rung_weights=rung_weight_tensor(bundle.ladder),
            checkpoint_path=checkpoint,
            max_epochs=args.max_epochs,
        )
        payload["hazard"] = {
            "epochs": result.epochs_run,
            "steps": result.steps_run,
            "first_loss": result.first_loss,
            "final_loss": result.final_loss,
            "gradient_norm": result.gradient_norm,
            "checkpoint_digest": result.checkpoint_digest,
            "seconds": result.seconds,
            "loss_decreased": result.loss_decreased,
        }
    target = Path(args.repo_root) / args.out / "fit_summary.json"
    atomic_write_json(target, payload)
    LOGGER.info("wrote %s", display(target, args.repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
