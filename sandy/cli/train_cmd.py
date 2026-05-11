"""CLI subcommand for model training.

Task 12.5: sandy train [--seed INT] [--output PATH]
Phase 1.5: extended with --target flag for game_winner/runs training.

Requirements: 6.4, 6.5, 6.6, 10.2, 4.5, 5.4
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from sandy.cli._helpers import require_config
from sandy.db import create_engine, get_connection
from sandy.logging import configure_logging
from sandy.train.artifact import save_artifact
from sandy.train.trainer import TrainingQualityError, train_model


@click.command()
@click.option("--seed", type=int, default=42, help="Random seed for training (default 42)")
@click.option("--output", type=click.Path(), default=None, help="Output path for model artifact")
@click.option(
    "--target",
    type=click.Choice(["reached_base", "game_winner", "runs", "volatility", "meta"], case_sensitive=False),
    default="reached_base",
    help="Training target (default: reached_base)",
)
@click.pass_context
def train(ctx: click.Context, seed: int, output: str | None, target: str) -> None:
    """Fit LightGBM model and write artifact."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)

    if target == "reached_base":
        output_path = Path(output) if output else cfg.model.path
        try:
            with get_connection(engine) as conn:
                artifact = train_model(conn, seed=seed)
        except TrainingQualityError as exc:
            click.echo(
                f"Training failed: validation ROC AUC {exc.roc_auc:.4f} "
                f"is below minimum threshold {exc.threshold:.2f}.",
                err=True,
            )
            sys.exit(1)
        except ValueError as exc:
            click.echo(f"Training failed: {exc}", err=True)
            sys.exit(1)

        save_artifact(artifact, output_path)
        click.echo(
            f"Model saved to {output_path}\n"
            f"  Target: reached_base\n"
            f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
            f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
        )

    elif target == "game_winner":
        from sandy.train.trainer import train_game_winner_model

        output_path = Path(output) if output else cfg.model.artifact_path("game_winner")
        try:
            with get_connection(engine) as conn:
                artifact = train_game_winner_model(conn, seed=seed)
        except TrainingQualityError as exc:
            click.echo(
                f"Training failed: validation ROC AUC {exc.roc_auc:.4f} "
                f"is below minimum threshold {exc.threshold:.2f}.",
                err=True,
            )
            sys.exit(1)
        except ValueError as exc:
            click.echo(f"Training failed: {exc}", err=True)
            sys.exit(1)

        save_artifact(artifact, output_path)
        click.echo(
            f"Model saved to {output_path}\n"
            f"  Target: game_winner\n"
            f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
            f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
        )

    elif target == "runs":
        from sandy.train.trainer import train_runs_model

        output_path = Path(output) if output else cfg.model.artifact_path("runs")
        try:
            with get_connection(engine) as conn:
                artifact = train_runs_model(conn, seed=seed)
        except ValueError as exc:
            click.echo(f"Training failed: {exc}", err=True)
            sys.exit(1)

        save_artifact(artifact, output_path)
        click.echo(
            f"Model saved to {output_path}\n"
            f"  Target: runs\n"
            f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
            f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
        )

    elif target == "volatility":
        from sandy.over_under.volatility import train_volatility_model

        output_path = Path(output) if output else cfg.model.artifact_path("volatility")
        try:
            artifact = train_volatility_model(cfg, seed=seed)
        except ValueError as exc:
            click.echo(f"Training failed: {exc}", err=True)
            sys.exit(1)

        save_artifact(artifact, output_path)
        click.echo(
            f"Model saved to {output_path}\n"
            f"  Target: volatility\n"
            f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
            f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
        )

    elif target == "meta":
        from sandy.over_under.meta_model import train_meta_model

        output_path = Path(output) if output else cfg.model.artifact_path("meta_over_5_5")
        try:
            artifact = train_meta_model(engine, cfg, seed=seed)
        except ValueError as exc:
            click.echo(f"Training failed: {exc}", err=True)
            sys.exit(1)

        save_artifact(artifact, output_path)
        click.echo(
            f"Model saved to {output_path}\n"
            f"  Target: meta_over_5_5\n"
            f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
            f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
        )
