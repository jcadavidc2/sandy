"""CLI subcommand for model training.

Task 12.5: sandy train [--seed INT] [--output PATH]

Requirements: 6.4, 6.5, 6.6, 10.2
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
@click.pass_context
def train(ctx: click.Context, seed: int, output: str | None) -> None:
    """Fit LightGBM model and write artifact."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
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
        f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}\n"
        f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})"
    )
