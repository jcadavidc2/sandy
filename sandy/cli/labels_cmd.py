"""CLI subcommand for label generation.

Task 12.3: sandy labels build [--game-pk INT]
Phase 1.5: extended with --target flag for game_winner/runs labels.

Requirements: 4.1, 10.2, 12.1, 12.2
"""
from __future__ import annotations

import click

from sandy.cli._helpers import require_config
from sandy.db import create_engine
from sandy.labels.runner import run_labels
from sandy.logging import configure_logging


@click.group()
def labels():
    """Manage derived labels (inning_labels, game_winner_labels, runs_labels)."""
    pass


@labels.command()
@click.option("--game-pk", type=int, default=None, help="Process only this game_pk")
@click.option(
    "--target",
    type=click.Choice(["reached_base", "game_winner", "runs"], case_sensitive=False),
    default="reached_base",
    help="Label target (default: reached_base)",
)
@click.pass_context
def build(ctx: click.Context, game_pk: int | None, target: str) -> None:
    """Generate labels for the specified target."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)

    if target == "reached_base":
        stats = run_labels(engine, game_pk=game_pk)
        click.echo(
            f"Labels complete: {stats.games_processed} games, "
            f"{stats.rows_written} labels written, "
            f"{stats.elapsed_seconds}s elapsed"
        )
    elif target == "game_winner":
        from sandy.labels.runner import run_game_winner_labels

        stats = run_game_winner_labels(engine, game_pk=game_pk)
        click.echo(
            f"Game winner labels complete: {stats.games_processed} games, "
            f"{stats.rows_written} labels written, "
            f"{stats.elapsed_seconds}s elapsed"
        )
    elif target == "runs":
        from sandy.labels.runner import run_runs_labels

        stats = run_runs_labels(engine, game_pk=game_pk)
        click.echo(
            f"Runs labels complete: {stats.games_processed} games, "
            f"{stats.rows_written} labels written, "
            f"{stats.elapsed_seconds}s elapsed"
        )
