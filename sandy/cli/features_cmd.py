"""CLI subcommand for feature building.

Task 12.4: sandy features build [--game-pk INT]
Phase 1.5: extended with --target flag for game-level features.

Requirements: 5.1, 10.2, 12.3
"""
from __future__ import annotations

import click

from sandy.cli._helpers import require_config
from sandy.db import create_engine
from sandy.features.runner import run_features
from sandy.logging import configure_logging


@click.group()
def features():
    """Manage derived features (inning_features, game_features)."""
    pass


@features.command()
@click.option("--game-pk", type=int, default=None, help="Process only this game_pk")
@click.option(
    "--target",
    type=click.Choice(["reached_base", "game_winner", "runs"], case_sensitive=False),
    default="reached_base",
    help="Feature target (default: reached_base). game_winner and runs both use game-level features.",
)
@click.pass_context
def build(ctx: click.Context, game_pk: int | None, target: str) -> None:
    """Build feature vectors for the specified target."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)

    if target == "reached_base":
        stats = run_features(engine, game_pk=game_pk)
        click.echo(
            f"Features complete: {stats.games_processed} games, "
            f"{stats.rows_written} rows written, {stats.rows_omitted} omitted, "
            f"{stats.elapsed_seconds}s elapsed"
        )
    else:
        # game_winner and runs both use game-level features
        from sandy.features.runner import run_game_features

        stats = run_game_features(engine, game_pk=game_pk)
        click.echo(
            f"Game features complete: {stats.games_processed} games, "
            f"{stats.rows_written} rows written, {stats.rows_omitted} omitted, "
            f"{stats.elapsed_seconds}s elapsed"
        )
