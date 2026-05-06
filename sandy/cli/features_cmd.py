"""CLI subcommand for feature building.

Task 12.4: sandy features build [--game-pk INT]

Requirements: 5.1, 10.2
"""
from __future__ import annotations

import click

from sandy.cli.main import _require_config
from sandy.db import create_engine
from sandy.features.runner import run_features
from sandy.logging import configure_logging


@click.group()
def features():
    """Manage derived.inning_features."""
    pass


@features.command()
@click.option("--game-pk", type=int, default=None, help="Process only this game_pk")
@click.pass_context
def build(ctx: click.Context, game_pk: int | None) -> None:
    """Build feature vectors for all labeled innings."""
    cfg = _require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
    stats = run_features(engine, game_pk=game_pk)

    click.echo(
        f"Features complete: {stats.games_processed} games, "
        f"{stats.rows_written} rows written, {stats.rows_omitted} omitted, "
        f"{stats.elapsed_seconds}s elapsed"
    )
