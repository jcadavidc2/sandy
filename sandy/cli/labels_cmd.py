"""CLI subcommand for label generation.

Task 12.3: sandy labels build [--game-pk INT]

Requirements: 4.1, 10.2
"""
from __future__ import annotations

import click

from sandy.cli.main import _require_config
from sandy.db import create_engine
from sandy.labels.runner import run_labels
from sandy.logging import configure_logging


@click.group()
def labels():
    """Manage derived.inning_labels."""
    pass


@labels.command()
@click.option("--game-pk", type=int, default=None, help="Process only this game_pk")
@click.pass_context
def build(ctx: click.Context, game_pk: int | None) -> None:
    """Generate reached_base labels for all Final games."""
    cfg = _require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
    stats = run_labels(engine, game_pk=game_pk)

    click.echo(
        f"Labels complete: {stats.games_processed} games, "
        f"{stats.rows_written} labels written, "
        f"{stats.elapsed_seconds}s elapsed"
    )
