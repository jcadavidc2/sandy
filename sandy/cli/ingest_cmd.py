"""CLI subcommands for data ingestion.

Task 12.2: sandy ingest backfill / sandy ingest incremental

Requirements: 1.1, 2.1, 10.2
"""
from __future__ import annotations

import click

from sandy.cli.main import _require_config
from sandy.db import bootstrap_schema, create_engine, get_connection
from sandy.ingest.client import MlbStatsClient
from sandy.ingest.service import backfill_seasons, incremental_ingest
from sandy.logging import configure_logging


@click.group()
def ingest():
    """Fetch data from MLB Stats API."""
    pass


@ingest.command()
@click.option("--seasons", type=int, default=3, help="Number of seasons to backfill (default 3)")
@click.option("--start-season", type=int, default=None, help="Earliest season year (default: auto)")
@click.pass_context
def backfill(ctx: click.Context, seasons: int, start_season: int | None) -> None:
    """Backfill historical regular-season data from MLB Stats API."""
    cfg = _require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
    with get_connection(engine) as conn:
        bootstrap_schema(conn)

    client = MlbStatsClient(cfg.ingest)

    # Determine season list
    from datetime import date
    current_year = date.today().year
    if start_season is not None:
        season_list = list(range(start_season, start_season + seasons))
    else:
        last_complete = current_year - 1
        season_list = list(range(last_complete - seasons + 1, last_complete + 1))

    with get_connection(engine) as conn:
        stats = backfill_seasons(conn, client, seasons=season_list)

    click.echo(
        f"Backfill complete: {stats.games_processed} processed, "
        f"{stats.games_skipped} skipped, {stats.games_failed} failed, "
        f"{stats.elapsed_seconds}s elapsed"
    )


@ingest.command()
@click.pass_context
def incremental(ctx: click.Context) -> None:
    """Fetch new/updated games since last ingestion."""
    cfg = _require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
    client = MlbStatsClient(cfg.ingest)

    with get_connection(engine) as conn:
        stats = incremental_ingest(conn, client)

    click.echo(
        f"Incremental complete: {stats.games_added} added, "
        f"{stats.games_updated} updated, {stats.games_skipped} skipped, "
        f"{stats.elapsed_seconds}s elapsed"
    )
