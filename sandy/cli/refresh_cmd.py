"""CLI subcommand: sandy refresh — daily data refresh.

Phase 2, Task 15.1: Runs ingest incremental + labels build + features build
for all targets in sequence. Equivalent to the daily cron job.

Requirements: 16.1–16.6
"""
from __future__ import annotations

import sys
import time

import click

from sandy.cli._helpers import require_config
from sandy.db import bootstrap_schema, create_engine, get_connection
from sandy.logging import configure_logging, get_logger

logger = get_logger("cli.refresh")


@click.command()
@click.pass_context
def refresh(ctx: click.Context) -> None:
    """Run daily data refresh: ingest + labels + features for all targets."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    engine = create_engine(cfg)
    t0 = time.monotonic()

    # Ensure schema is up to date
    with get_connection(engine) as conn:
        bootstrap_schema(conn)

    click.echo("Starting daily refresh...")

    # Step 1: Incremental ingest
    click.echo("  [1/5] Ingesting new games...")
    try:
        from sandy.ingest.client import MlbStatsClient
        from sandy.ingest.service import incremental_ingest

        client = MlbStatsClient(cfg.ingest)
        ingest_stats = incremental_ingest(engine, client)
        click.echo(
            f"        {ingest_stats.games_added} added, "
            f"{ingest_stats.games_updated} updated, "
            f"{ingest_stats.games_skipped} skipped"
        )
    except Exception as exc:
        click.echo(f"        Failed: {exc}", err=True)
        # Retry once after 30 seconds
        click.echo("        Retrying in 30 seconds...")
        time.sleep(30)
        try:
            ingest_stats = incremental_ingest(engine, client)
            click.echo(f"        Retry succeeded: {ingest_stats.games_added} added")
        except Exception as exc2:
            click.echo(f"        Retry failed: {exc2}", err=True)
            sys.exit(1)

    # Step 2: Labels (reached_base)
    click.echo("  [2/5] Building reached_base labels...")
    from sandy.labels.runner import run_labels
    label_stats = run_labels(engine)
    click.echo(f"        {label_stats.rows_written} labels")

    # Step 3: Labels (game_winner + runs)
    click.echo("  [3/5] Building game_winner + runs labels...")
    from sandy.labels.runner import run_game_winner_labels, run_runs_labels
    gw_stats = run_game_winner_labels(engine)
    runs_stats = run_runs_labels(engine)
    click.echo(f"        {gw_stats.rows_written} game_winner, {runs_stats.rows_written} runs")

    # Step 4: Features (inning-level)
    click.echo("  [4/5] Building inning features...")
    from sandy.features.runner import run_features
    feat_stats = run_features(engine)
    click.echo(f"        {feat_stats.rows_written} inning features")

    # Step 5: Features (game-level)
    click.echo("  [5/5] Building game features...")
    from sandy.features.runner import run_game_features
    game_feat_stats = run_game_features(engine)
    click.echo(f"        {game_feat_stats.rows_written} game features")

    elapsed = round(time.monotonic() - t0, 1)
    click.echo(f"\nRefresh complete in {elapsed}s")

    # Also reconcile any outstanding predictions
    try:
        from sandy.evaluation.reconciler import reconcile_outcomes
        updated = reconcile_outcomes(engine)
        if updated > 0:
            click.echo(f"  Reconciled {updated} prediction outcomes")
    except Exception:
        pass  # Non-critical
