"""CLI subcommand: sandy today — show today's MLB schedule.

Phase 1.5, Task 15.2: Fetches and displays today's schedule as a
formatted table with game times, teams, and probable pitchers.

Requirements: 9.1–9.5
"""
from __future__ import annotations

import sys

import click

from sandy.cli._helpers import require_config
from sandy.logging import configure_logging


@click.command()
@click.pass_context
def today(ctx: click.Context) -> None:
    """Show today's MLB schedule with probable pitchers."""
    cfg = require_config(ctx)
    configure_logging(cfg.logging.level)

    try:
        from sandy.schedule.client import get_todays_schedule

        schedule = get_todays_schedule(cfg)
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not schedule:
        click.echo("No MLB games scheduled for today.")
        return

    # Format as table
    header = f"{'Time (UTC)':<12} {'Away':<5} {'Home':<5} {'Away Pitcher':<22} {'Home Pitcher':<22} {'Status'}"
    click.echo(header)
    click.echo("-" * len(header))

    for game in sorted(schedule, key=lambda g: g.game_time_utc):
        time_str = game.game_time_utc.strftime("%H:%M")
        away_pitcher = game.away_probable_pitcher or "TBD"
        home_pitcher = game.home_probable_pitcher or "TBD"

        click.echo(
            f"{time_str:<12} "
            f"{game.away_team_code:<5} "
            f"{game.home_team_code:<5} "
            f"{away_pitcher:<22} "
            f"{home_pitcher:<22} "
            f"{game.status}"
        )
