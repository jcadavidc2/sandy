"""CLI commands for the Over/Under Feedback Loop.

Provides three subcommands under the 'over-under' group:
- predict: Run predictions for all games on a given date
- reconcile: Fill actual outcomes for finished games
- calibrate: Compute calibration metrics

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""
from __future__ import annotations

import sys
from datetime import date, datetime

import click

from sandy.logging import get_logger

logger = get_logger("cli.over_under")


@click.group("over-under")
def over_under():
    """Over/under prediction feedback loop commands."""
    pass


@over_under.command("predict")
@click.option("--date", "date_str", type=str, default=None, help="Game date (YYYY-MM-DD). Defaults to today.")
@click.option("--notify", is_flag=True, help="Send Telegram notification with results.")
@click.pass_context
def predict_cmd(ctx: click.Context, date_str: str | None, notify: bool) -> None:
    """Run over/under predictions for all scheduled games."""
    from sandy.cli.main import _require_config
    from sandy.db import create_engine
    from sandy.over_under.notifier import (
        format_morning_digest,
        format_no_games_message,
        send_telegram,
    )
    from sandy.over_under.predictor import persist_predictions, predict_all_games

    config = _require_config(ctx)
    game_date = _parse_date(date_str)

    click.echo(f"Running over/under predictions for {game_date}...")

    predictions = predict_all_games(config, game_date=game_date)

    if not predictions:
        msg = format_no_games_message()
        click.echo(msg)
        if notify:
            send_telegram(msg)
        return

    # Persist to database
    engine = create_engine(config)
    count = persist_predictions(engine, predictions)
    click.echo(f"Persisted {count} predictions.")

    # Load latest calibration for trust signal
    calibration = None
    try:
        from sandy.over_under.calibrator import compute_calibration
        calibration = compute_calibration(engine)
    except Exception:
        pass

    # Format and display
    for pred in sorted(predictions, key=lambda p: p.game_time_utc):
        p_6_5 = pred.p_over.get(6.5, 0.0)
        fb = " (fallback)" if pred.pitcher_fallback else ""
        click.echo(
            f"  {pred.home_team_code} vs {pred.away_team_code}: "
            f"P(over 6.5) = {p_6_5:.1%}{fb}"
        )

    if notify:
        message = format_morning_digest(predictions, calibration)
        send_telegram(message)
        click.echo("Telegram notification sent.")


@over_under.command("reconcile")
@click.option("--date", "date_str", type=str, default=None, help="Game date (YYYY-MM-DD). Defaults to today.")
@click.option("--notify", is_flag=True, help="Send Telegram notification with results.")
@click.pass_context
def reconcile_cmd(ctx: click.Context, date_str: str | None, notify: bool) -> None:
    """Reconcile actual outcomes for finished games."""
    from sandy.cli.main import _require_config
    from sandy.db import create_engine
    from sandy.over_under.notifier import (
        format_nightly_report,
        format_no_finals_message,
        send_telegram,
    )
    from sandy.over_under.reconciler import reconcile_over_under

    config = _require_config(ctx)
    engine = create_engine(config)

    click.echo("Reconciling over/under outcomes...")
    updated = reconcile_over_under(engine)
    click.echo(f"Updated {updated} outcomes.")

    if updated == 0:
        msg = format_no_finals_message()
        click.echo(msg)
        if notify:
            send_telegram(msg)
        return

    if notify:
        # Fetch reconciled outcomes for the report
        from sqlalchemy import text as sql_text

        with engine.connect() as conn:
            game_date = _parse_date(date_str)
            rows = conn.execute(
                sql_text("""
                    SELECT home_team_code, away_team_code,
                           p_over_6_5, actual_total_runs, was_correct_6_5,
                           feature_vector, home_starter_era, away_starter_era,
                           ballpark_id, home_trailing15_rpg, away_trailing15_rpg
                    FROM derived.over_under_outcomes
                    WHERE game_date = :game_date
                      AND actual_total_runs IS NOT NULL
                    ORDER BY game_pk
                """),
                {"game_date": game_date},
            ).fetchall()

        outcomes = [
            {
                "home_team_code": r[0].strip() if r[0] else "",
                "away_team_code": r[1].strip() if r[1] else "",
                "p_over_6_5": float(r[2]) if r[2] else 0.0,
                "actual_total_runs": int(r[3]) if r[3] is not None else None,
                "was_correct_6_5": r[4],
                "feature_vector": r[5],
                "home_starter_era": r[6],
                "away_starter_era": r[7],
                "ballpark_id": r[8],
                "home_trailing15_rpg": r[9],
                "away_trailing15_rpg": r[10],
            }
            for r in rows
        ]

        message = format_nightly_report(outcomes, None, None)
        send_telegram(message)
        click.echo("Telegram notification sent.")


@over_under.command("calibrate")
@click.option("--notify", is_flag=True, help="Send Telegram notification with results.")
@click.option("--weekly", is_flag=True, help="Run deeper weekly analysis with 4-week trends.")
@click.pass_context
def calibrate_cmd(ctx: click.Context, notify: bool, weekly: bool) -> None:
    """Compute calibration metrics from recent outcomes."""
    from sandy.cli.main import _require_config
    from sandy.db import create_engine
    from sandy.over_under.calibrator import compute_calibration, persist_calibration
    from sandy.over_under.notifier import send_telegram

    config = _require_config(ctx)
    engine = create_engine(config)

    lookback = 28 if weekly else 7
    click.echo(f"Computing calibration (lookback={lookback} days)...")

    snapshot = compute_calibration(engine, lookback_days=lookback)

    if snapshot is None:
        click.echo("Insufficient data for calibration (< 5 reconciled predictions).")
        if notify:
            send_telegram("📊 Over/under calibration: insufficient data (< 5 predictions).")
        return

    persist_calibration(engine, snapshot)

    click.echo(f"Calibration complete:")
    click.echo(f"  Sample size: {snapshot.sample_size}")
    click.echo(f"  Recommended threshold: {snapshot.recommended_threshold}")
    for t, acc in sorted(snapshot.accuracy_by_threshold.items()):
        click.echo(f"  Accuracy at {t}: {acc:.1%}")

    if snapshot.rolling_4w_accuracy is not None:
        click.echo(f"  Rolling 4-week accuracy (6.5): {snapshot.rolling_4w_accuracy:.1%}")

    if notify:
        msg_lines = [
            f"📊 Over/Under Calibration ({'Weekly' if weekly else 'Daily'})",
            f"Sample: {snapshot.sample_size} games, Lookback: {lookback} days",
            f"Recommended threshold: {snapshot.recommended_threshold}",
        ]
        for t, acc in sorted(snapshot.accuracy_by_threshold.items()):
            msg_lines.append(f"  {t}: {acc:.0%}")
        if snapshot.rolling_4w_accuracy is not None:
            msg_lines.append(f"Rolling 4-week (6.5): {snapshot.rolling_4w_accuracy:.0%}")
        send_telegram("\n".join(msg_lines))
        click.echo("Telegram notification sent.")


def _parse_date(date_str: str | None) -> date:
    """Parse a date string or return today."""
    if date_str is None:
        return date.today()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid date format: {date_str}. Use YYYY-MM-DD.", err=True)
        sys.exit(2)


__all__ = ["over_under"]
