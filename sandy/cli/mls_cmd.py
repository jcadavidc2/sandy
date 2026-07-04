"""`sandy mls ...` — MLS vertical commands (mirrors `sandy football ...`)."""
from __future__ import annotations

from datetime import date

import click

from sandy.config import load_config


@click.group("mls")
def mls() -> None:
    """MLS prediction loop commands."""


@mls.command("ingest")
def ingest_cmd() -> None:
    """Ingest yesterday/today/tomorrow + trickle match stats (corners)."""
    from sandy.mls.ingest import ingest_recent_window
    res = ingest_recent_window(load_config())
    click.echo(f"mls ingest: {res['matches']} matches, {res['stats']} summaries")


@mls.command("backfill")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), required=True)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--with-stats/--no-stats", default=True)
def backfill_cmd(start, end, with_stats) -> None:
    """Walk historical scoreboard dates (skips January)."""
    from sandy.mls.ingest import backfill
    res = backfill(load_config(), start=start.date(), end=end.date() if end else None,
                   with_stats=with_stats)
    click.echo(f"mls backfill: {res['matches']} match-upserts, {res['stats']} summaries")


@mls.command("ratings")
def ratings_cmd() -> None:
    """Refit goals + corners Dixon-Coles models and persist."""
    from sandy.mls.ratings import fit_and_persist
    res = fit_and_persist(load_config())
    click.echo(f"mls ratings: goals n={res['goals_matches']} corners n={res['corners_matches']} "
               f"home_adv={res['home_adv']}")


@mls.command("reconcile")
def reconcile_cmd() -> None:
    """Fill actuals + was_correct_* for finished matches."""
    from sandy.mls.reconciler import reconcile
    click.echo(f"mls reconcile: {reconcile(load_config())} predictions scored")


@mls.command("calibrate")
@click.option("--lookback-days", type=int, default=None)
def calibrate_cmd(lookback_days) -> None:
    """Recompute per-market calibration snapshots."""
    from sandy.mls.calibrator import calibrate
    snaps = calibrate(load_config(), lookback_days=lookback_days)
    for s in snaps:
        click.echo(f"{s['market']}: acc={s['accuracy']:.3f} n={s['sample_size']} "
                   f"thr={s['recommended_threshold']}")


@mls.command("predict")
@click.option("--notify", is_flag=True, default=False)
@click.option("--days-ahead", type=int, default=2)
def predict_cmd(notify, days_ahead) -> None:
    """Predict scheduled matches; optionally send the ⚽ Telegram digest."""
    from sandy.mls.predictor import predict_scheduled
    preds = predict_scheduled(load_config(), days_ahead=days_ahead)
    click.echo(f"mls predict: {len(preds)} matches")
    if notify:
        from sandy.mls.notifier import notify_daily
        click.echo(f"telegram sent: {notify_daily(load_config())}")


@mls.command("backtest")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default=None)
def backtest_cmd(start) -> None:
    """Walk-forward backtest to seed calibration (leakage-free)."""
    from sandy.mls.backtest import run_backtest
    res = run_backtest(load_config(), start=start.date() if start else None)
    click.echo(f"mls backtest: {res['predicted']} predicted, {res.get('reconciled')} reconciled")
