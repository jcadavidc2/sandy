"""`sandy football` CLI group — the daily World Cup loop.

Subcommands mirror the over-under group so the nightly script can call them in
sequence:

    sandy football ingest      # pull the +/-1 day fixture window (PST-correct)
    sandy football ratings     # refit Dixon-Coles on all data
    sandy football reconcile   # fill actuals + was_correct for finished games
    sandy football calibrate   # recompute calibration snapshots
    sandy football predict     # predict upcoming fixtures (+ --notify Telegram)
    sandy football backtest    # one-off walk-forward replay (manual)
"""
from __future__ import annotations

from datetime import date

import click

from sandy.config import load_config
from sandy.db import create_engine


@click.group("football")
def football() -> None:
    """World Cup prediction loop commands."""


@football.command("ingest")
def ingest_cmd() -> None:
    """Ingest the free-tier +/-1 day fixture window (yesterday/today/tomorrow)."""
    from sandy.football.ingest import ingest_recent_window
    cfg = load_config()
    total = sum(s.matches_upserted for s in ingest_recent_window(cfg))
    click.echo(f"football ingest: {total} matches upserted across the window")


@football.command("ratings")
def ratings_cmd() -> None:
    """Refit the Dixon-Coles model on all finished matches and persist it."""
    from sandy.football.ratings import fit_and_persist
    cfg = load_config()
    model = fit_and_persist(cfg, as_of=date.today())
    click.echo(f"football ratings: fit {model.n_matches} matches, "
               f"home_adv={model.home_adv:.3f} rho={model.rho:.3f}")


@football.command("reconcile")
@click.option("--notify", is_flag=True, help="(reserved) send a Telegram note")
def reconcile_cmd(notify: bool) -> None:
    """Fill actuals + was_correct for finished, unreconciled predictions."""
    from sandy.football.reconciler import reconcile
    n = reconcile(load_config())
    click.echo(f"football reconcile: {n} predictions reconciled")


@football.command("calibrate")
@click.option("--notify", is_flag=True, help="(reserved) send a Telegram note")
def calibrate_cmd(notify: bool) -> None:
    """Recompute and persist calibration snapshots."""
    from sandy.football.calibrator import calibrate
    snaps = calibrate(load_config())
    for s in snaps:
        click.echo(f"  {s['market']}: acc={s['accuracy']*100:.1f}% n={s['sample_size']} "
                   f"rec_threshold={s['recommended_threshold']}")
    click.echo(f"football calibrate: {len(snaps)} snapshots written")


@football.command("predict")
@click.option("--notify", is_flag=True, help="Send the digest to Telegram")
def predict_cmd(notify: bool) -> None:
    """Predict upcoming fixtures and print (optionally send) the daily digest."""
    from sandy.football.predictor import predict_scheduled
    from sandy.football.notifier import format_daily_digest, send_telegram
    from sandy.football.queries import (
        get_latest_calibration, get_recent_results, get_today_predictions,
    )
    cfg = load_config()
    predict_scheduled(cfg, upcoming_only=True)

    engine = create_engine(cfg)
    preds = get_today_predictions(engine, cfg)
    cal = get_latest_calibration(engine)
    results = get_recent_results(engine, cfg, days=1)
    msg = format_daily_digest(preds, cal, results)
    click.echo(msg)
    if notify:
        ok = send_telegram(msg)
        click.echo(f"\n[telegram sent: {ok}]")


@football.command("backtest")
@click.option("--start", default="2022-06-01")
@click.option("--end", default=None)
@click.option("--refit-every-days", default=30, type=int)
def backtest_cmd(start: str, end: str | None, refit_every_days: int) -> None:
    """One-off walk-forward backtest to (re)seed calibration data."""
    from sandy.football.backtest import run_backtest
    cfg = load_config()
    res = run_backtest(
        cfg, start=date.fromisoformat(start),
        end=date.fromisoformat(end) if end else date.today(),
        refit_every_days=refit_every_days,
    )
    click.echo(f"backtest: predicted={res.predicted} reconciled={res.reconciled} refits={res.refits}")


__all__ = ["football"]
