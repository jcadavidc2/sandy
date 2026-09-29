"""`sandy nba ...` — NBA vertical."""
from __future__ import annotations

import click

from sandy.config import load_config


@click.group("nba")
def nba() -> None:
    """NBA prediction loop commands."""


@nba.command("ingest")
def ingest_cmd() -> None:
    from sandy.nba.loop import ingest_recent_window
    click.echo(f"nba ingest: {ingest_recent_window(load_config())} games")


@nba.command("backfill")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), required=True)
def backfill_cmd(start) -> None:
    from sandy.nba.loop import backfill
    click.echo(f"nba backfill: {backfill(load_config(), start=start.date())} upserts")


@nba.command("ratings")
def ratings_cmd() -> None:
    from sandy.nba.loop import fit_and_persist
    res = fit_and_persist(load_config())
    click.echo(f"nba ratings: n={res['games']} mu={res['mu']} hfa={res['hfa']} sigma={res['sigma_total']}")


@nba.command("reconcile")
def reconcile_cmd() -> None:
    from sandy.nba.loop import reconcile
    click.echo(f"nba reconcile: {reconcile(load_config())}")


@nba.command("calibrate")
def calibrate_cmd() -> None:
    from sandy.nba.loop import calibrate
    for s in calibrate(load_config()):
        click.echo(f"{s['market']}: acc={s['accuracy']:.3f} n={s['sample_size']}")


@nba.command("predict")
@click.option("--notify", is_flag=True, default=False)
def predict_cmd(notify) -> None:
    from sandy.nba.loop import predict_scheduled
    click.echo(f"nba predict: {predict_scheduled(load_config())}")

    # Calibrated probability columns for the rows just written (walk-forward
    # isotonic per line — sandy/calibrate_lines.py). Non-fatal.
    try:
        from sandy.calibrate_lines import fill_league
        from sandy.db import create_engine as _ce
        _eng = _ce(load_config())
        for _lg in ("nba",):
            fill_league(_eng, _lg)
    except Exception as _exc:  # noqa: BLE001
        click.echo(f"calibrate_lines skipped: {_exc}")
    if notify:
        from sandy.nba.loop import notify_daily
        click.echo(f"telegram sent: {notify_daily(load_config())}")


@nba.command("backtest")
def backtest_cmd() -> None:
    from sandy.nba.loop import run_backtest
    res = run_backtest(load_config())
    click.echo(f"nba backtest: {res['predicted']} predicted, {res['reconciled']} reconciled")


@nba.command("meta")
def meta_cmd() -> None:
    from sandy.betmeta import train_meta
    res = train_meta("nba")
    click.echo(f"nba meta: rows={res['rows']} thr={res['threshold']} auc={res['auc']}")
