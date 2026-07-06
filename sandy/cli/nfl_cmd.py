"""`sandy nfl ...` — NFL vertical."""
from __future__ import annotations

import click

from sandy.config import load_config


@click.group("nfl")
def nfl() -> None:
    """NFL prediction loop commands."""


@nfl.command("ingest")
def ingest_cmd() -> None:
    from sandy.nfl.loop import ingest_recent_window
    click.echo(f"nfl ingest: {ingest_recent_window(load_config())} games")


@nfl.command("backfill")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), required=True)
def backfill_cmd(start) -> None:
    from sandy.nfl.loop import backfill
    click.echo(f"nfl backfill: {backfill(load_config(), start=start.date())} upserts")


@nfl.command("ratings")
def ratings_cmd() -> None:
    from sandy.nfl.loop import fit_and_persist
    res = fit_and_persist(load_config())
    click.echo(f"nfl ratings: n={res['games']} mu={res['mu']} hfa={res['hfa']} sigma={res['sigma_total']}")


@nfl.command("reconcile")
def reconcile_cmd() -> None:
    from sandy.nfl.loop import reconcile
    click.echo(f"nfl reconcile: {reconcile(load_config())}")


@nfl.command("calibrate")
def calibrate_cmd() -> None:
    from sandy.nfl.loop import calibrate
    for s in calibrate(load_config()):
        click.echo(f"{s['market']}: acc={s['accuracy']:.3f} n={s['sample_size']}")


@nfl.command("predict")
@click.option("--notify", is_flag=True, default=False)
def predict_cmd(notify) -> None:
    from sandy.nfl.loop import predict_scheduled
    click.echo(f"nfl predict: {predict_scheduled(load_config())}")
    if notify:
        from sandy.nfl.loop import notify_daily
        click.echo(f"telegram sent: {notify_daily(load_config())}")


@nfl.command("backtest")
def backtest_cmd() -> None:
    from sandy.nfl.loop import run_backtest
    res = run_backtest(load_config())
    click.echo(f"nfl backtest: {res['predicted']} predicted, {res['reconciled']} reconciled")


@nfl.command("meta")
def meta_cmd() -> None:
    from sandy.betmeta import train_meta
    res = train_meta("nfl")
    click.echo(f"nfl meta: rows={res['rows']} thr={res['threshold']} auc={res['auc']}")
