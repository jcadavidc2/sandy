"""`sandy soccer ...` — multi-league soccer vertical (ligas + copas)."""
from __future__ import annotations

import click

from sandy.config import load_config
from sandy.soccer import LEAGUES


@click.group("soccer")
def soccer() -> None:
    """Colombia / México / España / Inglaterra prediction loop."""


@soccer.command("ingest")
def ingest_cmd() -> None:
    from sandy.soccer.ingest import ingest_recent_window
    res = ingest_recent_window(load_config())
    click.echo(" ".join(f"{lg}:{v['matches']}m/{v['stats']}s" for lg, v in res.items()))


@soccer.command("backfill")
@click.option("--league", required=True, type=click.Choice(list(LEAGUES)))
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), required=True)
def backfill_cmd(league, start) -> None:
    from sandy.soccer.ingest import backfill
    res = backfill(load_config(), league=league, start=start.date())
    click.echo(f"soccer[{league}] backfill: {res['matches']} matches, {res['stats']} summaries")


@soccer.command("ratings")
def ratings_cmd() -> None:
    from sandy.soccer.loop import fit_all
    for lg, r in fit_all(load_config()).items():
        click.echo(f"{lg}: goals n={r['goals']} corners n={r['corners']}")


@soccer.command("reconcile")
def reconcile_cmd() -> None:
    from sandy.soccer.loop import reconcile
    click.echo(f"soccer reconcile: {reconcile(load_config())}")


@soccer.command("calibrate")
def calibrate_cmd() -> None:
    from sandy.soccer.loop import calibrate
    snaps = calibrate(load_config())
    click.echo(f"soccer calibrate: {len(snaps)} league-market snapshots")


@soccer.command("predict")
@click.option("--notify", is_flag=True, default=False)
def predict_cmd(notify) -> None:
    from sandy.soccer.loop import predict_scheduled
    click.echo(f"soccer predict: {predict_scheduled(load_config())}")
    if notify:
        from sandy.soccer.loop import notify_daily
        click.echo(f"telegram sent: {notify_daily(load_config())}")


@soccer.command("backtest")
@click.option("--league", "leagues", multiple=True, type=click.Choice(list(LEAGUES)),
              help="restrict to specific league(s); default all")
def backtest_cmd(leagues) -> None:
    from sandy.soccer.loop import run_backtest
    res = run_backtest(load_config(), leagues=tuple(leagues) or None)
    click.echo(f"soccer backtest: {res['predicted']} predicted, {res['reconciled']} reconciled")


@soccer.command("meta")
def meta_cmd() -> None:
    from sandy.betmeta import train_meta
    for lg in LEAGUES:  # ligas + copas — one meta artifact per competition
        try:
            res = train_meta(f"soccer_{lg}")
            click.echo(f"soccer_{lg} meta: rows={res['rows']} thr={res['threshold']} auc={res['auc']}")
        except RuntimeError as e:
            click.echo(f"soccer_{lg} meta: skipped ({e})")
