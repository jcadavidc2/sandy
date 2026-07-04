"""`sandy nhl ...` — NHL vertical commands."""
from __future__ import annotations

import click

from sandy.config import load_config


@click.group("nhl")
def nhl() -> None:
    """NHL prediction loop commands."""


@nhl.command("ingest")
def ingest_cmd() -> None:
    """Ingest yesterday/today/tomorrow from the official NHL API."""
    from sandy.nhl.ingest import ingest_recent_window
    click.echo(f"nhl ingest: {ingest_recent_window(load_config())} games upserted")


@nhl.command("backfill")
@click.option("--seasons", default="20222023,20232024,20242025,20252026", show_default=True)
def backfill_cmd(seasons) -> None:
    """Backfill seasons via per-team schedules (~33 requests/season)."""
    from sandy.nhl.ingest import backfill_seasons
    n = backfill_seasons(load_config(), seasons=[s.strip() for s in seasons.split(",")])
    click.echo(f"nhl backfill: {n} game-upserts")


@nhl.command("ratings")
def ratings_cmd() -> None:
    """Refit the regulation-goals Dixon-Coles model and persist."""
    from sandy.nhl.model import fit_and_persist
    res = fit_and_persist(load_config())
    click.echo(f"nhl ratings: n={res['games']} home_adv={res['home_adv']}")


@nhl.command("reconcile")
def reconcile_cmd() -> None:
    from sandy.nhl.loop import reconcile
    click.echo(f"nhl reconcile: {reconcile(load_config())} predictions scored")


@nhl.command("calibrate")
@click.option("--lookback-days", type=int, default=None)
def calibrate_cmd(lookback_days) -> None:
    from sandy.nhl.loop import calibrate
    for s in calibrate(load_config(), lookback_days=lookback_days):
        click.echo(f"{s['market']}: acc={s['accuracy']:.3f} n={s['sample_size']} "
                   f"thr={s['recommended_threshold']}")


@nhl.command("predict")
@click.option("--notify", is_flag=True, default=False)
@click.option("--days-ahead", type=int, default=1)
def predict_cmd(notify, days_ahead) -> None:
    from sandy.nhl.model import predict_scheduled
    n = predict_scheduled(load_config(), days_ahead=days_ahead)
    click.echo(f"nhl predict: {n} games")
    if notify:
        from sandy.nhl.loop import notify_daily
        click.echo(f"telegram sent: {notify_daily(load_config())}")


@nhl.command("backtest")
def backtest_cmd() -> None:
    from sandy.nhl.loop import run_backtest
    res = run_backtest(load_config())
    click.echo(f"nhl backtest: {res['predicted']} predicted, {res.get('reconciled')} reconciled")


@nhl.command("meta")
def meta_cmd() -> None:
    """Retrain the meta-model (P(pick correct)) and re-pick its threshold."""
    from sandy.betmeta import train_meta
    res = train_meta("nhl")
    click.echo(f"nhl meta: rows={res['rows']} threshold={res['threshold']}")
    for t in res["eval_table"]:
        click.echo(f"  meta>={t['thr']}: acc={t['acc']} n={t['n']}")
