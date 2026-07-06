"""Walk-forward backtest: replay history to seed calibration.

International results accrue slowly, so we bootstrap the calibration dataset by
replaying completed matches as if they were unknown: for each match we fit the
Dixon-Coles model on ONLY prior data (strict as-of cutoff — no leakage),
predict, persist, then reconcile against the known result. This is exactly the
"predict the previous World Cup" idea, generalized to the whole 2022-2026 span.

Refitting per match would be slow, so the model is refit every
``refit_every_days`` of match calendar and reused for matches in between (still
leakage-free: the model's as-of date never exceeds the match date).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.predictor import build_prediction, persist_predictions
from sandy.football.ratings import fit_dixon_coles, hyper, load_finished_matches
from sandy.football.reconciler import reconcile_predictions
from sandy.logging import get_logger

logger = get_logger("football.backtest")

# Competitions played at neutral venues (no home advantage in the prediction).
NEUTRAL_COMPETITIONS = frozenset({"World Cup"})


@dataclass
class BacktestResult:
    predicted: int
    reconciled: int
    refits: int
    start: date
    end: date


def _load_matches(engine, start: date, end: date) -> pd.DataFrame:
    sql = text("""
        SELECT m.fixture_id, m.match_date, m.kickoff_utc, m.competition,
               m.home_team_id, m.away_team_id, th.name AS home_name, ta.name AS away_name
        FROM football.matches m
        JOIN football.teams th ON th.team_id = m.home_team_id
        JOIN football.teams ta ON ta.team_id = m.away_team_id
        WHERE m.status IN ('FT','AET','PEN')
          AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
          AND m.match_date >= :start AND m.match_date <= :end
        ORDER BY m.match_date, m.fixture_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start": start, "end": end}).fetchall()
    return pd.DataFrame(rows, columns=[
        "fixture_id", "match_date", "kickoff_utc", "competition",
        "home_team_id", "away_team_id", "home_name", "away_name",
    ])


def run_backtest(
    config: Config | None = None,
    *,
    start: date = date(2022, 6, 1),
    end: date = date(2026, 6, 15),
    refit_every_days: int = 30,
    min_train: int = 300,
) -> BacktestResult:
    """Walk forward over [start, end], predicting each match leakage-free."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    matches = _load_matches(engine, start, end)

    model = None
    last_fit: date | None = None
    refits = 0
    preds = []

    for row in matches.itertuples(index=False):
        d = row.match_date
        if last_fit is None or (d - last_fit).days >= refit_every_days:
            train = load_finished_matches(engine, as_of=d)
            if len(train) >= min_train:
                model = fit_dixon_coles(train, as_of_date=d, xi=hyper()["xi"])
                last_fit = d
                refits += 1
        if model is None:
            continue
        neutral = row.competition in NEUTRAL_COMPETITIONS
        preds.append(build_prediction(
            model, fixture_id=row.fixture_id, match_date=d, kickoff_utc=row.kickoff_utc,
            home_team_id=row.home_team_id, away_team_id=row.away_team_id,
            home_name=row.home_name, away_name=row.away_name, neutral=neutral,
        ))

    n_pred = persist_predictions(engine, preds) if preds else 0
    n_rec = reconcile_predictions(engine)
    logger.info("Backtest complete", extra={
        "component": "football.backtest", "predicted": n_pred,
        "reconciled": n_rec, "refits": refits,
    })
    return BacktestResult(predicted=n_pred, reconciled=n_rec, refits=refits,
                          start=start, end=end)


__all__ = ["BacktestResult", "run_backtest"]
