"""Walk-forward backtest for MLS: refit both models every REFIT_DAYS of calendar,
predict the next block leakage-free, persist as is_backtest rows, reconcile —
this seeds calibration with years of scored predictions before day one."""
from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.ratings import fit_dixon_coles

from .predictor import build_prediction, persist_predictions, stamp_playoff_covariates
from .ratings import hyper, load_corner_matches, load_goal_matches
from .reconciler import reconcile

logger = logging.getLogger(__name__)

REFIT_DAYS = 14
MIN_TRAIN_MATCHES = 300


def run_backtest(config: Config | None = None, *, start: date | None = None,
                 end: date | None = None, with_features: bool = True) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    with engine.begin() as conn:
        lo, hi = conn.execute(text(
            "SELECT MIN(match_date), MAX(match_date) FROM mls.matches WHERE status='FT'"
        )).fetchone()
    if lo is None:
        return {"predicted": 0}
    start = start or (lo + timedelta(days=270))  # need a season of history before first predict
    end = end or hi
    predicted = 0
    block_start = start
    while block_start <= end:
        block_end = min(block_start + timedelta(days=REFIT_DAYS - 1), end)
        gdf = load_goal_matches(engine, as_of=block_start)
        if len(gdf) < MIN_TRAIN_MATCHES:
            block_start = block_end + timedelta(days=1)
            continue
        hp = hyper()
        goals = fit_dixon_coles(gdf, as_of_date=block_start, xi=hp["xi_goals"])
        cdf = load_corner_matches(engine, as_of=block_start)
        corners = fit_dixon_coles(cdf, as_of_date=block_start, xi=hp["xi_corners"]) if len(cdf) >= MIN_TRAIN_MATCHES else goals
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT m.event_id, m.match_date, m.home_team_id, m.away_team_id, t1.name, t2.name
                FROM mls.matches m
                JOIN mls.teams t1 ON t1.team_id = m.home_team_id
                JOIN mls.teams t2 ON t2.team_id = m.away_team_id
                WHERE m.status = 'FT' AND m.match_date BETWEEN :a AND :b
                ORDER BY m.match_date
            """), {"a": block_start, "b": block_end}).fetchall()
        preds = [build_prediction(engine, goals, corners, r, r[1],
                                  is_backtest=True, with_features=with_features)
                 for r in rows]
        persist_predictions(engine, preds)
        predicted += len(preds)
        logger.info("MLS backtest block %s→%s: %s predictions (train n=%s)",
                    block_start, block_end, len(preds), len(gdf))
        block_start = block_end + timedelta(days=1)
    stamp_playoff_covariates(engine)
    reconciled = reconcile(cfg)
    logger.info("MLS backtest done: %s predictions, %s reconciled", predicted, reconciled)
    return {"predicted": predicted, "reconciled": reconciled}
