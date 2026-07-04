"""MLS models: two Dixon-Coles fits reusing the football vertical's generic MLE —
one over GOALS, one over CORNERS (corners behave like a high-rate Poisson; the
DC low-score correction fits ~0 there and is harmless). Home advantage ON.
Artifacts: models/mls_goals.json + models/mls_corners.json.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.ratings import DixonColesModel, fit_dixon_coles, load_model, save_model

logger = logging.getLogger(__name__)

GOALS_XI = 0.0038   # ~6-month half-life: club form moves faster than national teams
CORNERS_XI = 0.0038


def _load_df(engine: Engine, cols: str, extra_where: str = "", as_of: date | None = None) -> pd.DataFrame:
    where = "status = 'FT'" + extra_where
    params: dict = {}
    if as_of is not None:
        where += " AND match_date < :as_of"
        params["as_of"] = as_of
    sql = f"""
        SELECT home_team_id, away_team_id, {cols}, match_date, 1.0 AS w
        FROM mls.matches WHERE {where}
        ORDER BY match_date
    """
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def load_goal_matches(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    df = _load_df(engine, "home_goals, away_goals",
                  " AND home_goals IS NOT NULL AND away_goals IS NOT NULL", as_of)
    return df


def load_corner_matches(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    df = _load_df(engine, "home_corners AS home_goals, away_corners AS away_goals",
                  " AND home_corners IS NOT NULL AND away_corners IS NOT NULL", as_of)
    return df


def fit_goals(engine: Engine, as_of: date | None = None) -> DixonColesModel:
    df = load_goal_matches(engine, as_of)
    model = fit_dixon_coles(df, as_of_date=as_of or date.today(), xi=GOALS_XI)
    logger.info("MLS goals model fit: %s matches, home_adv=%.3f rho=%.3f",
                model.n_matches, model.home_adv, model.rho)
    return model


def fit_corners(engine: Engine, as_of: date | None = None) -> DixonColesModel:
    df = load_corner_matches(engine, as_of)
    model = fit_dixon_coles(df, as_of_date=as_of or date.today(), xi=CORNERS_XI)
    logger.info("MLS corners model fit: %s matches, home_adv=%.3f",
                model.n_matches, model.home_adv)
    return model


def _persist_ratings(engine: Engine, goals: DixonColesModel, corners: DixonColesModel) -> None:
    as_of = goals.as_of_date
    with engine.begin() as conn:
        for tid in set(goals.attack) | set(corners.attack):
            conn.execute(text("""
                INSERT INTO mls.team_ratings (team_id, as_of_date, attack, defense, corner_attack, corner_defense)
                VALUES (:t, :d, :a, :de, :ca, :cd)
                ON CONFLICT (team_id, as_of_date) DO UPDATE SET attack=:a, defense=:de,
                    corner_attack=:ca, corner_defense=:cd
            """), {"t": tid, "d": as_of, "a": goals.attack.get(tid), "de": goals.defense.get(tid),
                   "ca": corners.attack.get(tid), "cd": corners.defense.get(tid)})


def fit_and_persist(config: Config | None = None) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    goals = fit_goals(engine)
    corners = fit_corners(engine)
    gpath = cfg.model.model_dir / "mls_goals.json"
    cpath = cfg.model.model_dir / "mls_corners.json"
    save_model(goals, gpath)
    save_model(corners, cpath)
    _persist_ratings(engine, goals, corners)
    return {"goals_matches": goals.n_matches, "corners_matches": corners.n_matches,
            "home_adv": round(goals.home_adv, 3)}


def load_models(cfg: Config) -> tuple[DixonColesModel, DixonColesModel]:
    return (load_model(cfg.model.model_dir / "mls_goals.json"),
            load_model(cfg.model.model_dir / "mls_corners.json"))
