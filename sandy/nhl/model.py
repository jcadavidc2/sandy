"""NHL model + predictor + features: Dixon-Coles over REGULATION goals (ties are
real at regulation, which is exactly what the double-chance market needs), with
an exact OT adjustment for final-total markets: any regulation tie adds exactly
one more goal (OT winner, or the shootout's counted goal), so
    final_total(i,j) = i + j + 1{i == j}.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from zoneinfo import ZoneInfo

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.predictor import compute_scoreline_matrix
from sandy.football.ratings import DixonColesModel, fit_dixon_coles, load_model, save_model
from sandy.hyper import load_hyper

logger = logging.getLogger(__name__)

DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
XI = 0.0038                      # ~6-month half-life
TOTAL_THRESHOLDS = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
MAX_GOALS = 12
FORM_BLEND_WEIGHT = 0.2
# Defaults above; models/hyper_nhl.json overrides (walk-forward tuned).
# b2b_factor multiplies a team's lambda when it plays on back-to-back nights
# (1.0 = no fatigue adjustment).
HYPER_DEFAULTS = {"xi": XI, "blend": FORM_BLEND_WEIGHT, "b2b_factor": 1.0}


def hyper() -> dict:
    return load_hyper("nhl", HYPER_DEFAULTS)


def load_reg_games(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    where = "status = 'FINAL' AND reg_home_goals IS NOT NULL"
    params: dict = {}
    if as_of is not None:
        where += " AND match_date < :as_of"
        params["as_of"] = as_of
    with engine.begin() as conn:
        return pd.read_sql(text(f"""
            SELECT home_team_id, away_team_id, reg_home_goals AS home_goals,
                   reg_away_goals AS away_goals, match_date, 1.0 AS w
            FROM nhl.games WHERE {where} ORDER BY match_date
        """), conn, params=params)


def fit_goals(engine: Engine, as_of: date | None = None) -> DixonColesModel:
    df = load_reg_games(engine, as_of)
    model = fit_dixon_coles(df, as_of_date=as_of or date.today(), xi=hyper()["xi"])
    logger.info("NHL goals model fit: %s games, home_adv=%.3f", model.n_matches, model.home_adv)
    return model


def fit_and_persist(config: Config | None = None) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    model = fit_goals(engine)
    save_model(model, cfg.model.model_dir / "nhl_goals.json")
    with engine.begin() as conn:
        for tid in model.attack:
            conn.execute(text("""
                INSERT INTO nhl.team_ratings (team_id, as_of_date, attack, defense)
                VALUES (:t, :d, :a, :de)
                ON CONFLICT (team_id, as_of_date) DO UPDATE SET attack=:a, defense=:de
            """), {"t": tid, "d": model.as_of_date, "a": model.attack.get(tid),
                   "de": model.defense.get(tid)})
    return {"games": model.n_matches, "home_adv": round(model.home_adv, 3)}


def team_form(engine: Engine, team_id: int, as_of: date) -> dict:
    """Rolling covariates: last-10 final goals for/against, points, rest, back-to-back."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT match_date, home_team_id, home_goals, away_goals, last_period_type
            FROM nhl.games
            WHERE status = 'FINAL' AND home_goals IS NOT NULL
              AND (home_team_id = :t OR away_team_id = :t) AND match_date < :d
            ORDER BY match_date DESC LIMIT 10
        """), {"t": team_id, "d": as_of}).fetchall()
    if not rows:
        return {}
    gf, ga, pts = [], [], []
    for r in rows:
        is_home = r.home_team_id == team_id
        f = r.home_goals if is_home else r.away_goals
        a = r.away_goals if is_home else r.home_goals
        gf.append(f); ga.append(a)
        if f > a:
            pts.append(2)
        elif r.last_period_type in ("OT", "SO"):
            pts.append(1)
        else:
            pts.append(0)
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    rest = (as_of - rows[0].match_date).days
    return {"gf_5": mean(gf[:5]), "ga_5": mean(ga[:5]), "gf_10": mean(gf), "ga_10": mean(ga),
            "points_10": sum(pts), "rest_days": rest, "back_to_back": rest <= 1,
            "played_10": len(rows)}


def _blend(lam: float, form_gf: float | None, opp_ga: float | None,
           weight: float = FORM_BLEND_WEIGHT) -> float:
    if form_gf is None or opp_ga is None or weight <= 0:
        return lam
    return (1 - weight) * lam + weight * (form_gf + opp_ga) / 2.0


def markets(model: DixonColesModel, engine: Engine, home_id: int, away_id: int,
            as_of: date, with_features: bool = True) -> dict:
    hp = hyper()
    feats = {"home": team_form(engine, home_id, as_of),
             "away": team_form(engine, away_id, as_of)} if with_features else {}
    lam, mu = model.expected_goals(home_id, away_id)
    hf, af = feats.get("home") or {}, feats.get("away") or {}
    lam = _blend(lam, hf.get("gf_5"), af.get("ga_5"), hp["blend"])
    mu = _blend(mu, af.get("gf_5"), hf.get("ga_5"), hp["blend"])
    # Tuned back-to-back fatigue: shrink a tired team's scoring rate.
    if hp["b2b_factor"] != 1.0:
        if hf.get("back_to_back"):
            lam *= hp["b2b_factor"]
        if af.get("back_to_back"):
            mu *= hp["b2b_factor"]
    m = compute_scoreline_matrix(lam, mu, model.rho, max_goals=MAX_GOALS)
    idx = np.arange(m.shape[0])
    hi, aj = np.meshgrid(idx, idx, indexing="ij")
    final_total = hi + aj + (hi == aj).astype(int)   # regulation tie ⇒ exactly one OT/SO goal
    p_home = float(np.tril(m, -1).sum())
    p_tie = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    p_over = {t: float(m[final_total > t].sum()) for t in TOTAL_THRESHOLDS}
    mli, mlj = np.unravel_index(int(np.argmax(m)), m.shape)
    return {"lambda_home": round(lam, 3), "lambda_away": round(mu, 3),
            "p_home_win_reg": p_home, "p_tie_reg": p_tie, "p_away_win_reg": p_away,
            "p_home_or_tie": p_home + p_tie, "p_over": p_over,
            "most_likely": (int(mli), int(mlj)), "features": feats}


def persist_prediction(conn, game_id: int, mdate: date, hid: int, aid: int,
                       hname: str, aname: str, mk: dict, *, is_backtest: bool = False) -> None:
    conn.execute(text("""
        INSERT INTO nhl.game_predictions (
            game_id, match_date, home_team_id, away_team_id, home_team, away_team,
            lambda_home, lambda_away, p_home_win_reg, p_tie_reg, p_away_win_reg,
            p_home_or_tie, p_over_3_5, p_over_4_5, p_over_5_5, p_over_6_5, p_over_7_5, p_over_8_5,
            most_likely_home, most_likely_away,
            features, is_backtest, predicted_at_utc)
        VALUES (:gid, :d, :hid, :aid, :hn, :an, :lh, :la, :ph, :pt, :pa, :pht,
                :o35, :o45, :o55, :o65, :o75, :o85, :mlh, :mla, :feats, :bt, :now)
        ON CONFLICT (game_id) DO UPDATE SET
            lambda_home=:lh, lambda_away=:la, p_home_win_reg=:ph, p_tie_reg=:pt,
            p_away_win_reg=:pa, p_home_or_tie=:pht,
            p_over_3_5=:o35, p_over_4_5=:o45, p_over_5_5=:o55, p_over_6_5=:o65,
            p_over_7_5=:o75, p_over_8_5=:o85,
            most_likely_home=:mlh, most_likely_away=:mla, features=:feats,
            is_backtest=:bt, predicted_at_utc=:now
    """), {"gid": game_id, "d": mdate, "hid": hid, "aid": aid, "hn": hname, "an": aname,
           "lh": mk["lambda_home"], "la": mk["lambda_away"], "ph": mk["p_home_win_reg"],
           "pt": mk["p_tie_reg"], "pa": mk["p_away_win_reg"], "pht": mk["p_home_or_tie"],
           "o35": mk["p_over"][3.5], "o45": mk["p_over"][4.5], "o55": mk["p_over"][5.5],
           "o65": mk["p_over"][6.5], "o75": mk["p_over"][7.5], "o85": mk["p_over"][8.5],
           "mlh": mk["most_likely"][0], "mla": mk["most_likely"][1],
           "feats": json.dumps(mk["features"]) if mk.get("features") else None,
           "bt": is_backtest, "now": datetime.now(timezone.utc)})


def predict_scheduled(config: Config | None = None, *, days_ahead: int = 1) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    model = load_model(cfg.model.model_dir / "nhl_goals.json")
    today = datetime.now(DISPLAY_TZ).date()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT g.game_id, g.match_date, g.home_team_id, g.away_team_id, t1.abbrev, t2.abbrev
            FROM nhl.games g
            JOIN nhl.teams t1 ON t1.team_id = g.home_team_id
            JOIN nhl.teams t2 ON t2.team_id = g.away_team_id
            WHERE g.status = 'FUT' AND g.match_date BETWEEN :a AND :b
            ORDER BY g.match_date, g.game_id
        """), {"a": today, "b": today + timedelta(days=days_ahead)}).fetchall()
    n = 0
    with engine.begin() as conn:
        for gid, mdate, hid, aid, hab, aab in rows:
            mk = markets(model, engine, hid, aid, today)
            persist_prediction(conn, gid, mdate, hid, aid, hab, aab, mk)
            n += 1
    logger.info("NHL predicted %s scheduled games", n)
    return n
