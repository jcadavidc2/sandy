"""MLS predictor: goals scoreline matrix (result + totals + double-chance) plus a
corners matrix for corner totals. Persists to mls.match_predictions with the
covariates snapshot."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.predictor import compute_scoreline_matrix, markets_from_matrix
from sandy.football.ratings import DixonColesModel

from .features import blend_lambda, match_features
from .parsers import DISPLAY_TZ
from .ratings import load_models
from .schemas import CORNER_THRESHOLDS, GOAL_THRESHOLDS, MlsPrediction

logger = logging.getLogger(__name__)

CORNER_MAX = 16          # per-side truncation for the corners matrix (totals to 32)
FORM_BLEND_WEIGHT = 0.2  # validated by walk-forward backtest vs weight=0


def corner_overs(m) -> dict[float, float]:
    import numpy as np
    n = m.shape[0]
    idx = np.arange(n)
    hi, aj = np.meshgrid(idx, idx, indexing="ij")
    totals = hi + aj
    return {t: float(m[totals > t].sum()) for t in CORNER_THRESHOLDS}


def build_prediction(engine: Engine, goals: DixonColesModel, corners: DixonColesModel,
                     row, as_of: date, *, is_backtest: bool = False,
                     with_features: bool = True) -> MlsPrediction:
    eid, mdate, hid, aid, hname, aname = row
    feats = match_features(engine, hid, aid, as_of) if with_features else {}
    lam, mu = goals.expected_goals(hid, aid)
    hf, af = feats.get("home") or {}, feats.get("away") or {}
    lam = blend_lambda(lam, hf.get("goals_for_5"), af.get("goals_against_5"), FORM_BLEND_WEIGHT)
    mu = blend_lambda(mu, af.get("goals_for_5"), hf.get("goals_against_5"), FORM_BLEND_WEIGHT)
    gm = compute_scoreline_matrix(lam, mu, goals.rho)
    mk = markets_from_matrix(gm, thresholds=GOAL_THRESHOLDS)

    clam, cmu = corners.expected_goals(hid, aid)
    cm = compute_scoreline_matrix(clam, cmu, 0.0, max_goals=CORNER_MAX)
    c_over = corner_overs(cm)

    return MlsPrediction(
        event_id=eid, match_date=mdate, home_team_id=hid, away_team_id=aid,
        home_team=hname, away_team=aname,
        lambda_home=round(lam, 3), lambda_away=round(mu, 3),
        p_home_win=mk["p_home_win"], p_draw=mk["p_draw"], p_away_win=mk["p_away_win"],
        p_home_or_draw=mk["p_home_win"] + mk["p_draw"],
        p_over=mk["p_over"],
        corner_lambda_home=round(clam, 2), corner_lambda_away=round(cmu, 2),
        p_corners_over=c_over,
        most_likely=mk["most_likely"],
        features=feats, is_backtest=is_backtest,
    )


def persist_predictions(engine: Engine, preds: list[MlsPrediction]) -> int:
    sql = text("""
        INSERT INTO mls.match_predictions (
            event_id, match_date, home_team_id, away_team_id, home_team, away_team,
            lambda_home, lambda_away, p_home_win, p_draw, p_away_win, p_home_or_draw,
            p_over_0_5, p_over_1_5, p_over_2_5, p_over_3_5, p_over_4_5, p_over_5_5,
            corner_lambda_home, corner_lambda_away,
            p_corners_over_7_5, p_corners_over_8_5, p_corners_over_9_5, p_corners_over_10_5,
            p_corners_over_11_5, p_corners_over_12_5,
            most_likely_home, most_likely_away, features, is_backtest, predicted_at_utc
        ) VALUES (
            :eid, :d, :hid, :aid, :hn, :an, :lh, :la, :phw, :pd, :paw, :phd,
            :o05, :o15, :o25, :o35, :o45, :o55, :clh, :cla,
            :c75, :c85, :c95, :c105, :c115, :c125,
            :mlh, :mla, :feats, :bt, :now
        )
        ON CONFLICT (event_id) DO UPDATE SET
            lambda_home=:lh, lambda_away=:la, p_home_win=:phw, p_draw=:pd, p_away_win=:paw,
            p_home_or_draw=:phd, p_over_0_5=:o05, p_over_1_5=:o15, p_over_2_5=:o25,
            p_over_3_5=:o35, p_over_4_5=:o45, p_over_5_5=:o55,
            corner_lambda_home=:clh, corner_lambda_away=:cla,
            p_corners_over_7_5=:c75, p_corners_over_8_5=:c85, p_corners_over_9_5=:c95,
            p_corners_over_10_5=:c105, p_corners_over_11_5=:c115, p_corners_over_12_5=:c125,
            most_likely_home=:mlh, most_likely_away=:mla,
            features=:feats, is_backtest=:bt, predicted_at_utc=:now
    """)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        for p in preds:
            conn.execute(sql, {
                "eid": p.event_id, "d": p.match_date, "hid": p.home_team_id, "aid": p.away_team_id,
                "hn": p.home_team, "an": p.away_team, "lh": p.lambda_home, "la": p.lambda_away,
                "phw": p.p_home_win, "pd": p.p_draw, "paw": p.p_away_win, "phd": p.p_home_or_draw,
                "o05": p.p_over.get(0.5), "o15": p.p_over.get(1.5), "o25": p.p_over.get(2.5),
                "o35": p.p_over.get(3.5), "o45": p.p_over.get(4.5), "o55": p.p_over.get(5.5),
                "clh": p.corner_lambda_home, "cla": p.corner_lambda_away,
                "c75": p.p_corners_over.get(7.5), "c85": p.p_corners_over.get(8.5),
                "c95": p.p_corners_over.get(9.5), "c105": p.p_corners_over.get(10.5),
                "c115": p.p_corners_over.get(11.5), "c125": p.p_corners_over.get(12.5),
                "mlh": p.most_likely[0], "mla": p.most_likely[1],
                "feats": json.dumps(p.features) if p.features else None,
                "bt": p.is_backtest, "now": now,
            })
    return len(preds)


def predict_scheduled(config: Config | None = None, *, days_ahead: int = 2) -> list[MlsPrediction]:
    """Predict every not-started match from today through today+days_ahead."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    goals, corners = load_models(cfg)
    today = datetime.now(DISPLAY_TZ).date()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT m.event_id, m.match_date, m.home_team_id, m.away_team_id, t1.name, t2.name
            FROM mls.matches m
            JOIN mls.teams t1 ON t1.team_id = m.home_team_id
            JOIN mls.teams t2 ON t2.team_id = m.away_team_id
            WHERE m.status = 'NS' AND m.match_date BETWEEN :a AND :b
            ORDER BY m.match_date, m.event_id
        """), {"a": today, "b": today + timedelta(days=days_ahead)}).fetchall()
    preds = [build_prediction(engine, goals, corners, r, today) for r in rows]
    persist_predictions(engine, preds)
    logger.info("MLS predicted %s scheduled matches", len(preds))
    return preds
