"""Football predictor: Dixon-Coles model -> scoreline distribution -> markets.

From a fitted :class:`~sandy.football.ratings.DixonColesModel` we compute the
expected goals for a fixture, build the full P(home=i, away=j) scoreline
matrix (with the Dixon-Coles low-score correction), and derive every market
from that one matrix:

- W/D/L  (lower-triangle / diagonal / upper-triangle)
- P(total > 1.5/2.5/3.5/4.5)
- BTTS   (both teams score)
- most-likely exact scoreline (argmax cell)

This mirrors how ``sandy.over_under.predictor`` turns an expected-runs number
into over/under probabilities, but here a single 2-D distribution feeds all
markets coherently.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
from scipy.stats import poisson
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.ratings import DixonColesModel, load_model
from sandy.football.schemas import GOAL_THRESHOLDS, FootballPrediction
from sandy.logging import get_logger

logger = get_logger("football.predictor")

MAX_GOALS = 10


# ---------------------------------------------------------------------------
# Core math (pure)
# ---------------------------------------------------------------------------


def compute_scoreline_matrix(
    lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """P(home=i, away=j) for i,j in 0..max_goals, with Dixon-Coles correction.

    Truncation + correction leave the mass slightly off 1, so we renormalize.
    """
    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, lam)
    pa = poisson.pmf(i, mu)
    m = np.outer(ph, pa)
    # Dixon-Coles low-score dependency adjustment.
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


def markets_from_matrix(m: np.ndarray, thresholds: list[float] | None = None) -> dict:
    """Derive W/D/L, over/under, BTTS, most-likely score from a scoreline matrix.

    `thresholds` lets callers (MLS, multi-league soccer) request a wider goals
    ladder without changing this module's default (used by the World Cup vertical).
    """
    n = m.shape[0]
    idx = np.arange(n)
    home_i, away_j = np.meshgrid(idx, idx, indexing="ij")
    totals = home_i + away_j

    p_home = float(np.tril(m, -1).sum())   # home_goals > away_goals
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())    # away_goals > home_goals

    p_over = {t: float(m[totals > t].sum()) for t in (thresholds or GOAL_THRESHOLDS)}
    p_btts = float(m[(home_i >= 1) & (away_j >= 1)].sum())

    mli, mlj = np.unravel_index(int(np.argmax(m)), m.shape)
    flat = sorted(
        (((int(a), int(b)), float(m[a, b])) for a in idx for b in idx),
        key=lambda x: -x[1],
    )
    top = [{"h": a, "a": b, "p": round(p, 4)} for (a, b), p in flat[:6]]

    return {
        "p_home_win": p_home,
        "p_draw": p_draw,
        "p_away_win": p_away,
        "p_over": p_over,
        "p_btts": p_btts,
        "most_likely": (int(mli), int(mlj)),
        "scoreline": top,
    }


def predict_match(
    model: DixonColesModel,
    home_team_id: int,
    away_team_id: int,
    *,
    neutral: bool = False,
) -> dict:
    """Full market set for one fixture. Pure (no DB)."""
    lam, mu = model.expected_goals(home_team_id, away_team_id, neutral=neutral)
    m = compute_scoreline_matrix(lam, mu, model.rho)
    out = markets_from_matrix(m)
    out["lambda_home"] = lam
    out["lambda_away"] = mu
    out["neutral"] = neutral
    return out


# ---------------------------------------------------------------------------
# DB-aware: predict scheduled fixtures and persist
# ---------------------------------------------------------------------------


def _load_model(cfg: Config) -> DixonColesModel:
    from sandy.football.ratings import ARTIFACT_NAME
    return load_model(cfg.model.model_dir / ARTIFACT_NAME)


def build_prediction(
    model: DixonColesModel,
    *,
    fixture_id: int,
    match_date: date,
    home_team_id: int,
    away_team_id: int,
    home_name: str,
    away_name: str,
    kickoff_utc: datetime | None,
    neutral: bool,
) -> FootballPrediction:
    mk = predict_match(model, home_team_id, away_team_id, neutral=neutral)
    return FootballPrediction(
        fixture_id=fixture_id, match_date=match_date,
        home_team_id=home_team_id, away_team_id=away_team_id,
        home_team_name=home_name, away_team_name=away_name,
        kickoff_utc=kickoff_utc, predicted_at_utc=datetime.now(timezone.utc),
        lambda_home=mk["lambda_home"], lambda_away=mk["lambda_away"],
        p_home_win=mk["p_home_win"], p_draw=mk["p_draw"], p_away_win=mk["p_away_win"],
        p_over=mk["p_over"], p_btts=mk["p_btts"],
        most_likely_home=mk["most_likely"][0], most_likely_away=mk["most_likely"][1],
        scoreline=mk["scoreline"],
        feature_vector={
            "lambda_home": mk["lambda_home"], "lambda_away": mk["lambda_away"],
            "atk_home": model.attack.get(home_team_id, 0.0),
            "def_home": model.defense.get(home_team_id, 0.0),
            "atk_away": model.attack.get(away_team_id, 0.0),
            "def_away": model.defense.get(away_team_id, 0.0),
            "home_adv": 0.0 if neutral else model.home_adv,
        },
    )


# Competitions played at neutral venues (no home advantage in the prediction).
NEUTRAL_COMPETITIONS = frozenset({"World Cup"})


def predict_scheduled(
    config: Config | None = None,
    *,
    upcoming_only: bool = True,
) -> list[FootballPrediction]:
    """Predict not-started fixtures in football.matches and persist them.

    Neutral-venue handling is per-match: World Cup games drop home advantage.
    """
    cfg = config or load_config()
    engine = create_engine(cfg)
    model = _load_model(cfg)

    clause = "m.status = 'NS'" if upcoming_only else "TRUE"
    sql = text(f"""
        SELECT m.fixture_id, m.match_date, m.kickoff_utc,
               m.home_team_id, m.away_team_id,
               th.name AS home_name, ta.name AS away_name, m.competition
        FROM football.matches m
        JOIN football.teams th ON th.team_id = m.home_team_id
        JOIN football.teams ta ON ta.team_id = m.away_team_id
        WHERE {clause}
        ORDER BY m.match_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    preds = [
        build_prediction(
            model, fixture_id=r[0], match_date=r[1], kickoff_utc=r[2],
            home_team_id=r[3], away_team_id=r[4], home_name=r[5], away_name=r[6],
            neutral=(r[7] in NEUTRAL_COMPETITIONS),
        )
        for r in rows
    ]
    if preds:
        persist_predictions(engine, preds)
    logger.info("Predicted scheduled fixtures",
                extra={"component": "football.predictor", "n": len(preds)})
    return preds


_PRED_UPSERT = text("""
    INSERT INTO football.match_predictions
        (fixture_id, match_date, home_team_id, away_team_id, predicted_at_utc,
         lambda_home, lambda_away, p_home_win, p_draw, p_away_win,
         p_over_1_5, p_over_2_5, p_over_3_5, p_over_4_5, p_btts,
         most_likely_home, most_likely_away, scoreline, feature_vector)
    VALUES
        (:fixture_id, :match_date, :home_team_id, :away_team_id, :predicted_at_utc,
         :lambda_home, :lambda_away, :p_home_win, :p_draw, :p_away_win,
         :p_over_1_5, :p_over_2_5, :p_over_3_5, :p_over_4_5, :p_btts,
         :most_likely_home, :most_likely_away, CAST(:scoreline AS JSONB), CAST(:feature_vector AS JSONB))
    ON CONFLICT (fixture_id) DO UPDATE SET
        match_date = EXCLUDED.match_date, predicted_at_utc = EXCLUDED.predicted_at_utc,
        lambda_home = EXCLUDED.lambda_home, lambda_away = EXCLUDED.lambda_away,
        p_home_win = EXCLUDED.p_home_win, p_draw = EXCLUDED.p_draw, p_away_win = EXCLUDED.p_away_win,
        p_over_1_5 = EXCLUDED.p_over_1_5, p_over_2_5 = EXCLUDED.p_over_2_5,
        p_over_3_5 = EXCLUDED.p_over_3_5, p_over_4_5 = EXCLUDED.p_over_4_5, p_btts = EXCLUDED.p_btts,
        most_likely_home = EXCLUDED.most_likely_home, most_likely_away = EXCLUDED.most_likely_away,
        scoreline = EXCLUDED.scoreline, feature_vector = EXCLUDED.feature_vector
""")


def persist_predictions(engine: Engine, predictions: list[FootballPrediction]) -> int:
    import json
    n = 0
    with engine.begin() as conn:
        for p in predictions:
            conn.execute(_PRED_UPSERT, {
                "fixture_id": p.fixture_id, "match_date": p.match_date,
                "home_team_id": p.home_team_id, "away_team_id": p.away_team_id,
                "predicted_at_utc": p.predicted_at_utc,
                "lambda_home": p.lambda_home, "lambda_away": p.lambda_away,
                "p_home_win": p.p_home_win, "p_draw": p.p_draw, "p_away_win": p.p_away_win,
                "p_over_1_5": p.p_over[1.5], "p_over_2_5": p.p_over[2.5],
                "p_over_3_5": p.p_over[3.5], "p_over_4_5": p.p_over[4.5], "p_btts": p.p_btts,
                "most_likely_home": p.most_likely_home, "most_likely_away": p.most_likely_away,
                "scoreline": json.dumps(p.scoreline),
                "feature_vector": json.dumps(p.feature_vector),
            })
            n += 1
    return n


__all__ = [
    "compute_scoreline_matrix",
    "markets_from_matrix",
    "predict_match",
    "build_prediction",
    "predict_scheduled",
    "persist_predictions",
]
