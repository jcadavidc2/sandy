"""Meta-model layer for the MLS/NHL verticals — the MLB recipe applied per league:

  1. The base (Dixon-Coles+form) model outputs a probability per market
     (e.g. P(over 1.5 goles) = 85%).
  2. A LightGBM META-MODEL predicts P(this pick is CORRECT) from the pick's
     confidence PLUS the covariates (form, rest, lambdas, market identity).
  3. A threshold on the meta score is chosen on held-out (chronologically later)
     data to maximize realized accuracy — the digest only recommends picks whose
     meta score clears it.

Training data = the walk-forward backtest predictions (already leakage-free),
one row per (prediction, market). Retrained nightly like MLB's meta.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 800
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# league → (schema, predictions table, markets: name → (prob col, kind, line))
SPECS = {
    "mls": {
        "schema": "mls", "table": "mls.match_predictions",
        "markets": {
            "double_chance": ("p_home_or_draw", "result", None),
            "over_1_5": ("p_over_1_5", "goals", 1.5),
            "over_2_5": ("p_over_2_5", "goals", 2.5),
            "over_3_5": ("p_over_3_5", "goals", 3.5),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "corners_over_8_5": ("p_corners_over_8_5", "corners", 8.5),
            "corners_over_9_5": ("p_corners_over_9_5", "corners", 9.5),
            "corners_over_10_5": ("p_corners_over_10_5", "corners", 10.5),
            "corners_over_11_5": ("p_corners_over_11_5", "corners", 11.5),
        },
        "num_cols": ["lambda_home", "lambda_away", "corner_lambda_home", "corner_lambda_away"],
        "form_keys": ["goals_for_5", "goals_against_5", "corners_for_5", "corners_against_5",
                      "form_points_5", "rest_days", "played_10"],
    },
    "nhl": {
        "schema": "nhl", "table": "nhl.game_predictions",
        "markets": {
            "double_chance": ("p_home_or_tie", "result", None),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "over_5_5": ("p_over_5_5", "goals", 5.5),
            "over_6_5": ("p_over_6_5", "goals", 6.5),
            "over_7_5": ("p_over_7_5", "goals", 7.5),
        },
        "num_cols": ["lambda_home", "lambda_away"],
        "form_keys": ["gf_5", "ga_5", "gf_10", "ga_10", "points_10", "rest_days", "played_10"],
    },
}


def _correct(row, kind: str, line: float | None, p: float) -> bool | None:
    pick_yes = p >= 0.5
    if kind == "result":
        res = row.get("actual_result") or row.get("actual_reg_result")
        if res is None:
            return None
        return pick_yes == (res != "A")
    actual = row.get("actual_total_goals") if kind == "goals" else row.get("actual_total_corners")
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return None
    return pick_yes == (actual > line)


def _row_features(row, spec, market, p) -> dict:
    feats = row.get("features")
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except (TypeError, ValueError):
            feats = None
    feats = feats or {}
    home, away = feats.get("home") or {}, feats.get("away") or {}
    out = {"p": p, "conf": max(p, 1 - p)}
    for c in spec["num_cols"]:
        out[c] = row.get(c)
    for k in spec["form_keys"]:
        hv, av = home.get(k), away.get(k)
        out[f"h_{k}"] = float(hv) if isinstance(hv, (int, float)) else np.nan
        out[f"a_{k}"] = float(av) if isinstance(av, (int, float)) else np.nan
    for m in spec["markets"]:
        out[f"mkt_{m}"] = 1.0 if m == market else 0.0
    return out


def _frame(engine, league: str) -> pd.DataFrame:
    spec = SPECS[league]
    with engine.begin() as conn:
        df = pd.read_sql(text(
            f"SELECT * FROM {spec['table']} WHERE outcome_filled_at_utc IS NOT NULL"), conn)
    rows, ys, dates = [], [], []
    for _, r in df.iterrows():
        rd = r.to_dict()
        for market, (pcol, kind, line) in spec["markets"].items():
            p = rd.get(pcol)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                continue
            y = _correct(rd, kind, line, float(p))
            if y is None:
                continue
            rows.append(_row_features(rd, spec, market, float(p)))
            ys.append(bool(y))
            dates.append(rd["match_date"])
    X = pd.DataFrame(rows)
    X["_y"] = ys
    X["_date"] = dates
    return X


def train_meta(league: str, config: Config | None = None) -> dict:
    """Train + evaluate (chronological holdout) + persist artifact and threshold table."""
    import lightgbm as lgb
    cfg = config or load_config()
    engine = create_engine(cfg)
    X = _frame(engine, league)
    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{league}: only {len(X)} meta rows (<{MIN_TRAIN_ROWS})")
    X = X.sort_values("_date").reset_index(drop=True)
    cut = X["_date"].quantile(0.7)
    train, hold = X[X["_date"] <= cut], X[X["_date"] > cut]
    feat_cols = [c for c in X.columns if c not in ("_y", "_date")]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.85, bagging_fraction=0.85,
                  bagging_freq=1, verbose=-1, seed=42)
    booster = lgb.train(params, lgb.Dataset(train[feat_cols], label=train["_y"]),
                        num_boost_round=300)
    hp = booster.predict(hold[feat_cols])
    hy = hold["_y"].to_numpy()
    table = []
    for thr in THRESHOLDS:
        m = hp >= thr
        n = int(m.sum())
        table.append({"thr": thr, "n": n,
                      "acc": round(float(hy[m].mean()), 4) if n else None})
    # The user's rule: the threshold that MAXIMIZES realized accuracy (with enough
    # holdout picks to trust the estimate).
    viable = [t for t in table if (t["n"] or 0) >= 50 and t["acc"] is not None]
    rec = max(viable, key=lambda t: t["acc"])["thr"] if viable else None
    # Final production model uses ALL data (the holdout only sized the threshold).
    booster_full = lgb.train(params, lgb.Dataset(X[feat_cols], label=X["_y"]),
                             num_boost_round=300)
    artifact = {"model_str": booster_full.model_to_string(), "features": feat_cols,
                "threshold": rec, "eval_table": table, "trained_rows": len(X),
                "holdout_rows": len(hold), "trained_at": date.today().isoformat()}
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {SPECS[league]['schema']}.calibration_snapshots
                (snapshot_date, market, lookback_days, sample_size, accuracy, brier,
                 reliability, recommended_threshold)
            VALUES (:d, 'meta_pick', NULL, :n, :acc, NULL, :rel, :thr)
        """), {"d": date.today(), "n": len(hold),
               "acc": next((t["acc"] for t in table if t["thr"] == rec), None) if rec else None,
               "rel": json.dumps(table), "thr": rec})
    logger.info("%s meta trained: %s rows, holdout thr=%s table=%s", league, len(X), rec, table)
    return {"rows": len(X), "threshold": rec, "eval_table": table}


_loaded: dict = {}


def load_meta(league: str, cfg: Config):
    """Cached artifact loader → (booster, features, threshold) or None."""
    import lightgbm as lgb
    if league in _loaded:
        return _loaded[league]
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        _loaded[league] = None
        return None
    with open(path, "rb") as f:
        a = pickle.load(f)
    booster = lgb.Booster(model_str=a["model_str"])
    _loaded[league] = (booster, a["features"], a["threshold"])
    return _loaded[league]


def score_candidate(league: str, cfg: Config, row_dict: dict, market: str, p: float) -> float | None:
    """Meta P(pick correct) for one candidate bet, or None if no artifact."""
    loaded = load_meta(league, cfg)
    if not loaded:
        return None
    booster, feat_cols, _thr = loaded
    feats = _row_features(row_dict, SPECS[league], market, p)
    X = pd.DataFrame([feats]).reindex(columns=feat_cols)
    return float(booster.predict(X)[0])
