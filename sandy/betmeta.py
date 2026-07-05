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
    # Multi-league soccer vertical: one meta per league, rows filtered by league.
    **{
        f"soccer_{lg}": {
            "schema": "soccer", "table": "soccer.match_predictions",
            "where": f"league = '{lg}'", "league": lg,
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
        }
        for lg in ("col", "mex", "esp", "eng")
    },
    "nba": {
        "schema": "nba", "table": "nba.game_predictions",
        "markets": {
            "winner": ("p_home_win", "winner", None),
            "over_215_5": ("p_over_215_5", "points", 215.5),
            "over_220_5": ("p_over_220_5", "points", 220.5),
            "over_225_5": ("p_over_225_5", "points", 225.5),
            "over_230_5": ("p_over_230_5", "points", 230.5),
            "over_235_5": ("p_over_235_5", "points", 235.5),
        },
        "num_cols": ["exp_home_points", "exp_away_points", "exp_total", "sigma_total", "p_home_win"],
        "form_keys": ["pf_5", "pa_5", "pf_10", "pa_10", "wins_10", "rest_days", "played_10"],
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
    # World Cup / national teams: no corners in the data source; covariates are the
    # DC model factors, surfaced as columns by the football.predictions_meta view.
    # Its calibration_snapshots table predates the shared shape -> no snapshot write,
    # and it has no is_backtest column (all reconciled rows are walk-forward).
    "worldcup": {
        "schema": "football", "table": "football.predictions_meta",
        "markets": {
            "double_chance": ("p_home_or_draw", "result", None),
            "over_1_5": ("p_over_1_5", "goals", 1.5),
            "over_2_5": ("p_over_2_5", "goals", 2.5),
            "over_3_5": ("p_over_3_5", "goals", 3.5),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "btts": ("p_btts", "btts", None),
        },
        "num_cols": ["lambda_home", "lambda_away", "atk_home", "atk_away",
                     "def_home", "def_away", "home_adv"],
        "form_keys": [],
        "no_snapshot": True, "no_backtest_col": True, "no_hist_gate": True,
    },
}


def _correct(row, kind: str, line: float | None, p: float) -> bool | None:
    pick_yes = p >= 0.5
    if kind == "result":
        res = row.get("actual_result") or row.get("actual_reg_result")
        if res is None:
            return None
        return pick_yes == (res != "A")
    if kind == "winner":
        w = row.get("actual_winner")
        if w is None:
            return None
        return pick_yes == (w == "H")
    if kind == "btts":
        b = row.get("actual_btts")
        if b is None:
            return None
        return pick_yes == bool(b)
    if kind == "points":
        actual = row.get("actual_total")
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            return None
        return pick_yes == (actual > line)
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
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    with engine.begin() as conn:
        df = pd.read_sql(text(
            f"SELECT * FROM {spec['table']} WHERE outcome_filled_at_utc IS NOT NULL{extra}"), conn)
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
    hp_raw = booster.predict(hold[feat_cols])
    hy = hold["_y"].to_numpy()
    # ISOTONIC CALIBRATION on the chronological holdout (data the booster never saw):
    # maps raw scores to honest probabilities, so a displayed "🤖 80%" really hits ~80%.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(hp_raw, hy.astype(float))
    hp = iso.predict(hp_raw)
    auc = round(float(roc_auc_score(hy, hp_raw)), 4)

    def _ladder(mask):
        rows = []
        for thr in THRESHOLDS:
            m = mask & (hp >= thr)
            n = int(m.sum())
            rows.append({"thr": thr, "n": n, "correct": int(hy[m].sum()),
                         "acc": round(float(hy[m].mean()), 4) if n else None})
        return rows

    all_mask = np.ones(len(hy), dtype=bool)
    table = _ladder(all_mask)
    # The user's rule: the threshold that MAXIMIZES realized accuracy (with enough
    # holdout picks to trust the estimate).
    viable = [t for t in table if (t["n"] or 0) >= 50 and t["acc"] is not None]
    rec = max(viable, key=lambda t: t["acc"])["thr"] if viable else None

    def _below(mask):
        if rec is None:
            return None
        m = mask & (hp < rec)
        n = int(m.sum())
        return {"n": n, "correct": int(hy[m].sum()),
                "acc": round(float(hy[m].mean()), 4) if n else None}

    # Per market-group ladders (goals / corners / double-chance / winner / points)
    # so the digest can show an MLB-style reliability block per prediction type.
    mkt_cols = [c for c in feat_cols if c.startswith("mkt_")]
    hold_markets = hold[mkt_cols].to_numpy().argmax(axis=1)
    kind_of = {i: SPECS[league]["markets"][c[4:]][1] for i, c in enumerate(mkt_cols)}
    row_kinds = np.array([kind_of[i] for i in hold_markets])
    eval_by_group = {}
    for kind in dict.fromkeys(kind_of.values()):
        gm = row_kinds == kind
        eval_by_group[kind] = {"table": _ladder(gm), "below": _below(gm)}

    # PER-EXACT-LINE ladders + per-line recommended threshold (the user's rule,
    # applied line by line: each market keeps the threshold that maximizes ITS
    # holdout accuracy, n>=30; falls back to the global threshold when thin).
    row_market = np.array([mkt_cols[i][4:] for i in hold_markets])
    eval_by_market, threshold_by_market = {}, {}
    for m in SPECS[league]["markets"]:
        mm = row_market == m
        lad = _ladder(mm)
        eval_by_market[m] = {"table": lad, "below": _below(mm)}
        viable_m = [t for t in lad if (t["n"] or 0) >= 30 and t["acc"] is not None]
        threshold_by_market[m] = (max(viable_m, key=lambda t: t["acc"])["thr"]
                                  if viable_m else rec)
    # Production = the split-trained booster + its isotonic map. (Retraining on ALL
    # data would orphan the calibration — integrity of the 🤖 probability wins.)
    imp = sorted(zip(feat_cols, booster.feature_importance("gain")),
                 key=lambda x: -x[1])[:10]
    artifact = {"model_str": booster.model_to_string(), "iso": iso, "features": feat_cols,
                "threshold": rec, "eval_table": table, "auc": auc,
                "eval_below": _below(all_mask), "eval_by_group": eval_by_group,
                "eval_by_market": eval_by_market, "threshold_by_market": threshold_by_market,
                "importances": [(f, round(float(g), 1)) for f, g in imp],
                "trained_rows": len(train), "holdout_rows": len(hold),
                "trained_at": date.today().isoformat()}
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    # League-scoped schemas (soccer) have a NOT NULL league column on snapshots.
    lg_col, lg_val = ("league, ", ":lg, ") if SPECS[league].get("league") else ("", "")
    if SPECS[league].get("no_snapshot"):
        logger.info("%s meta trained: auc=%s thr=%s (no snapshot table)", league, auc, rec)
        return {"rows": len(X), "threshold": rec, "eval_table": table, "auc": auc,
                "threshold_by_market": threshold_by_market,
                "importances": artifact["importances"]}
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {SPECS[league]['schema']}.calibration_snapshots
                (snapshot_date, {lg_col}market, lookback_days, sample_size, accuracy, brier,
                 reliability, recommended_threshold)
            VALUES (:d, {lg_val}'meta_pick', NULL, :n, :acc, NULL, :rel, :thr)
        """), {"d": date.today(), "n": len(hold), "lg": SPECS[league].get("league"),
               "acc": next((t["acc"] for t in table if t["thr"] == rec), None) if rec else None,
               "rel": json.dumps(table), "thr": rec})
    logger.info("%s meta trained: auc=%s thr=%s", league, auc, rec)
    return {"rows": len(X), "threshold": rec, "eval_table": table, "auc": auc,
            "threshold_by_market": threshold_by_market,
            "importances": artifact["importances"]}


_loaded: dict = {}


def load_meta(league: str, cfg: Config):
    """Cached artifact loader → (booster, features, global_thr, iso, thr_by_market) or None."""
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
    _loaded[league] = (booster, a["features"], a["threshold"], a.get("iso"),
                       a.get("threshold_by_market") or {})
    return _loaded[league]


def market_threshold(league: str, cfg: Config, market: str) -> float | None:
    """The per-line recommended threshold (falls back to the league's global)."""
    loaded = load_meta(league, cfg)
    if not loaded:
        return None
    _b, _f, global_thr, _i, by_market = loaded
    return by_market.get(market, global_thr)


def score_candidate(league: str, cfg: Config, row_dict: dict, market: str, p: float) -> float | None:
    """Meta P(pick correct) for one candidate bet, or None if no artifact."""
    loaded = load_meta(league, cfg)
    if not loaded:
        return None
    booster, feat_cols, _thr, iso, _by_mkt = loaded
    feats = _row_features(row_dict, SPECS[league], market, p)
    X = pd.DataFrame([feats]).reindex(columns=feat_cols)
    raw = float(booster.predict(X)[0])
    return float(iso.predict([raw])[0]) if iso is not None else raw


GROUP_LABELS_ES = {
    "result": "Doble oportunidad",
    "goals": "Goles (más de)",
    "corners": "Tiros de esquina (más de)",
    "winner": "Ganador",
    "points": "Puntos (más de)",
}


def format_meta_ladder(league: str, cfg: Config | None = None) -> str | None:
    """MLB-style P(correct) reliability block, one ladder per market group.

    Rendered from the holdout evaluation stored in the meta artifact, e.g.:
        🤖 Meta-modelo — Goles (más de):
          P(acierto) ≥70%: 82% (1029/1260)
          P(acierto) ≥80%: 85% (713/839) ← recomendado
          P(acierto) <80%: 64% (1071/1670) — evitar
    """
    cfg = cfg or load_config()
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        art = pickle.load(f)
    groups = art.get("eval_by_group")
    rec = art.get("threshold")
    if not groups:
        return None
    blocks = []
    for kind, g in groups.items():
        lines = [f"🤖 Meta-modelo — {GROUP_LABELS_ES.get(kind, kind)}:"]
        for t in g["table"]:
            if not t["n"]:
                continue
            mark = " ← recomendado" if rec is not None and t["thr"] == rec else ""
            lines.append(f"  P(acierto) ≥{round(t['thr'] * 100)}%: "
                         f"{t['acc'] * 100:.0f}% ({t['correct']}/{t['n']}){mark}")
        b = g.get("below")
        if b and b.get("n"):
            lines.append(f"  P(acierto) <{round(rec * 100)}%: "
                         f"{b['acc'] * 100:.0f}% ({b['correct']}/{b['n']}) — evitar")
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    by_mkt = art.get("threshold_by_market") or {}
    if by_mkt:
        pares = " · ".join(f"{m.replace('_', ' ')} {t:.0%}" for m, t in by_mkt.items() if t)
        blocks.append(f"📌 Umbral recomendado POR LÍNEA (el ✅ usa este):\n  {pares}")
    return "\n\n".join(blocks) if blocks else None
