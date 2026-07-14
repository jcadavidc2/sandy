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

# "Model's own recent errors" covariates: per team, the mean SIGNED and mean
# ABSOLUTE error of the base model's TOTAL prediction over that team's last
# ERR_WINDOW reconciled games STRICTLY BEFORE the row's match_date.
# Signed-error convention everywhere: EXPECTED minus ACTUAL (positive = the
# base model overshot the total). NaN when the team has no prior history.
ERR_WINDOW = 8
MODEL_ERR_FEATURES = ("h_model_err", "h_model_abs_err", "a_model_err", "a_model_abs_err")

# "Team base-model reliability" covariates: per team, the rolling mean (and
# count) of the base model's per-game correct-pick FRACTION (share of that
# game's graded markets where pick p>=0.5 matched the outcome) over the team's
# last REL_WINDOW reconciled games STRICTLY BEFORE the row's match_date.
# Same bulk (_attach_team_rel) / live (_team_recent_rel) equality contract as
# MODEL_ERR_FEATURES. NaN mean + 0 count with no history.
REL_WINDOW = 12
TEAM_REL_FEATURES = ("h_base_rel", "h_base_rel_n", "a_base_rel", "a_base_rel_n")

# Game-time WEATHER covariates (open-meteo via sandy/weather.py) — only for
# leagues whose spec declares a "wx_key" (the game-id column joining
# odds.game_weather.game_key: MLB game_pk / NFL event_id). wx_dome is 1.0 for
# fixed roofs, 0.5 retractable, 0.0 open (see weather.roof_score). Same bulk
# (_attach_weather) / live (weather.live_wx: DB row, else on-the-fly forecast
# upsert for pending games, NaN on failure) equality contract as MODEL_ERR.
WX_FEATURES = ("wx_temp", "wx_wind", "wx_precip", "wx_dome")

# Bump when the _frame/_row_features feature schema changes — downstream caches
# (betrefine's OOF store) key on this so a feature upgrade invalidates them.
FRAME_VERSION = 4  # v4 (2026-07-14): playoff/stage covariates (is_playoff, series_game_no, is_knockout)

# league → (schema, predictions table, markets: name → (prob col, kind, line))
SPECS = {
    "mls": {
        "schema": "mls", "table": "mls.match_predictions",
        "markets": {
            "double_chance": ("p_home_or_draw", "result", None),
            "over_0_5": ("p_over_0_5", "goals", 0.5),
            "over_1_5": ("p_over_1_5", "goals", 1.5),
            "over_2_5": ("p_over_2_5", "goals", 2.5),
            "over_3_5": ("p_over_3_5", "goals", 3.5),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "over_5_5": ("p_over_5_5", "goals", 5.5),
            "corners_over_7_5": ("p_corners_over_7_5", "corners", 7.5),
            "corners_over_8_5": ("p_corners_over_8_5", "corners", 8.5),
            "corners_over_9_5": ("p_corners_over_9_5", "corners", 9.5),
            "corners_over_10_5": ("p_corners_over_10_5", "corners", 10.5),
            "corners_over_11_5": ("p_corners_over_11_5", "corners", 11.5),
            "corners_over_12_5": ("p_corners_over_12_5", "corners", 12.5),
        },
        # is_playoff REJECTED by the 2026-07-14 gate (acc@thr .8827→.8619): the
        # stamped column stays on the table (dormant) but the meta doesn't use it.
        "num_cols": ["lambda_home", "lambda_away", "corner_lambda_home", "corner_lambda_away"],
        "form_keys": ["goals_for_5", "goals_against_5", "corners_for_5", "corners_against_5",
                      "form_points_5", "rest_days", "played_10"],
        "err_expected": ["lambda_home", "lambda_away"], "err_actual": "actual_total_goals",
    },
    # Multi-league soccer vertical: one meta per league, rows filtered by league.
    **{
        f"soccer_{lg}": {
            "schema": "soccer", "table": "soccer.match_predictions",
            "where": f"league = '{lg}'", "league": lg,
            "markets": {
                "double_chance": ("p_home_or_draw", "result", None),
                "over_0_5": ("p_over_0_5", "goals", 0.5),
                "over_1_5": ("p_over_1_5", "goals", 1.5),
                "over_2_5": ("p_over_2_5", "goals", 2.5),
                "over_3_5": ("p_over_3_5", "goals", 3.5),
                "over_4_5": ("p_over_4_5", "goals", 4.5),
                "over_5_5": ("p_over_5_5", "goals", 5.5),
                "corners_over_7_5": ("p_corners_over_7_5", "corners", 7.5),
                "corners_over_8_5": ("p_corners_over_8_5", "corners", 8.5),
                "corners_over_9_5": ("p_corners_over_9_5", "corners", 9.5),
                "corners_over_10_5": ("p_corners_over_10_5", "corners", 10.5),
                "corners_over_11_5": ("p_corners_over_11_5", "corners", 11.5),
                "corners_over_12_5": ("p_corners_over_12_5", "corners", 12.5),
            },
            # is_knockout (cuadrangulares) ADOPTED for col only — gate verdict
            # 2026-07-14: col improved AUC .724→.7253 AND acc .8851→.8876; mex
            # REJECTED (AUC .6804→.6731). esp/eng have no playoff round.
            "num_cols": ["lambda_home", "lambda_away", "corner_lambda_home", "corner_lambda_away"]
                        + (["is_knockout"] if lg == "col" else []),
            "form_keys": ["goals_for_5", "goals_against_5", "corners_for_5", "corners_against_5",
                          "form_points_5", "rest_days", "played_10"],
            "err_expected": ["lambda_home", "lambda_away"], "err_actual": "actual_total_goals",
        }
        for lg in ("col", "mex", "esp", "eng")
    },
    # Soccer CUP competitions (2026-07-13): same vertical + the competition-stage
    # covariate — is_knockout (group/league phase vs elimination football) changes
    # game character, so the cup metas see it; domestic leagues keep their exact
    # feature contract (existing artifacts untouched).
    **{
        f"soccer_{lg}": {
            "schema": "soccer", "table": "soccer.match_predictions",
            "where": f"league = '{lg}'", "league": lg,
            "markets": {
                "double_chance": ("p_home_or_draw", "result", None),
                "over_0_5": ("p_over_0_5", "goals", 0.5),
                "over_1_5": ("p_over_1_5", "goals", 1.5),
                "over_2_5": ("p_over_2_5", "goals", 2.5),
                "over_3_5": ("p_over_3_5", "goals", 3.5),
                "over_4_5": ("p_over_4_5", "goals", 4.5),
                "over_5_5": ("p_over_5_5", "goals", 5.5),
                "corners_over_7_5": ("p_corners_over_7_5", "corners", 7.5),
                "corners_over_8_5": ("p_corners_over_8_5", "corners", 8.5),
                "corners_over_9_5": ("p_corners_over_9_5", "corners", 9.5),
                "corners_over_10_5": ("p_corners_over_10_5", "corners", 10.5),
                "corners_over_11_5": ("p_corners_over_11_5", "corners", 11.5),
                "corners_over_12_5": ("p_corners_over_12_5", "corners", 12.5),
            },
            "num_cols": ["lambda_home", "lambda_away", "corner_lambda_home", "corner_lambda_away",
                         "is_knockout"],
            "form_keys": ["goals_for_5", "goals_against_5", "corners_for_5", "corners_against_5",
                          "form_points_5", "rest_days", "played_10"],
            "err_expected": ["lambda_home", "lambda_away"], "err_actual": "actual_total_goals",
        }
        for lg in ("ucl", "uel", "ccc", "lgc", "lib", "sud")
    },
    "nba": {
        "schema": "nba", "table": "nba.game_predictions",
        "markets": {
            "winner": ("p_home_win", "winner", None),
            "over_210_5": ("p_over_210_5", "points", 210.5),
            "over_215_5": ("p_over_215_5", "points", 215.5),
            "over_220_5": ("p_over_220_5", "points", 220.5),
            "over_225_5": ("p_over_225_5", "points", 225.5),
            "over_230_5": ("p_over_230_5", "points", 230.5),
            "over_235_5": ("p_over_235_5", "points", 235.5),
            "over_240_5": ("p_over_240_5", "points", 240.5),
            "over_245_5": ("p_over_245_5", "points", 245.5),
        },
        # Playoffs covariates (gated adoption 2026-07-14): flag + game N of the
        # best-of-7 (a game 7 is not a game 1).
        "num_cols": ["exp_home_points", "exp_away_points", "exp_total", "sigma_total", "p_home_win",
                     "is_playoff", "series_game_no"],
        "form_keys": ["pf_5", "pa_5", "pf_10", "pa_10", "wins_10", "rest_days", "played_10"],
        "err_expected": ["exp_total"], "err_actual": "actual_total",
    },
    "nfl": {
        "schema": "nfl", "table": "nfl.game_predictions",
        "markets": {
            "winner": ("p_home_win", "winner", None),
            "over_37_5": ("p_over_37_5", "points", 37.5),
            "over_41_5": ("p_over_41_5", "points", 41.5),
            "over_44_5": ("p_over_44_5", "points", 44.5),
            "over_47_5": ("p_over_47_5", "points", 47.5),
            "over_51_5": ("p_over_51_5", "points", 51.5),
        },
        # is_playoff REJECTED by the 2026-07-14 gate (AUC .6002→.5856, acc −3.2pp —
        # 13 playoff games/season is too thin). Column stays on the table, dormant.
        "num_cols": ["exp_home_points", "exp_away_points", "exp_total", "sigma_total", "p_home_win"],
        "form_keys": ["pf_5", "pa_5", "pf_10", "pa_10", "wins_10", "rest_days", "played_10"],
        "err_expected": ["exp_total"], "err_actual": "actual_total",
        # Weather covariates DORMANT for the NFL meta: the 2026-07-07 gated
        # retrain with wx_* fell on test AUC (0.6163 -> 0.6103), and the
        # SECOND try (2026-07-07, after weather moved into the BASE model's
        # expected-total adjustment — see nfl/loop.py _wx_adjust) also lost:
        # AUC 0.6002 -> 0.6022 but acc@thr collapsed 0.8241 -> 0.7748 (>1pp
        # rule). Weather now lives in the NFL BASE model only; the nightly
        # `sandy nfl meta` retrain is ungated and must not resurrect wx here.
        # odds.game_weather keeps accumulating NFL rows (backfilled 2022-09+
        # and refreshed daily by odds_daily.sh); to re-gate later add:
        # "wx_key": "event_id",
    },
    "nhl": {
        "schema": "nhl", "table": "nhl.game_predictions",
        "markets": {
            "double_chance": ("p_home_or_tie", "result", None),
            "over_3_5": ("p_over_3_5", "goals", 3.5),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "over_5_5": ("p_over_5_5", "goals", 5.5),
            "over_6_5": ("p_over_6_5", "goals", 6.5),
            "over_7_5": ("p_over_7_5", "goals", 7.5),
            "over_8_5": ("p_over_8_5", "goals", 8.5),
        },
        # is_playoff/series_game_no REJECTED by the 2026-07-14 gate (flat AUC,
        # acc −0.7pp). Columns stay stamped on the table, dormant.
        "num_cols": ["lambda_home", "lambda_away"],
        "form_keys": ["gf_5", "ga_5", "gf_10", "ga_10", "points_10", "rest_days", "played_10"],
        "err_expected": ["lambda_home", "lambda_away"], "err_actual": "actual_total_goals",
    },
    # World Cup / national teams: no corners in the data source; covariates are the
    # DC model factors, surfaced as columns by the football.predictions_meta view.
    # Its calibration_snapshots table predates the shared shape -> no snapshot write,
    # and it has no is_backtest column (all reconciled rows are walk-forward).
    "worldcup": {
        "schema": "football", "table": "football.predictions_meta",
        "markets": {
            "double_chance": ("p_home_or_draw", "result", None),
            "over_0_5": ("p_over_0_5", "goals", 0.5),
            "over_1_5": ("p_over_1_5", "goals", 1.5),
            "over_2_5": ("p_over_2_5", "goals", 2.5),
            "over_3_5": ("p_over_3_5", "goals", 3.5),
            "over_4_5": ("p_over_4_5", "goals", 4.5),
            "over_5_5": ("p_over_5_5", "goals", 5.5),
            "btts": ("p_btts", "btts", None),
        },
        "num_cols": ["lambda_home", "lambda_away", "atk_home", "atk_away",
                     "def_home", "def_away", "home_adv"],
        "form_keys": [],
        "err_expected": ["lambda_home", "lambda_away"], "err_actual": "actual_total_goals",
        "no_snapshot": True, "no_backtest_col": True, "no_hist_gate": True,
    },
    # MLB totals (the original over/under vertical): per-line meta over the
    # reconciled daily predictions, surfaced by the derived.mlb_predictions_meta
    # view (aliases game_date/team codes to the shared match_date/home_team shape).
    # Dashboard-only — the daily MLB digest keeps its own meta_over_5_5 model.
    "mlb": {
        "schema": "derived", "table": "derived.mlb_predictions_meta",
        "markets": {
            "over_5_5": ("p_over_5_5", "runs", 5.5),
            "over_6_5": ("p_over_6_5", "runs", 6.5),
            "over_7_5": ("p_over_7_5", "runs", 7.5),
            "over_8_5": ("p_over_8_5", "runs", 8.5),
            "over_9_5": ("p_over_9_5", "runs", 9.5),
            "over_10_5": ("p_over_10_5", "runs", 10.5),
            "over_11_5": ("p_over_11_5", "runs", 11.5),
        },
        "num_cols": ["home_starter_era", "away_starter_era", "home_trailing15_rpg",
                     "away_trailing15_rpg", "home_expected_runs", "away_expected_runs",
                     "sigma_used"],
        "form_keys": [],
        "err_expected": ["home_expected_runs", "away_expected_runs"],
        "err_actual": "actual_total_runs",
        "no_snapshot": True, "no_backtest_col": True,
        "wx_key": "game_pk",  # odds.game_weather join (weather covariates)
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
        if w is None or w == "T":  # NFL ties: a winner pick can't be graded
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
    if kind == "runs":
        actual = row.get("actual_total_runs")
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            return None
        return pick_yes == (actual > line)
    actual = row.get("actual_total_goals") if kind == "goals" else row.get("actual_total_corners")
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return None
    return pick_yes == (actual > line)


def _attach_model_err(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Bulk path for the MODEL_ERR_FEATURES covariates over a reconciled frame.

    For each game row: h_model_err / h_model_abs_err = mean signed / absolute
    error (EXPECTED minus ACTUAL total) of the base model over the HOME team's
    last ERR_WINDOW reconciled games strictly before the row's match_date
    (same-date games excluded); a_* likewise for the away team. Window ordering
    is (match_date, id) ascending, take the trailing 8 — exactly the set that
    _team_recent_err's `ORDER BY match_date DESC, id DESC LIMIT 8` selects, so
    the two code paths are value-identical. Leakage-safe: the row's own game
    (and any same-date game) never contributes to its own features.
    """
    df = df.copy()
    if df.empty:
        for c in MODEL_ERR_FEATURES:
            df[c] = np.nan
        return df
    expected = df[spec["err_expected"]].sum(axis=1, min_count=len(spec["err_expected"]))
    err = expected - pd.to_numeric(df[spec["err_actual"]], errors="coerce")
    hist_src = pd.DataFrame({"date": df["match_date"], "id": df["id"],
                             "home": df["home_team"], "away": df["away_team"], "err": err})
    hist_src = hist_src[hist_src["err"].notna()].sort_values(["date", "id"], kind="mergesort")
    hist: dict[str, tuple[list, list]] = {}  # team -> ([dates asc], [errs, (date,id) asc])
    for d, h, a, e in zip(hist_src["date"], hist_src["home"], hist_src["away"], hist_src["err"]):
        for team in (h, a):
            dates, errs = hist.setdefault(team, ([], []))
            dates.append(d)
            errs.append(float(e))
    from bisect import bisect_left

    def _pair(team, d):
        dv = hist.get(team)
        if not dv:
            return np.nan, np.nan
        dates, errs = dv
        i = bisect_left(dates, d)  # strictly-before cut (ties on date excluded)
        w = errs[max(0, i - ERR_WINDOW):i]
        if not w:
            return np.nan, np.nan
        arr = np.asarray(w)
        return float(arr.mean()), float(np.abs(arr).mean())

    cols = {c: [] for c in MODEL_ERR_FEATURES}
    for d, h, a in zip(df["match_date"], df["home_team"], df["away_team"]):
        he, ha = _pair(h, d)
        ae, aa = _pair(a, d)
        cols["h_model_err"].append(he)
        cols["h_model_abs_err"].append(ha)
        cols["a_model_err"].append(ae)
        cols["a_model_abs_err"].append(aa)
    for c, v in cols.items():
        df[c] = v
    return df


def _game_base_frac(rd: dict, spec: dict) -> float:
    """Share of this game's graded markets where the base pick was correct.
    Pure function of the row dict — bulk and live use this same code."""
    ok = tot = 0
    for _market, (pcol, kind, line) in spec["markets"].items():
        p = rd.get(pcol)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        y = _correct(rd, kind, line, float(p))
        if y is None:
            continue
        tot += 1
        ok += 1 if y else 0
    return ok / tot if tot else float("nan")


def _attach_team_rel(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Bulk path for TEAM_REL_FEATURES over a reconciled frame.

    Mirrors _attach_model_err: per-team history of per-game base-correct
    fractions ordered by (match_date, id) ascending; each row gets the mean and
    count of the HOME/AWAY team's trailing REL_WINDOW fractions strictly before
    its match_date (same-date games excluded). Value-identical with
    _team_recent_rel by construction (same ordering, same window, same frac)."""
    df = df.copy()
    if df.empty:
        for c in TEAM_REL_FEATURES:
            df[c] = np.nan
        return df
    fracs = [_game_base_frac(r, spec) for r in df.to_dict("records")]
    hist_src = pd.DataFrame({"date": df["match_date"], "id": df["id"],
                             "home": df["home_team"], "away": df["away_team"],
                             "frac": fracs})
    hist_src = hist_src[hist_src["frac"].notna()].sort_values(["date", "id"], kind="mergesort")
    hist: dict[str, tuple[list, list]] = {}
    for d, h, a, f in zip(hist_src["date"], hist_src["home"], hist_src["away"], hist_src["frac"]):
        for team in (h, a):
            dates, vals = hist.setdefault(team, ([], []))
            dates.append(d)
            vals.append(float(f))
    from bisect import bisect_left

    def _pair(team, d):
        dv = hist.get(team)
        if not dv:
            return np.nan, 0.0
        dates, vals = dv
        i = bisect_left(dates, d)
        w = vals[max(0, i - REL_WINDOW):i]
        if not w:
            return np.nan, 0.0
        return float(np.asarray(w).mean()), float(len(w))

    cols = {c: [] for c in TEAM_REL_FEATURES}
    for d, h, a in zip(df["match_date"], df["home_team"], df["away_team"]):
        hr, hn = _pair(h, d)
        ar, an = _pair(a, d)
        cols["h_base_rel"].append(hr)
        cols["h_base_rel_n"].append(hn)
        cols["a_base_rel"].append(ar)
        cols["a_base_rel_n"].append(an)
    for c, v in cols.items():
        df[c] = v
    return df


def _attach_weather(df: pd.DataFrame, spec: dict, league: str, engine) -> pd.DataFrame:
    """Bulk path for WX_FEATURES: join odds.game_weather by the spec's wx_key
    column (MLB game_pk / NFL event_id). Leagues without a wx_key are returned
    UNCHANGED (their feature schema stays weatherless). Values go through the
    same weather.wx_tuple conversion the live path uses — value-identical with
    weather.live_wx by construction; games without a stored row get NaNs
    (LightGBM routes missing to the default branch)."""
    key_col = spec.get("wx_key")
    if not key_col:
        return df
    df = df.copy()
    if df.empty or key_col not in df.columns:
        for c in WX_FEATURES:
            df[c] = np.nan
        return df
    from sandy import weather as _wx
    wmap = _wx.weather_map(league, engine)
    nan4 = (np.nan,) * 4
    vals = [wmap.get(str(int(k)) if pd.notna(k) else "", nan4) for k in df[key_col]]
    for i, c in enumerate(WX_FEATURES):
        df[c] = [v[i] for v in vals]
    return df


_err_cache: dict = {}
_rel_cache: dict = {}
_err_engine = None


def _team_recent_err(league: str, cfg: Config, team, match_date) -> tuple[float, float]:
    """Single-row (live) path for the model-error covariates: (mean signed,
    mean absolute) EXPECTED-minus-ACTUAL error over *team*'s last ERR_WINDOW
    reconciled games strictly before *match_date*; (nan, nan) with no history.
    Cached per (league, team, date) — at most 2 queries per live game.
    Must stay value-identical with _attach_model_err (audited at review time
    over random historical rows)."""
    global _err_engine
    key = (league, str(team), str(match_date))
    if key in _err_cache:
        return _err_cache[key]
    spec = SPECS[league]
    # Fetch raw columns and subtract in PYTHON (float64), not in SQL: several
    # columns are float4 in Postgres and server-side arithmetic would round
    # differently from the pandas bulk path — this keeps both bit-identical.
    sel_cols = ", ".join(spec["err_expected"] + [spec["err_actual"]])
    exp_not_null = " AND ".join(f"{c} IS NOT NULL" for c in spec["err_expected"])
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    if _err_engine is None:
        _err_engine = create_engine(cfg)
    with _err_engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT {sel_cols}
            FROM {spec['table']}
            WHERE outcome_filled_at_utc IS NOT NULL
              AND {exp_not_null} AND {spec['err_actual']} IS NOT NULL
              AND (home_team = :t OR away_team = :t)
              AND match_date < :d{extra}
            ORDER BY match_date DESC, id DESC
            LIMIT :n
        """), {"t": team, "d": match_date, "n": ERR_WINDOW}).fetchall()
    errs = []
    for r in rows:
        exp = float(r[0])
        for v in r[1:-1]:
            exp += float(v)
        errs.append(exp - float(r[-1]))
    if errs:
        arr = np.asarray(errs)
        out = (float(arr.mean()), float(np.abs(arr).mean()))
    else:
        out = (float("nan"), float("nan"))
    _err_cache[key] = out
    return out


def _team_recent_rel(league: str, cfg: Config, team, match_date) -> tuple[float, float]:
    """Single-row (live) path for TEAM_REL_FEATURES: (mean base-correct
    fraction, count) over *team*'s last REL_WINDOW reconciled games strictly
    before *match_date*. Fetches a 3x window and keeps the trailing REL_WINDOW
    games with a computable fraction — the exact set _attach_team_rel selects.
    Cached per (league, team, date). Must stay value-identical with the bulk
    path (audited like the model-err covariates)."""
    global _err_engine
    key = (league, str(team), str(match_date))
    if key in _rel_cache:
        return _rel_cache[key]
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    if _err_engine is None:
        _err_engine = create_engine(cfg)
    with _err_engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT * FROM {spec['table']}
            WHERE outcome_filled_at_utc IS NOT NULL
              AND (home_team = :t OR away_team = :t)
              AND match_date < :d{extra}
            ORDER BY match_date DESC, id DESC
            LIMIT :n
        """), {"t": team, "d": match_date, "n": REL_WINDOW * 3}).mappings().fetchall()
    fracs = []
    for r in rows:  # newest → oldest; stop at REL_WINDOW computable games
        f = _game_base_frac(dict(r), spec)
        if not np.isnan(f):
            fracs.append(f)
        if len(fracs) >= REL_WINDOW:
            break
    if fracs:
        out = (float(np.asarray(fracs).mean()), float(len(fracs)))
    else:
        out = (float("nan"), 0.0)
    _rel_cache[key] = out
    return out


def _num(v) -> float:
    """NaN-safe float of a row value."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _row_features(row, spec, market, p, league: str | None = None,
                  cfg: Config | None = None) -> dict:
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
    # ---- pure row+spec features (identical bulk/live by construction) -------
    # expected_total: the base model's expected total for its primary quantity
    # (λh+λa, exp_total, expected runs). line_value/margin/z_margin: the
    # numeric market line, distance of the KIND-matched expected total to it,
    # and that distance standardized (Poisson sd=sqrt(expected) for goals/
    # corners/runs; the row's sigma_total for points). NaN-safe throughout.
    exp_base = 0.0
    for c in spec["err_expected"]:
        exp_base += _num(row.get(c))
    out["expected_total"] = exp_base
    _pcol, kind, line = spec["markets"][market]
    out["line_value"] = float(line) if line is not None else np.nan
    margin = z_margin = float("nan")
    if line is not None:
        if kind == "corners":
            exp_kind = _num(row.get("corner_lambda_home")) + _num(row.get("corner_lambda_away"))
        else:
            exp_kind = exp_base
        margin = exp_kind - float(line)
        if kind == "points":
            sd = _num(row.get("sigma_total"))
        else:  # goals / corners / runs: Poisson-ish
            sd = np.sqrt(exp_kind) if exp_kind > 0 else float("nan")
        if sd and not np.isnan(sd) and sd > 0:
            z_margin = margin / sd
    out["margin"] = margin
    out["z_margin"] = z_margin
    # Season phase as month-of-year one-hots (trivially identical in bulk and
    # live paths — chosen over days-since-season-start for path equality).
    try:
        month = int(pd.Timestamp(row.get("match_date")).month)
    except (TypeError, ValueError):
        month = 0
    for m_i in range(1, 13):
        out[f"mon_{m_i}"] = 1.0 if m_i == month else 0.0
    # Model's-own-recent-errors covariates: bulk-attached upstream when present
    # (_attach_model_err on training/dashboard frames); live rows straight off
    # the DB lack them → cached single-row lookup, identical by construction.
    if all(k in row for k in MODEL_ERR_FEATURES):
        for k in MODEL_ERR_FEATURES:
            v = row.get(k)
            out[k] = float(v) if v is not None else np.nan
    elif league is not None and cfg is not None:
        he, ha = _team_recent_err(league, cfg, row.get("home_team"), row.get("match_date"))
        ae, aa = _team_recent_err(league, cfg, row.get("away_team"), row.get("match_date"))
        out["h_model_err"], out["h_model_abs_err"] = he, ha
        out["a_model_err"], out["a_model_abs_err"] = ae, aa
    else:
        for k in MODEL_ERR_FEATURES:
            out[k] = np.nan
    # Team base-model reliability: same bulk-attached/live-lookup contract.
    if all(k in row for k in TEAM_REL_FEATURES):
        for k in TEAM_REL_FEATURES:
            v = row.get(k)
            out[k] = float(v) if v is not None else np.nan
    elif league is not None and cfg is not None:
        hr, hn = _team_recent_rel(league, cfg, row.get("home_team"), row.get("match_date"))
        ar, an = _team_recent_rel(league, cfg, row.get("away_team"), row.get("match_date"))
        out["h_base_rel"], out["h_base_rel_n"] = hr, hn
        out["a_base_rel"], out["a_base_rel_n"] = ar, an
    else:
        out["h_base_rel"], out["h_base_rel_n"] = np.nan, np.nan
        out["a_base_rel"], out["a_base_rel_n"] = np.nan, np.nan
    # Weather covariates (only for specs with a wx_key — MLB/NFL): bulk-attached
    # upstream when present; live rows get the cached single-row lookup (stored
    # weather, else on-the-fly forecast for pending games; NaN on any failure).
    if spec.get("wx_key"):
        if all(k in row for k in WX_FEATURES):
            for k in WX_FEATURES:
                out[k] = _num(row.get(k))
        elif league is not None and cfg is not None:
            from sandy import weather as _wx
            (out["wx_temp"], out["wx_wind"], out["wx_precip"],
             out["wx_dome"]) = _wx.live_wx(league, row.get(spec["wx_key"]), cfg)
        else:
            for k in WX_FEATURES:
                out[k] = np.nan
    for m in spec["markets"]:
        out[f"mkt_{m}"] = 1.0 if m == market else 0.0
    return out


def _frame(engine, league: str, with_ids: bool = False) -> pd.DataFrame:
    """One row per (reconciled game × market) with the meta features + _y/_date.
    with_ids=True additionally keeps _game_id/_market/_home/_away bookkeeping
    columns (used by betrefine's OOF pipeline; train_meta never passes it, so
    its feat_cols stay unchanged)."""
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    with engine.begin() as conn:
        df = pd.read_sql(text(
            f"SELECT * FROM {spec['table']} WHERE outcome_filled_at_utc IS NOT NULL{extra}"), conn)
    df = _attach_model_err(df, spec)
    df = _attach_team_rel(df, spec)
    df = _attach_weather(df, spec, league, engine)
    rows, ys, dates, gids, mkts, homes, aways = [], [], [], [], [], [], []
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
            if with_ids:
                gids.append(rd["id"])
                mkts.append(market)
                homes.append(rd.get("home_team"))
                aways.append(rd.get("away_team"))
    X = pd.DataFrame(rows)
    X["_y"] = ys
    X["_date"] = dates
    if with_ids:
        X["_game_id"] = gids
        X["_market"] = mkts
        X["_home"] = homes
        X["_away"] = aways
    return X


def _wilson_lb(correct: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound — a small-sample-honest floor on accuracy, so a
    lucky 26/31 rung can never outrank a solid 474/592 one."""
    if n == 0:
        return 0.0
    phat = correct / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return (center - rad) / denom


def train_meta(league: str, config: Config | None = None) -> dict:
    """Train + calibrate + evaluate on a THREE-WAY chronological split:
      train (oldest 60%)  → fits the booster (with early stopping vs calib)
      calib (middle 20%)  → fits the isotonic map + selects thresholds (Wilson LB)
      test  (final 20%)   → NEVER touched by any choice; all reported ladders/acc
    So every number shown downstream is honest out-of-sample performance."""
    import lightgbm as lgb
    cfg = config or load_config()
    engine = create_engine(cfg)
    X = _frame(engine, league)
    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"{league}: only {len(X)} meta rows (<{MIN_TRAIN_ROWS})")
    X = X.sort_values("_date").reset_index(drop=True)
    c1, c2 = X["_date"].quantile(0.6), X["_date"].quantile(0.8)
    train = X[X["_date"] <= c1]
    calib = X[(X["_date"] > c1) & (X["_date"] <= c2)]
    test = X[X["_date"] > c2]
    feat_cols = [c for c in X.columns if c not in ("_y", "_date")]
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    # P(correct) must be non-decreasing in the pick's own confidence.
    mono = [1 if c == "conf" else 0 for c in feat_cols]
    base_params = dict(objective="binary", feature_fraction=0.85, bagging_fraction=0.85,
                       bagging_freq=1, verbose=-1, seed=42, monotone_constraints=mono)
    # Small honest grid, selected on CALIB AUC only — test stays untouched by
    # every choice (booster, hyperparams, isotonic, thresholds all come from
    # train+calib; test is evaluated exactly once, below).
    cy = calib["_y"].to_numpy()
    booster, chosen, calib_auc = None, None, -1.0
    for hp in [dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=mdl)
               for lr in (0.05, 0.03, 0.02) for nl in (15, 31) for mdl in (40, 100, 300)]:
        b = lgb.train({**base_params, **hp},
                      lgb.Dataset(train[feat_cols], label=train["_y"]),
                      num_boost_round=800,
                      valid_sets=[lgb.Dataset(calib[feat_cols], label=calib["_y"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        cauc = float(roc_auc_score(
            cy, b.predict(calib[feat_cols], num_iteration=b.best_iteration)))
        if cauc > calib_auc:
            booster, chosen, calib_auc = b, hp, cauc
    # Isotonic fitted on CALIB (booster never trained on it)…
    cp_raw = booster.predict(calib[feat_cols], num_iteration=booster.best_iteration)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(cp_raw, cy.astype(float))
    cp = iso.predict(cp_raw)
    # …and every REPORTED number comes from TEST, which nothing was tuned on.
    hp_raw = booster.predict(test[feat_cols], num_iteration=booster.best_iteration)
    hy = test["_y"].to_numpy()
    hp = iso.predict(hp_raw)
    hold = test  # reported rows
    auc = round(float(roc_auc_score(hy, hp_raw)), 4)

    def _ladder(mask, probs=None, ys=None):
        probs = hp if probs is None else probs
        ys = hy if ys is None else ys
        rows = []
        for thr in THRESHOLDS:
            m = mask & (probs >= thr)
            n = int(m.sum())
            rows.append({"thr": thr, "n": n, "correct": int(ys[m].sum()),
                         "acc": round(float(ys[m].mean()), 4) if n else None})
        return rows

    all_mask = np.ones(len(hy), dtype=bool)
    table = _ladder(all_mask)
    # The user's rule (threshold that maximizes accuracy), made small-sample-honest:
    # selected on CALIB by Wilson lower bound, n>=50 — never on the reported test set.
    calib_table = _ladder(np.ones(len(cy), dtype=bool), probs=cp, ys=cy)
    viable = [t for t in calib_table if (t["n"] or 0) >= 50]
    rec = (max(viable, key=lambda t: _wilson_lb(t["correct"], t["n"]))["thr"]
           if viable else None)

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

    # PER-EXACT-LINE ladders (reported from TEST) + per-line recommended threshold
    # SELECTED on CALIB via Wilson lower bound (n>=30; global fallback when thin).
    row_market = np.array([mkt_cols[i][4:] for i in hold_markets])
    calib_markets = calib[mkt_cols].to_numpy().argmax(axis=1)
    calib_row_market = np.array([mkt_cols[i][4:] for i in calib_markets])
    eval_by_market, threshold_by_market = {}, {}
    for m in SPECS[league]["markets"]:
        mm = row_market == m
        eval_by_market[m] = {"table": _ladder(mm), "below": _below(mm)}
        clad = _ladder(calib_row_market == m, probs=cp, ys=cy)
        viable_m = [t for t in clad if (t["n"] or 0) >= 30]
        threshold_by_market[m] = (max(viable_m, key=lambda t: _wilson_lb(t["correct"], t["n"]))["thr"]
                                  if viable_m else rec)
    # Production = the split-trained booster + its isotonic map. (Retraining on ALL
    # data would orphan the calibration — integrity of the 🤖 probability wins.)
    imp = sorted(zip(feat_cols, booster.feature_importance("gain")),
                 key=lambda x: -x[1])[:10]
    artifact = {"model_str": booster.model_to_string(num_iteration=booster.best_iteration),
                "iso": iso, "features": feat_cols,
                "threshold": rec, "eval_table": table, "auc": auc,
                "calib_auc": round(calib_auc, 4), "params": chosen,
                "eval_below": _below(all_mask), "eval_by_group": eval_by_group,
                "eval_by_market": eval_by_market, "threshold_by_market": threshold_by_market,
                "importances": [(f, round(float(g), 1)) for f, g in imp],
                "trained_rows": len(train), "calib_rows": len(calib), "holdout_rows": len(hold),
                "best_iteration": int(booster.best_iteration or 0),
                "trained_at": date.today().isoformat()}
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    # League-scoped schemas (soccer) have a NOT NULL league column on snapshots.
    lg_col, lg_val = ("league, ", ":lg, ") if SPECS[league].get("league") else ("", "")
    if SPECS[league].get("no_snapshot"):
        logger.info("%s meta trained: auc=%s thr=%s params=%s (no snapshot table)",
                    league, auc, rec, chosen)
        return {"rows": len(X), "threshold": rec, "eval_table": table, "auc": auc,
                "calib_auc": round(calib_auc, 4), "params": chosen,
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
    logger.info("%s meta trained: auc=%s thr=%s params=%s", league, auc, rec, chosen)
    return {"rows": len(X), "threshold": rec, "eval_table": table, "auc": auc,
            "calib_auc": round(calib_auc, 4), "params": chosen,
            "threshold_by_market": threshold_by_market,
            "importances": artifact["importances"]}


def _acc_at_threshold(artifact: dict) -> float | None:
    """Holdout accuracy at the artifact's own recommended threshold."""
    thr = artifact.get("threshold")
    if thr is None:
        return None
    for t in artifact.get("eval_table") or []:
        if t["thr"] == thr:
            return t["acc"]
    return None


def retrain_gated(leagues: list[str] | None = None, config: Config | None = None) -> dict:
    """Retrain every league's meta with a global honesty gate.

    Per league: back up the current artifact to models/backup_aucpass/, record
    its test AUC + accuracy@recommended-threshold, retrain, record the new
    numbers. SHIP RULE: keep the new artifacts only if the mean test AUC
    improves AND the mean acc@threshold does not drop >1pp; additionally any
    single league regressing >0.02 AUC is individually reverted to its backup.
    If the global gate fails, ALL leagues are reverted. Returns the full
    old->new report (also written to models/tuning/meta_retrain_report.json)."""
    import shutil
    cfg = config or load_config()
    mdir = cfg.model.model_dir
    bdir = mdir / "backup_aucpass"
    bdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for lg in (leagues or list(SPECS)):
        path = mdir / f"{lg}_meta.pkl"
        old: dict = {}
        if path.exists():
            shutil.copy2(path, bdir / path.name)
            with open(path, "rb") as f:
                a = pickle.load(f)
            old = {"auc": a.get("auc"), "thr": a.get("threshold"),
                   "acc_at_thr": _acc_at_threshold(a), "rows": a.get("holdout_rows")}
        try:
            res = train_meta(lg, cfg)
            with open(path, "rb") as f:
                a = pickle.load(f)
            new = {"auc": a.get("auc"), "thr": a.get("threshold"),
                   "acc_at_thr": _acc_at_threshold(a), "rows": a.get("holdout_rows"),
                   "params": res.get("params")}
        except Exception as e:  # noqa: BLE001 -- a failed league keeps its backup
            new = {"error": str(e)}
            if (bdir / path.name).exists():
                shutil.copy2(bdir / path.name, path)
        report[lg] = {"old": old, "new": new}
        logger.info("retrain_gated %s: old=%s new=%s", lg, old, new)
    scored = {lg: r for lg, r in report.items()
              if r["old"].get("auc") is not None and r["new"].get("auc") is not None}
    accs = {lg: r for lg, r in scored.items()
            if r["old"].get("acc_at_thr") is not None and r["new"].get("acc_at_thr") is not None}
    summary: dict = {"leagues": report, "reverted": []}
    if scored:
        mean_old_auc = sum(r["old"]["auc"] for r in scored.values()) / len(scored)
        mean_new_auc = sum(r["new"]["auc"] for r in scored.values()) / len(scored)
        mean_old_acc = (sum(r["old"]["acc_at_thr"] for r in accs.values()) / len(accs)) if accs else None
        mean_new_acc = (sum(r["new"]["acc_at_thr"] for r in accs.values()) / len(accs)) if accs else None
        ship = mean_new_auc > mean_old_auc and (
            mean_old_acc is None or mean_new_acc >= mean_old_acc - 0.01)
        if ship:
            for lg, r in scored.items():
                if r["old"]["auc"] - r["new"]["auc"] > 0.02:
                    shutil.copy2(bdir / f"{lg}_meta.pkl", mdir / f"{lg}_meta.pkl")
                    summary["reverted"].append(lg)
        else:
            for lg in report:
                b = bdir / f"{lg}_meta.pkl"
                if b.exists():
                    shutil.copy2(b, mdir / f"{lg}_meta.pkl")
                    summary["reverted"].append(lg)
        summary.update({"mean_old_auc": round(mean_old_auc, 4),
                        "mean_new_auc": round(mean_new_auc, 4),
                        "mean_old_acc": round(mean_old_acc, 4) if mean_old_acc is not None else None,
                        "mean_new_acc": round(mean_new_acc, 4) if mean_new_acc is not None else None,
                        "ship_all_new": ship})
    out = mdir / "tuning" / "meta_retrain_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    return summary


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
    feats = _row_features(row_dict, SPECS[league], market, p, league=league, cfg=cfg)
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
