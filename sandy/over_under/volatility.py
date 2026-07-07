"""Matchup-specific volatility model for over/under predictions.

Trains a LightGBM regression model that predicts |actual_total - predicted_total|
per game, giving a matchup-specific σ for the normal approximation in
compute_over_under_probabilities().

The volatility model uses the same 10 GAME_FEATURE_NAMES as the runs model.
If the volatility model artifact doesn't exist, predict_sigma() falls back
to 3.3 (data-driven average absolute error from historical analysis).

Requirements: 12.5, 13.1
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.config import Config, load_config
from sandy.features.schema import GAME_FEATURE_NAMES, GAME_FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger
from sandy.schemas import ModelArtifact
from sandy.train.artifact import load_artifact, save_artifact
from sandy.train.split import chronological_split

logger = get_logger("over_under.volatility")

# Fallback σ when the volatility model is not available (data-driven average)
DEFAULT_SIGMA: float = 3.3

# Game-time weather covariates (odds.game_weather via sandy/weather.py), the
# optional wx extension of the σ feature set. Order matters (model input).
# Missing weather -> NaN (LightGBM routes missing to the default branch); the
# ten base features keep their historical 0.0 default.
#
# DORMANT — judge gate 2026-07-07 said NO to weather in the MLB BASE models.
# In-memory walk-forward over the 7,794 backtest games (in-harness baseline
# reproduced the stored rows to 1e-6), pooled 7-line log-loss on the untouched
# final 25%:  base 0.64181 · wx-in-sigma 0.64183 · wx-total-adjust 0.64360 ·
# both 0.64366. Neither candidate improved -> production keeps wx=False and no
# total adjustment (weather stays at the MLB META level only, where it passed:
# AUC .6058->.6119). Machinery kept for future re-gating as data accumulates;
# do NOT flip defaults/call sites without a fresh judge pass.
WX_SIGMA_FEATURES: list[str] = ["wx_temp", "wx_wind", "wx_precip", "wx_dome"]

# Minimum open-air games with weather for the runs-total adjustment fit;
# below this the adjustment is disabled (None -> 0.0 everywhere).
WX_ADJ_MIN_OPEN_AIR = 150

# LightGBM hyperparameters for volatility (regression, same as runs model)
_LGB_VOLATILITY_PARAMS = {
    "objective": "regression",
    "metric": ["l1", "l2"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def attach_wx(df: pd.DataFrame, engine, wmap: dict | None = None) -> pd.DataFrame:
    """Add the WX_SIGMA_FEATURES columns to a per-game frame by joining
    odds.game_weather on game_pk (NaN where no weather is stored — e.g. the
    2022 season, which predates the weather backfill). *wmap* lets callers
    reuse one weather_map() across many blocks (walk-forward backtests)."""
    from sandy.weather import weather_map

    if wmap is None:
        wmap = weather_map("mlb", engine)
    df = df.copy()
    nan4 = (float("nan"),) * 4
    vals = [wmap.get(str(int(pk)), nan4) for pk in df["game_pk"]]
    for i, name in enumerate(WX_SIGMA_FEATURES):
        df[name] = [v[i] for v in vals]
    return df


def train_volatility_model(
    config: Config,
    *,
    seed: int = 42,
    training_window: tuple[date, date] | None = None,
    runs_artifact: ModelArtifact | None = None,
    wx: bool = False,
    frame: pd.DataFrame | None = None,
) -> ModelArtifact:
    """Train a model that predicts |actual_total - predicted_total| per game.

    Steps:
    1. Load all games from derived.game_features joined with raw.games (actual scores)
    2. For each game, run the runs model to get predicted_home + predicted_away
    3. Compute residual = abs(actual_total - predicted_total)
    4. Train LightGBM regression: game features (+ weather when wx=True) → residual
    5. Return ModelArtifact with target_name="volatility"

    wx=True appends WX_SIGMA_FEATURES (temp/wind/precip/dome from
    odds.game_weather); the artifact's feature_names record the choice, and
    predict_sigma builds its input from artifact.feature_names — so old and
    new artifacts both score correctly. *frame* lets walk-forward callers
    pass a pre-loaded _load_volatility_frame result (avoids reloading it for
    each model variant); production callers leave it None.
    """
    from sandy.db import create_engine, get_connection

    engine = create_engine(config)

    if frame is not None:
        df = frame
    else:
        with get_connection(engine) as conn:
            df = _load_volatility_frame(conn, config, training_window, runs_artifact=runs_artifact)

    feature_names = list(GAME_FEATURE_NAMES)
    if wx:
        if not all(c in df.columns for c in WX_SIGMA_FEATURES):
            df = attach_wx(df, engine)
        feature_names += WX_SIGMA_FEATURES

    if len(df) < 100:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows available. "
            f"Need at least 100 games with features and actual scores."
        )

    # Chronological split
    split = chronological_split(df, val_fraction=0.15)
    train_df = split.train
    val_df = split.val

    logger.info(
        "Volatility training split complete",
        extra={
            "component": "over_under.volatility",
            "target": "volatility",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_train_games": split.n_train_games,
            "n_val_games": split.n_val_games,
        },
    )

    # Build LightGBM datasets
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["residual"].values.astype(np.float32)
    X_val = val_df[feature_names].values.astype(np.float32)
    y_val = val_df["residual"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_names, reference=lgb_train
    )

    # Fit
    params = {**_LGB_VOLATILITY_PARAMS, "seed": seed}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # Validation metrics
    val_preds = booster.predict(X_val)
    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(mean_squared_error(y_val, val_preds) ** 0.5)

    logger.info(
        "Volatility training metrics",
        extra={
            "component": "over_under.volatility",
            "target": "volatility",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_residual": round(float(y_train.mean()), 4),
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "best_iteration": booster.best_iteration,
        },
    )

    return ModelArtifact(
        model=booster,
        feature_names=feature_names,
        feature_schema_version=GAME_FEATURE_SCHEMA_VERSION,
        training_window_start=df["game_date"].min(),
        training_window_end=df["game_date"].max(),
        created_at=datetime.now(timezone.utc),
        target_name="volatility",
    )


def sigma_feature_row(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """The σ model's input row, driven by the ARTIFACT's feature list so both
    the 10-feature and the wx-extended artifacts score correctly. Defaults:
    base game features -> 0.0 (historical behavior), wx features -> NaN
    (LightGBM missing-value branch — a game without weather behaves like the
    weatherless model saw it)."""
    return np.array(
        [[features.get(name, float("nan") if name in WX_SIGMA_FEATURES else 0.0)
          for name in feature_names]],
        dtype=np.float32,
    )


def predict_sigma(features: dict[str, float], config: Config) -> float:
    """Load the volatility model and predict σ for a given matchup.

    Returns the predicted absolute error (= σ for normal approximation).
    Falls back to 3.3 (data-driven average) if model not available.
    """
    artifact_path = config.model.artifact_path("volatility")

    try:
        artifact = load_artifact(artifact_path, expected_target="volatility")
    except (FileNotFoundError, Exception) as exc:
        logger.debug(
            f"Volatility model not available, using default σ={DEFAULT_SIGMA}: {exc}",
            extra={"component": "over_under.volatility"},
        )
        return DEFAULT_SIGMA

    # Build feature array in the artifact's own feature order
    feature_array = sigma_feature_row(features, list(artifact.feature_names))

    prediction = artifact.model.predict(feature_array)[0]

    # Ensure σ is positive and reasonable (floor at 1.0, cap at 8.0)
    sigma = float(max(1.0, min(8.0, prediction)))

    return sigma


# ---------------------------------------------------------------------------
# Weather adjustment of the expected TOTAL (open-air runs physics)
# ---------------------------------------------------------------------------


def fit_wx_runs_adjustment(
    config: Config,
    *,
    training_window: tuple[date, date] | None = None,
    runs_artifact: ModelArtifact | None = None,
    frame: pd.DataFrame | None = None,
    wmap: dict | None = None,
) -> dict | None:
    """Fit total_adj = b1·wind_kmh + b2·(temp_c-20) + b3·precip_mm on the runs
    model's SIGNED total residuals (actual − predicted) over OPEN-AIR games
    (wx_dome < 1; retractables half-weighted by (1-wx_dome)) in the training
    window. No intercept — at (wind 0, 20°C, dry) the adjustment is zero, and
    dome games (wx_dome=1) are untouched by construction at apply time.

    Returns {"coefs": [b_wind, b_temp, b_precip], "n_open": int} or None when
    fewer than WX_ADJ_MIN_OPEN_AIR usable games / weather unavailable — the
    caller then applies no adjustment (today's exact behavior)."""
    from sandy.db import create_engine, get_connection

    try:
        engine = create_engine(config)
        if frame is None:
            with get_connection(engine) as conn:
                frame = _load_volatility_frame(
                    conn, config, training_window, runs_artifact=runs_artifact
                )
        if frame.empty:
            return None
        if not all(c in frame.columns for c in WX_SIGMA_FEATURES):
            frame = attach_wx(frame, engine, wmap=wmap)
        resid = frame["actual_total"].to_numpy(dtype=float) - \
            frame["predicted_total"].to_numpy(dtype=float)
        dome = frame["wx_dome"].to_numpy(dtype=float)
        X = np.column_stack([
            frame["wx_wind"].to_numpy(dtype=float),
            frame["wx_temp"].to_numpy(dtype=float) - 20.0,
            frame["wx_precip"].to_numpy(dtype=float),
        ])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(dome) & (dome < 1.0) & np.isfinite(resid)
        n_open = int(ok.sum())
        if n_open < WX_ADJ_MIN_OPEN_AIR:
            logger.info(
                f"wx runs adjustment: only {n_open} open-air games — disabled",
                extra={"component": "over_under.volatility"},
            )
            return None
        sw = np.sqrt(1.0 - dome[ok])
        b, *_ = np.linalg.lstsq(X[ok] * sw[:, None], resid[ok] * sw, rcond=None)
        return {"coefs": [float(v) for v in b], "n_open": n_open}
    except Exception as exc:  # noqa: BLE001 — weather must never break training
        logger.warning(
            f"wx runs adjustment fit failed ({exc}) — disabled",
            extra={"component": "over_under.volatility"},
        )
        return None


def wx_total_adjustment(adj: dict | None, wx) -> float:
    """Apply a fit_wx_runs_adjustment result to one game.

    wx = (wx_temp, wx_wind, wx_precip, wx_dome) — the weather.wx_tuple /
    live_wx order. NaN-safe: no fit, no weather, or any NaN -> 0.0."""
    if not adj or not adj.get("coefs") or wx is None:
        return 0.0
    t, wnd, pr, dome = (float(v) for v in wx)
    if not np.all(np.isfinite([t, wnd, pr, dome])):
        return 0.0
    b1, b2, b3 = adj["coefs"]
    return float((1.0 - dome) * (b1 * wnd + b2 * (t - 20.0) + b3 * pr))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_volatility_frame(
    conn: Connection,
    config: Config,
    training_window: tuple[date, date] | None,
    runs_artifact: ModelArtifact | None = None,
) -> pd.DataFrame:
    """Load game features + actual totals, compute predicted totals via runs model.

    For each game:
    - Get the 10 game features (home perspective)
    - Get actual_total = home_score + away_score
    - Predict home_runs + away_runs using the runs model
    - residual = abs(actual_total - predicted_total)

    If *runs_artifact* is provided it is used directly (e.g. a walk-forward
    backtest's in-memory as-of model); otherwise the production runs artifact
    is loaded from disk (unchanged default behavior).
    """
    if runs_artifact is None:
        # Load the production runs model for predictions
        runs_artifact_path = config.model.artifact_path("runs")
        try:
            runs_artifact = load_artifact(runs_artifact_path, expected_target="runs")
        except FileNotFoundError:
            raise ValueError(
                "Runs model not found. Train the runs model first with "
                "'sandy train --target runs' before training volatility."
            )

    where_clause = ""
    params: dict = {}

    if training_window is not None:
        start, end = training_window
        where_clause = "AND g.game_date BETWEEN :start AND :end"
        params = {"start": start, "end": end}

    feature_cols = ", ".join(f"f.{name}" for name in GAME_FEATURE_NAMES)

    # Load home-perspective features with actual scores
    query = text(f"""
        SELECT
            f.game_pk,
            g.game_date,
            g.home_score,
            g.away_score,
            {feature_cols}
        FROM derived.game_features f
        JOIN raw.games g
            ON g.game_pk = f.game_pk
        WHERE f.feature_schema_version = :schema_version
          AND f.is_home = true
          AND g.status = 'Final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          {where_clause}
        ORDER BY g.game_date, f.game_pk
    """)

    params["schema_version"] = GAME_FEATURE_SCHEMA_VERSION
    result = conn.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if df.empty:
        return df

    # Coerce types
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in GAME_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Compute actual total
    df["actual_total"] = df["home_score"].astype(int) + df["away_score"].astype(int)

    # Predict runs for each game using the runs model
    X = df[GAME_FEATURE_NAMES].values.astype(np.float32)
    home_preds = runs_artifact.model.predict(X)

    # For away predictions, flip is_home and swap home/away features
    X_away = df[GAME_FEATURE_NAMES].values.astype(np.float32).copy()
    # Swap home/away starter ERA and WHIP
    is_home_idx = GAME_FEATURE_NAMES.index("is_home")
    home_era_idx = GAME_FEATURE_NAMES.index("home_starter_era")
    home_whip_idx = GAME_FEATURE_NAMES.index("home_starter_whip")
    away_era_idx = GAME_FEATURE_NAMES.index("away_starter_era")
    away_whip_idx = GAME_FEATURE_NAMES.index("away_starter_whip")
    home_rpg_idx = GAME_FEATURE_NAMES.index("home_trailing15_rpg")
    away_rpg_idx = GAME_FEATURE_NAMES.index("away_trailing15_rpg")
    home_obp_idx = GAME_FEATURE_NAMES.index("home_season_obp")
    away_obp_idx = GAME_FEATURE_NAMES.index("away_season_obp")

    # Flip is_home
    X_away[:, is_home_idx] = 0.0
    # Swap home/away features
    X_away[:, home_era_idx], X_away[:, away_era_idx] = (
        X[:, away_era_idx].copy(),
        X[:, home_era_idx].copy(),
    )
    X_away[:, home_whip_idx], X_away[:, away_whip_idx] = (
        X[:, away_whip_idx].copy(),
        X[:, home_whip_idx].copy(),
    )
    X_away[:, home_rpg_idx], X_away[:, away_rpg_idx] = (
        X[:, away_rpg_idx].copy(),
        X[:, home_rpg_idx].copy(),
    )
    X_away[:, home_obp_idx], X_away[:, away_obp_idx] = (
        X[:, away_obp_idx].copy(),
        X[:, home_obp_idx].copy(),
    )

    away_preds = runs_artifact.model.predict(X_away)

    # Compute predicted total and residual
    df["predicted_total"] = home_preds + away_preds
    df["residual"] = np.abs(df["actual_total"].values - df["predicted_total"].values)

    logger.info(
        f"Loaded {len(df)} games for volatility training, "
        f"mean residual: {df['residual'].mean():.2f}",
        extra={"component": "over_under.volatility", "n_games": len(df)},
    )

    return df


__all__ = [
    "DEFAULT_SIGMA",
    "WX_SIGMA_FEATURES",
    "attach_wx",
    "fit_wx_runs_adjustment",
    "predict_sigma",
    "sigma_feature_row",
    "train_volatility_model",
    "wx_total_adjustment",
]
