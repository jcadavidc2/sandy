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


def train_volatility_model(
    config: Config,
    *,
    seed: int = 42,
    training_window: tuple[date, date] | None = None,
) -> ModelArtifact:
    """Train a model that predicts |actual_total - predicted_total| per game.

    Steps:
    1. Load all games from derived.game_features joined with raw.games (actual scores)
    2. For each game, run the runs model to get predicted_home + predicted_away
    3. Compute residual = abs(actual_total - predicted_total)
    4. Train LightGBM regression: 10 game features → residual
    5. Return ModelArtifact with target_name="volatility"
    """
    from sandy.db import create_engine, get_connection

    engine = create_engine(config)

    with get_connection(engine) as conn:
        df = _load_volatility_frame(conn, config, training_window)

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
    X_train = train_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["residual"].values.astype(np.float32)
    X_val = val_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["residual"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=GAME_FEATURE_NAMES)
    lgb_val = lgb.Dataset(
        X_val, label=y_val, feature_name=GAME_FEATURE_NAMES, reference=lgb_train
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
        feature_names=GAME_FEATURE_NAMES,
        feature_schema_version=GAME_FEATURE_SCHEMA_VERSION,
        training_window_start=df["game_date"].min(),
        training_window_end=df["game_date"].max(),
        created_at=datetime.now(timezone.utc),
        target_name="volatility",
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

    # Build feature array in the correct order
    feature_array = np.array(
        [[features.get(name, 0.0) for name in GAME_FEATURE_NAMES]],
        dtype=np.float32,
    )

    prediction = artifact.model.predict(feature_array)[0]

    # Ensure σ is positive and reasonable (floor at 1.0, cap at 8.0)
    sigma = float(max(1.0, min(8.0, prediction)))

    return sigma


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_volatility_frame(
    conn: Connection,
    config: Config,
    training_window: tuple[date, date] | None,
) -> pd.DataFrame:
    """Load game features + actual totals, compute predicted totals via runs model.

    For each game:
    - Get the 10 game features (home perspective)
    - Get actual_total = home_score + away_score
    - Predict home_runs + away_runs using the runs model
    - residual = abs(actual_total - predicted_total)
    """
    # First, load the runs model for predictions
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
    "predict_sigma",
    "train_volatility_model",
]
