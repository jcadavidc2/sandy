"""Over/Under Meta-Model — predicts P(our O5.5 prediction will be correct).

A binary classifier trained on reconciled over/under outcomes. Uses the
prediction context (probability, sigma, pitcher ERAs, trailing RPG, ballpark,
fallback flag, total expected runs) to learn which combinations of features
lead to correct O5.5 predictions.

Integrated into the nightly pipeline (retrain after reconciliation) and
morning predictions (score each game with P(correct)).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config
from sandy.logging import get_logger
from sandy.schemas import ModelArtifact
from sandy.train.artifact import load_artifact, save_artifact

logger = get_logger("over_under.meta_model")

# Features used by the meta-model (all available in over_under_outcomes table)
META_FEATURE_NAMES: list[str] = [
    "p_over_5_5",
    "sigma_used",
    "home_starter_era",
    "away_starter_era",
    "home_trailing15_rpg",
    "away_trailing15_rpg",
    "ballpark_id",
    "pitcher_fallback",
    "total_expected_runs",
]

# Minimum reconciled games required to train
MIN_TRAINING_SAMPLES: int = 50

# LightGBM hyperparameters — heavily regularized for small data
_LGB_META_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
    "is_unbalance": True,
}


def train_meta_model(engine: Engine, config: Config, seed: int = 42) -> ModelArtifact:
    """Train the meta-model on reconciled over/under outcomes.

    Uses all reconciled games where actual_over_5_5 is populated.
    Returns a ModelArtifact with target_name="meta_over_5_5".

    Raises ValueError if insufficient training data (< 50 games).
    """
    with engine.connect() as conn:
        df = _load_meta_training_data(conn)

    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Insufficient training data: {len(df)} games available, "
            f"need at least {MIN_TRAINING_SAMPLES}."
        )

    # Chronological split: last 20% as validation
    df = df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[META_FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_val = val_df[META_FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=META_FEATURE_NAMES)
    lgb_val = lgb.Dataset(
        X_val, label=y_val, feature_name=META_FEATURE_NAMES, reference=lgb_train
    )

    params = {**_LGB_META_PARAMS, "seed": seed}
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # Validation metrics
    val_preds = booster.predict(X_val)
    val_correct = ((val_preds >= 0.5) == y_val).sum()
    val_accuracy = val_correct / len(y_val)

    # Training class balance
    pos_rate = y_train.mean()

    logger.info(
        "Meta-model training complete",
        extra={
            "component": "over_under.meta_model",
            "n_train": len(train_df),
            "n_val": len(val_df),
            "val_accuracy": round(float(val_accuracy), 4),
            "pos_rate": round(float(pos_rate), 4),
            "best_iteration": booster.best_iteration,
        },
    )

    return ModelArtifact(
        model=booster,
        feature_names=META_FEATURE_NAMES,
        feature_schema_version=1,
        training_window_start=df["game_date"].min(),
        training_window_end=df["game_date"].max(),
        created_at=datetime.now(timezone.utc),
        target_name="meta_over_5_5",
    )


def predict_correctness(
    predictions: list,
    config: Config,
) -> list[dict]:
    """Score each prediction with P(correct) using the meta-model.

    Returns a list of dicts: [{"game_pk": ..., "p_correct": 0.87}, ...]
    sorted by p_correct descending.

    Returns empty list if meta-model is not available.
    """
    artifact_path = config.model.artifact_path("meta_over_5_5")

    try:
        artifact = load_artifact(artifact_path, expected_target="meta_over_5_5")
    except (FileNotFoundError, Exception) as exc:
        logger.debug(
            f"Meta-model not available: {exc}",
            extra={"component": "over_under.meta_model"},
        )
        return []

    results = []
    for pred in predictions:
        features = _extract_meta_features(pred)
        feature_array = np.array(
            [[features.get(name, 0.0) for name in META_FEATURE_NAMES]],
            dtype=np.float32,
        )
        p_correct = float(artifact.model.predict(feature_array)[0])
        results.append({
            "game_pk": pred.game_pk,
            "home_team_code": pred.home_team_code,
            "away_team_code": pred.away_team_code,
            "p_over_5_5": pred.p_over.get(5.5, 0.0),
            "p_over_6_5": pred.p_over.get(6.5, 0.0),
            "sigma_used": pred.sigma_used,
            "p_correct": p_correct,
        })

    # Sort by p_correct descending
    results.sort(key=lambda x: x["p_correct"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_meta_training_data(conn) -> pd.DataFrame:
    """Load reconciled outcomes with all meta-features."""
    rows = conn.execute(text("""
        SELECT
            game_date,
            p_over_5_5,
            sigma_used,
            home_starter_era,
            away_starter_era,
            home_trailing15_rpg,
            away_trailing15_rpg,
            ballpark_id,
            pitcher_fallback,
            home_expected_runs,
            away_expected_runs,
            actual_over_5_5
        FROM derived.over_under_outcomes
        WHERE actual_over_5_5 IS NOT NULL
          AND sigma_used IS NOT NULL
          AND p_over_5_5 IS NOT NULL
        ORDER BY game_date
    """)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "game_date", "p_over_5_5", "sigma_used",
        "home_starter_era", "away_starter_era",
        "home_trailing15_rpg", "away_trailing15_rpg",
        "ballpark_id", "pitcher_fallback",
        "home_expected_runs", "away_expected_runs",
        "actual_over_5_5",
    ])

    # Compute derived feature
    df["total_expected_runs"] = (
        pd.to_numeric(df["home_expected_runs"], errors="coerce").fillna(4.5) +
        pd.to_numeric(df["away_expected_runs"], errors="coerce").fillna(4.5)
    )

    # Coerce types
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in META_FEATURE_NAMES:
        if col == "pitcher_fallback":
            df[col] = df[col].astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Label: 1 if our prediction was correct (actual went over 5.5)
    df["label"] = df["actual_over_5_5"].astype(int)

    return df


def _extract_meta_features(pred) -> dict[str, float]:
    """Extract meta-model features from an OverUnderPrediction."""
    return {
        "p_over_5_5": pred.p_over.get(5.5, 0.0),
        "sigma_used": pred.sigma_used or 3.3,
        "home_starter_era": pred.home_starter_era or 4.5,
        "away_starter_era": pred.away_starter_era or 4.5,
        "home_trailing15_rpg": pred.home_trailing15_rpg or 4.5,
        "away_trailing15_rpg": pred.away_trailing15_rpg or 4.5,
        "ballpark_id": pred.ballpark_id or 0,
        "pitcher_fallback": 1 if pred.pitcher_fallback else 0,
        "total_expected_runs": (pred.home_expected_runs or 4.5) + (pred.away_expected_runs or 4.5),
    }


__all__ = [
    "META_FEATURE_NAMES",
    "predict_correctness",
    "train_meta_model",
]
