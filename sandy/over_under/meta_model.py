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

# LightGBM hyperparameters — moderate regularization (230+ games available)
_LGB_META_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 16,
    "min_data_in_leaf": 8,
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

    # Two-phase training to prevent collapsing to 1 tree:
    # Phase 1: Train minimum 5 trees unconditionally (no early stopping)
    # Phase 2: Continue with early stopping up to 150 total
    # This guarantees at least 5 trees while allowing more if data supports it.
    MIN_TREES = 5

    # Phase 1: train 5 trees without early stopping
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=MIN_TREES,
        valid_sets=[lgb_val],
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    # Phase 2: continue training with early stopping (up to 150 total)
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=150 - MIN_TREES,
        valid_sets=[lgb_val],
        init_model=booster,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
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


def calibrate_meta_threshold(engine: Engine, config: Config) -> dict | None:
    """Compute the optimal P(correct) threshold from reconciled data.

    Scores all reconciled games with the current meta-model, then finds
    the P(correct) cutoff that maximizes accuracy (with ≥ 10 games).

    Returns a dict:
    {
        "recommended_threshold": 0.70,
        "breakdown": [
            {"threshold": 0.60, "accuracy": 0.73, "games": 230, "correct": 168},
            {"threshold": 0.65, "accuracy": 0.77, "games": 200, "correct": 154},
            ...
        ],
        "below_threshold": {"accuracy": 0.54, "games": 90, "correct": 49},
    }

    Returns None if meta-model is not available or insufficient data.
    """
    artifact_path = config.model.artifact_path("meta_over_5_5")

    try:
        artifact = load_artifact(artifact_path, expected_target="meta_over_5_5")
    except (FileNotFoundError, Exception) as exc:
        logger.debug(
            f"Meta-model not available for calibration: {exc}",
            extra={"component": "over_under.meta_model"},
        )
        return None

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                p_over_5_5, sigma_used,
                home_starter_era, away_starter_era,
                home_trailing15_rpg, away_trailing15_rpg,
                ballpark_id, pitcher_fallback,
                home_expected_runs, away_expected_runs,
                actual_over_5_5
            FROM derived.over_under_outcomes
            WHERE actual_over_5_5 IS NOT NULL
              AND sigma_used IS NOT NULL
              AND p_over_5_5 IS NOT NULL
            ORDER BY game_date
        """)).fetchall()

    if len(rows) < 20:
        return None

    # Score each game with the meta-model
    scored = []
    for row in rows:
        features = {
            "p_over_5_5": float(row[0]),
            "sigma_used": float(row[1]),
            "home_starter_era": float(row[2]) if row[2] else 4.5,
            "away_starter_era": float(row[3]) if row[3] else 4.5,
            "home_trailing15_rpg": float(row[4]) if row[4] else 4.5,
            "away_trailing15_rpg": float(row[5]) if row[5] else 4.5,
            "ballpark_id": float(row[6]) if row[6] else 0,
            "pitcher_fallback": 1 if row[7] else 0,
            "total_expected_runs": (float(row[8]) if row[8] else 4.5) + (float(row[9]) if row[9] else 4.5),
        }
        feat_array = np.array(
            [[features[n] for n in META_FEATURE_NAMES]], dtype=np.float32
        )
        p_correct = float(artifact.model.predict(feat_array)[0])
        actually_correct = bool(row[10])
        scored.append((p_correct, actually_correct))

    # Compute accuracy at each threshold
    thresholds = [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90]
    breakdown = []
    best_threshold = 0.50
    best_accuracy = 0.0

    for t in thresholds:
        above = [s for s in scored if s[0] >= t]
        if len(above) < 5:
            continue
        correct = sum(1 for _, c in above if c)
        total = len(above)
        accuracy = correct / total

        breakdown.append({
            "threshold": t,
            "accuracy": round(accuracy, 3),
            "games": total,
            "correct": correct,
        })

        # Best = highest accuracy with at least 10 games
        if accuracy >= best_accuracy and total >= 10:
            best_accuracy = accuracy
            best_threshold = t

    # Compute below-threshold stats
    below = [s for s in scored if s[0] < best_threshold]
    below_correct = sum(1 for _, c in below if c)
    below_stats = {
        "accuracy": round(below_correct / len(below), 3) if below else 0.0,
        "games": len(below),
        "correct": below_correct,
    }

    result = {
        "recommended_threshold": best_threshold,
        "recommended_accuracy": round(best_accuracy, 3),
        "breakdown": breakdown,
        "below_threshold": below_stats,
        "total_games": len(scored),
    }

    logger.info(
        "Meta-model calibration complete",
        extra={
            "component": "over_under.meta_model",
            "recommended_threshold": best_threshold,
            "recommended_accuracy": round(best_accuracy, 4),
            "total_games": len(scored),
        },
    )

    return result


__all__ = [
    "META_FEATURE_NAMES",
    "calibrate_meta_threshold",
    "predict_correctness",
    "train_meta_model",
]
