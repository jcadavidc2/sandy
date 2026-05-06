"""LightGBM trainer for Sandy Phase 1 + Phase 1.5.

Task 10.2: train_model() loads the labels⨝features join from Postgres,
splits chronologically, fits a LightGBM binary classifier, computes
validation metrics, and returns a ModelArtifact.

Phase 1.5 additions:
- train_game_winner_model(): binary classification for game winner
- train_runs_model(): regression for total runs scored

Key design decisions:
- deterministic=True + force_col_wise=True + fixed seed → bit-stable training
  across runs (required for the serializer round-trip PBT, requirement 7.2)
- Chronological split by game_pk prevents same-game leakage (requirement 6.2)
- TrainingQualityError raised if val ROC AUC < 0.52 (requirement 6.4)
- All metrics logged as a single JSON INFO line (requirement 6.3)

Requirements: 6.1, 6.3, 6.4, 6.6, 4.1–4.6, 5.1–5.5
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.features.schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    GAME_FEATURE_NAMES,
    GAME_FEATURE_SCHEMA_VERSION,
)
from sandy.logging import get_logger
from sandy.schemas import ModelArtifact
from sandy.train.split import chronological_split

logger = get_logger("train.trainer")


class TrainingQualityError(Exception):
    """Raised when validation ROC AUC falls below the minimum threshold."""

    def __init__(self, roc_auc: float, threshold: float = 0.52) -> None:
        super().__init__(
            f"Validation ROC AUC {roc_auc:.4f} is below minimum threshold {threshold:.2f}. "
            f"Check data quality or increase training data."
        )
        self.roc_auc = roc_auc
        self.threshold = threshold


class TrainingMetrics(NamedTuple):
    roc_auc: float
    log_loss: float
    brier_score: float
    n_positive: int
    n_negative: int
    n_train_rows: int
    n_val_rows: int
    n_train_games: int
    n_val_games: int


# Minimum acceptable validation ROC AUC (requirement 6.4)
MIN_ROC_AUC = 0.52

# LightGBM hyperparameters (requirement 6.1)
_LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "deterministic": True,   # bit-stable training (requirement 7.2)
    "force_col_wise": True,  # required with deterministic=True
}


def train_model(
    conn: Connection,
    *,
    seed: int = 42,
    training_window: tuple[date, date] | None = None,
) -> ModelArtifact:
    """Fit a LightGBM binary classifier on labeled feature rows.

    Parameters
    ----------
    conn:             Active DB connection (read-only).
    seed:             Random seed for LightGBM and any data operations.
                      Threads through all stochastic steps (requirement 6.6).
    training_window:  Optional (start, end) date range. If None, uses all
                      available data.

    Returns
    -------
    ModelArtifact ready to be saved via save_artifact().

    Raises
    ------
    TrainingQualityError: if validation ROC AUC < 0.52 (requirement 6.4).
    ValueError: if there is insufficient data to train.
    """
    # Step 1: load training frame
    df = _load_training_frame(conn, training_window)

    if len(df) < 100:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows available. "
            f"Run 'sandy ingest backfill' and 'sandy labels build' and "
            f"'sandy features build' first."
        )

    # Step 2: chronological split
    split = chronological_split(df, val_fraction=0.15)
    train_df = split.train
    val_df = split.val

    logger.info(
        "Training split complete",
        extra={
            "component": "train.trainer",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_train_games": split.n_train_games,
            "n_val_games": split.n_val_games,
        },
    )

    # Step 3: build LightGBM datasets
    X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["reached_base"].values.astype(np.float32)
    X_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["reached_base"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=lgb_train)

    # Step 4: fit
    params = {**_LGB_PARAMS, "seed": seed}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),  # suppress per-round output
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # Step 5: compute validation metrics
    val_preds = booster.predict(X_val)
    metrics = _compute_metrics(
        y_val, val_preds,
        n_train_rows=len(train_df),
        n_val_rows=len(val_df),
        n_train_games=split.n_train_games,
        n_val_games=split.n_val_games,
    )

    # Step 6: log metrics as single JSON line (requirement 6.3)
    logger.info(
        "Training metrics",
        extra={
            "component": "train.trainer",
            "roc_auc": round(metrics.roc_auc, 4),
            "log_loss": round(metrics.log_loss, 4),
            "brier_score": round(metrics.brier_score, 4),
            "n_positive": metrics.n_positive,
            "n_negative": metrics.n_negative,
            "n_train_rows": metrics.n_train_rows,
            "n_val_rows": metrics.n_val_rows,
            "n_train_games": metrics.n_train_games,
            "n_val_games": metrics.n_val_games,
            "best_iteration": booster.best_iteration,
        },
    )

    # Step 7: quality gate (requirement 6.4)
    if metrics.roc_auc < MIN_ROC_AUC:
        raise TrainingQualityError(metrics.roc_auc, MIN_ROC_AUC)

    return ModelArtifact(
        model=booster,
        feature_names=FEATURE_NAMES,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_window_start=df["game_date"].min(),
        training_window_end=df["game_date"].max(),
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_training_frame(
    conn: Connection,
    training_window: tuple[date, date] | None,
) -> pd.DataFrame:
    """Load the inner join of inning_features and inning_labels from Postgres.

    Returns a DataFrame with FEATURE_NAMES columns plus:
      game_pk, game_date, team_code, inning_number, reached_base
    """
    where_clause = ""
    params: dict = {}

    if training_window is not None:
        start, end = training_window
        where_clause = "AND g.game_date BETWEEN :start AND :end"
        params = {"start": start, "end": end}

    feature_cols = ", ".join(f"f.{name}" for name in FEATURE_NAMES)

    query = text(f"""
        SELECT
            f.game_pk,
            g.game_date,
            f.team_code,
            f.inning_number,
            l.reached_base,
            {feature_cols}
        FROM derived.inning_features f
        JOIN derived.inning_labels l
            ON  l.game_pk       = f.game_pk
            AND l.team_code     = f.team_code
            AND l.inning_number = f.inning_number
        JOIN raw.games g
            ON g.game_pk = f.game_pk
        WHERE f.feature_schema_version = :schema_version
          AND g.status = 'Final'
          {where_clause}
        ORDER BY g.game_date, f.game_pk, f.team_code, f.inning_number
    """)

    params["schema_version"] = FEATURE_SCHEMA_VERSION
    result = conn.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Coerce types
    df["reached_base"] = df["reached_base"].astype(bool)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_train_rows: int,
    n_val_rows: int,
    n_train_games: int,
    n_val_games: int,
) -> TrainingMetrics:
    return TrainingMetrics(
        roc_auc=float(roc_auc_score(y_true, y_pred)),
        log_loss=float(log_loss(y_true, y_pred)),
        brier_score=float(brier_score_loss(y_true, y_pred)),
        n_positive=int(y_true.sum()),
        n_negative=int((1 - y_true).sum()),
        n_train_rows=n_train_rows,
        n_val_rows=n_val_rows,
        n_train_games=n_train_games,
        n_val_games=n_val_games,
    )


# ---------------------------------------------------------------------------
# Phase 1.5: Game-level training functions
# ---------------------------------------------------------------------------

# LightGBM hyperparameters for game_winner (binary classification)
_LGB_GAME_WINNER_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
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

# LightGBM hyperparameters for runs (regression)
_LGB_RUNS_PARAMS = {
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


def train_game_winner_model(
    conn: Connection,
    *,
    seed: int = 42,
    training_window: tuple[date, date] | None = None,
) -> ModelArtifact:
    """Fit a LightGBM binary classifier for game winner prediction.

    Loads game_winner_labels ⨝ game_features from Postgres, splits
    chronologically, fits with binary objective, logs ROC AUC / log loss /
    Brier score.

    Raises TrainingQualityError if ROC AUC < 0.52.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.6
    """
    df = _load_game_winner_frame(conn, training_window)

    if len(df) < 100:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows available. "
            f"Run 'sandy labels build --target game_winner' and "
            f"'sandy features build --target game_winner' first."
        )

    # Chronological split
    split = chronological_split(df, val_fraction=0.15)
    train_df = split.train
    val_df = split.val

    logger.info(
        "Game winner training split complete",
        extra={
            "component": "train.trainer",
            "target": "game_winner",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_train_games": split.n_train_games,
            "n_val_games": split.n_val_games,
        },
    )

    # Build LightGBM datasets
    X_train = train_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["home_team_wins"].values.astype(np.float32)
    X_val = val_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["home_team_wins"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=GAME_FEATURE_NAMES)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=GAME_FEATURE_NAMES, reference=lgb_train)

    # Fit
    params = {**_LGB_GAME_WINNER_PARAMS, "seed": seed}
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
    auc = float(roc_auc_score(y_val, val_preds))
    ll = float(log_loss(y_val, val_preds))
    brier = float(brier_score_loss(y_val, val_preds))

    logger.info(
        "Game winner training metrics",
        extra={
            "component": "train.trainer",
            "target": "game_winner",
            "roc_auc": round(auc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "best_iteration": booster.best_iteration,
        },
    )

    # Quality gate
    if auc < MIN_ROC_AUC:
        raise TrainingQualityError(auc, MIN_ROC_AUC)

    return ModelArtifact(
        model=booster,
        feature_names=GAME_FEATURE_NAMES,
        feature_schema_version=GAME_FEATURE_SCHEMA_VERSION,
        training_window_start=df["game_date"].min(),
        training_window_end=df["game_date"].max(),
        created_at=datetime.now(timezone.utc),
        target_name="game_winner",
    )


def train_runs_model(
    conn: Connection,
    *,
    seed: int = 42,
    training_window: tuple[date, date] | None = None,
) -> ModelArtifact:
    """Fit a LightGBM regression model for runs prediction.

    Loads runs_labels ⨝ game_features from Postgres, splits chronologically,
    fits with regression objective, logs MAE and RMSE.

    Requirements: 5.1, 5.2, 5.3, 5.5
    """
    df = _load_runs_frame(conn, training_window)

    if len(df) < 100:
        raise ValueError(
            f"Insufficient training data: only {len(df)} rows available. "
            f"Run 'sandy labels build --target runs' and "
            f"'sandy features build --target runs' first."
        )

    # Chronological split
    split = chronological_split(df, val_fraction=0.15)
    train_df = split.train
    val_df = split.val

    logger.info(
        "Runs training split complete",
        extra={
            "component": "train.trainer",
            "target": "runs",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_train_games": split.n_train_games,
            "n_val_games": split.n_val_games,
        },
    )

    # Build LightGBM datasets
    X_train = train_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["runs"].values.astype(np.float32)
    X_val = val_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["runs"].values.astype(np.float32)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=GAME_FEATURE_NAMES)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=GAME_FEATURE_NAMES, reference=lgb_train)

    # Fit
    params = {**_LGB_RUNS_PARAMS, "seed": seed}
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
        "Runs training metrics",
        extra={
            "component": "train.trainer",
            "target": "runs",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
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
        target_name="runs",
    )


# ---------------------------------------------------------------------------
# Game-level data loading helpers
# ---------------------------------------------------------------------------


def _load_game_winner_frame(
    conn: Connection,
    training_window: tuple[date, date] | None,
) -> pd.DataFrame:
    """Load game_winner_labels ⨝ game_features from Postgres.

    Returns a DataFrame with GAME_FEATURE_NAMES columns plus:
      game_pk, game_date, home_team_wins
    """
    where_clause = ""
    params: dict = {}

    if training_window is not None:
        start, end = training_window
        where_clause = "AND g.game_date BETWEEN :start AND :end"
        params = {"start": start, "end": end}

    feature_cols = ", ".join(f"f.{name}" for name in GAME_FEATURE_NAMES)

    query = text(f"""
        SELECT
            f.game_pk,
            g.game_date,
            l.home_team_wins,
            {feature_cols}
        FROM derived.game_features f
        JOIN derived.game_winner_labels l
            ON l.game_pk = f.game_pk
        JOIN raw.games g
            ON g.game_pk = f.game_pk
        WHERE f.feature_schema_version = :schema_version
          AND f.is_home = true
          {where_clause}
        ORDER BY g.game_date, f.game_pk
    """)

    params["schema_version"] = GAME_FEATURE_SCHEMA_VERSION
    result = conn.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Coerce types
    df["home_team_wins"] = df["home_team_wins"].astype(bool)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in GAME_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _load_runs_frame(
    conn: Connection,
    training_window: tuple[date, date] | None,
) -> pd.DataFrame:
    """Load runs_labels ⨝ game_features from Postgres.

    Returns a DataFrame with GAME_FEATURE_NAMES columns plus:
      game_pk, game_date, team_code, runs
    """
    where_clause = ""
    params: dict = {}

    if training_window is not None:
        start, end = training_window
        where_clause = "AND g.game_date BETWEEN :start AND :end"
        params = {"start": start, "end": end}

    feature_cols = ", ".join(f"f.{name}" for name in GAME_FEATURE_NAMES)

    query = text(f"""
        SELECT
            f.game_pk,
            g.game_date,
            f.team_code,
            l.runs,
            {feature_cols}
        FROM derived.game_features f
        JOIN derived.runs_labels l
            ON l.game_pk = f.game_pk
            AND l.team_code = f.team_code
        JOIN raw.games g
            ON g.game_pk = f.game_pk
        WHERE f.feature_schema_version = :schema_version
          {where_clause}
        ORDER BY g.game_date, f.game_pk, f.team_code
    """)

    params["schema_version"] = GAME_FEATURE_SCHEMA_VERSION
    result = conn.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Coerce types
    df["runs"] = pd.to_numeric(df["runs"], errors="coerce").fillna(0).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    for col in GAME_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


__all__ = ["TrainingQualityError", "TrainingMetrics", "train_model", "train_game_winner_model", "train_runs_model"]
