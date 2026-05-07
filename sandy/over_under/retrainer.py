"""Over/Under Retrainer — retrain the runs model with guard condition.

Calls the existing train_runs_model() function, compares new model's
validation MAE against the current artifact. If new MAE > old MAE * 1.2,
does NOT overwrite (guard condition).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error
from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine, get_connection
from sandy.logging import get_logger
from sandy.over_under.schemas import RetrainingResult
from sandy.train.artifact import load_artifact, save_artifact

logger = get_logger("over_under.retrainer")


def retrain_runs_model(config: Config | None = None) -> RetrainingResult:
    """Retrain the runs model using all available game data.

    Uses the existing train_runs_model() function with standard LightGBM config.
    Compares new model's validation MAE against current artifact.
    If new_mae > previous_mae * 1.2, does NOT overwrite (guard condition).
    """
    from sandy.train.trainer import train_runs_model

    if config is None:
        config = load_config()

    engine = create_engine(config)
    artifact_path = config.model.artifact_path("runs")

    # Load current artifact's MAE for comparison (if it exists)
    previous_mae: float | None = None
    try:
        current_artifact = load_artifact(artifact_path, expected_target="runs")
        # Compute current model's MAE on validation data
        with get_connection(engine) as conn:
            previous_mae = _compute_artifact_mae(conn, current_artifact)
    except (FileNotFoundError, Exception) as exc:
        logger.info(
            f"No existing runs artifact to compare against: {exc}",
            extra={"component": "over_under.retrainer"},
        )

    # Train new model
    try:
        with get_connection(engine) as conn:
            new_artifact = train_runs_model(conn, seed=config.training.seed)
    except (ValueError, Exception) as exc:
        logger.error(
            f"Retraining failed: {exc}",
            extra={"component": "over_under.retrainer"},
        )
        return RetrainingResult(
            success=False,
            sample_size=0,
            new_mae=0.0,
            previous_mae=previous_mae,
            skipped_reason=f"Training failed: {exc}",
        )

    # Compute new model's MAE on validation data
    with get_connection(engine) as conn:
        new_mae = _compute_artifact_mae(conn, new_artifact)

    # Determine sample size from training window
    sample_size = _count_training_samples(engine)

    # Guard condition: if new MAE > old MAE * 1.2, don't overwrite
    if previous_mae is not None and new_mae > previous_mae * 1.2:
        reason = (
            f"New model MAE {new_mae:.4f} exceeds guard threshold "
            f"(previous {previous_mae:.4f} * 1.2 = {previous_mae * 1.2:.4f})"
        )
        logger.warning(
            reason,
            extra={"component": "over_under.retrainer"},
        )
        return RetrainingResult(
            success=False,
            sample_size=sample_size,
            new_mae=new_mae,
            previous_mae=previous_mae,
            skipped_reason=reason,
        )

    # Save new artifact
    save_artifact(new_artifact, artifact_path)
    logger.info(
        f"Runs model retrained: MAE={new_mae:.4f} (prev={previous_mae}), "
        f"sample_size={sample_size}",
        extra={
            "component": "over_under.retrainer",
            "new_mae": new_mae,
            "previous_mae": previous_mae,
            "sample_size": sample_size,
        },
    )

    return RetrainingResult(
        success=True,
        sample_size=sample_size,
        new_mae=new_mae,
        previous_mae=previous_mae,
        skipped_reason=None,
    )


def _compute_artifact_mae(conn, artifact) -> float:
    """Compute MAE of an artifact on the validation split of current data."""
    from sandy.features.schema import GAME_FEATURE_NAMES, GAME_FEATURE_SCHEMA_VERSION
    from sandy.train.split import chronological_split

    import pandas as pd

    feature_cols = ", ".join(f"f.{name}" for name in GAME_FEATURE_NAMES)

    result = conn.execute(
        text(f"""
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
            ORDER BY g.game_date, f.game_pk, f.team_code
        """),
        {"schema_version": GAME_FEATURE_SCHEMA_VERSION},
    )

    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    if len(df) < 100:
        return 999.0  # Not enough data

    df["runs"] = pd.to_numeric(df["runs"], errors="coerce").fillna(0).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in GAME_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    split = chronological_split(df, val_fraction=0.15)
    val_df = split.val

    X_val = val_df[GAME_FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df["runs"].values.astype(np.float32)

    preds = artifact.model.predict(X_val)
    return float(mean_absolute_error(y_val, preds))


def _count_training_samples(engine: Engine) -> int:
    """Count total training samples available."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM derived.runs_labels l
                JOIN raw.games g ON g.game_pk = l.game_pk
                WHERE g.status = 'Final'
            """)
        ).fetchone()
        return int(result[0]) if result else 0


__all__ = ["retrain_runs_model"]
