"""Prediction logger — persists every prediction for self-evaluation.

Phase 2, Task 9.1: Non-blocking DB writes. If the database is unreachable,
logs a warning and continues without blocking the prediction response.

Requirements: 7.1, 7.2, 7.3, 7.4
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.logging import get_logger

logger = get_logger("evaluation.logger")


class PredictionLogger:
    """Logs predictions to derived.prediction_log for calibration tracking."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def log_prediction(
        self,
        game_pk: int,
        target: str,
        team_code: str,
        inning_number: int | None,
        probability: float,
        confidence_level: str,
        features_snapshot: dict[str, Any],
    ) -> int:
        """Insert a prediction row. Returns the row ID, or -1 on failure.

        Non-blocking: if DB is unreachable, logs a warning and returns -1.
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        INSERT INTO derived.prediction_log (
                            game_pk, target, team_code, inning_number,
                            probability, confidence_level, features_snapshot
                        ) VALUES (
                            :game_pk, :target, :team_code, :inning_number,
                            :probability, :confidence_level, :features_snapshot
                        )
                        RETURNING id
                    """),
                    {
                        "game_pk": game_pk,
                        "target": target,
                        "team_code": team_code,
                        "inning_number": inning_number,
                        "probability": probability,
                        "confidence_level": confidence_level,
                        "features_snapshot": json.dumps(features_snapshot),
                    },
                )
                row = result.fetchone()
                return int(row[0]) if row else -1
        except Exception as exc:
            logger.warning(
                "Failed to log prediction (non-blocking)",
                extra={
                    "component": "evaluation.logger",
                    "error": str(exc),
                    "game_pk": game_pk,
                    "target": target,
                },
            )
            return -1


__all__ = ["PredictionLogger"]
