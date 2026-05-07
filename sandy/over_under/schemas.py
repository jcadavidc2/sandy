"""Domain dataclasses for the Over/Under Feedback Loop.

Defines the core data structures used across predictor, reconciler,
calibrator, retrainer, and notifier modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


# Standard over/under thresholds for MLB game totals
STANDARD_THRESHOLDS: list[float] = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]


@dataclass(frozen=True)
class OverUnderPrediction:
    """A single game's over/under prediction with all thresholds."""

    game_pk: int
    game_date: date
    home_team_code: str
    away_team_code: str
    game_time_utc: datetime
    predicted_at_utc: datetime
    p_over: dict[float, float]  # {5.5: 0.82, 6.5: 0.71, ...}
    feature_vector: dict[str, float]  # 10 GAME_FEATURE_NAMES keys
    home_starter_era: float | None
    away_starter_era: float | None
    ballpark_id: int | None
    home_trailing15_rpg: float | None
    away_trailing15_rpg: float | None
    pitcher_fallback: bool


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Calibration metrics computed from recent reconciled outcomes."""

    snapshot_date: date
    accuracy_by_threshold: dict[float, float]  # {5.5: 0.68, 6.5: 0.71, ...}
    recommended_threshold: float
    sample_size: int
    covariate_insights: dict[str, Any]  # miss rates by covariate quartile
    rolling_4w_accuracy: float | None


@dataclass(frozen=True)
class RetrainingResult:
    """Result of a model retraining attempt."""

    success: bool
    sample_size: int
    new_mae: float
    previous_mae: float | None
    skipped_reason: str | None  # Non-None if guard triggered


__all__ = [
    "CalibrationSnapshot",
    "OverUnderPrediction",
    "RetrainingResult",
    "STANDARD_THRESHOLDS",
]
