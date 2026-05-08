"""Over/Under Calibrator — compute accuracy metrics from reconciled outcomes.

Analyzes prediction accuracy per threshold, identifies the optimal threshold,
and computes per-covariate miss rates for EDA.

Requirements: 6.1, 6.2, 6.3, 6.5, 7.1, 7.2, 7.3, 7.7, 9.3
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.logging import get_logger
from sandy.over_under.schemas import (
    STANDARD_THRESHOLDS,
    CalibrationSnapshot,
)

logger = get_logger("over_under.calibrator")


def _threshold_col(t: float) -> str:
    """Convert threshold float to column name suffix, e.g. 5.5 -> '5_5'."""
    return str(t).replace(".", "_")


def compute_calibration(
    engine: Engine,
    lookback_days: int = 7,
) -> CalibrationSnapshot | None:
    """Compute calibration metrics from recent reconciled outcomes.

    Returns None if fewer than 5 reconciled predictions exist.
    Identifies optimal threshold (highest accuracy over lookback window).
    Computes per-covariate miss rates grouped by quartile.
    """
    today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    with engine.connect() as conn:
        # Get reconciled outcomes from the lookback window
        rows = conn.execute(
            text("""
                SELECT
                    was_correct_5_5, was_correct_6_5, was_correct_7_5,
                    was_correct_8_5, was_correct_9_5, was_correct_10_5,
                    was_correct_11_5,
                    home_starter_era, away_starter_era, ballpark_id,
                    home_trailing15_rpg, away_trailing15_rpg
                FROM derived.over_under_outcomes
                WHERE actual_total_runs IS NOT NULL
                  AND game_date >= :cutoff
            """),
            {"cutoff": cutoff},
        ).fetchall()

        if len(rows) < 5:
            return None

        sample_size = len(rows)

        # Compute accuracy per threshold
        accuracy_by_threshold: dict[float, float] = {}
        for i, t in enumerate(STANDARD_THRESHOLDS):
            correct_count = sum(1 for row in rows if row[i] is True)
            accuracy_by_threshold[t] = correct_count / sample_size

        # Find recommended threshold (highest accuracy)
        recommended_threshold = max(
            accuracy_by_threshold, key=lambda t: accuracy_by_threshold[t]
        )

        # Compute probability-binned accuracy for each threshold
        # This tells us: "for over 6.5, predictions above X% have Y% accuracy"
        probability_thresholds = _compute_probability_threshold_analysis(conn, cutoff)

        # Compute per-covariate miss rates (grouped by quartile)
        covariate_insights = _compute_covariate_insights(rows)

        # Add probability threshold recommendations to insights
        covariate_insights["probability_thresholds"] = probability_thresholds

        # Compute rolling 4-week accuracy for primary threshold (6.5)
        rolling_4w_accuracy = _compute_rolling_4w_accuracy(conn, today)

        covariate_insights["rolling_4w_accuracy"] = rolling_4w_accuracy

    return CalibrationSnapshot(
        snapshot_date=today,
        accuracy_by_threshold=accuracy_by_threshold,
        recommended_threshold=recommended_threshold,
        sample_size=sample_size,
        covariate_insights=covariate_insights,
        rolling_4w_accuracy=rolling_4w_accuracy,
    )


def _compute_probability_threshold_analysis(
    conn,
    cutoff: date,
) -> dict[str, Any]:
    """For each over threshold, find the probability cutoff that maximizes accuracy.

    Returns a dict like:
    {
        "5.5": {"best_prob_cutoff": 0.85, "accuracy_at_cutoff": 0.95, "games_at_cutoff": 19,
                "breakdown": [{"prob_min": 0.70, "accuracy": 0.80, "games": 20}, ...]},
        "6.5": {...},
        ...
    }
    """
    result: dict[str, Any] = {}

    for t in STANDARD_THRESHOLDS:
        col = _threshold_col(t)
        p_col = f"p_over_{col}"
        w_col = f"was_correct_{col}"

        rows = conn.execute(
            text(f"""
                SELECT {p_col}, {w_col}
                FROM derived.over_under_outcomes
                WHERE actual_total_runs IS NOT NULL
                  AND game_date >= :cutoff
                  AND {w_col} IS NOT NULL
                ORDER BY {p_col} DESC
            """),
            {"cutoff": cutoff},
        ).fetchall()

        if len(rows) < 3:
            result[str(t)] = {"insufficient_data": True}
            continue

        # Compute accuracy at different probability cutoffs
        prob_cutoffs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        breakdown = []
        best_cutoff = 0.50
        best_accuracy = 0.0
        best_games = 0

        for prob_min in prob_cutoffs:
            filtered = [(p, c) for p, c in rows if p >= prob_min]
            if len(filtered) < 2:
                continue
            correct = sum(1 for _, c in filtered if c is True)
            total = len(filtered)
            accuracy = correct / total

            breakdown.append({
                "prob_min": prob_min,
                "accuracy": round(accuracy, 3),
                "correct": correct,
                "total": total,
            })

            # Best = highest accuracy with at least 3 games
            if accuracy >= best_accuracy and total >= 3:
                best_accuracy = accuracy
                best_cutoff = prob_min
                best_games = total

        result[str(t)] = {
            "best_prob_cutoff": best_cutoff,
            "accuracy_at_cutoff": round(best_accuracy, 3),
            "games_at_cutoff": best_games,
            "breakdown": breakdown,
        }

    return result


def _compute_covariate_insights(rows: list) -> dict[str, Any]:
    """Compute miss rates by covariate quartile."""
    insights: dict[str, Any] = {}

    # Covariate indices in the row: 7=home_starter_era, 8=away_starter_era,
    # 9=ballpark_id, 10=home_trailing15_rpg, 11=away_trailing15_rpg
    covariates = {
        "home_starter_era": 7,
        "away_starter_era": 8,
        "ballpark_id": 9,
        "home_trailing15_rpg": 10,
        "away_trailing15_rpg": 11,
    }

    for name, idx in covariates.items():
        values = [(row[idx], row[1]) for row in rows if row[idx] is not None]
        if len(values) < 4:
            insights[name] = {"insufficient_data": True}
            continue

        # Sort by covariate value and split into quartiles
        values.sort(key=lambda x: x[0])
        q_size = len(values) // 4
        quartiles: dict[str, float] = {}

        for q_idx, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            start = q_idx * q_size
            end = start + q_size if q_idx < 3 else len(values)
            q_values = values[start:end]
            if q_values:
                miss_count = sum(1 for _, correct in q_values if correct is not True)
                quartiles[q_name] = miss_count / len(q_values)
            else:
                quartiles[q_name] = 0.0

        insights[name] = quartiles

    return insights


def _compute_rolling_4w_accuracy(conn: Any, today: date) -> float | None:
    """Compute rolling 4-week accuracy for the 6.5 threshold."""
    cutoff_4w = today - timedelta(days=28)

    result = conn.execute(
        text("""
            SELECT
                COUNT(*) FILTER (WHERE was_correct_6_5 = true) AS correct,
                COUNT(*) AS total
            FROM derived.over_under_outcomes
            WHERE actual_total_runs IS NOT NULL
              AND game_date >= :cutoff
        """),
        {"cutoff": cutoff_4w},
    ).fetchone()

    if result and result[1] > 0:
        return result[0] / result[1]
    return None


def persist_calibration(engine: Engine, snapshot: CalibrationSnapshot) -> None:
    """Insert calibration snapshot to derived.calibration_snapshots."""
    with engine.begin() as conn:
        # Insert one row per threshold with the snapshot data
        for t, accuracy in snapshot.accuracy_by_threshold.items():
            conn.execute(
                text("""
                    INSERT INTO derived.calibration_snapshots (
                        snapshot_date, threshold, accuracy, sample_size,
                        recommended_threshold, covariate_insights
                    ) VALUES (
                        :snapshot_date, :threshold, :accuracy, :sample_size,
                        :recommended_threshold, :covariate_insights
                    )
                """),
                {
                    "snapshot_date": snapshot.snapshot_date,
                    "threshold": t,
                    "accuracy": accuracy,
                    "sample_size": snapshot.sample_size,
                    "recommended_threshold": snapshot.recommended_threshold,
                    "covariate_insights": json.dumps(snapshot.covariate_insights),
                },
            )

    logger.info(
        f"Persisted calibration snapshot for {snapshot.snapshot_date}",
        extra={
            "component": "over_under.calibrator",
            "recommended_threshold": snapshot.recommended_threshold,
            "sample_size": snapshot.sample_size,
        },
    )


__all__ = ["compute_calibration", "persist_calibration"]
