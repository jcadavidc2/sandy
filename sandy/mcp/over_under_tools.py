"""MCP tool handlers for Over/Under predictions, reports, and calibration.

Three new tools:
- get_daily_over_under_predictions: today's predictions sorted by p_over_6_5 desc
- get_over_under_report: reconciled outcomes for a date with summary
- get_over_under_calibration: latest calibration snapshot

Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.2, 5.3, 5.4, 5.5, 7.6, 9.1, 9.2
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from sandy.logging import get_logger

logger = get_logger("mcp.over_under_tools")


def handle_get_daily_over_under_predictions(args: dict[str, Any]) -> dict[str, Any]:
    """Return today's (or specified date's) predictions sorted by p_over_6_5 desc."""
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    date_str = args.get("date")
    target_date = date.fromisoformat(date_str) if date_str else date.today()

    config = load_config()
    engine = create_engine(config)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT game_pk, game_date, home_team_code, away_team_code,
                       predicted_at_utc,
                       p_over_5_5, p_over_6_5, p_over_7_5, p_over_8_5,
                       p_over_9_5, p_over_10_5, p_over_11_5,
                       feature_vector,
                       home_starter_era, away_starter_era, ballpark_id,
                       home_trailing15_rpg, away_trailing15_rpg,
                       pitcher_fallback
                FROM derived.over_under_outcomes
                WHERE game_date = :game_date
                ORDER BY p_over_6_5 DESC
            """),
            {"game_date": target_date},
        ).fetchall()

    if not rows:
        return {"predictions": [], "count": 0, "date": str(target_date)}

    predictions = []
    for r in rows:
        fv = r[12]
        if isinstance(fv, str):
            try:
                fv = json.loads(fv)
            except (json.JSONDecodeError, TypeError):
                fv = {}

        predictions.append({
            "game_pk": r[0],
            "game_date": str(r[1]),
            "home_team": r[2].strip() if r[2] else "",
            "away_team": r[3].strip() if r[3] else "",
            "predicted_at_utc": r[4].isoformat() if r[4] else None,
            "p_over_5_5": round(float(r[5]), 4),
            "p_over_6_5": round(float(r[6]), 4),
            "p_over_7_5": round(float(r[7]), 4),
            "p_over_8_5": round(float(r[8]), 4),
            "p_over_9_5": round(float(r[9]), 4),
            "p_over_10_5": round(float(r[10]), 4),
            "p_over_11_5": round(float(r[11]), 4),
            "feature_vector": fv,
            "home_starter_era": float(r[13]) if r[13] is not None else None,
            "away_starter_era": float(r[14]) if r[14] is not None else None,
            "ballpark_id": int(r[15]) if r[15] is not None else None,
            "home_trailing15_rpg": float(r[16]) if r[16] is not None else None,
            "away_trailing15_rpg": float(r[17]) if r[17] is not None else None,
            "pitcher_fallback": bool(r[18]) if r[18] is not None else False,
        })

    return {
        "predictions": predictions,
        "count": len(predictions),
        "date": str(target_date),
    }


def handle_get_over_under_report(args: dict[str, Any]) -> dict[str, Any]:
    """Return reconciled outcomes for yesterday (or specified date) with summary."""
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    date_str = args.get("date")
    target_date = date.fromisoformat(date_str) if date_str else date.today() - timedelta(days=1)

    config = load_config()
    engine = create_engine(config)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT game_pk, home_team_code, away_team_code,
                       p_over_6_5, actual_total_runs, was_correct_6_5,
                       feature_vector,
                       home_starter_era, away_starter_era, ballpark_id,
                       home_trailing15_rpg, away_trailing15_rpg
                FROM derived.over_under_outcomes
                WHERE game_date = :game_date
                  AND actual_total_runs IS NOT NULL
                ORDER BY game_pk
            """),
            {"game_date": target_date},
        ).fetchall()

    if not rows:
        return {
            "outcomes": [],
            "summary": {"correct": 0, "total": 0, "accuracy": 0.0},
            "date": str(target_date),
        }

    outcomes = []
    correct_count = 0
    for r in rows:
        was_correct = r[5]
        if was_correct is True:
            correct_count += 1

        fv = r[6]
        if isinstance(fv, str):
            try:
                fv = json.loads(fv)
            except (json.JSONDecodeError, TypeError):
                fv = {}

        outcomes.append({
            "game_pk": r[0],
            "home_team": r[1].strip() if r[1] else "",
            "away_team": r[2].strip() if r[2] else "",
            "p_over_6_5": round(float(r[3]), 4) if r[3] is not None else None,
            "actual_total_runs": int(r[4]) if r[4] is not None else None,
            "was_correct": was_correct,
            "feature_vector": fv,
        })

    total = len(outcomes)
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        "outcomes": outcomes,
        "summary": {
            "correct": correct_count,
            "total": total,
            "accuracy": round(accuracy, 4),
        },
        "date": str(target_date),
    }


def handle_get_over_under_calibration(args: dict[str, Any]) -> dict[str, Any]:
    """Return latest calibration snapshot with optional weeks_back history."""
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    weeks_back = args.get("weeks_back", 1)

    config = load_config()
    engine = create_engine(config)

    with engine.connect() as conn:
        # Get the latest snapshot date
        latest = conn.execute(
            text("""
                SELECT DISTINCT snapshot_date
                FROM derived.calibration_snapshots
                ORDER BY snapshot_date DESC
                LIMIT :limit
            """),
            {"limit": weeks_back},
        ).fetchall()

        if not latest:
            return {"calibration": None, "message": "No calibration data available."}

        snapshots = []
        for (snap_date,) in latest:
            rows = conn.execute(
                text("""
                    SELECT threshold, accuracy, sample_size,
                           recommended_threshold, covariate_insights
                    FROM derived.calibration_snapshots
                    WHERE snapshot_date = :snap_date
                    ORDER BY threshold
                """),
                {"snap_date": snap_date},
            ).fetchall()

            accuracy_by_threshold = {}
            recommended = None
            sample_size = 0
            covariate_insights = {}

            for r in rows:
                accuracy_by_threshold[float(r[0])] = round(float(r[1]), 4)
                sample_size = int(r[2])
                recommended = float(r[3])
                insights = r[4]
                if isinstance(insights, str):
                    try:
                        covariate_insights = json.loads(insights)
                    except (json.JSONDecodeError, TypeError):
                        covariate_insights = {}
                elif isinstance(insights, dict):
                    covariate_insights = insights

            snapshots.append({
                "snapshot_date": str(snap_date),
                "accuracy_by_threshold": accuracy_by_threshold,
                "recommended_threshold": recommended,
                "sample_size": sample_size,
                "covariate_insights": covariate_insights,
            })

    return {
        "calibration": snapshots[0] if snapshots else None,
        "history": snapshots,
        "weeks_returned": len(snapshots),
    }


__all__ = [
    "handle_get_daily_over_under_predictions",
    "handle_get_over_under_calibration",
    "handle_get_over_under_report",
]
