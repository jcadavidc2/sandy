"""Calibration reporter — computes accuracy and calibration metrics.

Phase 2, Task 11.1 + 11.2: Generates calibration reports and natural-language
summaries suitable for agent system prompts.

Requirements: 9.1–9.5, 10.1–10.3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.logging import get_logger

logger = get_logger("evaluation.reporter")


@dataclass(frozen=True)
class CalibrationBucket:
    range_start: float
    range_end: float
    prediction_count: int
    actual_rate: float | None       # None if count < 10
    is_sufficient: bool             # True if count >= 10


@dataclass(frozen=True)
class CalibrationReport:
    total_predictions: int
    date_range_start: date
    date_range_end: date
    accuracy_by_target: dict[str, float]
    accuracy_by_confidence: dict[str, float]
    calibration_buckets: list[CalibrationBucket]
    natural_language_summary: str


def get_calibration_report(engine: Engine, days: int = 7) -> CalibrationReport:
    """Compute calibration metrics over the last N days of predictions.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    with engine.connect() as conn:
        # Get all reconciled predictions in the window
        rows = conn.execute(
            text("""
                SELECT target, confidence_level, probability, was_correct
                FROM derived.prediction_log
                WHERE predicted_at_utc >= :start
                  AND actual_outcome IS NOT NULL
                  AND was_correct IS NOT NULL
                ORDER BY predicted_at_utc
            """),
            {"start": start_date},
        ).fetchall()

    if not rows:
        return CalibrationReport(
            total_predictions=0,
            date_range_start=start_date,
            date_range_end=end_date,
            accuracy_by_target={},
            accuracy_by_confidence={},
            calibration_buckets=[],
            natural_language_summary="No reconciled predictions in this period.",
        )

    # Accuracy by target
    target_correct: dict[str, list[bool]] = {}
    confidence_correct: dict[str, list[bool]] = {}
    bucket_data: dict[int, list[bool]] = {i: [] for i in range(10)}

    for target, confidence, probability, was_correct in rows:
        target_correct.setdefault(target, []).append(bool(was_correct))
        confidence_correct.setdefault(confidence, []).append(bool(was_correct))

        # Bucket assignment (0-10%, 10-20%, ..., 90-100%)
        bucket_idx = min(int(probability * 10), 9)
        bucket_data[bucket_idx].append(bool(was_correct))

    accuracy_by_target = {
        t: sum(v) / len(v) if v else 0.0
        for t, v in target_correct.items()
    }
    accuracy_by_confidence = {
        c: sum(v) / len(v) if v else 0.0
        for c, v in confidence_correct.items()
    }

    # Calibration buckets
    buckets = []
    for i in range(10):
        count = len(bucket_data[i])
        is_sufficient = count >= 10
        actual_rate = sum(bucket_data[i]) / count if is_sufficient else None
        buckets.append(CalibrationBucket(
            range_start=i * 0.1,
            range_end=(i + 1) * 0.1,
            prediction_count=count,
            actual_rate=actual_rate,
            is_sufficient=is_sufficient,
        ))

    summary = get_calibration_summary_text(accuracy_by_target, accuracy_by_confidence, len(rows))

    return CalibrationReport(
        total_predictions=len(rows),
        date_range_start=start_date,
        date_range_end=end_date,
        accuracy_by_target=accuracy_by_target,
        accuracy_by_confidence=accuracy_by_confidence,
        calibration_buckets=buckets,
        natural_language_summary=summary,
    )


def get_calibration_summary_text(
    accuracy_by_target: dict[str, float],
    accuracy_by_confidence: dict[str, float],
    total: int,
) -> str:
    """Generate a natural-language summary for agent system prompts.

    Requirements: 10.1, 10.2, 10.3
    """
    parts = [f"Based on {total} reconciled predictions:"]

    for target, acc in accuracy_by_target.items():
        if acc < 0.55:
            parts.append(f"  - {target}: {acc:.0%} accuracy (unreliable for this target)")
        elif acc > 0.65:
            parts.append(f"  - {target}: {acc:.0%} accuracy (reliable)")
        else:
            parts.append(f"  - {target}: {acc:.0%} accuracy")

    high_acc = accuracy_by_confidence.get("HIGH")
    if high_acc is not None:
        if high_acc > 0.65:
            parts.append(f"  - HIGH confidence predictions: {high_acc:.0%} (reliable)")
        elif high_acc < 0.55:
            parts.append(f"  - HIGH confidence predictions: {high_acc:.0%} (unreliable)")

    return "\n".join(parts)


__all__ = ["CalibrationBucket", "CalibrationReport", "get_calibration_report"]
