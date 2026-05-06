"""Outcome reconciler — backfills actual outcomes for finished games.

Phase 2, Task 10.1: Updates prediction_log rows with actual outcomes once
games reach "Final" status. Idempotent — only updates rows where
actual_outcome IS NULL.

Requirements: 8.1–8.6
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.logging import get_logger

logger = get_logger("evaluation.reconciler")


def reconcile_outcomes(engine: Engine) -> int:
    """Backfill actual_outcome for all unresolved predictions whose games are Final.

    Returns the number of rows updated. Idempotent — safe to call repeatedly.

    Logic per target:
    - reached_base: actual_outcome = "true"/"false" based on inning_labels
    - game_winner: actual_outcome = "true"/"false" based on game scores
    - runs: actual_outcome = actual runs scored (as string)
    """
    total_updated = 0

    with engine.begin() as conn:
        # Get unresolved predictions for Final games
        unresolved = conn.execute(
            text("""
                SELECT pl.id, pl.game_pk, pl.target, pl.team_code, pl.inning_number
                FROM derived.prediction_log pl
                JOIN raw.games g ON g.game_pk = pl.game_pk
                WHERE pl.actual_outcome IS NULL
                  AND g.status = 'Final'
                ORDER BY pl.game_pk
            """)
        ).fetchall()

        for row in unresolved:
            pred_id, game_pk, target, team_code, inning_number = row
            team_code = team_code.strip()

            outcome = None
            was_correct = None

            if target == "reached_base":
                # Check if team reached base in that inning
                label_row = conn.execute(
                    text("""
                        SELECT reached_base FROM derived.inning_labels
                        WHERE game_pk = :game_pk
                          AND team_code = :team_code
                          AND inning_number = :inning_number
                    """),
                    {"game_pk": game_pk, "team_code": team_code, "inning_number": inning_number},
                ).fetchone()

                if label_row is not None:
                    outcome = "true" if label_row[0] else "false"

            elif target == "game_winner":
                # Check if the predicted team won
                game_row = conn.execute(
                    text("""
                        SELECT home_team_code, home_score, away_score
                        FROM raw.games WHERE game_pk = :game_pk
                    """),
                    {"game_pk": game_pk},
                ).fetchone()

                if game_row is not None:
                    home_code = game_row[0].strip()
                    home_won = game_row[1] > game_row[2]
                    team_is_home = (team_code == home_code)
                    team_won = home_won if team_is_home else not home_won
                    outcome = "true" if team_won else "false"

            elif target == "runs":
                # Get actual runs scored by the team
                game_row = conn.execute(
                    text("""
                        SELECT home_team_code, away_team_code, home_score, away_score
                        FROM raw.games WHERE game_pk = :game_pk
                    """),
                    {"game_pk": game_pk},
                ).fetchone()

                if game_row is not None:
                    home_code = game_row[0].strip()
                    if team_code == home_code:
                        outcome = str(game_row[2])
                    else:
                        outcome = str(game_row[3])

            if outcome is None:
                continue

            # Compute was_correct
            # For binary targets: prediction > 0.5 should match actual "true"
            # We'd need the probability, but we can get it from the same row
            prob_row = conn.execute(
                text("SELECT probability FROM derived.prediction_log WHERE id = :id"),
                {"id": pred_id},
            ).fetchone()

            if prob_row and target in ("reached_base", "game_winner"):
                predicted_positive = prob_row[0] > 0.5
                actual_positive = outcome == "true"
                was_correct = predicted_positive == actual_positive
            elif target == "runs":
                # For runs: "correct" if within 2 runs of actual
                try:
                    actual_runs = int(outcome)
                    predicted_runs = prob_row[0] if prob_row else 0
                    was_correct = abs(predicted_runs - actual_runs) <= 2
                except (ValueError, TypeError):
                    was_correct = None

            # Update the row
            conn.execute(
                text("""
                    UPDATE derived.prediction_log
                    SET actual_outcome = :outcome,
                        was_correct = :was_correct,
                        outcome_filled_at_utc = now()
                    WHERE id = :id
                """),
                {"id": pred_id, "outcome": outcome, "was_correct": was_correct},
            )
            total_updated += 1

    logger.info(
        "Outcome reconciliation complete",
        extra={
            "component": "evaluation.reconciler",
            "rows_updated": total_updated,
        },
    )
    return total_updated


__all__ = ["reconcile_outcomes"]
