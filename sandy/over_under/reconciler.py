"""Over/Under Reconciler — fill actual outcomes for finished games.

Queries Final games with NULL actual_total_runs in derived.over_under_outcomes,
computes actual totals and was_correct for each threshold.

Requirements: 3.1, 3.2, 3.3, 3.4
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.logging import get_logger
from sandy.over_under.schemas import STANDARD_THRESHOLDS

logger = get_logger("over_under.reconciler")


def _threshold_col(t: float) -> str:
    """Convert threshold float to column name suffix, e.g. 5.5 -> '5_5'."""
    return str(t).replace(".", "_")


def reconcile_over_under(engine: Engine, *, backtest_only: bool = False) -> int:
    """Fill actual outcomes for all Final games with NULL actual_total_runs.

    Computes:
    - actual_total_runs = home_score + away_score
    - actual_over_T = actual_total_runs > T (for each threshold)
    - was_correct_T = (p_over_T >= 0.5) == actual_over_T

    Returns number of rows updated. Idempotent — skips already-filled rows.

    With *backtest_only* (used by the walk-forward backtest) only rows with
    is_backtest = TRUE are touched, so the nightly live reconcile still finds
    (and reports on) today's freshly finished games itself.
    """
    total_updated = 0
    backtest_filter = "AND o.is_backtest" if backtest_only else ""

    with engine.begin() as conn:
        # Get unreconciled predictions for Final games
        rows = conn.execute(
            text(f"""
                SELECT
                    o.id, o.game_pk, o.game_date,
                    o.p_over_5_5, o.p_over_6_5, o.p_over_7_5, o.p_over_8_5,
                    o.p_over_9_5, o.p_over_10_5, o.p_over_11_5,
                    g.home_score, g.away_score
                FROM derived.over_under_outcomes o
                JOIN raw.games g ON g.game_pk = o.game_pk
                WHERE o.actual_total_runs IS NULL
                  AND g.status = 'Final'
                  AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
                  {backtest_filter}
            """)
        ).fetchall()

        for row in rows:
            row_id = row[0]
            home_score = int(row[10])
            away_score = int(row[11])
            actual_total = home_score + away_score

            # Map threshold -> p_over value from the row
            p_overs = {
                5.5: float(row[3]),
                6.5: float(row[4]),
                7.5: float(row[5]),
                8.5: float(row[6]),
                9.5: float(row[7]),
                10.5: float(row[8]),
                11.5: float(row[9]),
            }

            # Compute actual_over and was_correct for each threshold
            update_params: dict = {
                "id": row_id,
                "actual_total_runs": actual_total,
            }

            for t in STANDARD_THRESHOLDS:
                col = _threshold_col(t)
                actual_over = actual_total > t
                predicted_over = p_overs[t] >= 0.5
                was_correct = predicted_over == actual_over

                update_params[f"actual_over_{col}"] = actual_over
                update_params[f"was_correct_{col}"] = was_correct

            conn.execute(
                text("""
                    UPDATE derived.over_under_outcomes
                    SET actual_total_runs = :actual_total_runs,
                        actual_over_5_5 = :actual_over_5_5,
                        actual_over_6_5 = :actual_over_6_5,
                        actual_over_7_5 = :actual_over_7_5,
                        actual_over_8_5 = :actual_over_8_5,
                        actual_over_9_5 = :actual_over_9_5,
                        actual_over_10_5 = :actual_over_10_5,
                        actual_over_11_5 = :actual_over_11_5,
                        was_correct_5_5 = :was_correct_5_5,
                        was_correct_6_5 = :was_correct_6_5,
                        was_correct_7_5 = :was_correct_7_5,
                        was_correct_8_5 = :was_correct_8_5,
                        was_correct_9_5 = :was_correct_9_5,
                        was_correct_10_5 = :was_correct_10_5,
                        was_correct_11_5 = :was_correct_11_5,
                        outcome_filled_at_utc = now()
                    WHERE id = :id
                """),
                update_params,
            )
            total_updated += 1

    logger.info(
        f"Reconciled {total_updated} over/under outcomes",
        extra={"component": "over_under.reconciler", "rows_updated": total_updated},
    )
    return total_updated


__all__ = ["reconcile_over_under"]
