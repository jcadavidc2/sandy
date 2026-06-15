"""Reconcile football predictions against actual results.

Mirrors :func:`sandy.over_under.reconciler.reconcile_over_under`: for each
prediction whose fixture is now finished and not yet reconciled, fill the
actual outcome and the per-market ``was_correct_*`` flags. Idempotent — rows
already reconciled are skipped.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.logging import get_logger

logger = get_logger("football.reconciler")


def _predicted_result(p_home: float, p_draw: float, p_away: float) -> str:
    best = max((p_home, "H"), (p_draw, "D"), (p_away, "A"), key=lambda x: x[0])
    return best[1]


def reconcile_predictions(engine: Engine) -> int:
    """Fill actuals + was_correct flags for finished, unreconciled predictions.

    Returns the number of rows reconciled.
    """
    select_sql = text("""
        SELECT p.id, p.p_home_win, p.p_draw, p.p_away_win, p.p_over_2_5, p.p_btts,
               p.most_likely_home, p.most_likely_away,
               m.home_goals, m.away_goals
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        WHERE p.outcome_filled_at_utc IS NULL
          AND m.status IN ('FT','AET','PEN')
          AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
    """)
    update_sql = text("""
        UPDATE football.match_predictions SET
            actual_home_goals = :hg, actual_away_goals = :ag,
            actual_total_goals = :tot, actual_result = :res, actual_btts = :btts,
            was_correct_result = :c_res, was_correct_over_2_5 = :c_ou,
            was_correct_btts = :c_btts, was_correct_score = :c_score,
            outcome_filled_at_utc = :now
        WHERE id = :id
    """)

    now = datetime.now(timezone.utc)
    n = 0
    with engine.begin() as conn:
        rows = conn.execute(select_sql).fetchall()
        for r in rows:
            (pid, p_home, p_draw, p_away, p_ou, p_btts,
             ml_h, ml_a, hg, ag) = r
            hg, ag = int(hg), int(ag)
            tot = hg + ag
            res = "H" if hg > ag else ("A" if ag > hg else "D")
            btts = hg >= 1 and ag >= 1
            pred_res = _predicted_result(p_home, p_draw, p_away)
            conn.execute(update_sql, {
                "id": pid, "hg": hg, "ag": ag, "tot": tot, "res": res, "btts": btts,
                "c_res": pred_res == res,
                "c_ou": (p_ou >= 0.5) == (tot > 2.5),
                "c_btts": (p_btts >= 0.5) == btts,
                "c_score": (ml_h == hg and ml_a == ag),
                "now": now,
            })
            n += 1
    logger.info("Reconciled football predictions",
                extra={"component": "football.reconciler", "reconciled": n})
    return n


def reconcile(config: Config | None = None) -> int:
    cfg = config or load_config()
    return reconcile_predictions(create_engine(cfg))


__all__ = ["reconcile", "reconcile_predictions"]
