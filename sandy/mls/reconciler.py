"""Fill actuals + was_correct_* for finished MLS matches (idempotent)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)


def reconcile(config: Config | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    select_sql = text("""
        SELECT p.id, p.p_home_or_draw, p.p_over_2_5, p.p_corners_over_9_5,
               m.home_goals, m.away_goals, m.home_corners, m.away_corners
        FROM mls.match_predictions p
        JOIN mls.matches m ON m.event_id = p.event_id
        WHERE p.outcome_filled_at_utc IS NULL
          AND m.status = 'FT'
          AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
    """)
    update_sql = text("""
        UPDATE mls.match_predictions SET
            actual_home_goals=:hg, actual_away_goals=:ag, actual_total_goals=:tot,
            actual_result=:res, actual_total_corners=:tc,
            was_correct_double_chance=:c_dc, was_correct_over_2_5=:c_ou,
            was_correct_corners_9_5=:c_c, outcome_filled_at_utc=:now
        WHERE id=:id
    """)
    now = datetime.now(timezone.utc)
    n = 0
    with engine.begin() as conn:
        for r in conn.execute(select_sql).fetchall():
            pid, p_dc, p_ou, p_c95, hg, ag, hc, ac = r
            hg, ag = int(hg), int(ag)
            tot = hg + ag
            res = "H" if hg > ag else ("A" if ag > hg else "D")
            tc = (hc + ac) if (hc is not None and ac is not None) else None
            conn.execute(update_sql, {
                "id": pid, "hg": hg, "ag": ag, "tot": tot, "res": res, "tc": tc,
                # double chance: we "pick" home-or-draw when p >= 0.5; correct if result != Away win
                "c_dc": (p_dc >= 0.5) == (res != "A") if p_dc is not None else None,
                "c_ou": (p_ou >= 0.5) == (tot > 2.5) if p_ou is not None else None,
                "c_c": (p_c95 >= 0.5) == (tc > 9.5) if (p_c95 is not None and tc is not None) else None,
                "now": now,
            })
            n += 1
    logger.info("MLS reconciled %s predictions", n)
    return n
