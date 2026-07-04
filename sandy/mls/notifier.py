"""⚽ MLS Telegram digest — tip-sheet style: a 🎯 recommended-bets section (only
picks whose confidence bucket historically hits ≥60%), last night's scored
results, then the per-game probability board."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.over_under.notifier import send_telegram

from .parsers import DISPLAY_TZ
from .recommend import evaluate, load_reliability

logger = logging.getLogger(__name__)

GOAL_LINES = [(f"over_{str(t).replace('.', '_')}", f"p_over_{str(t).replace('.', '_')}", t) for t in (1.5, 2.5, 3.5, 4.5)]
CORNER_LINES = [(f"corners_over_{str(t).replace('.', '_')}", f"p_corners_over_{str(t).replace('.', '_')}", t)
                for t in (8.5, 9.5, 10.5, 11.5)]


def _candidates(reliability: dict, r) -> list[dict]:
    """All calibrated bet candidates for one match row."""
    out = []
    dc = evaluate(reliability, "double_chance", r.p_home_or_draw,
                  f"Local o empata (1X)", f"Gana visitante (2)")
    if dc:
        out.append(dc)
    for market, col, thr in GOAL_LINES:
        c = evaluate(reliability, market, getattr(r, col),
                     f"Más de {thr} goles", f"Menos de {thr} goles")
        if c:
            out.append(c)
    for market, col, thr in CORNER_LINES:
        c = evaluate(reliability, market, getattr(r, col),
                     f"Más de {thr} corners", f"Menos de {thr} corners")
        if c:
            out.append(c)
    return out


def _pending_rows(conn, day: date):
    return conn.execute(text("""
        SELECT * FROM mls.match_predictions
        WHERE match_date BETWEEN :a AND :b AND outcome_filled_at_utc IS NULL AND NOT is_backtest
        ORDER BY match_date, id
    """), {"a": day, "b": day + timedelta(days=1)}).fetchall()


def format_daily_digest(config: Config | None = None, *, for_date: date | None = None) -> str:
    cfg = config or load_config()
    engine = create_engine(cfg)
    today = for_date or datetime.now(DISPLAY_TZ).date()
    parts = [f"⚽ MLS ({today.strftime('%b %d')})"]
    with engine.begin() as conn:
        reliability = load_reliability(conn, "mls")

        # Last night, scored.
        night = conn.execute(text("""
            SELECT home_team, away_team, actual_home_goals, actual_away_goals,
                   actual_total_corners, was_correct_double_chance
            FROM mls.match_predictions
            WHERE match_date = :d AND outcome_filled_at_utc IS NOT NULL AND NOT is_backtest
            ORDER BY id LIMIT 8
        """), {"d": today - timedelta(days=1)}).fetchall()
        if night:
            hits = sum(1 for r in night if r.was_correct_double_chance)
            parts.append(f"🌙 Anoche: {hits}/{len(night)} 1X/2 correctas")
            for r in night:
                mark = "✅" if r.was_correct_double_chance else "❌"
                cor = f" · {r.actual_total_corners}c" if r.actual_total_corners is not None else ""
                parts.append(f"{mark} {r.home_team} {r.actual_home_goals}-{r.actual_away_goals} {r.away_team}{cor}")
            parts.append("")

        rows = _pending_rows(conn, today)
        if not rows:
            parts.append("😴 No hay partidos MLS hoy.")
        else:
            # 🎯 The tip sheet: every calibrated-trustworthy bet, best first.
            recs = []
            for r in rows:
                for c in _candidates(reliability, r):
                    recs.append((r, c))
            recs.sort(key=lambda x: (-x[1]["hist_acc"], -x[1]["conf"]))
            if recs:
                parts.append("🎯 APUESTAS RECOMENDADAS (históricamente ≥60% a esta confianza):")
                for r, c in recs[:8]:
                    parts.append(f"• {r.home_team} vs {r.away_team} → {c['label']} "
                                 f"({c['conf']:.0%}) · histórico {c['hist_acc']:.0%} (n={c['hist_n']})")
            else:
                parts.append("🎯 Hoy ningún pick supera el filtro de confianza — mejor no apostar.")
            parts.append("")
            parts.append(f"📋 Todos los partidos ({len(rows)}):")
            for r in rows:
                side = "1X" if r.p_home_or_draw >= r.p_away_win else "2"
                conf = max(r.p_home_or_draw, r.p_away_win)
                parts.append(f"· {r.home_team} vs {r.away_team} — {side} {conf:.0%} | "
                             f"O2.5 {r.p_over_2_5:.0%} | O9.5c {r.p_corners_over_9_5:.0%} | "
                             f"prob. {r.most_likely_home}-{r.most_likely_away}")
    parts.append("")
    parts.append("ℹ️ 1X = local gana o empata · O = más de · c = corners · histórico = % de acierto real de picks con esta confianza")
    return "\n".join(parts)


def notify_daily(config: Config | None = None) -> bool:
    ok = send_telegram(format_daily_digest(config))
    logger.info("MLS digest sent: %s", ok)
    return ok
