"""⚽ MLS Telegram digest — same shape as the World Cup one: calibration trust
line, last night's scored results, today's picks ranked by confidence."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.over_under.notifier import send_telegram

from .parsers import DISPLAY_TZ

logger = logging.getLogger(__name__)

MEDALS = ["🥇", "🥈", "🥉"]


def _calibration_line(conn) -> str:
    rows = conn.execute(text("""
        SELECT DISTINCT ON (market) market, accuracy, sample_size, recommended_threshold
        FROM mls.calibration_snapshots ORDER BY market, created_at DESC
    """)).fetchall()
    if not rows:
        return ""
    parts = [f"{m.replace('_', ' ')} {acc:.0%}" for m, acc, n, _ in rows]
    thr = next((t for m, a, n, t in rows if m == "double_chance" and t), None)
    trust = f" Confía en picks ≥{thr:.0%}." if thr else ""
    total = max(n for _, _, n, _ in rows)
    return f"📊 Calibración ({total} evaluadas): " + " · ".join(parts) + "." + trust


def _last_night_lines(conn, today) -> list[str]:
    rows = conn.execute(text("""
        SELECT home_team, away_team, actual_home_goals, actual_away_goals,
               actual_total_corners, was_correct_double_chance, was_correct_over_2_5,
               was_correct_corners_9_5
        FROM mls.match_predictions
        WHERE match_date = :d AND outcome_filled_at_utc IS NOT NULL AND NOT is_backtest
        ORDER BY id LIMIT 8
    """), {"d": today - timedelta(days=1)}).fetchall()
    if not rows:
        return []
    hits = sum(1 for r in rows if r.was_correct_double_chance)
    lines = [f"🌙 Anoche: {hits}/{len(rows)} dobles oportunidades correctas"]
    for r in rows:
        mark = "✅" if r.was_correct_double_chance else "❌"
        cor = f" · {r.actual_total_corners} corners" if r.actual_total_corners is not None else ""
        lines.append(f"{mark} {r.home_team} {r.actual_home_goals}-{r.actual_away_goals} {r.away_team}{cor}")
    return lines


def _pick_lines(conn, today) -> list[str]:
    rows = conn.execute(text("""
        SELECT home_team, away_team, p_home_or_draw, p_away_win, p_over_2_5,
               p_corners_over_9_5, most_likely_home, most_likely_away, match_date
        FROM mls.match_predictions
        WHERE match_date BETWEEN :a AND :b AND outcome_filled_at_utc IS NULL AND NOT is_backtest
        ORDER BY GREATEST(p_home_or_draw, p_away_win) DESC LIMIT 10
    """), {"a": today, "b": today + timedelta(days=1)}).fetchall()
    if not rows:
        return ["😴 No hay partidos MLS hoy."]
    lines = [f"🔮 Picks de hoy ({len(rows)}):"]
    for i, r in enumerate(rows):
        medal = MEDALS[i] if i < len(MEDALS) else "•"
        if r.p_home_or_draw >= r.p_away_win:
            pick, conf = "Local o empate (1X)", r.p_home_or_draw
        else:
            pick, conf = "Gana visitante (2)", r.p_away_win
        warn = " ⚠️ parejo" if conf < 0.55 else ""
        lines.append(
            f"{medal} {r.home_team} vs {r.away_team} → {pick} {conf:.0%}{warn} | "
            f"O2.5 {r.p_over_2_5:.0%} | corners O9.5 {r.p_corners_over_9_5:.0%} | "
            f"prob. {r.most_likely_home}-{r.most_likely_away}"
        )
    return lines


def format_daily_digest(config: Config | None = None) -> str:
    cfg = config or load_config()
    engine = create_engine(cfg)
    today = datetime.now(DISPLAY_TZ).date()
    parts = [f"⚽ MLS Predictions ({today.strftime('%b %d')})"]
    with engine.begin() as conn:
        cal = _calibration_line(conn)
        if cal:
            parts.append(cal)
        night = _last_night_lines(conn, today)
        if night:
            parts.append("")
            parts.extend(night)
        parts.append("")
        parts.extend(_pick_lines(conn, today))
    parts.append("")
    parts.append("ℹ️ 1X = local gana o empata · O2.5 = 3+ goles · O9.5 = 10+ corners")
    return "\n".join(parts)


def notify_daily(config: Config | None = None) -> bool:
    msg = format_daily_digest(config)
    ok = send_telegram(msg)
    logger.info("MLS digest sent: %s", ok)
    return ok
