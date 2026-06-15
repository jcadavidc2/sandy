"""Read-only query helpers for the football digest and dashboard.

Centralizes the SQL the notifier (CLI) and Streamlit app both need, so the two
surfaces stay consistent.
"""
from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config


def local_today(cfg: Config) -> date:
    from datetime import datetime
    return datetime.now(ZoneInfo(cfg.football.display_timezone)).date()


def get_today_predictions(
    engine: Engine, cfg: Config, *, for_date: date | None = None,
) -> list[dict]:
    """Upcoming (NS) World Cup picks for a specific local day (default: today).

    ``match_date`` is stored as the user-timezone local date (fixtures are
    ingested with the display timezone), so filtering on the local "today"
    yields exactly today's slate — not tomorrow's, which is also NS and already
    ingested in the +/-1 day window.
    """
    day = for_date or local_today(cfg)
    sql = text("""
        SELECT th.name AS home, ta.name AS away,
               p.p_home_win, p.p_draw, p.p_away_win,
               p.most_likely_home, p.most_likely_away,
               p.p_over_2_5, p.p_btts, m.kickoff_utc
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        JOIN football.teams th ON th.team_id = p.home_team_id
        JOIN football.teams ta ON ta.team_id = p.away_team_id
        WHERE m.status = 'NS' AND m.competition = 'World Cup'
          AND p.match_date = :day
        ORDER BY m.kickoff_utc
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"day": day}).mappings().all()
    return [dict(r) for r in rows]


def get_latest_calibration(engine: Engine) -> list[dict]:
    """Most recent calibration snapshot per market."""
    sql = text("""
        SELECT DISTINCT ON (market) market, accuracy, sample_size,
               recommended_threshold, snapshot_date, covariate_insights
        FROM football.calibration_snapshots
        ORDER BY market, snapshot_date DESC
    """)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql).mappings().all()]


def get_recent_results(engine: Engine, cfg: Config, days: int = 1) -> list[dict]:
    """Reconciled World Cup results from the last ``days`` days (user tz)."""
    cutoff = local_today(cfg) - timedelta(days=days)
    sql = text("""
        SELECT th.name AS home, ta.name AS away,
               p.actual_home_goals, p.actual_away_goals,
               p.was_correct_result, p.was_correct_over_2_5, p.was_correct_btts
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        JOIN football.teams th ON th.team_id = p.home_team_id
        JOIN football.teams ta ON ta.team_id = p.away_team_id
        WHERE p.actual_result IS NOT NULL AND m.competition = 'World Cup'
          AND p.match_date >= :cutoff
        ORDER BY p.match_date DESC
    """)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql, {"cutoff": cutoff}).mappings().all()]


def get_match_options(engine: Engine) -> list[dict]:
    """World Cup matches that have a prediction (past + upcoming), newest first."""
    sql = text("""
        SELECT p.fixture_id, m.match_date, m.status, m.season, m.round,
               th.name AS home, ta.name AS away,
               p.actual_home_goals, p.actual_away_goals
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        JOIN football.teams th ON th.team_id = p.home_team_id
        JOIN football.teams ta ON ta.team_id = p.away_team_id
        WHERE m.competition = 'World Cup'
        ORDER BY m.match_date DESC, p.fixture_id
    """)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql).mappings().all()]


def get_match_detail(engine: Engine, fixture_id: int) -> dict | None:
    """Full prediction (+ actual, if finished) for one fixture, plus team stats."""
    psql = text("""
        SELECT p.*, m.status, m.match_date, m.competition, m.round,
               th.name AS home, ta.name AS away
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        JOIN football.teams th ON th.team_id = p.home_team_id
        JOIN football.teams ta ON ta.team_id = p.away_team_id
        WHERE p.fixture_id = :fid
    """)
    ssql = text("""
        SELECT t.name AS team, s.is_home, s.possession, s.shots_total,
               s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards, s.xg
        FROM football.match_stats s JOIN football.teams t ON t.team_id = s.team_id
        WHERE s.fixture_id = :fid
        ORDER BY s.is_home DESC
    """)
    with engine.connect() as conn:
        row = conn.execute(psql, {"fid": fixture_id}).mappings().first()
        if not row:
            return None
        stats = [dict(r) for r in conn.execute(ssql, {"fid": fixture_id}).mappings().all()]
    out = dict(row)
    out["stats"] = stats
    return out


__all__ = [
    "get_latest_calibration", "get_match_detail", "get_match_options",
    "get_recent_results", "get_today_predictions", "local_today",
]
