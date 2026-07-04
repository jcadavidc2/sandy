"""Multi-league soccer ingestion (reuses the MLS ESPN client + parsers)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.mls.client import EspnClient
from sandy.mls.parsers import DISPLAY_TZ, parse_scoreboard_events, parse_summary_stats

from . import LEAGUES

logger = logging.getLogger(__name__)

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "add_soccer_tables.sql"


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())


def _client(league: str) -> EspnClient:
    return EspnClient(code=LEAGUES[league][0])


def _upsert_match(conn, league: str, m) -> None:
    for t in (m.home, m.away):
        conn.execute(text("""
            INSERT INTO soccer.teams (team_id, name, abbrev, league, logo_url)
            VALUES (:id, :name, :ab, :lg, :logo)
            ON CONFLICT (team_id) DO UPDATE SET name = EXCLUDED.name,
                abbrev = COALESCE(EXCLUDED.abbrev, soccer.teams.abbrev),
                league = EXCLUDED.league
        """), {"id": t.team_id, "name": t.name, "ab": t.abbrev, "lg": league, "logo": t.logo_url})
    conn.execute(text("""
        INSERT INTO soccer.matches (event_id, league, match_date, kickoff_utc, season, status,
                                    home_team_id, away_team_id, home_goals, away_goals)
        VALUES (:eid, :lg, :d, :ko, :season, :status, :h, :a, :hg, :ag)
        ON CONFLICT (event_id) DO UPDATE SET
            status = EXCLUDED.status,
            home_goals = COALESCE(EXCLUDED.home_goals, soccer.matches.home_goals),
            away_goals = COALESCE(EXCLUDED.away_goals, soccer.matches.away_goals),
            kickoff_utc = EXCLUDED.kickoff_utc, match_date = EXCLUDED.match_date
    """), {"eid": m.event_id, "lg": league, "d": m.match_date, "ko": m.kickoff_utc,
           "season": m.season, "status": m.status, "h": m.home.team_id, "a": m.away.team_id,
           "hg": m.home_goals, "ag": m.away_goals})


def ingest_dates(engine: Engine, league: str, dates: list[date],
                 client: EspnClient | None = None) -> int:
    client = client or _client(league)
    n = 0
    for d in dates:
        matches = parse_scoreboard_events(client.scoreboard(d.strftime("%Y%m%d")))
        with engine.begin() as conn:
            for m in matches:
                _upsert_match(conn, league, m)
                n += 1
    return n


def ingest_stats_for_unstatted(engine: Engine, league: str, limit: int = 60,
                               client: EspnClient | None = None) -> int:
    client = client or _client(league)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT event_id, home_team_id FROM soccer.matches
            WHERE league = :lg AND status = 'FT' AND stats_filled_at_utc IS NULL
            ORDER BY match_date DESC LIMIT :lim
        """), {"lg": league, "lim": limit}).fetchall()
    n = 0
    for eid, home_id in rows:
        try:
            stats = parse_summary_stats(eid, client.summary(eid))
        except RuntimeError as e:
            logger.warning("%s summary failed for %s: %s", league, eid, e)
            continue
        with engine.begin() as conn:
            corners = {}
            for s in stats:
                is_home = s.team_id == home_id
                conn.execute(text("""
                    INSERT INTO soccer.match_stats (event_id, team_id, is_home, corners, total_shots,
                        shots_on_target, possession_pct, fouls, offsides, yellow_cards, red_cards, saves)
                    VALUES (:eid, :tid, :ih, :c, :ts, :st, :pp, :f, :o, :yc, :rc, :sv)
                    ON CONFLICT (event_id, team_id) DO UPDATE SET corners = EXCLUDED.corners,
                        total_shots = EXCLUDED.total_shots, shots_on_target = EXCLUDED.shots_on_target,
                        possession_pct = EXCLUDED.possession_pct, fouls = EXCLUDED.fouls,
                        offsides = EXCLUDED.offsides, yellow_cards = EXCLUDED.yellow_cards,
                        red_cards = EXCLUDED.red_cards, saves = EXCLUDED.saves
                """), {"eid": eid, "tid": s.team_id, "ih": is_home, "c": s.corners,
                       "ts": s.total_shots, "st": s.shots_on_target, "pp": s.possession_pct,
                       "f": s.fouls, "o": s.offsides, "yc": s.yellow_cards, "rc": s.red_cards,
                       "sv": s.saves})
                corners["home" if is_home else "away"] = s.corners
            conn.execute(text("""
                UPDATE soccer.matches SET home_corners = :hc, away_corners = :ac,
                       stats_filled_at_utc = :now WHERE event_id = :eid
            """), {"hc": corners.get("home"), "ac": corners.get("away"),
                   "now": datetime.now(timezone.utc), "eid": eid})
        n += 1
    return n


def ingest_recent_window(config: Config | None = None, leagues: list[str] | None = None) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    today = datetime.now(DISPLAY_TZ).date()
    days = [today - timedelta(days=1), today, today + timedelta(days=1)]
    out = {}
    for lg in (leagues or list(LEAGUES)):
        n = ingest_dates(engine, lg, days)
        s = ingest_stats_for_unstatted(engine, lg, limit=30)
        out[lg] = {"matches": n, "stats": s}
        logger.info("soccer[%s] daily ingest: %s matches, %s summaries", lg, n, s)
    return out


def backfill(config: Config | None = None, *, league: str, start: date,
             end: date | None = None, with_stats: bool = True) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    client = _client(league)
    months = LEAGUES[league][3]
    end = end or datetime.now(DISPLAY_TZ).date()
    n_matches = 0
    d = start
    while d <= end:
        if d.month in months:
            n_matches += ingest_dates(engine, league, [d], client)
        d += timedelta(days=1)
        if d.day == 1:
            logger.info("soccer[%s] backfill through %s (%s upserts)", league, d, n_matches)
    n_stats = 0
    if with_stats:
        while True:
            got = ingest_stats_for_unstatted(engine, league, limit=200, client=client)
            n_stats += got
            if got == 0:
                break
    logger.info("soccer[%s] backfill complete: %s matches, %s summaries", league, n_matches, n_stats)
    return {"matches": n_matches, "stats": n_stats}
