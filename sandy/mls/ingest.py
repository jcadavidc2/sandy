"""MLS ingestion: idempotent upserts from ESPN into the `mls` schema.

Backfill walks scoreboard dates (season runs Feb–Dec; we skip Jan to save
requests), then trickles per-event summaries (corners + covariates) for any
finished match without stats. The daily window re-ingests yesterday/today/
tomorrow so late finals and postponements self-heal — and, unlike API-Football,
ESPN serves ANY historical date, so a missed cron day heals itself too.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine

from .client import EspnClient
from .parsers import DISPLAY_TZ, parse_scoreboard_events, parse_summary_stats
from .schemas import MlsMatch, MlsTeamStats

logger = logging.getLogger(__name__)

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "add_mls_tables.sql"
SEASON_MONTHS = range(2, 13)  # Feb..Dec — MLS never plays league games in January.


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())


def _upsert_match(conn, m: MlsMatch) -> None:
    for t in (m.home, m.away):
        conn.execute(text("""
            INSERT INTO mls.teams (team_id, name, abbrev, logo_url)
            VALUES (:id, :name, :ab, :logo)
            ON CONFLICT (team_id) DO UPDATE SET name = EXCLUDED.name,
                abbrev = COALESCE(EXCLUDED.abbrev, mls.teams.abbrev),
                logo_url = COALESCE(EXCLUDED.logo_url, mls.teams.logo_url)
        """), {"id": t.team_id, "name": t.name, "ab": t.abbrev, "logo": t.logo_url})
    conn.execute(text("""
        INSERT INTO mls.matches (event_id, match_date, kickoff_utc, season, status,
                                 home_team_id, away_team_id, home_goals, away_goals)
        VALUES (:eid, :d, :ko, :season, :status, :h, :a, :hg, :ag)
        ON CONFLICT (event_id) DO UPDATE SET
            status = EXCLUDED.status,
            home_goals = COALESCE(EXCLUDED.home_goals, mls.matches.home_goals),
            away_goals = COALESCE(EXCLUDED.away_goals, mls.matches.away_goals),
            kickoff_utc = EXCLUDED.kickoff_utc,
            match_date = EXCLUDED.match_date
    """), {"eid": m.event_id, "d": m.match_date, "ko": m.kickoff_utc, "season": m.season,
           "status": m.status, "h": m.home.team_id, "a": m.away.team_id,
           "hg": m.home_goals, "ag": m.away_goals})


def _upsert_stats(conn, event_id: int, home_team_id: int, rows: list[MlsTeamStats]) -> None:
    corners = {}
    for s in rows:
        is_home = s.team_id == home_team_id  # authoritative, not payload order
        conn.execute(text("""
            INSERT INTO mls.match_stats (event_id, team_id, is_home, corners, total_shots,
                shots_on_target, possession_pct, fouls, offsides, yellow_cards, red_cards, saves)
            VALUES (:eid, :tid, :ih, :c, :ts, :st, :pp, :f, :o, :yc, :rc, :sv)
            ON CONFLICT (event_id, team_id) DO UPDATE SET
                corners = EXCLUDED.corners, total_shots = EXCLUDED.total_shots,
                shots_on_target = EXCLUDED.shots_on_target, possession_pct = EXCLUDED.possession_pct,
                fouls = EXCLUDED.fouls, offsides = EXCLUDED.offsides,
                yellow_cards = EXCLUDED.yellow_cards, red_cards = EXCLUDED.red_cards,
                saves = EXCLUDED.saves
        """), {"eid": event_id, "tid": s.team_id, "ih": is_home, "c": s.corners,
               "ts": s.total_shots, "st": s.shots_on_target, "pp": s.possession_pct,
               "f": s.fouls, "o": s.offsides, "yc": s.yellow_cards, "rc": s.red_cards,
               "sv": s.saves})
        corners["home" if is_home else "away"] = s.corners
    conn.execute(text("""
        UPDATE mls.matches SET home_corners = :hc, away_corners = :ac,
               stats_filled_at_utc = :now WHERE event_id = :eid
    """), {"hc": corners.get("home"), "ac": corners.get("away"),
           "now": datetime.now(timezone.utc), "eid": event_id})


def ingest_dates(engine: Engine, dates: list[date], client: EspnClient | None = None) -> int:
    client = client or EspnClient()
    n = 0
    for d in dates:
        payload = client.scoreboard(d.strftime("%Y%m%d"))
        matches = parse_scoreboard_events(payload)
        with engine.begin() as conn:
            for m in matches:
                _upsert_match(conn, m)
                n += 1
    return n


def ingest_stats_for_unstatted(engine: Engine, limit: int = 60, client: EspnClient | None = None) -> int:
    """Fetch summaries (corners/covariates) for finished matches missing stats."""
    client = client or EspnClient()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT event_id, home_team_id FROM mls.matches
            WHERE status = 'FT' AND stats_filled_at_utc IS NULL
            ORDER BY match_date DESC LIMIT :lim
        """), {"lim": limit}).fetchall()
    n = 0
    for eid, home_id in rows:
        try:
            stats = parse_summary_stats(eid, client.summary(eid))
        except RuntimeError as e:
            logger.warning("summary fetch failed for %s: %s", eid, e)
            continue
        with engine.begin() as conn:
            if stats:
                _upsert_stats(conn, eid, home_id, stats)
            else:  # no boxscore available — mark so we don't refetch forever
                conn.execute(text("UPDATE mls.matches SET stats_filled_at_utc = :now WHERE event_id = :eid"),
                             {"now": datetime.now(timezone.utc), "eid": eid})
        n += 1
    return n


def ingest_recent_window(config: Config | None = None) -> dict:
    """Daily ingest: yesterday/today/tomorrow (display TZ) + stats trickle."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    today = datetime.now(DISPLAY_TZ).date()
    days = [today - timedelta(days=1), today, today + timedelta(days=1)]
    n_matches = ingest_dates(engine, days)
    n_stats = ingest_stats_for_unstatted(engine, limit=40)
    logger.info("MLS daily ingest: %s matches upserted, %s summaries fetched", n_matches, n_stats)
    return {"matches": n_matches, "stats": n_stats}


def backfill(config: Config | None = None, *, start: date, end: date | None = None,
             with_stats: bool = True) -> dict:
    """Historical backfill by walking scoreboard dates (skips January)."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    client = EspnClient()
    end = end or datetime.now(DISPLAY_TZ).date()
    n_matches = 0
    d = start
    while d <= end:
        if d.month in SEASON_MONTHS:
            n_matches += ingest_dates(engine, [d], client)
        d += timedelta(days=1)
        if d.day == 1:
            logger.info("MLS backfill progress: through %s (%s match-upserts)", d, n_matches)
    n_stats = 0
    if with_stats:
        while True:
            got = ingest_stats_for_unstatted(engine, limit=200, client=client)
            n_stats += got
            if got == 0:
                break
    logger.info("MLS backfill complete: %s match-upserts, %s summaries", n_matches, n_stats)
    return {"matches": n_matches, "stats": n_stats}
