"""NHL ingestion from the official keyless API (api-web.nhle.com).

Backfill: per-team season schedules (32 teams x N seasons ≈ 130 requests — each
game arrives twice and upserts idempotently). Daily: /v1/schedule/{date} for
yesterday/today/tomorrow. gameType 1 (preseason) and 4+ (all-star etc.) excluded.

Regulation score rule: lastPeriodType != REG  ⇒  regulation ended TIED at the
loser's score (OT winner scored once in OT; a shootout adds one display goal),
so reg_home = reg_away = min(home_goals, away_goals).
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

BASE = "https://api-web.nhle.com/v1"
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "add_nhl_tables.sql"
GAME_TYPES = (2, 3)  # regular season + playoffs

TEAM_ABBREVS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM",
    "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT",
    "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH", "ARI",
]

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


_last_call = 0.0


def _get(url: str, retries: int = 3) -> dict | None:
    """Polite (1 rps), retried, IPv4-forced GET. 404 → None (e.g. relocated team)."""
    global _last_call
    for attempt in range(retries):
        wait = 1.0 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            socket.getaddrinfo = _ipv4
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "sandy/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return json.loads(resp.read().decode())
            finally:
                socket.getaddrinfo = _orig_getaddrinfo
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            if e.code == 404:
                return None
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            logger.warning("NHL request failed (attempt %s): %s", attempt + 1, e)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"NHL request failed after {retries} attempts: {url}")


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())


def _status(game_state: str) -> str:
    if game_state in ("FINAL", "OFF"):
        return "FINAL"
    if game_state in ("FUT", "PRE"):
        return "FUT"
    return "LIVE"


def _upsert_game(conn, g: dict) -> bool:
    """Upsert one game dict from either schedule shape. Returns True if kept."""
    if g.get("gameType") not in GAME_TYPES:
        return False
    home, away = g["homeTeam"], g["awayTeam"]
    for t in (home, away):
        conn.execute(text("""
            INSERT INTO nhl.teams (team_id, abbrev, name) VALUES (:id, :ab, :name)
            ON CONFLICT (team_id) DO UPDATE SET abbrev = EXCLUDED.abbrev,
                name = COALESCE(EXCLUDED.name, nhl.teams.name)
        """), {"id": t["id"], "ab": t["abbrev"],
               "name": (t.get("placeName", {}) or {}).get("default")})
    start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
    status = _status(g.get("gameState", "FUT"))
    hg = home.get("score") if status == "FINAL" else None
    ag = away.get("score") if status == "FINAL" else None
    lpt = (g.get("gameOutcome") or {}).get("lastPeriodType")
    reg_h = reg_a = None
    if status == "FINAL" and hg is not None and ag is not None:
        if lpt and lpt != "REG":
            reg_h = reg_a = min(hg, ag)
        else:
            reg_h, reg_a = hg, ag
    conn.execute(text("""
        INSERT INTO nhl.games (game_id, match_date, start_utc, season, game_type, status,
            home_team_id, away_team_id, home_goals, away_goals, last_period_type,
            reg_home_goals, reg_away_goals)
        VALUES (:gid, :d, :ts, :season, :gt, :st, :h, :a, :hg, :ag, :lpt, :rh, :ra)
        ON CONFLICT (game_id) DO UPDATE SET
            status = EXCLUDED.status,
            home_goals = COALESCE(EXCLUDED.home_goals, nhl.games.home_goals),
            away_goals = COALESCE(EXCLUDED.away_goals, nhl.games.away_goals),
            last_period_type = COALESCE(EXCLUDED.last_period_type, nhl.games.last_period_type),
            reg_home_goals = COALESCE(EXCLUDED.reg_home_goals, nhl.games.reg_home_goals),
            reg_away_goals = COALESCE(EXCLUDED.reg_away_goals, nhl.games.reg_away_goals),
            start_utc = EXCLUDED.start_utc, match_date = EXCLUDED.match_date
    """), {"gid": g["id"], "d": start.astimezone(DISPLAY_TZ).date(), "ts": start,
           "season": g.get("season"), "gt": g.get("gameType"), "st": status,
           "h": home["id"], "a": away["id"], "hg": hg, "ag": ag, "lpt": lpt,
           "rh": reg_h, "ra": reg_a})
    return True


def backfill_seasons(config: Config | None = None, *, seasons: list[str] | None = None) -> int:
    """Per-team season schedules — ~33 requests per season, each game seen twice."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    seasons = seasons or ["20222023", "20232024", "20242025", "20252026"]
    n = 0
    for season in seasons:
        for ab in TEAM_ABBREVS:
            payload = _get(f"{BASE}/club-schedule-season/{ab}/{season}")
            if not payload:
                continue
            games = payload.get("games", [])
            with engine.begin() as conn:
                for g in games:
                    if _upsert_game(conn, g):
                        n += 1
        logger.info("NHL backfill: season %s done (%s upserts so far)", season, n)
    return n


def ingest_recent_window(config: Config | None = None) -> int:
    """Daily: yesterday/today/tomorrow via /v1/schedule/{date} (self-healing)."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    today = datetime.now(DISPLAY_TZ).date()
    n = 0
    for d in (today - timedelta(days=1), today, today + timedelta(days=1)):
        payload = _get(f"{BASE}/schedule/{d.isoformat()}")
        if not payload:
            continue
        for day in payload.get("gameWeek", []):
            if day.get("date") != d.isoformat():
                continue
            with engine.begin() as conn:
                for g in day.get("games", []):
                    if _upsert_game(conn, g):
                        n += 1
    logger.info("NHL daily ingest: %s games upserted", n)
    return n
