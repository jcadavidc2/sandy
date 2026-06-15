"""Ingestion: API-Football -> Postgres `football` schema.

Mirrors the baseball ingest service (``sandy.ingest.service``): pure parsers
produce typed rows, then idempotent UPSERTs persist them. Teams are written
before matches to satisfy the FK; per-match statistics are fetched separately
and throttled, because each is one request against the free-tier daily cap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football import parsers as P
from sandy.football.client import ApiFootballClient, FootballApiError
from sandy.football.schemas import FINISHED_STATUSES, MatchRow, MatchStatRow, TeamRow
from sandy.logging import get_logger

logger = get_logger("football.ingest")

# API-Football league ids for SENIOR MEN'S NATIONAL-TEAM competitions, with a
# match-importance weight (friendly < qualifier < continental < World Cup).
# Deliberately EXCLUDES youth (U17/U20/U23), women's, and club competitions so
# the ratings reflect senior men's national-team strength only.
WORLD_CUP_LEAGUE_ID = 1

COMPETITION_WEIGHTS: dict[int, float] = {
    1: 3.0,     # World Cup
    # Continental finals
    4: 2.5,     # Euro Championship
    9: 2.5,     # Copa America
    6: 2.5,     # Africa Cup of Nations
    7: 2.5,     # Asian Cup
    22: 2.5,    # CONCACAF Gold Cup
    # Nations League (group + finals)
    5: 2.0,     # UEFA Nations League
    536: 2.0,   # CONCACAF Nations League
    # World Cup qualifiers (all confederations + intercontinental playoff)
    29: 1.8, 30: 1.8, 31: 1.8, 32: 1.8, 33: 1.8, 34: 1.8, 37: 1.8,
    # Continental qualifiers
    960: 1.8,   # Euro Qualification
    36: 1.8,    # AFCON Qualification
    35: 1.8,    # Asian Cup Qualification
    858: 1.8,   # Gold Cup Qualification
    # Minor regional senior tournaments
    25: 1.3,    # Gulf Cup of Nations
    24: 1.3,    # ASEAN Championship
    28: 1.3,    # SAFF Championship
    859: 1.3,   # COSAFA Cup
    1008: 1.3,  # CAFA Nations Cup
    913: 1.3,   # CONMEBOL-UEFA Finalissima
    # Friendlies
    10: 1.0,    # Friendlies
}
DEFAULT_COMPETITION_WEIGHT = 1.0

# Senior men's national-team (league_id, season) pairs available on the free
# tier (2022-2024), derived from /leagues coverage. Used for the training
# backfill that powers the Dixon-Coles model + calibration backtest.
SENIOR_NT_BACKFILL: list[tuple[int, int]] = [
    (1, 2022),                                   # World Cup 2022
    (10, 2022), (10, 2023), (10, 2024),          # Friendlies
    (5, 2022), (5, 2024),                        # UEFA Nations League
    (536, 2022), (536, 2023), (536, 2024),       # CONCACAF Nations League
    (4, 2024), (960, 2023),                      # Euro 2024 + qualifying
    (9, 2024),                                   # Copa America 2024
    (6, 2023), (36, 2023),                       # AFCON 2023 + qualifying
    (7, 2023), (35, 2022), (35, 2024),           # Asian Cup 2023 + qualifying
    (22, 2023), (858, 2023),                     # Gold Cup 2023 + qualifying
    (25, 2023), (25, 2024),                      # Gulf Cup
    (24, 2022), (24, 2024),                      # ASEAN Championship
    (28, 2023), (859, 2022), (859, 2023), (859, 2024),  # SAFF + COSAFA
    (1008, 2023), (913, 2022),                   # CAFA Nations Cup + Finalissima
    (29, 2022), (29, 2023), (30, 2022), (31, 2022),     # WC qualifiers
    (32, 2024), (33, 2022), (34, 2022), (37, 2022),
]


def competition_weight(league_id: int | None) -> float:
    if league_id is None:
        return DEFAULT_COMPETITION_WEIGHT
    return COMPETITION_WEIGHTS.get(league_id, DEFAULT_COMPETITION_WEIGHT)


@dataclass
class IngestStats:
    league_id: int
    season: int
    teams_upserted: int
    matches_upserted: int
    fixtures_results: int


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def ingest_fixtures(
    engine: Engine,
    client: ApiFootballClient,
    league_id: int,
    season: int,
    *,
    enrich_teams: bool = True,
) -> IngestStats:
    """Fetch + upsert all fixtures (and their teams) for one league-season.

    One ``/fixtures`` request per league-season (cheap). When ``enrich_teams``
    is set, one extra ``/teams`` request adds fifa_code/country for the report.
    """
    weight = competition_weight(league_id)
    env = client.get("/fixtures", {"league": str(league_id), "season": str(season)})
    matches = P.parse_fixtures(env, competition_weight=weight)

    # Teams referenced by the fixtures (minimal rows satisfy the matches FK).
    teams: dict[int, TeamRow] = {t.team_id: t for t in P.parse_teams_from_fixtures(env)}
    if enrich_teams:
        tenv = client.get("/teams", {"league": str(league_id), "season": str(season)})
        for t in P.parse_teams(tenv):
            teams[t.team_id] = t  # richer row (code/country) wins

    with engine.begin() as conn:
        n_teams = _upsert_teams(conn, teams.values())
        n_matches = _upsert_matches(conn, matches)

    logger.info(
        "Ingested fixtures",
        extra={
            "component": "football.ingest",
            "league_id": league_id,
            "season": season,
            "teams": n_teams,
            "matches": n_matches,
        },
    )
    return IngestStats(
        league_id=league_id,
        season=season,
        teams_upserted=n_teams,
        matches_upserted=n_matches,
        fixtures_results=env.get("results", len(matches)),
    )


def ingest_statistics_for_unstatted(
    engine: Engine,
    client: ApiFootballClient,
    *,
    league_id: int | None = None,
    season: int | None = None,
    limit: int = 20,
) -> int:
    """Fetch per-team statistics for finished matches that lack them.

    Each fixture costs one request, so ``limit`` caps how many we pull per run
    to respect the daily quota. Idempotent: only targets fixtures with no
    rows in ``football.match_stats``. Returns the number of fixtures statted.
    """
    rows = _find_unstatted_finished(engine, league_id, season, limit)
    statted = 0
    for fixture_id, home_team_id in rows:
        senv = client.get("/fixtures/statistics", {"fixture": str(fixture_id)})
        stats = P.parse_fixture_statistics(fixture_id, senv, home_team_id=home_team_id)
        if not stats:
            continue
        with engine.begin() as conn:
            _upsert_match_stats(conn, stats)
        statted += 1
    logger.info(
        "Ingested match statistics",
        extra={"component": "football.ingest", "fixtures_statted": statted, "requested": len(rows)},
    )
    return statted


def ingest_by_date(
    engine: Engine,
    client: ApiFootballClient,
    the_date: date,
    *,
    timezone: str = "America/Los_Angeles",
) -> IngestStats:
    """Ingest national-team fixtures for a single calendar date.

    Used by the daily loop: the free tier exposes a +/-1 day date window
    (yesterday/today/tomorrow). Date queries return ALL leagues, so we keep
    only senior men's national-team competitions (those in COMPETITION_WEIGHTS)
    and assign each match its per-competition weight.

    ``timezone`` groups matches by the user's local day (an evening PST kickoff
    is returned on the day it was watched, not the next UTC date), and the
    parsed ``match_date`` becomes that local date.
    """
    env = client.get("/fixtures", {"date": the_date.isoformat(), "timezone": timezone})
    parsed = P.parse_fixtures(env)  # placeholder weight; re-assigned per league below
    matches = [
        replace(m, competition_weight=competition_weight(m.league_id))
        for m in parsed
        if m.league_id in COMPETITION_WEIGHTS
    ]
    keep_team_ids = {m.home_team_id for m in matches} | {m.away_team_id for m in matches}
    teams = {
        t.team_id: t
        for t in P.parse_teams_from_fixtures(env)
        if t.team_id in keep_team_ids
    }

    with engine.begin() as conn:
        n_teams = _upsert_teams(conn, teams.values())
        n_matches = _upsert_matches(conn, matches)

    logger.info("Ingested fixtures by date", extra={
        "component": "football.ingest", "date": the_date.isoformat(),
        "teams": n_teams, "matches": n_matches,
    })
    return IngestStats(league_id=-1, season=the_date.year, teams_upserted=n_teams,
                       matches_upserted=n_matches, fixtures_results=len(matches))


def ingest_recent_window(
    config: Config | None = None,
    *,
    days_back: int = 1,
    days_forward: int = 1,
) -> list[IngestStats]:
    """Ingest the free-tier date window (yesterday..tomorrow by default).

    This is the daily-loop entry point for live data: finished matches in the
    past day get reconciled, upcoming matches get predicted.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = config or load_config()
    engine = create_engine(cfg)
    client = ApiFootballClient(cfg.football)
    tz = cfg.football.display_timezone
    today = datetime.now(ZoneInfo(tz)).date()   # "today" in the user's timezone
    out: list[IngestStats] = []
    for offset in range(-days_back, days_forward + 1):
        d = today + timedelta(days=offset)
        try:
            out.append(ingest_by_date(engine, client, d, timezone=tz))
        except FootballApiError as e:
            # Dates outside the free-tier +/-1 day window return an error body;
            # skip them rather than aborting the whole window.
            logger.info("Skipping date outside free window", extra={
                "component": "football.ingest", "date": d.isoformat(), "reason": str(e)[-80:],
            })
    return out


def backfill_competitions(
    specs: Iterable[tuple[int, int]],
    *,
    config: Config | None = None,
    enrich_teams: bool = True,
) -> list[IngestStats]:
    """Backfill fixtures for a list of (league_id, season) pairs."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    client = ApiFootballClient(cfg.football)
    out: list[IngestStats] = []
    for league_id, season in specs:
        out.append(ingest_fixtures(engine, client, league_id, season, enrich_teams=enrich_teams))
    return out


# ---------------------------------------------------------------------------
# UPSERT helpers
# ---------------------------------------------------------------------------


_TEAMS_UPSERT = text("""
    INSERT INTO football.teams
        (team_id, name, fifa_code, country, confederation, fifa_rank, logo_url)
    VALUES (:team_id, :name, :fifa_code, :country, :confederation, :fifa_rank, :logo_url)
    ON CONFLICT (team_id) DO UPDATE SET
        name          = EXCLUDED.name,
        fifa_code     = COALESCE(EXCLUDED.fifa_code, football.teams.fifa_code),
        country       = COALESCE(EXCLUDED.country, football.teams.country),
        confederation = COALESCE(EXCLUDED.confederation, football.teams.confederation),
        fifa_rank     = COALESCE(EXCLUDED.fifa_rank, football.teams.fifa_rank),
        logo_url      = COALESCE(EXCLUDED.logo_url, football.teams.logo_url)
""")

_MATCHES_UPSERT = text("""
    INSERT INTO football.matches
        (fixture_id, match_date, kickoff_utc, league_id, season, competition,
         competition_weight, round, home_team_id, away_team_id, venue_id,
         venue_name, status, home_goals, away_goals, raw_payload_hash)
    VALUES
        (:fixture_id, :match_date, :kickoff_utc, :league_id, :season, :competition,
         :competition_weight, :round, :home_team_id, :away_team_id, :venue_id,
         :venue_name, :status, :home_goals, :away_goals, :raw_payload_hash)
    ON CONFLICT (fixture_id) DO UPDATE SET
        match_date         = EXCLUDED.match_date,
        kickoff_utc        = EXCLUDED.kickoff_utc,
        competition        = EXCLUDED.competition,
        competition_weight = EXCLUDED.competition_weight,
        round              = EXCLUDED.round,
        venue_id           = EXCLUDED.venue_id,
        venue_name         = EXCLUDED.venue_name,
        status             = EXCLUDED.status,
        home_goals         = EXCLUDED.home_goals,
        away_goals         = EXCLUDED.away_goals,
        raw_payload_hash   = EXCLUDED.raw_payload_hash,
        ingested_at        = now()
""")

_STATS_UPSERT = text("""
    INSERT INTO football.match_stats
        (fixture_id, team_id, is_home, possession, shots_total, shots_on_target,
         corners, fouls, yellow_cards, red_cards, xg)
    VALUES
        (:fixture_id, :team_id, :is_home, :possession, :shots_total, :shots_on_target,
         :corners, :fouls, :yellow_cards, :red_cards, :xg)
    ON CONFLICT (fixture_id, team_id) DO UPDATE SET
        is_home         = EXCLUDED.is_home,
        possession      = EXCLUDED.possession,
        shots_total     = EXCLUDED.shots_total,
        shots_on_target = EXCLUDED.shots_on_target,
        corners         = EXCLUDED.corners,
        fouls           = EXCLUDED.fouls,
        yellow_cards    = EXCLUDED.yellow_cards,
        red_cards       = EXCLUDED.red_cards,
        xg              = EXCLUDED.xg,
        ingested_at     = now()
""")


def _upsert_teams(conn, teams: Iterable[TeamRow]) -> int:
    n = 0
    for t in teams:
        conn.execute(_TEAMS_UPSERT, {
            "team_id": t.team_id, "name": t.name, "fifa_code": t.fifa_code,
            "country": t.country, "confederation": t.confederation,
            "fifa_rank": t.fifa_rank, "logo_url": t.logo_url,
        })
        n += 1
    return n


def _upsert_matches(conn, matches: Iterable[MatchRow]) -> int:
    n = 0
    for m in matches:
        conn.execute(_MATCHES_UPSERT, {
            "fixture_id": m.fixture_id, "match_date": m.match_date,
            "kickoff_utc": m.kickoff_utc, "league_id": m.league_id, "season": m.season,
            "competition": m.competition, "competition_weight": m.competition_weight,
            "round": m.round, "home_team_id": m.home_team_id, "away_team_id": m.away_team_id,
            "venue_id": m.venue_id, "venue_name": m.venue_name, "status": m.status,
            "home_goals": m.home_goals, "away_goals": m.away_goals,
            "raw_payload_hash": m.raw_payload_hash,
        })
        n += 1
    return n


def _upsert_match_stats(conn, stats: Iterable[MatchStatRow]) -> int:
    n = 0
    for s in stats:
        conn.execute(_STATS_UPSERT, {
            "fixture_id": s.fixture_id, "team_id": s.team_id, "is_home": s.is_home,
            "possession": s.possession, "shots_total": s.shots_total,
            "shots_on_target": s.shots_on_target, "corners": s.corners, "fouls": s.fouls,
            "yellow_cards": s.yellow_cards, "red_cards": s.red_cards, "xg": s.xg,
        })
        n += 1
    return n


def _find_unstatted_finished(
    engine: Engine, league_id: int | None, season: int | None, limit: int,
) -> list[tuple[int, int]]:
    clauses = ["m.status = ANY(:finished)", "s.fixture_id IS NULL"]
    params: dict = {"finished": list(FINISHED_STATUSES), "limit": limit}
    if league_id is not None:
        clauses.append("m.league_id = :league_id")
        params["league_id"] = league_id
    if season is not None:
        clauses.append("m.season = :season")
        params["season"] = season
    where = " AND ".join(clauses)
    sql = text(f"""
        SELECT m.fixture_id, m.home_team_id
        FROM football.matches m
        LEFT JOIN football.match_stats s ON s.fixture_id = m.fixture_id
        WHERE {where}
        ORDER BY m.match_date
        LIMIT :limit
    """)
    with engine.connect() as conn:
        return [(int(r[0]), int(r[1])) for r in conn.execute(sql, params).fetchall()]


__all__ = [
    "COMPETITION_WEIGHTS",
    "WORLD_CUP_LEAGUE_ID",
    "IngestStats",
    "backfill_competitions",
    "competition_weight",
    "ingest_fixtures",
    "ingest_statistics_for_unstatted",
]
