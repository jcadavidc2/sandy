"""Ingestion service — backfill and incremental ingest.

Tasks 6.1, 6.2, 6.3:
- backfill_seasons(): fetch 3 complete regular seasons from MLB Stats API
- incremental_ingest(): fetch only games newer than the DB's max final date
- Failure recording: exhausted retries → raw.ingest_failures, continue

All DB writes use a single transaction per game_pk so each game is atomic.
Idempotent: games already present with status='Final' are skipped.

Requirements: 1.1–1.7, 2.1–2.5, 10.2
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.ingest.client import MlbApiError, MlbApiRetryExhausted, MlbStatsClient
from sandy.ingest.parsers import (
    ParsedGame,
    ScheduleEntry,
    parse_live_feed,
    parse_schedule,
    parse_teams,
)
from sandy.logging import get_logger
from sandy.schemas import GameRow, IngestFailureRow

logger = get_logger("ingest.service")

# How many seasons to backfill (the 3 most recent complete seasons)
_BACKFILL_SEASON_COUNT = 3
# Progress log every N games
_PROGRESS_EVERY = 50


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class BackfillStats:
    seasons: list[int] = field(default_factory=list)
    games_processed: int = 0
    games_skipped: int = 0
    games_failed: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IncrementalStats:
    games_added: int = 0
    games_updated: int = 0
    games_skipped: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def backfill_seasons(
    conn: Connection,
    client: MlbStatsClient,
    seasons: list[int] | None = None,
) -> BackfillStats:
    """Fetch every regular-season game for the given seasons and persist them.

    If *seasons* is None, derives the three most recent complete seasons from
    the current year (requirement 1.1).

    Idempotent: games already in raw.games with status='Final' are skipped
    (requirement 1.2, 1.6). Progress is logged every 50 games (requirement
    1.5). Failures are recorded to raw.ingest_failures and processing
    continues (requirement 1.4).
    """
    if seasons is None:
        current_year = date.today().year
        # Most recent complete season ends the previous year
        last_complete = current_year - 1
        seasons = list(range(last_complete - _BACKFILL_SEASON_COUNT + 1, last_complete + 1))

    stats = BackfillStats(seasons=seasons)
    t0 = time.monotonic()

    # Ensure teams are loaded first (needed for FK constraints)
    _upsert_teams(conn, client)

    # Collect all game_pks across all seasons
    all_entries: list[ScheduleEntry] = []
    for season in seasons:
        entries = _fetch_season_schedule(conn, client, season)
        all_entries.extend(entries)
        logger.info(
            "Schedule fetched",
            extra={
                "component": "ingest.service",
                "season": season,
                "games_found": len(entries),
            },
        )

    # Filter to only regular-season games not already Final in DB
    final_pks = _get_final_game_pks(conn)
    pending = [e for e in all_entries if e.game_type == "R" and e.game_pk not in final_pks]
    total = len(pending)
    skipped = len(all_entries) - total
    stats.games_skipped = skipped

    logger.info(
        "Backfill starting",
        extra={
            "component": "ingest.service",
            "total_pending": total,
            "already_final": skipped,
            "seasons": seasons,
        },
    )

    for i, entry in enumerate(pending, start=1):
        success = _ingest_one_game(conn, client, entry.game_pk)
        if success:
            stats.games_processed += 1
        else:
            stats.games_failed += 1

        if i % _PROGRESS_EVERY == 0 or i == total:
            elapsed = time.monotonic() - t0
            logger.info(
                "Backfill progress",
                extra={
                    "component": "ingest.service",
                    "games_processed": stats.games_processed,
                    "games_remaining": total - i,
                    "games_failed": stats.games_failed,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

    stats.elapsed_seconds = round(time.monotonic() - t0, 1)
    logger.info(
        "Backfill complete",
        extra={
            "component": "ingest.service",
            "games_processed": stats.games_processed,
            "games_skipped": stats.games_skipped,
            "games_failed": stats.games_failed,
            "duration_seconds": stats.elapsed_seconds,
            "rows_read": 0,
            "rows_written": stats.games_processed,
        },
    )
    return stats


def incremental_ingest(
    conn: Connection,
    client: MlbStatsClient,
) -> IncrementalStats:
    """Fetch games newer than the most recent Final game in the DB.

    - Skips games already Final (requirement 2.2)
    - Replaces non-Final → Final games (requirement 2.3)
    - Idempotent if run twice (requirement 2.4)
    - Logs summary on completion (requirement 2.5)

    Requirements: 2.1–2.5
    """
    stats = IncrementalStats()
    t0 = time.monotonic()

    max_final_date = _get_max_final_date(conn)
    if max_final_date is None:
        logger.info(
            "No Final games in DB; run backfill first",
            extra={"component": "ingest.service"},
        )
        stats.elapsed_seconds = round(time.monotonic() - t0, 1)
        return stats

    # Fetch from max_final_date (inclusive — catches late-finalizing games)
    today = date.today()
    entries = _fetch_schedule_window(client, max_final_date, today)
    regular = [e for e in entries if e.game_type == "R"]

    final_pks = _get_final_game_pks(conn)
    all_pks_in_db = _get_all_game_pks(conn)

    for entry in regular:
        if entry.game_pk in final_pks:
            stats.games_skipped += 1
            continue

        success = _ingest_one_game(conn, client, entry.game_pk)
        if success:
            if entry.game_pk in all_pks_in_db:
                stats.games_updated += 1
            else:
                stats.games_added += 1

    stats.elapsed_seconds = round(time.monotonic() - t0, 1)
    logger.info(
        "Incremental ingest complete",
        extra={
            "component": "ingest.service",
            "games_added": stats.games_added,
            "games_updated": stats.games_updated,
            "games_skipped": stats.games_skipped,
            "duration_seconds": stats.elapsed_seconds,
            "rows_read": 0,
            "rows_written": stats.games_added + stats.games_updated,
        },
    )
    return stats


# ---------------------------------------------------------------------------
# Core per-game ingest
# ---------------------------------------------------------------------------


def _ingest_one_game(
    conn: Connection,
    client: MlbStatsClient,
    game_pk: int,
) -> bool:
    """Fetch and persist one game. Returns True on success, False on failure."""
    try:
        payload = client.get(f"/v1.1/game/{game_pk}/feed/live")
        parsed = parse_live_feed(game_pk, payload)
        _write_game_transaction(conn, parsed)
        return True

    except MlbApiRetryExhausted as exc:
        _record_failure(
            conn,
            game_pk=game_pk,
            endpoint=f"/v1.1/game/{game_pk}/feed/live",
            error_reason=str(exc),
            http_status=exc.http_status,
            retries=exc.retries,
        )
        logger.warning(
            "Game ingest failed after retries",
            extra={
                "component": "ingest.service",
                "game_pk": game_pk,
                "error": str(exc),
            },
        )
        return False

    except MlbApiError as exc:
        _record_failure(
            conn,
            game_pk=game_pk,
            endpoint=f"/v1.1/game/{game_pk}/feed/live",
            error_reason=str(exc),
            http_status=exc.http_status,
            retries=0,
        )
        logger.warning(
            "Game ingest non-retryable error",
            extra={
                "component": "ingest.service",
                "game_pk": game_pk,
                "error": str(exc),
            },
        )
        return False

    except Exception as exc:
        _record_failure(
            conn,
            game_pk=game_pk,
            endpoint=f"/v1.1/game/{game_pk}/feed/live",
            error_reason=str(exc),
            http_status=None,
            retries=0,
        )
        logger.error(
            "Unexpected error ingesting game",
            extra={
                "component": "ingest.service",
                "game_pk": game_pk,
                "error": str(exc),
            },
        )
        return False


def _write_game_transaction(conn: Connection, parsed: ParsedGame) -> None:
    """Write all rows for one game atomically.

    Order:
    1. UPSERT teams referenced by this game (FK prerequisite)
    2. UPSERT players
    3. DELETE existing plays (idempotent re-ingest)
    4. UPSERT game row
    5. INSERT plays
    6. UPSERT pitcher_game_stats

    Uses a savepoint so a failure here rolls back only this game, not the
    whole session (the caller's transaction remains open).
    """
    game_pk = parsed.game["game_pk"]

    with conn.begin_nested():  # savepoint
        # Ensure home/away teams exist (FK prerequisite for games table).
        # The /v1/teams endpoint may not return all historical teams, so we
        # also insert teams discovered in each game feed.
        for code in (parsed.game["home_team_code"], parsed.game["away_team_code"]):
            code = code.strip()
            conn.execute(
                text("""
                    INSERT INTO raw.teams (team_code, team_id, name)
                    VALUES (:code, :tid, :name)
                    ON CONFLICT (team_code) DO NOTHING
                """),
                {"code": code, "tid": abs(hash(code)) % 900000 + 100000, "name": code},
            )

        # Players (FK for games and plays)
        for player in parsed.players:
            conn.execute(
                text("""
                    INSERT INTO raw.players
                        (player_id, full_name, primary_position, throws, bats)
                    VALUES
                        (:player_id, :full_name, :primary_position, :throws, :bats)
                    ON CONFLICT (player_id) DO UPDATE SET
                        full_name        = EXCLUDED.full_name,
                        primary_position = EXCLUDED.primary_position,
                        throws           = EXCLUDED.throws,
                        bats             = EXCLUDED.bats,
                        ingested_at      = now()
                """),
                dict(player),
            )

        # Delete existing plays for this game (idempotent)
        conn.execute(
            text("DELETE FROM raw.plays WHERE game_pk = :game_pk"),
            {"game_pk": game_pk},
        )

        # Upsert game row
        g = parsed.game
        conn.execute(
            text("""
                INSERT INTO raw.games (
                    game_pk, game_date, season, game_type, status,
                    home_team_code, away_team_code, venue_id,
                    first_pitch_utc, home_score, away_score,
                    home_starter_id, away_starter_id, raw_payload_hash
                ) VALUES (
                    :game_pk, :game_date, :season, :game_type, :status,
                    :home_team_code, :away_team_code, :venue_id,
                    :first_pitch_utc, :home_score, :away_score,
                    :home_starter_id, :away_starter_id, :raw_payload_hash
                )
                ON CONFLICT (game_pk) DO UPDATE SET
                    status           = EXCLUDED.status,
                    home_score       = EXCLUDED.home_score,
                    away_score       = EXCLUDED.away_score,
                    home_starter_id  = EXCLUDED.home_starter_id,
                    away_starter_id  = EXCLUDED.away_starter_id,
                    raw_payload_hash = EXCLUDED.raw_payload_hash,
                    ingested_at      = now()
            """),
            {k: v for k, v in g.items()},
        )

        # Insert plays
        for play in parsed.plays:
            p = dict(play)
            # Serialize raw JSONB field
            import json as _json
            p["raw"] = _json.dumps(p["raw"])
            conn.execute(
                text("""
                    INSERT INTO raw.plays (
                        game_pk, at_bat_index, inning, half_inning,
                        batting_team_code, pitching_team_code,
                        batter_id, pitcher_id, batting_order,
                        event_type, event_code, is_reaches_base,
                        pitches_in_pa, start_time_utc, end_time_utc, raw
                    ) VALUES (
                        :game_pk, :at_bat_index, :inning, :half_inning,
                        :batting_team_code, :pitching_team_code,
                        :batter_id, :pitcher_id, :batting_order,
                        :event_type, :event_code, :is_reaches_base,
                        :pitches_in_pa, :start_time_utc, :end_time_utc,
                        CAST(:raw AS jsonb)
                    )
                    ON CONFLICT (game_pk, at_bat_index) DO NOTHING
                """),
                p,
            )

        # Upsert pitcher game stats
        for ps in parsed.pitcher_stats:
            conn.execute(
                text("""
                    INSERT INTO raw.pitcher_game_stats (
                        game_pk, pitcher_id, team_code,
                        pitches_thrown, outs_recorded, runs_allowed,
                        walks, hits_allowed, strikeouts, is_starter
                    ) VALUES (
                        :game_pk, :pitcher_id, :team_code,
                        :pitches_thrown, :outs_recorded, :runs_allowed,
                        :walks, :hits_allowed, :strikeouts, :is_starter
                    )
                    ON CONFLICT (game_pk, pitcher_id) DO UPDATE SET
                        pitches_thrown = EXCLUDED.pitches_thrown,
                        outs_recorded  = EXCLUDED.outs_recorded,
                        runs_allowed   = EXCLUDED.runs_allowed,
                        walks          = EXCLUDED.walks,
                        hits_allowed   = EXCLUDED.hits_allowed,
                        strikeouts     = EXCLUDED.strikeouts,
                        is_starter     = EXCLUDED.is_starter
                """),
                dict(ps),
            )


# ---------------------------------------------------------------------------
# Failure recording (requirement 1.4 / 6.2)
# ---------------------------------------------------------------------------


def _record_failure(
    conn: Connection,
    game_pk: int | None,
    endpoint: str,
    error_reason: str,
    http_status: int | None,
    retries: int,
) -> None:
    """Insert a row into raw.ingest_failures. Never raises."""
    try:
        conn.execute(
            text("""
                INSERT INTO raw.ingest_failures
                    (game_pk, endpoint, error_reason, http_status, retries)
                VALUES
                    (:game_pk, :endpoint, :error_reason, :http_status, :retries)
            """),
            {
                "game_pk": game_pk,
                "endpoint": endpoint,
                "error_reason": error_reason[:2000],
                "http_status": http_status,
                "retries": retries,
            },
        )
    except Exception:
        pass  # failure recording must never crash the pipeline


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _fetch_season_schedule(
    conn: Connection,
    client: MlbStatsClient,
    season: int,
) -> list[ScheduleEntry]:
    """Fetch all games for a season in one-month windows."""
    entries: list[ScheduleEntry] = []
    start = date(season, 3, 1)   # MLB regular season starts late March
    end = date(season, 11, 30)   # ends late October / early November

    current = start
    while current <= end:
        window_end = min(
            date(current.year, current.month + 1, 1) - timedelta(days=1)
            if current.month < 12
            else date(current.year, 12, 31),
            end,
        )
        try:
            payload = client.get(
                "/v1/schedule",
                params={
                    "sportId": "1",
                    "startDate": current.isoformat(),
                    "endDate": window_end.isoformat(),
                    "gameType": "R",
                    "season": str(season),
                },
            )
            entries.extend(parse_schedule(payload))
        except (MlbApiError, MlbApiRetryExhausted) as exc:
            logger.warning(
                "Schedule fetch failed for window",
                extra={
                    "component": "ingest.service",
                    "season": season,
                    "start": current.isoformat(),
                    "end": window_end.isoformat(),
                    "error": str(exc),
                },
            )
        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return entries


def _fetch_schedule_window(
    client: MlbStatsClient,
    start: date,
    end: date,
) -> list[ScheduleEntry]:
    """Fetch schedule for an arbitrary date window (used by incremental)."""
    entries: list[ScheduleEntry] = []
    current = start
    while current <= end:
        if current.month == 12:
            window_end = min(date(current.year, 12, 31), end)
        else:
            window_end = min(
                date(current.year, current.month + 1, 1) - timedelta(days=1),
                end,
            )
        try:
            payload = client.get(
                "/v1/schedule",
                params={
                    "sportId": "1",
                    "startDate": current.isoformat(),
                    "endDate": window_end.isoformat(),
                    "gameType": "R",
                },
            )
            entries.extend(parse_schedule(payload))
        except (MlbApiError, MlbApiRetryExhausted):
            pass
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return entries


# ---------------------------------------------------------------------------
# Teams bootstrap
# ---------------------------------------------------------------------------


def _upsert_teams(conn: Connection, client: MlbStatsClient) -> None:
    """Ensure raw.teams is populated (FK prerequisite for games/plays)."""
    count = conn.execute(text("SELECT COUNT(*) FROM raw.teams")).scalar()
    if count and count > 0:
        return  # already loaded

    try:
        payload = client.get("/v1/teams", params={"sportId": "1"})
        teams = parse_teams(payload)
        for team in teams:
            conn.execute(
                text("""
                    INSERT INTO raw.teams
                        (team_code, team_id, name, venue_id, league, division)
                    VALUES
                        (:team_code, :team_id, :name, :venue_id, :league, :division)
                    ON CONFLICT (team_code) DO UPDATE SET
                        team_id  = EXCLUDED.team_id,
                        name     = EXCLUDED.name,
                        venue_id = EXCLUDED.venue_id,
                        league   = EXCLUDED.league,
                        division = EXCLUDED.division
                """),
                dict(team),
            )
        logger.info(
            "Teams loaded",
            extra={"component": "ingest.service", "teams_count": len(teams)},
        )
    except Exception as exc:
        logger.warning(
            "Could not load teams",
            extra={"component": "ingest.service", "error": str(exc)},
        )


# ---------------------------------------------------------------------------
# DB query helpers
# ---------------------------------------------------------------------------


def _get_final_game_pks(conn: Connection) -> set[int]:
    rows = conn.execute(
        text("SELECT game_pk FROM raw.games WHERE status = 'Final'")
    )
    return {r[0] for r in rows}


def _get_all_game_pks(conn: Connection) -> set[int]:
    rows = conn.execute(text("SELECT game_pk FROM raw.games"))
    return {r[0] for r in rows}


def _get_max_final_date(conn: Connection) -> date | None:
    result = conn.execute(
        text("SELECT MAX(game_date) FROM raw.games WHERE status = 'Final'")
    ).scalar()
    return result  # already a date or None


__all__ = [
    "BackfillStats",
    "IncrementalStats",
    "backfill_seasons",
    "incremental_ingest",
]
