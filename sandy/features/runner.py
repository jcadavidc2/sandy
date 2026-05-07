"""Features batch runner — persists feature vectors for all Final game innings.

Task 8.3: iterates all (game_pk, team_code, inning_number) tuples from
derived.inning_labels (which only exist for Final games), calls
build_feature_vector() for each, omits rows where a feature is uncomputable,
and UPSERTs into derived.inning_features.

Idempotent via ON CONFLICT DO UPDATE (requirement 3.4).
Emits a final JSON log line with duration_seconds, rows_read, rows_written
(requirement 10.2).

Requirements: 3.4, 5.4, 5.5, 10.1, 10.2
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from sandy.db import get_connection
from sandy.features.builder import build_feature_vector
from sandy.features.schema import FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger

logger = get_logger("features.runner")

# Features that must be non-None for a row to be included in training.
# ERA/WHIP/K9 can be None for pitchers with no prior season data — we
# allow those rows through with 0.0 fallback (set in builder).
# trailing15 stats can also be None for teams with no prior games.
# We only hard-omit rows where the game itself is missing.
_REQUIRED_FEATURES = frozenset({
    "inning_number_feat",
})


@dataclass
class FeatureRunStats:
    games_processed: int = 0
    rows_read: int = 0
    rows_written: int = 0
    rows_omitted: int = 0
    elapsed_seconds: float = 0.0


def run_features(
    engine: Engine,
    game_pk: int | None = None,
) -> FeatureRunStats:
    """Build and persist feature vectors.

    If *game_pk* is given, process only that game's innings.
    Otherwise process all innings present in derived.inning_labels.

    Requirements: 5.4, 5.5, 10.2
    """
    stats = FeatureRunStats()
    t0 = time.monotonic()

    with get_connection(engine) as conn:
        innings = _get_innings_to_process(conn, game_pk)
        stats.rows_read = len(innings)

        for row in innings:
            g_pk, team_code, inning_number = row
            try:
                fv = _build_for_inning(conn, g_pk, team_code, inning_number)
            except Exception as exc:
                logger.warning(
                    "Feature build failed",
                    extra={
                        "component": "features.runner",
                        "game_pk": g_pk,
                        "team_code": team_code,
                        "inning_number": inning_number,
                        "error": str(exc),
                    },
                )
                stats.rows_omitted += 1
                continue

            if fv is None:
                stats.rows_omitted += 1
                continue

            _upsert_feature_vector(conn, fv.game_pk, team_code, inning_number, fv.values)
            stats.rows_written += 1

        stats.games_processed = len({r[0] for r in innings})

    stats.elapsed_seconds = round(time.monotonic() - t0, 1)
    logger.info(
        "Features run complete",
        extra={
            "component": "features.runner",
            "games_processed": stats.games_processed,
            "duration_seconds": stats.elapsed_seconds,
            "rows_read": stats.rows_read,
            "rows_written": stats.rows_written,
            "rows_omitted": stats.rows_omitted,
        },
    )
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_innings_to_process(
    conn: Connection,
    game_pk: int | None,
) -> list[tuple[int, str, int]]:
    """Return (game_pk, team_code, inning_number) tuples to process."""
    if game_pk is not None:
        rows = conn.execute(
            text("""
                SELECT il.game_pk, il.team_code, il.inning_number
                FROM derived.inning_labels il
                WHERE il.game_pk = :game_pk
                ORDER BY il.game_pk, il.team_code, il.inning_number
            """),
            {"game_pk": game_pk},
        ).fetchall()
    else:
        # Only process innings that don't already have features (incremental)
        rows = conn.execute(
            text("""
                SELECT il.game_pk, il.team_code, il.inning_number
                FROM derived.inning_labels il
                LEFT JOIN derived.inning_features f
                    ON f.game_pk = il.game_pk
                    AND f.team_code = il.team_code
                    AND f.inning_number = il.inning_number
                WHERE f.game_pk IS NULL
                ORDER BY il.game_pk, il.team_code, il.inning_number
            """)
        ).fetchall()
    return [(r[0], r[1].strip(), r[2]) for r in rows]


def _build_for_inning(
    conn: Connection,
    game_pk: int,
    team_code: str,
    inning_number: int,
):
    """Resolve game context and call build_feature_vector().

    Returns None if required context (game row, starter) is missing.
    Logs the missing feature name per requirement 5.4.
    """
    from sandy.schemas import FeatureVector

    # Get game context
    game_row = conn.execute(
        text("""
            SELECT game_date, home_team_code, away_team_code,
                   home_starter_id, away_starter_id
            FROM raw.games
            WHERE game_pk = :game_pk
        """),
        {"game_pk": game_pk},
    ).fetchone()

    if game_row is None:
        logger.warning(
            "Omitting inning: game not found",
            extra={
                "component": "features.runner",
                "game_pk": game_pk,
                "team_code": team_code,
                "inning_number": inning_number,
                "missing_feature": "game_row",
            },
        )
        return None

    game_date, home_code, away_code, home_starter_id, away_starter_id = game_row
    home_code = home_code.strip()
    away_code = away_code.strip()

    # Determine opposing starter
    is_home = (team_code == home_code)
    opp_team_code = away_code if is_home else home_code
    opp_starter_id = away_starter_id if is_home else home_starter_id

    if opp_starter_id is None:
        logger.warning(
            "Omitting inning: opposing starter unknown",
            extra={
                "component": "features.runner",
                "game_pk": game_pk,
                "team_code": team_code,
                "inning_number": inning_number,
                "missing_feature": "opp_starter_id",
            },
        )
        return None

    return build_feature_vector(
        conn=conn,
        team_code=team_code,
        opp_team_code=opp_team_code,
        inning_number=inning_number,
        opp_starter_id=opp_starter_id,
        game_date=game_date,
        game_pk=game_pk,
    )


def _upsert_feature_vector(
    conn: Connection,
    game_pk: int,
    team_code: str,
    inning_number: int,
    values: dict,
) -> None:
    conn.execute(
        text("""
            INSERT INTO derived.inning_features (
                game_pk, team_code, inning_number, feature_schema_version,
                opp_starter_era, opp_starter_whip, opp_starter_k9,
                opp_starter_pitches_before,
                lineup_spot_1, lineup_spot_2, lineup_spot_3,
                lineup_spot1_season_obp, lineup_spot2_season_obp, lineup_spot3_season_obp,
                is_home, ballpark_id, inning_number_feat,
                trailing15_rpg, trailing15_obp,
                prev_inning_reached_base, innings_reached_so_far, consecutive_reach_streak,
                team_season_obp, team_season_rpg
            ) VALUES (
                :game_pk, :team_code, :inning_number, :feature_schema_version,
                :opp_starter_era, :opp_starter_whip, :opp_starter_k9,
                :opp_starter_pitches_before,
                :lineup_spot_1, :lineup_spot_2, :lineup_spot_3,
                :lineup_spot1_season_obp, :lineup_spot2_season_obp, :lineup_spot3_season_obp,
                :is_home, :ballpark_id, :inning_number_feat,
                :trailing15_rpg, :trailing15_obp,
                :prev_inning_reached_base, :innings_reached_so_far, :consecutive_reach_streak,
                :team_season_obp, :team_season_rpg
            )
            ON CONFLICT (game_pk, team_code, inning_number) DO UPDATE SET
                feature_schema_version     = EXCLUDED.feature_schema_version,
                opp_starter_era            = EXCLUDED.opp_starter_era,
                opp_starter_whip           = EXCLUDED.opp_starter_whip,
                opp_starter_k9             = EXCLUDED.opp_starter_k9,
                opp_starter_pitches_before = EXCLUDED.opp_starter_pitches_before,
                lineup_spot_1              = EXCLUDED.lineup_spot_1,
                lineup_spot_2              = EXCLUDED.lineup_spot_2,
                lineup_spot_3              = EXCLUDED.lineup_spot_3,
                lineup_spot1_season_obp    = EXCLUDED.lineup_spot1_season_obp,
                lineup_spot2_season_obp    = EXCLUDED.lineup_spot2_season_obp,
                lineup_spot3_season_obp    = EXCLUDED.lineup_spot3_season_obp,
                is_home                    = EXCLUDED.is_home,
                ballpark_id                = EXCLUDED.ballpark_id,
                inning_number_feat         = EXCLUDED.inning_number_feat,
                trailing15_rpg             = EXCLUDED.trailing15_rpg,
                trailing15_obp             = EXCLUDED.trailing15_obp,
                prev_inning_reached_base   = EXCLUDED.prev_inning_reached_base,
                innings_reached_so_far     = EXCLUDED.innings_reached_so_far,
                consecutive_reach_streak   = EXCLUDED.consecutive_reach_streak,
                team_season_obp            = EXCLUDED.team_season_obp,
                team_season_rpg            = EXCLUDED.team_season_rpg,
                built_at                   = now()
        """),
        {
            "game_pk": game_pk,
            "team_code": team_code,
            "inning_number": inning_number,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            **values,
        },
    )


# ---------------------------------------------------------------------------
# Phase 1.5: Game-level feature runner
# ---------------------------------------------------------------------------


def run_game_features(
    engine: Engine,
    game_pk: int | None = None,
) -> FeatureRunStats:
    """Build and persist game-level feature vectors.

    If *game_pk* is given, process only that game.
    Otherwise process all Final regular-season games.

    Requirements: 12.3, 12.4, 12.5
    """
    from sandy.features.game_builder import build_game_feature_vector
    from sandy.features.schema import GAME_FEATURE_NAMES, GAME_FEATURE_SCHEMA_VERSION

    stats = FeatureRunStats()
    t0 = time.monotonic()

    with get_connection(engine) as conn:
        games = _get_games_for_game_features(conn, game_pk)
        stats.rows_read = len(games)

        for row in games:
            g_pk, game_date, home_code, away_code, home_starter_id, away_starter_id, venue_id = row

            # Build features for home team
            try:
                home_fv = build_game_feature_vector(
                    conn=conn,
                    game_pk=g_pk,
                    team_code=home_code,
                    opp_team_code=away_code,
                    home_starter_id=home_starter_id,
                    away_starter_id=away_starter_id,
                    game_date=game_date,
                    venue_id=venue_id,
                    is_home=True,
                )
                _upsert_game_feature_vector(conn, g_pk, home_code, home_fv.values)
                stats.rows_written += 1
            except Exception as exc:
                logger.warning(
                    "Game feature build failed (home)",
                    extra={
                        "component": "features.runner",
                        "game_pk": g_pk,
                        "team_code": home_code,
                        "error": str(exc),
                    },
                )
                stats.rows_omitted += 1

            # Build features for away team
            try:
                away_fv = build_game_feature_vector(
                    conn=conn,
                    game_pk=g_pk,
                    team_code=away_code,
                    opp_team_code=home_code,
                    home_starter_id=home_starter_id,
                    away_starter_id=away_starter_id,
                    game_date=game_date,
                    venue_id=venue_id,
                    is_home=False,
                )
                _upsert_game_feature_vector(conn, g_pk, away_code, away_fv.values)
                stats.rows_written += 1
            except Exception as exc:
                logger.warning(
                    "Game feature build failed (away)",
                    extra={
                        "component": "features.runner",
                        "game_pk": g_pk,
                        "team_code": away_code,
                        "error": str(exc),
                    },
                )
                stats.rows_omitted += 1

        stats.games_processed = len({r[0] for r in games})

    stats.elapsed_seconds = round(time.monotonic() - t0, 1)
    logger.info(
        "Game features run complete",
        extra={
            "component": "features.runner",
            "target": "game",
            "games_processed": stats.games_processed,
            "duration_seconds": stats.elapsed_seconds,
            "rows_written": stats.rows_written,
            "rows_omitted": stats.rows_omitted,
        },
    )
    return stats


def _get_games_for_game_features(
    conn: Connection,
    game_pk: int | None,
) -> list[tuple]:
    """Return game rows to process for game-level features."""
    if game_pk is not None:
        rows = conn.execute(
            text("""
                SELECT game_pk, game_date, home_team_code, away_team_code,
                       home_starter_id, away_starter_id, venue_id
                FROM raw.games
                WHERE game_pk = :game_pk AND status = 'Final' AND game_type = 'R'
                ORDER BY game_date, game_pk
            """),
            {"game_pk": game_pk},
        ).fetchall()
    else:
        # Only process games that don't already have game features (incremental)
        rows = conn.execute(
            text("""
                SELECT g.game_pk, g.game_date, g.home_team_code, g.away_team_code,
                       g.home_starter_id, g.away_starter_id, g.venue_id
                FROM raw.games g
                LEFT JOIN derived.game_features f
                    ON f.game_pk = g.game_pk AND f.team_code = g.home_team_code
                WHERE g.status = 'Final' AND g.game_type = 'R'
                  AND f.game_pk IS NULL
                ORDER BY g.game_date, g.game_pk
            """)
        ).fetchall()
    return [
        (r[0], r[1], r[2].strip(), r[3].strip(), r[4], r[5], r[6])
        for r in rows
    ]


def _upsert_game_feature_vector(
    conn: Connection,
    game_pk: int,
    team_code: str,
    values: dict,
) -> None:
    """UPSERT a game-level feature vector into derived.game_features."""
    from sandy.features.schema import GAME_FEATURE_SCHEMA_VERSION

    conn.execute(
        text("""
            INSERT INTO derived.game_features (
                game_pk, team_code, feature_schema_version,
                home_starter_era, home_starter_whip,
                away_starter_era, away_starter_whip,
                home_trailing15_rpg, away_trailing15_rpg,
                home_season_obp, away_season_obp,
                ballpark_id, is_home
            ) VALUES (
                :game_pk, :team_code, :feature_schema_version,
                :home_starter_era, :home_starter_whip,
                :away_starter_era, :away_starter_whip,
                :home_trailing15_rpg, :away_trailing15_rpg,
                :home_season_obp, :away_season_obp,
                :ballpark_id, :is_home
            )
            ON CONFLICT (game_pk, team_code) DO UPDATE SET
                feature_schema_version = EXCLUDED.feature_schema_version,
                home_starter_era       = EXCLUDED.home_starter_era,
                home_starter_whip      = EXCLUDED.home_starter_whip,
                away_starter_era       = EXCLUDED.away_starter_era,
                away_starter_whip      = EXCLUDED.away_starter_whip,
                home_trailing15_rpg    = EXCLUDED.home_trailing15_rpg,
                away_trailing15_rpg    = EXCLUDED.away_trailing15_rpg,
                home_season_obp        = EXCLUDED.home_season_obp,
                away_season_obp        = EXCLUDED.away_season_obp,
                ballpark_id            = EXCLUDED.ballpark_id,
                is_home                = EXCLUDED.is_home,
                computed_at            = now()
        """),
        {
            "game_pk": game_pk,
            "team_code": team_code,
            "feature_schema_version": GAME_FEATURE_SCHEMA_VERSION,
            "home_starter_era": values.get("home_starter_era", 0.0),
            "home_starter_whip": values.get("home_starter_whip", 0.0),
            "away_starter_era": values.get("away_starter_era", 0.0),
            "away_starter_whip": values.get("away_starter_whip", 0.0),
            "home_trailing15_rpg": values.get("home_trailing15_rpg", 0.0),
            "away_trailing15_rpg": values.get("away_trailing15_rpg", 0.0),
            "home_season_obp": values.get("home_season_obp", 0.0),
            "away_season_obp": values.get("away_season_obp", 0.0),
            "ballpark_id": values.get("ballpark_id", 0),
            "is_home": bool(values.get("is_home", False)),
        },
    )


__all__ = ["FeatureRunStats", "run_features", "run_game_features"]
