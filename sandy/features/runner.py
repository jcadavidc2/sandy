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
        rows = conn.execute(
            text("""
                SELECT il.game_pk, il.team_code, il.inning_number
                FROM derived.inning_labels il
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
                is_home, ballpark_id, inning_number_feat,
                trailing15_rpg, trailing15_obp
            ) VALUES (
                :game_pk, :team_code, :inning_number, :feature_schema_version,
                :opp_starter_era, :opp_starter_whip, :opp_starter_k9,
                :opp_starter_pitches_before,
                :lineup_spot_1, :lineup_spot_2, :lineup_spot_3,
                :is_home, :ballpark_id, :inning_number_feat,
                :trailing15_rpg, :trailing15_obp
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
                is_home                    = EXCLUDED.is_home,
                ballpark_id                = EXCLUDED.ballpark_id,
                inning_number_feat         = EXCLUDED.inning_number_feat,
                trailing15_rpg             = EXCLUDED.trailing15_rpg,
                trailing15_obp             = EXCLUDED.trailing15_obp,
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


__all__ = ["FeatureRunStats", "run_features"]
