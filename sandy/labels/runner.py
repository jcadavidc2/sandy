"""Labels batch runner — persists inning labels for all Final games.

Task 7.2: iterates all Final games, calls generate_labels_for_game() for
each, and UPSERTs results into derived.inning_labels.

Idempotent via ON CONFLICT DO UPDATE (requirement 4.4).
Emits a final JSON log line with duration_seconds, rows_read, rows_written
(requirement 10.2).

Requirements: 3.4, 4.4, 10.1, 10.2
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from sandy.db import get_connection
from sandy.labels.generator import generate_labels_for_game
from sandy.logging import get_logger

logger = get_logger("labels.runner")


@dataclass
class LabelRunStats:
    games_processed: int = 0
    rows_read: int = 0
    rows_written: int = 0
    elapsed_seconds: float = 0.0


def run_labels(
    engine: Engine,
    game_pk: int | None = None,
) -> LabelRunStats:
    """Generate and persist inning labels.

    If *game_pk* is given, process only that game.
    Otherwise process all Final games in raw.games.

    Requirements: 4.1, 4.4, 10.2
    """
    stats = LabelRunStats()
    t0 = time.monotonic()

    with get_connection(engine) as conn:
        if game_pk is not None:
            game_pks = [game_pk]
        else:
            game_pks = _get_final_game_pks(conn)

        stats.rows_read = len(game_pks)

        for pk in game_pks:
            labels = generate_labels_for_game(conn, pk)
            for label in labels:
                conn.execute(
                    text("""
                        INSERT INTO derived.inning_labels
                            (game_pk, team_code, inning_number, reached_base)
                        VALUES
                            (:game_pk, :team_code, :inning_number, :reached_base)
                        ON CONFLICT (game_pk, team_code, inning_number)
                        DO UPDATE SET
                            reached_base = EXCLUDED.reached_base,
                            labeled_at   = now()
                    """),
                    {
                        "game_pk": label.game_pk,
                        "team_code": label.team_code,
                        "inning_number": label.inning_number,
                        "reached_base": label.reached_base,
                    },
                )
                stats.rows_written += 1
            stats.games_processed += 1

    stats.elapsed_seconds = round(time.monotonic() - t0, 1)
    logger.info(
        "Labels run complete",
        extra={
            "component": "labels.runner",
            "games_processed": stats.games_processed,
            "duration_seconds": stats.elapsed_seconds,
            "rows_read": stats.rows_read,
            "rows_written": stats.rows_written,
        },
    )
    return stats


def _get_final_game_pks(conn: Connection) -> list[int]:
    rows = conn.execute(
        text("SELECT game_pk FROM raw.games WHERE status = 'Final' ORDER BY game_pk")
    ).fetchall()
    return [r[0] for r in rows]


__all__ = ["LabelRunStats", "run_labels"]
