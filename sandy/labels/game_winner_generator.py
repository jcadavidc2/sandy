"""Game winner label generator — pure function.

Phase 1.5, Task 4.1: Reads raw.games for a game_pk and produces a
GameWinnerLabel if the game is Final, regular-season, and not tied.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.schemas import GameWinnerLabel


def generate_game_winner_label(
    conn: Connection,
    game_pk: int,
) -> GameWinnerLabel | None:
    """Return a GameWinnerLabel for a Final regular-season game, or None.

    Returns None if:
    - Game not found
    - status != 'Final'
    - game_type != 'R' (not regular season)
    - home_score == away_score (tie/suspended)
    """
    row = conn.execute(
        text("""
            SELECT status, game_type, home_score, away_score
            FROM raw.games
            WHERE game_pk = :game_pk
        """),
        {"game_pk": game_pk},
    ).fetchone()

    if row is None:
        return None

    status, game_type, home_score, away_score = row

    if status != "Final":
        return None
    if game_type != "R":
        return None
    if home_score is None or away_score is None:
        return None
    if home_score == away_score:
        return None

    return GameWinnerLabel(
        game_pk=game_pk,
        home_team_wins=(home_score > away_score),
    )


__all__ = ["generate_game_winner_label"]
