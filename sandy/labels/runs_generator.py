"""Runs label generator — pure function.

Phase 1.5, Task 5.1: Reads raw.games for a game_pk and produces two
RunsLabel objects (one per team) for Final regular-season games.

Requirements: 2.1, 2.2, 2.3, 2.4
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.schemas import RunsLabel


def generate_runs_labels(
    conn: Connection,
    game_pk: int,
) -> list[RunsLabel]:
    """Return two RunsLabel objects for a Final regular-season game, or [].

    Returns [] if:
    - Game not found
    - status != 'Final'
    - game_type != 'R' (not regular season)
    """
    row = conn.execute(
        text("""
            SELECT status, game_type, home_team_code, away_team_code,
                   home_score, away_score
            FROM raw.games
            WHERE game_pk = :game_pk
        """),
        {"game_pk": game_pk},
    ).fetchone()

    if row is None:
        return []

    status, game_type, home_code, away_code, home_score, away_score = row

    if status != "Final":
        return []
    if game_type != "R":
        return []
    if home_score is None or away_score is None:
        return []

    return [
        RunsLabel(game_pk=game_pk, team_code=home_code.strip(), runs=int(home_score)),
        RunsLabel(game_pk=game_pk, team_code=away_code.strip(), runs=int(away_score)),
    ]


__all__ = ["generate_runs_labels"]
