"""Label generator — pure function over DB state.

Task 7.1: generate_labels_for_game() reads raw.plays for a game_pk and
produces one InningLabel per (batting_team_code, inning_number) group.

Pure: takes a DB connection, returns a list. No side effects, no writes.
The runner (task 7.2) handles persistence.

Requirements: 4.1, 4.2, 4.3, 4.6
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.schemas import InningLabel

# Event codes that count as "reaches base" — must match parsers.py
REACHES_BASE_EVENT_CODES: frozenset[str] = frozenset(
    {
        "single",
        "double",
        "triple",
        "home_run",
        "walk",
        "hit_by_pitch",
        "field_error",
    }
)


def generate_labels_for_game(
    conn: Connection,
    game_pk: int,
) -> list[InningLabel]:
    """Return one InningLabel per (team_code, inning_number) for a Final game.

    Steps:
    1. Check raw.games.status — return [] if not 'Final' (requirement 4.6).
    2. SELECT plays for this game_pk.
    3. Group by (batting_team_code, inning).
    4. reached_base = BOOL_OR(is_reaches_base) over the group.

    Groups that don't exist (e.g. bottom of 9th not played) are simply not
    emitted — no false zeros for innings that never happened.

    Idempotent: pure read, same inputs → same outputs (requirement 4.4).
    Monotonic: OR of booleans can only go false→true as events are added
    (requirement 4.5).
    """
    # Step 1: check game status
    status_row = conn.execute(
        text("SELECT status FROM raw.games WHERE game_pk = :game_pk"),
        {"game_pk": game_pk},
    ).fetchone()

    if status_row is None or status_row[0] != "Final":
        return []

    # Step 2: fetch plays
    rows = conn.execute(
        text("""
            SELECT batting_team_code, inning, is_reaches_base
            FROM raw.plays
            WHERE game_pk = :game_pk
            ORDER BY inning, at_bat_index
        """),
        {"game_pk": game_pk},
    ).fetchall()

    # Step 3 & 4: group and OR
    # key: (team_code, inning_number) → bool (reached_base so far)
    groups: dict[tuple[str, int], bool] = defaultdict(bool)

    for team_code, inning, is_reaches_base in rows:
        key = (team_code.strip(), int(inning))
        groups[key] = groups[key] or bool(is_reaches_base)

    return [
        InningLabel(
            game_pk=game_pk,
            team_code=team_code,
            inning_number=inning_number,
            reached_base=reached_base,
        )
        for (team_code, inning_number), reached_base in sorted(groups.items())
    ]


__all__ = ["REACHES_BASE_EVENT_CODES", "generate_labels_for_game"]
