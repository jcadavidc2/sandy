"""Property-based test: label monotonicity (Property 2).

Property: for any synthetic play-by-play where at least one reach-base event
exists for a (game_pk, team_code, inning), generate_labels_for_game()
produces reached_base = True for that row.

Uses Hypothesis to generate arbitrary lists of play rows with at least one
reach-base event injected per target inning, runs the generator against the
test DB, and asserts reached_base is True.

Validates: Requirements 4.2, 4.5, 11.3
"""
from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sqlalchemy import text

from sandy.labels.generator import REACHES_BASE_EVENT_CODES, generate_labels_for_game

# Non-reach-base event codes for generating "noise" plays
NON_REACH_CODES = ["strikeout", "field_out", "flyout", "groundout", "pop_out"]
REACH_CODES = sorted(REACHES_BASE_EVENT_CODES)
TEAM_CODES = ["SEA", "LAD"]


# ---------------------------------------------------------------------------
# DB seeding helpers
# ---------------------------------------------------------------------------

def _seed_prerequisites(conn, game_pk: int, home: str, away: str) -> None:
    """Insert the minimum rows needed for FK constraints."""
    for i, code in enumerate([home, away]):
        conn.execute(
            text("""
                INSERT INTO raw.teams (team_code, team_id, name)
                VALUES (:code, :tid, :name)
                ON CONFLICT (team_code) DO UPDATE SET
                    team_id = EXCLUDED.team_id, name = EXCLUDED.name
            """),
            {"code": code, "tid": abs(hash(code)) % 90000 + 10000 + i, "name": f"{code} Team"},
        )

    conn.execute(
        text("""
            INSERT INTO raw.games (
                game_pk, game_date, season, game_type, status,
                home_team_code, away_team_code, raw_payload_hash
            ) VALUES (
                :game_pk, :game_date, :season, :game_type, :status,
                :home, :away, :hash
            )
            ON CONFLICT (game_pk) DO UPDATE SET status = EXCLUDED.status
        """),
        {
            "game_pk": game_pk,
            "game_date": date(2023, 4, 15),
            "season": 2023,
            "game_type": "R",
            "status": "Final",
            "home": home,
            "away": away,
            "hash": "testhash",
        },
    )


def _insert_play(conn, game_pk: int, at_bat_index: int, inning: int,
                 half: str, batting: str, pitching: str,
                 event_code: str) -> None:
    conn.execute(
        text("""
            INSERT INTO raw.plays (
                game_pk, at_bat_index, inning, half_inning,
                batting_team_code, pitching_team_code,
                event_type, event_code, is_reaches_base,
                pitches_in_pa, raw
            ) VALUES (
                :game_pk, :at_bat_index, :inning, :half,
                :batting, :pitching,
                :event_type, :event_code, :is_reaches_base,
                0, '{}'::jsonb
            )
            ON CONFLICT (game_pk, at_bat_index) DO NOTHING
        """),
        {
            "game_pk": game_pk,
            "at_bat_index": at_bat_index,
            "inning": inning,
            "half": half,
            "batting": batting,
            "pitching": pitching,
            "event_type": "hit" if event_code in REACHES_BASE_EVENT_CODES else "out",
            "event_code": event_code,
            "is_reaches_base": event_code in REACHES_BASE_EVENT_CODES,
        },
    )


# ---------------------------------------------------------------------------
# The property test
# ---------------------------------------------------------------------------

@given(
    target_inning=st.integers(min_value=1, max_value=9),
    reach_code=st.sampled_from(REACH_CODES),
    noise_codes=st.lists(
        st.sampled_from(NON_REACH_CODES), min_size=0, max_size=5
    ),
    noise_before=st.booleans(),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_label_monotonicity(
    target_inning, reach_code, noise_codes, noise_before, clean_db
):
    """If at least one reach-base event exists in an inning, reached_base must be True."""
    engine = clean_db
    game_pk = 999001
    home, away = "SEA", "LAD"
    batting_team = away  # top of inning = away team bats

    with engine.begin() as conn:
        _seed_prerequisites(conn, game_pk, home, away)

        at_bat = 0
        # Optionally insert noise plays BEFORE the reach-base event
        if noise_before:
            for code in noise_codes:
                _insert_play(conn, game_pk, at_bat, target_inning,
                             "top", batting_team, home, code)
                at_bat += 1

        # Insert the guaranteed reach-base event
        _insert_play(conn, game_pk, at_bat, target_inning,
                     "top", batting_team, home, reach_code)
        at_bat += 1

        # Optionally insert noise plays AFTER the reach-base event
        if not noise_before:
            for code in noise_codes:
                _insert_play(conn, game_pk, at_bat, target_inning,
                             "top", batting_team, home, code)
                at_bat += 1

        # Run the generator
        labels = generate_labels_for_game(conn, game_pk)

    # Find the label for our target inning
    target_labels = [
        lbl for lbl in labels
        if lbl.team_code == batting_team and lbl.inning_number == target_inning
    ]

    assert len(target_labels) == 1, (
        f"Expected exactly 1 label for ({batting_team}, inning {target_inning}), "
        f"got {len(target_labels)}. All labels: {labels}"
    )
    assert target_labels[0].reached_base is True, (
        f"reached_base should be True when {reach_code!r} is in the inning. "
        f"Label: {target_labels[0]}"
    )
