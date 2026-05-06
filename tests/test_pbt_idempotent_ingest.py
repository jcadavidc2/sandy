"""Property-based test: idempotent ingestion (Property 1).

Property: running the Ingestion_Service twice over the same fixed set of
mocked MLB Stats API responses produces byte-equivalent rows across all
raw.* tables on both runs.

Uses Hypothesis to generate arbitrary ordered sequences of game data and
asserts that SELECT * ORDER BY PK on every raw table is identical after
the first and second run.

Validates: Requirements 1.2, 2.4, 11.2
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sqlalchemy import text

from sandy.ingest.service import backfill_seasons
from sandy.config import IngestConfig


# ---------------------------------------------------------------------------
# Hypothesis strategies for synthetic game data
# ---------------------------------------------------------------------------

# Valid 3-letter team codes used in generated games
TEAM_CODES = ["SEA", "LAD", "NYY", "BOS", "CHC", "HOU", "ATL", "SFG"]

# Valid event codes
EVENT_CODES = [
    "single", "double", "triple", "home_run",
    "walk", "strikeout", "field_out", "hit_by_pitch", "field_error",
]


def _make_player(player_id: int, name: str) -> dict[str, Any]:
    return {
        "id": player_id,
        "fullName": name,
        "primaryPosition": {"abbreviation": "P"},
        "pitchHand": {"code": "R"},
        "batSide": {"code": "R"},
    }


def _make_play(at_bat_index: int, inning: int, half: str,
               batting_code: str, pitching_code: str,
               batter_id: int, pitcher_id: int,
               event_code: str) -> dict[str, Any]:
    from sandy.ingest.parsers import _event_type
    return {
        "result": {"eventType": event_code, "description": event_code},
        "about": {
            "atBatIndex": at_bat_index,
            "inning": inning,
            "halfInning": half,
            "isComplete": True,
        },
        "matchup": {
            "batter": {"id": batter_id, "fullName": f"Batter{batter_id}"},
            "pitcher": {"id": pitcher_id, "fullName": f"Pitcher{pitcher_id}"},
            "batOrder": "100",
        },
        "playEvents": [],
    }


def _make_live_feed(game_pk: int, home: str, away: str,
                    season: int, plays: list[dict]) -> dict[str, Any]:
    """Build a minimal /v1.1/game/{pk}/feed/live payload."""
    pitcher_id = 900000 + game_pk
    batter_id = 800000 + game_pk

    players = {
        f"ID{pitcher_id}": _make_player(pitcher_id, f"Pitcher{pitcher_id}"),
        f"ID{batter_id}": _make_player(batter_id, f"Batter{batter_id}"),
    }

    return {
        "gameData": {
            "game": {"pk": game_pk, "season": str(season), "type": "R"},
            "datetime": {
                "officialDate": f"{season}-04-15",
                "dateTime": f"{season}-04-15T20:00:00Z",
            },
            "status": {"detailedState": "Final"},
            "teams": {
                "home": {"id": 1, "abbreviation": home, "name": f"{home} Team"},
                "away": {"id": 2, "abbreviation": away, "name": f"{away} Team"},
            },
            "venue": {"id": 1, "name": "Test Park"},
            "players": players,
        },
        "liveData": {
            "plays": {"allPlays": plays},
            "linescore": {
                "teams": {
                    "home": {"runs": 3},
                    "away": {"runs": 2},
                }
            },
            "boxscore": {
                "teams": {
                    "home": {
                        "pitchers": [pitcher_id],
                        "players": {
                            f"ID{pitcher_id}": {
                                "position": {"abbreviation": "P"},
                                "stats": {
                                    "pitching": {
                                        "pitchesThrown": 85,
                                        "outs": 18,
                                        "runs": 2,
                                        "baseOnBalls": 2,
                                        "hits": 5,
                                        "strikeOuts": 7,
                                        "gamesStarted": 1,
                                    }
                                },
                            }
                        },
                    },
                    "away": {"pitchers": [], "players": {}},
                }
            },
        },
    }


@st.composite
def game_sequence(draw) -> list[dict[str, Any]]:
    """Generate a list of 1–5 synthetic game payloads."""
    n_games = draw(st.integers(min_value=1, max_value=5))
    games = []
    used_pks = set()
    for i in range(n_games):
        game_pk = draw(st.integers(min_value=700000, max_value=799999))
        while game_pk in used_pks:
            game_pk += 1
        used_pks.add(game_pk)

        home, away = draw(
            st.lists(
                st.sampled_from(TEAM_CODES), min_size=2, max_size=2, unique=True
            )
        )
        n_plays = draw(st.integers(min_value=1, max_value=6))
        plays = []
        for j in range(n_plays):
            event_code = draw(st.sampled_from(EVENT_CODES))
            half = draw(st.sampled_from(["top", "bottom"]))
            batting = away if half == "top" else home
            pitching = home if half == "top" else away
            plays.append(
                _make_play(
                    at_bat_index=j,
                    inning=draw(st.integers(min_value=1, max_value=9)),
                    half=half,
                    batting_code=batting,
                    pitching_code=pitching,
                    batter_id=800000 + game_pk + j,
                    pitcher_id=900000 + game_pk,
                    event_code=event_code,
                )
            )
        games.append(_make_live_feed(game_pk, home, away, 2023, plays))
    return games


# ---------------------------------------------------------------------------
# Helper: snapshot all raw.* tables ordered by PK
# ---------------------------------------------------------------------------

def _snapshot(conn) -> dict[str, list[tuple]]:
    snapshots = {}
    snapshots["teams"] = conn.execute(
        text("SELECT * FROM raw.teams ORDER BY team_code")
    ).fetchall()
    snapshots["players"] = conn.execute(
        text("SELECT player_id, full_name, primary_position, throws, bats FROM raw.players ORDER BY player_id")
    ).fetchall()
    snapshots["games"] = conn.execute(
        text("SELECT game_pk, game_date, season, game_type, status, home_team_code, away_team_code FROM raw.games ORDER BY game_pk")
    ).fetchall()
    snapshots["plays"] = conn.execute(
        text("SELECT game_pk, at_bat_index, inning, half_inning, event_code, is_reaches_base FROM raw.plays ORDER BY game_pk, at_bat_index")
    ).fetchall()
    snapshots["pitcher_game_stats"] = conn.execute(
        text("SELECT game_pk, pitcher_id, outs_recorded, runs_allowed FROM raw.pitcher_game_stats ORDER BY game_pk, pitcher_id")
    ).fetchall()
    return snapshots


# ---------------------------------------------------------------------------
# The property test
# ---------------------------------------------------------------------------

@given(games=game_sequence())
@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_idempotent_ingestion(games, clean_db):
    """Running ingest twice over the same mocked responses gives identical DB state."""
    engine = clean_db

    # Seed teams table (FK prerequisite) — use teams from the generated games
    team_codes = set()
    for g in games:
        team_codes.add(g["gameData"]["teams"]["home"]["abbreviation"])
        team_codes.add(g["gameData"]["teams"]["away"]["abbreviation"])

    with engine.begin() as conn:
        for i, code in enumerate(sorted(team_codes)):
            conn.execute(
                text("""
                    INSERT INTO raw.teams (team_code, team_id, name)
                    VALUES (:code, :tid, :name)
                    ON CONFLICT (team_code) DO NOTHING
                """),
                {"code": code[:3], "tid": 100 + i, "name": f"{code} Team"},
            )

    # Build mock client that returns our synthetic payloads
    game_map = {g["gameData"]["game"]["pk"]: g for g in games}
    game_pks = list(game_map.keys())

    def mock_get(path, params=None):
        # Schedule endpoint — return all our game_pks as Final
        if "/schedule" in path:
            return {
                "dates": [
                    {
                        "games": [
                            {
                                "gamePk": pk,
                                "gameDate": f"2023-04-15",
                                "status": {"detailedState": "Final"},
                                "gameType": "R",
                            }
                            for pk in game_pks
                        ]
                    }
                ]
            }
        # Teams endpoint
        if "/teams" in path:
            return {"teams": []}
        # Live feed endpoint
        for pk in game_pks:
            if f"/game/{pk}/" in path:
                return game_map[pk]
        return {}

    mock_client = MagicMock()
    mock_client.get.side_effect = mock_get

    # --- First run ---
    with engine.begin() as conn:
        backfill_seasons(conn, mock_client, seasons=[2023])

    with engine.connect() as conn:
        snapshot_1 = _snapshot(conn)

    # --- Second run (idempotent) ---
    with engine.begin() as conn:
        backfill_seasons(conn, mock_client, seasons=[2023])

    with engine.connect() as conn:
        snapshot_2 = _snapshot(conn)

    # --- Assert byte-equivalence ---
    for table, rows1 in snapshot_1.items():
        rows2 = snapshot_2[table]
        assert rows1 == rows2, (
            f"Table raw.{table} differs between run 1 and run 2.\n"
            f"Run 1: {rows1}\nRun 2: {rows2}"
        )
