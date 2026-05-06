"""Pure parsers for MLB Stats API responses.

Task 5.2: Convert raw JSON from the four endpoints we use into TypedDict row
types defined in sandy.schemas. All functions are pure (no I/O, no DB) so
they are trivially testable and reusable.

Endpoints covered:
  /v1/schedule                        -> list[ScheduleEntry]
  /v1.1/game/{pk}/feed/live           -> ParsedGame
  /v1/teams                           -> list[TeamRow]
  /v1/people/{id}?hydrate=stats(...)  -> PlayerRow (+ optional season stats)

Requirements: 1.1, 4.2
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sandy.schemas import (
    GameRow,
    IngestFailureRow,
    PitcherGameStatRow,
    PlayRow,
    PlayerRow,
    TeamRow,
)

# ---------------------------------------------------------------------------
# Event-code sets
# ---------------------------------------------------------------------------

#: Plate-appearance outcomes that count as "reaches base" for label generation.
#: Requirement 4.2 / design §Label Generator.
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

# Grouped event types (coarser than event_code, stored in plays.event_type)
_HIT_CODES = frozenset({"single", "double", "triple", "home_run"})
_OUT_CODES = frozenset(
    {
        "strikeout",
        "field_out",
        "grounded_into_double_play",
        "double_play",
        "triple_play",
        "force_out",
        "fielders_choice_out",
        "fielders_choice",
        "sac_fly",
        "sac_bunt",
        "bunt_groundout",
        "bunt_lineout",
        "bunt_popout",
        "lineout",
        "pop_out",
        "flyout",
        "groundout",
    }
)


def _event_type(event_code: str) -> str:
    """Map a raw event_code to a coarser event_type bucket."""
    if event_code in _HIT_CODES:
        return "hit"
    if event_code == "walk":
        return "walk"
    if event_code == "hit_by_pitch":
        return "hit_by_pitch"
    if event_code == "field_error":
        return "error"
    if event_code == "strikeout":
        return "strikeout"
    if event_code in _OUT_CODES:
        return "out"
    return "other"


# ---------------------------------------------------------------------------
# Schedule parser  (/v1/schedule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleEntry:
    """Minimal game descriptor returned by the schedule endpoint."""
    game_pk: int
    game_date: date
    status: str          # 'Final', 'In Progress', 'Scheduled', ...
    game_type: str       # 'R', 'P', 'S', ...


def parse_schedule(payload: dict[str, Any]) -> list[ScheduleEntry]:
    """Parse a /v1/schedule response into a flat list of ScheduleEntry.

    The schedule endpoint returns a nested structure:
      payload["dates"][i]["games"][j]

    We flatten it and return one entry per game.
    """
    entries: list[ScheduleEntry] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            try:
                entries.append(
                    ScheduleEntry(
                        game_pk=int(game["gamePk"]),
                        game_date=date.fromisoformat(game["gameDate"][:10]),
                        status=game.get("status", {}).get("detailedState", "Unknown"),
                        game_type=game.get("gameType", "R"),
                    )
                )
            except (KeyError, ValueError):
                # Malformed entry — skip silently; the service layer logs failures
                continue
    return entries


# ---------------------------------------------------------------------------
# Teams parser  (/v1/teams)
# ---------------------------------------------------------------------------


def parse_teams(payload: dict[str, Any]) -> list[TeamRow]:
    """Parse a /v1/teams response into TeamRow dicts."""
    rows: list[TeamRow] = []
    for team in payload.get("teams", []):
        try:
            rows.append(
                TeamRow(
                    team_code=team["abbreviation"].upper()[:3],
                    team_id=int(team["id"]),
                    name=team["name"],
                    venue_id=team.get("venue", {}).get("id"),
                    league=team.get("league", {}).get("name"),
                    division=team.get("division", {}).get("name"),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Player parser  (/v1/people/{id})
# ---------------------------------------------------------------------------


def parse_player(payload: dict[str, Any]) -> PlayerRow | None:
    """Parse a /v1/people/{id} response into a PlayerRow.

    Returns None if the payload is malformed or missing the person record.
    """
    people = payload.get("people", [])
    if not people:
        return None
    p = people[0]
    try:
        return PlayerRow(
            player_id=int(p["id"]),
            full_name=p["fullName"],
            primary_position=p.get("primaryPosition", {}).get("abbreviation"),
            throws=p.get("pitchHand", {}).get("code"),
            bats=p.get("batSide", {}).get("code"),
        )
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Live feed parser  (/v1.1/game/{pk}/feed/live)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedGame:
    """All rows extracted from a single game's live feed."""
    game: GameRow
    plays: list[PlayRow]
    players: list[PlayerRow]
    pitcher_stats: list[PitcherGameStatRow]


def parse_live_feed(game_pk: int, payload: dict[str, Any]) -> ParsedGame:
    """Parse a /v1.1/game/{pk}/feed/live response into typed rows.

    The raw payload hash (sha256 of the full JSON) is stored on GameRow so
    we can detect upstream changes without re-parsing.
    """
    raw_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

    game_data = payload.get("gameData", {})
    live_data = payload.get("liveData", {})

    game_row = _parse_game_row(game_pk, game_data, live_data, raw_hash)
    players = _parse_players(game_data)
    plays = _parse_plays(game_pk, game_data, live_data)
    pitcher_stats = _parse_pitcher_stats(game_pk, live_data, game_data)

    return ParsedGame(
        game=game_row,
        plays=plays,
        players=players,
        pitcher_stats=pitcher_stats,
    )


# ---------------------------------------------------------------------------
# Internal helpers for live feed
# ---------------------------------------------------------------------------


def _parse_game_row(
    game_pk: int,
    game_data: dict[str, Any],
    live_data: dict[str, Any],
    raw_hash: str,
) -> GameRow:
    teams = game_data.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    datetime_info = game_data.get("datetime", {})
    status = game_data.get("status", {})
    venue = game_data.get("venue", {})

    # Scores from linescore
    linescore = live_data.get("linescore", {})
    home_score = linescore.get("teams", {}).get("home", {}).get("runs")
    away_score = linescore.get("teams", {}).get("away", {}).get("runs")

    # Starters from boxscore
    boxscore = live_data.get("boxscore", {})
    home_starter_id = _find_starter_id(boxscore, "home")
    away_starter_id = _find_starter_id(boxscore, "away")

    # first_pitch_utc
    first_pitch_utc: datetime | None = None
    dt_str = datetime_info.get("dateTime")
    if dt_str:
        try:
            first_pitch_utc = datetime.fromisoformat(
                dt_str.replace("Z", "+00:00")
            )
        except ValueError:
            pass

    return GameRow(
        game_pk=game_pk,
        game_date=date.fromisoformat(
            game_data.get("datetime", {}).get("officialDate", "1900-01-01")
        ),
        season=int(game_data.get("game", {}).get("season", 0)),
        game_type=game_data.get("game", {}).get("type", "R"),
        status=status.get("detailedState", "Unknown"),
        home_team_code=home.get("abbreviation", "UNK").upper()[:3],
        away_team_code=away.get("abbreviation", "UNK").upper()[:3],
        venue_id=venue.get("id"),
        first_pitch_utc=first_pitch_utc,
        home_score=int(home_score) if home_score is not None else None,
        away_score=int(away_score) if away_score is not None else None,
        home_starter_id=home_starter_id,
        away_starter_id=away_starter_id,
        raw_payload_hash=raw_hash,
    )


def _find_starter_id(boxscore: dict[str, Any], side: str) -> int | None:
    """Return the player_id of the starting pitcher for home or away."""
    team_box = boxscore.get("teams", {}).get(side, {})
    pitchers = team_box.get("pitchers", [])
    if not pitchers:
        return None
    # The first pitcher listed is the starter
    player_id = pitchers[0]
    players_info = team_box.get("players", {})
    key = f"ID{player_id}"
    player_data = players_info.get(key, {})
    stats = player_data.get("stats", {}).get("pitching", {})
    # Confirm they actually started (gameStarted flag or position)
    position = player_data.get("position", {}).get("abbreviation", "")
    if position == "P" or stats.get("gamesStarted", 0) >= 1:
        return int(player_id)
    return int(player_id)  # fallback: trust the first pitcher listed


def _parse_players(game_data: dict[str, Any]) -> list[PlayerRow]:
    """Extract all players mentioned in gameData.players."""
    rows: list[PlayerRow] = []
    for key, p in game_data.get("players", {}).items():
        try:
            rows.append(
                PlayerRow(
                    player_id=int(p["id"]),
                    full_name=p["fullName"],
                    primary_position=p.get("primaryPosition", {}).get("abbreviation"),
                    throws=p.get("pitchHand", {}).get("code"),
                    bats=p.get("batSide", {}).get("code"),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


def _parse_plays(
    game_pk: int,
    game_data: dict[str, Any],
    live_data: dict[str, Any],
) -> list[PlayRow]:
    """Parse allPlays from liveData into PlayRow dicts."""
    teams = game_data.get("teams", {})
    home_code = teams.get("home", {}).get("abbreviation", "UNK").upper()[:3]
    away_code = teams.get("away", {}).get("abbreviation", "UNK").upper()[:3]

    rows: list[PlayRow] = []
    for play in live_data.get("plays", {}).get("allPlays", []):
        try:
            row = _parse_single_play(game_pk, play, home_code, away_code)
            if row is not None:
                rows.append(row)
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def _parse_single_play(
    game_pk: int,
    play: dict[str, Any],
    home_code: str,
    away_code: str,
) -> PlayRow | None:
    """Parse one atBat object into a PlayRow. Returns None for incomplete plays."""
    result = play.get("result", {})
    about = play.get("about", {})
    matchup = play.get("matchup", {})

    event_code_raw = result.get("eventType", "")
    if not event_code_raw:
        return None  # incomplete / no result yet

    event_code = event_code_raw.lower().replace(" ", "_")
    is_top = about.get("halfInning", "top") == "top"
    half_inning = "top" if is_top else "bottom"
    batting_team = away_code if is_top else home_code
    pitching_team = home_code if is_top else away_code

    # Timestamps from playEvents (first and last pitch)
    play_events = play.get("playEvents", [])
    start_time_utc: datetime | None = None
    end_time_utc: datetime | None = None
    pitches_in_pa = 0
    for evt in play_events:
        if evt.get("isPitch"):
            pitches_in_pa += 1
            ts = evt.get("startTime") or evt.get("endTime")
            if ts and start_time_utc is None:
                try:
                    start_time_utc = datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
    if play_events:
        last_ts = play_events[-1].get("endTime") or play_events[-1].get("startTime")
        if last_ts:
            try:
                end_time_utc = datetime.fromisoformat(
                    last_ts.replace("Z", "+00:00")
                )
            except ValueError:
                pass

    return PlayRow(
        game_pk=game_pk,
        at_bat_index=int(about.get("atBatIndex", 0)),
        inning=int(about.get("inning", 1)),
        half_inning=half_inning,
        batting_team_code=batting_team,
        pitching_team_code=pitching_team,
        batter_id=_player_id(matchup.get("batter")),
        pitcher_id=_player_id(matchup.get("pitcher")),
        batting_order=_batting_order(matchup),
        event_type=_event_type(event_code),
        event_code=event_code,
        is_reaches_base=event_code in REACHES_BASE_EVENT_CODES,
        pitches_in_pa=pitches_in_pa,
        start_time_utc=start_time_utc,
        end_time_utc=end_time_utc,
        raw=play,
    )


def _player_id(player: dict[str, Any] | None) -> int | None:
    if not player:
        return None
    try:
        return int(player["id"])
    except (KeyError, ValueError, TypeError):
        return None


def _batting_order(matchup: dict[str, Any]) -> int | None:
    order = matchup.get("batterHotColdZoneStats") or matchup.get("postOnFirst")
    # batting order is in matchup.batOrder as a string like "100" (1st), "200" (2nd)
    bat_order = matchup.get("batOrder")
    if bat_order is not None:
        try:
            return int(str(bat_order)) // 100
        except (ValueError, TypeError):
            pass
    return None


def _parse_pitcher_stats(
    game_pk: int,
    live_data: dict[str, Any],
    game_data: dict[str, Any],
) -> list[PitcherGameStatRow]:
    """Extract per-pitcher game stats from the boxscore."""
    rows: list[PitcherGameStatRow] = []
    teams = game_data.get("teams", {})
    boxscore = live_data.get("boxscore", {})

    for side in ("home", "away"):
        team_code = teams.get(side, {}).get("abbreviation", "UNK").upper()[:3]
        team_box = boxscore.get("teams", {}).get(side, {})
        pitcher_ids = team_box.get("pitchers", [])
        players_info = team_box.get("players", {})

        for i, pid in enumerate(pitcher_ids):
            key = f"ID{pid}"
            player_data = players_info.get(key, {})
            stats = player_data.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            try:
                rows.append(
                    PitcherGameStatRow(
                        game_pk=game_pk,
                        pitcher_id=int(pid),
                        team_code=team_code,
                        pitches_thrown=int(stats.get("pitchesThrown", 0) or 0),
                        outs_recorded=int(stats.get("outs", 0) or 0),
                        runs_allowed=int(stats.get("runs", 0) or 0),
                        walks=int(stats.get("baseOnBalls", 0) or 0),
                        hits_allowed=int(stats.get("hits", 0) or 0),
                        strikeouts=int(stats.get("strikeOuts", 0) or 0),
                        is_starter=(i == 0),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
    return rows


__all__ = [
    "REACHES_BASE_EVENT_CODES",
    "ParsedGame",
    "ScheduleEntry",
    "parse_live_feed",
    "parse_player",
    "parse_schedule",
    "parse_teams",
]
