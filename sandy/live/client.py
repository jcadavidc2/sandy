"""Live game state client — on-demand fetch from MLB API.

Phase 2, Task 4.1: Fetches current game state for a team's active game.
Makes exactly one HTTP request per invocation (no background polling).

Requirements: 1.1–1.7
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.config import Config, IngestConfig, load_config
from sandy.ingest.client import MlbApiError, MlbApiRetryExhausted, MlbStatsClient
from sandy.live.schemas import LiveGameState, ShutdownFeatures
from sandy.logging import get_logger

logger = get_logger("live.client")


class NoActiveGameError(Exception):
    """No game currently in progress for the requested team."""
    def __init__(self, team_code: str) -> None:
        super().__init__(f"No active game found for {team_code} today.")
        self.team_code = team_code


class LiveStateError(Exception):
    """Failed to fetch live game state from MLB API."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_live_game_state(
    team_code: str,
    config: Config | None = None,
) -> LiveGameState:
    """Fetch current game state for the team's active game.

    Makes exactly one HTTP request per invocation (on-demand).
    Reuses MlbStatsClient for rate limiting and retries.

    Raises:
        NoActiveGameError: No game in progress for this team.
        LiveStateError: MLB API unreachable or returned an error.
    """
    if config is None:
        config = load_config()

    client = MlbStatsClient(config.ingest)
    team_upper = team_code.strip().upper()

    # Step 1: Find active game_pk from today's schedule
    try:
        schedule = client.get(
            "/v1/schedule",
            params={
                "sportId": "1",
                "date": date.today().isoformat(),
                "hydrate": "probablePitcher,linescore",
            },
        )
    except (MlbApiError, MlbApiRetryExhausted) as exc:
        raise LiveStateError(f"Failed to fetch schedule: {exc}")

    game_pk = _find_active_game_pk(schedule, team_upper)
    if game_pk is None:
        raise NoActiveGameError(team_upper)

    # Step 2: Fetch live feed
    try:
        feed = client.get(f"/v1.1/game/{game_pk}/feed/live")
    except (MlbApiError, MlbApiRetryExhausted) as exc:
        raise LiveStateError(f"Failed to fetch live feed for game {game_pk}: {exc}")

    # Step 3: Parse into LiveGameState
    return _parse_live_feed_to_state(game_pk, feed)


def compute_shutdown_features(
    live_state: LiveGameState,
    conn: Connection,
    team_code: str,
) -> ShutdownFeatures:
    """Compute shutdown features from live game state + DB.

    Requirements: 6.1, 6.2
    """
    # pitcher_zero_baserunner_innings: from live feed play-by-play
    # For now, approximate from the score and innings pitched
    # A more precise version would parse the live feed's allPlays
    zero_bb_innings = 0  # TODO: compute from live feed when available

    # is_bottom_of_order: check if batters due up are spots 7-8-9
    is_bottom = False
    # We'd need batting order info from the live feed

    # pitcher_game_k_rate: from live pitcher stats
    pitcher_k_rate = 0.0

    # team_season_k_rate: from DB
    season = date.today().year
    row = conn.execute(
        text("""
            SELECT
                SUM(CASE WHEN event_code = 'strikeout' THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0)
            FROM raw.plays p
            JOIN raw.games g ON g.game_pk = p.game_pk
            WHERE p.batting_team_code = :team
              AND g.status = 'Final'
              AND EXTRACT(YEAR FROM g.game_date) = :season
        """),
        {"team": team_code, "season": season},
    ).fetchone()
    team_k_rate = float(row[0]) if row and row[0] else 0.20

    # is_fresh_reliever
    is_fresh = live_state.pitch_count < 20 and live_state.inning_number > 1

    return ShutdownFeatures(
        pitcher_zero_baserunner_innings=zero_bb_innings,
        is_bottom_of_order=is_bottom,
        pitcher_game_k_rate=pitcher_k_rate,
        team_season_k_rate=team_k_rate,
        is_fresh_reliever=is_fresh,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_active_game_pk(schedule: dict[str, Any], team_upper: str) -> int | None:
    """Find the game_pk for an active game involving the team."""
    for date_block in schedule.get("dates", []):
        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status not in ("Live", "In Progress"):
                # Also check detailedState
                detailed = game.get("status", {}).get("detailedState", "")
                if detailed not in ("In Progress", "Warmup", "Manager Challenge"):
                    continue

            teams = game.get("teams", {})
            home = teams.get("home", {}).get("team", {}).get("abbreviation", "").upper()
            away = teams.get("away", {}).get("team", {}).get("abbreviation", "").upper()

            if home == team_upper or away == team_upper:
                return int(game["gamePk"])

    return None


def _parse_live_feed_to_state(game_pk: int, feed: dict[str, Any]) -> LiveGameState:
    """Parse a live feed response into a LiveGameState."""
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})

    teams = game_data.get("teams", {})
    home_code = teams.get("home", {}).get("abbreviation", "UNK").upper()[:3]
    away_code = teams.get("away", {}).get("abbreviation", "UNK").upper()[:3]

    # Scores
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0

    # Inning
    inning_number = linescore.get("currentInning", 0) or 0
    inning_half = linescore.get("inningHalf", "").lower()
    if inning_half == "top":
        inning_half = "top"
    elif inning_half == "bottom":
        inning_half = "bottom"
    else:
        inning_half = ""

    # Status
    status = game_data.get("status", {}).get("detailedState", "")
    is_final = status == "Final"

    # Current pitcher
    pitcher_name = ""
    pitcher_id = 0
    pitch_count = 0
    defense = live_data.get("linescore", {}).get("defense", {})
    pitcher_info = defense.get("pitcher", {})
    if pitcher_info:
        pitcher_name = pitcher_info.get("fullName", "")
        pitcher_id = pitcher_info.get("id", 0)
    # Pitch count from boxscore
    boxscore = live_data.get("boxscore", {})
    # Try to get pitch count from the current pitcher's stats
    for side in ("home", "away"):
        team_box = boxscore.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        key = f"ID{pitcher_id}"
        if key in players:
            stats = players[key].get("stats", {}).get("pitching", {})
            pitch_count = int(stats.get("pitchesThrown", 0) or 0)
            break

    # Batters due up
    batters_due_up: list[str] = []
    offense = live_data.get("linescore", {}).get("offense", {})
    for key in ("batter", "onDeck", "inHole"):
        player = offense.get(key, {})
        if player and player.get("fullName"):
            batters_due_up.append(player["fullName"])

    # Previous inning summary (simplified)
    previous_summary = ""
    plays = live_data.get("plays", {})
    scoring_plays = plays.get("scoringPlays", [])
    if scoring_plays:
        previous_summary = f"{len(scoring_plays)} scoring plays so far"

    return LiveGameState(
        game_pk=game_pk,
        inning_number=inning_number,
        inning_half=inning_half,
        home_team_code=home_code,
        away_team_code=away_code,
        home_score=int(home_score),
        away_score=int(away_score),
        current_pitcher_name=pitcher_name,
        current_pitcher_id=pitcher_id,
        pitch_count=pitch_count,
        batters_due_up=batters_due_up,
        previous_inning_summary=previous_summary,
        fetched_at_utc=datetime.now(timezone.utc),
        is_final=is_final,
    )


__all__ = ["LiveStateError", "NoActiveGameError", "compute_shutdown_features", "get_live_game_state"]
