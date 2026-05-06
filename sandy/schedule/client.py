"""Schedule client — fetch today's MLB games and probable pitchers.

Phase 1.5, Task 12.1: Reuses MlbStatsClient for rate limiting and retries.
Exposes get_todays_schedule() and resolve_starter_for_matchup() as
importable Python functions for Phase 2+ agents.

Requirements: 7.1–7.6, 8.1–8.5
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sandy.config import Config, IngestConfig, load_config
from sandy.ingest.client import MlbApiError, MlbApiRetryExhausted, MlbStatsClient
from sandy.logging import get_logger
from sandy.predict.predictor import InvalidInputError
from sandy.schemas import ScheduledGame

logger = get_logger("schedule.client")


def get_todays_schedule(config: Config | None = None) -> list[ScheduledGame]:
    """Fetch today's MLB schedule with probable pitchers.

    Reuses MlbStatsClient for rate limiting and retries.
    Returns a list of ScheduledGame objects.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    """
    if config is None:
        config = load_config()

    client = MlbStatsClient(config.ingest)
    today = date.today().isoformat()

    try:
        payload = client.get(
            "/v1/schedule",
            params={
                "sportId": "1",
                "date": today,
                "hydrate": "probablePitcher",
            },
        )
    except (MlbApiError, MlbApiRetryExhausted) as exc:
        logger.error(
            "Failed to fetch today's schedule",
            extra={"component": "schedule.client", "error": str(exc)},
        )
        raise RuntimeError(f"Could not fetch today's schedule: {exc}") from exc

    return _parse_schedule_response(payload)


def resolve_starter_for_matchup(
    schedule: list[ScheduledGame],
    team: str,
    opp: str,
) -> tuple[str, str]:
    """Find the home and away starters for a given matchup in today's schedule.

    Returns (home_starter_name, away_starter_name).
    Raises InvalidInputError if matchup not found or pitcher is TBD.

    Requirements: 8.1, 8.3, 8.4, 8.5
    """
    team_upper = team.strip().upper()
    opp_upper = opp.strip().upper()

    for game in schedule:
        home = game.home_team_code.strip().upper()
        away = game.away_team_code.strip().upper()

        # Match either direction
        if (home == team_upper and away == opp_upper) or \
           (home == opp_upper and away == team_upper):
            if game.home_probable_pitcher is None:
                raise InvalidInputError(
                    f"Home probable pitcher for {game.home_team_code} is TBD. "
                    f"Try again later or specify --starter manually."
                )
            if game.away_probable_pitcher is None:
                raise InvalidInputError(
                    f"Away probable pitcher for {game.away_team_code} is TBD. "
                    f"Try again later or specify --starter manually."
                )
            return (game.home_probable_pitcher, game.away_probable_pitcher)

    raise InvalidInputError(
        f"Matchup {team} vs {opp} not found in today's schedule. "
        f"Check team codes or try a different date."
    )


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------


def _parse_schedule_response(payload: dict[str, Any]) -> list[ScheduledGame]:
    """Parse the MLB schedule API response into ScheduledGame objects."""
    games: list[ScheduledGame] = []

    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            try:
                game_pk = int(game["gamePk"])
                teams = game.get("teams", {})
                home = teams.get("home", {}).get("team", {})
                away = teams.get("away", {}).get("team", {})

                home_code = home.get("abbreviation", "UNK").upper()[:3]
                away_code = away.get("abbreviation", "UNK").upper()[:3]

                # Probable pitchers
                home_pitcher = teams.get("home", {}).get("probablePitcher", {})
                away_pitcher = teams.get("away", {}).get("probablePitcher", {})
                home_pitcher_name = home_pitcher.get("fullName") if home_pitcher else None
                away_pitcher_name = away_pitcher.get("fullName") if away_pitcher else None

                # Game time
                game_time_str = game.get("gameDate", "")
                game_time_utc = datetime.now(timezone.utc)
                if game_time_str:
                    try:
                        game_time_utc = datetime.fromisoformat(
                            game_time_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                status = game.get("status", {}).get("detailedState", "Unknown")

                games.append(ScheduledGame(
                    game_pk=game_pk,
                    home_team_code=home_code,
                    away_team_code=away_code,
                    home_probable_pitcher=home_pitcher_name,
                    away_probable_pitcher=away_pitcher_name,
                    game_time_utc=game_time_utc,
                    status=status,
                ))
            except (KeyError, ValueError, TypeError):
                continue

    return games


__all__ = ["get_todays_schedule", "resolve_starter_for_matchup"]
