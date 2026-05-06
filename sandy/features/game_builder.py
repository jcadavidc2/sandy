"""Game-level feature builder for game_winner and runs predictions.

Phase 1.5, Task 7.1: Builds a 10-feature vector for a (game, team) pair.
Uses only data from before game_date (no future leakage).

Features: home/away starter ERA+WHIP, home/away trailing-15 RPG,
home/away season OBP, ballpark_id, is_home.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""
from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.features.schema import GAME_FEATURE_NAMES, GAME_FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger
from sandy.schemas import GameFeatureVector

logger = get_logger("features.game_builder")

# Minimum appearances for a starter before we use their stats (else league avg)
_MIN_STARTER_APPEARANCES = 3


def build_game_feature_vector(
    conn: Connection,
    game_pk: int | None,
    team_code: str,
    opp_team_code: str,
    home_starter_id: int | None,
    away_starter_id: int | None,
    game_date: date,
    venue_id: int | None = None,
    is_home: bool = True,
) -> GameFeatureVector:
    """Build a game-level feature vector.

    For training: pass game_pk and real starter IDs.
    For prediction: pass game_pk=None and resolve starters from schedule.
    """
    season = game_date.year

    # Pitcher stats (with fallback to league average)
    home_pitcher = _get_pitcher_stats(conn, home_starter_id, game_date, season) if home_starter_id else _league_avg()
    away_pitcher = _get_pitcher_stats(conn, away_starter_id, game_date, season) if away_starter_id else _league_avg()

    # Team trailing-15 stats
    home_t15 = _get_team_trailing15_rpg(conn, team_code if is_home else opp_team_code, game_date)
    away_t15 = _get_team_trailing15_rpg(conn, opp_team_code if is_home else team_code, game_date)

    # Team season OBP
    home_obp = _get_team_season_obp(conn, team_code if is_home else opp_team_code, game_date, season)
    away_obp = _get_team_season_obp(conn, opp_team_code if is_home else team_code, game_date, season)

    values: dict[str, float | int | bool] = {
        "home_starter_era": home_pitcher["era"],
        "home_starter_whip": home_pitcher["whip"],
        "away_starter_era": away_pitcher["era"],
        "away_starter_whip": away_pitcher["whip"],
        "home_trailing15_rpg": home_t15,
        "away_trailing15_rpg": away_t15,
        "home_season_obp": home_obp,
        "away_season_obp": away_obp,
        "ballpark_id": venue_id if venue_id is not None else 0,
        "is_home": int(is_home),
    }

    return GameFeatureVector(
        game_pk=game_pk,
        team_code=team_code,
        feature_schema_version=GAME_FEATURE_SCHEMA_VERSION,
        values=values,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _league_avg() -> dict[str, float]:
    """Fallback league-average pitcher stats."""
    return {"era": 4.20, "whip": 1.30}


def _get_pitcher_stats(
    conn: Connection,
    pitcher_id: int,
    game_date: date,
    season: int,
) -> dict[str, float]:
    """Compute ERA and WHIP for a pitcher this season before game_date."""
    row = conn.execute(
        text("""
            SELECT
                COUNT(*) AS appearances,
                SUM(pgs.outs_recorded) AS total_outs,
                SUM(pgs.runs_allowed) AS total_runs,
                SUM(pgs.walks) AS total_walks,
                SUM(pgs.hits_allowed) AS total_hits
            FROM raw.pitcher_game_stats pgs
            JOIN raw.games g ON g.game_pk = pgs.game_pk
            WHERE pgs.pitcher_id = :pitcher_id
              AND g.game_date < :game_date
              AND g.status = 'Final'
              AND EXTRACT(YEAR FROM g.game_date) = :season
        """),
        {"pitcher_id": pitcher_id, "game_date": game_date, "season": season},
    ).fetchone()

    if row is None or row[0] < _MIN_STARTER_APPEARANCES or row[1] == 0:
        return _league_avg()

    total_outs = float(row[1])
    innings = total_outs / 3.0
    if innings == 0:
        return _league_avg()

    era = 9.0 * float(row[2] or 0) / innings
    whip = (float(row[3] or 0) + float(row[4] or 0)) / innings

    return {"era": era, "whip": whip}


def _get_team_trailing15_rpg(
    conn: Connection,
    team_code: str,
    game_date: date,
) -> float:
    """Compute runs/game for a team over their 15 most recent games."""
    row = conn.execute(
        text("""
            SELECT
                SUM(CASE WHEN home_team_code = :team THEN home_score
                         ELSE away_score END) AS total_runs,
                COUNT(*) AS games
            FROM (
                SELECT game_pk, home_team_code, home_score, away_score
                FROM raw.games
                WHERE (home_team_code = :team OR away_team_code = :team)
                  AND game_date < :game_date
                  AND status = 'Final'
                ORDER BY game_date DESC, game_pk DESC
                LIMIT 15
            ) recent
        """),
        {"team": team_code, "game_date": game_date},
    ).fetchone()

    if row is None or row[1] == 0:
        return 4.5  # league average fallback

    return float(row[0] or 0) / int(row[1])


def _get_team_season_obp(
    conn: Connection,
    team_code: str,
    game_date: date,
    season: int,
) -> float:
    """Compute team OBP for the season before game_date."""
    row = conn.execute(
        text("""
            SELECT
                SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                             'walk','hit_by_pitch') THEN 1 ELSE 0 END) AS on_base,
                COUNT(*) AS pa
            FROM raw.plays p
            JOIN raw.games g ON g.game_pk = p.game_pk
            WHERE p.batting_team_code = :team
              AND g.game_date < :game_date
              AND g.status = 'Final'
              AND EXTRACT(YEAR FROM g.game_date) = :season
        """),
        {"team": team_code, "game_date": game_date, "season": season},
    ).fetchone()

    if row is None or row[1] == 0:
        return 0.320  # league average fallback

    return int(row[0] or 0) / int(row[1])


__all__ = ["build_game_feature_vector"]
