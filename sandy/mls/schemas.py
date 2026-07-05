"""Dataclasses + market thresholds for the MLS vertical."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

GOAL_THRESHOLDS: list[float] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
CORNER_THRESHOLDS: list[float] = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

# ESPN status.type.name → our normalized status.
_STATUS_MAP = {
    "STATUS_SCHEDULED": "NS",
    "STATUS_FULL_TIME": "FT",
    "STATUS_FINAL": "FT",
    "STATUS_FINAL_PEN": "FT",
    "STATUS_IN_PROGRESS": "LIVE",
    "STATUS_FIRST_HALF": "LIVE",
    "STATUS_SECOND_HALF": "LIVE",
    "STATUS_HALFTIME": "LIVE",
    "STATUS_POSTPONED": "PPD",
    "STATUS_CANCELED": "PPD",
    "STATUS_DELAYED": "PPD",
    "STATUS_ABANDONED": "PPD",
}


def normalize_status(espn_name: str) -> str:
    return _STATUS_MAP.get(espn_name, "NS" if "SCHEDULED" in espn_name else "LIVE")


@dataclass(frozen=True)
class MlsTeam:
    team_id: int
    name: str
    abbrev: str | None = None
    logo_url: str | None = None


@dataclass(frozen=True)
class MlsMatch:
    event_id: int
    match_date: date          # America/Los_Angeles calendar date of kickoff
    kickoff_utc: datetime | None
    season: int | None
    status: str               # NS/LIVE/FT/PPD
    home: MlsTeam
    away: MlsTeam
    home_goals: int | None
    away_goals: int | None


@dataclass(frozen=True)
class MlsTeamStats:
    event_id: int
    team_id: int
    is_home: bool
    corners: int | None = None
    total_shots: int | None = None
    shots_on_target: int | None = None
    possession_pct: float | None = None
    fouls: int | None = None
    offsides: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    saves: int | None = None


@dataclass
class MlsPrediction:
    event_id: int
    match_date: date
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_home_or_draw: float
    p_over: dict[float, float] = field(default_factory=dict)
    corner_lambda_home: float | None = None
    corner_lambda_away: float | None = None
    p_corners_over: dict[float, float] = field(default_factory=dict)
    most_likely: tuple[int, int] = (1, 1)
    features: dict | None = None
    is_backtest: bool = False
