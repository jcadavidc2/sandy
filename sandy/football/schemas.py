"""Typed rows and domain dataclasses for the football vertical.

Raw-ingest rows (TeamRow, MatchRow, MatchStatRow) are plain frozen dataclasses
produced by pure parser functions in :mod:`sandy.football.parsers`. Prediction
and calibration domain objects live here too, mirroring
:mod:`sandy.over_under.schemas`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Raw ingest rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamRow:
    team_id: int
    name: str
    fifa_code: str | None
    country: str | None
    confederation: str | None  # filled later via a static map; None at ingest
    fifa_rank: int | None
    logo_url: str | None


@dataclass(frozen=True)
class MatchRow:
    fixture_id: int
    match_date: date
    kickoff_utc: datetime | None
    league_id: int | None
    season: int | None
    competition: str | None
    competition_weight: float
    round: str | None
    home_team_id: int
    away_team_id: int
    venue_id: int | None
    venue_name: str | None
    status: str
    home_goals: int | None
    away_goals: int | None
    raw_payload_hash: str | None


@dataclass(frozen=True)
class MatchStatRow:
    fixture_id: int
    team_id: int
    is_home: bool
    possession: float | None
    shots_total: int | None
    shots_on_target: int | None
    corners: int | None
    fouls: int | None
    yellow_cards: int | None
    red_cards: int | None
    xg: float | None


# ---------------------------------------------------------------------------
# Domain objects (predictions / calibration) — mirror over_under.schemas
# ---------------------------------------------------------------------------

# Goal-total thresholds we score, analogous to baseball's STANDARD_THRESHOLDS.
GOAL_THRESHOLDS: list[float] = [1.5, 2.5, 3.5, 4.5]


@dataclass(frozen=True)
class FootballPrediction:
    fixture_id: int
    match_date: date
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    kickoff_utc: datetime | None
    predicted_at_utc: datetime
    lambda_home: float
    lambda_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over: dict[float, float]          # {1.5: .., 2.5: .., 3.5: .., 4.5: ..}
    p_btts: float
    most_likely_home: int
    most_likely_away: int
    scoreline: list[dict]               # top-N cells: [{"h":1,"a":0,"p":0.12}, ...]
    feature_vector: dict[str, float]
    p_correct: float | None = None      # filled by meta-model when available


@dataclass(frozen=True)
class FootballCalibrationSnapshot:
    snapshot_date: date
    market: str                          # 'result' | 'over_2_5' | 'btts'
    accuracy: float
    sample_size: int
    recommended_threshold: float | None
    covariate_insights: dict = field(default_factory=dict)


# Statuses API-Football uses for a completed match (goals are final).
FINISHED_STATUSES: frozenset[str] = frozenset({"FT", "AET", "PEN"})


__all__ = [
    "FINISHED_STATUSES",
    "GOAL_THRESHOLDS",
    "FootballCalibrationSnapshot",
    "FootballPrediction",
    "MatchRow",
    "MatchStatRow",
    "TeamRow",
]
