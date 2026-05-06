"""DB-agnostic row types and shared dataclasses for Sandy.

Task 3.4: TypedDict row types for every raw and derived table, plus the
public-API dataclasses used across pipelines: FeatureVector, ModelArtifact,
TopFeature, PredictionResult.

TypedDicts mirror the DB columns exactly so parsers and runners can pass them
directly to SQLAlchemy Core insert/upsert statements without an ORM layer.
Frozen dataclasses are used for the richer objects that cross pipeline
boundaries (features, artifacts, predictions).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# raw.teams
# ---------------------------------------------------------------------------


class TeamRow(TypedDict):
    team_code: str          # CHAR(3) PK
    team_id: int
    name: str
    venue_id: int | None
    league: str | None
    division: str | None


# ---------------------------------------------------------------------------
# raw.players
# ---------------------------------------------------------------------------


class PlayerRow(TypedDict):
    player_id: int          # PK
    full_name: str
    primary_position: str | None
    throws: str | None      # CHAR(1)
    bats: str | None        # CHAR(1)


# ---------------------------------------------------------------------------
# raw.games
# ---------------------------------------------------------------------------


class GameRow(TypedDict):
    game_pk: int            # PK
    game_date: date
    season: int
    game_type: str          # CHAR(1): 'R' regular, 'P' post
    status: str             # 'Final', 'In Progress', ...
    home_team_code: str
    away_team_code: str
    venue_id: int | None
    first_pitch_utc: datetime | None
    home_score: int | None
    away_score: int | None
    home_starter_id: int | None
    away_starter_id: int | None
    raw_payload_hash: str   # sha256 of full feed/live JSON


# ---------------------------------------------------------------------------
# raw.plays
# ---------------------------------------------------------------------------


class PlayRow(TypedDict):
    game_pk: int            # FK → games, part of composite PK
    at_bat_index: int       # part of composite PK
    inning: int
    half_inning: str        # 'top' | 'bottom'
    batting_team_code: str
    pitching_team_code: str
    batter_id: int | None
    pitcher_id: int | None
    batting_order: int | None   # 1..9; nullable for pinch etc.
    event_type: str         # grouped: 'hit', 'walk', 'strikeout', ...
    event_code: str         # raw: 'single', 'double', 'field_error', ...
    is_reaches_base: bool   # denormalized for label-gen speed
    pitches_in_pa: int
    start_time_utc: datetime | None
    end_time_utc: datetime | None
    raw: dict[str, Any]     # source atBat object (stored as JSONB)


# ---------------------------------------------------------------------------
# raw.pitcher_game_stats
# ---------------------------------------------------------------------------


class PitcherGameStatRow(TypedDict):
    game_pk: int            # FK → games, part of composite PK
    pitcher_id: int         # FK → players, part of composite PK
    team_code: str
    pitches_thrown: int
    outs_recorded: int
    runs_allowed: int
    walks: int
    hits_allowed: int
    strikeouts: int
    is_starter: bool


# ---------------------------------------------------------------------------
# raw.ingest_failures
# ---------------------------------------------------------------------------


class IngestFailureRow(TypedDict):
    game_pk: int | None
    endpoint: str | None
    error_reason: str
    http_status: int | None
    retries: int


# ---------------------------------------------------------------------------
# derived.inning_labels
# ---------------------------------------------------------------------------


class InningLabelRow(TypedDict):
    game_pk: int            # FK → games, part of composite PK
    team_code: str          # part of composite PK
    inning_number: int      # part of composite PK
    reached_base: bool


# ---------------------------------------------------------------------------
# derived.inning_features
# ---------------------------------------------------------------------------


class InningFeatureRow(TypedDict):
    game_pk: int
    team_code: str
    inning_number: int
    feature_schema_version: int
    opp_starter_era: float | None
    opp_starter_whip: float | None
    opp_starter_k9: float | None
    opp_starter_pitches_before: int | None
    lineup_spot_1: int | None
    lineup_spot_2: int | None
    lineup_spot_3: int | None
    is_home: bool | None
    ballpark_id: int | None
    inning_number_feat: int | None
    trailing15_rpg: float | None
    trailing15_obp: float | None


# ---------------------------------------------------------------------------
# Public-API dataclasses (cross-pipeline contracts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureVector:
    """Feature vector for a single (team, inning) prediction or training row."""
    game_pk: int | None         # None for prediction-time (hypothetical inning)
    team_code: str
    inning_number: int
    feature_schema_version: int
    values: dict[str, float | int | bool]   # keyed by FEATURE_NAMES


@dataclass(frozen=True)
class TopFeature:
    """A single feature contribution from SHAP-style pred_contrib output."""
    name: str
    contribution: float


@dataclass(frozen=True)
class PredictionResult:
    """Output of predict_from_features() — probability + top contributing features."""
    probability: float
    top_features: list[TopFeature]

    def to_json(self) -> str:
        """Serialize to the JSON format emitted by `sandy predict` on stdout."""
        return json.dumps(
            {
                "probability": self.probability,
                "top_features": [
                    {"name": f.name, "contribution": f.contribution}
                    for f in self.top_features
                ],
            }
        )


@dataclass(frozen=True)
class InningLabel:
    """In-memory label for a single (game, team, inning) — output of generator."""
    game_pk: int
    team_code: str
    inning_number: int
    reached_base: bool


@dataclass(frozen=True)
class ModelArtifact:
    """Loaded model artifact — wraps the LightGBM Booster plus metadata."""
    # model is typed as Any to avoid a hard import of lightgbm at schema-load
    # time; callers that use it will have lightgbm available.
    model: Any
    feature_names: list[str]
    feature_schema_version: int
    training_window_start: date
    training_window_end: date
    created_at: datetime
    target_name: str = "reached_base"  # Phase 1.5: identifies which model this is


# ---------------------------------------------------------------------------
# Phase 1.5: Game-level prediction dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameWinnerLabel:
    """Label for game winner prediction: did the home team win?"""
    game_pk: int
    home_team_wins: bool


@dataclass(frozen=True)
class RunsLabel:
    """Label for runs prediction: how many runs did this team score?"""
    game_pk: int
    team_code: str
    runs: int


@dataclass(frozen=True)
class GameFeatureVector:
    """Game-level feature vector for game_winner and runs predictions."""
    game_pk: int | None
    team_code: str
    feature_schema_version: int
    values: dict[str, float | int | bool]


@dataclass(frozen=True)
class ScheduledGame:
    """A game from today's MLB schedule with probable pitchers."""
    game_pk: int
    home_team_code: str
    away_team_code: str
    home_probable_pitcher: str | None
    away_probable_pitcher: str | None
    game_time_utc: datetime
    status: str


# ---------------------------------------------------------------------------
# Phase 2: Over/under and total runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverUnderLine:
    """A single over/under threshold with its probability."""
    threshold: float
    probability_over: float


@dataclass(frozen=True)
class TotalRunsResult:
    """Total runs prediction with per-team breakdown and over/under lines."""
    home_expected_runs: float
    away_expected_runs: float
    total_expected_runs: float
    over_under_lines: list[OverUnderLine]
    residual_std: float


__all__ = [
    "FeatureVector",
    "GameRow",
    "IngestFailureRow",
    "InningFeatureRow",
    "InningLabel",
    "InningLabelRow",
    "ModelArtifact",
    "PitcherGameStatRow",
    "PlayRow",
    "PlayerRow",
    "PredictionResult",
    "TeamRow",
    "TopFeature",
]
