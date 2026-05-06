"""Live game state dataclasses.

Phase 2, Task 2.1: LiveGameState and ShutdownFeatures frozen dataclasses.

Requirements: 2.1, 2.2, 2.3, 2.4, 6.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ShutdownFeatures:
    """Features indicating a pitcher is dominating (shutdown situation)."""
    pitcher_zero_baserunner_innings: int = 0
    is_bottom_of_order: bool = False
    pitcher_game_k_rate: float = 0.0
    team_season_k_rate: float = 0.0
    is_fresh_reliever: bool = False


@dataclass(frozen=True)
class LiveGameState:
    """Current state of a live MLB game, fetched on demand."""
    game_pk: int
    inning_number: int              # 0 if game hasn't started
    inning_half: str                # "top" | "bottom" | ""
    home_team_code: str
    away_team_code: str
    home_score: int
    away_score: int
    current_pitcher_name: str
    current_pitcher_id: int
    pitch_count: int
    batters_due_up: list[str] = field(default_factory=list)
    previous_inning_summary: str = ""
    fetched_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_final: bool = False

    def staleness_seconds(self) -> float:
        """Seconds elapsed since this state was fetched."""
        return (datetime.now(timezone.utc) - self.fetched_at_utc).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "game_pk": self.game_pk,
            "inning_number": self.inning_number,
            "inning_half": self.inning_half,
            "home_team_code": self.home_team_code,
            "away_team_code": self.away_team_code,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "current_pitcher_name": self.current_pitcher_name,
            "current_pitcher_id": self.current_pitcher_id,
            "pitch_count": self.pitch_count,
            "batters_due_up": list(self.batters_due_up),
            "previous_inning_summary": self.previous_inning_summary,
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
            "is_final": self.is_final,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveGameState:
        """Deserialize from a dict (round-trip with to_dict)."""
        fetched = d.get("fetched_at_utc")
        if isinstance(fetched, str):
            fetched = datetime.fromisoformat(fetched)
        elif fetched is None:
            fetched = datetime.now(timezone.utc)

        return cls(
            game_pk=int(d["game_pk"]),
            inning_number=int(d["inning_number"]),
            inning_half=str(d.get("inning_half", "")),
            home_team_code=str(d["home_team_code"]),
            away_team_code=str(d["away_team_code"]),
            home_score=int(d["home_score"]),
            away_score=int(d["away_score"]),
            current_pitcher_name=str(d.get("current_pitcher_name", "")),
            current_pitcher_id=int(d.get("current_pitcher_id", 0)),
            pitch_count=int(d.get("pitch_count", 0)),
            batters_due_up=list(d.get("batters_due_up", [])),
            previous_inning_summary=str(d.get("previous_inning_summary", "")),
            fetched_at_utc=fetched,
            is_final=bool(d.get("is_final", False)),
        )


__all__ = ["LiveGameState", "ShutdownFeatures"]
