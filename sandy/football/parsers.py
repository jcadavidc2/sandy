"""Pure parsers: API-Football JSON -> typed rows.

No network, no DB — deterministic transforms only, so they're trivially
unit-testable. Defensive against missing/null fields and the API's habit of
returning percentages as strings like ``"45%"``.

NOTE: response shapes follow API-Football v3 documented format. The exact
field presence is verified live against the account's plan during F1 backfill;
parsers degrade to None rather than raising on unexpected gaps.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from sandy.football.schemas import MatchRow, MatchStatRow, TeamRow

# Map a statistic "type" label (as API-Football names it) to our column.
_STAT_TYPE_MAP: dict[str, str] = {
    "Ball Possession": "possession",
    "Total Shots": "shots_total",
    "Shots on Goal": "shots_on_target",
    "Corner Kicks": "corners",
    "Fouls": "fouls",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "expected_goals": "xg",
}


# ---------------------------------------------------------------------------
# Scalar coercion helpers
# ---------------------------------------------------------------------------


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        if value == "":
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_to_float(value: Any) -> float | None:
    """'45%' -> 45.0; 45 -> 45.0; None -> None."""
    return _to_float(value)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def parse_teams(envelope: dict[str, Any]) -> list[TeamRow]:
    """Parse a /teams response envelope into TeamRow list.

    Only national teams are kept (``team.national == True``) when the flag is
    present; otherwise all are kept.
    """
    rows: list[TeamRow] = []
    for item in envelope.get("response", []):
        team = item.get("team") or {}
        team_id = _to_int(team.get("id"))
        if team_id is None:
            continue
        national = team.get("national")
        if national is False:
            continue
        rows.append(
            TeamRow(
                team_id=team_id,
                name=str(team.get("name") or "").strip() or f"team_{team_id}",
                fifa_code=(team.get("code") or None),
                country=(team.get("country") or None),
                confederation=None,  # filled later from a static map
                fifa_rank=None,
                logo_url=(team.get("logo") or None),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def parse_fixtures(
    envelope: dict[str, Any],
    *,
    competition_weight: float = 1.0,
) -> list[MatchRow]:
    """Parse a /fixtures response envelope into MatchRow list.

    ``competition_weight`` is supplied by the caller (it depends on the league
    being ingested: friendly < qualifier < continental < World Cup).
    """
    rows: list[MatchRow] = []
    for item in envelope.get("response", []):
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        venue = fixture.get("venue") or {}
        status = fixture.get("status") or {}

        fixture_id = _to_int(fixture.get("id"))
        home = (teams.get("home") or {})
        away = (teams.get("away") or {})
        home_id = _to_int(home.get("id"))
        away_id = _to_int(away.get("id"))
        if fixture_id is None or home_id is None or away_id is None:
            continue

        kickoff = _parse_iso8601(fixture.get("date"))
        match_d = kickoff.date() if kickoff else _date_from_timestamp(fixture.get("timestamp"))
        if match_d is None:
            continue

        rows.append(
            MatchRow(
                fixture_id=fixture_id,
                match_date=match_d,
                kickoff_utc=kickoff,
                league_id=_to_int(league.get("id")),
                season=_to_int(league.get("season")),
                competition=(league.get("name") or None),
                competition_weight=competition_weight,
                round=(league.get("round") or None),
                home_team_id=home_id,
                away_team_id=away_id,
                venue_id=_to_int(venue.get("id")),
                venue_name=(venue.get("name") or None),
                status=str(status.get("short") or "NS"),
                home_goals=_to_int(goals.get("home")),
                away_goals=_to_int(goals.get("away")),
                raw_payload_hash=_hash_payload(item),
            )
        )
    return rows


def parse_teams_from_fixtures(envelope: dict[str, Any]) -> list[TeamRow]:
    """Extract minimal TeamRows referenced by a /fixtures envelope.

    Fixtures must be able to satisfy the matches FK to teams even if /teams
    wasn't called first. These carry name + logo only; /teams enriches later.
    """
    seen: dict[int, TeamRow] = {}
    for item in envelope.get("response", []):
        teams = item.get("teams") or {}
        for side in ("home", "away"):
            t = teams.get(side) or {}
            tid = _to_int(t.get("id"))
            if tid is None or tid in seen:
                continue
            seen[tid] = TeamRow(
                team_id=tid,
                name=str(t.get("name") or "").strip() or f"team_{tid}",
                fifa_code=None,
                country=None,
                confederation=None,
                fifa_rank=None,
                logo_url=(t.get("logo") or None),
            )
    return list(seen.values())


# ---------------------------------------------------------------------------
# Fixture statistics
# ---------------------------------------------------------------------------


def parse_fixture_statistics(
    fixture_id: int,
    envelope: dict[str, Any],
    *,
    home_team_id: int | None = None,
) -> list[MatchStatRow]:
    """Parse a /fixtures/statistics response into per-team MatchStatRows.

    The response is a list of (usually 2) team blocks, each with a flat
    ``statistics`` array of {"type": label, "value": v}. We map known labels
    to our columns and ignore the rest.
    """
    rows: list[MatchStatRow] = []
    for block in envelope.get("response", []):
        team = block.get("team") or {}
        team_id = _to_int(team.get("id"))
        if team_id is None:
            continue

        cols: dict[str, Any] = {}
        for stat in block.get("statistics") or []:
            label = stat.get("type")
            if label not in _STAT_TYPE_MAP:
                continue
            cols[_STAT_TYPE_MAP[label]] = stat.get("value")

        rows.append(
            MatchStatRow(
                fixture_id=fixture_id,
                team_id=team_id,
                is_home=(team_id == home_team_id) if home_team_id is not None else False,
                possession=_pct_to_float(cols.get("possession")),
                shots_total=_to_int(cols.get("shots_total")),
                shots_on_target=_to_int(cols.get("shots_on_target")),
                corners=_to_int(cols.get("corners")),
                fouls=_to_int(cols.get("fouls")),
                yellow_cards=_to_int(cols.get("yellow_cards")),
                red_cards=_to_int(cols.get("red_cards")),
                xg=_to_float(cols.get("xg")),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _parse_iso8601(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        # API uses e.g. "2022-11-20T16:00:00+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _date_from_timestamp(value: Any) -> date | None:
    ts = _to_int(value)
    if ts is None:
        return None
    try:
        return datetime.utcfromtimestamp(ts).date()
    except (OverflowError, OSError, ValueError):
        return None


def _hash_payload(item: dict[str, Any]) -> str:
    blob = json.dumps(item, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "parse_fixture_statistics",
    "parse_fixtures",
    "parse_teams",
    "parse_teams_from_fixtures",
]
