"""Pure JSON → dataclass parsers for ESPN payloads (no I/O — property-testable)."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from .schemas import MlsMatch, MlsTeam, MlsTeamStats, normalize_status

logger = logging.getLogger(__name__)

DISPLAY_TZ = ZoneInfo("America/Los_Angeles")

# ESPN statistic name → MlsTeamStats field.
_STAT_FIELDS = {
    "wonCorners": "corners",
    "totalShots": "total_shots",
    "shotsOnTarget": "shots_on_target",
    "possessionPct": "possession_pct",
    "foulsCommitted": "fouls",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "saves": "saves",
}


def _num(v, cast=int):
    try:
        return cast(float(str(v).replace("%", "")))
    except (TypeError, ValueError):
        return None


def parse_scoreboard_events(payload: dict) -> list[MlsMatch]:
    """Every event on a scoreboard page → MlsMatch (teams embedded)."""
    out: list[MlsMatch] = []
    for ev in payload.get("events", []) or []:
        try:
            comp = ev["competitions"][0]
            kicked = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            teams: dict[str, MlsTeam] = {}
            scores: dict[str, int | None] = {}
            for c in comp["competitors"]:
                side = c["homeAway"]  # 'home' | 'away'
                t = c["team"]
                teams[side] = MlsTeam(
                    team_id=int(t["id"]),
                    name=t.get("displayName") or t.get("name") or str(t["id"]),
                    abbrev=t.get("abbreviation"),
                    logo_url=t.get("logo"),
                )
                scores[side] = _num(c.get("score"))
            status = normalize_status(ev["status"]["type"]["name"])
            out.append(MlsMatch(
                event_id=int(ev["id"]),
                match_date=kicked.astimezone(DISPLAY_TZ).date(),
                kickoff_utc=kicked,
                season=(ev.get("season") or {}).get("year"),
                status=status,
                home=teams["home"],
                away=teams["away"],
                home_goals=scores["home"] if status == "FT" else scores["home"],
                away_goals=scores["away"] if status == "FT" else scores["away"],
            ))
        except (KeyError, IndexError, ValueError) as e:
            logger.warning("Skipping unparseable ESPN event %s: %s", ev.get("id"), e)
    return out


def parse_summary_stats(event_id: int, payload: dict) -> list[MlsTeamStats]:
    """boxscore.teams[].statistics → per-team stats rows (corners & covariates)."""
    out: list[MlsTeamStats] = []
    teams = (payload.get("boxscore") or {}).get("teams") or []
    for entry in teams:
        try:
            fields: dict = {}
            for s in entry.get("statistics", []) or []:
                key = _STAT_FIELDS.get(s.get("name"))
                if key:
                    fields[key] = _num(s.get("displayValue"), float if key == "possession_pct" else int)
            out.append(MlsTeamStats(
                event_id=event_id,
                team_id=int(entry["team"]["id"]),
                is_home=(entry.get("homeAway") == "home"),
                **fields,
            ))
        except (KeyError, ValueError) as e:
            logger.warning("Skipping unparseable summary team for event %s: %s", event_id, e)
    # ESPN sometimes omits homeAway in boxscore — fall back to header order (home first).
    if len(out) == 2 and out[0].is_home == out[1].is_home:
        out[0] = MlsTeamStats(**{**out[0].__dict__, "is_home": True})
        out[1] = MlsTeamStats(**{**out[1].__dict__, "is_home": False})
    return out
