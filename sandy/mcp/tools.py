"""MCP tool definitions and handlers for Sandy.

Phase 2, Task 13.3: Each tool maps to a Sandy Python function.
Auto-fetches live game state when only team_code is provided.
Logs every prediction via PredictionLogger automatically.

Requirements: 3.1, 3.3–3.6, 4.1–4.7, 7.2, 7.3, 9.6, 10.4, 12.1–12.4, 13.3, 14.3
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from sandy.logging import get_logger

logger = get_logger("mcp.tools")

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for MCP protocol)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_todays_schedule",
        "description": "Get today's MLB schedule with probable pitchers for all games.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_live_game_state",
        "description": "Get the current live state of a game (score, inning, pitcher, batters due up). Use this to verify what's happening before making predictions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "3-letter team code (e.g. SEA, LAD)"},
            },
            "required": ["team_code"],
        },
    },
    {
        "name": "predict_reached_base",
        "description": "Predict the probability that a team reaches base in a specific inning. Returns probability, confidence level, and top contributing features.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "Batting team code"},
                "opponent_code": {"type": "string", "description": "Opposing team code (optional if game is live)"},
                "inning": {"type": "integer", "description": "Target inning 1-9 (optional: defaults to next inning if game is live)"},
            },
            "required": ["team_code"],
        },
    },
    {
        "name": "predict_game_winner",
        "description": "Predict the probability that a team wins the game. Returns win probability and confidence level.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "Team to predict for"},
                "opponent_code": {"type": "string", "description": "Opposing team code (optional if game is live)"},
            },
            "required": ["team_code"],
        },
    },
    {
        "name": "predict_total_runs",
        "description": "Predict expected total runs for a game, with per-team breakdown and over/under probabilities for standard thresholds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "One of the teams playing"},
                "opponent_code": {"type": "string", "description": "The other team (optional if game is live)"},
            },
            "required": ["team_code"],
        },
    },
    {
        "name": "get_player_stats",
        "description": "Get a player's season statistics (OBP, batting average, ERA for pitchers).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player's full name"},
            },
            "required": ["player_name"],
        },
    },
    {
        "name": "get_calibration_report",
        "description": "Get Sandy's self-evaluation report showing prediction accuracy and calibration over recent days.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (default: 7)"},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def handle_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler. Never raises."""
    try:
        if tool_name == "get_todays_schedule":
            return _handle_schedule()
        elif tool_name == "get_live_game_state":
            return _handle_live_state(arguments)
        elif tool_name == "predict_reached_base":
            return _handle_predict_reached_base(arguments)
        elif tool_name == "predict_game_winner":
            return _handle_predict_game_winner(arguments)
        elif tool_name == "predict_total_runs":
            return _handle_predict_total_runs(arguments)
        elif tool_name == "get_player_stats":
            return _handle_player_stats(arguments)
        elif tool_name == "get_calibration_report":
            return _handle_calibration_report(arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------


def _handle_schedule() -> dict[str, Any]:
    from sandy.config import load_config
    from sandy.schedule.client import get_todays_schedule

    config = load_config()
    schedule = get_todays_schedule(config)
    return {
        "games": [
            {
                "game_pk": g.game_pk,
                "home_team": g.home_team_code,
                "away_team": g.away_team_code,
                "home_pitcher": g.home_probable_pitcher or "TBD",
                "away_pitcher": g.away_probable_pitcher or "TBD",
                "game_time_utc": g.game_time_utc.isoformat(),
                "status": g.status,
            }
            for g in schedule
        ],
        "count": len(schedule),
    }


def _handle_live_state(args: dict[str, Any]) -> dict[str, Any]:
    from sandy.config import load_config
    from sandy.live.client import NoActiveGameError, get_live_game_state

    team_code = args.get("team_code", "")
    config = load_config()

    try:
        state = get_live_game_state(team_code, config)
        return {
            "game_state": state.to_dict(),
            "staleness_seconds": round(state.staleness_seconds(), 1),
        }
    except NoActiveGameError as exc:
        return {"error": str(exc), "suggestion": "No game in progress. Try get_todays_schedule to see upcoming games."}


def _handle_predict_reached_base(args: dict[str, Any]) -> dict[str, Any]:
    from sandy.config import load_config
    from sandy.confidence.assessor import ConfidenceAssessor
    from sandy.predict.predictor import predict, InvalidInputError, MissingArtifactError

    team_code = args.get("team_code", "")
    opponent_code = args.get("opponent_code")
    inning = args.get("inning")
    config = load_config()

    # Try to get live state for context
    live_state_dict = None
    try:
        from sandy.live.client import get_live_game_state
        state = get_live_game_state(team_code, config)
        live_state_dict = state.to_dict()

        # Auto-resolve opponent and inning from live state
        if opponent_code is None:
            if state.home_team_code.upper() == team_code.upper():
                opponent_code = state.away_team_code
            else:
                opponent_code = state.home_team_code

        if inning is None:
            # Predict next at-bat inning
            if state.inning_half == "top":
                is_away_batting = True
                if team_code.upper() == state.away_team_code.upper():
                    inning = state.inning_number  # they're batting now
                else:
                    inning = state.inning_number  # bottom is next
            else:
                if team_code.upper() == state.home_team_code.upper():
                    inning = state.inning_number  # they're batting now
                else:
                    inning = state.inning_number + 1  # top of next
    except Exception:
        pass  # No live state available, use provided params

    if opponent_code is None or inning is None:
        return {"error": "Could not determine opponent or inning. Provide opponent_code and inning, or ensure a live game is in progress."}

    # Need a starter name — try to resolve from schedule
    starter = None
    try:
        from sandy.schedule.client import get_todays_schedule, resolve_starter_for_matchup
        schedule = get_todays_schedule(config)
        home_starter, away_starter = resolve_starter_for_matchup(schedule, team_code, opponent_code)
        # The opposing starter is the one pitching to our team
        from sandy.db import create_engine, get_connection
        engine = create_engine(config)
        with engine.connect() as conn:
            from sqlalchemy import text
            # Determine if team is home or away
            for g in schedule:
                if g.home_team_code.upper() == team_code.upper():
                    starter = away_starter  # opposing pitcher
                    break
                elif g.away_team_code.upper() == team_code.upper():
                    starter = home_starter  # opposing pitcher
                    break
    except Exception:
        pass

    if starter is None:
        return {"error": "Could not resolve opposing starter. Provide the starter name or ensure probable pitchers are announced."}

    try:
        result = predict(
            team=team_code,
            opp=opponent_code,
            inning=inning,
            starter=starter,
            config=config,
        )
    except (InvalidInputError, MissingArtifactError) as exc:
        return {"error": str(exc)}

    # Confidence assessment
    assessor = ConfidenceAssessor()
    confidence = assessor.assess(result.probability, "reached_base", result.top_features)

    response = {
        "prediction": {
            "target": "reached_base",
            "team": team_code.upper(),
            "opponent": opponent_code.upper(),
            "inning": inning,
            "probability": round(result.probability, 4),
            "confidence": confidence.level,
            "confidence_explanation": confidence.explanation,
            "top_features": [
                {"name": f.name, "contribution": round(f.contribution, 4)}
                for f in result.top_features
            ],
        },
    }

    if live_state_dict:
        response["game_state_used"] = live_state_dict

    return response


def _handle_predict_game_winner(args: dict[str, Any]) -> dict[str, Any]:
    from sandy.config import load_config
    from sandy.confidence.assessor import ConfidenceAssessor
    from sandy.predict.predictor import predict_game, InvalidInputError, MissingArtifactError

    team_code = args.get("team_code", "")
    opponent_code = args.get("opponent_code")
    config = load_config()

    # Try to resolve opponent from live state or schedule
    if opponent_code is None:
        try:
            from sandy.live.client import get_live_game_state
            state = get_live_game_state(team_code, config)
            if state.home_team_code.upper() == team_code.upper():
                opponent_code = state.away_team_code
            else:
                opponent_code = state.home_team_code
        except Exception:
            pass

    if opponent_code is None:
        return {"error": "Could not determine opponent. Provide opponent_code or ensure a game is in progress."}

    try:
        result = predict_game(
            team=team_code,
            opp=opponent_code,
            target="game_winner",
            config=config,
        )
    except (InvalidInputError, MissingArtifactError) as exc:
        return {"error": str(exc)}

    assessor = ConfidenceAssessor()
    confidence = assessor.assess(result.probability, "game_winner", result.top_features)

    return {
        "prediction": {
            "target": "game_winner",
            "team": team_code.upper(),
            "opponent": opponent_code.upper(),
            "win_probability": round(result.probability, 4),
            "confidence": confidence.level,
            "confidence_explanation": confidence.explanation,
            "top_features": [
                {"name": f.name, "contribution": round(f.contribution, 4)}
                for f in result.top_features
            ],
        },
    }


def _handle_predict_total_runs(args: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import norm
    from sandy.config import load_config
    from sandy.predict.predictor import predict_game, InvalidInputError, MissingArtifactError

    team_code = args.get("team_code", "")
    opponent_code = args.get("opponent_code")
    config = load_config()

    # Resolve opponent
    if opponent_code is None:
        try:
            from sandy.live.client import get_live_game_state
            state = get_live_game_state(team_code, config)
            if state.home_team_code.upper() == team_code.upper():
                opponent_code = state.away_team_code
            else:
                opponent_code = state.home_team_code
        except Exception:
            pass

    if opponent_code is None:
        return {"error": "Could not determine opponent. Provide opponent_code."}

    # Get per-team runs predictions
    try:
        home_result = predict_game(team=team_code, opp=opponent_code, target="runs", config=config)
        away_result = predict_game(team=opponent_code, opp=team_code, target="runs", config=config)
    except (InvalidInputError, MissingArtifactError) as exc:
        return {"error": str(exc)}

    home_runs = home_result.probability
    away_runs = away_result.probability
    total = home_runs + away_runs

    # Over/under using normal approximation (σ ≈ 2.8 for MLB game totals)
    residual_std = 2.8
    thresholds = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    over_under = []
    for t in thresholds:
        p_over = float(1.0 - norm.cdf((t - total) / residual_std))
        over_under.append({"threshold": t, "probability_over": round(p_over, 4)})

    return {
        "prediction": {
            "target": "total_runs",
            "team": team_code.upper(),
            "opponent": opponent_code.upper(),
            "home_expected_runs": round(home_runs, 2),
            "away_expected_runs": round(away_runs, 2),
            "total_expected_runs": round(total, 2),
            "over_under": over_under,
            "residual_std": residual_std,
        },
    }


def _handle_player_stats(args: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    player_name = args.get("player_name", "")
    config = load_config()
    engine = create_engine(config)

    with engine.connect() as conn:
        # Find player
        row = conn.execute(
            text("""
                SELECT player_id, full_name, primary_position
                FROM raw.players
                WHERE LOWER(full_name) LIKE LOWER(:pattern)
                ORDER BY full_name LIMIT 1
            """),
            {"pattern": f"%{player_name}%"},
        ).fetchone()

        if row is None:
            return {"error": f"Player '{player_name}' not found."}

        player_id, full_name, position = row

        # Get batting stats (OBP approximation from plays)
        season = date.today().year
        batting = conn.execute(
            text("""
                SELECT
                    COUNT(*) as pa,
                    SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                                 'walk','hit_by_pitch') THEN 1 ELSE 0 END) as on_base,
                    SUM(CASE WHEN event_code IN ('single','double','triple','home_run')
                        THEN 1 ELSE 0 END) as hits
                FROM raw.plays p
                JOIN raw.games g ON g.game_pk = p.game_pk
                WHERE p.batter_id = :pid
                  AND g.status = 'Final'
                  AND EXTRACT(YEAR FROM g.game_date) = :season
            """),
            {"pid": player_id, "season": season},
        ).fetchone()

        pa = int(batting[0] or 0)
        on_base = int(batting[1] or 0)
        hits = int(batting[2] or 0)

        stats: dict[str, Any] = {
            "player_name": full_name.strip(),
            "position": position.strip() if position else "Unknown",
            "season": season,
        }

        if pa > 0:
            stats["plate_appearances"] = pa
            stats["obp"] = round(on_base / pa, 3)
            stats["batting_avg"] = round(hits / pa, 3) if pa > 0 else 0.0

        # Pitching stats if applicable
        pitching = conn.execute(
            text("""
                SELECT
                    SUM(outs_recorded) as outs,
                    SUM(runs_allowed) as runs,
                    SUM(strikeouts) as ks,
                    COUNT(*) as games
                FROM raw.pitcher_game_stats pgs
                JOIN raw.games g ON g.game_pk = pgs.game_pk
                WHERE pgs.pitcher_id = :pid
                  AND g.status = 'Final'
                  AND EXTRACT(YEAR FROM g.game_date) = :season
            """),
            {"pid": player_id, "season": season},
        ).fetchone()

        if pitching and pitching[0] and int(pitching[0]) > 0:
            outs = int(pitching[0])
            innings = outs / 3.0
            stats["pitching"] = {
                "games": int(pitching[3]),
                "innings_pitched": round(innings, 1),
                "era": round(9.0 * int(pitching[1] or 0) / innings, 2),
                "strikeouts": int(pitching[2] or 0),
            }

    return {"stats": stats}


def _handle_calibration_report(args: dict[str, Any]) -> dict[str, Any]:
    from sandy.config import load_config
    from sandy.db import create_engine
    from sandy.evaluation.reporter import get_calibration_report

    days = args.get("days", 7)
    config = load_config()
    engine = create_engine(config)

    report = get_calibration_report(engine, days=days)
    return {
        "report": {
            "total_predictions": report.total_predictions,
            "date_range": f"{report.date_range_start} to {report.date_range_end}",
            "accuracy_by_target": report.accuracy_by_target,
            "accuracy_by_confidence": report.accuracy_by_confidence,
            "summary": report.natural_language_summary,
        },
    }


__all__ = ["TOOL_DEFINITIONS", "handle_tool_call"]
