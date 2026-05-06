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
# Human-readable feature descriptions
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Inning-level features
    "opp_starter_era": "Opposing starter's season ERA",
    "opp_starter_whip": "Opposing starter's season WHIP",
    "opp_starter_k9": "Opposing starter's season K/9",
    "opp_starter_pitches_before": "Pitches thrown by starter before this inning",
    "lineup_spot_1": "Batting order spot of 1st batter due up (1-9)",
    "lineup_spot_2": "Batting order spot of 2nd batter due up (1-9)",
    "lineup_spot_3": "Batting order spot of 3rd batter due up (1-9)",
    "lineup_spot1_season_obp": "Season OBP of batter due up 1st",
    "lineup_spot2_season_obp": "Season OBP of batter due up 2nd",
    "lineup_spot3_season_obp": "Season OBP of batter due up 3rd",
    "is_home": "1 if batting team is home, 0 if away",
    "ballpark_id": "Venue ID (ballpark factor proxy)",
    "inning_number_feat": "Inning number (1-9)",
    "trailing15_rpg": "Team runs/game over trailing 15 games",
    "trailing15_obp": "Team OBP over trailing 15 games",
    "prev_inning_reached_base": "1 if team reached base in previous inning",
    "innings_reached_so_far": "Count of prior innings with a baserunner",
    "consecutive_reach_streak": "Current consecutive inning streak with baserunner",
    "team_season_obp": "Team season OBP before game date",
    "team_season_rpg": "Team season runs/game before game date",
    # Game-level features
    "home_starter_era": "Home starting pitcher's season ERA",
    "home_starter_whip": "Home starting pitcher's season WHIP",
    "away_starter_era": "Away starting pitcher's season ERA",
    "away_starter_whip": "Away starting pitcher's season WHIP",
    "home_trailing15_rpg": "Home team runs/game over trailing 15 games",
    "away_trailing15_rpg": "Away team runs/game over trailing 15 games",
    "home_season_obp": "Home team season OBP",
    "away_season_obp": "Away team season OBP",
}

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
    {
        "name": "get_team_recent_games",
        "description": "Get a team's recent game results (scores, opponents, wins/losses). Use this to answer questions like 'how have the Mariners been doing?' or 'what were SEA's last 5 games?'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "3-letter team code (e.g. SEA)"},
                "count": {"type": "integer", "description": "Number of recent games to return (default: 5, max: 20)"},
            },
            "required": ["team_code"],
        },
    },
    {
        "name": "query_team_stats",
        "description": "Get comprehensive team statistics: season record, batting stats (OBP, runs/game, hits/game), pitching stats (ERA, WHIP, K/9), last 5 game results, and trailing 15-game averages vs season averages. Use for EDA and descriptive stats questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "team_code": {"type": "string", "description": "3-letter team code (e.g. SEA, LAD, NYY)"},
            },
            "required": ["team_code"],
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
        elif tool_name == "get_team_recent_games":
            return _handle_team_recent_games(arguments)
        elif tool_name == "query_team_stats":
            return _handle_query_team_stats(arguments)
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

    # Build feature vector separately to get feature values for response
    feature_values: dict[str, float] = {}
    try:
        from sandy.features.builder import build_feature_vector
        from sandy.db import create_engine as _create_engine, get_connection as _get_conn
        from sandy.predict.predictor import _resolve_team_code, _resolve_starter
        from sqlalchemy import text as _text

        _engine = _create_engine(config)
        with _get_conn(_engine) as _conn:
            _team = _resolve_team_code(_conn, team_code)
            _opp = _resolve_team_code(_conn, opponent_code)
            _starter_id = _resolve_starter(_conn, starter)
            _features = build_feature_vector(
                conn=_conn,
                team_code=_team,
                opp_team_code=_opp,
                inning_number=inning,
                opp_starter_id=_starter_id,
                game_date=date.today(),
                game_pk=None,
                as_of=None,
            )
            feature_values = {k: float(v) for k, v in _features.values.items()}
    except Exception:
        pass  # Feature values are best-effort enrichment

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
            "feature_details": [
                {
                    "name": f.name,
                    "value": round(feature_values.get(f.name, 0.0), 4),
                    "contribution": round(f.contribution, 4),
                    "meaning": FEATURE_DESCRIPTIONS.get(f.name, f.name),
                }
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

    # Determine home/away from schedule
    home_code = team_code.upper()
    away_code = opponent_code.upper()
    try:
        from sandy.schedule.client import get_todays_schedule
        schedule = get_todays_schedule(config)
        for game in schedule:
            h = game.home_team_code.strip().upper()
            a = game.away_team_code.strip().upper()
            if h == team_code.strip().upper() and a == opponent_code.strip().upper():
                home_code = team_code.upper()
                away_code = opponent_code.upper()
                break
            elif h == opponent_code.strip().upper() and a == team_code.strip().upper():
                home_code = opponent_code.upper()
                away_code = team_code.upper()
                break
    except Exception:
        pass

    try:
        result = predict_game(
            team=team_code,
            opp=opponent_code,
            target="game_winner",
            config=config,
        )
    except (InvalidInputError, MissingArtifactError) as exc:
        return {"error": str(exc)}

    # Build game feature vector to get feature values
    feature_values: dict[str, float] = {}
    try:
        from sandy.features.game_builder import build_game_feature_vector
        from sandy.db import create_engine as _create_engine, get_connection as _get_conn
        from sandy.predict.predictor import _resolve_team_code, _resolve_starter, _get_team_venue
        from sandy.schedule.client import get_todays_schedule, resolve_starter_for_matchup

        _engine = _create_engine(config)
        with _get_conn(_engine) as _conn:
            _team = _resolve_team_code(_conn, team_code)
            _opp = _resolve_team_code(_conn, opponent_code)

            # Determine home/away and starters
            _schedule = get_todays_schedule(config)
            _home_starter_name, _away_starter_name = resolve_starter_for_matchup(
                _schedule, _team, _opp
            )
            _is_home = True
            for g in _schedule:
                h = g.home_team_code.strip().upper()
                a = g.away_team_code.strip().upper()
                if h == _opp.strip().upper() and a == _team.strip().upper():
                    _is_home = False
                    break

            _home_starter_id = _resolve_starter(
                _conn, _home_starter_name if _is_home else _away_starter_name
            ) if (_home_starter_name if _is_home else _away_starter_name) else None
            _away_starter_id = _resolve_starter(
                _conn, _away_starter_name if _is_home else _home_starter_name
            ) if (_away_starter_name if _is_home else _home_starter_name) else None

            _venue_id = _get_team_venue(_conn, home_code)

            _features = build_game_feature_vector(
                conn=_conn,
                game_pk=None,
                team_code=_team,
                opp_team_code=_opp,
                home_starter_id=_home_starter_id,
                away_starter_id=_away_starter_id,
                game_date=date.today(),
                venue_id=_venue_id,
                is_home=_is_home,
            )
            feature_values = {k: float(v) for k, v in _features.values.items()}
    except Exception:
        pass  # Feature values are best-effort enrichment

    assessor = ConfidenceAssessor()
    confidence = assessor.assess(result.probability, "game_winner", result.top_features)

    return {
        "prediction": {
            "target": "game_winner",
            "home_team": home_code,
            "away_team": away_code,
            "home_win_probability": round(result.probability, 4),
            "away_win_probability": round(1 - result.probability, 4),
            "predicted_for": team_code.upper(),
            "confidence": confidence.level,
            "confidence_explanation": confidence.explanation,
            "top_features": [
                {"name": f.name, "contribution": round(f.contribution, 4)}
                for f in result.top_features
            ],
            "feature_details": [
                {
                    "name": f.name,
                    "value": round(feature_values.get(f.name, 0.0), 4),
                    "contribution": round(f.contribution, 4),
                    "meaning": FEATURE_DESCRIPTIONS.get(f.name, f.name),
                }
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


def _handle_team_recent_games(args: dict[str, Any]) -> dict[str, Any]:
    """Return a team's recent game results."""
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    team_code = args.get("team_code", "").upper().strip()
    count = min(args.get("count", 5), 20)
    config = load_config()
    engine = create_engine(config)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT game_date, home_team_code, away_team_code,
                       home_score, away_score, status
                FROM raw.games
                WHERE (home_team_code = :team OR away_team_code = :team)
                  AND status = 'Final'
                ORDER BY game_date DESC, game_pk DESC
                LIMIT :count
            """),
            {"team": team_code, "count": count},
        ).fetchall()

    if not rows:
        return {"error": f"No games found for {team_code}."}

    games = []
    wins = 0
    losses = 0
    total_runs_for = 0
    total_runs_against = 0

    for row in rows:
        game_date, home, away, home_score, away_score, status = row
        home = home.strip()
        away = away.strip()

        is_home = (home == team_code)
        opponent = away if is_home else home
        team_runs = home_score if is_home else away_score
        opp_runs = away_score if is_home else home_score
        won = team_runs > opp_runs

        if won:
            wins += 1
        else:
            losses += 1
        total_runs_for += team_runs
        total_runs_against += opp_runs

        games.append({
            "date": str(game_date),
            "opponent": opponent,
            "score": f"{team_runs}-{opp_runs}",
            "result": "W" if won else "L",
            "home_away": "home" if is_home else "away",
        })

    return {
        "team": team_code,
        "recent_games": games,
        "summary": {
            "record": f"{wins}-{losses}",
            "runs_scored_avg": round(total_runs_for / len(games), 1),
            "runs_allowed_avg": round(total_runs_against / len(games), 1),
        },
    }


def _handle_query_team_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Return comprehensive team statistics for EDA questions."""
    from sqlalchemy import text
    from sandy.config import load_config
    from sandy.db import create_engine

    team_code = args.get("team_code", "").upper().strip()
    config = load_config()
    engine = create_engine(config)
    season = date.today().year

    with engine.connect() as conn:
        # Validate team code
        team_row = conn.execute(
            text("SELECT team_code, name FROM raw.teams WHERE UPPER(team_code) = UPPER(:code)"),
            {"code": team_code},
        ).fetchone()
        if team_row is None:
            return {"error": f"Unknown team code: '{team_code}'."}

        team_name = team_row[1].strip()

        # --- Season record (W-L) ---
        record_row = conn.execute(
            text("""
                SELECT
                    SUM(CASE
                        WHEN (home_team_code = :team AND home_score > away_score)
                          OR (away_team_code = :team AND away_score > home_score)
                        THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE
                        WHEN (home_team_code = :team AND home_score < away_score)
                          OR (away_team_code = :team AND away_score < home_score)
                        THEN 1 ELSE 0 END) AS losses,
                    COUNT(*) AS games_played
                FROM raw.games
                WHERE (home_team_code = :team OR away_team_code = :team)
                  AND status = 'Final'
                  AND EXTRACT(YEAR FROM game_date) = :season
            """),
            {"team": team_code, "season": season},
        ).fetchone()

        wins = int(record_row[0] or 0)
        losses = int(record_row[1] or 0)
        games_played = int(record_row[2] or 0)

        # --- Team batting stats (season) ---
        batting_row = conn.execute(
            text("""
                SELECT
                    COUNT(*) AS pa,
                    SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                                 'walk','hit_by_pitch') THEN 1 ELSE 0 END) AS on_base,
                    SUM(CASE WHEN event_code IN ('single','double','triple','home_run')
                        THEN 1 ELSE 0 END) AS hits
                FROM raw.plays p
                JOIN raw.games g ON g.game_pk = p.game_pk
                WHERE p.batting_team_code = :team
                  AND g.status = 'Final'
                  AND EXTRACT(YEAR FROM g.game_date) = :season
            """),
            {"team": team_code, "season": season},
        ).fetchone()

        pa = int(batting_row[0] or 0)
        on_base = int(batting_row[1] or 0)
        hits = int(batting_row[2] or 0)
        season_obp = round(on_base / pa, 3) if pa > 0 else 0.0
        hits_per_game = round(hits / games_played, 1) if games_played > 0 else 0.0

        # Runs per game
        runs_row = conn.execute(
            text("""
                SELECT SUM(CASE WHEN home_team_code = :team THEN home_score
                                ELSE away_score END) AS total_runs
                FROM raw.games
                WHERE (home_team_code = :team OR away_team_code = :team)
                  AND status = 'Final'
                  AND EXTRACT(YEAR FROM game_date) = :season
            """),
            {"team": team_code, "season": season},
        ).fetchone()
        total_runs = int(runs_row[0] or 0)
        runs_per_game = round(total_runs / games_played, 2) if games_played > 0 else 0.0

        # --- Team pitching stats (season) ---
        pitching_row = conn.execute(
            text("""
                SELECT
                    SUM(pgs.outs_recorded) AS total_outs,
                    SUM(pgs.runs_allowed) AS total_runs_allowed,
                    SUM(pgs.walks) AS total_walks,
                    SUM(pgs.hits_allowed) AS total_hits_allowed,
                    SUM(pgs.strikeouts) AS total_ks
                FROM raw.pitcher_game_stats pgs
                JOIN raw.games g ON g.game_pk = pgs.game_pk
                WHERE pgs.team_code = :team
                  AND g.status = 'Final'
                  AND EXTRACT(YEAR FROM g.game_date) = :season
            """),
            {"team": team_code, "season": season},
        ).fetchone()

        total_outs = int(pitching_row[0] or 0)
        innings_pitched = total_outs / 3.0 if total_outs > 0 else 0.0
        team_era = round(9.0 * int(pitching_row[1] or 0) / innings_pitched, 2) if innings_pitched > 0 else 0.0
        team_whip = round((int(pitching_row[2] or 0) + int(pitching_row[3] or 0)) / innings_pitched, 2) if innings_pitched > 0 else 0.0
        team_k9 = round(9.0 * int(pitching_row[4] or 0) / innings_pitched, 1) if innings_pitched > 0 else 0.0

        # --- Last 5 games ---
        last5_rows = conn.execute(
            text("""
                SELECT game_date, home_team_code, away_team_code,
                       home_score, away_score
                FROM raw.games
                WHERE (home_team_code = :team OR away_team_code = :team)
                  AND status = 'Final'
                ORDER BY game_date DESC, game_pk DESC
                LIMIT 5
            """),
            {"team": team_code},
        ).fetchall()

        last5_games = []
        for row in last5_rows:
            gd, home, away, hs, as_ = row
            home = home.strip()
            away = away.strip()
            is_home = (home == team_code)
            opp = away if is_home else home
            team_runs = hs if is_home else as_
            opp_runs = as_ if is_home else hs
            won = team_runs > opp_runs
            last5_games.append({
                "date": str(gd),
                "opponent": opp,
                "score": f"{team_runs}-{opp_runs}",
                "result": "W" if won else "L",
                "home_away": "home" if is_home else "away",
            })

        # --- Trailing 15 games vs season averages ---
        trailing15_row = conn.execute(
            text("""
                SELECT
                    SUM(CASE WHEN home_team_code = :team THEN home_score
                             ELSE away_score END) AS runs_for,
                    SUM(CASE WHEN home_team_code = :team THEN away_score
                             ELSE home_score END) AS runs_against,
                    COUNT(*) AS games
                FROM (
                    SELECT home_team_code, away_team_code, home_score, away_score
                    FROM raw.games
                    WHERE (home_team_code = :team OR away_team_code = :team)
                      AND status = 'Final'
                    ORDER BY game_date DESC, game_pk DESC
                    LIMIT 15
                ) recent
            """),
            {"team": team_code},
        ).fetchone()

        t15_games = int(trailing15_row[2] or 0)
        t15_rpg = round(int(trailing15_row[0] or 0) / t15_games, 2) if t15_games > 0 else 0.0
        t15_ra_pg = round(int(trailing15_row[1] or 0) / t15_games, 2) if t15_games > 0 else 0.0

        # Trailing 15 OBP
        t15_obp_row = conn.execute(
            text("""
                SELECT
                    SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                                 'walk','hit_by_pitch') THEN 1 ELSE 0 END) AS on_base,
                    COUNT(*) AS pa
                FROM raw.plays p
                WHERE p.batting_team_code = :team
                  AND p.game_pk IN (
                      SELECT game_pk FROM raw.games
                      WHERE (home_team_code = :team OR away_team_code = :team)
                        AND status = 'Final'
                      ORDER BY game_date DESC, game_pk DESC
                      LIMIT 15
                  )
            """),
            {"team": team_code},
        ).fetchone()
        t15_pa = int(t15_obp_row[1] or 0)
        t15_obp = round(int(t15_obp_row[0] or 0) / t15_pa, 3) if t15_pa > 0 else 0.0

    return {
        "team": team_code,
        "team_name": team_name,
        "season": season,
        "record": {"wins": wins, "losses": losses, "games_played": games_played},
        "batting": {
            "obp": season_obp,
            "runs_per_game": runs_per_game,
            "hits_per_game": hits_per_game,
        },
        "pitching": {
            "era": team_era,
            "whip": team_whip,
            "k_per_9": team_k9,
        },
        "last_5_games": last5_games,
        "trailing_15_vs_season": {
            "trailing_15_rpg": t15_rpg,
            "season_rpg": runs_per_game,
            "trailing_15_runs_allowed_pg": t15_ra_pg,
            "trailing_15_obp": t15_obp,
            "season_obp": season_obp,
        },
    }


__all__ = ["TOOL_DEFINITIONS", "handle_tool_call"]
