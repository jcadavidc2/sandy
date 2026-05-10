"""Over/Under Predictor — compute P(total > T) for all scheduled games.

Uses the existing predict_game(target="runs") for per-team expected runs,
then applies a normal approximation with residual_std to compute
P(total > T) for each standard threshold.

Requirements: 1.1, 1.2, 1.4, 1.7, 12.5, 13.1
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from scipy.stats import norm
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.logging import get_logger
from sandy.over_under.schemas import STANDARD_THRESHOLDS, OverUnderPrediction

logger = get_logger("over_under.predictor")


def compute_over_under_probabilities(
    total_expected_runs: float,
    residual_std: float = 2.8,
) -> dict[float, float]:
    """Compute P(total > T) for all standard thresholds via normal CDF.

    Returns dict mapping threshold -> probability_over.
    Probabilities are monotonically decreasing as threshold increases.
    """
    result: dict[float, float] = {}
    for t in STANDARD_THRESHOLDS:
        p_over = float(1.0 - norm.cdf((t - total_expected_runs) / residual_std))
        result[t] = p_over
    return result


def predict_all_games(
    config: Config | None = None,
    game_date: date | None = None,
) -> list[OverUnderPrediction]:
    """Run over/under predictions for all games on the given date.

    Uses predict_game(target="runs") for each team, sums expected runs,
    then applies normal approximation for each threshold.
    Falls back to team season ERA when no probable pitcher is announced.
    """
    from sandy.predict.predictor import (
        InvalidInputError,
        MissingArtifactError,
        predict_game,
    )
    from sandy.schedule.client import get_todays_schedule

    if config is None:
        config = load_config()

    schedule = get_todays_schedule(config)
    if not schedule:
        return []

    predictions: list[OverUnderPrediction] = []
    effective_date = game_date if game_date is not None else date.today()

    for game in schedule:
        home = game.home_team_code.strip().upper()
        away = game.away_team_code.strip().upper()
        pitcher_fallback = False

        try:
            # Predict runs for home team
            home_result = predict_game(
                team=home,
                opp=away,
                target="runs",
                config=config,
                as_of=effective_date,
            )
            home_runs = home_result.probability  # For runs target, probability = expected runs
        except (InvalidInputError, MissingArtifactError) as exc:
            logger.warning(
                f"Failed to predict runs for {home}: {exc}",
                extra={"component": "over_under.predictor", "team": home},
            )
            # Fallback: use a default expected runs
            home_runs = 4.5
            pitcher_fallback = True

        try:
            # Predict runs for away team
            away_result = predict_game(
                team=away,
                opp=home,
                target="runs",
                config=config,
                as_of=effective_date,
            )
            away_runs = away_result.probability
        except (InvalidInputError, MissingArtifactError) as exc:
            logger.warning(
                f"Failed to predict runs for {away}: {exc}",
                extra={"component": "over_under.predictor", "team": away},
            )
            away_runs = 4.5
            pitcher_fallback = True

        total_expected = home_runs + away_runs

        # Compute matchup-specific σ using the volatility model
        from sandy.over_under.volatility import predict_sigma

        sigma = predict_sigma(feature_vector, config) if feature_vector else 3.3
        p_over = compute_over_under_probabilities(total_expected, residual_std=sigma)

        # Extract feature values from the prediction results
        feature_vector: dict[str, float] = {}
        home_starter_era: float | None = None
        away_starter_era: float | None = None
        ballpark_id: int | None = None
        home_trailing15_rpg: float | None = None
        away_trailing15_rpg: float | None = None

        try:
            from sandy.db import create_engine, get_connection
            from sandy.features.game_builder import build_game_feature_vector
            from sandy.predict.predictor import (
                _get_team_venue,
                _resolve_starter,
                _resolve_team_code,
            )
            from sandy.schedule.client import resolve_starter_for_matchup

            engine = create_engine(config)
            with get_connection(engine) as conn:
                team_code = _resolve_team_code(conn, home)
                opp_code = _resolve_team_code(conn, away)

                # Resolve starters
                try:
                    home_starter_name, away_starter_name = resolve_starter_for_matchup(
                        schedule, home, away
                    )
                    home_starter_id = _resolve_starter(conn, home_starter_name)
                    away_starter_id = _resolve_starter(conn, away_starter_name)
                except (InvalidInputError, Exception):
                    home_starter_id = None
                    away_starter_id = None
                    pitcher_fallback = True

                venue_id = _get_team_venue(conn, home)

                features = build_game_feature_vector(
                    conn=conn,
                    game_pk=None,
                    team_code=team_code,
                    opp_team_code=opp_code,
                    home_starter_id=home_starter_id,
                    away_starter_id=away_starter_id,
                    game_date=effective_date,
                    venue_id=venue_id,
                    is_home=True,
                )
                feature_vector = {
                    k: round(float(v), 6) for k, v in features.values.items()
                }
                home_starter_era = feature_vector.get("home_starter_era")
                away_starter_era = feature_vector.get("away_starter_era")
                ballpark_id_val = feature_vector.get("ballpark_id")
                ballpark_id = int(ballpark_id_val) if ballpark_id_val else None
                home_trailing15_rpg = feature_vector.get("home_trailing15_rpg")
                away_trailing15_rpg = feature_vector.get("away_trailing15_rpg")
        except Exception as exc:
            logger.warning(
                f"Failed to build feature vector for {home} vs {away}: {exc}",
                extra={"component": "over_under.predictor"},
            )

        now_utc = datetime.now(timezone.utc)

        prediction = OverUnderPrediction(
            game_pk=game.game_pk,
            game_date=effective_date,
            home_team_code=home,
            away_team_code=away,
            game_time_utc=game.game_time_utc,
            predicted_at_utc=now_utc,
            p_over=p_over,
            feature_vector=feature_vector,
            home_starter_era=home_starter_era,
            away_starter_era=away_starter_era,
            ballpark_id=ballpark_id,
            home_trailing15_rpg=home_trailing15_rpg,
            away_trailing15_rpg=away_trailing15_rpg,
            pitcher_fallback=pitcher_fallback,
            home_expected_runs=home_runs,
            away_expected_runs=away_runs,
            sigma_used=sigma,
        )
        predictions.append(prediction)

    logger.info(
        f"Generated {len(predictions)} over/under predictions",
        extra={"component": "over_under.predictor", "count": len(predictions)},
    )
    return predictions


def persist_predictions(engine: Engine, predictions: list[OverUnderPrediction]) -> int:
    """Upsert predictions to derived.over_under_outcomes. Returns row count."""
    if not predictions:
        return 0

    count = 0
    with engine.begin() as conn:
        for pred in predictions:
            # Serialize feature vector with 6-decimal rounding
            fv_json = json.dumps(
                {k: round(v, 6) for k, v in pred.feature_vector.items()}
            )

            conn.execute(
                text("""
                    INSERT INTO derived.over_under_outcomes (
                        game_pk, game_date, home_team_code, away_team_code,
                        predicted_at_utc,
                        p_over_5_5, p_over_6_5, p_over_7_5, p_over_8_5,
                        p_over_9_5, p_over_10_5, p_over_11_5,
                        feature_vector,
                        home_starter_era, away_starter_era, ballpark_id,
                        home_trailing15_rpg, away_trailing15_rpg,
                        pitcher_fallback,
                        home_expected_runs, away_expected_runs, sigma_used
                    ) VALUES (
                        :game_pk, :game_date, :home_team_code, :away_team_code,
                        :predicted_at_utc,
                        :p_over_5_5, :p_over_6_5, :p_over_7_5, :p_over_8_5,
                        :p_over_9_5, :p_over_10_5, :p_over_11_5,
                        :feature_vector,
                        :home_starter_era, :away_starter_era, :ballpark_id,
                        :home_trailing15_rpg, :away_trailing15_rpg,
                        :pitcher_fallback,
                        :home_expected_runs, :away_expected_runs, :sigma_used
                    )
                    ON CONFLICT (game_pk, game_date) DO UPDATE SET
                        predicted_at_utc = EXCLUDED.predicted_at_utc,
                        p_over_5_5 = EXCLUDED.p_over_5_5,
                        p_over_6_5 = EXCLUDED.p_over_6_5,
                        p_over_7_5 = EXCLUDED.p_over_7_5,
                        p_over_8_5 = EXCLUDED.p_over_8_5,
                        p_over_9_5 = EXCLUDED.p_over_9_5,
                        p_over_10_5 = EXCLUDED.p_over_10_5,
                        p_over_11_5 = EXCLUDED.p_over_11_5,
                        feature_vector = EXCLUDED.feature_vector,
                        home_starter_era = EXCLUDED.home_starter_era,
                        away_starter_era = EXCLUDED.away_starter_era,
                        ballpark_id = EXCLUDED.ballpark_id,
                        home_trailing15_rpg = EXCLUDED.home_trailing15_rpg,
                        away_trailing15_rpg = EXCLUDED.away_trailing15_rpg,
                        pitcher_fallback = EXCLUDED.pitcher_fallback,
                        home_expected_runs = EXCLUDED.home_expected_runs,
                        away_expected_runs = EXCLUDED.away_expected_runs,
                        sigma_used = EXCLUDED.sigma_used
                """),
                {
                    "game_pk": pred.game_pk,
                    "game_date": pred.game_date,
                    "home_team_code": pred.home_team_code,
                    "away_team_code": pred.away_team_code,
                    "predicted_at_utc": pred.predicted_at_utc,
                    "p_over_5_5": pred.p_over[5.5],
                    "p_over_6_5": pred.p_over[6.5],
                    "p_over_7_5": pred.p_over[7.5],
                    "p_over_8_5": pred.p_over[8.5],
                    "p_over_9_5": pred.p_over[9.5],
                    "p_over_10_5": pred.p_over[10.5],
                    "p_over_11_5": pred.p_over[11.5],
                    "feature_vector": fv_json,
                    "home_starter_era": pred.home_starter_era,
                    "away_starter_era": pred.away_starter_era,
                    "ballpark_id": pred.ballpark_id,
                    "home_trailing15_rpg": pred.home_trailing15_rpg,
                    "away_trailing15_rpg": pred.away_trailing15_rpg,
                    "pitcher_fallback": pred.pitcher_fallback,
                    "home_expected_runs": pred.home_expected_runs,
                    "away_expected_runs": pred.away_expected_runs,
                    "sigma_used": pred.sigma_used,
                },
            )
            count += 1

    logger.info(
        f"Persisted {count} over/under predictions",
        extra={"component": "over_under.predictor", "rows": count},
    )
    return count


__all__ = [
    "compute_over_under_probabilities",
    "persist_predictions",
    "predict_all_games",
]
