"""Backfill existing over_under_outcomes rows with home_expected_runs, away_expected_runs, sigma_used.

For each existing row in derived.over_under_outcomes that has NULL sigma_used:
1. Load the feature_vector (JSONB) from the row
2. Predict home_runs and away_runs using the runs model
3. Predict sigma using the volatility model
4. UPDATE the row with the computed values

Usage:
    python -m sandy.scripts.backfill_sigma

Requires:
- Runs model trained (models/runs.pkl)
- Volatility model trained (models/volatility.pkl) — falls back to 3.3 if missing
- Database connection configured via environment variables
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sqlalchemy import text

from sandy.config import load_config
from sandy.db import create_engine, get_connection
from sandy.features.schema import GAME_FEATURE_NAMES
from sandy.logging import configure_logging, get_logger
from sandy.over_under.volatility import DEFAULT_SIGMA, predict_sigma
from sandy.train.artifact import load_artifact

logger = get_logger("scripts.backfill_sigma")


def main() -> None:
    config = load_config()
    configure_logging(config.logging.level)

    engine = create_engine(config)

    # Load the runs model
    runs_path = config.model.artifact_path("runs")
    try:
        runs_artifact = load_artifact(runs_path, expected_target="runs")
    except FileNotFoundError:
        logger.error("Runs model not found. Train it first: sandy train --target runs")
        sys.exit(1)

    # Load the volatility model (optional — falls back to DEFAULT_SIGMA)
    vol_path = config.model.artifact_path("volatility")
    vol_artifact = None
    try:
        vol_artifact = load_artifact(vol_path, expected_target="volatility")
        logger.info("Volatility model loaded for backfill")
    except (FileNotFoundError, Exception):
        logger.info(f"Volatility model not available, using default σ={DEFAULT_SIGMA}")

    updated = 0
    skipped = 0

    with get_connection(engine) as conn:
        # Fetch rows that need backfilling
        rows = conn.execute(
            text("""
                SELECT id, feature_vector
                FROM derived.over_under_outcomes
                WHERE sigma_used IS NULL
                ORDER BY id
            """)
        ).fetchall()

        logger.info(f"Found {len(rows)} rows to backfill")

        for row in rows:
            row_id = row[0]
            fv_raw = row[1]

            # Parse feature vector
            if isinstance(fv_raw, str):
                feature_vector = json.loads(fv_raw)
            elif isinstance(fv_raw, dict):
                feature_vector = fv_raw
            else:
                skipped += 1
                continue

            if not feature_vector:
                skipped += 1
                continue

            # Predict home runs (home perspective)
            home_features = np.array(
                [[float(feature_vector.get(name, 0.0)) for name in GAME_FEATURE_NAMES]],
                dtype=np.float32,
            )
            home_runs = float(runs_artifact.model.predict(home_features)[0])

            # Predict away runs (flip perspective)
            away_features = home_features.copy()
            is_home_idx = GAME_FEATURE_NAMES.index("is_home")
            home_era_idx = GAME_FEATURE_NAMES.index("home_starter_era")
            home_whip_idx = GAME_FEATURE_NAMES.index("home_starter_whip")
            away_era_idx = GAME_FEATURE_NAMES.index("away_starter_era")
            away_whip_idx = GAME_FEATURE_NAMES.index("away_starter_whip")
            home_rpg_idx = GAME_FEATURE_NAMES.index("home_trailing15_rpg")
            away_rpg_idx = GAME_FEATURE_NAMES.index("away_trailing15_rpg")
            home_obp_idx = GAME_FEATURE_NAMES.index("home_season_obp")
            away_obp_idx = GAME_FEATURE_NAMES.index("away_season_obp")

            # Flip is_home
            away_features[0, is_home_idx] = 0.0
            # Swap home/away features
            away_features[0, home_era_idx], away_features[0, away_era_idx] = (
                home_features[0, away_era_idx],
                home_features[0, home_era_idx],
            )
            away_features[0, home_whip_idx], away_features[0, away_whip_idx] = (
                home_features[0, away_whip_idx],
                home_features[0, home_whip_idx],
            )
            away_features[0, home_rpg_idx], away_features[0, away_rpg_idx] = (
                home_features[0, away_rpg_idx],
                home_features[0, home_rpg_idx],
            )
            away_features[0, home_obp_idx], away_features[0, away_obp_idx] = (
                home_features[0, away_obp_idx],
                home_features[0, home_obp_idx],
            )

            away_runs = float(runs_artifact.model.predict(away_features)[0])

            # Predict sigma
            if vol_artifact is not None:
                sigma_array = np.array(
                    [[float(feature_vector.get(name, 0.0)) for name in GAME_FEATURE_NAMES]],
                    dtype=np.float32,
                )
                sigma_pred = float(vol_artifact.model.predict(sigma_array)[0])
                sigma = max(1.0, min(8.0, sigma_pred))
            else:
                sigma = DEFAULT_SIGMA

            # Update the row
            conn.execute(
                text("""
                    UPDATE derived.over_under_outcomes
                    SET home_expected_runs = :home_runs,
                        away_expected_runs = :away_runs,
                        sigma_used = :sigma
                    WHERE id = :row_id
                """),
                {
                    "row_id": row_id,
                    "home_runs": round(home_runs, 4),
                    "away_runs": round(away_runs, 4),
                    "sigma": round(sigma, 4),
                },
            )
            updated += 1

            if updated % 100 == 0:
                logger.info(f"Backfilled {updated}/{len(rows)} rows...")

    logger.info(
        f"Backfill complete: {updated} updated, {skipped} skipped",
        extra={"component": "scripts.backfill_sigma", "updated": updated, "skipped": skipped},
    )
    print(f"✅ Backfill complete: {updated} rows updated, {skipped} skipped")


if __name__ == "__main__":
    main()
