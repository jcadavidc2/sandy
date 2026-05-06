"""Predictor — pure prediction function + high-level convenience wrapper.

Task 11.1: predict_from_features() — pure, no DB, no filesystem.
Task 11.2: predict() — high-level wrapper that resolves inputs via DB.

predict_from_features() is the STABLE entry point that Phase 2+ agents
will import directly. It takes a FeatureVector + ModelArtifact and returns
a PredictionResult with probability and top-5 feature contributions.

Requirements: 7.3, 8.1–8.8
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from sandy.config import Config
from sandy.db import create_engine, get_connection
from sandy.features.builder import build_feature_vector
from sandy.features.schema import FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger
from sandy.schemas import FeatureVector, ModelArtifact, PredictionResult, TopFeature
from sandy.train.artifact import FeatureSchemaMismatch, load_artifact

logger = get_logger("predict.predictor")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidInputError(Exception):
    """Raised for invalid user input (team code, inning, starter name).

    The CLI translates this to exit code 2 (requirement 8.4, 8.5, 8.6).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MissingArtifactError(Exception):
    """Raised when no model artifact exists at the configured path.

    The CLI translates this to exit code 3 (requirement 8.8).
    """

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"No model artifact found at {path}. "
            f"Run 'sandy train' to create one."
        )
        self.path = path


# ---------------------------------------------------------------------------
# Pure prediction (task 11.1)
# ---------------------------------------------------------------------------


def predict_from_features(
    features: FeatureVector,
    artifact: ModelArtifact,
) -> PredictionResult:
    """Pure prediction: no DB, no filesystem.

    This is the STABLE entry point Phase 2+ agents will import.

    Steps:
    1. Verify feature_schema_version matches (requirement 7.3)
    2. Build numpy array from feature values in the correct order
    3. Compute probability via model.predict()
    4. Compute feature contributions via pred_contrib=True
    5. Return PredictionResult with probability + top-5 features

    Requirements: 7.3, 8.2, 8.3
    """
    # Step 1: schema version check
    if features.feature_schema_version != artifact.feature_schema_version:
        raise FeatureSchemaMismatch(
            loaded=artifact.feature_schema_version,
            current=features.feature_schema_version,
        )

    # Step 2: build input array in feature_names order
    x = np.array(
        [[features.values.get(name, 0.0) for name in artifact.feature_names]],
        dtype=np.float64,
    )

    # Step 3: probability
    proba = float(artifact.model.predict(x)[0])
    # Clamp to [0, 1] for safety (LightGBM binary should already be in range)
    proba = max(0.0, min(1.0, proba))

    # Step 4: feature contributions (SHAP-style)
    contribs = artifact.model.predict(x, pred_contrib=True)[0]
    # contribs has len = n_features + 1 (last element is bias)
    named_contribs = list(zip(artifact.feature_names, contribs[:-1]))
    named_contribs.sort(key=lambda p: abs(p[1]), reverse=True)

    # Step 5: top 5
    top = [
        TopFeature(name=name, contribution=float(contrib))
        for name, contrib in named_contribs[:5]
    ]

    return PredictionResult(probability=proba, top_features=top)


# ---------------------------------------------------------------------------
# High-level predict wrapper (task 11.2)
# ---------------------------------------------------------------------------


def predict(
    team: str,
    opp: str,
    inning: int,
    starter: str,
    *,
    as_of: date | None = None,
    config: Config | None = None,
) -> PredictionResult:
    """High-level convenience: resolves inputs via DB, builds features,
    loads latest artifact, returns result.

    Parameters
    ----------
    team:    Batting team code (e.g. "SEA")
    opp:     Opposing team code (e.g. "LAD")
    inning:  Target inning (1-9)
    starter: Opposing starting pitcher name
    as_of:   Optional date ceiling for feature computation
    config:  Optional Config; if None, loads from env/TOML

    Raises
    ------
    InvalidInputError:   bad team code, inning, or starter name (exit 2)
    MissingArtifactError: no model file (exit 3)
    FeatureSchemaMismatch: model version mismatch

    Requirements: 8.1, 8.4–8.8
    """
    from sandy.config import load_config

    if config is None:
        config = load_config()

    # Validate inning (requirement 8.5)
    if not isinstance(inning, int) or inning < 1 or inning > 9:
        raise InvalidInputError(
            f"Invalid inning: {inning}. Must be an integer between 1 and 9."
        )

    engine = create_engine(config)

    with get_connection(engine) as conn:
        # Resolve team codes (requirement 8.4)
        team_code = _resolve_team_code(conn, team)
        opp_code = _resolve_team_code(conn, opp)

        # Resolve starter (requirement 8.6)
        starter_id = _resolve_starter(conn, starter)

        # Determine as_of date
        effective_date = as_of if as_of is not None else date.today()

        # Build feature vector (prediction path: game_pk=None)
        features = build_feature_vector(
            conn=conn,
            team_code=team_code,
            opp_team_code=opp_code,
            inning_number=inning,
            opp_starter_id=starter_id,
            game_date=effective_date,
            game_pk=None,
            as_of=None,
        )

    # Load artifact (requirement 8.8)
    model_path = config.model.path
    try:
        artifact = load_artifact(model_path)
    except FileNotFoundError:
        raise MissingArtifactError(model_path)

    # Predict
    return predict_from_features(features, artifact)


# ---------------------------------------------------------------------------
# Input resolution helpers
# ---------------------------------------------------------------------------


def _resolve_team_code(conn: Connection, code: str) -> str:
    """Resolve a team code case-insensitively. Raises InvalidInputError if not found."""
    row = conn.execute(
        text("SELECT team_code FROM raw.teams WHERE UPPER(team_code) = UPPER(:code)"),
        {"code": code.strip()},
    ).fetchone()

    if row is None:
        # Get all valid codes for the error message
        all_codes = conn.execute(
            text("SELECT team_code FROM raw.teams ORDER BY team_code")
        ).fetchall()
        valid = [r[0].strip() for r in all_codes]
        raise InvalidInputError(
            f"Unrecognized team code: '{code}'. "
            f"Valid codes: {', '.join(valid[:10])}{'...' if len(valid) > 10 else ''}"
        )

    return row[0].strip()


def _resolve_starter(conn: Connection, name: str) -> int:
    """Resolve a pitcher name to player_id. Raises InvalidInputError on miss."""
    # Exact case-insensitive match
    row = conn.execute(
        text("""
            SELECT player_id FROM raw.players
            WHERE LOWER(full_name) = LOWER(:name)
        """),
        {"name": name.strip()},
    ).fetchone()

    if row is not None:
        return int(row[0])

    # Fuzzy fallback: find closest matches using simple substring/prefix matching
    candidates = conn.execute(
        text("""
            SELECT player_id, full_name FROM raw.players
            WHERE LOWER(full_name) LIKE LOWER(:pattern)
            ORDER BY full_name
            LIMIT 5
        """),
        {"pattern": f"%{name.strip()}%"},
    ).fetchall()

    if len(candidates) == 1:
        return int(candidates[0][0])

    if candidates:
        matches = [f"  - {r[1].strip()}" for r in candidates]
        raise InvalidInputError(
            f"Starter '{name}' not found. Did you mean one of:\n"
            + "\n".join(matches)
        )

    # No matches at all — try broader search
    candidates = conn.execute(
        text("""
            SELECT full_name FROM raw.players
            WHERE LOWER(full_name) LIKE LOWER(:pattern)
            ORDER BY full_name
            LIMIT 5
        """),
        {"pattern": f"%{name.strip().split()[-1]}%"},
    ).fetchall()

    if candidates:
        matches = [f"  - {r[0].strip()}" for r in candidates]
        raise InvalidInputError(
            f"Starter '{name}' not found. Closest matches:\n"
            + "\n".join(matches)
        )

    raise InvalidInputError(
        f"Starter '{name}' not found in the database. "
        f"Check spelling or run 'sandy ingest' to update player data."
    )


__all__ = [
    "InvalidInputError",
    "MissingArtifactError",
    "predict",
    "predict_from_features",
]
