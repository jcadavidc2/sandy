"""Property-based test: probability range (Property 4).

Property: the probability returned by predict_from_features() falls in the
closed interval [0.0, 1.0] for every generated input.

Uses Hypothesis to generate arbitrary valid FeatureVectors against a fixed
artifact and asserts 0.0 <= result.probability <= 1.0.

Validates: Requirements 8.2, 11.1
"""
from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from sandy.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from sandy.predict.predictor import predict_from_features
from sandy.schemas import FeatureVector, ModelArtifact


# ---------------------------------------------------------------------------
# Fixture: a trained model artifact (session-scoped, built once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixed_artifact():
    """Train a small deterministic LightGBM model for testing."""
    n_features = len(FEATURE_NAMES)
    rng = np.random.default_rng(42)

    X = rng.random((500, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=500).astype(np.float32)

    params = {
        "objective": "binary",
        "num_leaves": 16,
        "min_data_in_leaf": 10,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 42,
    }

    lgb_train = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    booster = lgb.train(params, lgb_train, num_boost_round=30)

    return ModelArtifact(
        model=booster,
        feature_names=FEATURE_NAMES,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_window_start=date(2022, 4, 1),
        training_window_end=date(2024, 9, 30),
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Strategy: generate arbitrary feature vectors
# ---------------------------------------------------------------------------

@st.composite
def feature_vectors(draw):
    """Generate a FeatureVector with random but valid values."""
    values = {}
    for name in FEATURE_NAMES:
        if "obp" in name:
            values[name] = draw(st.floats(min_value=0.0, max_value=0.6))
        elif "era" in name:
            values[name] = draw(st.floats(min_value=0.0, max_value=15.0))
        elif "whip" in name:
            values[name] = draw(st.floats(min_value=0.0, max_value=4.0))
        elif "k9" in name:
            values[name] = draw(st.floats(min_value=0.0, max_value=20.0))
        elif "rpg" in name:
            values[name] = draw(st.floats(min_value=0.0, max_value=15.0))
        elif "pitches_before" in name:
            values[name] = draw(st.integers(min_value=0, max_value=120))
        elif "lineup_spot" in name and "obp" not in name:
            values[name] = draw(st.integers(min_value=1, max_value=9))
        elif "is_home" in name:
            values[name] = draw(st.integers(min_value=0, max_value=1))
        elif "ballpark_id" in name:
            values[name] = draw(st.integers(min_value=1, max_value=50))
        elif "inning_number" in name:
            values[name] = draw(st.integers(min_value=1, max_value=9))
        elif "prev_inning" in name:
            values[name] = draw(st.integers(min_value=0, max_value=1))
        elif "innings_reached" in name:
            values[name] = draw(st.integers(min_value=0, max_value=8))
        elif "streak" in name:
            values[name] = draw(st.integers(min_value=0, max_value=8))
        else:
            values[name] = draw(st.floats(min_value=-10.0, max_value=10.0))

    return FeatureVector(
        game_pk=None,
        team_code="SEA",
        inning_number=draw(st.integers(min_value=1, max_value=9)),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        values=values,
    )


# ---------------------------------------------------------------------------
# The property test
# ---------------------------------------------------------------------------

@given(fv=feature_vectors())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_probability_in_range(fv, fixed_artifact):
    """predict_from_features() always returns probability in [0.0, 1.0]."""
    result = predict_from_features(fv, fixed_artifact)

    assert 0.0 <= result.probability <= 1.0, (
        f"Probability {result.probability} is outside [0.0, 1.0].\n"
        f"Feature values: {fv.values}"
    )

    # Also verify top_features structure
    assert len(result.top_features) <= 5
    for tf in result.top_features:
        assert tf.name in FEATURE_NAMES
        assert isinstance(tf.contribution, float)
