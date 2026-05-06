"""Property-based test: model serializer round-trip (Property 3).

Property: saving a trained LightGBM model via save_artifact() and loading
it via load_artifact() produces predictions matching the in-memory model
within 1e-9 absolute tolerance on a fixed input.

Uses Hypothesis to generate fixed input feature matrices, fits a
deterministic LightGBM, saves and reloads, then asserts numerical
equivalence.

Validates: Requirements 7.2, 11.4
"""
from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from sandy.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from sandy.schemas import ModelArtifact
from sandy.train.artifact import load_artifact, save_artifact


# ---------------------------------------------------------------------------
# Strategy: generate a small training dataset and fit a deterministic model
# ---------------------------------------------------------------------------

@st.composite
def trained_model_and_input(draw):
    """Generate a fitted LightGBM model + a random input matrix for prediction.

    The model is trained on synthetic data with deterministic=True so the
    round-trip property is testable.
    """
    n_features = len(FEATURE_NAMES)
    seed = draw(st.integers(min_value=1, max_value=10000))

    rng = np.random.default_rng(seed)

    # Generate synthetic training data (small but enough for LightGBM)
    n_train = draw(st.integers(min_value=200, max_value=500))
    X_train = rng.random((n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, size=n_train).astype(np.float32)

    # Fit a deterministic LightGBM model
    params = {
        "objective": "binary",
        "num_leaves": 8,
        "min_data_in_leaf": 10,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    booster = lgb.train(params, lgb_train, num_boost_round=20)

    # Generate a small input matrix for prediction comparison
    n_pred = draw(st.integers(min_value=1, max_value=10))
    X_pred = rng.random((n_pred, n_features)).astype(np.float32)

    artifact = ModelArtifact(
        model=booster,
        feature_names=FEATURE_NAMES,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_window_start=date(2022, 4, 1),
        training_window_end=date(2024, 9, 30),
        created_at=datetime.now(timezone.utc),
    )

    return artifact, X_pred


# ---------------------------------------------------------------------------
# The property test
# ---------------------------------------------------------------------------

@given(data=trained_model_and_input())
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_serializer_roundtrip(data):
    """save_artifact → load_artifact produces numerically identical predictions."""
    artifact, X_pred = data

    # Get predictions from the in-memory model
    original_preds = artifact.model.predict(X_pred)

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pkl"
        save_artifact(artifact, path)
        reloaded = load_artifact(path)

    # Get predictions from the reloaded model
    reloaded_preds = reloaded.model.predict(X_pred)

    # Assert numerical equivalence within 1e-9
    assert np.allclose(original_preds, reloaded_preds, atol=1e-9), (
        f"Predictions differ after round-trip.\n"
        f"Max absolute diff: {np.max(np.abs(original_preds - reloaded_preds))}\n"
        f"Original:  {original_preds[:5]}\n"
        f"Reloaded:  {reloaded_preds[:5]}"
    )

    # Also verify metadata survived
    assert reloaded.feature_names == artifact.feature_names
    assert reloaded.feature_schema_version == artifact.feature_schema_version
    assert reloaded.training_window_start == artifact.training_window_start
    assert reloaded.training_window_end == artifact.training_window_end
