# %% [markdown]
# # Sandy Phase 1 — Prototype Notebook
#
# This notebook walks through the entire Sandy pipeline end-to-end:
# 1. Connect to the DB and explore the raw data
# 2. Generate labels
# 3. Build features
# 4. Train the model
# 5. Make a prediction
# 6. Inspect feature importance
#
# Run this on the EC2 with the DB env vars set, or point it at your Postgres.

# %% [markdown]
# ## 1. Setup & DB Connection

# %%
import os
import pandas as pd
import numpy as np
from sqlalchemy import text

# Set these if not already in your environment
# os.environ["MLB_DB_HOST"] = "localhost"
# os.environ["MLB_DB_PORT"] = "5432"
# os.environ["MLB_DB_NAME"] = "sandy"
# os.environ["MLB_DB_USER"] = "sandy"
# os.environ["MLB_DB_PASSWORD"] = "sandydev"

from sandy.config import load_config
from sandy.db import create_engine, get_connection, bootstrap_schema

cfg = load_config()
engine = create_engine(cfg)
print(f"Connected to: {cfg.database.host}:{cfg.database.port}/{cfg.database.name}")

# %% [markdown]
# ## 2. Explore the Raw Data

# %%
# How much data do we have?
with engine.connect() as conn:
    games_count = conn.execute(text("SELECT COUNT(*) FROM raw.games WHERE status = 'Final'")).scalar()
    plays_count = conn.execute(text("SELECT COUNT(*) FROM raw.plays")).scalar()
    teams_count = conn.execute(text("SELECT COUNT(*) FROM raw.teams")).scalar()
    players_count = conn.execute(text("SELECT COUNT(*) FROM raw.players")).scalar()

print(f"Final games:  {games_count:,}")
print(f"Total plays:  {plays_count:,}")
print(f"Teams:        {teams_count}")
print(f"Players:      {players_count:,}")

# %%
# Sample of games
with engine.connect() as conn:
    df_games = pd.read_sql(
        "SELECT game_pk, game_date, season, home_team_code, away_team_code, "
        "home_score, away_score FROM raw.games WHERE status = 'Final' "
        "ORDER BY game_date DESC LIMIT 20",
        conn,
    )
df_games

# %%
# Sample of plays from one game
with engine.connect() as conn:
    sample_pk = conn.execute(
        text("SELECT game_pk FROM raw.games WHERE status = 'Final' ORDER BY game_date LIMIT 1")
    ).scalar()

    df_plays = pd.read_sql(
        f"SELECT at_bat_index, inning, half_inning, batting_team_code, "
        f"event_code, is_reaches_base, pitches_in_pa "
        f"FROM raw.plays WHERE game_pk = {sample_pk} "
        f"ORDER BY at_bat_index",
        conn,
    )
print(f"Game {sample_pk}: {len(df_plays)} plate appearances")
df_plays.head(20)

# %%
# Event code distribution
with engine.connect() as conn:
    df_events = pd.read_sql(
        "SELECT event_code, COUNT(*) as count, "
        "SUM(CASE WHEN is_reaches_base THEN 1 ELSE 0 END) as reaches_base "
        "FROM raw.plays GROUP BY event_code ORDER BY count DESC LIMIT 15",
        conn,
    )
df_events

# %% [markdown]
# ## 3. Generate Labels

# %%
from sandy.labels.runner import run_labels

label_stats = run_labels(engine)
print(f"Labels generated: {label_stats.rows_written} rows from {label_stats.games_processed} games")
print(f"Elapsed: {label_stats.elapsed_seconds}s")

# %%
# Look at the labels
with engine.connect() as conn:
    df_labels = pd.read_sql(
        "SELECT * FROM derived.inning_labels ORDER BY game_pk, team_code, inning_number LIMIT 50",
        conn,
    )
print(f"Total label rows: {len(df_labels)}")
print(f"Base rate (reached_base=True): {df_labels['reached_base'].mean():.1%}")
df_labels.head(20)

# %%
# Base rate by inning
with engine.connect() as conn:
    df_by_inning = pd.read_sql(
        "SELECT inning_number, "
        "COUNT(*) as total, "
        "SUM(CASE WHEN reached_base THEN 1 ELSE 0 END) as reached, "
        "ROUND(AVG(CASE WHEN reached_base THEN 1.0 ELSE 0.0 END)::numeric, 3) as rate "
        "FROM derived.inning_labels GROUP BY inning_number ORDER BY inning_number",
        conn,
    )
df_by_inning

# %% [markdown]
# ## 4. Build Features

# %%
from sandy.features.runner import run_features

feature_stats = run_features(engine)
print(f"Features built: {feature_stats.rows_written} rows, {feature_stats.rows_omitted} omitted")
print(f"Elapsed: {feature_stats.elapsed_seconds}s")

# %%
# Look at the features
from sandy.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

with engine.connect() as conn:
    df_features = pd.read_sql(
        f"SELECT game_pk, team_code, inning_number, "
        f"{', '.join(FEATURE_NAMES)} "
        f"FROM derived.inning_features "
        f"WHERE feature_schema_version = {FEATURE_SCHEMA_VERSION} "
        f"ORDER BY game_pk, team_code, inning_number LIMIT 50",
        conn,
    )
print(f"Feature schema version: {FEATURE_SCHEMA_VERSION}")
print(f"Features per row: {len(FEATURE_NAMES)}")
df_features.head(10)

# %%
# Feature distributions
df_features[FEATURE_NAMES].describe().T

# %% [markdown]
# ## 5. Train the Model

# %%
from sandy.train.trainer import train_model
from sandy.train.artifact import save_artifact, load_artifact

with get_connection(engine) as conn:
    artifact = train_model(conn, seed=42)

print(f"Model trained successfully!")
print(f"  Training window: {artifact.training_window_start} to {artifact.training_window_end}")
print(f"  Features: {len(artifact.feature_names)} (schema v{artifact.feature_schema_version})")
print(f"  Best iteration: {artifact.model.best_iteration}")

# %%
# Save the model
from pathlib import Path

model_path = cfg.model.path
save_artifact(artifact, model_path)
print(f"Model saved to: {model_path}")

# %% [markdown]
# ## 6. Feature Importance

# %%
# Built-in LightGBM feature importance (gain)
importance = artifact.model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({
    "feature": artifact.feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

print("Top features by gain:")
feat_imp

# %%
# Visualize (if matplotlib available)
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp_sorted = feat_imp.sort_values("importance", ascending=True)
    ax.barh(feat_imp_sorted["feature"], feat_imp_sorted["importance"])
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Sandy Phase 1 — Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.show()
    print("Saved to feature_importance.png")
except ImportError:
    print("matplotlib not installed — skipping plot")

# %% [markdown]
# ## 7. Make a Prediction

# %%
from sandy.predict.predictor import predict, predict_from_features
from sandy.features.builder import build_feature_vector
import json

# Example: Will the Mariners reach base in inning 3 vs the Dodgers?
# (Uses whatever starter is in the DB)
with engine.connect() as conn:
    # Find a real starter from the data
    starter_row = conn.execute(text(
        "SELECT p.player_id, p.full_name FROM raw.players p "
        "JOIN raw.pitcher_game_stats pgs ON pgs.pitcher_id = p.player_id "
        "WHERE pgs.is_starter = true "
        "GROUP BY p.player_id, p.full_name "
        "ORDER BY COUNT(*) DESC LIMIT 1"
    )).fetchone()

if starter_row:
    starter_name = starter_row[1]
    print(f"Using starter: {starter_name}")

    result = predict(
        team="SEA",
        opp="LAD",
        inning=3,
        starter=starter_name,
        config=cfg,
    )

    print(f"\n{'='*50}")
    print(f"Prediction: SEA reaches base in inning 3 vs LAD")
    print(f"  Starter: {starter_name}")
    print(f"  Probability: {result.probability:.1%}")
    print(f"\n  Top contributing features:")
    for feat in result.top_features:
        direction = "↑" if feat.contribution > 0 else "↓"
        print(f"    {direction} {feat.name}: {feat.contribution:+.4f}")
    print(f"\nJSON output:")
    print(json.dumps(json.loads(result.to_json()), indent=2))
else:
    print("No starters found in DB yet — wait for more backfill data")

# %% [markdown]
# ## 8. Model Evaluation Summary

# %%
# Load the training frame and check validation metrics
with engine.connect() as conn:
    total_labels = conn.execute(text("SELECT COUNT(*) FROM derived.inning_labels")).scalar()
    total_features = conn.execute(text(
        f"SELECT COUNT(*) FROM derived.inning_features WHERE feature_schema_version = {FEATURE_SCHEMA_VERSION}"
    )).scalar()
    total_joined = conn.execute(text(f"""
        SELECT COUNT(*)
        FROM derived.inning_features f
        JOIN derived.inning_labels l
            ON l.game_pk = f.game_pk AND l.team_code = f.team_code AND l.inning_number = f.inning_number
        WHERE f.feature_schema_version = {FEATURE_SCHEMA_VERSION}
    """)).scalar()

print(f"Pipeline summary:")
print(f"  Labels:           {total_labels:,}")
print(f"  Features (v{FEATURE_SCHEMA_VERSION}):    {total_features:,}")
print(f"  Training rows:    {total_joined:,}")
print(f"  Model features:   {len(artifact.feature_names)}")
print(f"  Training window:  {artifact.training_window_start} → {artifact.training_window_end}")

# %% [markdown]
# ---
# ## Next Steps
#
# - Wait for full backfill to complete (7,424 games)
# - Retrain with full data for better accuracy
# - Phase 1.5: Add game winner + total runs prediction targets
# - Phase 2: Live game state integration
