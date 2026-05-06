"""Build game features and train all 3 models with full data."""
from sandy.config import load_config
from sandy.db import create_engine, get_connection
from sandy.features.runner import run_game_features
from sandy.train.trainer import train_model, train_game_winner_model, train_runs_model
from sandy.train.artifact import save_artifact
from sandy.logging import configure_logging

configure_logging("INFO")
cfg = load_config()
engine = create_engine(cfg)

print("Building game-level features...")
stats = run_game_features(engine)
print(f"  Done: {stats.rows_written} rows, {stats.elapsed_seconds}s")

print("Training reached_base model...")
with engine.begin() as conn:
    art = train_model(conn, seed=42)
save_artifact(art, cfg.model.artifact_path("reached_base"))
print("  Saved reached_base model")

print("Training game_winner model...")
with engine.begin() as conn:
    art = train_game_winner_model(conn, seed=42)
save_artifact(art, cfg.model.artifact_path("game_winner"))
print("  Saved game_winner model")

print("Training runs model...")
with engine.begin() as conn:
    art = train_runs_model(conn, seed=42)
save_artifact(art, cfg.model.artifact_path("runs"))
print("  Saved runs model")

print("All 3 models trained!")
