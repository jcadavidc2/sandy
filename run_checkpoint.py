"""Run checkpoint 9+14: features for first 200 games, train, predict."""
from sqlalchemy import text
from sandy.config import load_config
from sandy.db import create_engine, get_connection
from sandy.features.runner import run_features
from sandy.train.trainer import train_model
from sandy.train.artifact import save_artifact
from sandy.predict.predictor import predict
from sandy.logging import configure_logging
import time

configure_logging("INFO")
cfg = load_config()
engine = create_engine(cfg)

# Build features for first 200 games only (fast enough for checkpoint)
print("Building features for first 200 games...")
t0 = time.monotonic()
with engine.connect() as conn:
    pks = conn.execute(text(
        "SELECT game_pk FROM derived.inning_labels GROUP BY game_pk ORDER BY game_pk LIMIT 200"
    )).fetchall()

for row in pks:
    run_features(engine, game_pk=row[0])

elapsed = time.monotonic() - t0
print(f"Features built for {len(pks)} games in {elapsed:.0f}s")

# Check how many training rows we have
with engine.connect() as conn:
    count = conn.execute(text("""
        SELECT COUNT(*) FROM derived.inning_features f
        JOIN derived.inning_labels l
            ON l.game_pk = f.game_pk AND l.team_code = f.team_code AND l.inning_number = f.inning_number
        WHERE f.feature_schema_version = 3
    """)).scalar()
print(f"Training rows available: {count}")

# Train
print("\nTraining model...")
with get_connection(engine) as conn:
    artifact = train_model(conn, seed=42)
save_artifact(artifact, cfg.model.path)
print(f"Model saved to {cfg.model.path}")

# Predict
print("\nMaking prediction...")
with engine.connect() as conn:
    starter = conn.execute(text(
        "SELECT full_name FROM raw.players p "
        "JOIN raw.pitcher_game_stats pgs ON pgs.pitcher_id = p.player_id "
        "WHERE pgs.is_starter = true "
        "GROUP BY full_name ORDER BY COUNT(*) DESC LIMIT 1"
    )).scalar()

result = predict(team="SEA", opp="LAD", inning=3, starter=starter, config=cfg)
print(f"\nPrediction: SEA reaches base in inning 3 vs LAD")
print(f"  Starter: {starter}")
print(f"  Probability: {result.probability:.1%}")
print(f"  Top features:")
for f in result.top_features:
    print(f"    {'↑' if f.contribution > 0 else '↓'} {f.name}: {f.contribution:+.4f}")
print(f"\nCheckpoint PASSED!")
