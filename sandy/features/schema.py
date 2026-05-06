"""Feature schema constants for Sandy.

Feature set evolution:
  v1 (original 12): pitcher stats, lineup spots, park, trailing-15
  v2 (17 features): + within-game momentum, season team stats
  v3 (20 features): + individual batter season OBP for the 3 batters due up

At prediction time when the exact lineup is unknown, batter OBP features
fall back to the team season OBP as a reasonable approximation.

Bumping FEATURE_SCHEMA_VERSION invalidates existing inning_features rows and
model artifacts — enforced at artifact load time (requirement 7.3).

Requirements: 5.2, 5.5
"""
from __future__ import annotations

#: Increment this when the feature set changes in a way that invalidates
#: existing derived.inning_features rows or saved model artifacts.
FEATURE_SCHEMA_VERSION: int = 3

#: Ordered list of feature names. Order is significant — it must match the
#: column order used when building the numpy array for model inference.
FEATURE_NAMES: list[str] = [
    # --- Opposing pitcher quality ---
    "opp_starter_era",            # opposing starter season ERA (before game_date)
    "opp_starter_whip",           # opposing starter season WHIP
    "opp_starter_k9",             # opposing starter season K/9
    "opp_starter_pitches_before", # pitches thrown by starter before target inning

    # --- Lineup context ---
    "lineup_spot_1",              # batting order spot of 1st batter due up (1-9)
    "lineup_spot_2",              # batting order spot of 2nd batter due up (1-9)
    "lineup_spot_3",              # batting order spot of 3rd batter due up (1-9)

    # --- Individual batter quality (season OBP before game_date) ---
    "lineup_spot1_season_obp",    # season OBP of batter due up 1st (fallback: team avg)
    "lineup_spot2_season_obp",    # season OBP of batter due up 2nd (fallback: team avg)
    "lineup_spot3_season_obp",    # season OBP of batter due up 3rd (fallback: team avg)

    # --- Game context ---
    "is_home",                    # 1 if batting team is home, 0 if away
    "ballpark_id",                # venue_id from raw.games
    "inning_number_feat",         # inning number (explicit copy, decoupled from PK)

    # --- Cross-game team offensive form (trailing 15 games) ---
    "trailing15_rpg",             # batting team runs/game over trailing 15 games
    "trailing15_obp",             # batting team OBP over trailing 15 games

    # --- Within-game momentum/context ---
    "prev_inning_reached_base",   # 1 if team reached base in the previous inning
    "innings_reached_so_far",     # count of prior innings this game with a baserunner
    "consecutive_reach_streak",   # current consecutive inning streak with baserunner

    # --- Season-level team offensive baseline ---
    "team_season_obp",            # team OBP this season before game_date
    "team_season_rpg",            # team runs/game this season before game_date
]

assert len(FEATURE_NAMES) == 20, "FEATURE_NAMES must have exactly 20 entries"

# ---------------------------------------------------------------------------
# Phase 1.5: Game-level feature schema (for game_winner and runs targets)
# ---------------------------------------------------------------------------

GAME_FEATURE_SCHEMA_VERSION: int = 1

GAME_FEATURE_NAMES: list[str] = [
    "home_starter_era",       # home starting pitcher season ERA
    "home_starter_whip",      # home starting pitcher season WHIP
    "away_starter_era",       # away starting pitcher season ERA
    "away_starter_whip",      # away starting pitcher season WHIP
    "home_trailing15_rpg",    # home team runs/game over trailing 15 games
    "away_trailing15_rpg",    # away team runs/game over trailing 15 games
    "home_season_obp",        # home team season OBP
    "away_season_obp",        # away team season OBP
    "ballpark_id",            # venue ID
    "is_home",                # 1 if predicting for home team, 0 for away
]

assert len(GAME_FEATURE_NAMES) == 10, "GAME_FEATURE_NAMES must have exactly 10 entries"

__all__ = ["FEATURE_NAMES", "FEATURE_SCHEMA_VERSION", "GAME_FEATURE_NAMES", "GAME_FEATURE_SCHEMA_VERSION"]
