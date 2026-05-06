"""Feature schema constants for Sandy.

Task 8.1: Defines FEATURE_SCHEMA_VERSION and FEATURE_NAMES.

Bumping FEATURE_SCHEMA_VERSION invalidates existing inning_features rows and
model artifacts — enforced at artifact load time (requirement 7.3).

Requirements: 5.2, 5.5
"""
from __future__ import annotations

#: Increment this when the feature set changes in a way that invalidates
#: existing derived.inning_features rows or saved model artifacts.
FEATURE_SCHEMA_VERSION: int = 1

#: Ordered list of feature names. Order is significant — it must match the
#: column order used when building the numpy array for model inference.
FEATURE_NAMES: list[str] = [
    "opp_starter_era",           # opposing starter season ERA (before game_date)
    "opp_starter_whip",          # opposing starter season WHIP
    "opp_starter_k9",            # opposing starter season K/9
    "opp_starter_pitches_before",# pitches thrown by starter before target inning
    "lineup_spot_1",             # batting order spot of 1st batter due up (1-9)
    "lineup_spot_2",             # batting order spot of 2nd batter due up (1-9)
    "lineup_spot_3",             # batting order spot of 3rd batter due up (1-9)
    "is_home",                   # 1 if batting team is home, 0 if away
    "ballpark_id",               # venue_id from raw.games
    "inning_number_feat",        # inning number (explicit copy, decoupled from PK)
    "trailing15_rpg",            # batting team runs/game over trailing 15 games
    "trailing15_obp",            # batting team OBP over trailing 15 games
]

assert len(FEATURE_NAMES) == 12, "FEATURE_NAMES must have exactly 12 entries"

__all__ = ["FEATURE_NAMES", "FEATURE_SCHEMA_VERSION"]
