"""Feature vector builder — pure function over DB state.

Task 8.2: build_feature_vector() reads raw.* tables and returns a
FeatureVector dataclass. No writes, no network calls.

Leakage prevention (requirement 5.1):
- cutoff_ts = MIN(start_time_utc) of plays in the target half-inning.
  Every same-game aggregate is bounded to start_time_utc < cutoff_ts.
- Cross-game trailing-15 aggregates use game_date < :game_date.
- For prediction (game_pk=None), same-game features use empty history:
  pitches_before=0, lineup_spots=(1,2,3).

Requirements: 5.1, 5.2, 5.3
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sandy.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger
from sandy.schemas import FeatureVector

logger = get_logger("features.builder")


def build_feature_vector(
    conn: Connection,
    team_code: str,
    opp_team_code: str,
    inning_number: int,
    opp_starter_id: int,
    game_date: date,
    game_pk: int | None = None,
    as_of: datetime | None = None,
) -> FeatureVector:
    """Build a FeatureVector for a (team, inning) row.

    Parameters
    ----------
    conn:           Active DB connection (read-only usage).
    team_code:      The batting team's 3-letter code.
    opp_team_code:  The opposing (pitching) team's code.
    inning_number:  Target inning (1-9).
    opp_starter_id: player_id of the opposing starting pitcher.
    game_date:      Date of the game (used for trailing-window cutoffs).
    game_pk:        If set, build the historical vector for that game row.
                    If None, build a hypothetical prediction vector.
    as_of:          Optional datetime ceiling for same-game features when
                    game_pk is None (prediction path).

    Returns
    -------
    FeatureVector with all 12 features populated (None values allowed for
    features that cannot be computed — the runner omits those rows).
    """
    # ------------------------------------------------------------------
    # 1. Determine cutoff_ts for same-game leakage prevention
    # ------------------------------------------------------------------
    cutoff_ts: datetime | None = None

    if game_pk is not None:
        # Historical path: cutoff = first pitch of the target half-inning
        half = "top" if _is_away_team(conn, game_pk, team_code) else "bottom"
        cutoff_ts = _get_cutoff_ts(conn, game_pk, inning_number, half)
        # If no timestamps available, fall back to as_of or None
        if cutoff_ts is None and as_of is not None:
            cutoff_ts = as_of

    # ------------------------------------------------------------------
    # 2. Opposing starter season stats (cross-game, before game_date)
    # ------------------------------------------------------------------
    starter_stats = _get_starter_season_stats(
        conn, opp_starter_id, game_date
    )

    era = starter_stats.get("era")
    whip = starter_stats.get("whip")
    k9 = starter_stats.get("k9")

    # ------------------------------------------------------------------
    # 3. Same-game features
    # ------------------------------------------------------------------
    if game_pk is not None and cutoff_ts is not None:
        pitches_before = _get_starter_pitches_before(
            conn, game_pk, opp_starter_id, inning_number, cutoff_ts
        )
        lineup_spots = _get_lineup_spots(
            conn, game_pk, team_code, inning_number, cutoff_ts
        )
    else:
        # Prediction path: no same-game history available
        pitches_before = 0
        lineup_spots = (1, 2, 3)

    # ------------------------------------------------------------------
    # 4. Home/away and ballpark
    # ------------------------------------------------------------------
    is_home: bool | None = None
    ballpark_id: int | None = None

    if game_pk is not None:
        game_info = _get_game_info(conn, game_pk)
        if game_info:
            is_home = game_info["home_team_code"].strip() == team_code.strip()
            ballpark_id = game_info["venue_id"]
    else:
        # For prediction we don't know home/away without a game_pk
        is_home = None
        ballpark_id = None

    # ------------------------------------------------------------------
    # 6. Individual batter season OBP (new in schema v3)
    # ------------------------------------------------------------------
    # Resolve the actual batter IDs for the 3 due-up spots
    if game_pk is not None and cutoff_ts is not None:
        batter_ids = _get_due_up_batter_ids(
            conn, game_pk, team_code, inning_number, cutoff_ts
        )
    else:
        batter_ids = (None, None, None)

    # Compute season OBP for each batter; fall back to team_season_obp if unknown
    team_season_stats_early = _get_team_season_stats(conn, team_code, game_date)
    fallback_obp = team_season_stats_early.get("obp") or 0.0

    batter_obps = []
    for batter_id in batter_ids:
        if batter_id is not None:
            obp = _get_batter_season_obp(conn, batter_id, game_date)
            batter_obps.append(obp if obp is not None else fallback_obp)
        else:
            batter_obps.append(fallback_obp)

    # ------------------------------------------------------------------
    # 7. Trailing-15 team offensive stats
    # ------------------------------------------------------------------
    trailing = _get_trailing15_stats(conn, team_code, game_date)
    trailing15_rpg = trailing.get("rpg")
    trailing15_obp = trailing.get("obp")

    # ------------------------------------------------------------------
    # 6. Within-game momentum features (new in schema v2)
    # ------------------------------------------------------------------
    if game_pk is not None and cutoff_ts is not None:
        momentum = _get_within_game_context(
            conn, game_pk, team_code, inning_number, cutoff_ts
        )
    else:
        momentum = {"prev_reached": 0, "innings_reached": 0, "streak": 0}

    # ------------------------------------------------------------------
    # 7. Season-level team offensive baseline (new in schema v2)
    # ------------------------------------------------------------------
    season_stats = _get_team_season_stats(conn, team_code, game_date)
    team_season_obp = season_stats.get("obp")
    team_season_rpg = season_stats.get("rpg")

    # ------------------------------------------------------------------
    # 8. Assemble values dict
    # ------------------------------------------------------------------
    values: dict[str, float | int | bool] = {
        "opp_starter_era": era if era is not None else 0.0,
        "opp_starter_whip": whip if whip is not None else 0.0,
        "opp_starter_k9": k9 if k9 is not None else 0.0,
        "opp_starter_pitches_before": pitches_before,
        "lineup_spot_1": lineup_spots[0],
        "lineup_spot_2": lineup_spots[1],
        "lineup_spot_3": lineup_spots[2],
        "lineup_spot1_season_obp": batter_obps[0],
        "lineup_spot2_season_obp": batter_obps[1],
        "lineup_spot3_season_obp": batter_obps[2],
        "is_home": int(is_home) if is_home is not None else 0,
        "ballpark_id": ballpark_id if ballpark_id is not None else 0,
        "inning_number_feat": inning_number,
        "trailing15_rpg": trailing15_rpg if trailing15_rpg is not None else 0.0,
        "trailing15_obp": trailing15_obp if trailing15_obp is not None else 0.0,
        "prev_inning_reached_base": momentum["prev_reached"],
        "innings_reached_so_far": momentum["innings_reached"],
        "consecutive_reach_streak": momentum["streak"],
        "team_season_obp": team_season_obp if team_season_obp is not None else 0.0,
        "team_season_rpg": team_season_rpg if team_season_rpg is not None else 0.0,
    }

    return FeatureVector(
        game_pk=game_pk,
        team_code=team_code,
        inning_number=inning_number,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        values=values,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_away_team(conn: Connection, game_pk: int, team_code: str) -> bool:
    """Return True if team_code is the away team for this game."""
    row = conn.execute(
        text("SELECT away_team_code FROM raw.games WHERE game_pk = :pk"),
        {"pk": game_pk},
    ).fetchone()
    if row is None:
        return True  # default to top/away if unknown
    return row[0].strip() == team_code.strip()


def _get_cutoff_ts(
    conn: Connection,
    game_pk: int,
    inning_number: int,
    half_inning: str,
) -> datetime | None:
    """Return MIN(start_time_utc) for plays in the target half-inning."""
    row = conn.execute(
        text("""
            SELECT MIN(start_time_utc)
            FROM raw.plays
            WHERE game_pk = :game_pk
              AND inning = :inning
              AND half_inning = :half
              AND start_time_utc IS NOT NULL
        """),
        {"game_pk": game_pk, "inning": inning_number, "half": half_inning},
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def _get_starter_season_stats(
    conn: Connection,
    pitcher_id: int,
    game_date: date,
) -> dict[str, float | None]:
    """Compute ERA, WHIP, K/9 for a pitcher over games before game_date."""
    row = conn.execute(
        text("""
            SELECT
                SUM(pgs.outs_recorded)   AS total_outs,
                SUM(pgs.runs_allowed)    AS total_runs,
                SUM(pgs.walks)           AS total_walks,
                SUM(pgs.hits_allowed)    AS total_hits,
                SUM(pgs.strikeouts)      AS total_ks
            FROM raw.pitcher_game_stats pgs
            JOIN raw.games g ON g.game_pk = pgs.game_pk
            WHERE pgs.pitcher_id = :pitcher_id
              AND g.game_date < :game_date
              AND g.status = 'Final'
              AND EXTRACT(YEAR FROM g.game_date) = :season
        """),
        {
            "pitcher_id": pitcher_id,
            "game_date": game_date,
            "season": game_date.year,
        },
    ).fetchone()

    if row is None or row[0] is None or row[0] == 0:
        return {"era": None, "whip": None, "k9": None}

    total_outs = float(row[0])
    innings = total_outs / 3.0
    if innings == 0:
        return {"era": None, "whip": None, "k9": None}

    era = 9.0 * float(row[1] or 0) / innings
    whip = (float(row[2] or 0) + float(row[3] or 0)) / innings
    k9 = 9.0 * float(row[4] or 0) / innings

    return {"era": era, "whip": whip, "k9": k9}


def _get_starter_pitches_before(
    conn: Connection,
    game_pk: int,
    pitcher_id: int,
    inning_number: int,
    cutoff_ts: datetime,
) -> int:
    """Count plate appearances by the starter before the target inning cutoff."""
    row = conn.execute(
        text("""
            SELECT COUNT(*)
            FROM raw.plays
            WHERE game_pk = :game_pk
              AND pitcher_id = :pitcher_id
              AND (
                  inning < :inning
                  OR (inning = :inning AND start_time_utc < :cutoff_ts)
              )
        """),
        {
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "inning": inning_number,
            "cutoff_ts": cutoff_ts,
        },
    ).fetchone()
    return int(row[0]) if row else 0


def _get_lineup_spots(
    conn: Connection,
    game_pk: int,
    team_code: str,
    inning_number: int,
    cutoff_ts: datetime,
) -> tuple[int, int, int]:
    """Determine the three batting order spots due up in the target inning.

    Finds the last out recorded before cutoff_ts for the batting team,
    then the next three spots in the 1-9 rotation.
    For inning 1 or no prior history, returns (1, 2, 3).
    """
    if inning_number == 1:
        return (1, 2, 3)

    # Find the batting_order of the last plate appearance before cutoff
    row = conn.execute(
        text("""
            SELECT batting_order
            FROM raw.plays
            WHERE game_pk = :game_pk
              AND batting_team_code = :team_code
              AND start_time_utc < :cutoff_ts
              AND batting_order IS NOT NULL
            ORDER BY start_time_utc DESC, at_bat_index DESC
            LIMIT 1
        """),
        {
            "game_pk": game_pk,
            "team_code": team_code,
            "cutoff_ts": cutoff_ts,
        },
    ).fetchone()

    if row is None or row[0] is None:
        return (1, 2, 3)

    last_spot = int(row[0])
    s1 = (last_spot % 9) + 1
    s2 = (s1 % 9) + 1
    s3 = (s2 % 9) + 1
    return (s1, s2, s3)


def _get_game_info(conn: Connection, game_pk: int) -> dict[str, Any] | None:
    """Return home_team_code and venue_id for a game."""
    row = conn.execute(
        text("""
            SELECT home_team_code, venue_id
            FROM raw.games
            WHERE game_pk = :game_pk
        """),
        {"game_pk": game_pk},
    ).fetchone()
    if row is None:
        return None
    return {"home_team_code": row[0], "venue_id": row[1]}


def _get_trailing15_stats(
    conn: Connection,
    team_code: str,
    game_date: date,
) -> dict[str, float | None]:
    """Compute RPG and OBP for the batting team over the 15 most recent games."""
    # Get the 15 most recent Final game_pks for this team before game_date
    rows = conn.execute(
        text("""
            SELECT game_pk
            FROM raw.games
            WHERE (home_team_code = :team OR away_team_code = :team)
              AND game_date < :game_date
              AND status = 'Final'
            ORDER BY game_date DESC, game_pk DESC
            LIMIT 15
        """),
        {"team": team_code, "game_date": game_date},
    ).fetchall()

    if not rows:
        return {"rpg": None, "obp": None}

    game_pks = [r[0] for r in rows]
    pk_list = ",".join(str(pk) for pk in game_pks)

    # Runs scored: count run-scoring events (home_run counts as run + RBI)
    # Simpler: sum runs from linescore via games table
    runs_row = conn.execute(
        text(f"""
            SELECT
                SUM(CASE WHEN home_team_code = :team THEN home_score
                         ELSE away_score END) AS total_runs,
                COUNT(*) AS games_count
            FROM raw.games
            WHERE game_pk IN ({pk_list})
        """),
        {"team": team_code},
    ).fetchone()

    total_runs = float(runs_row[0] or 0)
    games_count = int(runs_row[1] or 1)
    rpg = total_runs / games_count if games_count > 0 else None

    # OBP: (H + BB + HBP) / (AB + BB + HBP + SF)
    # Approximate from plays: hits + walks + hbp / total plate appearances
    obp_row = conn.execute(
        text(f"""
            SELECT
                SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                             'walk','hit_by_pitch') THEN 1 ELSE 0 END) AS on_base,
                COUNT(*) AS plate_appearances
            FROM raw.plays
            WHERE game_pk IN ({pk_list})
              AND batting_team_code = :team
        """),
        {"team": team_code},
    ).fetchone()

    on_base = int(obp_row[0] or 0)
    pa = int(obp_row[1] or 0)
    obp = on_base / pa if pa > 0 else None

    return {"rpg": rpg, "obp": obp}


def _get_within_game_context(
    conn: Connection,
    game_pk: int,
    team_code: str,
    inning_number: int,
    cutoff_ts: datetime,
) -> dict[str, int]:
    """Compute within-game momentum features for the batting team.

    Uses only innings strictly before the target inning (leakage-safe).

    Returns:
        prev_reached:    1 if team reached base in inning_number-1, else 0
        innings_reached: count of prior innings this game where team reached base
        streak:          current consecutive inning streak with a baserunner
    """
    if inning_number <= 1:
        return {"prev_reached": 0, "innings_reached": 0, "streak": 0}

    # Get is_reaches_base for each prior inning (grouped by inning number)
    rows = conn.execute(
        text("""
            SELECT inning, BOOL_OR(is_reaches_base) AS reached
            FROM raw.plays
            WHERE game_pk = :game_pk
              AND batting_team_code = :team_code
              AND inning < :inning_number
              AND (start_time_utc IS NULL OR start_time_utc < :cutoff_ts)
            GROUP BY inning
            ORDER BY inning ASC
        """),
        {
            "game_pk": game_pk,
            "team_code": team_code,
            "inning_number": inning_number,
            "cutoff_ts": cutoff_ts,
        },
    ).fetchall()

    if not rows:
        return {"prev_reached": 0, "innings_reached": 0, "streak": 0}

    # Build a dict of inning -> reached
    inning_results: dict[int, bool] = {int(r[0]): bool(r[1]) for r in rows}

    # prev_inning_reached_base
    prev_reached = int(inning_results.get(inning_number - 1, False))

    # innings_reached_so_far
    innings_reached = sum(1 for v in inning_results.values() if v)

    # consecutive_reach_streak — count backwards from inning_number-1
    streak = 0
    for inn in range(inning_number - 1, 0, -1):
        if inning_results.get(inn, False):
            streak += 1
        else:
            break

    return {
        "prev_reached": prev_reached,
        "innings_reached": innings_reached,
        "streak": streak,
    }


def _get_team_season_stats(
    conn: Connection,
    team_code: str,
    game_date: date,
) -> dict[str, float | None]:
    """Compute team OBP and RPG for the current season before game_date."""
    season = game_date.year

    rows = conn.execute(
        text("""
            SELECT game_pk
            FROM raw.games
            WHERE (home_team_code = :team OR away_team_code = :team)
              AND game_date < :game_date
              AND status = 'Final'
              AND EXTRACT(YEAR FROM game_date) = :season
        """),
        {"team": team_code, "game_date": game_date, "season": season},
    ).fetchall()

    if not rows:
        return {"obp": None, "rpg": None}

    game_pks = [r[0] for r in rows]
    pk_list = ",".join(str(pk) for pk in game_pks)

    runs_row = conn.execute(
        text(f"""
            SELECT
                SUM(CASE WHEN home_team_code = :team THEN home_score
                         ELSE away_score END) AS total_runs,
                COUNT(*) AS games_count
            FROM raw.games
            WHERE game_pk IN ({pk_list})
        """),
        {"team": team_code},
    ).fetchone()

    total_runs = float(runs_row[0] or 0)
    games_count = int(runs_row[1] or 1)
    rpg = total_runs / games_count if games_count > 0 else None

    obp_row = conn.execute(
        text(f"""
            SELECT
                SUM(CASE WHEN event_code IN ('single','double','triple','home_run',
                                             'walk','hit_by_pitch') THEN 1 ELSE 0 END),
                COUNT(*)
            FROM raw.plays
            WHERE game_pk IN ({pk_list})
              AND batting_team_code = :team
        """),
        {"team": team_code},
    ).fetchone()

    on_base = int(obp_row[0] or 0)
    pa = int(obp_row[1] or 0)
    obp = on_base / pa if pa > 0 else None

    return {"obp": obp, "rpg": rpg}


def _get_due_up_batter_ids(
    conn: Connection,
    game_pk: int,
    team_code: str,
    inning_number: int,
    cutoff_ts: datetime,
) -> tuple[int | None, int | None, int | None]:
    """Resolve the batter_ids of the 3 batters due up in the target inning.

    Uses the same lineup-spot logic as _get_lineup_spots but returns the
    actual player IDs instead of order numbers.
    For inning 1 or no prior history, returns the first 3 batters seen.
    """
    if inning_number == 1:
        # Get the first 3 batters from the lineup (batting_order 1,2,3)
        rows = conn.execute(
            text("""
                SELECT DISTINCT ON (batting_order) batter_id
                FROM raw.plays
                WHERE game_pk = :game_pk
                  AND batting_team_code = :team_code
                  AND batting_order IS NOT NULL
                  AND batting_order <= 3
                ORDER BY batting_order, at_bat_index
            """),
            {"game_pk": game_pk, "team_code": team_code},
        ).fetchall()
        ids = [r[0] for r in rows]
        while len(ids) < 3:
            ids.append(None)
        return (ids[0], ids[1], ids[2])

    # Find the last batter who had a PA before cutoff
    last_row = conn.execute(
        text("""
            SELECT batting_order, batter_id
            FROM raw.plays
            WHERE game_pk = :game_pk
              AND batting_team_code = :team_code
              AND start_time_utc < :cutoff_ts
              AND batting_order IS NOT NULL
            ORDER BY start_time_utc DESC, at_bat_index DESC
            LIMIT 1
        """),
        {"game_pk": game_pk, "team_code": team_code, "cutoff_ts": cutoff_ts},
    ).fetchone()

    if last_row is None:
        return (None, None, None)

    last_spot = int(last_row[0])
    # Next 3 spots in the 1-9 rotation
    spots = [
        (last_spot % 9) + 1,
        ((last_spot % 9 + 1) % 9) + 1,
        ((last_spot % 9 + 2) % 9) + 1,
    ]

    # Resolve batter_id for each spot from the game's lineup
    ids = []
    for spot in spots:
        row = conn.execute(
            text("""
                SELECT batter_id
                FROM raw.plays
                WHERE game_pk = :game_pk
                  AND batting_team_code = :team_code
                  AND batting_order = :spot
                  AND batter_id IS NOT NULL
                ORDER BY at_bat_index DESC
                LIMIT 1
            """),
            {"game_pk": game_pk, "team_code": team_code, "spot": spot},
        ).fetchone()
        ids.append(row[0] if row else None)

    return (ids[0], ids[1], ids[2])


def _get_batter_season_obp(
    conn: Connection,
    batter_id: int,
    game_date: date,
) -> float | None:
    """Compute a batter's season OBP from raw.plays before game_date.

    OBP = (H + BB + HBP) / total plate appearances
    Only counts games in the same season before game_date.
    Returns None if the batter has no plate appearances this season.
    """
    season = game_date.year
    row = conn.execute(
        text("""
            SELECT
                SUM(CASE WHEN p.event_code IN ('single','double','triple','home_run',
                                               'walk','hit_by_pitch') THEN 1 ELSE 0 END) AS on_base,
                COUNT(*) AS plate_appearances
            FROM raw.plays p
            JOIN raw.games g ON g.game_pk = p.game_pk
            WHERE p.batter_id = :batter_id
              AND g.game_date < :game_date
              AND g.status = 'Final'
              AND EXTRACT(YEAR FROM g.game_date) = :season
        """),
        {"batter_id": batter_id, "game_date": game_date, "season": season},
    ).fetchone()

    if row is None or row[1] is None or int(row[1]) == 0:
        return None

    return int(row[0] or 0) / int(row[1])


__all__ = ["build_feature_vector"]
