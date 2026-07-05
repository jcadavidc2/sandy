"""Walk-forward historical backtest for the MLB over/under vertical.

"Pretend it's last year": replay 2023-04-01 → 2026-05-05 in 14-day blocks so the
meta-models can train on ~3 seasons of scored predictions instead of 2 months —
the same treatment the mls/nhl/nba/soccer verticals got (see sandy/mls/backtest.py).

Per block:
  1. Refit the runs model AND the volatility (σ) model in memory using ONLY
     games with game_date < block_start (production artifacts on disk are
     NEVER touched).
  2. For each finished regular-season game in the block, build the 10-feature
     vector as-of that game's date via the same build_game_feature_vector()
     the live predictor uses (every sub-query filters game_date < :game_date,
     so nothing after the game leaks in).
  3. Compute the 7 line probabilities with the exact live function
     (compute_over_under_probabilities) and persist with is_backtest=TRUE and
     a deterministic predicted_at_utc (game_date 12:00 UTC, clamped to 2h
     before first pitch for the handful of Seoul/Tokyo morning games).
  4. Reconcile actual outcomes from raw.games scores via the live reconciler.

HONESTY NOTES (differences vs the live morning pipeline, documented on purpose):
  * Starters are the ACTUAL starters recorded in raw.games
    (home_starter_id/away_starter_id), not the announced probables the live
    pipeline sees in the morning — a slight optimism vs live (late scratches
    are invisible here), so pitcher_fallback is effectively always FALSE.
  * σ comes from a volatility model refit as-of each block (the live one on
    disk was trained on future data relative to these games).

The 847 live daily rows (2026-05-06 →) are SACRED: this module asserts their
count and checksum are identical before/after, and its upsert can only ever
touch rows that are already is_backtest=TRUE.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine, get_connection
from sandy.features.game_builder import build_game_feature_vector
from sandy.features.schema import GAME_FEATURE_NAMES, GAME_FEATURE_SCHEMA_VERSION
from sandy.logging import get_logger
from sandy.over_under.predictor import compute_over_under_probabilities
from sandy.over_under.reconciler import reconcile_over_under
from sandy.over_under.schemas import STANDARD_THRESHOLDS
from sandy.over_under.volatility import DEFAULT_SIGMA, train_volatility_model
from sandy.schemas import GameFeatureVector, ModelArtifact

logger = get_logger("over_under.backtest")

REFIT_DAYS = 14
# Window: first predictable date (a full 2022 season of history exists) up to
# the day BEFORE the live daily predictions begin (2026-05-06). Never overlap.
DEFAULT_START = date(2023, 4, 1)
DEFAULT_END = date(2026, 5, 5)
# Training data may reach back to the earliest ingested game (2022-04-07).
TRAINING_FLOOR = date(2020, 1, 1)
PROGRESS_EVERY = 100


class BacktestSafetyError(RuntimeError):
    """Raised when a live-row or sanity invariant is violated. Do not commit."""


@dataclass
class BacktestStats:
    predicted: int = 0
    reconciled: int = 0
    already_present: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    blocks_trained: int = 0

    def skip(self, reason: str, n: int = 1) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + n


# ---------------------------------------------------------------------------
# Live-row protection
# ---------------------------------------------------------------------------


def live_snapshot(engine: Engine) -> tuple[int, str, str]:
    """(count, sum of p_over_5_5, md5 checksum) over the NON-backtest rows."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*),
                   COALESCE(ROUND(SUM(p_over_5_5)::numeric, 6), 0)::text,
                   COALESCE(md5(string_agg(
                       id::text || '|' || game_pk::text || '|' || predicted_at_utc::text
                       || '|' || p_over_5_5::text
                       || '|' || COALESCE(actual_total_runs::text, '')
                       || '|' || COALESCE(outcome_filled_at_utc::text, ''),
                       ',' ORDER BY id)), '')
            FROM derived.over_under_outcomes
            WHERE NOT is_backtest
        """)).fetchone()
    return int(row[0]), str(row[1]), str(row[2])


# ---------------------------------------------------------------------------
# Per-block model refits (in memory — production artifacts untouched)
# ---------------------------------------------------------------------------


def _fit_block_models(
    engine: Engine,
    config: Config,
    block_start: date,
    seed: int,
) -> tuple[ModelArtifact | None, ModelArtifact | None]:
    """Fit runs + volatility models on games strictly before *block_start*."""
    from sandy.train.trainer import train_runs_model

    window = (TRAINING_FLOOR, block_start - timedelta(days=1))
    try:
        with get_connection(engine) as conn:
            runs_artifact = train_runs_model(conn, seed=seed, training_window=window)
    except ValueError as exc:
        logger.warning(
            f"Backtest block {block_start}: runs model not trainable ({exc})",
            extra={"component": "over_under.backtest"},
        )
        return None, None

    try:
        vol_artifact = train_volatility_model(
            config, seed=seed, training_window=window, runs_artifact=runs_artifact
        )
    except ValueError as exc:
        logger.warning(
            f"Backtest block {block_start}: volatility model not trainable ({exc}); "
            f"falling back to σ={DEFAULT_SIGMA}",
            extra={"component": "over_under.backtest"},
        )
        vol_artifact = None

    return runs_artifact, vol_artifact


def _predict_runs(artifact: ModelArtifact, values: dict) -> float:
    x = np.array(
        [[float(values.get(name, 0.0)) for name in artifact.feature_names]],
        dtype=np.float64,
    )
    return max(0.0, float(artifact.model.predict(x)[0]))


def _predict_sigma_from_artifact(artifact: ModelArtifact | None, values: dict) -> float:
    """Matchup σ from the block's volatility artifact (mirrors predict_sigma)."""
    if artifact is None:
        return DEFAULT_SIGMA
    x = np.array(
        [[float(values.get(name, 0.0)) for name in GAME_FEATURE_NAMES]],
        dtype=np.float32,
    )
    return float(max(1.0, min(8.0, artifact.model.predict(x)[0])))


def _deterministic_predicted_at(game_date: date, first_pitch_utc: datetime | None) -> datetime:
    """game_date 12:00 UTC — clamped to 2h before first pitch for the few
    international morning games (Seoul/Tokyo series start ~10:00 UTC), so the
    'prediction made before first pitch' invariant always holds. Never now()."""
    predicted_at = datetime.combine(game_date, time(12, 0), tzinfo=timezone.utc)
    if first_pitch_utc is not None:
        if first_pitch_utc.tzinfo is None:
            first_pitch_utc = first_pitch_utc.replace(tzinfo=timezone.utc)
        cutoff = first_pitch_utc - timedelta(hours=2)
        if predicted_at >= cutoff:
            predicted_at = cutoff
    return predicted_at


# ---------------------------------------------------------------------------
# Persistence (backtest rows only — can never touch a live row)
# ---------------------------------------------------------------------------

_INSERT_SQL = text("""
    INSERT INTO derived.over_under_outcomes (
        game_pk, game_date, home_team_code, away_team_code,
        predicted_at_utc,
        p_over_5_5, p_over_6_5, p_over_7_5, p_over_8_5,
        p_over_9_5, p_over_10_5, p_over_11_5,
        feature_vector,
        home_starter_era, away_starter_era, ballpark_id,
        home_trailing15_rpg, away_trailing15_rpg,
        pitcher_fallback,
        home_expected_runs, away_expected_runs, sigma_used,
        is_backtest
    ) VALUES (
        :game_pk, :game_date, :home_team_code, :away_team_code,
        :predicted_at_utc,
        :p_over_5_5, :p_over_6_5, :p_over_7_5, :p_over_8_5,
        :p_over_9_5, :p_over_10_5, :p_over_11_5,
        :feature_vector,
        :home_starter_era, :away_starter_era, :ballpark_id,
        :home_trailing15_rpg, :away_trailing15_rpg,
        :pitcher_fallback,
        :home_expected_runs, :away_expected_runs, :sigma_used,
        TRUE
    )
    ON CONFLICT (game_pk, game_date) DO UPDATE SET
        predicted_at_utc = EXCLUDED.predicted_at_utc,
        p_over_5_5 = EXCLUDED.p_over_5_5,
        p_over_6_5 = EXCLUDED.p_over_6_5,
        p_over_7_5 = EXCLUDED.p_over_7_5,
        p_over_8_5 = EXCLUDED.p_over_8_5,
        p_over_9_5 = EXCLUDED.p_over_9_5,
        p_over_10_5 = EXCLUDED.p_over_10_5,
        p_over_11_5 = EXCLUDED.p_over_11_5,
        feature_vector = EXCLUDED.feature_vector,
        home_starter_era = EXCLUDED.home_starter_era,
        away_starter_era = EXCLUDED.away_starter_era,
        ballpark_id = EXCLUDED.ballpark_id,
        home_trailing15_rpg = EXCLUDED.home_trailing15_rpg,
        away_trailing15_rpg = EXCLUDED.away_trailing15_rpg,
        pitcher_fallback = EXCLUDED.pitcher_fallback,
        home_expected_runs = EXCLUDED.home_expected_runs,
        away_expected_runs = EXCLUDED.away_expected_runs,
        sigma_used = EXCLUDED.sigma_used,
        -- reset outcome fields so the reconciler refills them for the new probs
        actual_total_runs = NULL,
        outcome_filled_at_utc = NULL
    -- guard: a conflicting row may only be overwritten if it is itself a
    -- backtest row. Live rows can never conflict (disjoint game_pk ranges),
    -- but this makes clobbering one structurally impossible.
    WHERE derived.over_under_outcomes.is_backtest
""")


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------


def run_backtest(
    config: Config | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    seed: int = 42,
) -> dict:
    """Walk forward from *start* to *end* in REFIT_DAYS blocks.

    Returns a dict with predicted / skipped / reconciled counts.
    Raises BacktestSafetyError if the live rows changed underneath us.
    """
    cfg = config or load_config()
    engine = create_engine(cfg)
    start = start or DEFAULT_START
    end = end or DEFAULT_END
    if end > DEFAULT_END:
        raise BacktestSafetyError(
            f"Backtest end {end} would enter the live prediction window "
            f"(live rows begin {DEFAULT_END + timedelta(days=1)})."
        )

    live_before = live_snapshot(engine)
    logger.info(
        f"Backtest starting {start} → {end}; live rows snapshot: "
        f"count={live_before[0]} sum_p55={live_before[1]}",
        extra={"component": "over_under.backtest"},
    )

    stats = BacktestStats()
    block_start = start

    while block_start <= end:
        block_end = min(block_start + timedelta(days=REFIT_DAYS - 1), end)

        with engine.connect() as conn:
            games = conn.execute(text("""
                SELECT game_pk, game_date, home_team_code, away_team_code,
                       home_starter_id, away_starter_id, venue_id, first_pitch_utc
                FROM raw.games
                WHERE status = 'Final'
                  AND game_type = 'R'
                  AND home_score IS NOT NULL AND away_score IS NOT NULL
                  AND game_date BETWEEN :a AND :b
                ORDER BY game_date, game_pk
            """), {"a": block_start, "b": block_end}).fetchall()

            existing = {
                r[0] for r in conn.execute(text("""
                    SELECT game_pk FROM derived.over_under_outcomes
                    WHERE game_date BETWEEN :a AND :b
                """), {"a": block_start, "b": block_end}).fetchall()
            }

        todo = [g for g in games if g[0] not in existing]
        stats.already_present += len(games) - len(todo)

        if not todo:
            block_start = block_end + timedelta(days=1)
            continue

        # Refit models on strictly-prior data only
        runs_artifact, vol_artifact = _fit_block_models(engine, cfg, block_start, seed)
        if runs_artifact is None:
            stats.skip("no_trainable_runs_model", len(todo))
            block_start = block_end + timedelta(days=1)
            continue
        stats.blocks_trained += 1

        with engine.begin() as conn:
            for g in todo:
                (game_pk, game_date, home_raw, away_raw,
                 home_starter_id, away_starter_id, venue_id, first_pitch_utc) = g
                home = home_raw.strip().upper()
                away = away_raw.strip().upper()

                try:
                    # Home-perspective feature vector — the exact live builder,
                    # every sub-query bounded by game_date < this game's date.
                    home_fv = build_game_feature_vector(
                        conn=conn,
                        game_pk=game_pk,
                        team_code=home,
                        opp_team_code=away,
                        home_starter_id=home_starter_id,
                        away_starter_id=away_starter_id,
                        game_date=game_date,
                        venue_id=venue_id,
                        is_home=True,
                    )
                    values = home_fv.values
                    # Away perspective = identical covariates with is_home=0
                    # (verified equivalent to the live predictor's second
                    # build_game_feature_vector call with team/opp swapped).
                    away_values = dict(values)
                    away_values["is_home"] = 0
                    away_fv = GameFeatureVector(
                        game_pk=game_pk,
                        team_code=away,
                        feature_schema_version=GAME_FEATURE_SCHEMA_VERSION,
                        values=away_values,
                    )

                    home_runs = _predict_runs(runs_artifact, home_fv.values)
                    away_runs = _predict_runs(runs_artifact, away_fv.values)
                    total_expected = home_runs + away_runs

                    feature_vector = {
                        k: round(float(v), 6) for k, v in values.items()
                    }
                    sigma = _predict_sigma_from_artifact(vol_artifact, feature_vector)
                    p_over = compute_over_under_probabilities(
                        total_expected, residual_std=sigma
                    )

                    ballpark_val = feature_vector.get("ballpark_id")
                    predicted_at = _deterministic_predicted_at(game_date, first_pitch_utc)

                    conn.execute(_INSERT_SQL, {
                        "game_pk": game_pk,
                        "game_date": game_date,
                        "home_team_code": home,
                        "away_team_code": away,
                        "predicted_at_utc": predicted_at,
                        "p_over_5_5": p_over[5.5],
                        "p_over_6_5": p_over[6.5],
                        "p_over_7_5": p_over[7.5],
                        "p_over_8_5": p_over[8.5],
                        "p_over_9_5": p_over[9.5],
                        "p_over_10_5": p_over[10.5],
                        "p_over_11_5": p_over[11.5],
                        "feature_vector": json.dumps(feature_vector),
                        "home_starter_era": feature_vector.get("home_starter_era"),
                        "away_starter_era": feature_vector.get("away_starter_era"),
                        "ballpark_id": int(ballpark_val) if ballpark_val else None,
                        "home_trailing15_rpg": feature_vector.get("home_trailing15_rpg"),
                        "away_trailing15_rpg": feature_vector.get("away_trailing15_rpg"),
                        # actual starters from raw.games are always known here
                        "pitcher_fallback": home_starter_id is None or away_starter_id is None,
                        "home_expected_runs": home_runs,
                        "away_expected_runs": away_runs,
                        "sigma_used": sigma,
                    })
                    stats.predicted += 1
                except Exception as exc:  # noqa: BLE001 — log, skip, keep walking
                    stats.skip("game_error")
                    logger.warning(
                        f"Backtest game {game_pk} ({game_date} {home} vs {away}) "
                        f"failed: {exc}",
                        extra={"component": "over_under.backtest"},
                    )

                if stats.predicted and stats.predicted % PROGRESS_EVERY == 0:
                    logger.info(
                        f"Backtest progress: {stats.predicted} predicted "
                        f"(through {game_date})",
                        extra={"component": "over_under.backtest"},
                    )

        logger.info(
            f"Backtest block {block_start} → {block_end}: "
            f"{len(todo)} games (total {stats.predicted})",
            extra={"component": "over_under.backtest"},
        )
        block_start = block_end + timedelta(days=1)

    # Reconcile actual outcomes for everything just written (live reconciler,
    # idempotent). backtest_only leaves today's freshly-finished LIVE games for
    # the nightly pipeline to reconcile + report on, as always.
    stats.reconciled = reconcile_over_under(engine, backtest_only=True)

    live_after = live_snapshot(engine)
    if live_after != live_before:
        raise BacktestSafetyError(
            f"LIVE ROWS CHANGED during backtest: before={live_before} "
            f"after={live_after}. Investigate immediately."
        )

    result = {
        "predicted": stats.predicted,
        "reconciled": stats.reconciled,
        "already_present": stats.already_present,
        "skipped": stats.skipped,
        "blocks_trained": stats.blocks_trained,
        "live_rows": live_after[0],
        "live_checksum": live_after[2],
    }
    logger.info(
        f"Backtest complete: {result}",
        extra={"component": "over_under.backtest"},
    )
    return result


# ---------------------------------------------------------------------------
# Leakage audit + sanity checks
# ---------------------------------------------------------------------------


def verify_no_leakage(engine: Engine, sample_size: int = 20, seed: int = 7) -> dict:
    """Mandatory leakage audit over the persisted backtest rows.

    1. For *sample_size* random backtest rows, recompute home_trailing15_rpg
       directly via SQL using ONLY games strictly before that game_date and
       assert it matches the stored value.
    2. Assert no backtest row has predicted_at_utc >= its game's first_pitch_utc.

    Raises BacktestSafetyError on any violation.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT o.id, o.game_pk, o.game_date, o.home_team_code, o.home_trailing15_rpg
            FROM derived.over_under_outcomes o
            WHERE o.is_backtest
        """)).fetchall()
        if not rows:
            raise BacktestSafetyError("Leakage audit: no backtest rows found.")

        rng = random.Random(seed)
        sample = rng.sample(rows, min(sample_size, len(rows)))

        mismatches = []
        for row_id, game_pk, game_date, home_team, stored in sample:
            recomputed = conn.execute(text("""
                SELECT COALESCE(SUM(CASE WHEN home_team_code = :team THEN home_score
                                         ELSE away_score END)::float
                                / NULLIF(COUNT(*), 0), 4.5)
                FROM (
                    SELECT game_pk, home_team_code, home_score, away_score
                    FROM raw.games
                    WHERE (home_team_code = :team OR away_team_code = :team)
                      AND game_date < :game_date
                      AND status = 'Final'
                    ORDER BY game_date DESC, game_pk DESC
                    LIMIT 15
                ) recent
            """), {"team": home_team, "game_date": game_date}).fetchone()[0]
            if stored is None or abs(float(stored) - float(recomputed)) > 1e-4:
                mismatches.append((row_id, game_pk, game_date, stored, recomputed))

        bad_ts = conn.execute(text("""
            SELECT COUNT(*)
            FROM derived.over_under_outcomes o
            JOIN raw.games g ON g.game_pk = o.game_pk
            WHERE o.is_backtest
              AND g.first_pitch_utc IS NOT NULL
              AND o.predicted_at_utc >= g.first_pitch_utc
        """)).fetchone()[0]

    if mismatches:
        raise BacktestSafetyError(
            f"Leakage audit FAILED: {len(mismatches)} of {len(sample)} sampled rows "
            f"have home_trailing15_rpg != strictly-prior SQL recompute: {mismatches[:5]}"
        )
    if bad_ts:
        raise BacktestSafetyError(
            f"Leakage audit FAILED: {bad_ts} backtest rows have "
            f"predicted_at_utc >= first_pitch_utc."
        )

    result = {"sampled": len(sample), "feature_mismatches": 0, "predicted_at_violations": 0}
    logger.info(
        f"Leakage audit passed: {result}",
        extra={"component": "over_under.backtest"},
    )
    return result


def backtest_accuracy(engine: Engine) -> dict:
    """Base (unfiltered) accuracy per line over reconciled rows, backtest vs live."""
    out: dict = {}
    with engine.connect() as conn:
        for scope, cond in (("backtest", "o.is_backtest"), ("live", "NOT o.is_backtest")):
            per_line = {}
            for t in STANDARD_THRESHOLDS:
                col = str(t).replace(".", "_")
                r = conn.execute(text(f"""
                    SELECT COUNT(*) FILTER (WHERE was_correct_{col}), COUNT(*)
                    FROM derived.over_under_outcomes o
                    WHERE {cond} AND o.actual_total_runs IS NOT NULL
                      AND was_correct_{col} IS NOT NULL
                """)).fetchone()
                per_line[str(t)] = {
                    "n": int(r[1]),
                    "accuracy": round(r[0] / r[1], 4) if r[1] else None,
                }
            out[scope] = per_line
    return out


def sanity_check(engine: Engine) -> dict:
    """O5.5 base accuracy must land in a plausible band (0.50 – 0.95).

    Raises BacktestSafetyError otherwise — investigate before committing.
    """
    acc = backtest_accuracy(engine)
    o55 = acc["backtest"]["5.5"]["accuracy"]
    if o55 is None or not (0.50 <= o55 <= 0.95):
        raise BacktestSafetyError(
            f"Backtest O5.5 base accuracy {o55} outside plausible band [0.50, 0.95] "
            f"— STOP and investigate (full table: {acc})."
        )
    logger.info(
        f"Sanity check passed: backtest O5.5 accuracy={o55}",
        extra={"component": "over_under.backtest"},
    )
    return acc


__all__ = [
    "BacktestSafetyError",
    "DEFAULT_END",
    "DEFAULT_START",
    "backtest_accuracy",
    "live_snapshot",
    "run_backtest",
    "sanity_check",
    "verify_no_leakage",
]
