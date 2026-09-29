"""Calibrated MLB over/under probabilities — empirical distribution of past errors.

WHY (audit 2026-09-28): the live pipeline turned E[runs] into P(total > L) with a
symmetric Normal, ``1 − Φ((L − E)/σ)`` (predictor.compute_over_under_probabilities).
Two structural defects made every probability over-confident:

  * σ is the volatility model's *mean absolute error* (≈3.45), not a standard
    deviation (the real residual sd is ≈4.49) — every z was stretched ~1.3×.
  * run totals are right-skewed (mean 8.97, median 8) and discrete with an
    odd-number excess; a symmetric bell curve cannot represent that.

Result: +6 to +9pp over-bias on the 6.5/7.5/8.5 lines (the model said "over
8.5" in 90% of live games; reality ≈49%), which the odds layer then read as
"edge" against a market that IS calibrated. 61% of all value picks were MLB.

FIX — replace the Normal with the empirical CDF of the model's OWN past errors
r = actual_total − E, using only games strictly BEFORE the prediction date
(walk-forward, leakage-free by construction):

    P(over L) = #{ r_i > L − E } / N          (N = prior reconciled games)

One map for every line, monotone across lines, captures skew and parity, and
needs no fitting — it is a count. Walk-forward validation on the 1,922 live
2026 games (fit only on earlier rows): per-line bias 6.5/7.5/8.5 went from
+6.5/+8.5/+5.3pp to about −0.5/+1.6/+1.1pp, Brier and log-loss improved on
every line and in every month; the value-pick replay dropped the 22% of MLB
picks that only existed because of the bias (they lost) and lifted flat ROI
from +1.0% to ≈+4%.

STORAGE — the calibrated probabilities live in NEW nullable columns
``p_cal_over_{L}`` on derived.over_under_outcomes; the raw Normal columns are
untouched (the backtest safety hash and the legacy meta_over_5_5 read them).
The shared view derived.mlb_predictions_meta exposes both; betmeta
SPECS["mlb"] points the markets at the calibrated columns, so the value log,
both portfolios, the dashboard and the MLB meta all use the same numbers.

The σ model is not used here at all: its σ carries no information (corr with
|residual| ≈ 0.01), so the RAW residual r is used rather than r/σ — the two
performed identically in validation and the raw version has one fewer moving part.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.over_under.schemas import STANDARD_THRESHOLDS

logger = logging.getLogger(__name__)

MIN_RESIDUALS = 1000        # below this the empirical CDF is too coarse → no calibrated p (NULL)
WINDOW_DAYS = 3 * 365       # rolling fit window: the runs model drifts season to season
P_FLOOR, P_CEIL = 0.001, 0.999


def _col(t: float) -> str:
    return str(t).replace(".", "_")


CAL_COLUMNS = [f"p_cal_over_{_col(t)}" for t in STANDARD_THRESHOLDS]


# ------------------------------------------------------------------- schema --
def ensure_columns(engine: Engine) -> None:
    """Idempotent migration: add the p_cal_over_* columns + rebuild the meta view."""
    sql = (Path(__file__).resolve().parents[1] / "migrations" / "add_mlb_calibrated_probs.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))


# -------------------------------------------------------------- pure kernel --
def calibrated_probabilities(
    total_expected: float,
    residuals_sorted: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[float, float] | None:
    """P(total > L) for each line from the empirical CDF of past residuals.

    ``residuals_sorted`` must be ascending (np.sort of actual − expected over
    prior games). Returns None when fewer than MIN_RESIDUALS are available —
    the caller then stores NULL rather than a badly estimated probability.
    Monotone non-increasing in L by construction.
    """
    n = len(residuals_sorted)
    if n < MIN_RESIDUALS:
        return None
    out: dict[float, float] = {}
    for t in thresholds or STANDARD_THRESHOLDS:
        # over L  ⇔  actual > L  ⇔  r > L − E   (strict; L is a half-integer so ties can't occur)
        k = int(np.searchsorted(residuals_sorted, t - total_expected, side="right"))
        p = 1.0 - k / n
        out[t] = float(min(max(p, P_FLOOR), P_CEIL))
    return out


# ----------------------------------------------------------------- loaders --
def load_residuals(engine: Engine, before: date, window_days: int = WINDOW_DAYS) -> np.ndarray:
    """Sorted residuals actual − E of reconciled games with game_date < before
    (and within the rolling window). Backtest and live rows both count: they are
    all as-of predictions of the same model family."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT actual_total_runs - (home_expected_runs + away_expected_runs)
            FROM derived.over_under_outcomes
            WHERE actual_total_runs IS NOT NULL
              AND home_expected_runs IS NOT NULL AND away_expected_runs IS NOT NULL
              AND game_date < :before AND game_date >= :start
        """), {"before": before, "start": before - timedelta(days=window_days)}).fetchall()
    return np.sort(np.array([float(r[0]) for r in rows], dtype=float))


# ---------------------------------------------------------------- backfill --
def backfill_calibrated(engine: Engine, *, force: bool = False,
                        window_days: int = WINDOW_DAYS) -> int:
    """Walk-forward fill of p_cal_over_* for stored rows (default: only NULLs).

    For every game_date d, the CDF is built from reconciled rows with
    game_date < d — exactly what the live predictor would have seen that
    morning — so historical calibrated probabilities are honest as-of values.
    Idempotent; a nightly call after reconcile fills nothing unless a morning
    prediction ran with too few residuals. Returns rows updated.
    """
    ensure_columns(engine)
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT id, game_date, home_expected_runs + away_expected_runs AS e,
                   actual_total_runs AS y,
                   {CAL_COLUMNS[0]} IS NULL AS missing
            FROM derived.over_under_outcomes
            WHERE home_expected_runs IS NOT NULL AND away_expected_runs IS NOT NULL
            ORDER BY game_date, id
        """)).fetchall()
    if not rows:
        return 0
    dates = np.array([r.game_date for r in rows])
    e = np.array([float(r.e) for r in rows])
    y = np.array([np.nan if r.y is None else float(r.y) for r in rows])
    missing = np.array([bool(r.missing) for r in rows])
    ids = [r.id for r in rows]
    resid = y - e
    updates: list[dict] = []
    for d in np.unique(dates):
        target = (dates == d) & (missing | force)
        if not target.any():
            continue
        prior = (dates < d) & (dates >= d - timedelta(days=window_days)) & ~np.isnan(resid)
        rs = np.sort(resid[prior])
        for i in np.flatnonzero(target):
            p = calibrated_probabilities(e[i], rs)
            if p is None:
                continue
            updates.append({"id": ids[i], **{f"p{_col(t)}": p[t] for t in STANDARD_THRESHOLDS}})
    if not updates:
        return 0
    sets = ", ".join(f"p_cal_over_{_col(t)} = :p{_col(t)}" for t in STANDARD_THRESHOLDS)
    with engine.begin() as conn:
        for chunk in range(0, len(updates), 2000):
            conn.execute(text(f"UPDATE derived.over_under_outcomes SET {sets} WHERE id = :id"),
                         updates[chunk:chunk + 2000])
    logger.info("calibrated backfill: %d rows updated (force=%s)", len(updates), force)
    return len(updates)


def calibration_report(engine: Engine, since: date) -> list[dict]:
    """Per-line bias/Brier of raw vs calibrated p on reconciled LIVE rows since
    `since` — the honest scoreboard of this module (used by the CLI)."""
    out = []
    with engine.begin() as conn:
        for t in STANDARD_THRESHOLDS:
            c = _col(t)
            r = conn.execute(text(f"""
                SELECT count(*) AS n,
                       avg(p_over_{c}) - avg(actual_over_{c}::int) AS bias_raw,
                       avg(p_cal_over_{c}) - avg(actual_over_{c}::int) AS bias_cal,
                       avg((p_over_{c} - actual_over_{c}::int)^2) AS brier_raw,
                       avg((p_cal_over_{c} - actual_over_{c}::int)^2) AS brier_cal
                FROM derived.over_under_outcomes
                WHERE NOT is_backtest AND outcome_filled_at_utc IS NOT NULL
                  AND game_date >= :s AND p_cal_over_{c} IS NOT NULL
            """), {"s": since}).fetchone()
            out.append({"line": t, "n": int(r.n or 0),
                        "bias_raw_pp": None if r.bias_raw is None else round(100 * float(r.bias_raw), 1),
                        "bias_cal_pp": None if r.bias_cal is None else round(100 * float(r.bias_cal), 1),
                        "brier_raw": None if r.brier_raw is None else round(float(r.brier_raw), 4),
                        "brier_cal": None if r.brier_cal is None else round(float(r.brier_cal), 4)})
    return out
