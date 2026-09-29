"""Per-line probability calibration for every vertical except MLB — the MLB lesson
(sandy/over_under/calibration.py) generalized.

WHY (audit 2026-09-28/29): the MLB Normal-CDF probabilities were biased and produced
61% of the losing value picks. The discrete models of the other verticals (Dixon-Coles
goals, WLS points) have no systematic bias, but they ARE over-confident at the
extremes — e.g. NHL over 5.5: when the model says 63% the truth is 58%, when it says
47% the truth is 53%; NHL 1X is under-confident (46% → 51%, 55% → 60%); soccer UEL/SUD
over-predict the 3.5/4.5 lines by 4–8pp. Value picks live exactly in those extremes,
so an uncalibrated tail manufactures edge that isn't there.

WHAT: for each (league, market) an isotonic regression P(yes | model p) is fitted
WALK-FORWARD on the league's own reconciled predictions strictly BEFORE the row's
match_date (rolling WINDOW_DAYS, at least MIN_ROWS rows), and stored in a parallel
nullable column ``p_cal_<...>`` next to the raw ``p_<...>`` column (raw columns are
untouched; rows without enough history stay NULL and simply drop out of the meta
frame and of the candidate scans). betmeta.SPECS points the markets of the leagues in
CALIBRATED_LEAGUES at the calibrated columns, so the value log, both paper
portfolios, the dashboard and the per-league meta-models all use the same numbers.

WHEN: ``fill_league(league)`` is called right after each vertical's daily predict
(the CLI predict commands) and again defensively at the start of the odds daily pass
(sandy.odds.run_daily), so no candidate is ever scored on a NULL calibrated column.
Refit granularity is monthly blocks — a count-based map over thousands of rows does
not move day to day.

Honesty: isotonic on strictly-prior rows cannot leak; the reliability of the
calibrated column is reported by ``report(league)`` (raw vs calibrated bias/Brier on
live rows).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

MIN_ROWS = 300              # rows strictly before the block needed to trust a map (else identity)
WINDOW_DAYS = 4 * 365       # rolling fit window (ratings drift season to season)
P_FLOOR, P_CEIL = 0.001, 0.999


def calibrated_leagues() -> tuple[str, ...]:
    """Leagues whose SPECS markets read the calibrated columns (single source of
    truth lives in betmeta so the SPECS repoint and this module never disagree)."""
    from sandy.betmeta import CALIBRATED_LEAGUES
    return CALIBRATED_LEAGUES


def cal_col(pcol: str) -> str:
    """p_over_5_5 → p_cal_over_5_5, p_home_or_draw → p_cal_home_or_draw, ..."""
    assert pcol.startswith("p_"), pcol
    return "p_cal_" + pcol[2:]


def _raw_markets(league: str) -> dict[str, tuple[str, str, float | None]]:
    """The league's markets keyed on the RAW probability column (SPECS may already
    point at the calibrated one)."""
    from sandy.betmeta import SPECS
    out = {}
    for market, (pcol, kind, line) in SPECS[league]["markets"].items():
        raw = pcol.replace("p_cal_", "p_", 1) if pcol.startswith("p_cal_") else pcol
        out[market] = (raw, kind, line)
    return out


def ensure_columns(engine: Engine, league: str) -> None:
    """Idempotent ALTER TABLE ... ADD COLUMN IF NOT EXISTS for every calibrated column."""
    from sandy.betmeta import SPECS
    table = SPECS[league]["table"]
    with engine.begin() as conn:
        for _m, (raw, _k, _l) in _raw_markets(league).items():
            conn.execute(text(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {cal_col(raw)} DOUBLE PRECISION"))


def _yes_happened(rd: dict, kind: str, line: float | None) -> bool | None:
    """Did the YES side (over / home / home-or-draw / btts) happen? None if ungradeable."""
    from sandy.betmeta import _correct
    # _correct scores "the pick at prob p": p=0.99 encodes the YES side.
    return _correct(rd, kind, line, 0.99)


def fill_league(engine: Engine, league: str, *, force: bool = False,
                min_rows: int = MIN_ROWS, window_days: int = WINDOW_DAYS) -> int:
    """Walk-forward isotonic fill of p_cal_* for `league` (default: only NULL rows).

    Month by month: fit on reconciled rows with match_date < month start (rolling
    window), predict every target row of that month (reconciled or pending). Rows in
    months without enough history stay NULL. Returns rows updated."""
    from sklearn.isotonic import IsotonicRegression
    from sandy.betmeta import SPECS
    spec = SPECS[league]
    table = spec["table"]
    where = f" AND {spec['where']}" if spec.get("where") else ""
    ensure_columns(engine, league)
    markets = _raw_markets(league)
    raw_cols = [raw for raw, _k, _l in markets.values()]
    cal_cols = [cal_col(c) for c in raw_cols]
    with engine.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table} WHERE TRUE{where} ORDER BY match_date, id"), conn)
    if df.empty:
        return 0
    df["match_date"] = pd.to_datetime(df["match_date"])
    recon = df["outcome_filled_at_utc"].notna().to_numpy()
    # labels (YES happened) for reconciled rows, per market — one pass over the records
    records = df.to_dict("records")
    labels: dict[str, np.ndarray] = {raw: np.full(len(df), np.nan) for raw in raw_cols}
    for i in np.flatnonzero(recon):
        rd = records[i]
        for market, (raw, kind, line) in markets.items():
            got = _yes_happened(rd, kind, line)
            if got is not None:
                labels[raw][i] = float(got)
    ids = df["id"].to_numpy()
    updates: dict[int, dict] = {}
    months = sorted(df["match_date"].dt.to_period("M").unique())
    for m in months:
        start = m.to_timestamp()
        in_month = (df["match_date"].dt.to_period("M") == m).to_numpy()
        target = in_month & (force | df[cal_cols].isna().any(axis=1).to_numpy())
        if not target.any():
            continue
        prior = ((df["match_date"] < start) & (df["match_date"] >= start - timedelta(days=window_days))).to_numpy() & recon
        for raw in raw_cols:
            x_all = pd.to_numeric(df[raw], errors="coerce").to_numpy(dtype=float)
            rows = np.flatnonzero(target & ~np.isnan(x_all))
            if not len(rows):
                continue
            ok = prior & ~np.isnan(labels[raw]) & ~np.isnan(x_all)
            if ok.sum() >= min_rows:
                iso = IsotonicRegression(y_min=P_FLOOR, y_max=P_CEIL, out_of_bounds="clip")
                iso.fit(x_all[ok], labels[raw][ok])
                preds = iso.predict(x_all[rows])
            else:
                # Not enough own history yet (new cups, first season): identity map —
                # the league keeps working on its raw probability until it earns a map.
                preds = x_all[rows]
            for i, p in zip(rows, preds):
                updates.setdefault(int(ids[i]), {})[cal_col(raw)] = float(min(max(p, P_FLOOR), P_CEIL))
    if not updates:
        return 0
    n = 0
    with engine.begin() as conn:
        for rid, vals in updates.items():
            sets = ", ".join(f"{c} = :{c}" for c in vals)
            conn.execute(text(f"UPDATE {table} SET {sets} WHERE id = :id"), {"id": rid, **vals})
            n += 1
    logger.info("%s: calibrated columns filled for %d rows (force=%s)", league, n, force)
    return n


def fill_all(engine: Engine, leagues: tuple[str, ...] | None = None, **kw) -> dict[str, int]:
    out = {}
    for lg in leagues or calibrated_leagues():
        try:
            out[lg] = fill_league(engine, lg, **kw)
        except Exception:  # noqa: BLE001 — one league must never block the others
            logger.exception("calibrate_lines: %s failed", lg)
            out[lg] = -1
    return out


def report(engine: Engine, league: str, since: date | None = None) -> list[dict]:
    """Raw vs calibrated bias (mean p − hit rate) and Brier on reconciled LIVE rows."""
    from sandy.betmeta import SPECS
    spec = SPECS[league]
    where = f" AND {spec['where']}" if spec.get("where") else ""
    bt = "" if spec.get("no_backtest_col") else " AND NOT is_backtest"
    since = since or (date.today() - timedelta(days=120))
    with engine.begin() as conn:
        df = pd.read_sql(text(
            f"SELECT * FROM {spec['table']} WHERE outcome_filled_at_utc IS NOT NULL "
            f"AND match_date >= :s{where}{bt}"), conn, params={"s": since})
    out = []
    for market, (raw, kind, line) in _raw_markets(league).items():
        c = cal_col(raw)
        if c not in df.columns:
            continue
        sub = df.dropna(subset=[raw, c])
        if sub.empty:
            continue
        y = np.array([float(v) if (v := _yes_happened(r, kind, line)) is not None else np.nan
                      for r in sub.to_dict("records")])
        ok = ~np.isnan(y)
        if ok.sum() < 20:
            continue
        p_raw, p_cal, yy = sub[raw].to_numpy(dtype=float)[ok], sub[c].to_numpy(dtype=float)[ok], y[ok]
        out.append({"market": market, "n": int(ok.sum()),
                    "bias_raw_pp": round(100 * float(p_raw.mean() - yy.mean()), 1),
                    "bias_cal_pp": round(100 * float(p_cal.mean() - yy.mean()), 1),
                    "brier_raw": round(float(np.mean((p_raw - yy) ** 2)), 4),
                    "brier_cal": round(float(np.mean((p_cal - yy) ** 2)), 4)})
    return out


def main() -> None:
    import argparse, json
    from sandy.config import load_config
    from sandy.db import create_engine
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Per-line isotonic calibration (walk-forward) for the non-MLB verticals")
    ap.add_argument("cmd", choices=["fill", "report"])
    ap.add_argument("--league", action="append", help="repeatable; default = all CALIBRATED_LEAGUES")
    ap.add_argument("--force", action="store_true", help="recompute every row, not only NULLs")
    args = ap.parse_args()
    engine = create_engine(load_config())
    leagues = tuple(args.league) if args.league else calibrated_leagues()
    if args.cmd == "fill":
        print(json.dumps(fill_all(engine, leagues, force=args.force), indent=2))
    else:
        for lg in leagues:
            for r in report(engine, lg):
                print(f"{lg:11s} {r['market']:18s} n={r['n']:5d} bias raw {r['bias_raw_pp']:+5.1f}pp → cal "
                      f"{r['bias_cal_pp']:+5.1f}pp | Brier {r['brier_raw']} → {r['brier_cal']}")


if __name__ == "__main__":
    main()
