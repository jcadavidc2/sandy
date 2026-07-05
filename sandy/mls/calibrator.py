"""Per-market calibration for MLS: double_chance, over_2_5, corners_over_9_5.
Same treatment as football: accuracy, confidence-bucket reliability, Brier,
recommended trust threshold → mls.calibration_snapshots."""
from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30
BUCKETS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]

# Every stored line gets its own calibration (MLB-style threshold ladder).
# Correctness is computed from the reconciled actual totals, so adding a line
# never needs a schema change. Confidence for a binary market is max(p, 1-p).
def _goal_line(pcol: str, thr: float):
    return (pcol, "actual_total_goals", lambda r: (r[pcol] >= 0.5) == (r["actual_total_goals"] > thr))


def _corner_line(pcol: str, thr: float):
    return (pcol, "actual_total_corners", lambda r: (r[pcol] >= 0.5) == (r["actual_total_corners"] > thr))


MARKETS = {
    "double_chance": ("p_home_or_draw", "actual_result",
                      lambda r: (r["p_home_or_draw"] >= 0.5) == (r["actual_result"] != "A")),
    "over_0_5": _goal_line("p_over_0_5", 0.5),
    "over_1_5": _goal_line("p_over_1_5", 1.5),
    "over_2_5": _goal_line("p_over_2_5", 2.5),
    "over_3_5": _goal_line("p_over_3_5", 3.5),
    "over_4_5": _goal_line("p_over_4_5", 4.5),
    "over_5_5": _goal_line("p_over_5_5", 5.5),
    "corners_over_7_5": _corner_line("p_corners_over_7_5", 7.5),
    "corners_over_8_5": _corner_line("p_corners_over_8_5", 8.5),
    "corners_over_9_5": _corner_line("p_corners_over_9_5", 9.5),
    "corners_over_10_5": _corner_line("p_corners_over_10_5", 10.5),
    "corners_over_11_5": _corner_line("p_corners_over_11_5", 11.5),
    "corners_over_12_5": _corner_line("p_corners_over_12_5", 12.5),
}


def _reliability(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    out = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        out.append({"lo": lo, "hi": round(hi, 2), "n": n,
                    "acc": round(float(correct[m].mean()), 4) if n else None})
    return out


def _recommended_threshold(table: list[dict], target: float = 0.6) -> float | None:
    for b in table:
        if (b["n"] or 0) >= 10 and (b["acc"] or 0) >= target:
            return b["lo"]
    return None


def compute_calibration(engine: Engine, lookback_days: int | None = None) -> list[dict]:
    where = "outcome_filled_at_utc IS NOT NULL"
    if lookback_days:
        where += f" AND match_date >= CURRENT_DATE - INTERVAL '{int(lookback_days)} days'"
    with engine.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM mls.match_predictions WHERE {where}"), conn)
    snaps = []
    for market, (pcol, actual_col, correct_fn) in MARKETS.items():
        sub = df.dropna(subset=[pcol, actual_col])
        if len(sub) < MIN_SAMPLES:
            logger.info("MLS calibration: market %s has %s samples (<%s) — skipped",
                        market, len(sub), MIN_SAMPLES)
            continue
        p = sub[pcol].to_numpy(dtype=float)
        correct = sub.apply(correct_fn, axis=1).to_numpy(dtype=bool)
        conf = np.maximum(p, 1 - p)
        picked_yes = p >= 0.5
        outcome_yes = np.where(picked_yes, correct, ~correct)  # reconstruct the actual binary outcome
        brier = float(np.mean((p - outcome_yes.astype(float)) ** 2))
        table = _reliability(conf, correct)
        snaps.append({
            "snapshot_date": date.today(), "market": market,
            "lookback_days": lookback_days, "sample_size": int(len(sub)),
            "accuracy": round(float(correct.mean()), 4), "brier": round(brier, 4),
            "reliability": table, "recommended_threshold": _recommended_threshold(table),
        })
    return snaps


def persist_calibration(engine: Engine, snaps: list[dict]) -> int:
    with engine.begin() as conn:
        for s in snaps:
            conn.execute(text("""
                INSERT INTO mls.calibration_snapshots
                    (snapshot_date, market, lookback_days, sample_size, accuracy, brier,
                     reliability, recommended_threshold)
                VALUES (:snapshot_date, :market, :lookback_days, :sample_size, :accuracy,
                        :brier, :rel, :recommended_threshold)
            """), {**s, "rel": json.dumps(s["reliability"])})
    return len(snaps)


def calibrate(config: Config | None = None, *, lookback_days: int | None = None) -> list[dict]:
    cfg = config or load_config()
    engine = create_engine(cfg)
    snaps = compute_calibration(engine, lookback_days)
    persist_calibration(engine, snaps)
    for s in snaps:
        logger.info("MLS calibration %s: acc=%.3f brier=%.3f n=%s thr=%s",
                    s["market"], s["accuracy"], s["brier"], s["sample_size"],
                    s["recommended_threshold"])
    return snaps
