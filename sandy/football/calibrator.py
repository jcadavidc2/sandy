"""Calibration for football predictions.

Consumes the reconciled rows in ``football.match_predictions`` (seeded by the
walk-forward backtest and grown nightly) and measures, per market, how well our
probabilities match reality:

- **result**  — accuracy of the W/D/L pick (argmax), bucketed by pick confidence
- **over_2_5** — accuracy of the over/under 2.5 call, plus Brier score
- **btts**    — accuracy of the both-teams-score call, plus Brier score

Mirrors :mod:`sandy.over_under.calibrator`: it writes one
``football.calibration_snapshots`` row per market, with a reliability table in
``covariate_insights`` and a recommended confidence threshold.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.logging import get_logger

logger = get_logger("football.calibrator")

MIN_SAMPLES = 30


def _load_reconciled(engine: Engine, lookback_days: int | None) -> pd.DataFrame:
    clause = "p.outcome_filled_at_utc IS NOT NULL"
    params: dict = {}
    if lookback_days is not None:
        clause += " AND p.match_date >= :cutoff"
        params["cutoff"] = date.today() - pd.Timedelta(days=lookback_days)
    sql = text(f"""
        SELECT p.p_home_win, p.p_draw, p.p_away_win, p.p_over_2_5, p.p_btts,
               p.was_correct_result, p.was_correct_over_2_5, p.was_correct_btts,
               p.was_correct_score, p.actual_total_goals, p.actual_btts,
               m.competition_weight
        FROM football.match_predictions p
        JOIN football.matches m ON m.fixture_id = p.fixture_id
        WHERE {clause}
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame(rows, columns=[
        "p_home_win", "p_draw", "p_away_win", "p_over_2_5", "p_btts",
        "was_correct_result", "was_correct_over_2_5", "was_correct_btts",
        "was_correct_score", "actual_total_goals", "actual_btts", "competition_weight",
    ])


def _reliability_by_bucket(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        table.append({
            "bucket": f"{lo:.2f}-{hi:.2f}",
            "n": nb,
            "accuracy": round(float(correct[mask].mean()), 3),
        })
    return table


def _recommended_threshold(table: list[dict], target: float = 0.6) -> float | None:
    for row in table:
        if row["n"] >= 10 and row["accuracy"] >= target:
            return float(row["bucket"].split("-")[0])
    return None


def compute_calibration(engine: Engine, lookback_days: int | None = None) -> list[dict]:
    df = _load_reconciled(engine, lookback_days)
    if len(df) < MIN_SAMPLES:
        logger.info("Insufficient data for calibration",
                    extra={"component": "football.calibrator", "n": len(df)})
        return []

    snapshots: list[dict] = []

    # --- result market: confidence = max(W/D/L), accuracy of the pick ---
    conf = df[["p_home_win", "p_draw", "p_away_win"]].max(axis=1).to_numpy()
    correct = df["was_correct_result"].astype(float).to_numpy()
    rel = _reliability_by_bucket(conf, correct)
    snapshots.append({
        "market": "result",
        "accuracy": round(float(correct.mean()), 4),
        "sample_size": len(df),
        "recommended_threshold": _recommended_threshold(rel, 0.6),
        "covariate_insights": {
            "reliability": rel,
            "exact_score_accuracy": round(float(df["was_correct_score"].mean()), 4),
        },
    })

    # --- over/under 2.5 ---
    ou_correct = df["was_correct_over_2_5"].astype(float).to_numpy()
    over_actual = (df["actual_total_goals"] > 2.5).astype(float).to_numpy()
    ou_brier = float(np.mean((df["p_over_2_5"].to_numpy() - over_actual) ** 2))
    ou_conf = np.abs(df["p_over_2_5"].to_numpy() - 0.5) + 0.5
    snapshots.append({
        "market": "over_2_5",
        "accuracy": round(float(ou_correct.mean()), 4),
        "sample_size": len(df),
        "recommended_threshold": _recommended_threshold(_reliability_by_bucket(ou_conf, ou_correct), 0.6),
        "covariate_insights": {
            "brier": round(ou_brier, 4),
            "reliability": _reliability_by_bucket(ou_conf, ou_correct),
        },
    })

    # --- BTTS ---
    btts_correct = df["was_correct_btts"].astype(float).to_numpy()
    btts_actual = df["actual_btts"].astype(float).to_numpy()
    btts_brier = float(np.mean((df["p_btts"].to_numpy() - btts_actual) ** 2))
    btts_conf = np.abs(df["p_btts"].to_numpy() - 0.5) + 0.5
    snapshots.append({
        "market": "btts",
        "accuracy": round(float(btts_correct.mean()), 4),
        "sample_size": len(df),
        "recommended_threshold": _recommended_threshold(_reliability_by_bucket(btts_conf, btts_correct), 0.6),
        "covariate_insights": {
            "brier": round(btts_brier, 4),
            "reliability": _reliability_by_bucket(btts_conf, btts_correct),
        },
    })
    return snapshots


def persist_calibration(engine: Engine, snapshots: list[dict]) -> int:
    import json
    sql = text("""
        INSERT INTO football.calibration_snapshots
            (snapshot_date, market, accuracy, sample_size, recommended_threshold, covariate_insights)
        VALUES (:d, :market, :accuracy, :n, :rt, CAST(:ci AS JSONB))
    """)
    with engine.begin() as conn:
        for s in snapshots:
            conn.execute(sql, {
                "d": date.today(), "market": s["market"], "accuracy": s["accuracy"],
                "n": s["sample_size"], "rt": s["recommended_threshold"],
                "ci": json.dumps(s["covariate_insights"]),
            })
    return len(snapshots)


def calibrate(config: Config | None = None, *, lookback_days: int | None = None) -> list[dict]:
    cfg = config or load_config()
    engine = create_engine(cfg)
    snaps = compute_calibration(engine, lookback_days)
    if snaps:
        persist_calibration(engine, snaps)
    return snaps


__all__ = ["calibrate", "compute_calibration", "persist_calibration"]
