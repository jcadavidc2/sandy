"""Bet recommendations: cross every pending prediction with the LATEST calibration
reliability tables and keep only picks whose confidence bucket has proven itself
(bucket accuracy ≥ RECOMMEND_MIN_ACC on ≥ RECOMMEND_MIN_N historical picks).
Each recommendation carries its expected hit rate so the digest reads like a
tip sheet: '→ Más de 4.5 goles NO (85%) · histórico 86%'."""
from __future__ import annotations

import json

RECOMMEND_MIN_ACC = 0.60
RECOMMEND_MIN_N = 30


def load_reliability(conn, schema: str) -> dict:
    from sqlalchemy import text
    rows = conn.execute(text(f"""
        SELECT DISTINCT ON (market) market, reliability, recommended_threshold
        FROM {schema}.calibration_snapshots ORDER BY market, created_at DESC
    """)).fetchall()
    out = {}
    for m, rel, thr in rows:
        out[m] = rel if isinstance(rel, list) else json.loads(rel)
    return out


def bucket_acc(rel: list[dict], conf: float) -> tuple[float | None, int]:
    for b in rel or []:
        if b["lo"] <= conf < b["hi"]:
            return b.get("acc"), b.get("n") or 0
    return None, 0


def evaluate(reliability: dict, market: str, p: float | None,
             yes_label: str, no_label: str) -> dict | None:
    """One binary market → a candidate bet with its historical hit rate (or None)."""
    if p is None or market not in reliability:
        return None
    yes = p >= 0.5
    conf = p if yes else 1 - p
    acc, n = bucket_acc(reliability[market], conf)
    if acc is None or n < RECOMMEND_MIN_N or acc < RECOMMEND_MIN_ACC:
        return None
    return {"market": market, "label": yes_label if yes else no_label,
            "conf": conf, "hist_acc": acc, "hist_n": n}
