"""NHL reconcile + calibrate + backtest + Telegram digest (one compact module —
same treatment as MLS/football)."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.football.ratings import fit_dixon_coles
from sandy.over_under.notifier import send_telegram

from .model import DISPLAY_TZ, XI, load_reg_games, markets, persist_prediction

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30
BUCKETS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
MARKETS = {
    "double_chance": ("p_home_or_tie", "was_correct_double_chance"),
    "over_5_5": ("p_over_5_5", "was_correct_over_5_5"),
    "over_6_5": ("p_over_6_5", "was_correct_over_6_5"),
}
MEDALS = ["🥇", "🥈", "🥉"]


# ------------------------------- reconcile ---------------------------------
def reconcile(config: Config | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    now = datetime.now(timezone.utc)
    n = 0
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT p.id, p.p_home_or_tie, p.p_over_5_5, p.p_over_6_5,
                   g.home_goals, g.away_goals, g.reg_home_goals, g.reg_away_goals
            FROM nhl.game_predictions p
            JOIN nhl.games g ON g.game_id = p.game_id
            WHERE p.outcome_filled_at_utc IS NULL AND g.status = 'FINAL'
              AND g.home_goals IS NOT NULL
        """)).fetchall()
        for r in rows:
            pid, p_dc, p55, p65, hg, ag, rh, ra = r
            tot = int(hg) + int(ag)
            reg_res = "T" if rh == ra else ("H" if rh > ra else "A")
            conn.execute(text("""
                UPDATE nhl.game_predictions SET
                    actual_home_goals=:hg, actual_away_goals=:ag, actual_total_goals=:tot,
                    actual_reg_result=:res,
                    was_correct_double_chance=:c_dc, was_correct_over_5_5=:c55,
                    was_correct_over_6_5=:c65, outcome_filled_at_utc=:now
                WHERE id=:id
            """), {"id": pid, "hg": hg, "ag": ag, "tot": tot, "res": reg_res,
                   "c_dc": (p_dc >= 0.5) == (reg_res != "A") if p_dc is not None else None,
                   "c55": (p55 >= 0.5) == (tot > 5.5) if p55 is not None else None,
                   "c65": (p65 >= 0.5) == (tot > 6.5) if p65 is not None else None,
                   "now": now})
            n += 1
    logger.info("NHL reconciled %s predictions", n)
    return n


# ------------------------------- calibrate ---------------------------------
def calibrate(config: Config | None = None, *, lookback_days: int | None = None) -> list[dict]:
    cfg = config or load_config()
    engine = create_engine(cfg)
    where = "outcome_filled_at_utc IS NOT NULL"
    if lookback_days:
        where += f" AND match_date >= CURRENT_DATE - INTERVAL '{int(lookback_days)} days'"
    with engine.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM nhl.game_predictions WHERE {where}"), conn)
    snaps = []
    for market, (pcol, ccol) in MARKETS.items():
        sub = df.dropna(subset=[pcol, ccol])
        if len(sub) < MIN_SAMPLES:
            continue
        p = sub[pcol].to_numpy(dtype=float)
        correct = sub[ccol].to_numpy(dtype=bool)
        conf = np.maximum(p, 1 - p)
        picked_yes = p >= 0.5
        outcome_yes = np.where(picked_yes, correct, ~correct)
        table = []
        for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
            m = (conf >= lo) & (conf < hi)
            cnt = int(m.sum())
            table.append({"lo": lo, "hi": round(hi, 2), "n": cnt,
                          "acc": round(float(correct[m].mean()), 4) if cnt else None})
        thr = next((b["lo"] for b in table if (b["n"] or 0) >= 10 and (b["acc"] or 0) >= 0.6), None)
        snaps.append({"snapshot_date": date.today(), "market": market,
                      "lookback_days": lookback_days, "sample_size": int(len(sub)),
                      "accuracy": round(float(correct.mean()), 4),
                      "brier": round(float(np.mean((p - outcome_yes.astype(float)) ** 2)), 4),
                      "reliability": table, "recommended_threshold": thr})
    with engine.begin() as conn:
        for s in snaps:
            conn.execute(text("""
                INSERT INTO nhl.calibration_snapshots
                    (snapshot_date, market, lookback_days, sample_size, accuracy, brier,
                     reliability, recommended_threshold)
                VALUES (:snapshot_date, :market, :lookback_days, :sample_size, :accuracy,
                        :brier, :rel, :recommended_threshold)
            """), {**s, "rel": json.dumps(s["reliability"])})
    for s in snaps:
        logger.info("NHL calibration %s: acc=%.3f n=%s thr=%s",
                    s["market"], s["accuracy"], s["sample_size"], s["recommended_threshold"])
    return snaps


# ------------------------------- backtest ----------------------------------
def run_backtest(config: Config | None = None, *, refit_days: int = 14,
                 with_features: bool = True) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    with engine.begin() as conn:
        lo, hi = conn.execute(text(
            "SELECT MIN(match_date), MAX(match_date) FROM nhl.games WHERE status='FINAL'"
        )).fetchone()
    if lo is None:
        return {"predicted": 0}
    start = lo + timedelta(days=240)   # first season is training-only
    predicted = 0
    block_start = start
    while block_start <= hi:
        block_end = min(block_start + timedelta(days=refit_days - 1), hi)
        df = load_reg_games(engine, as_of=block_start)
        if len(df) >= 600:
            model = fit_dixon_coles(df, as_of_date=block_start, xi=XI)
            with engine.begin() as conn:
                rows = conn.execute(text("""
                    SELECT g.game_id, g.match_date, g.home_team_id, g.away_team_id, t1.abbrev, t2.abbrev
                    FROM nhl.games g
                    JOIN nhl.teams t1 ON t1.team_id = g.home_team_id
                    JOIN nhl.teams t2 ON t2.team_id = g.away_team_id
                    WHERE g.status = 'FINAL' AND g.match_date BETWEEN :a AND :b
                    ORDER BY g.match_date
                """), {"a": block_start, "b": block_end}).fetchall()
                for gid, mdate, hid, aid, hab, aab in rows:
                    mk = markets(model, engine, hid, aid, mdate, with_features=with_features)
                    persist_prediction(conn, gid, mdate, hid, aid, hab, aab, mk, is_backtest=True)
                    predicted += 1
            logger.info("NHL backtest block %s→%s: %s games (train n=%s)",
                        block_start, block_end, len(rows), len(df))
        block_start = block_end + timedelta(days=1)
    reconciled = reconcile(cfg)
    return {"predicted": predicted, "reconciled": reconciled}


# ------------------------------- digest ------------------------------------
def format_daily_digest(config: Config | None = None) -> str:
    cfg = config or load_config()
    engine = create_engine(cfg)
    today = datetime.now(DISPLAY_TZ).date()
    parts = [f"🏒 NHL Predictions ({today.strftime('%b %d')})"]
    with engine.begin() as conn:
        cal = conn.execute(text("""
            SELECT DISTINCT ON (market) market, accuracy, sample_size, recommended_threshold
            FROM nhl.calibration_snapshots ORDER BY market, created_at DESC
        """)).fetchall()
        if cal:
            total = max(r.sample_size for r in cal)
            parts.append(f"📊 Calibración ({total} evaluadas): " +
                         " · ".join(f"{r.market.replace('_', ' ')} {r.accuracy:.0%}" for r in cal) + ".")
        night = conn.execute(text("""
            SELECT home_team, away_team, actual_home_goals, actual_away_goals,
                   actual_reg_result, was_correct_double_chance
            FROM nhl.game_predictions
            WHERE match_date = :d AND outcome_filled_at_utc IS NOT NULL AND NOT is_backtest
            ORDER BY id LIMIT 10
        """), {"d": today - timedelta(days=1)}).fetchall()
        if night:
            hits = sum(1 for r in night if r.was_correct_double_chance)
            parts.append("")
            parts.append(f"🌙 Anoche: {hits}/{len(night)} dobles oportunidades correctas")
            for r in night:
                mark = "✅" if r.was_correct_double_chance else "❌"
                parts.append(f"{mark} {r.home_team} {r.actual_home_goals}-{r.actual_away_goals} {r.away_team}")
        picks = conn.execute(text("""
            SELECT home_team, away_team, p_home_or_tie, p_away_win_reg, p_over_5_5, p_over_6_5,
                   most_likely_home, most_likely_away
            FROM nhl.game_predictions
            WHERE match_date BETWEEN :a AND :b AND outcome_filled_at_utc IS NULL AND NOT is_backtest
            ORDER BY GREATEST(p_home_or_tie, p_away_win_reg) DESC LIMIT 12
        """), {"a": today, "b": today + timedelta(days=1)}).fetchall()
        parts.append("")
        if not picks:
            parts.append("😴 No hay juegos NHL hoy.")
        else:
            parts.append(f"🔮 Picks de hoy ({len(picks)}):")
            for i, r in enumerate(picks):
                medal = MEDALS[i] if i < len(MEDALS) else "•"
                if r.p_home_or_tie >= r.p_away_win_reg:
                    pick, conf = "Local o empate (reg.)", r.p_home_or_tie
                else:
                    pick, conf = "Gana visitante (reg.)", r.p_away_win_reg
                warn = " ⚠️ parejo" if conf < 0.55 else ""
                parts.append(f"{medal} {r.home_team} vs {r.away_team} → {pick} {conf:.0%}{warn} | "
                             f"O5.5 {r.p_over_5_5:.0%} | O6.5 {r.p_over_6_5:.0%} | "
                             f"prob. {r.most_likely_home}-{r.most_likely_away}")
    parts.append("")
    parts.append("ℹ️ reg. = al reglamento (OT/SO cuenta como empate) · totales incluyen OT/SO")
    return "\n".join(parts)


def notify_daily(config: Config | None = None) -> bool:
    ok = send_telegram(format_daily_digest(config))
    logger.info("NHL digest sent: %s", ok)
    return ok
