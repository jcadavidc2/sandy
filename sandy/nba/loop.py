"""NBA: ingest + model + predict + reconcile + calibrate + backtest + digest.

Model (points are NOT Poisson — high counts, variance≠mean): weighted least
squares on team offense/defense with exponential time decay and home advantage:
    E[points(i @ j home h)] = mu + off_i - def_j + hfa·is_home
σ of the TOTAL from weighted residuals. P(over L) = 1 − Φ((L − E[total])/σ_total),
P(home win) = Φ(E[margin]/σ_margin) — the same normal machinery as MLB's runs stack.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date, datetime, timedelta, timezone
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from sandy.config import Config, load_config
from sandy.db import create_engine
from sandy.mls.client import EspnClient
from sandy.mls.parsers import DISPLAY_TZ
from sandy.over_under.notifier import send_telegram

logger = logging.getLogger(__name__)

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "add_nba_tables.sql"
SEASON_MONTHS = {1, 2, 3, 4, 5, 6, 10, 11, 12}
TOTAL_LINES = [210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5, 245.5]
HALF_LIFE_DAYS = 120.0
MIN_TRAIN = 400
MIN_SAMPLES = 30
BUCKETS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _client() -> EspnClient:
    return EspnClient(sport="basketball", code="nba")


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())


# ------------------------------- ingest ------------------------------------
_STATUS = {"STATUS_FINAL": "FT", "STATUS_SCHEDULED": "NS", "STATUS_IN_PROGRESS": "LIVE",
           "STATUS_HALFTIME": "LIVE", "STATUS_END_PERIOD": "LIVE", "STATUS_POSTPONED": "PPD",
           "STATUS_CANCELED": "PPD"}


def _upsert_event(conn, ev: dict) -> bool:
    try:
        comp = ev["competitions"][0]
        started = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        status = _STATUS.get(ev["status"]["type"]["name"], "NS")
        sides: dict = {}
        for c in comp["competitors"]:
            t = c["team"]
            conn.execute(text("""
                INSERT INTO nba.teams (team_id, name, abbrev) VALUES (:id, :n, :ab)
                ON CONFLICT (team_id) DO UPDATE SET name = EXCLUDED.name, abbrev = EXCLUDED.abbrev
            """), {"id": int(t["id"]), "n": t.get("displayName") or t["id"], "ab": t.get("abbreviation")})
            score = None
            if status == "FT":
                try:
                    score = int(c.get("score"))
                except (TypeError, ValueError):
                    score = None
            sides[c["homeAway"]] = (int(t["id"]), score)
        conn.execute(text("""
            INSERT INTO nba.games (event_id, match_date, start_utc, season, status,
                home_team_id, away_team_id, home_points, away_points)
            VALUES (:eid, :d, :ts, :season, :st, :h, :a, :hp, :ap)
            ON CONFLICT (event_id) DO UPDATE SET status = EXCLUDED.status,
                home_points = COALESCE(EXCLUDED.home_points, nba.games.home_points),
                away_points = COALESCE(EXCLUDED.away_points, nba.games.away_points),
                start_utc = EXCLUDED.start_utc, match_date = EXCLUDED.match_date
        """), {"eid": int(ev["id"]), "d": started.astimezone(DISPLAY_TZ).date(), "ts": started,
               "season": (ev.get("season") or {}).get("year"), "st": status,
               "h": sides["home"][0], "a": sides["away"][0],
               "hp": sides["home"][1], "ap": sides["away"][1]})
        return True
    except (KeyError, IndexError, ValueError) as e:
        logger.warning("skip nba event %s: %s", ev.get("id"), e)
        return False


def ingest_dates(engine: Engine, dates: list[date], client: EspnClient | None = None) -> int:
    client = client or _client()
    n = 0
    for d in dates:
        payload = client.scoreboard(d.strftime("%Y%m%d"))
        with engine.begin() as conn:
            for ev in payload.get("events", []) or []:
                if _upsert_event(conn, ev):
                    n += 1
    return n


def ingest_recent_window(config: Config | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    today = datetime.now(DISPLAY_TZ).date()
    n = ingest_dates(engine, [today - timedelta(days=1), today, today + timedelta(days=1)])
    logger.info("nba daily ingest: %s games", n)
    return n


def backfill(config: Config | None = None, *, start: date, end: date | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_schema(engine)
    client = _client()
    end = end or datetime.now(DISPLAY_TZ).date()
    n = 0
    d = start
    while d <= end:
        if d.month in SEASON_MONTHS:
            n += ingest_dates(engine, [d], client)
        d += timedelta(days=1)
        if d.day == 1:
            logger.info("nba backfill through %s (%s upserts)", d, n)
    return n


# ------------------------------- model -------------------------------------
class NbaModel:
    def __init__(self, mu, hfa, off, dfn, sigma_total, sigma_margin, as_of, n):
        self.mu, self.hfa, self.off, self.dfn = mu, hfa, off, dfn
        self.sigma_total, self.sigma_margin = sigma_total, sigma_margin
        self.as_of, self.n = as_of, n

    def expected(self, home_id: int, away_id: int) -> tuple[float, float]:
        eh = self.mu + self.off.get(home_id, 0.0) - self.dfn.get(away_id, 0.0) + self.hfa
        ea = self.mu + self.off.get(away_id, 0.0) - self.dfn.get(home_id, 0.0)
        return eh, ea


def _load_games(engine: Engine, as_of: date | None = None) -> pd.DataFrame:
    where = "status = 'FT' AND home_points IS NOT NULL"
    params: dict = {}
    if as_of is not None:
        where += " AND match_date < :as_of"
        params["as_of"] = as_of
    with engine.begin() as conn:
        return pd.read_sql(text(f"""
            SELECT match_date, home_team_id, away_team_id, home_points, away_points
            FROM nba.games WHERE {where} ORDER BY match_date
        """), conn, params=params)


def fit_model(engine: Engine, as_of: date | None = None) -> NbaModel:
    df = _load_games(engine, as_of)
    if len(df) < MIN_TRAIN:
        raise ValueError(f"only {len(df)} NBA games to fit")
    ref = as_of or date.today()
    ages = (pd.Timestamp(ref) - pd.to_datetime(df["match_date"])).dt.days.to_numpy(dtype=float)
    w = np.exp(-np.log(2) * ages / HALF_LIFE_DAYS)
    teams = sorted(set(df["home_team_id"]) | set(df["away_team_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    rows, ys, ws = [], [], []
    for (hi, ai, hp, ap), wt in zip(df[["home_team_id", "away_team_id", "home_points", "away_points"]].to_numpy(), w):
        r1 = np.zeros(2 * n_t + 1)
        r1[idx[hi]] = 1.0; r1[n_t + idx[ai]] = -1.0; r1[-1] = 1.0  # home offense − away defense + hfa
        rows.append(r1); ys.append(hp); ws.append(wt)
        r2 = np.zeros(2 * n_t + 1)
        r2[idx[ai]] = 1.0; r2[n_t + idx[hi]] = -1.0                 # away offense − home defense
        rows.append(r2); ys.append(ap); ws.append(wt)
    X = np.asarray(rows); y = np.asarray(ys, dtype=float); sw = np.sqrt(np.asarray(ws))
    mu = float(np.average(y, weights=np.repeat(w, 2)))  # baseline points per team (2 rows/game)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], (y - mu) * sw, rcond=None)
    off = {t: float(beta[idx[t]]) for t in teams}
    dfn = {t: float(beta[n_t + idx[t]]) for t in teams}
    hfa = float(beta[-1])
    # residual σ for totals and margins
    eh = np.array([mu + off[h] - dfn[a] + hfa for h, a in df[["home_team_id", "away_team_id"]].to_numpy()])
    ea = np.array([mu + off[a] - dfn[h] for h, a in df[["home_team_id", "away_team_id"]].to_numpy()])
    tot_res = (df["home_points"] + df["away_points"]).to_numpy() - (eh + ea)
    mar_res = (df["home_points"] - df["away_points"]).to_numpy() - (eh - ea)
    sigma_total = float(np.sqrt(np.average(tot_res ** 2, weights=w)))
    sigma_margin = float(np.sqrt(np.average(mar_res ** 2, weights=w)))
    model = NbaModel(mu, hfa, off, dfn, sigma_total, sigma_margin, ref, len(df))
    logger.info("nba model: n=%s mu=%.1f hfa=%.2f sigma_total=%.1f", len(df), mu, hfa, sigma_total)
    return model


def _model_path(cfg: Config) -> Path:
    return cfg.model.model_dir / "nba_points.pkl"


def fit_and_persist(config: Config | None = None) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    model = fit_model(engine)
    with open(_model_path(cfg), "wb") as f:
        pickle.dump(model, f)
    return {"games": model.n, "mu": round(model.mu, 1), "hfa": round(model.hfa, 2),
            "sigma_total": round(model.sigma_total, 1)}


def team_form(engine: Engine, team_id: int, as_of: date) -> dict:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT match_date, home_team_id, home_points, away_points FROM nba.games
            WHERE status = 'FT' AND home_points IS NOT NULL
              AND (home_team_id = :t OR away_team_id = :t) AND match_date < :d
            ORDER BY match_date DESC LIMIT 10
        """), {"t": team_id, "d": as_of}).fetchall()
    if not rows:
        return {}
    pf, pa, wins = [], [], 0
    for r in rows:
        is_home = r.home_team_id == team_id
        f = r.home_points if is_home else r.away_points
        a = r.away_points if is_home else r.home_points
        pf.append(f); pa.append(a); wins += 1 if f > a else 0
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    rest = (as_of - rows[0].match_date).days
    return {"pf_5": mean(pf[:5]), "pa_5": mean(pa[:5]), "pf_10": mean(pf), "pa_10": mean(pa),
            "wins_10": wins, "rest_days": rest, "back_to_back": rest <= 1, "played_10": len(rows)}


def _markets(model: NbaModel, engine, hid, aid, as_of, with_features=True) -> dict:
    feats = ({"home": team_form(engine, hid, as_of), "away": team_form(engine, aid, as_of)}
             if with_features else {})
    eh, ea = model.expected(hid, aid)
    total = eh + ea
    margin = eh - ea
    p_over = {ln: 1.0 - _phi((ln - total) / model.sigma_total) for ln in TOTAL_LINES}
    return {"eh": round(eh, 1), "ea": round(ea, 1), "total": round(total, 1),
            "sigma": round(model.sigma_total, 1),
            "p_home_win": _phi(margin / model.sigma_margin), "p_over": p_over, "features": feats}


_UPSERT = text("""
    INSERT INTO nba.game_predictions (
        event_id, match_date, home_team_id, away_team_id, home_team, away_team,
        exp_home_points, exp_away_points, exp_total, sigma_total, p_home_win,
        p_over_210_5, p_over_215_5, p_over_220_5, p_over_225_5, p_over_230_5,
        p_over_235_5, p_over_240_5, p_over_245_5,
        features, is_backtest, predicted_at_utc)
    VALUES (:eid, :d, :hid, :aid, :hn, :an, :eh, :ea, :tot, :sig, :phw,
            :o210, :o215, :o220, :o225, :o230, :o235, :o240, :o245, :feats, :bt, :now)
    ON CONFLICT (event_id) DO UPDATE SET
        exp_home_points=:eh, exp_away_points=:ea, exp_total=:tot, sigma_total=:sig,
        p_home_win=:phw, p_over_210_5=:o210, p_over_215_5=:o215, p_over_220_5=:o220,
        p_over_225_5=:o225, p_over_230_5=:o230, p_over_235_5=:o235, p_over_240_5=:o240,
        p_over_245_5=:o245, features=:feats, is_backtest=:bt,
        predicted_at_utc=:now
""")


def _persist(conn, eid, mdate, hid, aid, hn, an, mk, is_backtest=False):
    conn.execute(_UPSERT, {"eid": eid, "d": mdate, "hid": hid, "aid": aid, "hn": hn, "an": an,
                           "eh": mk["eh"], "ea": mk["ea"], "tot": mk["total"], "sig": mk["sigma"],
                           "phw": mk["p_home_win"],
                           "o210": mk["p_over"][210.5], "o215": mk["p_over"][215.5],
                           "o220": mk["p_over"][220.5], "o225": mk["p_over"][225.5],
                           "o230": mk["p_over"][230.5], "o235": mk["p_over"][235.5],
                           "o240": mk["p_over"][240.5], "o245": mk["p_over"][245.5],
                           "feats": json.dumps(mk["features"]) if mk.get("features") else None,
                           "bt": is_backtest, "now": datetime.now(timezone.utc)})


_ROWS_SQL = """
    SELECT g.event_id, g.match_date, g.home_team_id, g.away_team_id, t1.abbrev, t2.abbrev
    FROM nba.games g
    JOIN nba.teams t1 ON t1.team_id = g.home_team_id
    JOIN nba.teams t2 ON t2.team_id = g.away_team_id
    WHERE g.status = {status} AND g.match_date BETWEEN :a AND :b
    ORDER BY g.match_date, g.event_id
"""


def predict_scheduled(config: Config | None = None, *, days_ahead: int = 1) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    with open(_model_path(cfg), "rb") as f:
        model = pickle.load(f)
    today = datetime.now(DISPLAY_TZ).date()
    n = 0
    with engine.begin() as conn:
        rows = conn.execute(text(_ROWS_SQL.format(status="'NS'")),
                            {"a": today, "b": today + timedelta(days=days_ahead)}).fetchall()
        for eid, mdate, hid, aid, hab, aab in rows:
            _persist(conn, eid, mdate, hid, aid, hab, aab, _markets(model, engine, hid, aid, today))
            n += 1
    logger.info("nba predicted %s games", n)
    return n


# ------------------------------ reconcile / calibrate ----------------------
def reconcile(config: Config | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    now = datetime.now(timezone.utc)
    n = 0
    with engine.begin() as conn:
        for r in conn.execute(text("""
            SELECT p.id, p.p_home_win, p.p_over_225_5, g.home_points, g.away_points
            FROM nba.game_predictions p JOIN nba.games g ON g.event_id = p.event_id
            WHERE p.outcome_filled_at_utc IS NULL AND g.status = 'FT' AND g.home_points IS NOT NULL
        """)).fetchall():
            pid, phw, p225, hp, ap = r
            tot = int(hp) + int(ap)
            winner = "H" if hp > ap else "A"
            conn.execute(text("""
                UPDATE nba.game_predictions SET actual_home_points=:hp, actual_away_points=:ap,
                    actual_total=:tot, actual_winner=:w,
                    was_correct_winner=:cw, was_correct_over_225_5=:c225, outcome_filled_at_utc=:now
                WHERE id=:id
            """), {"id": pid, "hp": hp, "ap": ap, "tot": tot, "w": winner,
                   "cw": (phw >= 0.5) == (winner == "H") if phw is not None else None,
                   "c225": (p225 >= 0.5) == (tot > 225.5) if p225 is not None else None,
                   "now": now})
            n += 1
    logger.info("nba reconciled %s", n)
    return n


def _line_market(pcol, thr):
    return (pcol, "actual_total", lambda r: (r[pcol] >= 0.5) == (r["actual_total"] > thr))


MARKETS = {
    "winner": ("p_home_win", "actual_winner",
               lambda r: (r["p_home_win"] >= 0.5) == (r["actual_winner"] == "H")),
    "over_210_5": _line_market("p_over_210_5", 210.5),
    "over_215_5": _line_market("p_over_215_5", 215.5),
    "over_220_5": _line_market("p_over_220_5", 220.5),
    "over_225_5": _line_market("p_over_225_5", 225.5),
    "over_230_5": _line_market("p_over_230_5", 230.5),
    "over_235_5": _line_market("p_over_235_5", 235.5),
    "over_240_5": _line_market("p_over_240_5", 240.5),
    "over_245_5": _line_market("p_over_245_5", 245.5),
}


def calibrate(config: Config | None = None, *, lookback_days: int | None = None) -> list[dict]:
    cfg = config or load_config()
    engine = create_engine(cfg)
    where = "outcome_filled_at_utc IS NOT NULL"
    if lookback_days:
        where += f" AND match_date >= CURRENT_DATE - INTERVAL '{int(lookback_days)} days'"
    with engine.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM nba.game_predictions WHERE {where}"), conn)
    snaps = []
    for market, (pcol, acol, fn) in MARKETS.items():
        sub = df.dropna(subset=[pcol, acol])
        if len(sub) < MIN_SAMPLES:
            continue
        p = sub[pcol].to_numpy(dtype=float)
        correct = sub.apply(fn, axis=1).to_numpy(dtype=bool)
        conf = np.maximum(p, 1 - p)
        table = []
        for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
            m = (conf >= lo) & (conf < hi)
            cnt = int(m.sum())
            table.append({"lo": lo, "hi": round(hi, 2), "n": cnt,
                          "acc": round(float(correct[m].mean()), 4) if cnt else None})
        thr = next((b["lo"] for b in table if (b["n"] or 0) >= 10 and (b["acc"] or 0) >= 0.6), None)
        picked_yes = p >= 0.5
        outcome_yes = np.where(picked_yes, correct, ~correct)
        snaps.append({"snapshot_date": date.today(), "market": market,
                      "lookback_days": lookback_days, "sample_size": int(len(sub)),
                      "accuracy": round(float(correct.mean()), 4),
                      "brier": round(float(np.mean((p - outcome_yes.astype(float)) ** 2)), 4),
                      "reliability": table, "recommended_threshold": thr})
    with engine.begin() as conn:
        for s in snaps:
            conn.execute(text("""
                INSERT INTO nba.calibration_snapshots
                    (snapshot_date, market, lookback_days, sample_size, accuracy, brier,
                     reliability, recommended_threshold)
                VALUES (:snapshot_date, :market, :lookback_days, :sample_size, :accuracy,
                        :brier, :rel, :recommended_threshold)
            """), {**s, "rel": json.dumps(s["reliability"])})
    for s in snaps:
        logger.info("nba calibration %s: acc=%.3f n=%s", s["market"], s["accuracy"], s["sample_size"])
    return snaps


def run_backtest(config: Config | None = None, *, refit_days: int = 14) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    with engine.begin() as conn:
        lo, hi = conn.execute(text(
            "SELECT MIN(match_date), MAX(match_date) FROM nba.games WHERE status='FT'")).fetchone()
    if lo is None:
        return {"predicted": 0}
    block_start = lo + timedelta(days=240)
    predicted = 0
    while block_start <= hi:
        block_end = min(block_start + timedelta(days=refit_days - 1), hi)
        try:
            model = fit_model(engine, as_of=block_start)
        except ValueError:
            block_start = block_end + timedelta(days=1)
            continue
        with engine.begin() as conn:
            rows = conn.execute(text(_ROWS_SQL.format(status="'FT'")),
                                {"a": block_start, "b": block_end}).fetchall()
            for eid, mdate, hid, aid, hab, aab in rows:
                _persist(conn, eid, mdate, hid, aid, hab, aab,
                         _markets(model, engine, hid, aid, mdate), is_backtest=True)
                predicted += 1
        logger.info("nba backtest %s→%s: %s (train to date)", block_start, block_end, len(rows))
        block_start = block_end + timedelta(days=1)
    reconciled = reconcile(cfg)
    return {"predicted": predicted, "reconciled": reconciled}


# ------------------------------- digest ------------------------------------
def format_daily_digest(config: Config | None = None, *, for_date: date | None = None) -> str:
    from sandy.mls.recommend import evaluate, load_reliability, meta_gate
    cfg = config or load_config()
    engine = create_engine(cfg)
    sim = for_date is not None  # render a historical day from backtest rows
    today = for_date or datetime.now(DISPLAY_TZ).date()
    parts = [f"🏀 NBA ({today.strftime('%b %d')})"]
    with engine.begin() as conn:
        reliability = load_reliability(conn, "nba")
        night = conn.execute(text(f"""
            SELECT home_team, away_team, actual_home_points, actual_away_points, was_correct_over_225_5
            FROM nba.game_predictions
            WHERE match_date = :d AND outcome_filled_at_utc IS NOT NULL
              AND {"is_backtest" if sim else "NOT is_backtest"}
            ORDER BY id LIMIT 10
        """), {"d": today - timedelta(days=1)}).fetchall()
        if night:
            hits = sum(1 for r in night if r.was_correct_over_225_5)
            parts.append(f"🌙 Anoche O225.5: {hits}/{len(night)} · " + " · ".join(
                f"{r.home_team} {r.actual_home_points}-{r.actual_away_points} {r.away_team}" for r in night[:5]))
        rows = conn.execute(text(f"""
            SELECT * FROM nba.game_predictions
            WHERE match_date BETWEEN :a AND :b
              AND {"is_backtest" if sim else "outcome_filled_at_utc IS NULL AND NOT is_backtest"}
            ORDER BY match_date, id
        """), {"a": today, "b": today + timedelta(days=1)}).fetchall()
        parts.append("")
        if not rows:
            parts.append("😴 No hay juegos NBA hoy.")
        else:
            recs = []
            for r in rows:
                cands = []
                w = evaluate(reliability, "winner", r.p_home_win, "Gana local", "Gana visitante")
                if w:
                    cands.append(w)
                for market, col, thr in (("over_210_5", "p_over_210_5", 210.5),
                                         ("over_215_5", "p_over_215_5", 215.5),
                                         ("over_220_5", "p_over_220_5", 220.5),
                                         ("over_225_5", "p_over_225_5", 225.5),
                                         ("over_230_5", "p_over_230_5", 230.5),
                                         ("over_235_5", "p_over_235_5", 235.5),
                                         ("over_240_5", "p_over_240_5", 240.5),
                                         ("over_245_5", "p_over_245_5", 245.5)):
                    c = evaluate(reliability, market, getattr(r, col),
                                 f"Más de {thr} puntos", f"Menos de {thr} puntos")
                    if c:
                        cands.append(c)
                for c in meta_gate("nba", cfg, r, cands):
                    recs.append((r, c))
            recs.sort(key=lambda x: (-x[1].get("meta", 0), -x[1]["hist_acc"]))
            if recs:
                parts.append("🎯 APUESTAS RECOMENDADAS:")
                for r, c in recs[:8]:
                    meta = f" · 🤖 {c['meta']:.0%}" if c.get("meta") is not None else ""
                    parts.append(f"• {r.home_team} vs {r.away_team} → {c['label']} "
                                 f"({c['conf']:.0%}) · hist {c['hist_acc']:.0%}{meta}")
            else:
                parts.append("🎯 Ningún pick supera el filtro hoy.")
            parts.append("")
            parts.append(f"📋 Todos ({len(rows)}):")
            for r in rows:
                parts.append(f"· {r.home_team} vs {r.away_team} — total esperado {r.exp_total:.0f} | "
                             f"O225.5 {r.p_over_225_5:.0%} | gana local {r.p_home_win:.0%}")
    parts.append("")
    parts.append("ℹ️ totales = puntos de ambos equipos · hist = % acierto real · 🤖 = meta-modelo")
    return "\n".join(parts)


def notify_daily(config: Config | None = None) -> bool:
    msg = format_daily_digest(config)
    ok = send_telegram(msg)
    # On game days, follow with the MLB-style meta-model reliability ladder
    # (separate message so a full slate can never overflow Telegram's limit).
    if "😴" not in msg:
        from sandy.betmeta import format_meta_ladder
        ladder = format_meta_ladder("nba", config or load_config())
        if ladder:
            send_telegram("🏀 NBA — fiabilidad del meta-modelo (histórico)\n\n" + ladder)
    logger.info("nba digest sent: %s", ok)
    return ok
