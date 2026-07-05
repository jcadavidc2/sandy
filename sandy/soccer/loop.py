"""Multi-league soccer: models + predictor + reconciler + calibrator + backtest +
digest in one module (per-league Dixon-Coles fits, shared machinery from football/mls)."""
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
from sandy.football.predictor import compute_scoreline_matrix, markets_from_matrix
from sandy.football.ratings import DixonColesModel, fit_dixon_coles, load_model, save_model
from sandy.mls.parsers import DISPLAY_TZ
from sandy.mls.predictor import CORNER_MAX, corner_overs
from sandy.mls.schemas import GOAL_THRESHOLDS
from sandy.over_under.notifier import send_telegram

from . import LEAGUES

logger = logging.getLogger(__name__)

XI = 0.0038
FORM_BLEND_WEIGHT = 0.2
MIN_TRAIN = 250
MIN_SAMPLES = 30
BUCKETS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]


# ------------------------------ data + form --------------------------------
def _load_df(engine: Engine, league: str, cols: str, extra: str, as_of: date | None) -> pd.DataFrame:
    where = f"league = :lg AND status = 'FT'{extra}"
    params: dict = {"lg": league}
    if as_of is not None:
        where += " AND match_date < :as_of"
        params["as_of"] = as_of
    with engine.begin() as conn:
        return pd.read_sql(text(f"""
            SELECT home_team_id, away_team_id, {cols}, match_date, 1.0 AS w
            FROM soccer.matches WHERE {where} ORDER BY match_date
        """), conn, params=params)


def load_goals(engine, league, as_of=None):
    return _load_df(engine, league, "home_goals, away_goals",
                    " AND home_goals IS NOT NULL", as_of)


def load_corners(engine, league, as_of=None):
    return _load_df(engine, league, "home_corners AS home_goals, away_corners AS away_goals",
                    " AND home_corners IS NOT NULL", as_of)


def team_form(engine: Engine, team_id: int, as_of: date) -> dict:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT match_date, home_team_id, home_goals, away_goals, home_corners, away_corners
            FROM soccer.matches
            WHERE status = 'FT' AND home_goals IS NOT NULL
              AND (home_team_id = :t OR away_team_id = :t) AND match_date < :d
            ORDER BY match_date DESC LIMIT 10
        """), {"t": team_id, "d": as_of}).fetchall()
    if not rows:
        return {}
    gf, ga, cf, ca, pts = [], [], [], [], []
    for r in rows:
        is_home = r.home_team_id == team_id
        f = r.home_goals if is_home else r.away_goals
        a = r.away_goals if is_home else r.home_goals
        gf.append(f); ga.append(a)
        pts.append(3 if f > a else (1 if f == a else 0))
        c_f = r.home_corners if is_home else r.away_corners
        if c_f is not None:
            cf.append(c_f); ca.append(r.away_corners if is_home else r.home_corners)
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return {"goals_for_5": mean(gf[:5]), "goals_against_5": mean(ga[:5]),
            "goals_for_10": mean(gf), "goals_against_10": mean(ga),
            "corners_for_5": mean(cf[:5]), "corners_against_5": mean(ca[:5]),
            "form_points_5": sum(pts[:5]), "rest_days": (as_of - rows[0].match_date).days,
            "played_10": len(rows)}


def _blend(lam, form_gf, opp_ga):
    if form_gf is None or opp_ga is None:
        return lam
    return (1 - FORM_BLEND_WEIGHT) * lam + FORM_BLEND_WEIGHT * (form_gf + opp_ga) / 2.0


# ------------------------------ ratings ------------------------------------
def _paths(cfg: Config, league: str):
    return (cfg.model.model_dir / f"soccer_{league}_goals.json",
            cfg.model.model_dir / f"soccer_{league}_corners.json")


def fit_league(engine: Engine, cfg: Config, league: str) -> dict:
    gdf = load_goals(engine, league)
    goals = fit_dixon_coles(gdf, as_of_date=date.today(), xi=XI)
    cdf = load_corners(engine, league)
    corners = fit_dixon_coles(cdf, as_of_date=date.today(), xi=XI) if len(cdf) >= MIN_TRAIN else goals
    gp, cp = _paths(cfg, league)
    save_model(goals, gp)
    save_model(corners, cp)
    with engine.begin() as conn:
        for tid in set(goals.attack) | set(corners.attack):
            conn.execute(text("""
                INSERT INTO soccer.team_ratings (team_id, league, as_of_date, attack, defense,
                                                 corner_attack, corner_defense)
                VALUES (:t, :lg, :d, :a, :de, :ca, :cd)
                ON CONFLICT (team_id, league, as_of_date) DO UPDATE SET attack=:a, defense=:de,
                    corner_attack=:ca, corner_defense=:cd
            """), {"t": tid, "lg": league, "d": goals.as_of_date,
                   "a": goals.attack.get(tid), "de": goals.defense.get(tid),
                   "ca": corners.attack.get(tid), "cd": corners.defense.get(tid)})
    logger.info("soccer[%s] fit: goals n=%s corners n=%s home_adv=%.3f",
                league, goals.n_matches, corners.n_matches, goals.home_adv)
    return {"goals": goals.n_matches, "corners": corners.n_matches}


def fit_all(config: Config | None = None) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    return {lg: fit_league(engine, cfg, lg) for lg in LEAGUES}


# ------------------------------ predict ------------------------------------
def _predict_row(engine, goals: DixonColesModel, corners: DixonColesModel, league, row,
                 as_of: date, is_backtest=False, with_features=True) -> dict:
    eid, mdate, hid, aid, hname, aname = row
    feats = ({"home": team_form(engine, hid, as_of), "away": team_form(engine, aid, as_of)}
             if with_features else {})
    lam, mu = goals.expected_goals(hid, aid)
    hf, af = feats.get("home") or {}, feats.get("away") or {}
    lam = _blend(lam, hf.get("goals_for_5"), af.get("goals_against_5"))
    mu = _blend(mu, af.get("goals_for_5"), hf.get("goals_against_5"))
    mk = markets_from_matrix(compute_scoreline_matrix(lam, mu, goals.rho),
                             thresholds=GOAL_THRESHOLDS)
    clam, cmu = corners.expected_goals(hid, aid)
    c_over = corner_overs(compute_scoreline_matrix(clam, cmu, 0.0, max_goals=CORNER_MAX))
    return {"eid": eid, "lg": league, "d": mdate, "hid": hid, "aid": aid, "hn": hname, "an": aname,
            "lh": round(lam, 3), "la": round(mu, 3),
            "phw": mk["p_home_win"], "pd": mk["p_draw"], "paw": mk["p_away_win"],
            "phd": mk["p_home_win"] + mk["p_draw"],
            "o05": mk["p_over"][0.5], "o15": mk["p_over"][1.5], "o25": mk["p_over"][2.5],
            "o35": mk["p_over"][3.5], "o45": mk["p_over"][4.5], "o55": mk["p_over"][5.5],
            "clh": round(clam, 2), "cla": round(cmu, 2),
            "c75": c_over[7.5], "c85": c_over[8.5], "c95": c_over[9.5], "c105": c_over[10.5],
            "c115": c_over[11.5], "c125": c_over[12.5],
            "mlh": mk["most_likely"][0], "mla": mk["most_likely"][1],
            "feats": json.dumps(feats) if feats else None, "bt": is_backtest,
            "now": datetime.now(timezone.utc)}


_UPSERT = text("""
    INSERT INTO soccer.match_predictions (
        event_id, league, match_date, home_team_id, away_team_id, home_team, away_team,
        lambda_home, lambda_away, p_home_win, p_draw, p_away_win, p_home_or_draw,
        p_over_0_5, p_over_1_5, p_over_2_5, p_over_3_5, p_over_4_5, p_over_5_5,
        corner_lambda_home, corner_lambda_away,
        p_corners_over_7_5, p_corners_over_8_5, p_corners_over_9_5, p_corners_over_10_5,
        p_corners_over_11_5, p_corners_over_12_5,
        most_likely_home, most_likely_away, features, is_backtest, predicted_at_utc)
    VALUES (:eid, :lg, :d, :hid, :aid, :hn, :an, :lh, :la, :phw, :pd, :paw, :phd,
            :o05, :o15, :o25, :o35, :o45, :o55, :clh, :cla,
            :c75, :c85, :c95, :c105, :c115, :c125,
            :mlh, :mla, :feats, :bt, :now)
    ON CONFLICT (event_id) DO UPDATE SET
        lambda_home=:lh, lambda_away=:la, p_home_win=:phw, p_draw=:pd, p_away_win=:paw,
        p_home_or_draw=:phd, p_over_0_5=:o05, p_over_1_5=:o15, p_over_2_5=:o25,
        p_over_3_5=:o35, p_over_4_5=:o45, p_over_5_5=:o55,
        corner_lambda_home=:clh, corner_lambda_away=:cla,
        p_corners_over_7_5=:c75, p_corners_over_8_5=:c85, p_corners_over_9_5=:c95,
        p_corners_over_10_5=:c105, p_corners_over_11_5=:c115, p_corners_over_12_5=:c125,
        most_likely_home=:mlh, most_likely_away=:mla,
        features=:feats, is_backtest=:bt, predicted_at_utc=:now
""")

_SCHED_SQL = """
    SELECT m.event_id, m.match_date, m.home_team_id, m.away_team_id, t1.name, t2.name
    FROM soccer.matches m
    JOIN soccer.teams t1 ON t1.team_id = m.home_team_id
    JOIN soccer.teams t2 ON t2.team_id = m.away_team_id
    WHERE m.league = :lg AND m.status = {status} AND m.match_date BETWEEN :a AND :b
    ORDER BY m.match_date, m.event_id
"""


def predict_scheduled(config: Config | None = None, *, days_ahead: int = 2) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    today = datetime.now(DISPLAY_TZ).date()
    n = 0
    for lg in LEAGUES:
        gp, cp = _paths(cfg, lg)
        if not gp.exists():
            continue
        goals, corners = load_model(gp), load_model(cp)
        with engine.begin() as conn:
            rows = conn.execute(text(_SCHED_SQL.format(status="'NS'")),
                                {"lg": lg, "a": today, "b": today + timedelta(days=days_ahead)}).fetchall()
            for r in rows:
                conn.execute(_UPSERT, _predict_row(engine, goals, corners, lg, r, today))
                n += 1
    logger.info("soccer predicted %s matches across leagues", n)
    return n


# ------------------------------ reconcile ----------------------------------
def reconcile(config: Config | None = None) -> int:
    cfg = config or load_config()
    engine = create_engine(cfg)
    now = datetime.now(timezone.utc)
    n = 0
    with engine.begin() as conn:
        for r in conn.execute(text("""
            SELECT p.id, p.p_home_or_draw, p.p_over_2_5, p.p_corners_over_9_5,
                   m.home_goals, m.away_goals, m.home_corners, m.away_corners
            FROM soccer.match_predictions p
            JOIN soccer.matches m ON m.event_id = p.event_id
            WHERE p.outcome_filled_at_utc IS NULL AND m.status = 'FT'
              AND m.home_goals IS NOT NULL
        """)).fetchall():
            pid, p_dc, p_ou, p_c95, hg, ag, hc, ac = r
            hg, ag = int(hg), int(ag)
            tot = hg + ag
            res = "H" if hg > ag else ("A" if ag > hg else "D")
            tc = (hc + ac) if (hc is not None and ac is not None) else None
            conn.execute(text("""
                UPDATE soccer.match_predictions SET
                    actual_home_goals=:hg, actual_away_goals=:ag, actual_total_goals=:tot,
                    actual_result=:res, actual_total_corners=:tc,
                    was_correct_double_chance=:c_dc, was_correct_over_2_5=:c_ou,
                    was_correct_corners_9_5=:c_c, outcome_filled_at_utc=:now
                WHERE id=:id
            """), {"id": pid, "hg": hg, "ag": ag, "tot": tot, "res": res, "tc": tc,
                   "c_dc": (p_dc >= 0.5) == (res != "A") if p_dc is not None else None,
                   "c_ou": (p_ou >= 0.5) == (tot > 2.5) if p_ou is not None else None,
                   "c_c": (p_c95 >= 0.5) == (tc > 9.5) if (p_c95 is not None and tc is not None) else None,
                   "now": now})
            n += 1
    logger.info("soccer reconciled %s", n)
    return n


# ------------------------------ calibrate ----------------------------------
def _goal_line(pcol, thr):
    return (pcol, "actual_total_goals", lambda r: (r[pcol] >= 0.5) == (r["actual_total_goals"] > thr))


def _corner_line(pcol, thr):
    return (pcol, "actual_total_corners", lambda r: (r[pcol] >= 0.5) == (r["actual_total_corners"] > thr))


MARKETS = {
    "double_chance": ("p_home_or_draw", "actual_result",
                      lambda r: (r["p_home_or_draw"] >= 0.5) == (r["actual_result"] != "A")),
    "over_0_5": _goal_line("p_over_0_5", 0.5), "over_1_5": _goal_line("p_over_1_5", 1.5),
    "over_2_5": _goal_line("p_over_2_5", 2.5), "over_3_5": _goal_line("p_over_3_5", 3.5),
    "over_4_5": _goal_line("p_over_4_5", 4.5), "over_5_5": _goal_line("p_over_5_5", 5.5),
    "corners_over_7_5": _corner_line("p_corners_over_7_5", 7.5),
    "corners_over_8_5": _corner_line("p_corners_over_8_5", 8.5),
    "corners_over_9_5": _corner_line("p_corners_over_9_5", 9.5),
    "corners_over_10_5": _corner_line("p_corners_over_10_5", 10.5),
    "corners_over_11_5": _corner_line("p_corners_over_11_5", 11.5),
    "corners_over_12_5": _corner_line("p_corners_over_12_5", 12.5),
}


def calibrate(config: Config | None = None, *, lookback_days: int | None = None) -> list[dict]:
    cfg = config or load_config()
    engine = create_engine(cfg)
    where = "outcome_filled_at_utc IS NOT NULL"
    if lookback_days:
        where += f" AND match_date >= CURRENT_DATE - INTERVAL '{int(lookback_days)} days'"
    with engine.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM soccer.match_predictions WHERE {where}"), conn)
    snaps = []
    for lg in LEAGUES:
        sub_lg = df[df["league"] == lg]
        for market, (pcol, actual_col, fn) in MARKETS.items():
            sub = sub_lg.dropna(subset=[pcol, actual_col])
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
            snaps.append({"snapshot_date": date.today(), "league": lg, "market": market,
                          "lookback_days": lookback_days, "sample_size": int(len(sub)),
                          "accuracy": round(float(correct.mean()), 4),
                          "brier": round(float(np.mean((p - outcome_yes.astype(float)) ** 2)), 4),
                          "reliability": table, "recommended_threshold": thr})
    with engine.begin() as conn:
        for s in snaps:
            conn.execute(text("""
                INSERT INTO soccer.calibration_snapshots
                    (snapshot_date, league, market, lookback_days, sample_size, accuracy, brier,
                     reliability, recommended_threshold)
                VALUES (:snapshot_date, :league, :market, :lookback_days, :sample_size,
                        :accuracy, :brier, :rel, :recommended_threshold)
            """), {**s, "rel": json.dumps(s["reliability"])})
    logger.info("soccer calibrated %s league-market snapshots", len(snaps))
    return snaps


# ------------------------------ backtest -----------------------------------
def run_backtest(config: Config | None = None, *, refit_days: int = 14) -> dict:
    cfg = config or load_config()
    engine = create_engine(cfg)
    predicted = 0
    for lg in LEAGUES:
        with engine.begin() as conn:
            lo, hi = conn.execute(text(
                "SELECT MIN(match_date), MAX(match_date) FROM soccer.matches WHERE league=:lg AND status='FT'"
            ), {"lg": lg}).fetchone()
        if lo is None:
            continue
        block_start = lo + timedelta(days=270)
        while block_start <= hi:
            block_end = min(block_start + timedelta(days=refit_days - 1), hi)
            gdf = load_goals(engine, lg, as_of=block_start)
            if len(gdf) >= MIN_TRAIN:
                goals = fit_dixon_coles(gdf, as_of_date=block_start, xi=XI)
                cdf = load_corners(engine, lg, as_of=block_start)
                corners = fit_dixon_coles(cdf, as_of_date=block_start, xi=XI) if len(cdf) >= MIN_TRAIN else goals
                with engine.begin() as conn:
                    rows = conn.execute(text(_SCHED_SQL.format(status="'FT'")),
                                        {"lg": lg, "a": block_start, "b": block_end}).fetchall()
                    for r in rows:
                        conn.execute(_UPSERT, _predict_row(engine, goals, corners, lg, r, r[1],
                                                           is_backtest=True))
                        predicted += 1
                logger.info("soccer[%s] backtest %s→%s: %s (train %s)",
                            lg, block_start, block_end, len(rows), len(gdf))
            block_start = block_end + timedelta(days=1)
    reconciled = reconcile(cfg)
    return {"predicted": predicted, "reconciled": reconciled}


# ------------------------------ digest --------------------------------------
def format_daily_digest(config: Config | None = None, *, for_date: date | None = None) -> str:
    from sandy.mls.recommend import evaluate, meta_gate
    cfg = config or load_config()
    engine = create_engine(cfg)
    sim = for_date is not None  # render a historical day from backtest rows
    today = for_date or datetime.now(DISPLAY_TZ).date()
    parts = [f"🌍 Fútbol Ligas ({today.strftime('%b %d')})"]
    any_games = False
    with engine.begin() as conn:
        for lg, (_code, name, flag, _m) in LEAGUES.items():
            rel_rows = conn.execute(text("""
                SELECT DISTINCT ON (market) market, reliability FROM soccer.calibration_snapshots
                WHERE league = :lg ORDER BY market, created_at DESC
            """), {"lg": lg}).fetchall()
            reliability = {m: (r if isinstance(r, list) else json.loads(r)) for m, r in rel_rows}
            rows = conn.execute(text(f"""
                SELECT * FROM soccer.match_predictions
                WHERE league = :lg AND match_date BETWEEN :a AND :b
                  AND {"is_backtest" if sim else "outcome_filled_at_utc IS NULL AND NOT is_backtest"}
                ORDER BY match_date, id
            """), {"lg": lg, "a": today, "b": today + timedelta(days=1)}).fetchall()
            night = conn.execute(text(f"""
                SELECT home_team, away_team, actual_home_goals, actual_away_goals, was_correct_double_chance
                FROM soccer.match_predictions
                WHERE league = :lg AND match_date = :d AND outcome_filled_at_utc IS NOT NULL
                  AND {"is_backtest" if sim else "NOT is_backtest"} ORDER BY id LIMIT 6
            """), {"lg": lg, "d": today - timedelta(days=1)}).fetchall()
            if not rows and not night:
                continue
            any_games = True
            parts.append("")
            parts.append(f"{flag} {name}")
            if night:
                hits = sum(1 for r in night if r.was_correct_double_chance)
                parts.append(f"  🌙 Ayer {hits}/{len(night)}: " + " · ".join(
                    f"{'✅' if r.was_correct_double_chance else '❌'} {r.home_team} {r.actual_home_goals}-{r.actual_away_goals} {r.away_team}"
                    for r in night[:4]))
            recs = []
            for r in rows:
                cands = []
                dc = evaluate(reliability, "double_chance", r.p_home_or_draw,
                              "Local o empata (1X)", "Gana visitante (2)")
                if dc:
                    cands.append(dc)
                for market, col, thr in (("over_0_5", "p_over_0_5", 0.5), ("over_1_5", "p_over_1_5", 1.5),
                                         ("over_2_5", "p_over_2_5", 2.5), ("over_3_5", "p_over_3_5", 3.5),
                                         ("over_4_5", "p_over_4_5", 4.5), ("over_5_5", "p_over_5_5", 5.5),
                                         ("corners_over_7_5", "p_corners_over_7_5", 7.5),
                                         ("corners_over_9_5", "p_corners_over_9_5", 9.5),
                                         ("corners_over_11_5", "p_corners_over_11_5", 11.5),
                                         ("corners_over_12_5", "p_corners_over_12_5", 12.5)):
                    c = evaluate(reliability, market, getattr(r, col),
                                 f"Más de {thr} {'corners' if 'corners' in market else 'goles'}",
                                 f"Menos de {thr} {'corners' if 'corners' in market else 'goles'}")
                    if c:
                        cands.append(c)
                for c in meta_gate(f"soccer_{lg}", cfg, r, cands):
                    recs.append((r, c))
            recs.sort(key=lambda x: (-x[1].get("meta", 0), -x[1]["hist_acc"]))
            if recs:
                parts.append("  🎯 Recomendadas:")
                from sandy.betrefine import star_for_candidate
                for r, c in recs[:5]:
                    meta = f" · 🤖 {c['meta']:.0%}" if c.get("meta") is not None else ""
                    star = star_for_candidate(f"soccer_{lg}", cfg, r, c)
                    parts.append(f"  • {star}{r.home_team} vs {r.away_team} → {c['label']} "
                                 f"({c['conf']:.0%}) · hist {c['hist_acc']:.0%}{meta}")
            elif rows:
                parts.append(f"  📋 {len(rows)} partidos — ninguno supera el filtro hoy.")
    if not any_games:
        parts.append("😴 No hay partidos hoy en Colombia, México, España ni Inglaterra.")
    parts.append("")
    parts.append("ℹ️ 1X = local gana o empata · hist = % acierto real a esta confianza · 🤖 = meta-modelo")
    return "\n".join(parts)


def notify_daily(config: Config | None = None) -> bool:
    cfg = config or load_config()
    ok = send_telegram(format_daily_digest(cfg))
    # One MLB-style meta-model reliability ladder per league WITH games today —
    # separate messages so a full weekend slate can never overflow Telegram's limit.
    from sandy.betmeta import format_meta_ladder
    engine = create_engine(cfg)
    today = datetime.now(DISPLAY_TZ).date()
    with engine.begin() as conn:
        active = {r[0] for r in conn.execute(text("""
            SELECT DISTINCT league FROM soccer.match_predictions
            WHERE match_date BETWEEN :a AND :b
              AND outcome_filled_at_utc IS NULL AND NOT is_backtest
        """), {"a": today, "b": today + timedelta(days=1)})}
    for lg, (_code, name, flag, _m) in LEAGUES.items():
        if lg not in active:
            continue
        ladder = format_meta_ladder(f"soccer_{lg}", cfg)
        if ladder:
            send_telegram(f"{flag} {name} — fiabilidad del meta-modelo (histórico)\n\n{ladder}")
    logger.info("soccer digest sent: %s", ok)
    return ok
