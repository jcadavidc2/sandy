"""Confederation scouting report (the original-prompt deliverable).

Per team, over the last N official matches (most recent first): an estimated
W/D/L (from the Dixon-Coles model), goals for/against per game, Over/Under 2.5
trend, BTTS%, last-5 form, a HOT STAT, and the stat-sheet columns
(corners/cards/possession/shots on target). Teams are grouped by confederation,
derived from the confederation-specific qualifier/continental leagues they've
played in. Honest about gaps: columns with no data render "insufficient data".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Exclude youth / Olympic / women's sides that leak in via friendlies.
_NON_SENIOR = re.compile(r"\bU(15|16|17|18|19|20|21|23)\b|Olympic|Women|\bW\b", re.IGNORECASE)

from sandy.config import Config
from sandy.football.predictor import compute_scoreline_matrix, markets_from_matrix

# Confederation-specific leagues (excludes friendlies + World Cup, which mix
# confederations). A team's confederation = the one it played most in.
LEAGUE_CONFEDERATION: dict[int, str] = {
    32: "UEFA", 5: "UEFA", 4: "UEFA", 960: "UEFA",
    29: "CAF", 36: "CAF", 6: "CAF", 859: "CAF", 19: "CAF", 1008: "CAF",
    30: "AFC", 35: "AFC", 7: "AFC", 25: "AFC", 24: "AFC", 28: "AFC",
    31: "CONCACAF", 536: "CONCACAF", 22: "CONCACAF", 858: "CONCACAF",
    34: "CONMEBOL", 9: "CONMEBOL",
    33: "OFC",
}
INSUFFICIENT = "insufficient data"


def team_confederations(engine: Engine) -> dict[int, str]:
    """Most-played confederation per team, from confederation-specific leagues."""
    sql = text("""
        SELECT team_id, league_id, count(*) AS n FROM (
            SELECT home_team_id AS team_id, league_id FROM football.matches
            UNION ALL
            SELECT away_team_id AS team_id, league_id FROM football.matches
        ) s
        WHERE league_id = ANY(:lids)
        GROUP BY team_id, league_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"lids": list(LEAGUE_CONFEDERATION)}).fetchall()
    best: dict[int, tuple[int, str]] = {}
    for team_id, league_id, n in rows:
        conf = LEAGUE_CONFEDERATION[league_id]
        if team_id not in best or n > best[team_id][0]:
            best[team_id] = (n, conf)
    return {tid: c for tid, (_, c) in best.items()}


def _hot_stat(results: list[str], gf: list[int], ga: list[int]) -> str:
    """Pick the most notable recent trend from the last 10 matches."""
    r10, gf10, ga10 = results[:10], gf[:10], ga[:10]
    # Unbeaten streak from most recent.
    unbeaten = 0
    for x in results:
        if x == "L":
            break
        unbeaten += 1
    scored_in = sum(1 for g in gf10 if g >= 1)
    wins = r10.count("W")
    clean = sum(1 for g in ga10 if g == 0)
    if unbeaten >= 5:
        return f"Unbeaten in last {unbeaten}"
    if wins >= 7:
        return f"Won {wins} of last 10"
    if clean >= 5:
        return f"{clean} clean sheets in last 10"
    return f"Scored in {scored_in} of last 10"


def _stat_avgs(engine: Engine, team_id: int, fixture_ids: list[int]) -> dict:
    """Average corners/cards/possession/SoT over the given fixtures (if any)."""
    if not fixture_ids:
        return {}
    sql = text("""
        SELECT avg(possession) poss, avg(shots_on_target) sot,
               avg(corners) corners, avg(yellow_cards + red_cards) cards,
               count(*) n
        FROM football.match_stats
        WHERE team_id = :tid AND fixture_id = ANY(:fids)
    """)
    with engine.connect() as conn:
        r = conn.execute(sql, {"tid": team_id, "fids": fixture_ids}).mappings().first()
    if not r or not r["n"]:
        return {}
    return {k: (float(v) if v is not None else None) for k, v in r.items() if k != "n"}


def build_form_report(
    engine: Engine, cfg: Config, *, last_n: int = 20,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return (report_by_confederation, top_picks)."""
    from sandy.football.ratings import ARTIFACT_NAME, load_model
    try:
        model = load_model(cfg.model.model_dir / ARTIFACT_NAME)
    except Exception:  # noqa: BLE001
        model = None

    confs = team_confederations(engine)
    names = pd.read_sql("SELECT team_id, name FROM football.teams", engine)
    name_map = dict(zip(names["team_id"], names["name"]))

    # Per-team match perspective, most recent first.
    df = pd.read_sql(text("""
        SELECT home_team_id AS team_id, fixture_id, match_date, home_goals AS gf, away_goals AS ga
        FROM football.matches WHERE status IN ('FT','AET','PEN') AND home_goals IS NOT NULL
        UNION ALL
        SELECT away_team_id, fixture_id, match_date, away_goals, home_goals
        FROM football.matches WHERE status IN ('FT','AET','PEN') AND home_goals IS NOT NULL
    """), engine)
    df = df.sort_values("match_date", ascending=False)

    min_matches = cfg.football.min_matches_for_report
    report: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for team_id, g in df.groupby("team_id"):
        if _NON_SENIOR.search(name_map.get(team_id, "")):
            continue  # skip youth/Olympic/women's sides
        g = g.head(last_n)
        n = len(g)
        gf = g["gf"].astype(int).tolist()
        ga = g["ga"].astype(int).tolist()
        results = ["W" if a > b else ("D" if a == b else "L") for a, b in zip(gf, ga)]
        conf = confs.get(team_id, "Other")

        row: dict = {"team": name_map.get(team_id, str(team_id)), "confederation": conf,
                     "matches": n}
        if n < min_matches:
            row.update({"status": INSUFFICIENT})
            report.setdefault(conf, []).append(row)
            continue

        # Model estimate of W/D/L vs an average team (neutral).
        if model is not None and team_id in model.attack:
            lam = float(np.exp(model.mean_goals + model.attack[team_id]))
            mu = float(np.exp(model.mean_goals - model.defense.get(team_id, 0.0)))
            mk = markets_from_matrix(compute_scoreline_matrix(lam, mu, model.rho))
            est_w, est_d, est_l = mk["p_home_win"], mk["p_draw"], mk["p_away_win"]
        else:
            est_w = est_d = est_l = None

        stats = _stat_avgs(engine, team_id, g["fixture_id"].tolist())
        ppg = (results.count("W") * 3 + results.count("D")) / n

        row.update({
            "est_win": est_w, "est_draw": est_d, "est_loss": est_l,
            "gf_pg": round(float(np.mean(gf)), 2), "ga_pg": round(float(np.mean(ga)), 2),
            "over25_pct": round(float(np.mean([(a + b) > 2.5 for a, b in zip(gf, ga)])) * 100),
            "btts_pct": round(float(np.mean([(a >= 1 and b >= 1) for a, b in zip(gf, ga)])) * 100),
            "last5": "-".join(results[:5]),
            "corners": round(stats["corners"], 1) if stats.get("corners") is not None else INSUFFICIENT,
            "cards": round(stats["cards"], 1) if stats.get("cards") is not None else INSUFFICIENT,
            "possession": round(stats["poss"], 1) if stats.get("poss") is not None else INSUFFICIENT,
            "shots_on_target": round(stats["sot"], 1) if stats.get("sot") is not None else INSUFFICIENT,
            "hot_stat": _hot_stat(results, gf, ga),
            "ppg": round(ppg, 2),
        })
        report.setdefault(conf, []).append(row)
        all_rows.append(row)

    # Sort each confederation by estimated win prob (fallback ppg).
    for conf in report:
        report[conf].sort(key=lambda r: (r.get("est_win") or 0, r.get("ppg", 0)), reverse=True)

    # TOP picks: best recent form across all confederations.
    top = sorted(all_rows, key=lambda r: (r.get("ppg", 0), r.get("est_win") or 0), reverse=True)[:10]
    return report, top


__all__ = ["INSUFFICIENT", "LEAGUE_CONFEDERATION", "build_form_report", "team_confederations"]
