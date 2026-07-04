"""Covariate features for MLS predictions — persisted with every prediction so the
calibration loop can learn what actually helps. Computed strictly from matches
BEFORE the given date (no leakage).

Per team, as of a date:
  goals_for_5/10, goals_against_5/10   rolling means (last 5 / last 10 played)
  corners_for_5, corners_against_5     rolling means where stats exist
  rest_days                            days since last match
  form_points_5                        points from last 5 (W3/D1/L0)
  home_gf_5 / away_gf_5                venue-split scoring (last 5 at that venue)
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Engine


def team_form(engine: Engine, team_id: int, as_of: date) -> dict:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT match_date, home_team_id, away_team_id, home_goals, away_goals,
                   home_corners, away_corners
            FROM mls.matches
            WHERE status = 'FT' AND home_goals IS NOT NULL
              AND (home_team_id = :t OR away_team_id = :t) AND match_date < :d
            ORDER BY match_date DESC LIMIT 10
        """), {"t": team_id, "d": as_of}).fetchall()
    if not rows:
        return {}
    gf, ga, cf, ca, pts, home_gf, away_gf = [], [], [], [], [], [], []
    for r in rows:
        is_home = r.home_team_id == team_id
        f = r.home_goals if is_home else r.away_goals
        a = r.away_goals if is_home else r.home_goals
        gf.append(f); ga.append(a)
        pts.append(3 if f > a else (1 if f == a else 0))
        (home_gf if is_home else away_gf).append(f)
        c_f = r.home_corners if is_home else r.away_corners
        c_a = r.away_corners if is_home else r.home_corners
        if c_f is not None:
            cf.append(c_f); ca.append(c_a)
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return {
        "goals_for_5": mean(gf[:5]), "goals_against_5": mean(ga[:5]),
        "goals_for_10": mean(gf), "goals_against_10": mean(ga),
        "corners_for_5": mean(cf[:5]), "corners_against_5": mean(ca[:5]),
        "form_points_5": sum(pts[:5]),
        "rest_days": (as_of - rows[0].match_date).days,
        "home_gf_5": mean(home_gf[:5]), "away_gf_5": mean(away_gf[:5]),
        "played_10": len(rows),
    }


def match_features(engine: Engine, home_team_id: int, away_team_id: int, as_of: date) -> dict:
    return {"home": team_form(engine, home_team_id, as_of),
            "away": team_form(engine, away_team_id, as_of)}


def blend_lambda(lam_dc: float, form_gf: float | None, opp_form_ga: float | None,
                 weight: float = 0.2) -> float:
    """Blend the DC expectation with recent-form scoring (form attack vs opp form defense).
    weight=0 → pure Dixon-Coles. The backtest measures whether the blend earns its keep."""
    if form_gf is None or opp_form_ga is None or weight <= 0:
        return lam_dc
    lam_form = (form_gf + opp_form_ga) / 2.0
    return (1 - weight) * lam_dc + weight * lam_form
