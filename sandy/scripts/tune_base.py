"""Base-model hyperparameter tuning via honest walk-forward evaluation.

Methodology (identical for every vertical):
  * Walk-forward: train strictly-prior, predict forward in refit-day blocks —
    exactly the shape of the production backtests (14d blocks; 30d for the
    worldcup vertical, matching football/backtest.py).
  * Metric: mean log-loss of the league's binary market probabilities, pooled
    over all its markets (one term per (match, market) with a graded outcome).
  * Split: the FINAL 25% of predicted history (by date quantile) is the
    untouched JUDGE window; hyperparameters are chosen on the earlier 75%
    (TUNE) only. A candidate ships only if it beats the CURRENT hypers on the
    judge window — both evaluated under this same harness (same refit cadence,
    same warm-started optimizer), so the comparison is apples-to-apples.

Speed: Dixon-Coles fits are warm-started across blocks (fixed team-index over
the league's full history; unseen-so-far teams are pinned to ~0 by the same L2
regularizer production uses). The objective/bounds/penalties are copied from
sandy.football.ratings.fit_dixon_coles verbatim.

Outputs one JSON per vertical under --out (default: models/tuning/) with the
tune/judge tables; writing models/hyper_{league}.json is a SEPARATE, explicit
decision made by the operator after reading the gate results.

Usage:
    python scripts/tune_base.py soccer_col soccer_mex ... mls nhl nba nfl worldcup
"""
from __future__ import annotations

import json
import sys
import time
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sqlalchemy import text

from sandy.config import load_config
from sandy.db import create_engine
from sandy.football.ratings import _tau

XI_GRID = [0.001, 0.002, 0.0038, 0.006, 0.01]
BLEND_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
B2B_GRID = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90]
NBA_HL_GRID = [60, 90, 120, 200, 300]
NFL_HL_GRID = [100, 200, 300, 400]
RIDGE_GRID = [0.0, 1.0, 5.0, 20.0]
EPS = 1e-6

GOAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
NHL_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
NBA_LINES = [210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5, 245.5]
NFL_LINES = [37.5, 41.5, 44.5, 47.5, 51.5]


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------ DC fits --
class WarmDC:
    """Warm-started Dixon-Coles MLE over a FIXED team index (same objective,
    bounds and penalties as sandy.football.ratings.fit_dixon_coles)."""

    def __init__(self, teams: list):
        self.idx = {t: i for i, t in enumerate(teams)}
        self.n = len(teams)
        self.x = np.concatenate([np.zeros(2 * self.n), [0.25, -0.05]])

    def fit(self, df: pd.DataFrame, as_of: date, xi: float, reg: float = 0.01,
            max_iter: int = 200):
        n = self.n
        hi = df["home_team_id"].map(self.idx).to_numpy()
        ai = df["away_team_id"].map(self.idx).to_numpy()
        hg = df["home_goals"].to_numpy(dtype=float)
        ag = df["away_goals"].to_numpy(dtype=float)
        days = (pd.Timestamp(as_of) - pd.to_datetime(df["match_date"])).dt.days
        w = df["w"].to_numpy(dtype=float) * np.exp(-xi * np.clip(days.to_numpy(dtype=float), 0, None))
        mean_goals = float(np.log(max((hg.sum() + ag.sum()) / (2.0 * len(df)), 1e-3)))

        def nll(p):
            attack, defense, adv, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            loglam = mean_goals + attack[hi] - defense[ai] + adv
            logmu = mean_goals + attack[ai] - defense[hi]
            lam, mu = np.exp(loglam), np.exp(logmu)
            tau = _tau(hg, ag, lam, mu, rho)
            ll = np.log(tau) + hg * loglam - lam + ag * logmu - mu
            neg = -np.sum(w * ll)
            neg += reg * (np.sum(attack ** 2) + np.sum(defense ** 2))
            neg += 100.0 * (attack.mean() ** 2)
            return neg

        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
        res = minimize(nll, self.x, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "maxfun": max_iter * 50})
        self.x = res.x
        attack, defense = res.x[:n], res.x[n:2 * n]
        return {"attack": attack - attack.mean(), "defense": defense,
                "adv": float(res.x[2 * n]), "rho": float(res.x[2 * n + 1]),
                "mean_goals": mean_goals}


def dc_walk(df: pd.DataFrame, xi: float, refit_days: int, offset_days: int,
            min_train: int) -> pd.DataFrame:
    """Walk-forward DC lambdas per match. Returns rows for predicted matches:
    [row index of df] lam, mu, rho, plus passthrough columns."""
    teams = sorted(set(df["home_team_id"]) | set(df["away_team_id"]))
    warm = WarmDC(teams)
    lo, hi = df["match_date"].min(), df["match_date"].max()
    out = []
    block_start = lo + timedelta(days=offset_days)
    model = None
    while block_start <= hi:
        block_end = block_start + timedelta(days=refit_days - 1)
        train = df[df["match_date"] < block_start]
        if len(train) >= min_train:
            model = warm.fit(train, block_start, xi)
        if model is not None:
            blk = df[(df["match_date"] >= block_start) & (df["match_date"] <= block_end)]
            for i, r in blk.iterrows():
                ah = model["attack"][warm.idx[r["home_team_id"]]]
                dh = model["defense"][warm.idx[r["home_team_id"]]]
                aa = model["attack"][warm.idx[r["away_team_id"]]]
                da = model["defense"][warm.idx[r["away_team_id"]]]
                adv = 0.0 if r.get("neutral", False) else model["adv"]
                lam = float(np.exp(model["mean_goals"] + ah - da + adv))
                mu = float(np.exp(model["mean_goals"] + aa - dh))
                out.append({"i": i, "match_date": r["match_date"], "lam": lam, "mu": mu,
                            "rho": model["rho"]})
        block_start = block_end + timedelta(days=1)
    return pd.DataFrame(out)


def score_matrix(lam, mu, rho, max_goals=10):
    i = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(i, lam), poisson.pmf(i, mu))
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    s = m.sum()
    return m / s if s > 0 else m


# --------------------------------------------------------------- form utils --
def build_form(df: pd.DataFrame, k: int = 5):
    """Per (team, match) rolling last-k means of goals for/against, strictly
    before the row's match_date (same-date games excluded — mirrors team_form's
    match_date < :d). Returns dict: df-index -> (h_gf, h_ga, a_gf, a_ga)."""
    hist: dict = {}
    for r in df.sort_values(["match_date"]).itertuples():
        for team, gf, ga in ((r.home_team_id, r.home_goals, r.away_goals),
                             (r.away_team_id, r.away_goals, r.home_goals)):
            dates, gfs, gas = hist.setdefault(team, ([], [], []))
            dates.append(r.match_date)
            gfs.append(float(gf))
            gas.append(float(ga))

    def last5(team, d):
        dv = hist.get(team)
        if not dv:
            return None, None
        dates, gfs, gas = dv
        j = bisect_left(dates, d)
        w_gf, w_ga = gfs[max(0, j - k):j], gas[max(0, j - k):j]
        if not w_gf:
            return None, None
        # production rounds the rolling means to 2dp before blending
        return round(sum(w_gf) / len(w_gf), 2), round(sum(w_ga) / len(w_ga), 2)

    out = {}
    for r in df.itertuples():
        h_gf, h_ga = last5(r.home_team_id, r.match_date)
        a_gf, a_ga = last5(r.away_team_id, r.match_date)
        out[r.Index] = (h_gf, h_ga, a_gf, a_ga)
    return out


def blend(lam, form_gf, opp_ga, wt):
    if form_gf is None or opp_ga is None or wt <= 0:
        return lam
    return (1 - wt) * lam + wt * (form_gf + opp_ga) / 2.0


# ------------------------------------------------------ soccer-family tuner --
def eval_soccer_goals(df, preds, form, wt, lines=GOAL_LINES, max_goals=10):
    """Per predicted match: [date, sum log-loss over DC + over lines, count]."""
    rows = []
    idx = np.arange(max_goals + 1)
    hi_g, aj_g = np.meshgrid(idx, idx, indexing="ij")
    totals = hi_g + aj_g
    for p in preds.itertuples():
        r = df.loc[p.i]
        h_gf, h_ga, a_gf, a_ga = form[p.i]
        lam = blend(p.lam, h_gf, a_ga, wt)
        mu = blend(p.mu, a_gf, h_ga, wt)
        m = score_matrix(lam, mu, p.rho, max_goals)
        p_hd = float(np.tril(m, -1).sum() + np.trace(m))
        tot = r["home_goals"] + r["away_goals"]
        ps = [p_hd] + [float(m[totals > ln].sum()) for ln in lines]
        ys = [1.0 if r["home_goals"] >= r["away_goals"] else 0.0] + \
             [1.0 if tot > ln else 0.0 for ln in lines]
        ll = _ll(np.array(ps), np.array(ys))
        rows.append((p.match_date, float(ll.sum()), len(ll)))
    return pd.DataFrame(rows, columns=["d", "ll", "n"])


def eval_corners(df, preds, lines=CORNER_LINES, max_goals=16):
    rows = []
    idx = np.arange(max_goals + 1)
    hi_g, aj_g = np.meshgrid(idx, idx, indexing="ij")
    totals = hi_g + aj_g
    for p in preds.itertuples():
        r = df.loc[p.i]
        m = score_matrix(p.lam, p.mu, 0.0, max_goals)
        tot = r["home_goals"] + r["away_goals"]  # corners aliased to goals cols
        ps = [float(m[totals > ln].sum()) for ln in lines]
        ys = [1.0 if tot > ln else 0.0 for ln in lines]
        ll = _ll(np.array(ps), np.array(ys))
        rows.append((p.match_date, float(ll.sum()), len(ll)))
    return pd.DataFrame(rows, columns=["d", "ll", "n"])


def split_mean(frame: pd.DataFrame, cut) -> tuple[float, float]:
    d = pd.to_datetime(frame["d"])
    tune = frame[d <= cut]
    judge = frame[d > cut]
    f = lambda fr: float(fr["ll"].sum() / fr["n"].sum()) if fr["n"].sum() else float("nan")  # noqa: E731
    return f(tune), f(judge)


def date_cut(preds: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(preds["match_date"]).quantile(0.75)


def tune_dc_family(name: str, gdf: pd.DataFrame, cdf: pd.DataFrame | None,
                   current: dict, refit_days=14, offset=270, min_train=250,
                   goal_lines=GOAL_LINES, eval_goals=eval_soccer_goals,
                   blend_grid=BLEND_GRID) -> dict:
    t0 = time.time()
    form = build_form(gdf)
    res = {"league": name, "goals": {}, "corners": None}
    # goals: xi × blend
    tables = {}
    for xi in sorted(set(XI_GRID + [current["xi_goals"]])):
        preds = dc_walk(gdf, xi, refit_days, offset, min_train)
        if preds.empty:
            continue
        cut = date_cut(preds)
        for wt in sorted(set(blend_grid + [current["blend_goals"]])):
            fr = eval_goals(gdf, preds, form, wt, lines=goal_lines)
            tables[(xi, wt)] = split_mean(fr, cut)
        print(f"  [{name}] goals xi={xi} done ({time.time()-t0:.0f}s)", flush=True)
    cur_key = (current["xi_goals"], current["blend_goals"])
    best_key = min(tables, key=lambda k: tables[k][0])
    res["goals"] = {
        "grid": {f"xi={k[0]},blend={k[1]}": {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                 for k, v in sorted(tables.items())},
        "current": {"xi": cur_key[0], "blend": cur_key[1],
                    "tune": round(tables[cur_key][0], 5), "judge": round(tables[cur_key][1], 5)},
        "chosen": {"xi": best_key[0], "blend": best_key[1],
                   "tune": round(tables[best_key][0], 5), "judge": round(tables[best_key][1], 5)},
        "ships": bool(tables[best_key][1] < tables[cur_key][1]),
    }
    # corners: xi only
    if cdf is not None and len(cdf) >= min_train:
        ctables = {}
        for xi in sorted(set(XI_GRID + [current["xi_corners"]])):
            preds = dc_walk(cdf, xi, refit_days, offset, min_train)
            if preds.empty:
                continue
            cut = date_cut(preds)
            ctables[xi] = split_mean(eval_corners(cdf, preds), cut)
            print(f"  [{name}] corners xi={xi} done ({time.time()-t0:.0f}s)", flush=True)
        if ctables:
            cxi = current["xi_corners"]
            bxi = min(ctables, key=lambda k: ctables[k][0])
            res["corners"] = {
                "grid": {f"xi={k}": {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                         for k, v in sorted(ctables.items())},
                "current": {"xi": cxi, "tune": round(ctables[cxi][0], 5),
                            "judge": round(ctables[cxi][1], 5)},
                "chosen": {"xi": bxi, "tune": round(ctables[bxi][0], 5),
                           "judge": round(ctables[bxi][1], 5)},
                "ships": bool(ctables[bxi][1] < ctables[cxi][1]),
            }
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------- NHL -----
def tune_nhl(engine) -> dict:
    t0 = time.time()
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT match_date, home_team_id, away_team_id,
                   reg_home_goals, reg_away_goals, home_goals, away_goals, 1.0 AS w
            FROM nhl.games WHERE status='FINAL' AND reg_home_goals IS NOT NULL
              AND home_goals IS NOT NULL
            ORDER BY match_date
        """), conn)
    # DC frame over REGULATION goals
    reg = df.rename(columns={"reg_home_goals": "home_goals", "reg_away_goals": "away_goals"})[
        ["match_date", "home_team_id", "away_team_id", "w"]].copy()
    reg["home_goals"] = df["reg_home_goals"]
    reg["away_goals"] = df["reg_away_goals"]
    # form + b2b from FINAL goals (production team_form uses final goals)
    fin = df[["match_date", "home_team_id", "away_team_id", "home_goals", "away_goals"]].copy()
    form = build_form(fin)
    # rest days per (team, date) for b2b
    last_date: dict = {}
    b2b_rows = []
    for r in df.itertuples():
        hb = (r.match_date - last_date[r.home_team_id]).days <= 1 if r.home_team_id in last_date else False
        ab = (r.match_date - last_date[r.away_team_id]).days <= 1 if r.away_team_id in last_date else False
        b2b_rows.append((hb, ab))
        last_date[r.home_team_id] = r.match_date
        last_date[r.away_team_id] = r.match_date
    b2b = dict(zip(df.index, b2b_rows))

    idx = np.arange(13)
    hi_g, aj_g = np.meshgrid(idx, idx, indexing="ij")
    final_total = hi_g + aj_g + (hi_g == aj_g).astype(int)

    def eval_nhl(preds, wt, fatigue=1.0):
        rows = []
        for p in preds.itertuples():
            r = df.loc[p.i]
            h_gf, h_ga, a_gf, a_ga = form[p.i]
            lam = blend(p.lam, h_gf, a_ga, wt)
            mu = blend(p.mu, a_gf, h_ga, wt)
            hb, ab = b2b[p.i]
            if fatigue != 1.0:
                if hb:
                    lam *= fatigue
                if ab:
                    mu *= fatigue
            m = score_matrix(lam, mu, p.rho, 12)
            p_ht = float(np.tril(m, -1).sum() + np.trace(m))
            tot = r["home_goals"] + r["away_goals"]
            reg_home_ge = r["reg_home_goals"] >= r["reg_away_goals"]
            ps = [p_ht] + [float(m[final_total > ln].sum()) for ln in NHL_LINES]
            ys = [1.0 if reg_home_ge else 0.0] + [1.0 if tot > ln else 0.0 for ln in NHL_LINES]
            ll = _ll(np.array(ps), np.array(ys))
            rows.append((p.match_date, float(ll.sum()), len(ll)))
        return pd.DataFrame(rows, columns=["d", "ll", "n"])

    tables, preds_by_xi = {}, {}
    for xi in sorted(set(XI_GRID + [0.0038])):
        preds = dc_walk(reg, xi, 14, 240, 600)
        preds_by_xi[xi] = preds
        cut = date_cut(preds)
        for wt in BLEND_GRID:
            tables[(xi, wt)] = split_mean(eval_nhl(preds, wt), cut)
        print(f"  [nhl] xi={xi} done ({time.time()-t0:.0f}s)", flush=True)
    cur_key = (0.0038, 0.2)
    best_key = min(tables, key=lambda k: tables[k][0])
    # fatigue factor on TUNE with the chosen xi/blend
    preds = preds_by_xi[best_key[0]]
    cut = date_cut(preds)
    ftables = {f: split_mean(eval_nhl(preds, best_key[1], f), cut) for f in B2B_GRID}
    best_f = min(ftables, key=lambda k: ftables[k][0])
    out = {
        "league": "nhl",
        "grid": {f"xi={k[0]},blend={k[1]}": {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                 for k, v in sorted(tables.items())},
        "current": {"xi": cur_key[0], "blend": cur_key[1], "b2b": 1.0,
                    "tune": round(tables[cur_key][0], 5), "judge": round(tables[cur_key][1], 5)},
        "chosen": {"xi": best_key[0], "blend": best_key[1], "b2b": best_f,
                   "tune": round(ftables[best_f][0], 5), "judge": round(ftables[best_f][1], 5)},
        "b2b_grid": {str(f): {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                     for f, v in ftables.items()},
        "ships": bool(ftables[best_f][1] < tables[cur_key][1]),
        "seconds": round(time.time() - t0, 1),
    }
    return out


# --------------------------------------------------------------- NBA / NFL ---
def fit_wls(df, as_of, half_life, ridge):
    ages = (pd.Timestamp(as_of) - pd.to_datetime(df["match_date"])).dt.days.to_numpy(dtype=float)
    w = np.exp(-np.log(2) * ages / half_life)
    teams = sorted(set(df["home_team_id"]) | set(df["away_team_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    hidx = df["home_team_id"].map(idx).to_numpy()
    aidx = df["away_team_id"].map(idx).to_numpy()
    hp = df["home_points"].to_numpy(dtype=float)
    ap = df["away_points"].to_numpy(dtype=float)
    n_g = len(df)
    X = np.zeros((2 * n_g, 2 * n_t + 1))
    r1 = np.arange(0, 2 * n_g, 2)
    r2 = np.arange(1, 2 * n_g, 2)
    X[r1, hidx] = 1.0
    X[r1, n_t + aidx] = -1.0
    X[r1, -1] = 1.0
    X[r2, aidx] = 1.0
    X[r2, n_t + hidx] = -1.0
    y = np.empty(2 * n_g)
    y[r1] = hp
    y[r2] = ap
    ws = np.repeat(w, 2)
    sw = np.sqrt(ws)
    mu = float(np.average(y, weights=ws))
    Xs, ys = X * sw[:, None], (y - mu) * sw
    if ridge > 0:
        R = np.zeros((2 * n_t, 2 * n_t + 1))
        R[np.arange(2 * n_t), np.arange(2 * n_t)] = np.sqrt(ridge)
        Xs = np.vstack([Xs, R])
        ys = np.concatenate([ys, np.zeros(2 * n_t)])
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    off = beta[:n_t]
    dfn = beta[n_t:2 * n_t]
    hfa = float(beta[-1])
    eh = mu + off[hidx] - dfn[aidx] + hfa
    ea = mu + off[aidx] - dfn[hidx]
    tot_res = (hp + ap) - (eh + ea)
    mar_res = (hp - ap) - (eh - ea)
    sigma_total = float(np.sqrt(np.average(tot_res ** 2, weights=w)))
    sigma_margin = float(np.sqrt(np.average(mar_res ** 2, weights=w)))
    return {"idx": idx, "mu": mu, "off": off, "dfn": dfn, "hfa": hfa,
            "sigma_total": sigma_total, "sigma_margin": sigma_margin}


def wls_walk(df, half_life, ridge, refit_days=14, offset=240, min_train=400):
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))  # noqa: E731
    lo, hi = df["match_date"].min(), df["match_date"].max()
    out = []
    block_start = lo + timedelta(days=offset)
    model = None
    while block_start <= hi:
        block_end = block_start + timedelta(days=refit_days - 1)
        train = df[df["match_date"] < block_start]
        if len(train) >= min_train:
            model = fit_wls(train, block_start, half_life, ridge)
        if model is not None:
            blk = df[(df["match_date"] >= block_start) & (df["match_date"] <= block_end)]
            for i, r in blk.iterrows():
                ih = model["idx"].get(r["home_team_id"])
                ia = model["idx"].get(r["away_team_id"])
                offh = model["off"][ih] if ih is not None else 0.0
                dfnh = model["dfn"][ih] if ih is not None else 0.0
                offa = model["off"][ia] if ia is not None else 0.0
                dfna = model["dfn"][ia] if ia is not None else 0.0
                eh = model["mu"] + offh - dfna + model["hfa"]
                ea = model["mu"] + offa - dfnh
                out.append({"i": i, "match_date": r["match_date"], "eh": eh, "ea": ea,
                            "sigma_total": model["sigma_total"],
                            "sigma_margin": model["sigma_margin"],
                            "p_home": phi((eh - ea) / model["sigma_margin"])})
        block_start = block_end + timedelta(days=1)
    return pd.DataFrame(out)


def tune_points(name, engine, table, lines, hl_grid, hl_current, min_train) -> dict:
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))  # noqa: E731
    t0 = time.time()
    with engine.begin() as conn:
        df = pd.read_sql(text(f"""
            SELECT match_date, home_team_id, away_team_id, home_points, away_points
            FROM {table} WHERE status='FT' AND home_points IS NOT NULL ORDER BY match_date
        """), conn)
    tot = (df["home_points"] + df["away_points"]).to_numpy(dtype=float)
    winner_y = (df["home_points"] > df["away_points"]).to_numpy()
    ties = (df["home_points"] == df["away_points"]).to_numpy()

    def eval_combo(preds, sigma_override=None):
        rows = []
        for p in preds.itertuples():
            j = p.i
            st = sigma_override[j] if sigma_override is not None else p.sigma_total
            exp_t = p.eh + p.ea
            ps = [1.0 - phi((ln - exp_t) / st) for ln in lines]
            ys = [1.0 if tot[j] > ln else 0.0 for ln in lines]
            if not ties[j]:
                ps.append(p.p_home)
                ys.append(1.0 if winner_y[j] else 0.0)
            ll = _ll(np.array(ps), np.array(ys))
            rows.append((p.match_date, float(ll.sum()), len(ll)))
        return pd.DataFrame(rows, columns=["d", "ll", "n"])

    tables, preds_cache = {}, {}
    for hl in sorted(set(hl_grid + [hl_current])):
        for rg in RIDGE_GRID:
            preds = wls_walk(df, hl, rg, min_train=min_train)
            if preds.empty:
                continue
            preds_cache[(hl, rg)] = preds
            cut = date_cut(preds)
            tables[(hl, rg)] = split_mean(eval_combo(preds), cut)
        print(f"  [{name}] hl={hl} done ({time.time()-t0:.0f}s)", flush=True)
    cur_key = (hl_current, 0.0)
    best_key = min(tables, key=lambda k: tables[k][0])

    # ---- per-game sigma for P(over): squared-residual model on TUNE ----------
    preds = preds_cache[best_key]
    cut = date_cut(preds)
    # rolling variance of each team's game totals (last 10, strictly prior)
    hist: dict = {}
    var_h, var_a = {}, {}
    for r in df.itertuples():
        for team, store in ((r.home_team_id, var_h), (r.away_team_id, var_a)):
            dv = hist.get(team, [])
            store[r.Index] = float(np.var(dv[-10:])) if len(dv) >= 4 else float("nan")
        t = float(r.home_points + r.away_points)
        hist.setdefault(r.home_team_id, []).append(t)
        hist.setdefault(r.away_team_id, []).append(t)
    pi = preds.set_index("i")
    exp_t = (pi["eh"] + pi["ea"])
    resid2 = (pd.Series(tot, index=df.index).loc[pi.index] - exp_t) ** 2
    vsum = pd.Series({i: var_h.get(i, np.nan) + var_a.get(i, np.nan) for i in pi.index})
    is_tune = pd.to_datetime(pi["match_date"]) <= cut
    mask = is_tune & vsum.notna()
    sig_g = float(pi["sigma_total"].median())
    coefs = None
    sigma_over = None
    if mask.sum() >= 200:
        A = np.column_stack([np.ones(mask.sum()), exp_t[mask], vsum[mask]])
        b, *_ = np.linalg.lstsq(A, resid2[mask], rcond=None)
        coefs = [float(x) for x in b]
        s2 = pd.Series(b[0] + b[1] * exp_t + b[2] * vsum, index=pi.index)
        s2 = s2.fillna(pi["sigma_total"] ** 2)
        sig = np.sqrt(np.clip(s2, (0.5 * sig_g) ** 2, (2.0 * sig_g) ** 2))
        sigma_over = sig.to_dict()
    base_tab = split_mean(eval_combo(preds), cut)
    sig_tab = split_mean(eval_combo(preds, sigma_override=sigma_over), cut) if sigma_over else None
    sigma_ships = bool(sig_tab and sig_tab[1] < base_tab[1])

    return {
        "league": name,
        "grid": {f"hl={k[0]},ridge={k[1]}": {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                 for k, v in sorted(tables.items())},
        "current": {"half_life": cur_key[0], "ridge": cur_key[1],
                    "tune": round(tables[cur_key][0], 5), "judge": round(tables[cur_key][1], 5)},
        "chosen": {"half_life": best_key[0], "ridge": best_key[1],
                   "tune": round(tables[best_key][0], 5), "judge": round(tables[best_key][1], 5)},
        "ships": bool(tables[best_key][1] < tables[cur_key][1]),
        "sigma_model": {"coefs": coefs, "global_sigma": round(sig_g, 2),
                        "base_judge": round(base_tab[1], 5),
                        "sigma_judge": round(sig_tab[1], 5) if sig_tab else None,
                        "ships": sigma_ships},
        "seconds": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ loaders --
def load_league(engine, vertical: str):
    from sandy.mls.ratings import load_corner_matches, load_goal_matches
    from sandy.soccer.loop import load_corners, load_goals
    if vertical == "mls":
        return load_goal_matches(engine), load_corner_matches(engine)
    lg = vertical.split("_", 1)[1]
    return load_goals(engine, lg), load_corners(engine, lg)


def tune_worldcup(engine) -> dict:
    from sandy.football.backtest import NEUTRAL_COMPETITIONS
    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT match_date, home_team_id, away_team_id, home_goals, away_goals,
                   competition, COALESCE(competition_weight, 1.0) AS w
            FROM football.matches
            WHERE status IN ('FT','AET','PEN') AND home_goals IS NOT NULL
            ORDER BY match_date
        """), conn)
    df["neutral"] = df["competition"].isin(NEUTRAL_COMPETITIONS)
    t0 = time.time()
    tables = {}

    def eval_wc(preds):
        rows = []
        idx = np.arange(11)
        hi_g, aj_g = np.meshgrid(idx, idx, indexing="ij")
        totals = hi_g + aj_g
        btts_m = (hi_g >= 1) & (aj_g >= 1)
        for p in preds.itertuples():
            r = df.loc[p.i]
            m = score_matrix(p.lam, p.mu, p.rho, 10)
            p_hd = float(np.tril(m, -1).sum() + np.trace(m))
            tot = r["home_goals"] + r["away_goals"]
            ps = ([p_hd] + [float(m[totals > ln].sum()) for ln in GOAL_LINES]
                  + [float(m[btts_m].sum())])
            ys = ([1.0 if r["home_goals"] >= r["away_goals"] else 0.0]
                  + [1.0 if tot > ln else 0.0 for ln in GOAL_LINES]
                  + [1.0 if (r["home_goals"] > 0 and r["away_goals"] > 0) else 0.0])
            ll = _ll(np.array(ps), np.array(ys))
            rows.append((p.match_date, float(ll.sum()), len(ll)))
        return pd.DataFrame(rows, columns=["d", "ll", "n"])

    for xi in sorted(set(XI_GRID + [0.0019])):
        preds = dc_walk(df, xi, 30, 400, 300)
        if preds.empty:
            continue
        cut = date_cut(preds)
        tables[xi] = split_mean(eval_wc(preds), cut)
        print(f"  [worldcup] xi={xi} done ({time.time()-t0:.0f}s)", flush=True)
    cur, best = 0.0019, min(tables, key=lambda k: tables[k][0])
    return {"league": "worldcup",
            "grid": {f"xi={k}": {"tune": round(v[0], 5), "judge": round(v[1], 5)}
                     for k, v in sorted(tables.items())},
            "current": {"xi": cur, "tune": round(tables[cur][0], 5),
                        "judge": round(tables[cur][1], 5)},
            "chosen": {"xi": best, "tune": round(tables[best][0], 5),
                       "judge": round(tables[best][1], 5)},
            "ships": bool(tables[best][1] < tables[cur][1]),
            "seconds": round(time.time() - t0, 1)}


def main():
    cfg = load_config()
    engine = create_engine(cfg)
    out_dir = Path(cfg.model.model_dir) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or ["soccer_col", "soccer_mex", "soccer_esp", "soccer_eng",
                               "mls", "worldcup", "nhl", "nba", "nfl"]
    for tgt in targets:
        print(f"=== tuning {tgt} ===", flush=True)
        if tgt in ("mls",) or tgt.startswith("soccer_"):
            gdf, cdf = load_league(engine, tgt)
            cur = {"xi_goals": 0.0038, "blend_goals": 0.2, "xi_corners": 0.0038}
            res = tune_dc_family(tgt, gdf, cdf, cur,
                                 min_train=300 if tgt == "mls" else 250)
        elif tgt == "nhl":
            res = tune_nhl(engine)
        elif tgt == "nba":
            res = tune_points("nba", engine, "nba.games", NBA_LINES, NBA_HL_GRID, 120, 400)
        elif tgt == "nfl":
            res = tune_points("nfl", engine, "nfl.games", NFL_LINES, NFL_HL_GRID, 200, 280)
        elif tgt == "worldcup":
            res = tune_worldcup(engine)
        else:
            print(f"unknown target {tgt}")
            continue
        path = out_dir / f"tune_{tgt}.json"
        path.write_text(json.dumps(res, indent=2))
        summary = {k: res.get(k) for k in ("current", "chosen", "ships", "goals",
                                           "corners", "sigma_model", "seconds") if k in res}
        print(json.dumps(summary, indent=2, default=str), flush=True)
        print(f"=== {tgt} done -> {path} ===", flush=True)


if __name__ == "__main__":
    main()
