"""Meta² — a GLOBAL second-stage "refiner" over the per-league meta-models.

The per-league meta (betmeta) already gates picks: a pick is ✅ when its 🤖
score clears its market's recommended threshold. This module trains ONE model
across all 9 leagues that re-scores those already-approved picks and surfaces
an elite subset (⭐ / 💎) with honestly-higher accuracy than a simple 🤖 floor.

PIPELINE
  Stage 1  build_oof(league)            Honest OUT-OF-FOLD meta scores. The
           production artifacts' scores on their own training rows are
           inflated, so per league the meta frame is sorted chronologically,
           cut into 5 equal time blocks, and for i in 3..5 a meta is trained
           on blocks 1..i-1 (league's stored params, monotone conf, early
           stopping + isotonic on the training span's last 20%) and scores
           block i. Blocks 1-2 are seed-only. Cached under models/oof/.
  Stage 2  build_refiner_dataset()      One row per (game, market) pick that
           passes its league's CURRENT per-line threshold using the OOF iso
           score (mirrors production gating honestly). Label = pick correct.
  Stage 3  train_refiner()              Pooled 60/20/20 chronological split;
           LightGBM binary, monotone +1 on the OOF meta score and conf; small
           grid on calib AUC; isotonic on calib; tier floors on CALIB via
           Wilson LB (⭐ ≥ 0.90, 💎 ≥ 0.94) over best-per-game picks.
  Stage 4  SHIP-GATE                    On the untouched pooled TEST, best
           pick per game (highest 🤖, like the finals list): each tier must
           beat a 🤖-floor baseline AT MATCHED VOLUME (the top-n test picks
           by 🤖, n = the tier's volume). Ships only if BOTH tiers win; the
           artifact records the verdict and score_pick stays inert on a fail.

FEATURES — every one computable at live scoring time:
  meta_iso        the meta 🤖 score. Training: OOF isotonic score. Live: the
                  production artifact's isotonic score (what gates the pick).
  meta_raw        raw booster score behind meta_iso (OOF / live artifact).
  p, conf         base-model probability of the YES side; conf = max(p, 1-p).
  side_yes        1.0 when the pick is the YES side (over / 1X / home / BTTS-sí).
  kind_*          market-kind one-hots (result winner btts goals corners points runs).
  lg_*            league one-hots (9 leagues).
  line_margin     |expected_total − line|; expected_total = Σ spec err_expected
                  (λh+λa, exp_total, expected runs) for goals/points/runs,
                  corner λh+λa for corners; NaN for result/winner/btts.
  sib_n           other markets of the SAME game currently approved (count).
  sib_frac        sib_n / (scored markets in the game − 1).
  sib_adj_same_side  1.0 if an ADJACENT line (the neighbouring rung of the same
                  kind's ladder) is approved with the same side; NaN for
                  line-less kinds. Training approvals use OOF scores; live
                  approvals use the production meta (same thresholds).
  xm_z_goals/xm_z_corners/xm_z_agree  cross-model story (soccer/MLS only):
                  z-scores of the goals-model and corners-model expected
                  totals vs the league's seed-block (blocks 1-2) distribution,
                  and their product (positive = both high / both low). NaN
                  where a league has no corners model.
  form_acc_30/form_n_30  league recent form: accuracy (and count) of the
                  league's OOF-APPROVED picks over the prior 30 days,
                  STRICTLY before the row's date. Live reads the artifact's
                  OOF history store (refreshed by the nightly retrain).
  slate_size      approved-pick count for (league, date). Training/historical:
                  from the OOF store; live dates not in the store: computed
                  from the day's candidates via the production meta (cached).
  h_/a_model_err, h_/a_model_abs_err  the base model's own recent errors
                  (betmeta covariates; bulk = _attach_model_err, live =
                  _team_recent_err — value-identical, audited in betmeta).

STRICTLY PRIOR everywhere: rolling features exclude the row's own date.
Historical rows are scored from the artifact's OOF store (so dashboard levels
on past dates stay honest); genuinely-live rows fall back to the production
meta for the meta/sibling/slate inputs — the same semantics, fresher source.

audit_equality() re-scores 30 sampled TEST rows through the LIVE path
(score_pick on the raw DB row) and asserts |bulk − live| < 1e-6, like
betmeta's bulk/live model-err audit.
"""
from __future__ import annotations

import logging
import pickle
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from sandy.betmeta import (MODEL_ERR_FEATURES, SPECS, _frame, _row_features,
                           _team_recent_err, _wilson_lb)
from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

LEAGUES = tuple(SPECS)  # the 9 leagues
KINDS = ("result", "winner", "btts", "goals", "corners", "points", "runs")
N_BLOCKS = 5
SEED_BLOCKS = 2          # blocks 1-2: seed-only, never in the refiner dataset
FORM_DAYS = 30
STAR_LB, DIAMOND_LB = 0.90, 0.94
FLOOR_GRID = [round(f, 3) for f in np.arange(0.50, 0.9951, 0.005)]
DEFAULT_HP = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=40)
MIN_DATASET_ROWS = 2000

FEATURES = (["meta_iso", "meta_raw", "p", "conf", "side_yes"]
            + [f"kind_{k}" for k in KINDS]
            + [f"lg_{lg}" for lg in LEAGUES]
            + ["line_margin", "sib_n", "sib_frac", "sib_adj_same_side",
               "xm_z_goals", "xm_z_corners", "xm_z_agree",
               "form_acc_30", "form_n_30", "slate_size",
               *MODEL_ERR_FEATURES])


# ---------------------------------------------------------------- helpers ---
def _as_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.to_datetime(v).date()


def _league_artifact(league: str, cfg: Config) -> dict:
    path = cfg.model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _market_info(league: str) -> dict[str, tuple[str, str, float | None]]:
    """market -> (pcol, kind, line)."""
    return SPECS[league]["markets"]


def _line_neighbors(league: str) -> dict[str, list[str]]:
    """market -> the adjacent rungs (line ±1 step) of the same kind's ladder."""
    ladders: dict[str, list[tuple[float, str]]] = {}
    for m, (_p, kind, line) in _market_info(league).items():
        if line is not None:
            ladders.setdefault(kind, []).append((float(line), m))
    out: dict[str, list[str]] = {}
    for pairs in ladders.values():
        pairs.sort()
        for j, (_ln, mk) in enumerate(pairs):
            adj = []
            if j > 0:
                adj.append(pairs[j - 1][1])
            if j + 1 < len(pairs):
                adj.append(pairs[j + 1][1])
            out[mk] = adj
    return out


def _expected_totals(league: str, get) -> tuple[float, float]:
    """(goals/points/runs expected total, corners expected total) via `get(col)`.
    NaN when a component is missing (e.g. no corners model in that league)."""
    spec = SPECS[league]

    def _sum(cols):
        tot = 0.0
        for c in cols:
            v = get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return float("nan")
            tot += float(v)
        return tot

    base = _sum(spec["err_expected"])
    corners = (_sum(["corner_lambda_home", "corner_lambda_away"])
               if "corner_lambda_home" in spec["num_cols"] else float("nan"))
    return base, corners


def _form_window(hist: tuple[np.ndarray, np.ndarray], d: date) -> tuple[float, float]:
    """(accuracy, n) of approved picks in [d-30d, d) — strictly prior."""
    dates, corr = hist
    if len(dates) == 0:
        return float("nan"), 0.0
    d64 = np.datetime64(d)
    lo = int(np.searchsorted(dates, d64 - np.timedelta64(FORM_DAYS, "D"), side="left"))
    hi = int(np.searchsorted(dates, d64, side="left"))
    n = hi - lo
    return (float(corr[lo:hi].mean()) if n else float("nan")), float(n)


def _sibling_feats(rows: list[dict], neighbors: dict[str, list[str]]) -> dict[str, dict]:
    """Sibling features for ONE game. `rows`: dicts with market/kind/line/
    side_yes/approved for every SCORED market of the game. Returns
    market -> {sib_n, sib_frac, sib_adj_same_side}."""
    n_scored = len(rows)
    approved_side = {r["market"]: r["side_yes"] for r in rows if r["approved"]}
    n_appr = len(approved_side)
    out = {}
    for r in rows:
        own = 1 if r["approved"] else 0
        sib_n = float(n_appr - own)
        sib_frac = sib_n / max(n_scored - 1, 1)
        if r["line"] is None or (isinstance(r["line"], float) and np.isnan(r["line"])):
            adj = float("nan")
        else:
            adj = 1.0 if any(mk in approved_side and approved_side[mk] == r["side_yes"]
                             for mk in neighbors.get(r["market"], [])) else 0.0
        out[r["market"]] = {"sib_n": sib_n, "sib_frac": sib_frac, "sib_adj_same_side": adj}
    return out


# --------------------------------------------- Stage 1: out-of-fold scores --
def _data_sig(engine, league: str) -> tuple:
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    with engine.begin() as conn:
        n, mx = conn.execute(text(
            f"SELECT COUNT(*), MAX(match_date) FROM {spec['table']} "
            f"WHERE outcome_filled_at_utc IS NOT NULL{extra}")).fetchone()
    return int(n), str(mx)


def build_oof(league: str, cfg: Config, engine=None, force: bool = False) -> pd.DataFrame:
    """The league's meta frame + `_block` (1..5) + `oof_raw`/`oof_iso` (NaN on
    the seed blocks 1-2). Cached under models/oof/ keyed on the reconciled
    row-count + max date, so unchanged data never retrains."""
    engine = engine or create_engine(cfg)
    sig = _data_sig(engine, league)
    oof_dir = cfg.model.model_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    path = oof_dir / f"{league}_oof.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("sig") == sig:
            return cached["df"]
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    X = _frame(engine, league, with_ids=True)
    X = X.sort_values(["_date", "_game_id"], kind="mergesort").reset_index(drop=True)
    dts = pd.to_datetime(X["_date"])
    cuts = [dts.quantile(q) for q in (0.2, 0.4, 0.6, 0.8)]
    X["_block"] = np.searchsorted(pd.to_datetime(cuts).values, dts.values, side="left") + 1
    feat_cols = [c for c in X.columns if not c.startswith("_")]
    hp = _league_artifact(league, cfg).get("params") or DEFAULT_HP
    mono = [1 if c == "conf" else 0 for c in feat_cols]
    base = dict(objective="binary", feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=1, verbose=-1, seed=42, monotone_constraints=mono)
    X["oof_raw"] = np.nan
    X["oof_iso"] = np.nan
    for blk in range(SEED_BLOCKS + 1, N_BLOCKS + 1):
        tr = X[X["_block"] < blk]
        sc = X[X["_block"] == blk]
        if tr.empty or sc.empty or tr["_y"].nunique() < 2:
            continue
        # early stopping + isotonic against the training span's last 20%
        tail_cut = pd.to_datetime(tr["_date"]).quantile(0.8)
        fit = tr[pd.to_datetime(tr["_date"]) <= tail_cut]
        val = tr[pd.to_datetime(tr["_date"]) > tail_cut]
        if fit.empty or val.empty or fit["_y"].nunique() < 2:
            continue
        b = lgb.train({**base, **hp}, lgb.Dataset(fit[feat_cols], label=fit["_y"]),
                      num_boost_round=500,
                      valid_sets=[lgb.Dataset(val[feat_cols], label=val["_y"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(b.predict(val[feat_cols], num_iteration=b.best_iteration),
                val["_y"].to_numpy().astype(float))
        raw = b.predict(sc[feat_cols], num_iteration=b.best_iteration)
        X.loc[sc.index, "oof_raw"] = raw
        X.loc[sc.index, "oof_iso"] = iso.predict(raw)
        logger.info("%s OOF block %s: train=%s score=%s", league, blk, len(tr), len(sc))
    with open(path, "wb") as f:
        pickle.dump({"sig": sig, "df": X}, f)
    return X


# ------------------------------------------------ Stage 2: refiner dataset --
def build_refiner_dataset(cfg: Config, force_oof: bool = False):
    """Returns (dataset, stores):
      dataset — one row per OOF-APPROVED (game, market) pick with FEATURES,
                _y label and _league/_game_id/_market/_date keys.
      stores  — everything the live path needs: the compact OOF pick index,
                per-league form histories, slate counts, z-stats, thresholds,
                and per-league coverage counts for the report."""
    engine = create_engine(cfg)
    per_league, coverage, z_stats, thresholds = [], {}, {}, {}
    for lg in LEAGUES:
        X = build_oof(lg, cfg, engine=engine, force=force_oof)
        art = _league_artifact(lg, cfg)
        thr_by = dict(art.get("threshold_by_market") or {})
        gthr = art.get("threshold")
        thresholds[lg] = {"by_market": thr_by, "global": gthr}
        info = _market_info(lg)
        neighbors = _line_neighbors(lg)
        spec = SPECS[lg]
        # expected totals (vectorized; 2-term IEEE sums match the live path's
        # sequential adds exactly)
        X["_base_et"] = X[spec["err_expected"]].sum(axis=1, min_count=len(spec["err_expected"]))
        X["_corners_et"] = (X[["corner_lambda_home", "corner_lambda_away"]].sum(axis=1, min_count=2)
                            if "corner_lambda_home" in X.columns else np.nan)
        # cross-model z-stats from the SEED blocks only (strictly prior to all
        # refiner rows), game-level (dedupe the per-market repetition).
        seed_games = X[X["_block"] <= SEED_BLOCKS].drop_duplicates("_game_id")
        zs = None
        if seed_games["_corners_et"].notna().any():
            g, c = seed_games["_base_et"].dropna(), seed_games["_corners_et"].dropna()
            if len(g) >= 30 and len(c) >= 30 and g.std() > 0 and c.std() > 0:
                zs = {"g_mean": float(g.mean()), "g_std": float(g.std()),
                      "c_mean": float(c.mean()), "c_std": float(c.std())}
        z_stats[lg] = zs

        D = X[X["oof_iso"].notna()].copy()
        coverage[lg] = {"frame_rows": len(X), "oof_rows": len(D)}
        if D.empty:
            continue
        D["_league"] = lg
        D["_kind"] = D["_market"].map(lambda m: info[m][1])
        D["_line"] = pd.to_numeric(D["_market"].map(lambda m: info[m][2]))  # None → NaN
        D["_thr"] = pd.to_numeric(D["_market"].map(lambda m: thr_by.get(m, gthr)))
        D["approved"] = D["oof_iso"] >= D["_thr"]  # NaN threshold → not approved
        D["side_yes"] = (D["p"] >= 0.5).astype(float)
        # line margin: |expected_total − line| for the market's own kind
        exp_for_kind = np.where(D["_kind"] == "corners", D["_corners_et"],
                                np.where(D["_kind"].isin(["goals", "points", "runs"]),
                                         D["_base_et"], np.nan))
        D["line_margin"] = np.abs(exp_for_kind - D["_line"].astype(float).to_numpy())
        if zs:
            D["xm_z_goals"] = (D["_base_et"] - zs["g_mean"]) / zs["g_std"]
            D["xm_z_corners"] = (D["_corners_et"] - zs["c_mean"]) / zs["c_std"]
            D["xm_z_agree"] = D["xm_z_goals"] * D["xm_z_corners"]
        else:
            D["xm_z_goals"] = D["xm_z_corners"] = D["xm_z_agree"] = np.nan
        # sibling agreement within each game (from the OOF approvals)
        for k in ("sib_n", "sib_frac", "sib_adj_same_side"):
            D[k] = np.nan
        for _gid, grp in D.groupby("_game_id", sort=False):
            rows = [{"market": m, "line": ln, "side_yes": sy, "approved": ap}
                    for m, ln, sy, ap in zip(grp["_market"], grp["_line"],
                                             grp["side_yes"], grp["approved"])]
            feats = _sibling_feats(rows, neighbors)
            for k in ("sib_n", "sib_frac", "sib_adj_same_side"):
                D.loc[grp.index, k] = [feats[m][k] for m in grp["_market"]]
        # only the pooled/global columns survive (league frames differ otherwise)
        per_league.append(D[["_date", "_game_id", "_market", "_league", "_kind", "_line",
                             "_y", "p", "conf", "oof_raw", "oof_iso", "approved", "side_yes",
                             "line_margin", "xm_z_goals", "xm_z_corners", "xm_z_agree",
                             "sib_n", "sib_frac", "sib_adj_same_side", *MODEL_ERR_FEATURES]])
        coverage[lg]["approved_rows"] = int(D["approved"].sum())
    pooled = pd.concat(per_league, ignore_index=True)

    # league recent form + slate size, from the OOF-APPROVED history only
    form_hist, slate_map = {}, {}
    for lg, grp in pooled.groupby("_league"):
        appr = grp[grp["approved"]].sort_values("_date", kind="mergesort")
        form_hist[lg] = (pd.to_datetime(appr["_date"]).to_numpy(dtype="datetime64[D]"),
                         appr["_y"].to_numpy().astype(float))
        for d, n in appr.groupby("_date").size().items():
            slate_map[(lg, _as_date(d))] = float(n)
    accs, ns, slates = [], [], []
    for lg, d, appr in zip(pooled["_league"], pooled["_date"], pooled["approved"]):
        dd = _as_date(d)
        a, n = _form_window(form_hist[lg], dd)
        accs.append(a)
        ns.append(n)
        slates.append(slate_map.get((lg, dd), 0.0))
    pooled["form_acc_30"], pooled["form_n_30"], pooled["slate_size"] = accs, ns, slates
    pooled = pooled.rename(columns={"oof_iso": "meta_iso", "oof_raw": "meta_raw"})
    for k in KINDS:
        pooled[f"kind_{k}"] = (pooled["_kind"] == k).astype(float)
    for lg in LEAGUES:
        pooled[f"lg_{lg}"] = (pooled["_league"] == lg).astype(float)

    # compact OOF pick index: the live path's store for historical rows
    oof_index = pooled[["_league", "_game_id", "_market", "_date", "_kind", "_line",
                        "side_yes", "meta_raw", "meta_iso", "approved", "_y"]].copy()
    dataset = pooled[pooled["approved"]].copy().sort_values(
        ["_date", "_league", "_game_id"], kind="mergesort").reset_index(drop=True)
    stores = {"oof_index": oof_index, "z_stats": z_stats, "thresholds": thresholds,
              "coverage": coverage}
    return dataset, stores


# ------------------------------------- Stage 3+4: train, floors, ship-gate --
def _best_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Best pick per game = highest 🤖 (meta_iso), like the finals list."""
    idx = df.groupby(["_league", "_game_id"])["meta_iso"].idxmax()
    return df.loc[idx]


def _tier_floor(scores: np.ndarray, ys: np.ndarray, target_lb: float) -> float | None:
    """Smallest refiner-score floor whose Wilson LB(accuracy) >= target."""
    for f in FLOOR_GRID:
        m = scores >= f
        n = int(m.sum())
        if n == 0:
            return None
        if _wilson_lb(int(ys[m].sum()), n) >= target_lb:
            return f
    return None


def _matched_baseline(meta: np.ndarray, ys: np.ndarray, n: int) -> tuple[float, float] | None:
    """Accuracy of the 🤖 floor giving the same n test picks. When that volume
    lands inside a mass of TIED 🤖 scores (isotonic plateaus, e.g. a big group
    at exactly 1.0) no such floor exists — use the EXPECTED accuracy under
    random tie-breaking (strict-above correct + pro-rata share of the tied
    group), which is the fair matched-volume comparator."""
    if n == 0 or n > len(meta):
        return None
    thr = np.sort(meta)[::-1][n - 1]
    strict = meta > thr
    ties = meta == thr
    k = n - int(strict.sum())
    exp_correct = float(ys[strict].sum()) + k * float(ys[ties].mean())
    return exp_correct / n, float(thr)


def _eval_tier(best: pd.DataFrame, ref_scores: np.ndarray, floor: float) -> dict:
    """Tier accuracy/volume on best-per-game rows + the 🤖 baseline at MATCHED
    volume (the 🤖 floor giving the same n; fair tie handling)."""
    ys = best["_y"].to_numpy()
    m = ref_scores >= floor
    n = int(m.sum())
    days = max(best["_date"].nunique(), 1)  # all test slate days, same for baselines
    tier_acc = float(ys[m].mean()) if n else None
    base = _matched_baseline(best["meta_iso"].to_numpy(), ys, n)
    return {"floor": floor, "n": n, "correct": int(ys[m].sum()),
            "acc": round(tier_acc, 4) if tier_acc is not None else None,
            "picks_per_day": round(n / days, 2),
            "baseline_matched_acc": round(base[0], 4) if base else None,
            "baseline_matched_floor": round(base[1], 4) if base else None,
            "beats_baseline": bool(tier_acc is not None and base is not None
                                   and tier_acc > base[0])}


def train_refiner(config: Config | None = None, force_oof: bool = False) -> dict:
    """Stages 1-4: build the dataset, train the global refiner, pick tier
    floors on CALIB, run the ship-gate on TEST, persist models/refiner.pkl
    (always — the artifact records gate_passed and the loaders stay inert on
    a fail, so a failed gate can never leak into dashboards or digests)."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    cfg = config or load_config()
    ds, stores = build_refiner_dataset(cfg, force_oof=force_oof)
    if len(ds) < MIN_DATASET_ROWS:
        raise RuntimeError(f"refiner: only {len(ds)} approved rows (<{MIN_DATASET_ROWS})")
    dts = pd.to_datetime(ds["_date"])
    c1, c2 = dts.quantile(0.6), dts.quantile(0.8)
    tr, ca, te = ds[dts <= c1], ds[(dts > c1) & (dts <= c2)], ds[dts > c2]
    mono = [1 if f in ("meta_iso", "conf") else 0 for f in FEATURES]
    base = dict(objective="binary", feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=1, verbose=-1, seed=42, monotone_constraints=mono)
    cy = ca["_y"].to_numpy().astype(int)
    booster, chosen, calib_auc = None, None, -1.0
    for hp in [dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=mdl)
               for lr in (0.05, 0.03) for nl in (15, 31) for mdl in (100, 300)]:
        b = lgb.train({**base, **hp}, lgb.Dataset(tr[FEATURES], label=tr["_y"]),
                      num_boost_round=500,
                      valid_sets=[lgb.Dataset(ca[FEATURES], label=ca["_y"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        cauc = float(roc_auc_score(cy, b.predict(ca[FEATURES], num_iteration=b.best_iteration)))
        if cauc > calib_auc:
            booster, chosen, calib_auc = b, hp, cauc
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ca_raw = booster.predict(ca[FEATURES], num_iteration=booster.best_iteration)
    iso.fit(ca_raw, cy.astype(float))

    # tier floors on CALIB, best-per-game (the consumption mode), Wilson LB
    ca_best = _best_per_game(ca)
    ca_best_ref = iso.predict(booster.predict(ca_best[FEATURES],
                                              num_iteration=booster.best_iteration))
    ca_best_y = ca_best["_y"].to_numpy()
    star = _tier_floor(ca_best_ref, ca_best_y, STAR_LB)
    diamond = _tier_floor(ca_best_ref, ca_best_y, DIAMOND_LB)

    # ------- TEST: evaluated exactly once, nothing above was tuned on it -----
    te_best = _best_per_game(te)
    te_best_ref = iso.predict(booster.predict(te_best[FEATURES],
                                              num_iteration=booster.best_iteration))
    te_raw = booster.predict(te[FEATURES], num_iteration=booster.best_iteration)
    test_auc = float(roc_auc_score(te["_y"].to_numpy().astype(int), te_raw))
    tiers = {}
    if star is not None:
        tiers["star"] = _eval_tier(te_best, te_best_ref, star)
    if diamond is not None:
        tiers["diamond"] = _eval_tier(te_best, te_best_ref, diamond)
    gate_passed = (star is not None and diamond is not None
                   and tiers["star"]["beats_baseline"] and tiers["diamond"]["beats_baseline"])
    # fixed 🤖-floor baselines for context (the numbers the refiner must beat)
    by = te_best["_y"].to_numpy()
    bm = te_best["meta_iso"].to_numpy()
    days = max(te_best["_date"].nunique(), 1)
    fixed_baselines = {f"meta>={f}": {"n": int((bm >= f).sum()),
                                      "acc": round(float(by[bm >= f].mean()), 4) if (bm >= f).any() else None,
                                      "picks_per_day": round(int((bm >= f).sum()) / days, 2)}
                       for f in (0.90, 0.95)}
    ladder = []
    for f in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975):
        m = te_best_ref >= f
        n = int(m.sum())
        ladder.append({"floor": f, "n": n, "correct": int(by[m].sum()),
                       "acc": round(float(by[m].mean()), 4) if n else None})
    audit_rows = te.sample(min(30, len(te)), random_state=7)
    audit_scores = iso.predict(booster.predict(audit_rows[FEATURES],
                                               num_iteration=booster.best_iteration))
    audit_sample = [{"league": r["_league"], "game_id": r["_game_id"], "market": r["_market"],
                     "date": str(r["_date"]), "p": float(r["p"]),
                     "features": {f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES},
                     "score": float(s)}
                    for (_i, r), s in zip(audit_rows.iterrows(), audit_scores)]
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])[:15]
    artifact = {"model_str": booster.model_to_string(num_iteration=booster.best_iteration),
                "iso": iso, "features": FEATURES, "params": chosen,
                "floors": {"star": star, "diamond": diamond},
                "gate_passed": gate_passed, "tiers_test": tiers,
                "calib_auc": round(calib_auc, 4), "test_auc": round(test_auc, 4),
                "eval_ladder_test": ladder, "fixed_baselines_test": fixed_baselines,
                "importances": [(f, round(float(g), 1)) for f, g in imp],
                "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
                "best_iteration": int(booster.best_iteration or 0),
                "oof_index": stores["oof_index"], "z_stats": stores["z_stats"],
                "thresholds": stores["thresholds"], "coverage": stores["coverage"],
                "audit_sample": audit_sample,
                "trained_at": date.today().isoformat()}
    path = cfg.model.model_dir / "refiner.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    global _bundle_cache
    _bundle_cache = {}
    logger.info("refiner trained: rows=%s auc(calib)=%.4f auc(test)=%.4f floors=%s gate=%s",
                len(ds), calib_auc, test_auc, artifact["floors"], gate_passed)
    return {"n_dataset": len(ds), "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
            "params": chosen, "calib_auc": round(calib_auc, 4), "test_auc": round(test_auc, 4),
            "floors": artifact["floors"], "tiers_test": tiers, "gate_passed": gate_passed,
            "fixed_baselines_test": fixed_baselines, "coverage": stores["coverage"],
            "importances": artifact["importances"]}


# --------------------------------------------------------- live scoring -----
_bundle_cache: dict = {}
_live_sib_cache: dict = {}
_live_slate_cache: dict = {}


def _load_bundle(cfg: Config, require_gate: bool = True):
    """Artifact + runtime lookup structures. None when there is no artifact —
    or when the ship-gate FAILED (require_gate), so a failed refiner can never
    decorate dashboards/digests."""
    key = ("gate" if require_gate else "any")
    if key in _bundle_cache:
        return _bundle_cache[key]
    path = cfg.model.model_dir / "refiner.pkl"
    if not path.exists():
        _bundle_cache[key] = None
        return None
    with open(path, "rb") as f:
        art = pickle.load(f)
    if require_gate and not art.get("gate_passed"):
        _bundle_cache[key] = None
        return None
    import lightgbm as lgb
    booster = lgb.Booster(model_str=art["model_str"])
    idx = art["oof_index"]
    store: dict = {}
    for (lg, gid), grp in idx.groupby(["_league", "_game_id"], sort=False):
        store[(lg, gid)] = [
            {"market": m, "kind": k, "line": ln, "side_yes": sy, "approved": bool(ap),
             "meta_raw": mr, "meta_iso": mi}
            for m, k, ln, sy, ap, mr, mi in zip(grp["_market"], grp["_kind"], grp["_line"],
                                                grp["side_yes"], grp["approved"],
                                                grp["meta_raw"], grp["meta_iso"])]
    form_hist, slate_map = {}, {}
    appr = idx[idx["approved"]]
    for lg, grp in appr.groupby("_league"):
        g = grp.sort_values("_date", kind="mergesort")
        form_hist[lg] = (pd.to_datetime(g["_date"]).to_numpy(dtype="datetime64[D]"),
                         g["_y"].to_numpy().astype(float))
        for d, n in g.groupby("_date").size().items():
            slate_map[(lg, _as_date(d))] = float(n)
    bundle = {"art": art, "booster": booster, "iso": art["iso"], "features": art["features"],
              "floors": art["floors"], "z_stats": art["z_stats"],
              "thresholds": art["thresholds"], "store": store,
              "form_hist": form_hist, "slate_map": slate_map}
    _bundle_cache[key] = bundle
    return bundle


def _meta_scores_live(league: str, cfg: Config, rd: dict, market: str, p: float):
    """(raw, iso) from the CURRENT production meta artifact for a live row."""
    from sandy.betmeta import load_meta
    loaded = load_meta(league, cfg)
    if not loaded:
        return float("nan"), float("nan")
    booster, feat_cols, _thr, iso, _by = loaded
    feats = _row_features(rd, SPECS[league], market, p, league=league, cfg=cfg)
    X = pd.DataFrame([feats]).reindex(columns=feat_cols)
    raw = float(booster.predict(X)[0])
    return raw, (float(iso.predict([raw])[0]) if iso is not None else raw)


def _live_game_rows(league: str, cfg: Config, rd: dict, thresholds: dict) -> list[dict]:
    """Sibling rows for a LIVE (not-in-store) game, from the production meta.
    Cached per (league, row id, date)."""
    key = (league, rd.get("id"), str(rd.get("match_date")))
    if key in _live_sib_cache:
        return _live_sib_cache[key]
    if len(_live_sib_cache) > 4000:
        _live_sib_cache.clear()
    thr_by, gthr = thresholds["by_market"], thresholds["global"]
    rows = []
    for m, (pcol, kind, line) in _market_info(league).items():
        pv = rd.get(pcol)
        if pv is None or (isinstance(pv, float) and np.isnan(pv)):
            continue
        pv = float(pv)
        _raw, iso_s = _meta_scores_live(league, cfg, rd, m, pv)
        thr = thr_by.get(m, gthr)
        rows.append({"market": m, "kind": kind, "line": line,
                     "side_yes": 1.0 if pv >= 0.5 else 0.0,
                     "approved": bool(thr is not None and not np.isnan(iso_s) and iso_s >= thr),
                     "meta_raw": _raw, "meta_iso": iso_s})
    _live_sib_cache[key] = rows
    return rows


def _live_slate(league: str, cfg: Config, d: date, thresholds: dict) -> float:
    """Approved-pick count for a LIVE date: score the day's candidates (pending
    rows preferred, else backtest replay) with the production meta. Cached."""
    key = (league, d)
    if key in _live_slate_cache:
        return _live_slate_cache[key]
    if len(_live_slate_cache) > 2000:
        _live_slate_cache.clear()
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    live_cond = ("" if spec.get("no_backtest_col")
                 else " AND outcome_filled_at_utc IS NULL AND NOT is_backtest")
    engine = create_engine(cfg)
    with engine.begin() as conn:
        rows = conn.execute(text(f"SELECT * FROM {spec['table']} "
                                 f"WHERE match_date = :d{extra}{live_cond}"), {"d": d}).fetchall()
        if not rows and not spec.get("no_backtest_col"):
            rows = conn.execute(text(f"SELECT * FROM {spec['table']} "
                                     f"WHERE match_date = :d{extra} AND is_backtest"),
                                {"d": d}).fetchall()
    n = 0.0
    for r in rows:
        rd = dict(r._mapping)
        n += sum(1.0 for row in _live_game_rows(league, cfg, rd, thresholds) if row["approved"])
    _live_slate_cache[key] = n
    return n


def _pick_features(league: str, rd: dict, market: str, p: float,
                   meta_p: float | None, cfg: Config, bundle: dict) -> dict:
    """The refiner feature vector for one pick — historical rows come straight
    from the OOF store (exactly the training values, audited); live rows use
    the production meta for the meta/sibling/slate inputs."""
    info = _market_info(league)
    kind, line = info[market][1], info[market][2]
    d = _as_date(rd["match_date"])
    store_rows = bundle["store"].get((league, rd.get("id")))
    stored = (next((r for r in store_rows if r["market"] == market), None)
              if store_rows else None)
    thresholds = bundle["thresholds"][league]
    if stored is not None:
        meta_raw, meta_iso = stored["meta_raw"], stored["meta_iso"]
        game_rows = store_rows
        slate = bundle["slate_map"].get((league, d), 0.0)
    else:
        if meta_p is None:
            meta_raw, meta_iso = _meta_scores_live(league, cfg, rd, market, p)
        else:
            meta_raw, _ = _meta_scores_live(league, cfg, rd, market, p)
            meta_iso = float(meta_p)
        game_rows = _live_game_rows(league, cfg, rd, thresholds)
        slate = bundle["slate_map"].get((league, d))
        if slate is None:
            slate = _live_slate(league, cfg, d, thresholds)
    sib = _sibling_feats(game_rows, _line_neighbors(league)).get(market) or \
        {"sib_n": float("nan"), "sib_frac": float("nan"), "sib_adj_same_side": float("nan")}
    base_et, corners_et = _expected_totals(league, rd.get)
    exp_for_kind = (corners_et if kind == "corners"
                    else base_et if kind in ("goals", "points", "runs") else float("nan"))
    line_margin = abs(exp_for_kind - line) if line is not None else float("nan")
    zs = bundle["z_stats"].get(league)
    if zs:
        z_g = (base_et - zs["g_mean"]) / zs["g_std"]
        z_c = (corners_et - zs["c_mean"]) / zs["c_std"]
        z_a = z_g * z_c
    else:
        z_g = z_c = z_a = float("nan")
    form_acc, form_n = _form_window(bundle["form_hist"].get(league, (np.array([], dtype="datetime64[D]"), np.array([]))), d)
    he, ha = _team_recent_err(league, cfg, rd.get("home_team"), rd["match_date"])
    ae, aa = _team_recent_err(league, cfg, rd.get("away_team"), rd["match_date"])
    out = {"meta_iso": meta_iso, "meta_raw": meta_raw, "p": p, "conf": max(p, 1 - p),
           "side_yes": 1.0 if p >= 0.5 else 0.0,
           "line_margin": line_margin, **sib,
           "xm_z_goals": z_g, "xm_z_corners": z_c, "xm_z_agree": z_a,
           "form_acc_30": form_acc, "form_n_30": form_n, "slate_size": slate,
           "h_model_err": he, "h_model_abs_err": ha, "a_model_err": ae, "a_model_abs_err": aa}
    for k in KINDS:
        out[f"kind_{k}"] = 1.0 if k == kind else 0.0
    for lg in LEAGUES:
        out[f"lg_{lg}"] = 1.0 if lg == league else 0.0
    return out


def score_pick(league: str, row_dict: dict, market: str, p: float,
               meta_p: float | None, cfg: Config, *, require_gate: bool = True) -> float | None:
    """Refiner P(pick correct) for one already-approved pick, or None when no
    shipped artifact exists (missing, or the ship-gate failed)."""
    bundle = _load_bundle(cfg, require_gate=require_gate)
    if not bundle:
        return None
    feats = _pick_features(league, dict(row_dict), market, float(p), meta_p, cfg, bundle)
    X = pd.DataFrame([feats]).reindex(columns=bundle["features"])
    raw = float(bundle["booster"].predict(X)[0])
    return float(bundle["iso"].predict([raw])[0])


def refiner_floors(cfg: Config) -> dict | None:
    bundle = _load_bundle(cfg)
    return bundle["floors"] if bundle else None


def nivel_for_pick(league: str, cfg: Config, row_dict: dict, market: str,
                   p: float, meta_p: float | None) -> str | None:
    """'💎' / '⭐' / '✅' for an approved pick — or None when the refiner is not
    shipped (callers fall back to plain ✅)."""
    try:
        s = score_pick(league, row_dict, market, p, meta_p, cfg)
    except Exception:  # never let the refiner break a digest/dashboard
        logger.exception("refiner score_pick failed for %s/%s", league, market)
        return None
    if s is None:
        return None
    floors = refiner_floors(cfg) or {}
    if floors.get("diamond") is not None and s >= floors["diamond"]:
        return "💎"
    if floors.get("star") is not None and s >= floors["star"]:
        return "⭐"
    return "✅"


# -------------------------------------------------------------- audit -------
def audit_equality(cfg: Config | None = None, tol: float = 1e-6) -> dict:
    """Bulk-vs-live equality: re-score the artifact's 30 sampled TEST rows
    through the LIVE path (raw DB row → score_pick) and compare to the bulk
    training-time scores, feature by feature. Like betmeta's model-err audit."""
    cfg = cfg or load_config()
    bundle = _load_bundle(cfg, require_gate=False)
    if not bundle:
        raise RuntimeError("no refiner artifact")
    engine = create_engine(cfg)
    worst, n_ok, mism = 0.0, 0, []
    for s in bundle["art"]["audit_sample"]:
        lg, gid, market = s["league"], s["game_id"], s["market"]
        spec = SPECS[lg]
        with engine.begin() as conn:
            row = conn.execute(text(f"SELECT * FROM {spec['table']} WHERE id = :i"),
                               {"i": gid}).fetchone()
        rd = dict(row._mapping)
        feats = _pick_features(lg, rd, market, s["p"], None, cfg, bundle)
        for f, want in s["features"].items():
            got = feats.get(f)
            if want is None:
                ok = got is None or (isinstance(got, float) and np.isnan(got))
            else:
                ok = got is not None and not (isinstance(got, float) and np.isnan(got)) \
                    and abs(float(got) - want) <= tol
            if not ok:
                mism.append((lg, gid, market, f, want, got))
        live = score_pick(lg, rd, market, s["p"], None, cfg, require_gate=False)
        d = abs(live - s["score"])
        worst = max(worst, d)
        n_ok += d <= tol
    return {"rows": len(bundle["art"]["audit_sample"]), "within_tol": n_ok,
            "worst_abs_diff": worst, "feature_mismatches": mism[:20]}


if __name__ == "__main__":
    import json as _json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rep = train_refiner()
    print(_json.dumps(rep, indent=2, default=str))
