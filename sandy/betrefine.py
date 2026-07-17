"""Meta² — a GLOBAL second-stage "refiner" over the per-league meta-models,
shipped as an ADAPTIVE HYBRID: each tier (⭐ / 💎) is served by whichever
engine — the refiner's floors or the plain meta floors (0.90 / 0.95) — wins a
nightly matched-volume comparison on the CALIB slice. Having the refiner can
therefore never cost accuracy: a tier where the meta is already at its ceiling
simply keeps the meta floor.

PIPELINE
  Stage 1  build_oof(league)            Honest OUT-OF-FOLD meta scores. The
           production artifacts' scores on their own training rows are
           inflated, so per league the meta frame is sorted chronologically,
           cut into 10 equal time blocks, and for i in 2..10 a meta is trained
           on blocks 1..i-1 (league's stored params, monotone conf, early
           stopping + isotonic on the training span's last 20%) and scores
           block i. Block 1 is seed-only → ~90% OOF coverage. Cached under
           models/oof/, keyed on (row-count, max-date, block config).
  Stage 2  build_refiner_dataset()      One row per (game, market) pick that
           passes its league's CURRENT per-line threshold using the OOF iso
           score (mirrors production gating honestly). Label = pick correct.
  Stage 3  train_refiner()              Pooled 60/20/20 chronological split;
           LightGBM binary, monotone +1 on the OOF meta score and conf; small
           grid on calib AUC; isotonic on calib; tier floors on CALIB via
           Wilson LB (⭐ ≥ 0.90, 💎 ≥ 0.94) over best-per-game picks.
  Stage 4  choose_engines()             THE HYBRID GATE, re-run nightly. Per
           tier, on CALIB best-per-game rows: refiner-floor accuracy vs the
           meta's expected accuracy AT MATCHED VOLUME (fair tie handling on
           isotonic plateaus). The winner is persisted in the artifact as
           {"star_engine": "refiner"|"meta", "diamond_engine": ...}. One
           conservative one-way clamp uses TEST: if the shipped hybrid tier is
           materially worse (>0.5pp) than the pure-meta floor there, that tier
           is forced back to "meta" (test can only ever push TOWARD the
           status-quo meta, never promote the refiner — so it stays an honest
           holdout for the reported numbers).

CONSUMPTION — nivel_for_pick(league, row_dict, market, p, meta_p, cfg) returns
'💎'/'⭐'/'✅' for an already-approved pick using the CHOSEN engine per tier.
Missing/stale (>10 days) artifact, legacy artifact, or any scoring error →
plain meta floors (0.90/0.95). It never raises and never returns None.

FEATURES — every one computable at live scoring time:
  meta_iso        the meta 🤖 score. Training: OOF isotonic score. Live: the
                  production artifact's isotonic score (what gates the pick).
  meta_raw        raw booster score behind meta_iso (OOF / live artifact).
  iso_raw_gap     |meta_iso − meta_raw|: how hard the calibration bent the raw
                  score — plateau/extrapolation regions are less trustworthy.
  p, conf         base-model probability of the YES side; conf = max(p, 1-p).
  side_yes        1.0 when the pick is the YES side (over / 1X / home / BTTS-sí).
  kind_*          market-kind one-hots (result winner btts goals corners points runs).
  sk_yes_*        side×kind interactions: side_yes ⋅ kind_k (an over-goals pick
                  and an under-goals pick can have very different reliability).
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
  sib_same_n/sib_same_mean, sib_opp_n/sib_opp_mean  the same game's OTHER
                  approved picks split by side agreement: count and mean 🤖 of
                  those on the SAME side as this pick vs the OPPOSITE side
                  (mean NaN when the count is 0). A game whose whole board
                  leans one way is a different story from a split board.
  xm_z_goals/xm_z_corners/xm_z_agree  cross-model story (soccer/MLS only):
                  z-scores of the goals-model and corners-model expected
                  totals vs the league's seed-block distribution, and their
                  product (positive = both high / both low). NaN where a
                  league has no corners model.
  form_acc_30/form_n_30  league recent form: accuracy (and count) of the
                  league's OOF-APPROVED picks over the prior 30 days,
                  STRICTLY before the row's date.
  slate_size      approved-pick count for (league, date) — the day's slate.
  day_rank/day_rank_frac  rank of this pick's 🤖 within that day's approved
                  league slate (1 = best; ties share the better rank via a
                  strictly-greater count) and rank/slate_size. Day-level info
                  only — the slate's scores are known before any game starts.
  season_days/season_frac  season phase. Seasons are runs of league game dates
                  split on gaps ≥ 45 days (offseason); season_days = days from
                  the season's first game date (0 for an opener, including a
                  live date that starts a new run), season_frac = season_days /
                  the league's typical season length (median completed-run
                  length; schedule-derived only, involves NO outcomes, so the
                  normalizing constant is leakage-free). season_frac > 1 is the
                  late-season/playoffs proxy — no league table carries an
                  explicit playoff flag (checked 2026-07-05), so a true
                  playoff-vs-regular flag is not derivable.
  team_rel_h/team_rel_h_n, team_rel_a/team_rel_a_n  team-level meta
                  reliability: how often meta-approved picks INVOLVING this
                  team hit, over the team's last 20 approved picks STRICTLY
                  before the row's date (mean NaN + n=0 with no history).
  h_/a_model_err, h_/a_model_abs_err  the base model's own recent errors
                  (betmeta covariates; bulk = _attach_model_err, live =
                  _team_recent_err — value-identical, audited in betmeta).
  (Weather covariates deliberately deferred by the user.)

STRICTLY PRIOR everywhere: rolling features exclude the row's own date.
Historical rows are scored from the artifact's OOF store (so dashboard levels
on past dates stay honest); genuinely-live rows fall back to the production
meta for the meta/sibling/slate/day-rank inputs — the same semantics, fresher
source.

audit_equality() re-scores 30 sampled TEST rows through the LIVE path
(score_pick on the raw DB row) and asserts |bulk − live| < 1e-6 feature by
feature, like betmeta's bulk/live model-err audit.
"""
from __future__ import annotations

import logging
import pickle
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from sandy.betmeta import (FRAME_VERSION, MODEL_ERR_FEATURES, SPECS, _frame,
                           _row_features, _team_recent_err, _wilson_lb)
from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

LEAGUES = tuple(SPECS)  # the 9 leagues
KINDS = ("result", "winner", "btts", "goals", "corners", "points", "runs")
N_BLOCKS = 10
SEED_BLOCKS = 1          # block 1: seed-only, never in the refiner dataset
FORM_DAYS = 30
TEAM_REL_WINDOW = 20
SEASON_GAP_DAYS = 45
TYPICAL_SEASON_FALLBACK = 180.0
STAR_LB, DIAMOND_LB = 0.90, 0.94
META_STAR_FLOOR, META_DIAMOND_FLOOR = 0.90, 0.95   # the meta engine's tiers
STALE_DAYS = 10          # artifact older than this → meta floors (never crash)
MIN_TIER_N = 25          # min calib tier volume for the refiner to win a tier
TEST_CLAMP_PP = 0.005    # hybrid >0.5pp worse than meta on test → force meta
FLOOR_GRID = [round(f, 3) for f in np.arange(0.50, 0.9951, 0.005)]
DEFAULT_HP = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=40)
MIN_DATASET_ROWS = 2000

FEATURES = (["meta_iso", "meta_raw", "iso_raw_gap", "p", "conf", "side_yes"]
            + [f"kind_{k}" for k in KINDS]
            + [f"sk_yes_{k}" for k in KINDS]
            + [f"lg_{lg}" for lg in LEAGUES]
            + ["line_margin", "sib_n", "sib_frac", "sib_adj_same_side",
               "sib_same_n", "sib_same_mean", "sib_opp_n", "sib_opp_mean",
               "xm_z_goals", "xm_z_corners", "xm_z_agree",
               "form_acc_30", "form_n_30", "slate_size",
               "day_rank", "day_rank_frac", "season_days", "season_frac",
               "team_rel_h", "team_rel_h_n", "team_rel_a", "team_rel_a_n",
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
    side_yes/approved/meta_iso for every SCORED market of the game. Returns
    market -> {sib_n, sib_frac, sib_adj_same_side, sib_same_n, sib_same_mean,
    sib_opp_n, sib_opp_mean}."""
    n_scored = len(rows)
    approved = [(r["market"], r["side_yes"], r["meta_iso"]) for r in rows if r["approved"]]
    approved_side = {m: sy for m, sy, _iso in approved}
    n_appr = len(approved)
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
        same = [iso for m, sy, iso in approved if m != r["market"] and sy == r["side_yes"]]
        opp = [iso for m, sy, iso in approved if m != r["market"] and sy != r["side_yes"]]
        out[r["market"]] = {
            "sib_n": sib_n, "sib_frac": sib_frac, "sib_adj_same_side": adj,
            "sib_same_n": float(len(same)),
            "sib_same_mean": float(np.mean(same)) if same else float("nan"),
            "sib_opp_n": float(len(opp)),
            "sib_opp_mean": float(np.mean(opp)) if opp else float("nan")}
    return out


# ---------------------------------------------- shared bulk/live builders ---
# Every rolling store is built by ONE function used by BOTH the dataset build
# and the live bundle, so bulk and live values are identical by construction.
def _season_runs(dates: np.ndarray) -> tuple[np.ndarray, float]:
    """`dates`: sorted unique datetime64[D] league game dates. Returns
    (per-date season-start array, typical season length in days). Seasons are
    runs split on gaps >= SEASON_GAP_DAYS; typical = median COMPLETED run
    length (the open last run is excluded), fallback 180. Schedule-derived
    only — no outcomes."""
    starts = dates.copy()
    lengths = []
    s = 0
    for i in range(1, len(dates)):
        if int((dates[i] - dates[i - 1]).astype(int)) >= SEASON_GAP_DAYS:
            lengths.append(int((dates[i - 1] - dates[s]).astype(int)) + 1)
            s = i
        starts[i] = dates[s]
    typical = float(np.median(lengths)) if lengths else TYPICAL_SEASON_FALLBACK
    return starts, typical


def _season_feats(dates: np.ndarray, starts: np.ndarray, typical: float,
                  d: date) -> tuple[float, float]:
    """(season_days, season_frac) for date d — strictly schedule-prior: only
    game dates <= d matter (a gap >= 45d back to the last known date means d
    itself opens a new season)."""
    if len(dates) == 0:
        return float("nan"), float("nan")
    d64 = np.datetime64(d)
    i = int(np.searchsorted(dates, d64, side="right")) - 1
    if i < 0 or int((d64 - dates[i]).astype(int)) >= SEASON_GAP_DAYS:
        return 0.0, 0.0
    days = float((d64 - starts[i]).astype(int))
    return days, days / max(typical, 1.0)


def _build_form_hist(idx: pd.DataFrame) -> dict:
    """league -> (approved-pick dates asc, correctness) for _form_window."""
    out = {}
    for lg, grp in idx[idx["approved"]].groupby("_league"):
        g = grp.sort_values("_date", kind="mergesort")
        out[lg] = (pd.to_datetime(g["_date"]).to_numpy(dtype="datetime64[D]"),
                   g["_y"].to_numpy().astype(float))
    return out


def _build_day_scores(idx: pd.DataFrame) -> dict:
    """(league, date) -> DESC-sorted meta_iso array of that day's approved
    picks. len() is the slate size; strictly-greater counts give day_rank."""
    out = {}
    ap = idx[idx["approved"]]
    for (lg, d), grp in ap.groupby(["_league", "_date"], sort=False):
        out[(lg, _as_date(d))] = np.sort(grp["meta_iso"].to_numpy())[::-1]
    return out


def _build_team_hist(idx: pd.DataFrame) -> dict:
    """(league, team) -> (approved-pick dates asc, correctness) over picks
    INVOLVING the team (home or away)."""
    ap = idx[idx["approved"]].sort_values("_date", kind="mergesort")
    d64 = pd.to_datetime(ap["_date"]).to_numpy(dtype="datetime64[D]")
    tmp: dict = {}
    for lg, h, a, d, y in zip(ap["_league"], ap["_home"], ap["_away"], d64,
                              ap["_y"].to_numpy().astype(float)):
        for t in (h, a):
            dates, ys = tmp.setdefault((lg, t), ([], []))
            dates.append(d)
            ys.append(y)
    return {k: (np.array(v[0], dtype="datetime64[D]"), np.array(v[1]))
            for k, v in tmp.items()}


def _team_rel(hist: tuple[np.ndarray, np.ndarray] | None, d: date) -> tuple[float, float]:
    """(hit rate, n) of the team's last TEAM_REL_WINDOW approved picks strictly
    before d (same-date picks excluded, like the model-err covariates)."""
    if hist is None:
        return float("nan"), 0.0
    dates, ys = hist
    i = int(np.searchsorted(dates, np.datetime64(d), side="left"))
    w = ys[max(0, i - TEAM_REL_WINDOW):i]
    return (float(w.mean()) if len(w) else float("nan")), float(len(w))


def _day_rank(scores: np.ndarray | None, own_iso: float) -> tuple[float, float]:
    """(rank, rank/slate) of own_iso within the day's DESC approved scores.
    Ties share the better rank (strictly-greater count + 1)."""
    if scores is None or len(scores) == 0 or np.isnan(own_iso):
        return float("nan"), float("nan")
    rank = 1.0 + float((scores > own_iso).sum())
    return rank, rank / float(len(scores))


# --------------------------------------------- Stage 1: out-of-fold scores --
def _data_sig(engine, league: str) -> tuple:
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    with engine.begin() as conn:
        n, mx = conn.execute(text(
            f"SELECT COUNT(*), MAX(match_date) FROM {spec['table']} "
            f"WHERE outcome_filled_at_utc IS NOT NULL{extra}")).fetchone()
    # block config + feature-schema version in the signature → changing the OOF
    # scheme OR betmeta's feature set invalidates caches. NOTE: a backtest
    # re-run that rewrites the same games does not move (n, max-date) — callers
    # that regenerate backtests must pass force=True (train_refiner(force_oof=..)).
    return int(n), str(mx), N_BLOCKS, SEED_BLOCKS, FRAME_VERSION


def build_oof(league: str, cfg: Config, engine=None, force: bool = False) -> pd.DataFrame:
    """The league's meta frame + `_block` (1..N_BLOCKS) + `oof_raw`/`oof_iso`
    (NaN on the seed block). Cached under models/oof/ keyed on the reconciled
    row-count + max date + block config, so unchanged data never retrains."""
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
    cuts = [dts.quantile(i / N_BLOCKS) for i in range(1, N_BLOCKS)]
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
      stores  — everything the live path needs: the compact OOF pick index
                (incl. home/away teams), per-league full game-date lists,
                z-stats, thresholds, and per-league coverage counts."""
    engine = create_engine(cfg)
    per_league, coverage, z_stats, thresholds, date_store = [], {}, {}, {}, {}
    for lg in LEAGUES:
        X = build_oof(lg, cfg, engine=engine, force=force_oof)
        spec = SPECS[lg]
        # Skip a league with no usable meta frame yet — an empty OOF (a cup still
        # awaiting its season, e.g. soccer_lgc before the Aug edition → 0 rows) or one
        # missing the base-expectation columns. Without this the whole meta² refiner
        # crashes on that one league (found: soccer_lgc, broken nightly 2026-07-14→17).
        if X is None or getattr(X, "empty", True) or any(c not in X.columns for c in spec["err_expected"]):
            logger.info("refiner: skipping %s — no usable OOF frame yet (below its history bar / off-season)", lg)
            continue
        art = _league_artifact(lg, cfg)
        thr_by = dict(art.get("threshold_by_market") or {})
        gthr = art.get("threshold")
        thresholds[lg] = {"by_market": thr_by, "global": gthr}
        info = _market_info(lg)
        neighbors = _line_neighbors(lg)
        # full-frame game-date list (season phase); schedule info, no outcomes
        date_store[lg] = np.sort(pd.to_datetime(X["_date"]).unique()).astype("datetime64[D]")
        # expected totals (vectorized; 2-term IEEE sums match the live path's
        # sequential adds exactly)
        X["_base_et"] = X[spec["err_expected"]].sum(axis=1, min_count=len(spec["err_expected"]))
        X["_corners_et"] = (X[["corner_lambda_home", "corner_lambda_away"]].sum(axis=1, min_count=2)
                            if "corner_lambda_home" in X.columns else np.nan)
        # cross-model z-stats from the SEED block only (strictly prior to all
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
        sib_keys = ("sib_n", "sib_frac", "sib_adj_same_side",
                    "sib_same_n", "sib_same_mean", "sib_opp_n", "sib_opp_mean")
        for k in sib_keys:
            D[k] = np.nan
        for _gid, grp in D.groupby("_game_id", sort=False):
            rows = [{"market": m, "line": ln, "side_yes": sy, "approved": ap, "meta_iso": mi}
                    for m, ln, sy, ap, mi in zip(grp["_market"], grp["_line"],
                                                 grp["side_yes"], grp["approved"],
                                                 grp["oof_iso"])]
            feats = _sibling_feats(rows, neighbors)
            for k in sib_keys:
                D.loc[grp.index, k] = [feats[m][k] for m in grp["_market"]]
        # only the pooled/global columns survive (league frames differ otherwise)
        per_league.append(D[["_date", "_game_id", "_market", "_league", "_kind", "_line",
                             "_home", "_away",
                             "_y", "p", "conf", "oof_raw", "oof_iso", "approved", "side_yes",
                             "line_margin", "xm_z_goals", "xm_z_corners", "xm_z_agree",
                             *sib_keys, *MODEL_ERR_FEATURES]])
        coverage[lg]["approved_rows"] = int(D["approved"].sum())
    pooled = pd.concat(per_league, ignore_index=True)
    pooled = pooled.rename(columns={"oof_iso": "meta_iso", "oof_raw": "meta_raw"})
    pooled["iso_raw_gap"] = (pooled["meta_iso"] - pooled["meta_raw"]).abs()
    for k in KINDS:
        pooled[f"kind_{k}"] = (pooled["_kind"] == k).astype(float)
        pooled[f"sk_yes_{k}"] = pooled["side_yes"] * pooled[f"kind_{k}"]
    for lg in LEAGUES:
        pooled[f"lg_{lg}"] = (pooled["_league"] == lg).astype(float)

    # compact OOF pick index: the live path's store for historical rows —
    # built FIRST so the rolling stores below come from the exact same frame
    # the live bundle rebuilds them from (value-identical by construction).
    oof_index = pooled[["_league", "_game_id", "_market", "_date", "_kind", "_line",
                        "_home", "_away", "side_yes", "meta_raw", "meta_iso",
                        "approved", "_y"]].copy()
    form_hist = _build_form_hist(oof_index)
    day_scores = _build_day_scores(oof_index)
    team_hist = _build_team_hist(oof_index)
    season = {lg: (date_store[lg], *_season_runs(date_store[lg])) for lg in date_store}

    cols = {k: [] for k in ("form_acc_30", "form_n_30", "slate_size", "day_rank",
                            "day_rank_frac", "season_days", "season_frac",
                            "team_rel_h", "team_rel_h_n", "team_rel_a", "team_rel_a_n")}
    empty_hist = (np.array([], dtype="datetime64[D]"), np.array([]))
    for lg, d, iso_v, h, a in zip(pooled["_league"], pooled["_date"], pooled["meta_iso"],
                                  pooled["_home"], pooled["_away"]):
        dd = _as_date(d)
        fa, fn = _form_window(form_hist.get(lg, empty_hist), dd)
        scores = day_scores.get((lg, dd))
        dr, drf = _day_rank(scores, float(iso_v))
        sd_, sf = _season_feats(*season[lg], dd)
        th, thn = _team_rel(team_hist.get((lg, h)), dd)
        ta, tan = _team_rel(team_hist.get((lg, a)), dd)
        for k, v in zip(cols, (fa, fn, float(len(scores)) if scores is not None else 0.0,
                               dr, drf, sd_, sf, th, thn, ta, tan)):
            cols[k].append(v)
    for k, v in cols.items():
        pooled[k] = v

    dataset = pooled[pooled["approved"]].copy().sort_values(
        ["_date", "_league", "_game_id"], kind="mergesort").reset_index(drop=True)
    stores = {"oof_index": oof_index, "z_stats": z_stats, "thresholds": thresholds,
              "coverage": coverage, "date_store": date_store}
    return dataset, stores


# ------------------------------------- Stage 3: train, floors, tier evals ---
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
    """Accuracy of the 🤖 floor giving the same n picks. When that volume
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


# --------------------------------- Stage 4: the adaptive hybrid engine gate --
def _choose_engines_from_artifact(art: dict) -> tuple[dict, dict]:
    """Per-tier engine choice on the CALIB best-per-game arrays (matched-volume
    vs the meta), then the one-way TEST clamp (meta-ward only). Returns
    (engines, detail) — engines = {"star_engine": ..., "diamond_engine": ...}."""
    ev_ca, ev_te = art["eval_arrays"]["calib"], art["eval_arrays"]["test"]
    engines, detail = {}, {}
    for tier, meta_floor in (("star", META_STAR_FLOOR), ("diamond", META_DIAMOND_FLOOR)):
        floor = (art.get("floors") or {}).get(tier)
        choice, info = "meta", {"refiner_floor": floor}
        if floor is not None:
            r, m, y = ev_ca["ref"], ev_ca["meta_iso"], ev_ca["y"]
            msk = r >= floor
            n = int(msk.sum())
            info["calib_n"] = n
            if n >= MIN_TIER_N:
                acc_r = float(y[msk].mean())
                base = _matched_baseline(m, y, n)
                info["calib_acc_refiner"] = round(acc_r, 4)
                info["calib_acc_meta_matched"] = round(base[0], 4) if base else None
                if base is not None and acc_r > base[0]:
                    choice = "refiner"
        if choice == "refiner":
            # conservative one-way clamp: test may only demote to meta
            rt, mt, yt = ev_te["ref"], ev_te["meta_iso"], ev_te["y"]
            hyb, pure = rt >= floor, mt >= meta_floor
            acc_h = float(yt[hyb].mean()) if hyb.any() else None
            acc_p = float(yt[pure].mean()) if pure.any() else None
            if acc_h is not None and acc_p is not None and acc_h < acc_p - TEST_CLAMP_PP:
                choice = "meta"
                info["test_clamp"] = {"hybrid_acc": round(acc_h, 4),
                                      "pure_meta_acc": round(acc_p, 4),
                                      "note": f"refiner won calib but was >{TEST_CLAMP_PP:.1%}"
                                              " worse on test → forced to meta"}
        engines[f"{tier}_engine"] = choice
        detail[tier] = info
    return engines, detail


def _hybrid_test_report(art: dict) -> dict:
    """FINAL untouched-test table: the shipped hybrid (chosen engines) vs the
    pure-meta floors, per tier: accuracy + volume."""
    ev = art["eval_arrays"]["test"]
    r, m, y = ev["ref"], ev["meta_iso"], ev["y"]
    days = max(len(set(ev["dates"])), 1)
    out = {}
    for tier, meta_floor in (("star", META_STAR_FLOOR), ("diamond", META_DIAMOND_FLOOR)):
        engine = (art.get("engines") or {}).get(f"{tier}_engine", "meta")
        floor = (art.get("floors") or {}).get(tier)
        hyb = (r >= floor) if (engine == "refiner" and floor is not None) else (m >= meta_floor)
        pure = m >= meta_floor
        out[tier] = {"engine": engine,
                     "hybrid": {"n": int(hyb.sum()), "correct": int(y[hyb].sum()),
                                "acc": round(float(y[hyb].mean()), 4) if hyb.any() else None,
                                "picks_per_day": round(int(hyb.sum()) / days, 2)},
                     "pure_meta": {"n": int(pure.sum()), "correct": int(y[pure].sum()),
                                   "acc": round(float(y[pure].mean()), 4) if pure.any() else None,
                                   "picks_per_day": round(int(pure.sum()) / days, 2)}}
    return out


def choose_engines(config: Config | None = None) -> dict:
    """Nightly hybrid gate: re-choose the per-tier engine on the artifact's
    CALIB arrays, apply the one-way test clamp, persist into the artifact.
    Safe no-op (meta everywhere) when the artifact is missing or legacy."""
    cfg = config or load_config()
    path = cfg.model.model_dir / "refiner.pkl"
    if not path.exists():
        return {"engines": None, "reason": "no refiner artifact"}
    with open(path, "rb") as f:
        art = pickle.load(f)
    if "eval_arrays" not in art:
        return {"engines": None, "reason": "legacy artifact without eval arrays"}
    engines, detail = _choose_engines_from_artifact(art)
    art["engines"] = engines
    art["engines_detail"] = detail
    art["engines_chosen_at"] = date.today().isoformat()
    art["hybrid_test_report"] = _hybrid_test_report(art)
    with open(path, "wb") as f:
        pickle.dump(art, f)
    global _bundle_cache
    _bundle_cache = {}
    logger.info("refiner engines chosen: %s", engines)
    return {"engines": engines, "detail": detail,
            "hybrid_test_report": art["hybrid_test_report"]}


def train_refiner(config: Config | None = None, force_oof: bool = False) -> dict:
    """Stages 1-4: build the dataset, train the global refiner, pick tier
    floors on CALIB, choose the per-tier engines (calib matched-volume + the
    one-way test clamp), persist models/refiner.pkl. The artifact ALWAYS
    ships — safety lives in the per-tier engine choice, and consumers fall
    back to meta floors whenever the artifact is missing/stale/broken."""
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

    # ------- TEST: scored exactly once; used only for REPORTING and the
    # one-way meta-ward clamp inside the engine choice ------------------------
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
    # fixed 🤖-floor baselines for context (what the meta engine would serve)
    by = te_best["_y"].to_numpy()
    bm = te_best["meta_iso"].to_numpy()
    days = max(te_best["_date"].nunique(), 1)
    fixed_baselines = {f"meta>={f}": {"n": int((bm >= f).sum()),
                                      "acc": round(float(by[bm >= f].mean()), 4) if (bm >= f).any() else None,
                                      "picks_per_day": round(int((bm >= f).sum()) / days, 2)}
                       for f in (META_STAR_FLOOR, META_DIAMOND_FLOOR)}
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
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])[:25]
    eval_arrays = {
        "calib": {"meta_iso": ca_best["meta_iso"].to_numpy().astype(float),
                  "ref": np.asarray(ca_best_ref, dtype=float),
                  "y": ca_best_y.astype(float),
                  "dates": np.array([str(d) for d in ca_best["_date"]])},
        "test": {"meta_iso": bm.astype(float),
                 "ref": np.asarray(te_best_ref, dtype=float),
                 "y": by.astype(float),
                 "dates": np.array([str(d) for d in te_best["_date"]])}}
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
                "date_store": stores["date_store"],
                "eval_arrays": eval_arrays,
                "audit_sample": audit_sample,
                "trained_at": date.today().isoformat()}
    # the adaptive hybrid: choose engines now so the saved artifact ships ready
    engines, detail = _choose_engines_from_artifact(artifact)
    artifact["engines"] = engines
    artifact["engines_detail"] = detail
    artifact["engines_chosen_at"] = date.today().isoformat()
    artifact["hybrid_test_report"] = _hybrid_test_report(artifact)
    path = cfg.model.model_dir / "refiner.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    global _bundle_cache
    _bundle_cache = {}
    logger.info("refiner trained: rows=%s auc(calib)=%.4f auc(test)=%.4f floors=%s engines=%s",
                len(ds), calib_auc, test_auc, artifact["floors"], engines)
    return {"n_dataset": len(ds), "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
            "params": chosen, "calib_auc": round(calib_auc, 4), "test_auc": round(test_auc, 4),
            "floors": artifact["floors"], "tiers_test": tiers, "gate_passed": gate_passed,
            "engines": engines, "engines_detail": detail,
            "hybrid_test_report": artifact["hybrid_test_report"],
            "fixed_baselines_test": fixed_baselines, "coverage": stores["coverage"],
            "importances": artifact["importances"]}


# --------------------------------------------------------- live scoring -----
_bundle_cache: dict = {}
_live_sib_cache: dict = {}
_live_day_cache: dict = {}


def _load_bundle(cfg: Config, require_fresh: bool = True):
    """Artifact + booster. None when there is no artifact — or (require_fresh)
    when it is legacy (no engines) or stale (> STALE_DAYS old), so consumers
    fall back to plain meta floors. Heavy runtime lookup structures are built
    lazily by _runtime() only when the refiner engine is actually used."""
    key = ("fresh" if require_fresh else "any")
    if key in _bundle_cache:
        return _bundle_cache[key]
    path = cfg.model.model_dir / "refiner.pkl"
    if not path.exists():
        _bundle_cache[key] = None
        return None
    with open(path, "rb") as f:
        art = pickle.load(f)
    if require_fresh:
        stale = True
        try:
            stale = (date.today() - date.fromisoformat(art.get("trained_at", ""))).days > STALE_DAYS
        except ValueError:
            pass
        if not art.get("engines") or stale:
            _bundle_cache[key] = None
            return None
    import lightgbm as lgb
    booster = lgb.Booster(model_str=art["model_str"])
    bundle = {"art": art, "booster": booster, "iso": art["iso"], "features": art["features"],
              "floors": art["floors"], "z_stats": art["z_stats"],
              "thresholds": art["thresholds"]}
    _bundle_cache[key] = bundle
    return bundle


def _runtime(bundle: dict) -> dict:
    """Build (once) the store/rolling structures the live path needs, from the
    artifact's oof_index — via the SAME builders the dataset build used."""
    if "store" in bundle:
        return bundle
    art = bundle["art"]
    idx = art["oof_index"]
    store: dict = {}
    for (lg, gid), grp in idx.groupby(["_league", "_game_id"], sort=False):
        store[(lg, gid)] = [
            {"market": m, "kind": k, "line": ln, "side_yes": sy, "approved": bool(ap),
             "meta_raw": mr, "meta_iso": mi}
            for m, k, ln, sy, ap, mr, mi in zip(grp["_market"], grp["_kind"], grp["_line"],
                                                grp["side_yes"], grp["approved"],
                                                grp["meta_raw"], grp["meta_iso"])]
    date_store = art.get("date_store") or {}
    bundle.update(
        store=store,
        form_hist=_build_form_hist(idx),
        day_scores=_build_day_scores(idx),
        team_hist=_build_team_hist(idx) if "_home" in idx.columns else {},
        season={lg: (ds, *_season_runs(ds)) for lg, ds in date_store.items()})
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


def _live_day_scores(league: str, cfg: Config, d: date, thresholds: dict) -> np.ndarray:
    """DESC approved-🤖 scores for a LIVE date's league slate: score the day's
    candidates (pending rows preferred, else backtest replay) with the
    production meta. Cached. len() = slate size; also feeds day_rank."""
    key = (league, d)
    if key in _live_day_cache:
        return _live_day_cache[key]
    if len(_live_day_cache) > 2000:
        _live_day_cache.clear()
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
    scores = []
    for r in rows:
        rd = dict(r._mapping)
        scores.extend(row["meta_iso"] for row in _live_game_rows(league, cfg, rd, thresholds)
                      if row["approved"])
    arr = np.sort(np.asarray(scores, dtype=float))[::-1]
    _live_day_cache[key] = arr
    return arr


def _pick_features(league: str, rd: dict, market: str, p: float,
                   meta_p: float | None, cfg: Config, bundle: dict) -> dict:
    """The refiner feature vector for one pick — historical rows come straight
    from the OOF store (exactly the training values, audited); live rows use
    the production meta for the meta/sibling/slate/day-rank inputs."""
    _runtime(bundle)
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
        scores = bundle["day_scores"].get((league, d))
    else:
        if meta_p is None:
            meta_raw, meta_iso = _meta_scores_live(league, cfg, rd, market, p)
        else:
            meta_raw, _ = _meta_scores_live(league, cfg, rd, market, p)
            meta_iso = float(meta_p)
        game_rows = _live_game_rows(league, cfg, rd, thresholds)
        scores = bundle["day_scores"].get((league, d))
        if scores is None:
            scores = _live_day_scores(league, cfg, d, thresholds)
    slate = float(len(scores)) if scores is not None else 0.0
    day_rank, day_rank_frac = _day_rank(scores, float(meta_iso))
    sib = _sibling_feats(game_rows, _line_neighbors(league)).get(market) or \
        {k: float("nan") for k in ("sib_n", "sib_frac", "sib_adj_same_side",
                                   "sib_same_n", "sib_same_mean", "sib_opp_n", "sib_opp_mean")}
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
    form_acc, form_n = _form_window(bundle["form_hist"].get(
        league, (np.array([], dtype="datetime64[D]"), np.array([]))), d)
    season = bundle["season"].get(league)
    sd_, sf = _season_feats(*season, d) if season else (float("nan"), float("nan"))
    trh, trh_n = _team_rel(bundle["team_hist"].get((league, rd.get("home_team"))), d)
    tra, tra_n = _team_rel(bundle["team_hist"].get((league, rd.get("away_team"))), d)
    he, ha = _team_recent_err(league, cfg, rd.get("home_team"), rd["match_date"])
    ae, aa = _team_recent_err(league, cfg, rd.get("away_team"), rd["match_date"])
    side_yes = 1.0 if p >= 0.5 else 0.0
    out = {"meta_iso": meta_iso, "meta_raw": meta_raw,
           "iso_raw_gap": abs(meta_iso - meta_raw),
           "p": p, "conf": max(p, 1 - p), "side_yes": side_yes,
           "line_margin": line_margin, **sib,
           "xm_z_goals": z_g, "xm_z_corners": z_c, "xm_z_agree": z_a,
           "form_acc_30": form_acc, "form_n_30": form_n, "slate_size": slate,
           "day_rank": day_rank, "day_rank_frac": day_rank_frac,
           "season_days": sd_, "season_frac": sf,
           "team_rel_h": trh, "team_rel_h_n": trh_n,
           "team_rel_a": tra, "team_rel_a_n": tra_n,
           "h_model_err": he, "h_model_abs_err": ha, "a_model_err": ae, "a_model_abs_err": aa}
    for k in KINDS:
        out[f"kind_{k}"] = 1.0 if k == kind else 0.0
        out[f"sk_yes_{k}"] = side_yes if k == kind else 0.0
    for lg in LEAGUES:
        out[f"lg_{lg}"] = 1.0 if lg == league else 0.0
    return out


def score_pick(league: str, row_dict: dict, market: str, p: float,
               meta_p: float | None, cfg: Config, *, require_fresh: bool = True) -> float | None:
    """Refiner P(pick correct) for one already-approved pick, or None when no
    usable artifact exists (missing / legacy / stale)."""
    bundle = _load_bundle(cfg, require_fresh=require_fresh)
    if not bundle:
        return None
    feats = _pick_features(league, dict(row_dict), market, float(p), meta_p, cfg, bundle)
    X = pd.DataFrame([feats]).reindex(columns=bundle["features"])
    raw = float(bundle["booster"].predict(X)[0])
    return float(bundle["iso"].predict([raw])[0])


def refiner_floors(cfg: Config) -> dict | None:
    bundle = _load_bundle(cfg)
    return bundle["floors"] if bundle else None


def shipped_engines(cfg: Config) -> dict:
    """The per-tier engines the hybrid currently serves (meta when unshipped)."""
    bundle = _load_bundle(cfg)
    eng = (bundle["art"].get("engines") or {}) if bundle else {}
    return {"star_engine": eng.get("star_engine", "meta"),
            "diamond_engine": eng.get("diamond_engine", "meta")}


def nivel_for_pick(league: str, row_dict: dict, market: str, p: float,
                   meta_p: float | None, cfg: Config) -> str:
    """'💎' / '⭐' / '✅' for an approved pick, via the CHOSEN engine per tier.
    Never raises, never returns None: any failure (missing/stale artifact,
    scoring error) degrades to the plain meta floors 0.90/0.95."""
    star_e = dia_e = "meta"
    floors: dict = {}
    bundle = None
    try:
        bundle = _load_bundle(cfg)
        if bundle:
            eng = bundle["art"].get("engines") or {}
            star_e = eng.get("star_engine", "meta")
            dia_e = eng.get("diamond_engine", "meta")
            floors = bundle.get("floors") or {}
    except Exception:  # never let the refiner break a digest/dashboard
        logger.exception("refiner bundle load failed — using meta floors")
    s = None
    if "refiner" in (star_e, dia_e):
        if bundle is not None and league not in bundle.get("thresholds", {}):
            # league newer than the artifact (e.g. a cup added today) — the nightly
            # retrain will include it; until then meta floors apply, quietly.
            logger.debug("refiner artifact predates league %s — meta floors", league)
        else:
            try:
                s = score_pick(league, row_dict, market, float(p), meta_p, cfg)
            except Exception:
                logger.exception("refiner score_pick failed for %s/%s — meta floors", league, market)
    mp = float(meta_p) if meta_p is not None else float("nan")

    def _tier(engine: str, tier: str, meta_floor: float) -> bool:
        if engine == "refiner" and s is not None and floors.get(tier) is not None:
            return s >= floors[tier]
        return bool(not np.isnan(mp) and mp >= meta_floor)

    if _tier(dia_e, "diamond", META_DIAMOND_FLOOR):
        return "💎"
    if _tier(star_e, "star", META_STAR_FLOOR):
        return "⭐"
    return "✅"


def star_for_candidate(league: str, cfg: Config, row, cand: dict) -> str:
    """Digest prefix '💎 ' / '⭐ ' / '' for one meta-approved candidate dict
    (as produced by meta_gate: has 'market' and 'meta'). Never raises."""
    try:
        rd = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        pcol = _market_info(league)[cand["market"]][0]
        pv = rd.get(pcol)
        p = float(pv) if pv is not None else float(cand.get("conf") or 0.5)
        nv = nivel_for_pick(league, rd, cand["market"], p, cand.get("meta"), cfg)
    except Exception:
        logger.exception("star_for_candidate failed for %s — meta floors", league)
        m = cand.get("meta") or 0
        nv = ("💎" if m >= META_DIAMOND_FLOOR
              else ("⭐" if m >= META_STAR_FLOOR else "✅"))
    return {"💎": "💎 ", "⭐": "⭐ "}.get(nv, "")


# -------------------------------------------------------------- audit -------
def audit_equality(cfg: Config | None = None, tol: float = 1e-6) -> dict:
    """Bulk-vs-live equality: re-score the artifact's 30 sampled TEST rows
    through the LIVE path (raw DB row → score_pick) and compare to the bulk
    training-time scores, feature by feature. Like betmeta's model-err audit."""
    cfg = cfg or load_config()
    bundle = _load_bundle(cfg, require_fresh=False)
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
        live = score_pick(lg, rd, market, s["p"], None, cfg, require_fresh=False)
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
