"""Data layer for the Sandy dashboard — plain pandas/SQL, no streamlit imports.

Every function creates its own engine (cheap) so the streamlit app can wrap
them with st.cache_data keyed on the plain-value arguments.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from sandy.betmeta import (SPECS, _attach_model_err, _attach_team_rel,
                           _attach_weather, _correct, _row_features, load_meta)
from sandy.config import load_config
from sandy.db import create_engine
from sandy.mls.recommend import RECOMMEND_MIN_ACC, RECOMMEND_MIN_N, bucket_acc

LEAGUES = {
    "soccer_col": ("🇨🇴", "Liga Colombia"),
    "soccer_mex": ("🇲🇽", "Liga MX"),
    "soccer_esp": ("🇪🇸", "La Liga"),
    "soccer_eng": ("🏴", "Premier League"),
    "mls": ("⚽", "MLS"),
    "nhl": ("🏒", "NHL"),
    "nba": ("🏀", "NBA"),
    "nfl": ("🏈", "NFL"),
    "worldcup": ("🏆", "Mundial 2026"),
    "mlb": ("⚾", "MLB"),
}


def league_title(key: str) -> str:
    flag, name = LEAGUES[key]
    return f"{flag} {name}"


def market_label(market: str, kind: str | None = None) -> str:
    if market == "double_chance":
        return "Doble oportunidad (1X/2)"
    if market == "winner":
        return "Ganador"
    if market == "btts":
        return "Ambos anotan (BTTS)"
    line = market.rsplit("over_", 1)[-1].replace("_", ".")
    if kind == "runs":
        return f"Carreras {line}"
    if kind == "points":  # NBA and NFL totals are points at ANY line (NFL ~44.5)
        return f"Puntos {line}"
    if market.startswith("corners_"):
        return f"Corners {line}"
    return f"Puntos {line}" if float(line) > 50 else f"Goles {line}"


def _pick_labels(kind: str, line) -> tuple[str, str]:
    """(label if p>=.5, label if p<.5)"""
    if kind == "result":
        return "1X (local o empata)", "2 (gana visitante)"
    if kind == "winner":
        return "Gana local", "Gana visitante"
    if kind == "btts":
        return "Ambos anotan: SÍ", "Ambos anotan: NO"
    unit = {"goals": "goles", "corners": "corners", "points": "puntos", "runs": "carreras"}[kind]
    return f"Más de {line} {unit}", f"Menos de {line} {unit}"


def _actual_str(rd: dict, kind: str) -> str:
    if kind in ("winner", "points"):
        return f"{rd.get('actual_home_points')}-{rd.get('actual_away_points')}"
    if kind == "corners":
        c = rd.get("actual_total_corners")
        return f"{c} corners" if c is not None else ""
    if kind == "runs":
        t = rd.get("actual_total_runs")
        return f"{int(t)} carreras" if t is not None and not pd.isna(t) else ""
    return f"{rd.get('actual_home_goals')}-{rd.get('actual_away_goals')}"


def _meta(league: str):
    return load_meta(league, load_config())  # (booster, feats, thr, iso) or None


def meta_threshold(league: str) -> float | None:
    loaded = _meta(league)
    return loaded[2] if loaded else None


def _reliability(conn, league: str) -> dict:
    spec = SPECS[league]
    lg_filter = "WHERE league = :lg" if spec.get("league") else ""
    rows = conn.execute(text(f"""
        SELECT DISTINCT ON (market) market, reliability
        FROM {spec['schema']}.calibration_snapshots {lg_filter}
        ORDER BY market, created_at DESC
    """), {"lg": spec.get("league")}).fetchall()
    import json
    return {m: (r if isinstance(r, list) else json.loads(r)) for m, r in rows}


def scored_results(league: str) -> pd.DataFrame:
    """Every reconciled (game × market) prediction, scored by the meta-model.

    Columns: match_date, home, away, market, pick, conf, correct, meta, holdout.
    `holdout` marks rows after the meta's chronological 70% training cut — only
    those are honest for evaluating the meta (it never trained on them).
    """
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    engine = create_engine(load_config())
    with engine.begin() as conn:
        df = pd.read_sql(text(
            f"SELECT * FROM {spec['table']} WHERE outcome_filled_at_utc IS NOT NULL{extra}"
        ), conn)
    # Same bulk model-error covariates as train_meta's _frame — keeps the meta
    # scores here identical to what the artifact saw in training (the raw rd
    # dicts alone lack these columns).
    df = _attach_model_err(df, spec)
    df = _attach_team_rel(df, spec)
    df = _attach_weather(df, spec, league, engine)
    out, feat_rows = [], []
    for _, r in df.iterrows():
        rd = r.to_dict()
        for market, (pcol, kind, line) in spec["markets"].items():
            p = rd.get(pcol)
            if p is None or pd.isna(p):
                continue
            y = _correct(rd, kind, line, float(p))
            if y is None:
                continue
            p = float(p)
            yes, no = _pick_labels(kind, line)
            out.append({
                "match_date": rd["match_date"], "home": rd["home_team"], "away": rd["away_team"],
                "market": market, "pick": yes if p >= 0.5 else no,
                "conf": p if p >= 0.5 else 1 - p, "correct": bool(y),
                "resultado": _actual_str(rd, kind),
            })
            feat_rows.append(_row_features(rd, spec, market, p))
    res = pd.DataFrame(out)
    if res.empty:
        return res
    loaded = _meta(league)
    if loaded:
        booster, feats, _thr, iso, _by_mkt = loaded
        X = pd.DataFrame(feat_rows).reindex(columns=feats)
        raw = booster.predict(X.to_numpy())
        res["meta"] = iso.predict(raw) if iso is not None else raw
    else:
        res["meta"] = None
    res = res.sort_values("match_date").reset_index(drop=True)
    cut = res["match_date"].quantile(0.8)  # train_meta's TEST split (final 20%)
    res["holdout"] = res["match_date"] > cut
    return res


def today_board(league: str, day: date) -> pd.DataFrame:
    """Every (game × market) candidate for one day, with hist + meta + gate flag.

    Uses live pending predictions when they exist for that day; otherwise falls
    back to walk-forward backtest rows (historical replay, leakage-free).
    """
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    engine = create_engine(load_config())
    rec_thr = meta_threshold(league)
    from sandy.betmeta import score_candidate
    cfg = load_config()
    with engine.begin() as conn:
        params = {"a": day, "b": day + timedelta(days=1)}
        rows = conn.execute(text(f"""
            SELECT * FROM {spec['table']}
            WHERE match_date BETWEEN :a AND :b{extra}
              AND outcome_filled_at_utc IS NULL AND NOT is_backtest
            ORDER BY match_date, id
        """), params).fetchall()
        sim = not rows
        if sim:
            rows = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date BETWEEN :a AND :b{extra} AND is_backtest
                ORDER BY match_date, id
            """), params).fetchall()
        reliability = _reliability(conn, league)
    out = []
    for r in rows:
        rd = dict(r._mapping)
        for market, (pcol, kind, line) in spec["markets"].items():
            p = rd.get(pcol)
            if p is None:
                continue
            p = float(p)
            yes, no = _pick_labels(kind, line)
            conf = p if p >= 0.5 else 1 - p
            acc, n = bucket_acc(reliability.get(market), conf)
            meta_p = score_candidate(league, cfg, rd, market, p)
            hist_ok = acc is not None and n >= RECOMMEND_MIN_N and acc >= RECOMMEND_MIN_ACC
            meta_ok = meta_p is not None and rec_thr is not None and meta_p >= rec_thr
            out.append({
                "liga": league_title(league), "partido": f"{rd['home_team']} vs {rd['away_team']}",
                "fecha": rd["match_date"], "mercado": market_label(market, kind), "pick": yes if p >= 0.5 else no,
                "prob": conf, "hist": acc, "hist_n": n, "meta": meta_p,
                "umbral": rec_thr, "recomendada": bool(hist_ok and meta_ok), "replay": sim,
            })
    df = pd.DataFrame(out)
    if not df.empty:
        for c in ("prob", "hist", "meta", "umbral"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def meta_artifact(league: str) -> dict:
    """Full meta artifact (eval_by_market ladders, per-line thresholds, …) — last trained."""
    import pickle
    path = load_config().model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _acc_at(art: dict, market: str, thr: float | None):
    """Holdout accuracy (and n) of this market at its recommended threshold."""
    if thr is None:
        return None, 0
    for t in (art.get("eval_by_market", {}).get(market, {}).get("table") or []):
        if abs(t["thr"] - thr) < 1e-9:
            return t.get("acc"), t.get("n") or 0
    return None, 0


def date_bounds(league: str):
    spec = SPECS[league]
    extra = f" WHERE {spec['where']}" if spec.get("where") else ""
    engine = create_engine(load_config())
    with engine.begin() as conn:
        lo, hi = conn.execute(text(
            f"SELECT MIN(match_date), MAX(match_date) FROM {spec['table']}{extra}")).fetchone()
    return lo, hi


def board_range(league: str, start: date, end: date) -> pd.DataFrame:
    """The sketch's Games table: one row per game in [start, end]; per market a
    composite cell `prob (🤖meta / Th→Acu) ✅` — with the real result and per-pick
    ✓/✗ for finished games (future games have no result yet).

    Pick rows also carry cuota / mercado % / edge / EV when TheOddsAPI stored
    matched odds for that game+line (see sandy/odds.py). edge/EV use the pick's
    OWN `prob` (base-model calibrated side probability) — NOT 🤖, which is
    P(pick correct), a different quantity."""
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    live_cond = ("TRUE" if spec.get("no_backtest_col")
                 else "((outcome_filled_at_utc IS NULL AND NOT is_backtest) OR is_backtest)")
    engine = create_engine(load_config())
    cfg = load_config()
    art = meta_artifact(league)
    thr_by = art.get("threshold_by_market") or {}
    global_thr = art.get("threshold")
    from sandy.betmeta import score_candidate
    from sandy.betrefine import nivel_for_pick
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT * FROM {spec['table']}
            WHERE match_date BETWEEN :a AND :b{extra} AND {live_cond}
            ORDER BY match_date, id"""), {"a": start, "b": end}).fetchall()
    try:  # odds are decoration: any failure leaves the board intact
        from sandy import odds as _odds
        oidx = _odds.odds_index(league, start, end, engine)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("odds_index failed — board served without cuotas")
        oidx = {}
    out, picks, all_ok = [], [], []
    # Doubleheaders (two rows, same date + teams — e.g. a postponed MLB game made up the
    # next day): market odds are keyed by (date, teams) and CANNOT tell the two games
    # apart, so cuota/edge/EV are suppressed for those rows — never price the wrong game.
    # The `hora` column (first pitch, Bogota) is what tells the games apart for the human.
    from collections import Counter
    _key = lambda rd: (rd["match_date"], (rd.get("home_team") or rd.get("home") or "").strip(),
                       (rd.get("away_team") or rd.get("away") or "").strip())
    dh_keys = {k for k, n in Counter(_key(dict(r._mapping)) for r in rows).items() if n > 1}
    for r in rows:
        rd = dict(r._mapping)
        fp = rd.get("first_pitch_utc")
        hora = None
        if fp is not None:
            try:
                hora = pd.Timestamp(fp).tz_convert("America/Bogota").strftime("%I:%M %p").lstrip("0")
            except Exception:
                hora = None
        base = {"fecha": rd["match_date"], "hora": hora or "—",
                "local": rd["home_team"] if "home_team" in rd else rd.get("home"),
                "visitante": rd["away_team"] if "away_team" in rd else rd.get("away")}
        finished = rd.get("outcome_filled_at_utc") is not None
        res_str = ""
        best = None
        for market, (pcol, kind, line) in spec["markets"].items():
            p = rd.get(pcol)
            if p is None:
                base[market_label(market, kind)] = "—"
                continue
            p = float(p)
            conf = p if p >= 0.5 else 1 - p
            thr = thr_by.get(market, global_thr)
            acc_thr, _n = _acc_at(art, market, thr)
            mp = score_candidate(league, cfg, rd, market, p)
            ok = mp is not None and thr is not None and mp >= thr
            correct = _correct(rd, kind, line, p) if finished else None
            cell = f"{conf:.0%}"
            if mp is not None and thr is not None:
                cell += f" (🤖{mp:.0%} / Th{thr:.0%}→{(acc_thr or 0):.0%})"
            if ok:
                cell += " ✅"
            if correct is not None:
                cell += " ✓" if correct else " ✗"
            base[market_label(market, kind)] = cell
            if finished and not res_str:
                res_str = _actual_str(rd, kind)
            if ok:
                yes, no = _pick_labels(kind, line)
                # adaptive hybrid: per-tier engine (refiner or meta floors),
                # chosen nightly on calib — falls back to meta floors on its own
                nivel = nivel_for_pick(league, rd, market, p, mp, cfg)
                pick_row = {"nivel": nivel,
                            "fecha": rd["match_date"], "hora": base["hora"],
                            "partido": f"{base['local']} vs {base['visitante']}",
                            "mercado": market_label(market, kind), "pick": yes if p >= 0.5 else no,
                            "prob": conf, "🤖": mp, "umbral": thr, "acierto_hist": acc_thr,
                            "resultado": res_str or "(pendiente)",
                            "acertó": ("✓" if correct else "✗") if correct is not None else "—"}
                # cuota/mercado %/edge/EV from matched TheOddsAPI odds (may be
                # empty). edge uses `conf` = the pick's own side prob, NOT 🤖.
                vstats = {"cuota": None, "mercado %": None, "edge": None, "EV": None}
                if oidx and _key(rd) not in dh_keys:
                    from sandy import odds as _odds
                    mapping = _odds.market_to_api(league, market)
                    if mapping:
                        api_m, pt = mapping
                        hit = oidx.get((rd["match_date"], (base["local"] or "").strip(),
                                        (base["visitante"] or "").strip(), api_m, pt,
                                        _odds.pick_side(kind, p)))
                        vstats = _odds.value_stats(conf, hit)
                pick_row.update(vstats)
                all_ok.append(pick_row)
                if best is None or (mp or 0) > best["🤖"]:
                    best = pick_row
        base["resultado"] = res_str or "(por jugar)"
        out.append(base)
        if best:
            picks.append(best)
    games = pd.DataFrame(out)
    finals = pd.DataFrame(picks).sort_values("🤖", ascending=False) if picks else pd.DataFrame()
    todos = pd.DataFrame(all_ok).sort_values("🤖", ascending=False) if all_ok else pd.DataFrame()
    return games, finals, todos


def calibration_latest(league: str) -> pd.DataFrame:
    """Latest calibration snapshot per market (accuracy, n, recommended threshold)."""
    spec = SPECS[league]
    lg_filter = "WHERE league = :lg" if spec.get("league") else ""
    engine = create_engine(load_config())
    with engine.begin() as conn:
        df = pd.read_sql(text(f"""
            SELECT DISTINCT ON (market) market, snapshot_date, sample_size, accuracy,
                   recommended_threshold, reliability
            FROM {spec['schema']}.calibration_snapshots {lg_filter}
            ORDER BY market, created_at DESC
        """), conn, params={"lg": spec.get("league")})
    return df


COVARIATE_LABELS = {
    "goals_for_5": "Goles a favor (últ. 5)", "goals_against_5": "Goles en contra (últ. 5)",
    "corners_for_5": "Corners a favor (últ. 5)", "corners_against_5": "Corners en contra (últ. 5)",
    "form_points_5": "Puntos de forma (últ. 5)", "rest_days": "Días de descanso",
    "played_10": "Partidos jugados (vent. 10)", "gf_5": "Goles a favor (últ. 5)",
    "ga_5": "Goles en contra (últ. 5)", "gf_10": "Goles a favor (últ. 10)",
    "ga_10": "Goles en contra (últ. 10)", "points_10": "Puntos (últ. 10)",
    "back_to_back": "Back-to-back", "pf_5": "Puntos anotados (últ. 5)",
    "pa_5": "Puntos recibidos (últ. 5)", "pf_10": "Puntos anotados (últ. 10)",
    "pa_10": "Puntos recibidos (últ. 10)", "wins_10": "Victorias (últ. 10)",
    "lambda_home": "λ goles local", "lambda_away": "λ goles visitante",
    "corner_lambda_home": "λ corners local", "corner_lambda_away": "λ corners visitante",
    "exp_home_points": "Puntos esperados local", "exp_away_points": "Puntos esperados visitante",
    "exp_total": "Total esperado", "sigma_total": "σ total", "p_home_win": "P(gana local)",
    "p": "Prob. del modelo", "conf": "Confianza",
    "model_err": "Error medio del modelo (últ. 8)",
    "model_abs_err": "Error abs. medio del modelo (últ. 8)",
    # clima del partido (open-meteo; hora del primer lanzamiento / kickoff)
    "wx_temp": "Temperatura (°C)", "wx_wind": "Viento (km/h)",
    "wx_precip": "Lluvia (mm)", "wx_dome": "Techo cerrado",
}


def covariate_label(key: str) -> str:
    base = COVARIATE_LABELS.get(key)
    if base:
        return base
    if key.startswith("h_"):
        return f"Local · {COVARIATE_LABELS.get(key[2:], key[2:])}"
    if key.startswith("a_"):
        return f"Visitante · {COVARIATE_LABELS.get(key[2:], key[2:])}"
    if key.startswith("mkt_"):
        return f"Mercado: {market_label(key[4:])}"
    return key


def game_covariates(league: str, day: date) -> pd.DataFrame:
    """One row per game with every covariate the models see (form, rest, lambdas).
    Live rows when the day has pending predictions; backtest rows otherwise."""
    import json as _json
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    engine = create_engine(load_config())
    with engine.begin() as conn:
        params = {"a": day, "b": day + timedelta(days=1)}
        if spec.get("no_backtest_col"):
            rows = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date BETWEEN :a AND :b{extra}
                ORDER BY match_date, id"""), params).fetchall()
        else:
            rows = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date BETWEEN :a AND :b{extra}
                  AND outcome_filled_at_utc IS NULL AND NOT is_backtest
                ORDER BY match_date, id"""), params).fetchall()
            if not rows:
                rows = conn.execute(text(f"""
                    SELECT * FROM {spec['table']}
                    WHERE match_date BETWEEN :a AND :b{extra} AND is_backtest
                    ORDER BY match_date, id"""), params).fetchall()
    # weather covariates (MLB/NFL when the spec has a wx_key) — same stored
    # rows the meta sees (odds.game_weather), labeled in Spanish
    wx_map: dict = {}
    if spec.get("wx_key") and rows:
        from sandy import weather as _wx
        wx_map = _wx.weather_map(league, engine)
    out = []
    for r in rows:
        rd = dict(r._mapping)
        feats = rd.get("features")
        if isinstance(feats, str):
            try:
                feats = _json.loads(feats)
            except (TypeError, ValueError):
                feats = None
        feats = feats or {}
        home, away = feats.get("home") or {}, feats.get("away") or {}
        row_out = {"partido": f"{rd['home_team']} vs {rd['away_team']}", "fecha": rd["match_date"]}
        for c in spec["num_cols"]:
            row_out[covariate_label(c)] = rd.get(c)
        for k in spec["form_keys"]:
            row_out[covariate_label(f"h_{k}")] = home.get(k)
            row_out[covariate_label(f"a_{k}")] = away.get(k)
        if spec.get("wx_key"):
            wx = wx_map.get(str(rd.get(spec["wx_key"])))
            for i, k in enumerate(("wx_temp", "wx_wind", "wx_precip", "wx_dome")):
                v = wx[i] if wx else None
                row_out[covariate_label(k)] = (None if v is None or v != v
                                               else round(float(v), 1))
        out.append(row_out)
    return pd.DataFrame(out)


def meta_importances(league: str) -> pd.DataFrame:
    """Which covariates the meta-model leans on (LightGBM gain, normalized)."""
    import pickle
    path = load_config().model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        art = pickle.load(f)
    imp = art.get("importances") or []
    total = sum(g for _f, g in imp) or 1.0
    return pd.DataFrame([{"covariable": covariate_label(f), "importancia %": round(100 * g / total, 1)}
                         for f, g in imp])


def meta_summary(league: str) -> dict:
    """Headline artifact facts: threshold, AUC, holdout size, trained_at."""
    import pickle
    path = load_config().model.model_dir / f"{league}_meta.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        art = pickle.load(f)
    return {"threshold": art.get("threshold"), "auc": art.get("auc"),
            "holdout_rows": art.get("holdout_rows"), "trained_rows": art.get("trained_rows"),
            "trained_at": art.get("trained_at")}


def spec_markets(league: str) -> dict[str, str]:
    return {m: market_label(m, SPECS[league]["markets"][m][1]) for m in SPECS[league]["markets"]}


def ladder(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Accuracy ladder over meta thresholds for an already-filtered scored frame."""
    rows = []
    for thr in thresholds:
        m = df[df["meta"] >= thr]
        rows.append({"umbral 🤖": f"≥{thr:.0%}", "picks": len(m),
                     "aciertos": int(m["correct"].sum()),
                     "acierto %": round(100 * m["correct"].mean(), 1) if len(m) else None})
    return pd.DataFrame(rows)
