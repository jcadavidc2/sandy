"""Sandy Football dashboard (Streamlit).

Phone/laptop view of the World Cup model: today's picks, calibration/trust,
team strength ratings, recent results, and data coverage. Reads the `football`
Postgres tables directly. Run:

    streamlit run sandy/dashboard/app.py

Exposed remotely via Cloudflare Tunnel (no open inbound ports).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from sandy.config import load_config
from sandy.db import create_engine
from sandy.football.queries import get_latest_calibration, get_recent_results, get_today_predictions

st.set_page_config(page_title="Sandy · World Cup 2026", page_icon="⚽", layout="wide")


@st.cache_resource
def _engine():
    return create_engine(load_config())


@st.cache_data(ttl=300)
def _df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _conf_color(v: float) -> str:
    """GREEN = confident pick, YELLOW = lean, RED = coin-flip."""
    if v >= 0.65:
        return "background-color: #1b5e20; color: white"
    if v >= 0.50:
        return "background-color: #f9a825; color: black"
    return "background-color: #b71c1c; color: white"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
cfg = load_config()
st.title("⚽ Sandy · World Cup 2026")
st.caption(
    "Dixon-Coles model · single free data source (API-Football) · "
    "calibrated on a 3,000+ match walk-forward backtest. Vibes modeling, not a wagering tool."
)

# ---------------------------------------------------------------------------
# Today's picks
# ---------------------------------------------------------------------------
st.header("🔮 Upcoming picks")
preds = get_today_predictions(_engine(), cfg)
if not preds:
    st.info("No upcoming World Cup matches in the current window. The nightly pipeline refreshes these.")
else:
    rows = []
    for p in preds:
        conf = max(p["p_home_win"], p["p_draw"], p["p_away_win"])
        fav = max([(p["home"], p["p_home_win"]), ("Draw", p["p_draw"]), (p["away"], p["p_away_win"])],
                  key=lambda x: x[1])
        rows.append({
            "Match": f"{p['home']} vs {p['away']}",
            "Pick": fav[0],
            "Win%": round(p["p_home_win"] * 100),
            "Draw%": round(p["p_draw"] * 100),
            "Loss%": round(p["p_away_win"] * 100),
            "Likely": f"{p['most_likely_home']}-{p['most_likely_away']}",
            "O2.5%": round(p["p_over_2_5"] * 100),
            "BTTS%": round(p["p_btts"] * 100),
            "Conf": conf,
        })
    df = pd.DataFrame(rows).sort_values("Conf", ascending=False).reset_index(drop=True)
    styled = (df.style
              .map(_conf_color, subset=["Conf"])
              .format({"Conf": "{:.0%}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("**O2.5%** = chance of 3+ total goals · **BTTS%** = both teams score · "
               "**Conf** = how sure the top pick is. 🟩 confident · 🟨 lean · 🟥 coin-flip.")

# ---------------------------------------------------------------------------
# Game explorer — pick any match, see prediction (+ result if finished)
# ---------------------------------------------------------------------------
st.header("🔍 Game explorer")
st.caption("Pick any World Cup match — upcoming (prediction only) or past "
           "(prediction vs actual result + stats).")
from sandy.football.queries import get_match_detail, get_match_options  # noqa: E402

_opts = get_match_options(_engine())
if not _opts:
    st.info("No World Cup predictions yet.")
else:
    def _label(o: dict) -> str:
        d = o["match_date"]
        if o["status"] in ("FT", "AET", "PEN") and o["actual_home_goals"] is not None:
            return f"{d} · {o['home']} {o['actual_home_goals']}-{o['actual_away_goals']} {o['away']} (final)"
        return f"{d} · {o['home']} vs {o['away']} (upcoming)"

    labels = {_label(o): o["fixture_id"] for o in _opts}
    pick = st.selectbox("Match", list(labels.keys()))
    d = get_match_detail(_engine(), labels[pick])
    if d:
        finished = d["status"] in ("FT", "AET", "PEN") and d["actual_home_goals"] is not None
        st.subheader(f"{d['home']} vs {d['away']}")
        st.caption(f"{d['match_date']} · {d['competition']} · {d['round'] or ''}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{d['home']} win", f"{d['p_home_win']*100:.0f}%")
        c2.metric("Draw", f"{d['p_draw']*100:.0f}%")
        c3.metric(f"{d['away']} win", f"{d['p_away_win']*100:.0f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("Most likely score", f"{d['most_likely_home']}-{d['most_likely_away']}")
        c5.metric("Over 2.5 goals", f"{d['p_over_2_5']*100:.0f}%")
        c6.metric("Both teams score", f"{d['p_btts']*100:.0f}%")

        # Top scorelines from the stored distribution.
        sl = d.get("scoreline")
        if isinstance(sl, list) and sl:
            st.caption("Most likely scorelines: " + " · ".join(
                f"{c['h']}-{c['a']} ({c['p']*100:.0f}%)" for c in sl[:5]))

        if finished:
            ah, aa = d["actual_home_goals"], d["actual_away_goals"]
            st.markdown(f"### Actual result: **{d['home']} {ah}-{aa} {d['away']}**")
            badges = []
            badges.append(("Result", d.get("was_correct_result")))
            badges.append(("Over/Under 2.5", d.get("was_correct_over_2_5")))
            badges.append(("BTTS", d.get("was_correct_btts")))
            badges.append(("Exact score", d.get("was_correct_score")))
            cols = st.columns(len(badges))
            for col, (name, ok) in zip(cols, badges):
                col.metric(name, "✅" if ok else "❌")
            if d.get("stats"):
                st.caption("Match stats")
                st.dataframe(pd.DataFrame(d["stats"]), use_container_width=True, hide_index=True)
            else:
                st.caption("Match stats: insufficient data (per-match stats still trickling in).")
        else:
            st.info("Upcoming match — result and stats will appear here after it's played.")

# ---------------------------------------------------------------------------
# Confederation scouting report (original-prompt deliverable)
# ---------------------------------------------------------------------------
st.header("📋 Confederation report")
st.caption("Last-20-match form per team. 🟩 favorable · 🟨 even · 🟥 unfavorable "
           "(by estimated win probability). Stat columns show *insufficient data* "
           "until per-match stats finish trickling in (API daily cap).")
from sandy.football.report import build_form_report, INSUFFICIENT  # noqa: E402


@st.cache_data(ttl=300)
def _report():
    return build_form_report(_engine(), cfg)


report, top = _report()
if not report:
    st.info("No report data yet.")
else:
    confs = [c for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC", "Other"] if c in report]
    choice = st.selectbox("Confederation", confs)

    def _wdl_color(v):
        if not isinstance(v, (int, float)) or pd.isna(v):
            return ""
        if v >= 0.55:
            return "background-color: #1b5e20; color: white"
        if v >= 0.40:
            return "background-color: #f9a825; color: black"
        return "background-color: #b71c1c; color: white"

    rows = []
    for r in report[choice]:
        if r.get("status") == INSUFFICIENT:
            rows.append({"Team": r["team"], "Est Win": None, "GF/g": INSUFFICIENT})
            continue
        rows.append({
            "Team": r["team"],
            "Est Win": r["est_win"], "Draw": r["est_draw"], "Loss": r["est_loss"],
            "GF/g": r["gf_pg"], "GA/g": r["ga_pg"],
            "O2.5%": r["over25_pct"], "BTTS%": r["btts_pct"], "Last 5": r["last5"],
            "Corners": r["corners"], "Cards": r["cards"], "Poss%": r["possession"],
            "SoT": r["shots_on_target"], "🔥 Hot stat": r["hot_stat"],
        })
    rdf = pd.DataFrame(rows)
    fmt = {c: "{:.0%}" for c in ["Est Win", "Draw", "Loss"] if c in rdf.columns}
    styled = rdf.style.format(fmt, na_rep="—")
    if "Est Win" in rdf.columns:
        styled = styled.map(_wdl_color, subset=["Est Win"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("⭐ TOP picks (best recent form)")
    tdf = pd.DataFrame([{
        "Team": r["team"], "Confederation": r["confederation"],
        "Form (pts/g)": r["ppg"], "Est Win": r.get("est_win"),
        "Last 5": r["last5"], "Most bettable": r["hot_stat"],
    } for r in top])
    st.dataframe(tdf.style.format({"Est Win": "{:.0%}", "Form (pts/g)": "{:.2f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Calibration / trust
# ---------------------------------------------------------------------------
st.header("📊 Model calibration (trust signal)")
cal = get_latest_calibration(_engine())
if not cal:
    st.info("No calibration snapshots yet.")
else:
    cols = st.columns(len(cal))
    for c, snap in zip(cols, cal):
        c.metric(
            label=f"{snap['market']} accuracy",
            value=f"{snap['accuracy']*100:.0f}%",
            help=f"n={snap['sample_size']} · trust ≥ {snap['recommended_threshold']}",
        )
    # Reliability curve for the result market (confidence bucket -> accuracy).
    result_snap = next((c for c in cal if c["market"] == "result"), None)
    if result_snap:
        ci = result_snap.get("covariate_insights") or {}
        rel = ci.get("reliability") if isinstance(ci, dict) else None
        if rel:
            rel_df = pd.DataFrame(rel).set_index("bucket")
            st.caption("Reliability — when the model is more confident, is it actually more right?")
            st.bar_chart(rel_df["accuracy"])

# ---------------------------------------------------------------------------
# Team strength ratings
# ---------------------------------------------------------------------------
st.header("🏆 Team strength")
try:
    from sandy.football.ratings import ARTIFACT_NAME, load_model
    model = load_model(cfg.model.model_dir / ARTIFACT_NAME)
    names = _df("SELECT team_id, name FROM football.teams")
    name_map = dict(zip(names["team_id"], names["name"]))
    mc = _df("""SELECT team_id, count(*) n FROM (
        SELECT home_team_id team_id FROM football.matches WHERE status IN ('FT','AET','PEN')
        UNION ALL SELECT away_team_id FROM football.matches WHERE status IN ('FT','AET','PEN')) s
        GROUP BY team_id""")
    mc_map = dict(zip(mc["team_id"], mc["n"]))
    mg = model.mean_goals
    strength = []
    for tid, atk in model.attack.items():
        if mc_map.get(tid, 0) < 25:
            continue
        gd = float(np.exp(mg + atk) - np.exp(mg - model.defense.get(tid, 0.0)))
        strength.append({"Team": name_map.get(tid, tid), "Exp. goal diff /game": round(gd, 2),
                         "Matches": mc_map.get(tid, 0)})
    sdf = pd.DataFrame(strength).sort_values("Exp. goal diff /game", ascending=False).head(25)
    st.dataframe(sdf, use_container_width=True, hide_index=True)
    st.caption("Expected goal difference vs an average team (neutral venue). Min 25 matches.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Ratings artifact not available yet: {exc}")

# ---------------------------------------------------------------------------
# Recent results
# ---------------------------------------------------------------------------
st.header("✅ Recent results")
results = get_recent_results(_engine(), cfg, days=10)
if not results:
    st.info("No reconciled World Cup results yet — they appear after matches finish.")
else:
    rr = []
    for r in results:
        rr.append({
            "Match": f"{r['home']} {r['actual_home_goals']}-{r['actual_away_goals']} {r['away']}",
            "Result pick": "✅" if r["was_correct_result"] else "❌",
            "O/U pick": "✅" if r["was_correct_over_2_5"] else "❌",
            "BTTS pick": "✅" if r["was_correct_btts"] else "❌",
        })
    st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Data coverage
# ---------------------------------------------------------------------------
st.header("🗂️ Data coverage")
cov = _df("""
    SELECT EXTRACT(YEAR FROM match_date)::int AS year, count(*) AS matches,
           count(*) FILTER (WHERE status IN ('FT','AET','PEN')) AS finished
    FROM football.matches GROUP BY year ORDER BY year
""")
st.bar_chart(cov.set_index("year")["matches"])
st.caption(f"{int(cov['matches'].sum()):,} matches across senior national-team competitions, 2019–2026.")
